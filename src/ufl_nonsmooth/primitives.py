# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/primitives.py
@brief Directional actions and selected pullbacks for continuous scalar operations.
@details Native Unified Form Language values retain native solver linearisations. Directional
    evaluation selects max, min, and absolute-value branches using the supplied perturbation,
    then applies native differentiation. Its result is an action on that perturbation, not a
    linear operator whose transpose represents the whole non-smooth relation. All remaining
    operations must be continuously differentiable on their active domains. Fixed-topology
    coordinate responses use native reference pullbacks. Moving general conditionals require
    continuity, interface, or event contracts.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from firedrake import Function, Interpolate, MeshGeometry
from numpy.typing import NDArray
from ufl import (
    And,
    Argument,
    Form,
    Or,
    SpatialCoordinate,
    ZeroBaseForm,
    as_ufl,
    conditional,
    derivative,
    diff,
    dot,
    eq,
    grad,
    max_value,
    min_value,
    replace,
    zero,
)
from ufl.algorithms.analysis import extract_arguments, extract_coefficients
from ufl.algorithms.apply_algebra_lowering import apply_algebra_lowering
from ufl.algorithms.apply_derivatives import apply_coordinate_derivatives, apply_derivatives
from ufl.algorithms.apply_function_pullbacks import apply_function_pullbacks, apply_inverse_pullback
from ufl.algorithms.apply_geometry_lowering import apply_geometry_lowering
from ufl.algorithms.apply_integral_scaling import compute_integrand_scaling_factor
from ufl.classes import (
    Abs,
    Coefficient,
    Conditional,
    GeometricQuantity,
    Jacobian,
    MaxValue,
    MinValue,
    ReferenceGrad,
    ReferenceValue,
    Variable,
)
from ufl.classes import Interpolate as UFLInterpolate
from ufl.core.base_form_operator import BaseFormOperator
from ufl.core.expr import Expr
from ufl.corealg.traversal import unique_post_traversal
from ufl.domain import extract_unique_domain
from ufl.form import BaseForm

ArrayValue = float | NDArray[np.float64]
DerivativeTarget = Coefficient | SpatialCoordinate | Variable


@runtime_checkable
class NativeOperatorResponse(Protocol):
    """
    @brief Selected linear and joint directional actions of a native local operator.
    @details All input dependence is carried by its operands in the fixed reference output
        space. Selected actions are linear in their perturbations. Joint actions retain
        perturbation coefficients so a regular implicit residual can differentiate with
        respect to its unknown response. Absent operand actions freeze those operands.
    """

    def selected_action(self, operand_actions: list[Expr | None]) -> Expr:
        """
        @brief Apply the selected native linearisation in its output space.
        """
        ...

    def directional_action(self, operand_actions: list[Expr | None]) -> Expr:
        """
        @brief Return the exact joint finite action in the native output space.
        """
        ...


@runtime_checkable
class NativeCoordinateResponse(Protocol):
    """
    @brief Explicit fixed-topology geometry action of an opaque native operator.
    """

    def coordinate_action(self, target: SpatialCoordinate, direction: Expr) -> Expr:
        """
        @brief Return the material response in the operator's native output space.
        """
        ...


@dataclass(frozen=True)
class ScalarGraphResponse:
    """
    @brief A finite union of closed intervals of finite scalar graph responses.
    @details Infinite endpoints describe unbounded sets of finite outputs. An empty tuple is an
        empty relation, and separated singleton intervals can describe a nonconvex relation.
    @param intervals: Ordered interval endpoints, with no NaNs or reversed intervals.
    """

    intervals: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        """
        @brief Validate intervals as sets of finite real outputs.
        """
        for lower, upper in self.intervals:
            if (
                np.isnan(lower)
                or np.isnan(upper)
                or lower > upper
                or lower == np.inf
                or upper == -np.inf
            ):
                raise ValueError("graph response intervals must contain finite real outputs")

    @property
    def empty(self) -> bool:
        """
        @brief Whether the graph has no finite output in the supplied direction.
        """
        return not self.intervals

    @property
    def single_valued(self) -> bool:
        """
        @brief Whether the relation contains exactly one finite response.
        """
        return bool(self.intervals) and all(
            lower == upper == self.intervals[0][0] for lower, upper in self.intervals
        )

    def contains(self, candidate: float) -> bool:
        """
        @brief Test a finite candidate against every interval of the relation.
        """
        return bool(np.isfinite(candidate)) and any(
            lower <= candidate <= upper for lower, upper in self.intervals
        )

    def value(self) -> float:
        """
        @brief Read a unique finite response without selecting from a set-valued result.
        """
        if not self.single_valued:
            raise ValueError("an empty or set-valued response has no unique value")
        return self.intervals[0][0]


class DirectionalResponseError(ValueError):
    """
    @brief A requested finite action has an empty or multivalued graphical response.
    @param message: Operation and domain obligation that prevented a finite action.
    @param response: Exact known scalar graph response at the requested point and direction.
    """

    def __init__(self, message: str, response: ScalarGraphResponse) -> None:
        super().__init__(message)
        self.response = response


def clip(value: Expr | float, lower: Expr | float, upper: Expr | float) -> Expr:
    """
    @brief Bound a scalar expression by native maximum and minimum operations.
    @param value: Expression to bound.
    @param lower: Lower bound, which may depend on a coefficient.
    @param upper: Upper bound, which may depend on a coefficient.
    @return Native scalar expression with the second operand selected at equality in each operation.
    @pre The lower bound does not exceed the upper bound.
    """
    expression = as_ufl(value)
    bottom, top = as_ufl(lower), as_ufl(upper)
    if expression.ufl_shape or bottom.ufl_shape or top.ufl_shape:
        raise ValueError("clipping requires scalar expressions")
    return min_value(max_value(expression, bottom), top)


def absolute(value: Expr | float) -> Expr:
    """
    @brief Absolute value with the positive branch selected by native differentiation at zero.
    @param value: Real scalar expression.
    @return Native maximum of the negative and positive branches.
    """
    expression = as_ufl(value)
    if expression.ufl_shape:
        raise ValueError("absolute value requires a scalar expression")
    return max_value(-expression, expression)


def oriented_flux(flux: Expr, positive_payload: Expr, negative_payload: Expr) -> Expr:
    """
    @brief Split an oriented flux into its continuous positive and negative contributions.
    @param flux: Signed scalar flux.
    @param positive_payload: Payload multiplying a positive flux.
    @param negative_payload: Payload multiplying the magnitude of a negative flux.
    @return Positive and negative flux magnitudes times their respective payloads.
    """
    if flux.ufl_shape:
        raise ValueError("an oriented flux must be scalar")
    return max_value(flux, 0.0) * positive_payload + max_value(-flux, 0.0) * negative_payload


def directional(
    expression: Expr | BaseForm,
    coefficient: DerivativeTarget | list[DerivativeTarget],
    direction: Expr | list[Expr],
) -> Expr | BaseForm:
    """
    @brief Evaluate a joint coefficient and fixed-topology coordinate action.
    @details Native algebra and spatial derivatives are lowered before switches are classified.
        The direction is a coefficient expression, since the response can be nonlinear in it.
        General moving predicates require a separate continuity or interface contract. Native
        coordinate derivatives retain reference-cell field values, geometric factors, normals,
        and integration measures. Remaining operations must be continuously differentiable on
        the active domain. The result is an action, not a global reverse relation.
    @param expression: Native expression, interpolation, or weak form.
    @param coefficient: State or parameter coefficient, spatial coordinates, or joint input list.
    @param direction: Corresponding perturbation or list, evaluated at the base point and held
        fixed during differentiation. A geometry direction is a reference coordinate field.
    @return Native directional expression or form.
    """
    coefficients = coefficient if isinstance(coefficient, list) else [coefficient]
    directions = direction if isinstance(direction, list) else [direction]
    if not coefficients or len(coefficients) != len(directions):
        raise ValueError("each joint coefficient needs one supplied direction")
    targets = _canonical_targets(coefficients, directions)
    if any(extract_arguments(perturbation) for perturbation in directions):
        raise ValueError("a non-smooth direction must be a coefficient expression, not an Argument")
    expression = apply_derivatives(apply_algebra_lowering(expression))
    integrands = (
        [integral.integrand() for integral in expression.integrals()]
        if isinstance(expression, Form)
        else [expression]
    )
    replacements: dict[Expr, Expr] = {}
    visited: set[Expr] = set()
    for integrand in integrands:
        for node in unique_post_traversal(integrand):
            if node in visited:
                continue
            visited.add(node)
            if isinstance(node, Conditional) and _depends_on(node.ufl_operands[0], targets):
                raise ValueError(
                    "a parameter-dependent conditional requires a separate response contract"
                )
            if not isinstance(node, (MaxValue, MinValue, Abs)):
                continue
            operands = [replace(operand, replacements) for operand in node.ufl_operands]
            if isinstance(node, Abs):
                operands = [-operands[0], operands[0]]
            left, right = operands
            if left.ufl_shape or right.ufl_shape or extract_arguments(node):
                raise ValueError(
                    "non-smooth primitives require real scalar coefficient expressions"
                )
            left_action = _expression_action(left, targets, directions, True)
            right_action = _expression_action(right, targets, directions, True)
            assert isinstance(left_action, Expr) and isinstance(right_action, Expr)
            if isinstance(node, MinValue):
                selected = Or(left < right, And(eq(left, right), left_action <= right_action))
            else:
                selected = Or(left > right, And(eq(left, right), left_action >= right_action))
            replacements[node] = conditional(selected, left, right)
    return _expression_action(replace(expression, replacements), targets, directions, True)


def _depends_on(expression: Expr, coefficients: list[DerivativeTarget]) -> bool:
    """
    @brief Test dependency on state fields and the geometry of a differentiated mesh.
    """
    fields = extract_coefficients(expression)
    geometries = [item for item in coefficients if isinstance(item, SpatialCoordinate)]
    variables = {node for node in unique_post_traversal(expression) if isinstance(node, Variable)}
    return any(item in fields or item in variables for item in coefficients) or any(
        isinstance(node, GeometricQuantity)
        and any(extract_unique_domain(node) == extract_unique_domain(item) for item in geometries)
        for node in unique_post_traversal(expression)
    )


def _value_nodes(expression: Expr | BaseForm) -> list[Expr]:
    """
    @brief Traverse differentiated value paths, excluding native conditional predicates.
    """
    active: set[Expr | BaseForm] = set()
    pending = [expression]
    while pending:
        node = pending.pop()
        if node in active:
            continue
        active.add(node)
        operands = node.ufl_operands[1:] if isinstance(node, Conditional) else node.ufl_operands
        pending.extend(operands)
    return [node for node in unique_post_traversal(expression) if node in active]


def selected_directional(
    expression: Expr | BaseForm,
    coefficients: list[DerivativeTarget],
    directions: list[Expr],
) -> Expr | BaseForm:
    """
    @brief Apply the recorded native linearisation to one joint supplied direction.
    @details Primitive switches retain their native tie selection. Opaque operator jets retain
        their output spaces through temporary symbolic coefficients, which are substituted
        before assembly. This action is linear in the perturbation. Its transpose at a switch
        is not the full reverse relation or necessarily a limiting normal. On a persistent
        smooth stratum it is the ordinary derivative. An empty set of inputs has the zero action.
    @param expression: Native expression, interpolation, or weak form.
    @param coefficients: Distinct native coefficients, formal variables, and coordinate targets.
    @param directions: Corresponding shape-compatible perturbations, frozen at the base point.
    @return Native selected branch action.
    """
    if len(coefficients) != len(directions):
        raise ValueError("each selected input requires one supplied direction")
    if not coefficients:
        return (
            ZeroBaseForm(expression.arguments())
            if isinstance(expression, BaseForm)
            else zero(expression.ufl_shape)
        )
    targets = _canonical_targets(coefficients, directions)
    return _expression_action(expression, targets, directions, False)


def _canonical_targets(
    coefficients: list[DerivativeTarget], directions: list[Expr]
) -> list[DerivativeTarget]:
    """
    @brief Validate independent native coefficients, formal operands, and geometry targets.
    @details Formal scalar variables are independent local operands. A formal operand and a
        target occurring inside it cannot both be differentiated as independent inputs.
    """
    if len(coefficients) != len(directions):
        raise ValueError("each joint coefficient needs one supplied direction")
    targets: list[DerivativeTarget] = []
    for target, perturbation in zip(coefficients, directions, strict=True):
        if not isinstance(target, (Coefficient, SpatialCoordinate, Variable)):
            raise TypeError(
                "directional evaluation requires coefficients, local variables, or spatial coordinates"
            )
        if isinstance(target, Function):
            mesh = target.function_space().mesh()
            if isinstance(mesh, MeshGeometry) and target is mesh.coordinates:
                target = SpatialCoordinate(mesh)
        if target in targets:
            raise ValueError("joint directional coefficients must be distinct")
        if isinstance(target, Variable) and target.ufl_shape:
            raise ValueError("local formal operands must be scalar variables")
        if target.ufl_shape != perturbation.ufl_shape:
            raise ValueError("the coefficient and its direction must have the same value shape")
        targets.append(target)
    for target in targets:
        if isinstance(target, Variable):
            nested = set(unique_post_traversal(target.ufl_operands[0]))
            if any(other != target and other in nested for other in targets):
                raise ValueError("formal local operands must be independent derivative targets")
    return targets


def _expression_action(
    expression: Expr | BaseForm,
    coefficients: list[DerivativeTarget],
    directions: list[Expr],
    joint: bool,
) -> Expr | BaseForm:
    """
    @brief Lower native operators while retaining selected or joint action semantics.
    """
    if isinstance(expression, BaseFormOperator):
        return _branch_action(expression, coefficients, directions, joint)
    integrands = (
        [integral.integrand() for integral in expression.integrals()]
        if isinstance(expression, Form)
        else [expression]
    )
    replacements: dict[Expr, Expr] = {}
    proxies: list[DerivativeTarget] = []
    actions: list[Expr] = []
    for integrand in integrands:
        for node in _value_nodes(integrand):
            if not isinstance(node, BaseFormOperator) or node in replacements:
                continue
            if len(node.arguments()) != 1:
                raise ValueError("a response operand must be a value-valued native operator")
            operand = replace(node, replacements)
            if not _depends_on(node, coefficients) and not any(
                isinstance(item, SpatialCoordinate) for item in coefficients
            ):
                action = zero(node.ufl_shape)
            else:
                action = _branch_action(
                    operand, coefficients + proxies, directions + actions, joint
                )
            if not isinstance(action, Expr):
                raise TypeError(
                    "a pointwise native operator must provide a coefficient-valued action"
                )
            proxy = Coefficient(node.ufl_function_space())
            replacements[node] = proxy
            proxies.append(proxy)
            actions.append(action)
    if not replacements:
        return _branch_action(expression, coefficients, directions, joint)
    result = _branch_action(
        replace(expression, replacements), coefficients + proxies, directions + actions, joint
    )
    return replace(result, {proxy: node for node, proxy in replacements.items()})


def _branch_action(
    expression: Expr | BaseForm,
    coefficients: list[DerivativeTarget],
    directions: list[Expr],
    joint: bool,
) -> Expr | BaseForm:
    """
    @brief Differentiate one selected branch jointly using native field and geometry rules.
    """
    if isinstance(expression, NativeOperatorResponse):
        operand_actions: list[Expr | None] = []
        for operand in expression.ufl_operands:
            if not _depends_on(operand, coefficients):
                operand_actions.append(None)
                continue
            action = _expression_action(operand, coefficients, directions, joint)
            if not isinstance(action, Expr):
                raise TypeError("a native operator operand requires an expression-valued action")
            operand_actions.append(action)
        return (
            expression.directional_action(operand_actions)
            if joint
            else expression.selected_action(operand_actions)
        )
    fields = [item for item in coefficients if isinstance(item, Coefficient)]
    field_directions = [
        perturbation
        for item, perturbation in zip(coefficients, directions, strict=True)
        if isinstance(item, Coefficient)
    ]
    actions: list[Expr | BaseForm] = []
    for target, perturbation in zip(coefficients, directions, strict=True):
        if isinstance(target, Variable):
            if isinstance(expression, BaseForm) or target.ufl_shape:
                raise ValueError("formal local operands require a scalar expression law")
            actions.append(apply_derivatives(diff(expression, target)) * perturbation)
    if fields:
        arguments = extract_arguments(expression)
        number = max((item.number() for item in arguments), default=-1) + 1
        for field, perturbation in zip(fields, field_directions, strict=True):
            variation = Argument(field.ufl_function_space(), number)
            action = apply_derivatives(derivative(expression, field, variation))
            actions.append(replace(action, {variation: perturbation}))
    for target, perturbation in zip(coefficients, directions, strict=True):
        if not isinstance(target, SpatialCoordinate):
            continue
        if isinstance(expression, NativeCoordinateResponse):
            action = expression.coordinate_action(target, perturbation)
        elif isinstance(expression, UFLInterpolate):
            operand = expression.ufl_operands[0]
            domain = expression.ufl_function_space().ufl_domain()
            source_domain = extract_unique_domain(operand)
            if source_domain is not None and source_domain != domain:
                raise NotImplementedError(
                    "nonmatching interpolation requires a mesh-transfer response contract"
                )
            element = expression.ufl_element()
            reference_operand = apply_inverse_pullback(operand, element, domain)
            reference_action = _coordinate_expression_action(
                reference_operand, target, perturbation
            )
            action = Interpolate(
                element.pullback.apply(reference_action, domain), expression.argument_slots()[0]
            )
        elif isinstance(expression, BaseFormOperator):
            raise NotImplementedError(
                "an opaque native operator requires a coordinate_action contract"
            )
        elif isinstance(expression, Form):
            integrals = []
            for integral in expression.integrals():
                integrand = integral.integrand()
                scaling, _ = compute_integrand_scaling_factor(integral)
                density_action = _coordinate_expression_action(integrand, target, perturbation)
                scaling_action = _coordinate_expression_action(
                    as_ufl(scaling), target, perturbation
                )
                integrals.append(
                    integral.reconstruct(
                        integrand=density_action + integrand * scaling_action / scaling
                    )
                )
            action = Form(integrals)
        elif isinstance(expression, Expr) and not isinstance(expression, BaseForm):
            action = _coordinate_expression_action(expression, target, perturbation)
        else:
            mesh = extract_unique_domain(target)
            geometry_fields = (
                {mesh.coordinates: perturbation} if isinstance(mesh, MeshGeometry) else {}
            )
            action = derivative(expression, target, perturbation, geometry_fields)
        actions.append(action)
    result = actions[0]
    for action in actions[1:]:
        result += action
    return result


def _coordinate_expression_action(
    expression: Expr, target: SpatialCoordinate, perturbation: Expr
) -> Expr:
    """
    @brief Differentiate a pulled-back expression and restore its physical representation.
    @details Native field pullbacks retain Piola maps. Reference gradients are converted with
        the Jacobian before the resulting expression enters native assembly again.
    """
    mesh = extract_unique_domain(target)
    geometry_fields = {mesh.coordinates: perturbation} if isinstance(mesh, MeshGeometry) else {}
    action = derivative(expression, target, perturbation, geometry_fields)
    action = apply_function_pullbacks(apply_derivatives(action))
    action = apply_derivatives(apply_geometry_lowering(action))
    action = apply_derivatives(apply_geometry_lowering(action))
    action = apply_coordinate_derivatives(action)
    replacements: dict[Expr, Expr] = {}
    for node in unique_post_traversal(action):
        if isinstance(node, ReferenceValue):
            field = node.ufl_operands[0]
            replacements[node] = apply_inverse_pullback(
                field, field.ufl_element(), extract_unique_domain(field)
            )
        elif isinstance(node, ReferenceGrad):
            operand = replace(node.ufl_operands[0], replacements)
            replacements[node] = dot(grad(operand), Jacobian(extract_unique_domain(node)))
    return apply_derivatives(apply_algebra_lowering(replace(action, replacements)))


def maximum_values(
    left: ArrayValue, right: ArrayValue
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    @brief Maximum and partial weights of its selected linear branch, selecting right at equality.
    @param left: Left values.
    @param right: Right values, broadcast with the left values.
    @return Values, left weights, and right weights.
    """
    first, second = np.broadcast_arrays(left, right)
    weight = np.asarray(first > second, dtype=np.float64)
    return (
        np.asarray(np.maximum(first, second), dtype=np.float64),
        weight,
        np.asarray(1.0 - weight, dtype=np.float64),
    )


def absolute_values(values: ArrayValue) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    @brief Absolute values and selected slopes, choosing the positive branch at zero.
    @param values: Real input values.
    @return Magnitudes and selected slopes.
    """
    points = np.asarray(values, dtype=np.float64)
    return np.abs(points), np.asarray(np.where(points >= 0.0, 1.0, -1.0), dtype=np.float64)


def clip_values(
    values: ArrayValue, lower: ArrayValue, upper: ArrayValue
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    @brief Clipped values and selected partial weights for the input and both moving bounds.
    @details The bound is selected at either endpoint, with the upper bound selected when the
        bounds coincide. These are branch weights, not the nonlinear directional response.
    @param values: Values to clip.
    @param lower: Lower bounds.
    @param upper: Upper bounds.
    @return Values, input weights, lower-bound weights, and upper-bound weights.
    """
    if np.any(np.asarray(lower) > np.asarray(upper)):
        raise ValueError("the lower clipping bound exceeds the upper bound")
    raised, value_weight, lower_weight = maximum_values(values, lower)
    raised, top = np.broadcast_arrays(raised, upper)
    free = np.asarray(raised < top, dtype=np.float64)
    return (
        np.asarray(np.minimum(raised, top), dtype=np.float64),
        np.asarray(free * value_weight, dtype=np.float64),
        np.asarray(free * lower_weight, dtype=np.float64),
        np.asarray(1.0 - free, dtype=np.float64),
    )


def maximum_directional(
    left: ArrayValue, right: ArrayValue, left_direction: ArrayValue, right_direction: ArrayValue
) -> NDArray[np.float64]:
    """
    @brief Directional response of a maximum, retaining both input directions at a tie.
    @param left: Left values.
    @param right: Right values.
    @param left_direction: Left perturbations.
    @param right_direction: Right perturbations.
    @return Response, nonlinear in the perturbations at a tie.
    """
    first, second, first_rate, second_rate = np.broadcast_arrays(
        left, right, left_direction, right_direction
    )
    return np.asarray(
        np.where(
            first > second,
            first_rate,
            np.where(first < second, second_rate, np.maximum(first_rate, second_rate)),
        ),
        dtype=np.float64,
    )


def clip_directional(
    values: ArrayValue,
    lower: ArrayValue,
    upper: ArrayValue,
    direction: ArrayValue,
    lower_direction: ArrayValue = 0.0,
    upper_direction: ArrayValue = 0.0,
) -> NDArray[np.float64]:
    """
    @brief Directional response of clipping, including perturbations of both bounds.
    @param values: Input values.
    @param lower: Lower bounds.
    @param upper: Upper bounds.
    @param direction: Input perturbations.
    @param lower_direction: Lower-bound perturbations.
    @param upper_direction: Upper-bound perturbations.
    @return Response within directions preserving the ordering of coincident bounds.
    """
    if np.any(np.asarray(lower) > np.asarray(upper)) or np.any(
        (np.asarray(lower) == np.asarray(upper))
        & (np.asarray(lower_direction) > np.asarray(upper_direction))
    ):
        raise ValueError("the clipping direction must preserve the ordering of the bounds")
    raised = np.maximum(values, lower)
    raised_direction = maximum_directional(values, lower, direction, lower_direction)
    return -maximum_directional(
        -raised, -np.asarray(upper), -raised_direction, -np.asarray(upper_direction)
    )


def masked_pullback(seeds: ArrayValue, slopes: ArrayValue) -> NDArray[np.float64]:
    """
    @brief Multiply branch slopes by nonzero seeds, omitting unobserved singular contributions.
    @param seeds: Incoming covectors.
    @param slopes: Selected branch slopes, which may be infinite at a singular boundary.
    @return Pointwise pullback, zero wherever the incoming covector is zero.
    """
    weights, rates = np.broadcast_arrays(seeds, slopes)
    result = np.zeros(weights.shape, dtype=np.float64)
    np.multiply(weights, rates, out=result, where=weights != 0.0)
    return result


def kink_coderivative(left_slope: float, right_slope: float, seed: float) -> ScalarGraphResponse:
    """
    @brief Basic coderivative of a continuous scalar law with two smooth incident branches.
    @details The slopes are the limiting derivatives from smaller and larger inputs. The
        result includes the interval only when the seed-weighted left slope is no larger
        than the right one. Otherwise it contains the two branch covectors separately.
        This is a local scalar contract, not a coderivative composition rule.
    """
    if not np.isfinite((left_slope, right_slope, seed)).all():
        raise ValueError("scalar kink slopes and output covectors must be finite")
    left, right = seed * left_slope, seed * right_slope
    if not np.isfinite((left, right)).all():
        raise ValueError("scalar kink covectors exceed the finite numerical range")
    intervals = ((left, right),) if left <= right else ((right, right), (left, left))
    return ScalarGraphResponse(intervals)


def maximum_coderivative(value: float, seed: float, threshold: float = 0.0) -> ScalarGraphResponse:
    """
    @brief Basic coderivative of the maximum of one scalar input and a fixed threshold.
    @details Only value varies. The threshold is held fixed.
    """
    if not np.isfinite((value, threshold)).all():
        raise ValueError("maximum inputs must be finite")
    left, right = float(value > threshold), float(value >= threshold)
    return kink_coderivative(left, right, seed)


def minimum_coderivative(value: float, seed: float, threshold: float = 0.0) -> ScalarGraphResponse:
    """
    @brief Basic coderivative of the minimum of one scalar input and a fixed threshold.
    @details Only value varies. The threshold is held fixed.
    """
    if not np.isfinite((value, threshold)).all():
        raise ValueError("minimum inputs must be finite")
    left, right = float(value <= threshold), float(value < threshold)
    return kink_coderivative(left, right, seed)


def absolute_coderivative(value: float, seed: float) -> ScalarGraphResponse:
    """
    @brief Basic coderivative of the absolute value of one scalar input.
    """
    if not np.isfinite(value):
        raise ValueError("the absolute-value input must be finite")
    left, right = (-1.0 if value <= 0.0 else 1.0), (-1.0 if value < 0.0 else 1.0)
    return kink_coderivative(left, right, seed)


def clip_coderivative(value: float, lower: float, upper: float, seed: float) -> ScalarGraphResponse:
    """
    @brief Basic coderivative of one scalar input clipped between fixed finite bounds.
    @details Both bounds are held fixed. Equal bounds give a constant map.
    """
    if not np.isfinite((value, lower, upper)).all() or lower > upper:
        raise ValueError("clipping needs finite inputs and ordered bounds")
    left, right = float(lower < value <= upper), float(lower <= value < upper)
    return kink_coderivative(left, right, seed)
