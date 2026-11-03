# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_primitives.py
@brief Tests for native non-smooth directional actions and selected array pullbacks.
@details Hand-calculated responses at scalar switches distinguish directional actions from
    linear branch pullbacks. Moving clipping bounds, joint coefficients, weak forms, and
    one-sided numerical differences check the supported compositions and their boundaries.
"""

import numpy as np
import pytest
from firedrake import (
    FacetNormal,
    Function,
    FunctionSpace,
    Interpolate,
    MeshGeometry,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    SpatialCoordinate,
    TestFunction,
    VectorFunctionSpace,
    assemble,
    ds,
    dx,
    grad,
    sqrt,
)
from ufl import Form, ZeroBaseForm, as_ufl, as_vector, conditional, max_value, min_value, replace
from ufl.core.expr import Expr

from ufl_nonsmooth.primitives import (
    ScalarGraphResponse,
    absolute,
    absolute_coderivative,
    absolute_values,
    clip,
    clip_coderivative,
    clip_directional,
    clip_values,
    directional,
    masked_pullback,
    maximum_coderivative,
    maximum_directional,
    maximum_values,
    minimum_coderivative,
    selected_directional,
)


@pytest.fixture
def scalar_coefficient(box_mesh: MeshGeometry) -> Function:
    """
    @brief A real-space coefficient for algebraic directional evaluations.
    """
    space = FunctionSpace(box_mesh, "R", 0)
    return Function(space)


@pytest.mark.parametrize(
    ("operation", "point", "perturbation", "expected"),
    [
        ("maximum", -1.0, 2.0, 0.0),
        ("maximum", 1.0, -2.0, -2.0),
        ("maximum", 0.0, 2.0, 2.0),
        ("maximum", 0.0, -2.0, 0.0),
        ("minimum", -1.0, 2.0, 2.0),
        ("minimum", 1.0, -2.0, 0.0),
        ("minimum", 0.0, 2.0, 0.0),
        ("minimum", 0.0, -2.0, -2.0),
        ("absolute", -1.0, 2.0, -2.0),
        ("absolute", 1.0, -2.0, -2.0),
        ("absolute", 0.0, 2.0, 2.0),
        ("absolute", 0.0, -2.0, 2.0),
        ("native_absolute", 0.0, 2.0, 2.0),
        ("native_absolute", 0.0, -2.0, 2.0),
    ],
)
def test_native_scalar_actions_follow_the_perturbed_branch(
    scalar_coefficient: Function,
    operation: str,
    point: float,
    perturbation: float,
    expected: float,
) -> None:
    """
    @brief Maximum, minimum, and both absolute-value forms have their analytical one-sided actions.
    """
    coefficient = scalar_coefficient
    expressions: dict[str, Expr] = {
        "maximum": max_value(coefficient, 0.0),
        "minimum": min_value(coefficient, 0.0),
        "absolute": absolute(coefficient),
        "native_absolute": abs(as_ufl(coefficient)),
    }
    action = directional(expressions[operation], coefficient, as_ufl(perturbation))
    assert isinstance(action, Expr)
    assert float(replace(action, {coefficient: as_ufl(point)})) == pytest.approx(expected)


@pytest.mark.parametrize("perturbation", [-2.0, 0.0, 3.0])
def test_compositions_retain_the_shared_coefficient_direction(
    scalar_coefficient: Function, perturbation: float
) -> None:
    """
    @brief Opposite positive parts compose to the identity and their minimum to zero.
    """
    coefficient = scalar_coefficient
    positive = max_value(coefficient, 0.0)
    negative = max_value(-coefficient, 0.0)
    for expression, expected in (
        (positive - negative, perturbation),
        (min_value(positive, negative), 0.0),
        (max_value(positive, negative), abs(perturbation)),
        ((positive - negative) ** 2 + 2.0 * positive, 2.0 * max(perturbation, 0.0)),
    ):
        action = directional(expression, coefficient, as_ufl(perturbation))
        assert isinstance(action, Expr)
        assert float(replace(action, {coefficient: as_ufl(0.0)})) == pytest.approx(expected)


def test_joint_maximum_action_is_not_the_sum_of_independent_actions(
    scalar_coefficient: Function,
) -> None:
    """
    @brief Moving both equal inputs up by one moves their maximum by one, whereas the two
        independent one-sided responses each equal one.
    """
    first = scalar_coefficient
    second = Function(first.function_space())
    expression = max_value(first, second)
    joint = directional(expression, [first, second], [as_ufl(1.0), as_ufl(1.0)])
    first_action = directional(expression, first, as_ufl(1.0))
    second_action = directional(expression, second, as_ufl(1.0))
    values = {first: as_ufl(0.0), second: as_ufl(0.0)}
    assert isinstance(joint, Expr)
    assert isinstance(first_action, Expr)
    assert isinstance(second_action, Expr)
    assert float(replace(joint, values)) == 1.0
    assert float(replace(first_action + second_action, values)) == 2.0


@pytest.mark.parametrize(
    ("point", "value_rate", "lower_rate", "upper_rate", "perturbation", "expected"),
    [
        (0.0, 2.0, -1.0, 0.0, 1.0, 2.0),
        (0.0, 2.0, -1.0, 0.0, -1.0, 1.0),
        (2.0, 2.0, 0.0, -1.0, 1.0, -1.0),
        (2.0, 2.0, 0.0, -1.0, -1.0, -2.0),
        (1.0, 2.0, -1.0, 1.0, -1.0, -2.0),
        (-1.0, 2.0, -1.0, 1.0, 1.0, -1.0),
        (3.0, 2.0, -1.0, 1.0, -1.0, -1.0),
    ],
)
def test_native_clipping_includes_moving_bound_actions(
    scalar_coefficient: Function,
    point: float,
    value_rate: float,
    lower_rate: float,
    upper_rate: float,
    perturbation: float,
    expected: float,
) -> None:
    """
    @brief Clipping follows the moving bound or value selected by the direction at each tie.
    """
    coefficient = scalar_coefficient
    expression = clip(
        point + value_rate * coefficient,
        lower_rate * coefficient,
        2.0 + upper_rate * coefficient,
    )
    action = directional(expression, coefficient, as_ufl(perturbation))
    assert isinstance(action, Expr)
    assert float(replace(action, {coefficient: as_ufl(0.0)})) == pytest.approx(expected)


def test_joint_clipping_moves_independent_value_and_bounds(scalar_coefficient: Function) -> None:
    """
    @brief The joint response at a lower clipping tie compares the value and lower-bound directions.
    """
    value = scalar_coefficient
    lower = Function(value.function_space())
    upper = Function(value.function_space())
    expression = clip(value, lower, upper)
    action = directional(expression, [value, lower, upper], [as_ufl(-2), as_ufl(1), as_ufl(0)])
    assert isinstance(action, Expr)
    assert float(replace(action, {value: as_ufl(0), lower: as_ufl(0), upper: as_ufl(2)})) == 1.0


@pytest.mark.parametrize(("perturbation", "expected"), [((2.0, -1.0), 2.0), ((-2.0, 1.0), 1.0)])
def test_directional_actions_use_vector_coefficient_components(
    box_mesh: MeshGeometry, perturbation: tuple[float, float], expected: float
) -> None:
    """
    @brief A scalar maximum of two components retains their joint vector perturbation.
    """
    space = VectorFunctionSpace(box_mesh, "R", 0, dim=2)
    coefficient = Function(space)
    action = directional(
        max_value(coefficient[0], coefficient[1]), coefficient, as_vector(perturbation)
    )
    assert isinstance(action, Expr)
    assert float(replace(action, {coefficient: as_vector((0.0, 0.0))})) == expected


def test_directional_action_of_a_weak_residual_is_nonlinear_in_its_direction(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief The weak response of the positive part at zero integrates only positive perturbations.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    coefficient = Function(space)
    perturbation = Function(space)
    perturbation.dat.data[:] = np.where(np.arange(perturbation.dat.data_ro.size) % 2, -2.0, 3.0)
    test = TestFunction(space)
    residual = max_value(coefficient, 0.0) * test * dx
    action = directional(residual, coefficient, perturbation)
    opposite = directional(residual, coefficient, -perturbation)
    assert isinstance(action, Form)
    assert isinstance(opposite, Form)
    volumes = assemble(test * dx).dat.data_ro
    response = assemble(action).dat.data_ro
    opposite_response = assemble(opposite).dat.data_ro
    expected = volumes * np.where(perturbation.dat.data_ro > 0.0, 3.0, 0.0)
    assert response == pytest.approx(expected)
    assert not np.allclose(response, -opposite_response)
    step = 1e-6
    coefficient.assign(step * perturbation)
    assert assemble(residual).dat.data_ro / step == pytest.approx(response)


@pytest.mark.parametrize(("parameter_direction", "expected"), [(1.0, 0.5), (-1.0, -1.0)])
def test_native_implicit_directional_solve_retains_the_switch(
    box_mesh: MeshGeometry, parameter_direction: float, expected: float
) -> None:
    r"""
    @brief The directional equation of \f$ u + \max(u,0) = p \f$ uses the native nonlinear solver.
    @details At \f$ u=p=0 \f$ a positive parameter direction gives half that direction, while a
        negative direction passes through unchanged.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    state, parameter = Function(space), Function(space)
    response = Function(space)
    perturbation = Function(space).assign(parameter_direction)
    test = TestFunction(space)
    residual = (state + max_value(state, 0.0) - parameter) * test * dx
    action = directional(residual, [state, parameter], [response, perturbation])
    assert isinstance(action, Form)
    problem = NonlinearVariationalProblem(action, response)
    solver = NonlinearVariationalSolver(
        problem,
        solver_parameters={
            "snes_type": "newtonls",
            "snes_linesearch_type": "bt",
            "ksp_type": "preonly",
            "pc_type": "lu",
        },
    )
    solver.solve()
    assert response.dat.data_ro == pytest.approx(expected, abs=1e-12)


def test_general_parameter_conditionals_require_a_separate_contract(
    scalar_coefficient: Function,
) -> None:
    """
    @brief A general coefficient-dependent predicate is refused even for an absolute-value formula.
    """
    coefficient = scalar_coefficient
    expression = conditional(coefficient >= 0.0, coefficient, -coefficient)
    with pytest.raises(ValueError, match="separate response contract"):
        directional(expression, coefficient, as_ufl(1.0))


@pytest.mark.parametrize(("gate_value", "expected"), [(1.0, 0.0), (-1.0, -2.0)])
def test_parameter_independent_predicates_retain_their_branch(
    scalar_coefficient: Function, gate_value: float, expected: float
) -> None:
    """
    @brief A predicate on an independent coefficient selects its recorded branch.
    """
    coefficient = scalar_coefficient
    gate = Function(coefficient.function_space())
    expression = conditional(gate > 0.0, max_value(coefficient, 0.0), min_value(coefficient, 0.0))
    action = directional(expression, coefficient, as_ufl(-2.0))
    assert isinstance(action, Expr)
    assert float(replace(action, {coefficient: as_ufl(0), gate: as_ufl(gate_value)})) == expected


def test_directional_inputs_refuse_arguments_and_freeze_supplied_directions(
    scalar_coefficient: Function,
) -> None:
    """
    @brief A nonlinear directional response requires a shape-compatible independent expression.
    """
    coefficient = scalar_coefficient
    expression = max_value(coefficient, 0.0)
    with pytest.raises(ValueError, match="not an Argument"):
        directional(expression, coefficient, TestFunction(coefficient.function_space()))
    with pytest.raises(ValueError, match="same value shape"):
        directional(expression, coefficient, as_vector((1.0, 2.0)))
    action = directional(coefficient**2, coefficient, coefficient)
    assert float(replace(action, {coefficient: as_ufl(2.0)})) == 8.0
    action = directional(expression, coefficient, coefficient + 1.0)
    assert float(replace(action, {coefficient: as_ufl(0.0)})) == 1.0


def test_joint_inputs_refuse_empty_mismatched_and_duplicate_entries(
    scalar_coefficient: Function,
) -> None:
    """
    @brief Joint directions name distinct coefficients and one independent perturbation per input.
    """
    first = scalar_coefficient
    second = Function(first.function_space())
    expression = max_value(first, second)
    with pytest.raises(ValueError, match="one supplied direction"):
        directional(expression, [], [])
    with pytest.raises(ValueError, match="one supplied direction"):
        directional(expression, [first, second], [as_ufl(1.0)])
    with pytest.raises(ValueError, match="distinct"):
        directional(expression, [first, first], [as_ufl(1), as_ufl(2)])
    action = directional(expression, [first, second], [second, as_ufl(1)])
    assert float(replace(action, {first: as_ufl(0.0), second: as_ufl(0.0)})) == 1.0


@pytest.mark.parametrize("coordinate_function", [False, True])
def test_native_coordinate_volume_surface_and_normal_actions(
    box_mesh: MeshGeometry, coordinate_function: bool
) -> None:
    """
    @brief Native shape actions include volume measures, surface measures, and moving normals.
    """
    position = SpatialCoordinate(box_mesh)
    target = box_mesh.coordinates if coordinate_function else position
    dilation = Function(box_mesh.coordinates.function_space()).interpolate(position)
    volume = assemble(1.0 * dx(domain=box_mesh))
    area = assemble(1.0 * ds(domain=box_mesh))
    assert assemble(directional(1.0 * dx(domain=box_mesh), target, dilation)) == pytest.approx(
        3.0 * volume
    )
    assert assemble(directional(1.0 * ds(domain=box_mesh), target, dilation)) == pytest.approx(
        2.0 * area
    )
    shear = Function(box_mesh.coordinates.function_space()).interpolate(
        as_vector((position[1], 0.0, 0.0))
    )
    normal = FacetNormal(box_mesh)
    response = directional(normal[0] * normal[1] * ds, target, shear)
    assert assemble(response) == pytest.approx(-2.0 * 0.5 * 0.02, abs=1e-12)


def test_joint_state_and_shape_response_includes_both_actions(box_mesh: MeshGeometry) -> None:
    """
    @brief A growing constant state on a dilating three-dimensional volume has both responses.
    """
    coefficient = Function(FunctionSpace(box_mesh, "R", 0)).assign(1.0)
    position = SpatialCoordinate(box_mesh)
    dilation = Function(box_mesh.coordinates.function_space()).interpolate(position)
    action = directional(
        max_value(coefficient, 0.0) * dx, [coefficient, position], [as_ufl(2.0), dilation]
    )
    assert assemble(action) == pytest.approx(5.0 * assemble(1.0 * dx(domain=box_mesh)))


def test_shape_switch_uses_the_geometric_operand_response(box_mesh: MeshGeometry) -> None:
    """
    @brief A positive affine coordinate carries its material action and its volume action.
    """
    position = SpatialCoordinate(box_mesh)
    dilation = Function(box_mesh.coordinates.function_space()).interpolate(position)
    action = directional(max_value(position[0] + 1.0, 0.0) * dx, position, dilation)
    assert assemble(action) == pytest.approx(assemble(1.0 * dx(domain=box_mesh)))


def test_spatial_lowering_refuses_an_uncontracted_moving_interface(
    box_mesh: MeshGeometry,
) -> None:
    r"""
    @brief The integral of a differentiated positive part has a moving-interface contribution.
    @details On the unit-length plate, the derivative of the active volume with respect to its
        offset is the cross-sectional area, not zero. A pointwise branch compiler must expose
        the resulting moving predicate instead of silently commuting the two derivatives.
    """
    parameter = Function(FunctionSpace(box_mesh, "R", 0))
    position = SpatialCoordinate(box_mesh)
    form = grad(max_value(position[0] + 0.5 + parameter, 0.0))[0] * dx
    with pytest.raises(ValueError, match="separate response contract"):
        directional(form, parameter, as_ufl(1.0))
    assert assemble(form) == pytest.approx(0.5 * assemble(1.0 * dx(domain=box_mesh)))


def test_maximum_of_a_spatial_gradient_and_native_interpolation(box_mesh: MeshGeometry) -> None:
    """
    @brief Native algebra lowering admits a smooth gradient operand before a scalar switch.
    """
    space = FunctionSpace(box_mesh, "CG", 1)
    coefficient = Function(space)
    position = SpatialCoordinate(box_mesh)
    perturbation = Function(space).interpolate(position[0])
    action = directional(max_value(grad(coefficient)[0], 0.0), coefficient, perturbation)
    assert assemble(action * dx) == pytest.approx(assemble(1.0 * dx(domain=box_mesh)))
    interpolation = directional(
        Interpolate(max_value(coefficient, 0.0), space), coefficient, perturbation
    )
    assert assemble(interpolation).dat.data_ro == pytest.approx(0.0)


def test_native_scalar_constructors_refuse_vector_inputs() -> None:
    """
    @brief Scalar clipping and absolute value require scalar operands.
    """
    vector = as_vector((1.0, 2.0))
    with pytest.raises(ValueError, match="scalar"):
        absolute(vector)
    with pytest.raises(ValueError, match="scalar"):
        clip(vector, 0.0, 1.0)
    with pytest.raises(ValueError, match="scalar"):
        clip(0.0, vector, 1.0)


def test_array_maximum_values_and_weights_broadcast_and_select_the_right_tie() -> None:
    """
    @brief Each broadcast maximum assigns its unit branch weight to the selected operand.
    """
    values, left, right = maximum_values(np.array([[-1.0], [2.0]]), np.array([0.0, 2.0]))
    assert values == pytest.approx(np.array([[0.0, 2.0], [2.0, 2.0]]))
    assert left == pytest.approx(np.array([[0.0, 0.0], [1.0, 0.0]]))
    assert right == pytest.approx(np.array([[1.0, 1.0], [0.0, 1.0]]))
    scalar = maximum_values(1.0, 1.0)
    assert all(value.shape == () for value in scalar)


def test_array_absolute_values_select_the_positive_slope_at_zero() -> None:
    """
    @brief Selected absolute-value slopes are minus one below zero and one at or above it.
    """
    values, slopes = absolute_values(np.array([-2.0, 0.0, 3.0]))
    assert values == pytest.approx([2.0, 0.0, 3.0])
    assert slopes == pytest.approx([-1.0, 1.0, 1.0])


def test_array_clipping_weights_include_both_moving_bounds() -> None:
    """
    @brief Interior values use the input weight and endpoint ties use their bound's weight.
    """
    values, free, lower, upper = clip_values(
        np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 0.0]), 0.0, np.array([2.0] * 5 + [0.0])
    )
    assert values == pytest.approx([0.0, 0.0, 1.0, 2.0, 2.0, 0.0])
    assert free == pytest.approx([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert lower == pytest.approx([1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert upper == pytest.approx([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    assert free + lower + upper == pytest.approx(np.ones(6))
    base = np.array([-1.0, 1.0, 3.0])
    _, free, lower, upper = clip_values(base, 0.0, 2.0)
    step = 1e-6
    difference = (
        clip_values(base + step * 0.3, -step * 0.5, 2.0 + step * 0.7)[0] - np.clip(base, 0.0, 2.0)
    ) / step
    assert free * 0.3 - lower * 0.5 + upper * 0.7 == pytest.approx(difference, abs=1e-9)


@pytest.mark.parametrize(
    ("sign", "expected"), [(1.0, [1.0, 3.0, -3.0, 5.0]), (-1.0, [-1.0, 2.0, 3.0, 6.0])]
)
def test_array_maximum_directional_matches_analytical_and_one_sided_responses(
    sign: float, expected: list[float]
) -> None:
    """
    @brief At equality the maximum compares both perturbations for each supplied direction.
    """
    left, right = np.array([-1.0, 0.0, 2.0, 1.0]), np.array([0.0, 0.0, 1.0, 1.0])
    left_rate, right_rate = (
        sign * np.array([4.0, -2.0, -3.0, 5.0]),
        sign * np.array([1.0, 3.0, 7.0, -6.0]),
    )
    action = maximum_directional(left, right, left_rate, right_rate)
    assert action == pytest.approx(expected)
    step = 1e-6
    difference = (
        np.maximum(left + step * left_rate, right + step * right_rate) - np.maximum(left, right)
    ) / step
    assert action == pytest.approx(difference, abs=1e-9)


def test_array_clipping_refuses_reversed_values_and_bound_directions() -> None:
    """
    @brief Clipping requires ordered bounds and directions that preserve a collapsed interval.
    """
    with pytest.raises(ValueError, match="lower clipping bound"):
        clip_values(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="ordering"):
        clip_directional(0.0, 0.0, 0.0, 0.0, 1.0, -1.0)
    with pytest.raises(ValueError, match="ordering"):
        clip_directional(0.0, 1.0, 0.0, 0.0)


def test_masked_pullback_omits_unobserved_infinite_slopes() -> None:
    """
    @brief Zero seeds contribute zero even at infinite slopes, while observed signs are retained.
    """
    seeds = np.array([[0.0, 1.0], [-2.0, 0.0]])
    with np.errstate(invalid="raise"):
        result = masked_pullback(seeds, np.array([np.inf, -np.inf]))
        scalar = masked_pullback(0.0, np.inf)
    assert result[0, 0] == result[1, 1] == float(scalar) == 0.0
    assert np.isneginf(result[0, 1])
    assert np.isneginf(result[1, 0])
    assert scalar.shape == ()
    assert masked_pullback(np.array([2.0, -3.0]), np.array([4.0, 5.0])) == pytest.approx(
        [8.0, -15.0]
    )


@pytest.mark.parametrize(
    ("intervals", "empty", "single", "members"),
    [
        ((), True, False, []),
        (((2.0, 2.0),), False, True, [2.0]),
        (((-np.inf, 0.0),), False, False, [-3.0, 0.0]),
        (((-1.0, -1.0), (1.0, 1.0)), False, False, [-1.0, 1.0]),
    ],
)
def test_scalar_graph_response_preserves_empty_unbounded_and_nonconvex_sets(
    intervals: tuple[tuple[float, float], ...], empty: bool, single: bool, members: list[float]
) -> None:
    """
    @brief Finite tangent outputs can be empty, singleton, unbounded, or nonconvex.
    """
    response = ScalarGraphResponse(intervals)
    assert response.empty == empty
    assert response.single_valued == single
    assert all(response.contains(value) for value in members)
    assert not response.contains(np.inf)
    assert not response.contains(np.nan)
    if len(intervals) == 2:
        assert not response.contains(0.0)


@pytest.mark.parametrize("intervals", [((1.0, 0.0),), ((np.inf, np.inf),), ((np.nan, 1.0),)])
def test_scalar_graph_response_refuses_invalid_intervals(
    intervals: tuple[tuple[float, float], ...],
) -> None:
    """
    @brief Interval endpoints describe nonempty sets of finite real values.
    """
    with pytest.raises(ValueError):
        ScalarGraphResponse(intervals)


def test_an_inactive_singular_branch_keeps_native_lazy_domain_semantics(
    scalar_coefficient: Function,
) -> None:
    """
    @brief An inactive square-root branch imposes no false singularity on the selected identity.
    """
    coefficient = scalar_coefficient
    gate = Function(coefficient.function_space())
    expression = conditional(gate > 0.0, sqrt(coefficient), coefficient)
    response = directional(expression, coefficient, as_ufl(2.0))
    gate.assign(-1.0)
    coefficient.assign(0.0)
    assert isinstance(response, Expr)
    result = Function(coefficient.function_space()).interpolate(response)
    assert result.dat.data_ro == pytest.approx(2.0)


def test_native_interpolation_coordinate_action_retains_reference_dofs(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Scalar interpolation follows moving nodes and a Piola field retains its nodal response.
    """
    position = SpatialCoordinate(box_mesh)
    geometry_direction = Function(box_mesh.coordinates.function_space()).interpolate(position)
    scalar_space = FunctionSpace(box_mesh, "CG", 1)
    interpolation = Interpolate(position[0], scalar_space)
    response = assemble(directional(interpolation, position, geometry_direction))
    expected = Function(scalar_space).interpolate(position[0])
    assert response.dat.data_ro == pytest.approx(expected.dat.data_ro)
    piola_space = FunctionSpace(box_mesh, "RT", 1)
    vector = as_vector((1.0, 0.0, 0.0))
    piola_response = assemble(
        directional(Interpolate(vector, piola_space), position, geometry_direction)
    )
    expected_piola = Function(piola_space).interpolate(2.0 * vector)
    assert piola_response.dat.data_ro == pytest.approx(expected_piola.dat.data_ro)


def test_selected_empty_action_preserves_value_shape_and_form_arguments(
    scalar_coefficient: Function,
) -> None:
    """
    @brief An empty joint selected direction is zero in the original value or form space.
    """
    coefficient = scalar_coefficient
    vector = selected_directional(as_vector((coefficient, coefficient**2)), [], [])
    assert isinstance(vector, Expr)
    assert vector.ufl_shape == (2,)
    assert vector == as_vector((0.0, 0.0))
    form = coefficient * TestFunction(coefficient.function_space()) * dx
    action = selected_directional(form, [], [])
    assert isinstance(action, ZeroBaseForm)
    assert action.arguments() == form.arguments()
    operator = Interpolate(coefficient, coefficient.function_space())
    action = selected_directional(operator, [], [])
    assert isinstance(action, ZeroBaseForm)
    assert action.arguments() == operator.arguments()
    with pytest.raises(ValueError, match="one supplied direction"):
        selected_directional(coefficient, [], [as_ufl(1.0)])


def test_recorded_linearisation_and_hadamard_action_are_distinct_at_a_switch(
    scalar_coefficient: Function,
) -> None:
    """
    @brief Native tie linearisations remain linear while the true directional action changes branch.
    """
    coefficient = scalar_coefficient
    positive_part = max_value(coefficient, 0.0)
    selected = selected_directional(positive_part, [coefficient], [as_ufl(2.0)])
    hadamard = directional(positive_part, coefficient, as_ufl(2.0))
    point = {coefficient: as_ufl(0.0)}
    assert float(replace(selected, point)) == 0.0
    assert float(replace(hadamard, point)) == 2.0
    raw_absolute = selected_directional(abs(as_ufl(coefficient)), [coefficient], [as_ufl(2.0)])
    branch_absolute = selected_directional(absolute(coefficient), [coefficient], [as_ufl(2.0)])
    assert float(replace(raw_absolute, point)) == 0.0
    assert float(replace(branch_absolute, point)) == 2.0


@pytest.mark.parametrize(
    ("operation", "seed", "included", "excluded"),
    [
        ("maximum", 2.0, [0.0, 1.0, 2.0], [-1.0, 3.0]),
        ("maximum", -2.0, [-2.0, 0.0], [-1.0, 1.0]),
        ("minimum", 2.0, [0.0, 2.0], [1.0, -1.0]),
        ("minimum", -2.0, [-2.0, -1.0, 0.0], [1.0, -3.0]),
        ("absolute", 2.0, [-2.0, 0.0, 2.0], [-3.0, 3.0]),
        ("absolute", -2.0, [-2.0, 2.0], [0.0, 1.0]),
        ("clip_lower", -2.0, [-2.0, 0.0], [-1.0, 1.0]),
        ("clip_upper", 2.0, [0.0, 2.0], [1.0, -1.0]),
    ],
)
def test_scalar_basic_coderivatives_preserve_the_actual_reverse_sets(
    operation: str, seed: float, included: list[float], excluded: list[float]
) -> None:
    """
    @brief Analytical graph normals distinguish intervals from disconnected branch covectors.
    """
    responses = {
        "maximum": maximum_coderivative(0.0, seed),
        "minimum": minimum_coderivative(0.0, seed),
        "absolute": absolute_coderivative(0.0, seed),
        "clip_lower": clip_coderivative(0.0, 0.0, 1.0, seed),
        "clip_upper": clip_coderivative(1.0, 0.0, 1.0, seed),
    }
    response = responses[operation]
    assert all(response.contains(value) for value in included)
    assert all(not response.contains(value) for value in excluded)
    with pytest.raises(ValueError, match="unique"):
        response.value()


def test_scalar_reverse_relations_reduce_to_the_smooth_derivative() -> None:
    """
    @brief Smooth branches, zero output covectors, and collapsed clipping have unique results.
    """
    assert maximum_coderivative(-1.0, 3.0).value() == 0.0
    assert maximum_coderivative(3.0, 3.0, threshold=2.0).value() == 3.0
    assert minimum_coderivative(1.0, 3.0, threshold=2.0).value() == 3.0
    assert minimum_coderivative(3.0, 3.0, threshold=2.0).value() == 0.0
    assert absolute_coderivative(-1.0, 3.0).value() == -3.0
    assert absolute_coderivative(1.0, 3.0).value() == 3.0
    assert absolute_coderivative(0.0, 0.0).value() == 0.0
    assert clip_coderivative(0.5, 0.0, 1.0, 3.0).value() == 3.0
    assert clip_coderivative(1.0, 1.0, 1.0, 3.0).value() == 0.0
    with pytest.raises(ValueError, match="unique"):
        ScalarGraphResponse(()).value()


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_scalar_reverse_relations_refuse_nonfinite_data(invalid: float) -> None:
    """
    @brief Invalid local data cannot manufacture a finite reverse relation.
    """
    with pytest.raises(ValueError, match="finite"):
        maximum_coderivative(invalid, 1.0)
    with pytest.raises(ValueError, match="finite"):
        minimum_coderivative(0.0, invalid)
    with pytest.raises(ValueError, match="finite"):
        absolute_coderivative(invalid, 1.0)
    with pytest.raises(ValueError, match="finite"):
        clip_coderivative(0.0, invalid, 1.0, 1.0)


@pytest.mark.parametrize(
    ("sign", "expected"), [(1.0, [-1.0, 2.0, 2.0, 1.0, 1.0]), (-1.0, [1.0, 1.0, -2.0, -2.0, -1.0])]
)
def test_array_clipping_directional_compares_moving_bounds_at_ties(
    sign: float, expected: list[float]
) -> None:
    """
    @brief One-sided clipping responses retain input, lower-bound, and upper-bound perturbations.
    """
    values = np.array([-1.0, 0.0, 1.0, 2.0, 3.0])
    action = clip_directional(values, 0.0, 2.0, 2.0 * sign, -sign, sign)
    assert action == pytest.approx(expected)
    step = 1e-6
    difference = (
        np.clip(values + step * 2.0 * sign, -step * sign, 2.0 + step * sign)
        - np.clip(values, 0.0, 2.0)
    ) / step
    assert action == pytest.approx(difference, abs=1e-9)


@pytest.mark.parametrize(
    ("input_direction", "lower_direction", "upper_direction", "expected"),
    [(2.0, -1.0, 1.0, 1.0), (-2.0, -1.0, 1.0, -1.0), (0.0, -1.0, 1.0, 0.0), (0.0, 1.0, 2.0, 1.0)],
)
def test_coincident_clipping_bounds_admit_only_order_preserving_directions(
    input_direction: float, lower_direction: float, upper_direction: float, expected: float
) -> None:
    """
    @brief A collapsed interval opens according to its two bound directions.
    """
    response = clip_directional(0.0, 0.0, 0.0, input_direction, lower_direction, upper_direction)
    assert float(response) == expected
