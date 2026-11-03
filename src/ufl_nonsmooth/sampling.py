# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/sampling.py
@brief Active native operands and separated sampled branch predicates.
@details Native tape dependencies retain active zero perturbations. Explicit quadrature and
    interpolation samples define the finite programme on which moving predicates can be frozen.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from firedrake import (
    Cofunction,
    Constant,
    DirichletBC,
    Function,
    MeshGeometry,
    SpatialCoordinate,
    assemble,
)
from mpi4py import MPI
from ufl import BaseForm, Form, as_ufl, conditional, eq, replace
from ufl.algorithms.analysis import extract_coefficients
from ufl.algorithms.apply_restrictions import apply_restrictions
from ufl.classes import EQ, GE, GT, LE, LT, NE, Coefficient, Conditional, Integral, Interpolate
from ufl.core.expr import Expr
from ufl.corealg.traversal import unique_post_traversal

from ufl_nonsmooth.primitives import DerivativeTarget

if TYPE_CHECKING:
    from pyadjoint import Block
    from pyadjoint.block_variable import BlockVariable


def active_dependencies(block: Block, exclude: Function | None = None) -> list[BlockVariable]:
    """
    @brief Retain varying value operands and omit a designated nonlinear initial guess.
    @details Boundary perturbations belong to their native boundary conditions. An explicit
        zero direction remains active, while an absent direction means the operand is frozen.
    """
    return [
        dependency
        for dependency in block.get_dependencies()
        if dependency.tlm_value is not None
        and not isinstance(dependency.output, DirichletBC)
        and dependency.output is not exclude
    ]


def operation_directions(
    block: Block,
    constants: dict[Constant, Function] | None = None,
    exclude: Function | None = None,
) -> tuple[list[DerivativeTarget], list[Expr]]:
    """
    @brief Collect the active saved coefficients and their joint incoming directions.
    @details Mesh dependencies use native spatial-coordinate targets. Implicit scalar Constants
        use the supplied value-preserving coefficient proxies. Boundary conditions are handled
        by their implicit solve, and have no coefficient entry in the residual.
    """
    coefficients: list[DerivativeTarget] = []
    directions: list[Expr] = []
    for dependency in active_dependencies(block, exclude):
        perturbation = dependency.tlm_value
        coefficient = dependency.saved_output
        if isinstance(coefficient, Constant) and constants is not None:
            coefficient = constants[coefficient]
        if isinstance(coefficient, MeshGeometry):
            coefficient = SpatialCoordinate(coefficient)
        if not isinstance(coefficient, (Coefficient, SpatialCoordinate)):
            raise NotImplementedError(
                f"a {type(coefficient).__name__} dependency has no native expression direction"
            )
        coefficients.append(coefficient)
        if isinstance(perturbation, Cofunction):
            perturbation = perturbation.riesz_representation("l2")
        directions.append(as_ufl(perturbation))
    return coefficients, directions


def sampled_stratum(
    expression: Expr | BaseForm, coefficients: list[DerivativeTarget]
) -> Expr | BaseForm:
    """
    @brief Admit predicates separated from events in an explicitly sampled native operation.
    @details Weak forms require piecewise-constant active fields and explicit degree-zero
        quadrature. Scalar interpolation checks its actual target nodes. Each comparison is
        sampled with the original domain, subdomain, and integration or interpolation metadata.
        Strictly separated predicates retain saved values during differentiation. Equality at
        a sample requires a graphical event response. This is a finite programme contract,
        without a continuum moving-interface claim.
    """
    if isinstance(expression, Form):
        expression = Form(
            [
                integral.reconstruct(integrand=apply_restrictions(integral.integrand()))
                for integral in expression.integrals()
            ]
        )
        samples = expression.integrals()
    elif isinstance(expression, Interpolate):
        samples = [expression]
    else:
        return expression
    snapshots: dict[Expr, Expr] = {}
    replacements: dict[Expr, Expr] = {}
    for sample in samples:
        integrand = sample.integrand() if isinstance(sample, Integral) else sample.ufl_operands[0]
        predicates = [
            node.ufl_operands[0]
            for node in unique_post_traversal(integrand)
            if isinstance(node, Conditional)
            and any(target in extract_coefficients(node.ufl_operands[0]) for target in coefficients)
        ]
        for predicate in predicates:
            if isinstance(sample, Integral) and sample.metadata().get("quadrature_degree") != 0:
                raise NotImplementedError(
                    "moving predicates require an explicitly declared degree-zero sampling contract"
                )
            if isinstance(sample, Interpolate) and sample.ufl_shape:
                raise NotImplementedError(
                    "sampled predicate interpolation requires a scalar output"
                )
            active = [
                target for target in coefficients if target in extract_coefficients(predicate)
            ]
            active_fields: list[Function] = []
            for target in active:
                if not isinstance(target, Function):
                    raise NotImplementedError("sampled predicate admission requires native fields")
                if isinstance(sample, Integral) and target.ufl_element().degree() != 0:
                    raise NotImplementedError(
                        "sampled weak-form predicates require piecewise-constant active fields"
                    )
                active_fields.append(target)
            for comparison in unique_post_traversal(predicate):
                if not isinstance(comparison, (EQ, NE, LE, LT, GE, GT)):
                    continue
                if any(target in extract_coefficients(comparison) for target in active):
                    validate_predicate_sample(comparison, sample)
            for target in active_fields:
                if target not in snapshots:
                    snapshots[target] = target.copy(deepcopy=True)
            replacements[predicate] = replace(predicate, snapshots)
    return replace(expression, replacements) if replacements else expression


def validate_predicate_sample(comparison: Expr, sample: Integral | Interpolate) -> None:
    """
    @brief Check finite strict comparison margins at an operation's exact evaluation samples.
    """
    left, right = comparison.ufl_operands
    margin = left - right
    event = conditional(eq(left, right), 1.0, 0.0)
    if isinstance(sample, Integral):
        squared = assemble(Form([sample.reconstruct(integrand=margin**2)]))
        if not np.isfinite(float(squared)):
            raise ValueError("a sampled predicate has a non-finite comparison margin")
        active = float(assemble(Form([sample.reconstruct(integrand=event)]))) > 0.0
    else:
        squared = assemble(replace(sample, {sample.ufl_operands[0]: margin**2}))
        events = assemble(replace(sample, {sample.ufl_operands[0]: event}))
        if not isinstance(squared, Function) or not isinstance(events, Function):
            raise NotImplementedError("sampled predicates require a field interpolation output")
        comm = events.function_space().mesh().comm
        if not comm.allreduce(bool(np.isfinite(squared.dat.data_ro).all()), op=MPI.LAND):
            raise ValueError("a sampled predicate has a non-finite comparison margin")
        active = comm.allreduce(bool(np.any(events.dat.data_ro > 0.0)), op=MPI.LOR)
    if active:
        raise NotImplementedError("an active sampled predicate requires a graphical event response")
