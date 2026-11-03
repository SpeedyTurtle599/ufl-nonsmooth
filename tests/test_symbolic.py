# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_symbolic.py
@brief Shared native local-law values, joint actions, and selected transposes.
@details A maximum law distinguishes a nonlinear directional response from its selected
    linearisation at an equal-operand switch.
"""

import numpy as np
import pytest
from firedrake import (
    Cofunction,
    Function,
    FunctionSpace,
    Interpolate,
    MeshGeometry,
    action,
    adjoint,
    assemble,
    derivative,
)
from ufl import as_ufl, conditional, eq, max_value
from ufl.core.expr import Expr

from ufl_nonsmooth.application import apply_law
from ufl_nonsmooth.laws import LocalEvaluation, SymbolicLaw
from ufl_nonsmooth.primitives import directional, selected_directional


class MaximumLaw(SymbolicLaw):
    """
    @brief Maximum of two independent native scalar operands.
    """

    def evaluate(self, inputs: list[Expr]) -> LocalEvaluation[Expr]:
        """
        @brief Select the second operand at an equal maximum.
        """
        left, right = inputs
        weight = conditional(left > right, 1.0, 0.0)
        return LocalEvaluation([max_value(left, right)], [[weight, 1.0 - weight]])

    def directional(self, inputs: list[Expr], directions: list[Expr | None]) -> list[Expr]:
        """
        @brief Compare both incoming perturbations at an equal maximum.
        """
        left, right = inputs
        first = as_ufl(0.0) if directions[0] is None else directions[0]
        second = as_ufl(0.0) if directions[1] is None else directions[1]
        return [
            conditional(
                eq(left, right), max_value(first, second), conditional(left > right, first, second)
            )
        ]


def test_native_joint_action_and_selected_transpose_have_distinct_switch_semantics(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A joint maximum action compares directions while its selected transpose uses one branch.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first, second = Function(space), Function(space)
    directions = [Function(space).assign(1.0), Function(space).assign(-1.0)]
    expression = apply_law(MaximumLaw(), [first, second], function_space=space)[0]
    response = assemble(directional(expression, [first, second], directions))
    selected = assemble(selected_directional(expression, [first, second], directions))
    assert np.allclose(response.dat.data_ro, 1.0)
    assert np.allclose(selected.dat.data_ro, -1.0)
    seed = Cofunction(space.dual()).assign(2.0)
    first_pullback = assemble(action(adjoint(derivative(expression, first)), seed))
    second_pullback = assemble(action(adjoint(derivative(expression, second)), seed))
    assert np.allclose(first_pullback.dat.data_ro, 0.0)
    assert np.allclose(second_pullback.dat.data_ro, 2.0)
    with directions[1].dat.vec_ro as vector, second_pullback.dat.vec_ro as covector:
        pairing = covector.dot(vector)
    with selected.dat.vec_ro as vector, seed.dat.vec_ro as covector:
        assert pairing == pytest.approx(covector.dot(vector))


def test_native_real_operand_composes_through_interpolation(box_mesh: MeshGeometry) -> None:
    """
    @brief A shared scalar's selected Jacobian composes with a native interpolation matrix.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    real = FunctionSpace(box_mesh, "R", 0)
    first = Function(real).assign(3.0)
    second = Function(space).assign(2.0)
    expression = apply_law(MaximumLaw(), [first, second], function_space=space)[0]
    outer = Interpolate(expression, space)
    variation = derivative(outer, first)
    matrix = assemble(variation)
    direction = Function(real).assign(1.5)
    response = Function(space)
    with direction.dat.vec_ro as vector, response.dat.vec as output:
        matrix.petscmat.mult(vector, output)
    assert np.allclose(response.dat.data_ro, 1.5)
    seed = Cofunction(space.dual()).assign(2.0)
    pullback = assemble(action(adjoint(variation), seed))
    expected = box_mesh.comm.allreduce(float(seed.dat.data_ro.sum()))
    assert np.allclose(pullback.dat.data_ro, expected)


def test_native_local_law_declares_its_output_pullback_contract(box_mesh: MeshGeometry) -> None:
    """
    @brief A non-identity output pullback requires a different local-space action contract.
    """
    space = FunctionSpace(box_mesh, "DG L2", 0)
    operand = Function(space).assign(1.0)
    with pytest.raises(ValueError, match="identity output pullback"):
        apply_law(MaximumLaw(), [operand, operand], function_space=space)
