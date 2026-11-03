# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_field.py
@brief Numerical local-law field responses, selected pullbacks, and replay.
@details A two-output analytical law checks joint actions and shared real-space reductions on
    native mesh communicators.
"""

import numpy as np
import pytest
from firedrake import Function, FunctionSpace, MeshGeometry, assemble, dx
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)

from tests.fixtures.laws import CoupledLaw
from ufl_nonsmooth.application import apply_law
from ufl_nonsmooth.tape import evaluate_directional


@pytest.mark.parametrize("family", ["DG", "R"])
def test_multioutput_real_operand_retains_joint_response_pullback_and_replay(
    box_mesh: MeshGeometry, family: str
) -> None:
    """
    @brief One field operation retains all outputs and globally sums the shared scalar covector.
    """
    space = FunctionSpace(box_mesh, family, 0)
    real = FunctionSpace(box_mesh, "R", 0)
    first = Function(space).assign(2.0)
    second = Function(real).assign(3.0)
    outputs = [Function(space), Function(space)]
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(first), Control(second)]
        apply_law(CoupledLaw(), [first, second], outputs)
        objective = assemble((outputs[0] + 2.0 * outputs[1]) * dx)
        pause_annotation()
        assert np.allclose(outputs[0].dat.data_ro, 5.0)
        assert np.allclose(outputs[1].dat.data_ro, 6.0)
        evaluate_directional(
            tape, controls, [Function(space).assign(0.25), Function(real).assign(-0.5)]
        )
        assert np.allclose(outputs[0].block_variable.tlm_value.dat.data_ro, -0.25)
        assert np.allclose(outputs[1].block_variable.tlm_value.dat.data_ro, -0.25)
        assert float(objective.block_variable.tlm_value) == pytest.approx(-0.75 * volume)
        reduced = ReducedFunctional(objective, controls)
        gradients = reduced.derivative()
        unit = Function(space).assign(1.0)
        with gradients[0].dat.vec_ro as gradient, unit.dat.vec_ro as direction:
            assert gradient.dot(direction) == pytest.approx(7.0 * volume)
        assert np.allclose(gradients[1].dat.data_ro, 5.0 * volume)
        assert float(
            reduced([Function(space).assign(-1.0), Function(real).assign(4.0)])
        ) == pytest.approx(-5.0 * volume)


def test_numerical_local_law_declares_its_output_pullback_contract(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Numerical samples use identity-pullback scalar spaces rather than mapped coefficients.
    """
    space = FunctionSpace(box_mesh, "DG L2", 0)
    first, second = Function(space), Function(space)
    with pytest.raises(ValueError, match="identity output pullback"):
        apply_law(CoupledLaw(), [first, second], [Function(space), Function(space)])
