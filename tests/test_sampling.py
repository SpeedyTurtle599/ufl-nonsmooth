# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_sampling.py
@brief Native operand activity and finite sampled branch contracts.
@details Strict branches retain their values and analytical actions, while active events and
    undeclared sampling are refused. Native dependency selection distinguishes zero from absence.
"""

import numpy as np
import pytest
from firedrake import Function, FunctionSpace, MeshGeometry, assemble, dx
from firedrake.adjoint import (
    Control,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)
from ufl import conditional

from ufl_nonsmooth.primitives import directional
from ufl_nonsmooth.sampling import (
    active_dependencies,
    operation_directions,
    sampled_stratum,
)


@pytest.mark.parametrize("point", [-2.0, 2.0])
def test_strict_sampled_branch_retains_its_value_and_action(
    box_mesh: MeshGeometry, point: float
) -> None:
    """
    @brief Degree-zero absolute-value branches have the expected value and signed response.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(point)
    form = conditional(source > 0.0, source, -source) * dx(degree=0)
    frozen = sampled_stratum(form, [source])
    assert assemble(frozen) == pytest.approx(assemble(form))
    response = directional(frozen, source, Function(space).assign(1.0))
    volume = assemble(1.0 * dx(domain=box_mesh))
    assert assemble(response) == pytest.approx(np.sign(point) * volume)


@pytest.mark.parametrize("point", [0.0, np.nan])
def test_sampled_event_or_nonfinite_margin_is_refused(box_mesh: MeshGeometry, point: float) -> None:
    """
    @brief Active or non-finite comparisons do not become fixed native predicates.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(point)
    form = conditional(source > 0.0, source, -source) * dx(degree=0)
    error = NotImplementedError if point == 0.0 else ValueError
    with pytest.raises(error, match="active sampled predicate|non-finite comparison"):
        sampled_stratum(form, [source])


def test_native_operands_keep_active_zero_and_omit_frozen_inputs(box_mesh: MeshGeometry) -> None:
    """
    @brief Actual recorded assembly dependencies preserve zero perturbations and independent cuts.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(2.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        assemble(source * dx)
        pause_annotation()
        block = tape.get_blocks()[-1]
        control.tlm_value = Function(space)
        assert active_dependencies(block) == [control.block_variable]
        coefficients, directions = operation_directions(block)
        assert len(coefficients) == len(directions) == 1
        assert coefficients[0] is control.block_variable.saved_output
        assert isinstance(directions[0], Function)
        assert np.allclose(directions[0].dat.data_ro, 0.0)
        assert active_dependencies(block, source) == []
        control.tlm_value = None
        assert active_dependencies(block) == []
