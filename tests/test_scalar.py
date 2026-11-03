# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_scalar.py
@brief Shared scalar local-law recording, joint actions, replay, and selected pullbacks.
@details Analytical charts and sum-product laws exercise multiple operands and outputs,
    repeated controls, and exact local scalar reverse queries through the common recorder.
"""

import numpy as np
import pytest
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    continue_annotation,
    get_working_tape,
    pause_annotation,
)
from pyadjoint import AdjFloat

from tests.fixtures.laws import CoupledLaw
from ufl_nonsmooth.piecewise import PiecewiseLinear
from ufl_nonsmooth.scalar import evaluate_scalar, scalar_coderivative
from ufl_nonsmooth.tape import evaluate_directional


@pytest.mark.parametrize(
    ("time_direction", "knot_direction", "expected"),
    [(1.0, 1.0, 3.0), (0.0, 1.0, 1.0), (-1.0, 0.0, 1.0), (1.0, 0.0, 2.0)],
)
def test_chart_joint_moving_knot_response(
    time_direction: float, knot_direction: float, expected: float
) -> None:
    """
    @brief Relative query and knot motion selects the adjacent slope with the moving ordinate.
    """
    tape = get_working_tape()
    tape.clear_tape()
    time, knot, ordinate = AdjFloat(1.0), AdjFloat(1.0), AdjFloat(2.0)
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
    continue_annotation()
    try:
        result = evaluate_scalar(chart, [time, 0.0, 0.0, knot, ordinate, 2.0, 1.0])[0]
        assert isinstance(result, AdjFloat)
        evaluate_directional(
            tape,
            [Control(time), Control(knot), Control(ordinate)],
            [time_direction, knot_direction, 3.0],
        )
        assert result.block_variable.tlm_value == pytest.approx(expected)
    finally:
        pause_annotation()
        tape.clear_tape()


def test_chart_replay_and_selected_adjoint_share_runtime_rows() -> None:
    """
    @brief Replay and the selected reverse branch retain knot and ordinate controls.
    """
    tape = get_working_tape()
    tape.clear_tape()
    time, knot, ordinate = AdjFloat(0.5), AdjFloat(1.0), AdjFloat(2.0)
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
    continue_annotation()
    try:
        result = evaluate_scalar(chart, [time, 0.0, 0.0, knot, ordinate, 2.0, 1.0])[0]
        functional = ReducedFunctional(result, [Control(time), Control(knot), Control(ordinate)])
        assert functional.derivative() == pytest.approx([2.0, -1.0, 0.5])
        assert functional([0.5, 1.5, 3.0]) == pytest.approx(1.0)
    finally:
        pause_annotation()
        tape.clear_tape()


def test_repeated_scalar_control_accumulates_every_operand_position() -> None:
    """
    @brief Moving the query with its lower knot leaves the lower ordinate unchanged.
    """
    tape = get_working_tape()
    tape.clear_tape()
    point = AdjFloat(0.5)
    chart = PiecewiseLinear([(0.0, 1.0), (2.0, 3.0)])
    continue_annotation()
    try:
        value = evaluate_scalar(chart, [point, point, 1.0, 2.0, 3.0])[0]
        assert isinstance(value, AdjFloat)
        control = Control(point)
        evaluate_directional(tape, [control], [1.0])
        assert value.block_variable.tlm_value == pytest.approx(0.0)
        reduced = ReducedFunctional(value, control)
        assert reduced.derivative() == pytest.approx(0.0)
        assert reduced(1.0) == pytest.approx(1.0)
    finally:
        pause_annotation()
        tape.clear_tape()


def test_scalar_reverse_query_preserves_table_nonconvexity() -> None:
    """
    @brief A decreasing table slope keeps the two limiting covectors for a positive seed.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)])
    result = scalar_coderivative(chart, 1.0, [2.0])
    assert result.contains(2.0)
    assert result.contains(4.0)
    assert not result.contains(3.0)
    assert result.intervals == ((2.0, 2.0), (4.0, 4.0))


@pytest.mark.parametrize(("point", "seed"), [(np.nan, 1.0), (0.0, np.inf)])
def test_scalar_reverse_refuses_nonfinite_coordinates_and_covectors(
    point: float, seed: float
) -> None:
    """
    @brief Exact local reverse queries require finite inputs and output covectors.
    """
    law = PiecewiseLinear([(0.0, 0.0), (1.0, 1.0)])
    with pytest.raises(ValueError, match="must be finite"):
        scalar_coderivative(law, point, [seed])


def test_multioutput_scalar_law_retains_every_observed_output() -> None:
    """
    @brief The sum-product objective has analytical partials seven and five at two and three.
    """
    tape = get_working_tape()
    tape.clear_tape()
    first, second = AdjFloat(2.0), AdjFloat(3.0)
    continue_annotation()
    try:
        total, product = evaluate_scalar(CoupledLaw(), [first, second])
        objective = total + 2.0 * product
        assert isinstance(objective, AdjFloat)
        controls = [Control(first), Control(second)]
        pause_annotation()
        evaluate_directional(tape, controls, [0.25, -0.5])
        assert objective.block_variable.tlm_value == pytest.approx(-0.75)
        functional = ReducedFunctional(objective, controls)
        assert functional.derivative() == pytest.approx([7.0, 5.0])
        assert functional([-1.0, 4.0]) == pytest.approx(-5.0)
    finally:
        pause_annotation()
        tape.clear_tape()


def test_scalar_reverse_requires_the_declared_local_capability() -> None:
    """
    @brief A smooth array law does not declare a scalar-input two-branch reverse relation.
    """
    with pytest.raises(TypeError, match="declared finite two-sided law"):
        scalar_coderivative(CoupledLaw(), 2.0, [1.0])  # ty: ignore[invalid-argument-type]
