# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_laws.py
@brief Shared local joins and selected evaluation contracts.
@details Analytical branch actions exercise exact switching coordinates, moving guards,
    broadcast operands, and finite smooth incident branches.
"""

import numpy as np
import pytest
from numpy.typing import NDArray
from ufl import as_ufl, max_value
from ufl.core.expr import Expr
from ufl.operators import variable

from ufl_nonsmooth.laws import LocalEvaluation, join, ordered_branch
from ufl_nonsmooth.primitives import directional, selected_directional


@pytest.mark.parametrize(
    ("margin", "rate", "expected"),
    [
        (-1.0, 3.0, -2.0),
        (1.0, -3.0, 4.0),
        (0.0, -1.0, -2.0),
        (0.0, 1.0, 4.0),
        (0.0, 0.0, 4.0),
        (0.0, None, 4.0),
    ],
)
def test_join_uses_the_joint_margin_action(
    margin: float, rate: float | None, expected: float
) -> None:
    """
    @brief Incident actions follow the sign of the full switching-margin perturbation.
    """
    assert float(join(margin, -2.0, 4.0, rate)) == expected
    symbolic = join(as_ufl(margin), as_ufl(-2.0), as_ufl(4.0), rate)
    assert float(symbolic) == expected


def test_join_broadcasts_independent_local_queries() -> None:
    """
    @brief Base sides and tie directions select incident actions independently at each point.
    """
    margins = np.array([[-1.0], [0.0], [1.0]])
    rates = np.array([-1.0, 0.0, 1.0])
    result = join(margins, 2.0, 5.0, rates)
    assert np.array_equal(result, [[2.0, 2.0, 2.0], [2.0, 5.0, 5.0], [5.0, 5.0, 5.0]])


def test_ordered_branches_share_exact_coordinate_ties() -> None:
    """
    @brief Ordered intervals select their left or right jet only at exact switching coordinates.
    """
    knots = np.array([1.0, 3.0])
    points = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    assert np.array_equal(ordered_branch(knots, points), [0, 1, 1, 2, 2])
    assert np.array_equal(ordered_branch(knots, points, -1.0), [0, 0, 1, 1, 2])
    assert np.array_equal(ordered_branch(knots, points, 0.0), [0, 1, 1, 2, 2])


@pytest.mark.parametrize("knots", [np.array([2.0, 1.0]), np.array([1.0, 1.0]), np.array([np.nan])])
def test_ordered_branches_refuse_invalid_switching_coordinates(knots: NDArray[np.float64]) -> None:
    """
    @brief Branch coordinates must be finite and strictly increasing.
    """
    with pytest.raises(ValueError, match="breakpoints"):
        ordered_branch(knots, 0.0)


@pytest.mark.parametrize(
    ("values", "partials"),
    [([], []), ([1.0], []), ([1.0, 2.0], [[1.0], [1.0, 2.0]])],
)
def test_selected_evaluation_requires_a_complete_rectangular_map(
    values: list[float], partials: list[list[float]]
) -> None:
    """
    @brief Selected linearisation rows describe every output over one common operand list.
    """
    with pytest.raises(ValueError):
        LocalEvaluation(values, partials)


@pytest.mark.parametrize(("rate", "expected"), [(-2.0, 0.0), (0.0, 0.0), (3.0, 3.0)])
def test_formal_local_operands_have_joint_and_selected_native_actions(
    rate: float, expected: float
) -> None:
    """
    @brief Formal scalar operands retain the actual positive-part action at a local kink.
    """
    point = variable(as_ufl(0.0))
    expression = max_value(point, 0.0)
    action = directional(expression, point, as_ufl(rate))
    selected = selected_directional(expression, [point], [as_ufl(rate)])
    assert isinstance(action, Expr) and isinstance(selected, Expr)
    assert float(action) == pytest.approx(expected)
    assert float(selected) == 0.0


def test_formal_operands_refuse_overlapping_derivative_targets() -> None:
    """
    @brief Nested local variables cannot be independent coordinates of the same request.
    """
    inner = variable(as_ufl(2.0))
    outer = variable(inner**2)
    with pytest.raises(ValueError, match="independent derivative targets"):
        directional(outer, [inner, outer], [as_ufl(1.0), as_ufl(1.0)])
    with pytest.raises(ValueError, match="independent derivative targets"):
        selected_directional(outer, [inner, outer], [as_ufl(1.0), as_ufl(1.0)])
