# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_piecewise.py
@brief Tests for continuous piecewise-linear charts.
@details Hand-calculated affine segments check values, held endpoints, directional responses,
    and the nonconvex scalar basic coderivative. Native Unified Form Language differentiation
    checks the right slope at each knot and zero slope outside the chart.
"""

import numpy as np
import pytest
from firedrake import Function, FunctionSpace, MeshGeometry
from numpy.typing import NDArray
from ufl import as_ufl, diff, replace
from ufl.algorithms.ad import expand_derivatives
from ufl.operators import variable

from ufl_nonsmooth.piecewise import PiecewiseLinear
from ufl_nonsmooth.primitives import directional
from ufl_nonsmooth.scalar import scalar_coderivative


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(0.0, 1.0)],
        [(0.0, 1.0), (0.0, 2.0)],
        [(1.0, 1.0), (0.0, 2.0)],
        [(0.0, 1.0), (float("inf"), 2.0)],
        [(0.0, 1.0), (1.0, float("nan"))],
    ],
)
def test_chart_refuses_invalid_rows(rows: list[tuple[float, float]]) -> None:
    """
    @brief A chart needs at least two finite rows in strictly increasing coordinate order.
    """
    with pytest.raises(ValueError):
        PiecewiseLinear(rows)


def test_chart_owns_immutable_data() -> None:
    """
    @brief Changing input rows cannot change a chart and its array data cannot be written.
    """
    rows = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)]
    chart = PiecewiseLinear(rows)
    rows[1] = (1.0, 100.0)
    assert float(chart.evaluate([np.asarray(1.0, dtype=np.float64)]).values[0]) == 3.0
    for data in (chart.coordinates, chart.values, chart.segment_slopes):
        with pytest.raises(ValueError, match="read-only"):
            data[0] = 100.0


def test_values_interpolate_each_segment_and_hold_the_ends() -> None:
    """
    @brief Slopes two and minus one give the segment midpoints and the chart holds both ends.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    points = np.array([[-1.0, 0.0, 0.5], [1.0, 1.5, 3.0]])
    assert chart.evaluate([np.asarray(points, dtype=np.float64)]).values[0] == pytest.approx(
        np.array([[1.0, 1.0, 2.0], [3.0, 2.5, 2.0]])
    )
    scalar = chart.evaluate([np.asarray(0.5, dtype=np.float64)]).values[0]
    assert scalar.shape == ()
    assert float(scalar) == 2.0


def test_one_sided_slopes_include_the_held_endpoint_branches() -> None:
    """
    @brief Each knot retains its two adjacent slopes, including zero on the held branches.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    left, right = chart.slopes(np.array([-1.0, 0.0, 0.5, 1.0, 1.5, 2.0, 3.0]))
    assert left == pytest.approx([0.0, 0.0, 2.0, 2.0, -1.0, -1.0, 0.0])
    assert right == pytest.approx([0.0, 2.0, 2.0, -1.0, -1.0, 0.0, 0.0])
    left_scalar, right_scalar = chart.slopes(1.0)
    assert left_scalar.shape == right_scalar.shape == ()


def test_directional_responses_broadcast_and_follow_the_perturbed_side() -> None:
    """
    @brief Negative directions use the left segment and positive directions use the right one.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    points = np.array([[0.0], [1.0], [2.0]])
    directions = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    expected = np.array(
        [[0.0, 0.0, 0.0, 2.0, 4.0], [-4.0, -2.0, 0.0, -1.0, -2.0], [2.0, 1.0, 0.0, 0.0, 0.0]]
    )
    response = chart.directional(
        [np.asarray(points, dtype=np.float64)], [np.asarray(directions, dtype=np.float64)]
    )[0]
    assert response == pytest.approx(expected)
    step = 1e-6
    difference = (
        chart.evaluate([np.asarray(points + step * directions, dtype=np.float64)]).values[0]
        - chart.evaluate([np.asarray(points, dtype=np.float64)]).values[0]
    ) / step
    assert response == pytest.approx(difference, abs=1e-8)


@pytest.mark.parametrize(
    ("point", "seed", "expected"),
    [
        (0.0, 1.0, [False, False, False, False, True, True, True, True, False]),
        (0.0, -1.0, [False, True, False, False, True, False, False, False, False]),
        (1.0, 1.0, [False, False, True, False, False, False, False, True, False]),
        (1.0, -1.0, [False, True, True, True, True, True, True, False, False]),
        (2.0, 1.0, [False, False, True, True, True, False, False, False, False]),
        (2.0, -1.0, [False, False, False, False, True, False, True, False, False]),
        (0.5, -1.0, [False, True, False, False, False, False, False, False, False]),
    ],
)
def test_basic_coderivative_keeps_ordered_intervals_and_limiting_branches(
    point: float, seed: float, expected: list[bool]
) -> None:
    """
    @brief A convex corner includes an interval for a positive seed and only endpoints for a
        negative seed, with the order reversed at a concave corner.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    candidates = np.array([-4.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0])
    assert chart.coderivative_contains(point, seed, candidates).tolist() == expected


def test_basic_coderivative_broadcasts_points_seeds_and_candidates() -> None:
    """
    @brief Reverse membership is evaluated independently for each broadcast scalar relation.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 1.0)])
    points = np.array([[0.0], [1.0]])
    seeds = np.array([1.0, -1.0])
    candidates = np.array([[0.5, -0.5], [0.5, -0.5]])
    assert chart.coderivative_contains(points, seeds, candidates).tolist() == [
        [True, False],
        [False, True],
    ]
    scalar = chart.coderivative_contains(0.0, -1.0, -0.5)
    assert scalar.shape == ()
    assert not bool(scalar)


@pytest.mark.parametrize(
    ("point", "expected", "slope"),
    [
        (-1e20, 1.0, 0.0),
        (0.0, 1.0, 2.0),
        (0.5, 2.0, 2.0),
        (1.0, 3.0, -1.0),
        (1.5, 2.5, -1.0),
        (2.0, 2.0, 0.0),
        (1e20, 2.0, 0.0),
    ],
)
def test_native_expression_has_the_chart_values_and_right_selected_slopes(
    point: float, expected: float, slope: float
) -> None:
    """
    @brief Native symbolic differentiation selects the right segment at knots and held branches
        at exterior coordinates, including coordinates far outside the chart.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    coordinate = variable(as_ufl(point))
    expression = chart.expression(coordinate)
    assert float(expression) == pytest.approx(expected)
    assert float(expand_derivatives(diff(expression, coordinate))) == pytest.approx(slope)


def test_numeric_and_native_values_share_the_same_chart() -> None:
    """
    @brief A nonuniform chart evaluates identically through its numerical and symbolic readings.
    """
    chart = PiecewiseLinear([(-2.0, 4.0), (-0.5, 1.0), (3.0, 2.0), (7.0, -5.0)])
    points: NDArray[np.float64] = np.linspace(-4.0, 9.0, 101)
    expected = chart.evaluate([np.asarray(points, dtype=np.float64)]).values[0]
    symbolic = np.array([float(chart.expression(as_ufl(point))) for point in points])
    assert symbolic == pytest.approx(expected, abs=1e-13)


@pytest.mark.parametrize(
    "rows",
    [
        [(0.0, 1.0), (1.0, 1e16), (2.0, 1.0)],
        [(0.0, 1e16), (1.0, 1.0), (2.0, 1.0)],
        [(0.0, 1.0), (1.0, -1e16), (2.0, 2.0)],
    ],
)
def test_native_values_preserve_nodes_with_large_neighbouring_values(
    rows: list[tuple[float, float]],
) -> None:
    """
    @brief Nodal values and held endpoints are exact even when adjacent values differ by sixteen
        orders of magnitude.
    """
    chart = PiecewiseLinear(rows)
    for coordinate, value in rows:
        assert float(chart.expression(as_ufl(coordinate))) == value
    assert float(chart.expression(as_ufl(-1e20))) == rows[0][1]
    assert float(chart.expression(as_ufl(1e20))) == rows[-1][1]


@pytest.mark.parametrize("point", [-1.0, 0.0, 0.4, 1.0, 1.4, 2.0, 3.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_moving_chart_knots_and_ordinates_have_joint_one_sided_responses(
    point: float, sign: float
) -> None:
    """
    @brief Relative input and knot velocity selects the correct segment at every chart boundary.
    """
    rows = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)]
    rates = [(0.2 * sign, 0.4 * sign), (-0.3 * sign, 0.7 * sign), (0.5 * sign, -0.8 * sign)]
    chart = PiecewiseLinear(rows)
    response = chart.directional(
        [
            np.asarray(point, dtype=np.float64),
            *[np.asarray(value) for row in rows for value in row],
        ],
        [
            np.asarray(sign, dtype=np.float64),
            *[np.asarray(value) for row in rates for value in row],
        ],
    )[0]
    step = 1e-7
    perturbed = [
        (x + step * dx, y + step * dy) for (x, y), (dx, dy) in zip(rows, rates, strict=True)
    ]
    difference = (
        chart.evaluate(
            [
                np.asarray(point + step * sign, dtype=np.float64),
                *[np.asarray(value) for row in perturbed for value in row],
            ]
        ).values[0]
        - chart.evaluate([np.asarray(point, dtype=np.float64)]).values[0]
    ) / step
    assert float(response) == pytest.approx(float(difference), abs=1e-6)


@pytest.mark.parametrize("point", [-1.0, 0.0, 0.4, 1.0, 1.4, 2.0, 3.0])
def test_chart_selected_linearisation_has_all_parameter_weights(point: float) -> None:
    """
    @brief The selected branch pairs with all input, knot, and ordinate perturbations.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)])
    evaluation = chart.evaluate(
        [
            np.asarray(point),
            *[
                np.asarray(value)
                for row in zip(chart.coordinates, chart.values, strict=True)
                for value in row
            ],
        ]
    )
    slope = float(evaluation.partials[0][0])
    knots = np.asarray(evaluation.partials[0][1::2])
    values = np.asarray(evaluation.partials[0][2::2])
    assert float(np.sum(values)) == pytest.approx(1.0)
    assert float(np.sum(knots)) == pytest.approx(-slope)
    if point >= 2.0:
        assert slope == 0.0
        assert values[-1] == 1.0
    if point == 1.0:
        assert slope == -1.0
        assert knots[1] == 1.0
        assert values[1] == 1.0


@pytest.mark.parametrize("point", [0.0, 0.4, 1.0, 1.4, 2.0])
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_native_parametric_chart_matches_joint_numeric_response(
    box_mesh: MeshGeometry, point: float, sign: float
) -> None:
    """
    @brief One native chart graph differentiates the input and every knot and ordinate jointly.
    """
    rows = [(0.0, 1.0), (1.0, 3.0), (2.0, 2.0)]
    rates = [(0.2 * sign, 0.4 * sign), (-0.3 * sign, 0.7 * sign), (0.5 * sign, -0.8 * sign)]
    chart = PiecewiseLinear(rows)
    space = FunctionSpace(box_mesh, "R", 0)
    parameter = Function(space)
    symbolic_rows = [
        (x + dx * parameter, y + dy * parameter)
        for (x, y), (dx, dy) in zip(rows, rates, strict=True)
    ]
    expression = chart.expression(point + sign * parameter, symbolic_rows)
    response = directional(expression, parameter, as_ufl(1.0))
    assert float(replace(response, {parameter: as_ufl(0.0)})) == pytest.approx(
        float(
            chart.directional(
                [
                    np.asarray(point, dtype=np.float64),
                    *[
                        np.asarray(value)
                        for row in list(zip(chart.coordinates, chart.values, strict=True))
                        for value in row
                    ],
                ],
                [
                    np.asarray(sign, dtype=np.float64),
                    *[np.asarray(value) for row in rates for value in row],
                ],
            )[0]
        )
    )


@pytest.mark.parametrize(
    ("point", "seed", "candidate", "expected"),
    [
        (1.0, 1.0, 2.0, True),
        (1.0, 1.0, -1.0, True),
        (1.0, 1.0, 0.0, False),
        (1.0, -1.0, 0.0, True),
        (0.0, 1.0, 1.0, True),
        (2.0, 1.0, -0.5, True),
    ],
)
def test_joint_chart_coderivative_retains_knot_motion_and_nonconvexity(
    point: float, seed: float, candidate: float, expected: bool
) -> None:
    """
    @brief A knot's query and knot covectors oppose one another without convexifying the relation.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
    rows = np.zeros((3, 2))
    knot = int(point)
    rows[knot] = (-candidate, seed)
    assert bool(chart.joint_coderivative_contains(point, seed, candidate, rows)) == expected
    rows[knot, 0] += 0.25
    assert not chart.joint_coderivative_contains(point, seed, candidate, rows)


def test_joint_chart_coderivative_checks_all_smooth_and_held_row_covectors() -> None:
    """
    @brief Smooth intervals have the interpolation partials and held values observe one ordinate.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 1.0)])
    interior = np.array([[-1.0, 0.5], [-1.0, 0.5], [0.0, 0.0]])
    held = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    assert chart.joint_coderivative_contains(0.5, 1.0, 2.0, interior)
    assert chart.joint_coderivative_contains(3.0, 1.0, 0.0, held)
    assert not chart.joint_coderivative_contains(0.5, 1.0, 2.0, held)
    assert chart.joint_coderivative_contains(
        np.array([0.5, 3.0]), 1.0, np.array([2.0, 0.0]), np.stack((interior, held))
    ).all()


def test_scalar_chart_coderivative_exposes_the_whole_reverse_relation() -> None:
    """
    @brief A decreasing slope at a table knot retains disconnected positive-seed covectors.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)])
    response = scalar_coderivative(chart, 1.0, [2.0])
    assert response.contains(2.0)
    assert response.contains(4.0)
    assert not response.contains(3.0)
    assert not response.single_valued
    assert scalar_coderivative(chart, 0.5, [2.0]).value() == 4.0
    assert scalar_coderivative(chart, -1.0, [2.0]).value() == 0.0
    candidates = np.array([2.0, 3.0, 4.0])
    assert chart.coderivative_contains(1.0, 2.0, candidates) == pytest.approx(
        [response.contains(float(value)) for value in candidates]
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_basic_coderivative_membership_excludes_nonfinite_points(invalid: float) -> None:
    """
    @brief A held endpoint cannot turn non-real input data into a graph member.
    """
    chart = PiecewiseLinear([(0.0, 0.0), (1.0, 1.0)])
    assert not chart.coderivative_contains(invalid, 1.0, 0.0)
    assert not chart.joint_coderivative_contains(
        invalid, 1.0, 0.0, np.array([[0.0, 0.0], [0.0, 1.0]])
    )


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_chart_evaluation_refuses_nonfinite_query_coordinates(invalid: float) -> None:
    """
    @brief An invalid coordinate cannot silently select a held endpoint value.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 2.0)])
    with pytest.raises(ValueError, match="coordinates must be finite"):
        chart.evaluate([np.asarray(invalid)])
    with pytest.raises(ValueError, match="coordinates must be finite"):
        chart.directional([np.asarray(invalid)], [np.asarray(1.0)])


def test_chart_directional_refuses_nonfinite_operand_actions() -> None:
    """
    @brief Local chart actions require finite perturbations of every active operand.
    """
    chart = PiecewiseLinear([(0.0, 1.0), (1.0, 2.0)])
    with pytest.raises(ValueError, match="directions must be finite"):
        chart.directional([np.asarray(0.5)], [np.asarray(np.nan)])
