# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/piecewise.py
@brief Continuous piecewise-linear charts with held endpoint values.
@details Immutable default coordinates, values, and segment slopes supply numerical evaluation
    and native Unified Form Language expressions. Optional ordered knots and ordinates use the
    same chart for joint parameter actions and the exact local basic coderivative. Native
    expression differentiation selects the right slope at a knot. Directional actions and
    reverse relations retain both sides.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from ufl import as_ufl, max_value, min_value

from ufl_nonsmooth.laws import LocalEvaluation, ScalarKinkLaw, join

if TYPE_CHECKING:
    from ufl.core.expr import Expr


class PiecewiseLinear(ScalarKinkLaw):
    """
    @brief A continuous scalar chart, constant beyond its first and last coordinates.
    @param coordinates: Strictly increasing chart coordinates.
    @param values: Chart values at the coordinates.
    @param segment_slopes: Slopes between adjacent coordinates.
    """

    coordinates: NDArray[np.float64]
    values: NDArray[np.float64]
    segment_slopes: NDArray[np.float64]

    def __init__(self, rows: list[tuple[float, float]]) -> None:
        """
        @brief Copy and validate a chart with at least two finite, ordered rows.
        @param rows: Coordinate and value pairs.
        @throws ValueError: Rows are non-finite, contain fewer than two pairs, or do not have
            strictly increasing coordinates.
        """
        data = np.asarray(rows, dtype=np.float64)
        if data.shape != (len(rows), 2) or len(rows) < 2:
            raise ValueError("a piecewise-linear chart needs two or more coordinate-value pairs")
        if not np.all(np.isfinite(data)):
            raise ValueError("piecewise-linear chart rows must be finite")
        if np.any(np.diff(data[:, 0]) <= 0.0):
            raise ValueError("piecewise-linear chart coordinates must be strictly increasing")
        self.coordinates = data[:, 0].copy()
        self.values = data[:, 1].copy()
        self.segment_slopes = np.diff(self.values) / np.diff(self.coordinates)
        for array in (self.coordinates, self.values, self.segment_slopes):
            array.setflags(write=False)

    def _weights(
        self,
        inputs: list[NDArray[np.float64]],
        directions: list[NDArray[np.float64] | None] | None = None,
    ) -> tuple[NDArray[np.float64], list[NDArray[np.float64]]]:
        """
        @brief Evaluate a selected segment and every coordinate, knot, and ordinate partial.
        @details The input coordinate comes first. Optional alternating knot and ordinate
            operands replace every default row. Relative query and knot motion selects the
            adjacent segment at a tie. All queries broadcast independently.
        """
        count = len(self.coordinates)
        if len(inputs) not in (1, 1 + 2 * count):
            raise ValueError("a chart needs its coordinate and optionally every knot and ordinate")
        if directions is not None and len(directions) != len(inputs):
            raise ValueError("each chart operand needs one direction")
        parameters = (
            inputs
            if len(inputs) > 1
            else [
                inputs[0],
                *[
                    np.asarray(value)
                    for row in zip(self.coordinates, self.values, strict=True)
                    for value in row
                ],
            ]
        )
        count_parameters = len(parameters)
        if not np.isfinite(inputs[0]).all():
            raise ValueError("chart query coordinates must be finite")
        varying = [] if directions is None else [rate for rate in directions if rate is not None]
        if any(not np.isfinite(rate).all() for rate in varying):
            raise ValueError("chart operand directions must be finite")
        parameters = list(np.broadcast_arrays(*parameters, *varying))[:count_parameters]
        point = parameters[0]
        knots, ordinates = parameters[1::2], parameters[2::2]
        if any(not np.isfinite(value).all() for value in parameters[1:]) or any(
            np.any(right <= left) for left, right in itertools.pairwise(knots)
        ):
            raise ValueError("runtime chart rows must be finite and strictly ordered")
        rates: list[NDArray[np.float64] | None] = [None] * len(parameters)
        if directions is not None:
            rates[: len(directions)] = directions
        index = np.zeros(point.shape, dtype=np.int64)
        for row, knot in enumerate(knots):
            active = rates[0] is not None or rates[1 + 2 * row] is not None
            relative = (
                np.asarray(0.0 if rates[0] is None else rates[0])
                - np.asarray(0.0 if rates[1 + 2 * row] is None else rates[1 + 2 * row])
                if active
                else None
            )
            selected = join(point - knot, np.zeros_like(point), np.ones_like(point), relative)
            index = index + np.asarray(selected, dtype=np.int64)
        segment = np.clip(index - 1, 0, count - 2)
        stacked_knots = np.stack(knots, axis=-1)
        stacked_values = np.stack(ordinates, axis=-1)
        left = np.take_along_axis(stacked_knots, segment[..., None], axis=-1)[..., 0]
        right = np.take_along_axis(stacked_knots, (segment + 1)[..., None], axis=-1)[..., 0]
        lower = np.take_along_axis(stacked_values, segment[..., None], axis=-1)[..., 0]
        upper = np.take_along_axis(stacked_values, (segment + 1)[..., None], axis=-1)[..., 0]
        fraction = np.clip((point - left) / (right - left), 0.0, 1.0)
        slope = np.where((index == 0) | (index == count), 0.0, (upper - lower) / (right - left))
        weights = [
            np.asarray(
                np.where(
                    index == 0,
                    float(row == 0),
                    np.where(
                        index == count,
                        float(row == count - 1),
                        np.where(
                            segment == row,
                            1.0 - fraction,
                            np.where(segment + 1 == row, fraction, 0.0),
                        ),
                    ),
                ),
                dtype=np.float64,
            )
            for row in range(count)
        ]
        value = sum(
            (weight * ordinate for weight, ordinate in zip(weights, ordinates, strict=True)),
            start=np.zeros_like(point),
        )
        partials = [np.asarray(slope, dtype=np.float64)]
        for weight in weights:
            partials.extend([np.asarray(-slope * weight), weight])
        return np.asarray(value, dtype=np.float64), partials

    def evaluate(self, inputs: list[NDArray[np.float64]]) -> LocalEvaluation[NDArray[np.float64]]:
        """
        @brief Evaluate chart values and right-selected partials of every supplied operand.
        @param inputs: Coordinate, optionally followed by alternating runtime knots and ordinates.
        @return One broadcast output with its selected operand partials.
        """
        value, partials = self._weights(inputs)
        return LocalEvaluation([value], [partials[: len(inputs)]])

    def directional(
        self,
        inputs: list[NDArray[np.float64]],
        directions: list[NDArray[np.float64] | None],
    ) -> list[NDArray[np.float64]]:
        """
        @brief Apply the joint action of the query and every varying knot and ordinate.
        @param inputs: Coordinate, optionally followed by alternating runtime knots and ordinates.
        @param directions: Matching operand perturbations, with absent directions held fixed.
        @return One broadcast response following relative motion at each knot.
        """
        _, partials = self._weights(inputs, directions)
        response = sum(
            (
                partial * direction
                for partial, direction in zip(partials, directions)
                if direction is not None
            ),
            start=np.zeros_like(partials[0]),
        )
        return [np.asarray(response, dtype=np.float64)]

    def slopes(
        self, points: float | NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        @brief Read the slopes on both sides, including the held endpoint branches.
        """
        coordinate = np.asarray(points, dtype=np.float64)
        _, left = self._weights([coordinate], [-np.ones_like(coordinate)])
        _, right = self._weights([coordinate], [np.ones_like(coordinate)])
        return left[0], right[0]

    def coderivative_contains(
        self,
        points: float | NDArray[np.float64],
        seeds: float | NDArray[np.float64],
        candidates: float | NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        r"""
        @brief Test exact membership in the chart's scalar basic coderivative.
        @details With left and right slopes \f$ p \f$ and \f$ q \f$, the two limiting branch
            covectors are \f$ \ell p \f$ and \f$ \ell q \f$. The interval between those
            values is included only when \f$ \ell p \leq \ell q \f$.
        @param points: Base coordinates of the reverse relation.
        @param seeds: Output covectors.
        @param candidates: Proposed input covectors, broadcast against the coordinates and seeds.
        @return Membership in the union of the limiting branch values and the ordered interval.
        """
        points, seeds, candidates = np.broadcast_arrays(points, seeds, candidates)
        finite = np.isfinite(points) & np.isfinite(seeds) & np.isfinite(candidates)
        left, right = self.slopes(np.where(finite, points, self.coordinates[0]))
        seeds = np.where(finite, seeds, 0.0)
        candidates = np.where(finite, candidates, 0.0)
        lower = seeds * left
        upper = seeds * right
        membership = (
            (candidates == lower)
            | (candidates == upper)
            | ((lower <= candidates) & (candidates <= upper))
        )
        return np.asarray(membership & finite, dtype=np.bool_)

    def joint_coderivative_contains(
        self,
        points: float | NDArray[np.float64],
        seeds: float | NDArray[np.float64],
        input_candidates: float | NDArray[np.float64],
        row_candidates: NDArray[np.float64],
        rows: list[tuple[float, float]] | None = None,
    ) -> NDArray[np.bool_]:
        r"""
        @brief Test the exact basic coderivative jointly in the input, knots, and ordinates.
        @details At a knot, its covector is the negative input covector, its ordinate covector
            is the output seed, and all other row covectors vanish. The input covector follows
            the unsorted two-sided scalar relation. Strictly positive knot gaps define the
            parameter domain. Array arguments batch independent scalar queries and do not
            combine output seeds into a vector-output relation with shared table parameters.
        @param points: Base input coordinates.
        @param seeds: Output covectors.
        @param input_candidates: Proposed input-coordinate covectors.
        @param row_candidates: Proposed row covectors, with final axes for rows and knot/ordinate.
        @param rows: Optional finite, strictly ordered runtime knot and ordinate pairs.
        @return Exact joint membership with the broadcast shape of coordinates and seeds.
        """
        chart = self if rows is None else PiecewiseLinear(rows)
        if len(chart.coordinates) != len(self.coordinates):
            raise ValueError("runtime chart rows must preserve the chart topology")
        points, seeds, candidates = np.broadcast_arrays(points, seeds, input_candidates)
        expected = np.zeros(points.shape + (len(chart.coordinates), 2), dtype=np.float64)
        membership = np.ones(points.shape, dtype=np.bool_)
        for index in np.ndindex(points.shape):
            point = float(points[index])
            seed = float(seeds[index])
            candidate = float(candidates[index])
            if not np.isfinite((point, seed, candidate)).all():
                membership[index] = False
                continue
            knots = np.flatnonzero(chart.coordinates == point)
            if knots.size:
                knot = int(knots[0])
                expected[index + (knot, 0)] = -candidate
                expected[index + (knot, 1)] = seed
                membership[index] = chart.coderivative_contains(point, seed, candidate)
            else:
                _, partials = chart._weights([np.asarray(point)])
                expected[index + (slice(None), 0)] = seed * np.asarray(partials[1::2])
                expected[index + (slice(None), 1)] = seed * np.asarray(partials[2::2])
                membership[index] = candidate == seed * partials[0]
        return np.asarray(
            membership
            & np.all(np.isfinite(row_candidates), axis=(-2, -1))
            & np.all(np.asarray(row_candidates) == expected, axis=(-2, -1)),
            dtype=np.bool_,
        )

    def expression(
        self, coordinate: Expr, rows: list[tuple[Expr | float, Expr | float]] | None = None
    ) -> Expr:
        """
        @brief Build the native expression with the right slope selected at each knot.
        @details Bounded rising ramps give nonnegative nodal weights, with only adjacent nodes
            contributing inside an interval. The coordinate is clamped before the ramps, so held
            endpoint values are independent of the distance travelled outside the chart. The
            selected slope enters the chart at its lower endpoint and leaves it at its upper one.
        @param coordinate: Scalar Unified Form Language coordinate of the chart.
        @param rows: Optional symbolic knot and ordinate pairs, with strictly positive knot gaps.
        @pre Runtime knots remain strictly increasing and all parameters remain finite.
        @return Continuous piecewise-linear expression with held endpoint values.
        """
        parameters = (
            [
                (float(knot), float(value))
                for knot, value in zip(self.coordinates, self.values, strict=True)
            ]
            if rows is None
            else rows
        )
        if len(parameters) != len(self.coordinates):
            raise ValueError("runtime chart rows must preserve the chart topology")
        knots = [as_ufl(row[0]) for row in parameters]
        values = [as_ufl(row[1]) for row in parameters]
        if coordinate.ufl_shape or any(item.ufl_shape for item in knots + values):
            raise ValueError("chart coordinates and ordinates must be scalar expressions")
        bounded = min_value(max_value(knots[0], coordinate), knots[-1])
        ramps = [
            min_value(max_value(0.0, (bounded - lower) / (upper - lower)), 1.0)
            for lower, upper in itertools.pairwise(knots)
        ]
        expression = values[0] * (1.0 - ramps[0])
        for index in range(1, len(values) - 1):
            expression += values[index] * ramps[index - 1] * (1.0 - ramps[index])
        expression += values[-1] * ramps[-1]
        return expression
