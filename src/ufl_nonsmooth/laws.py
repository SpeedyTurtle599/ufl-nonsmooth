# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/laws.py
@brief Local values, selected linearisations, and joint directional actions.
@details A local law declares every varying operand together. Its selected partials form a
    linear map, while its directional action follows the supplied perturbation through joins.
    An absent perturbation freezes an operand. An explicit zero keeps it active. Numerical and
    symbolic laws share this contract and use their native representations for evaluation.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, overload, runtime_checkable

import numpy as np
from numpy.typing import NDArray
from ufl import And, Or, as_ufl, conditional, eq
from ufl.core.expr import Expr

from ufl_nonsmooth.primitives import ArrayValue

Value = TypeVar("Value")


@dataclass(frozen=True)
class LocalEvaluation(Generic[Value]):
    """
    @brief Values and selected partials of a local law at one set of operands.
    @param values: One value per scalar output, optionally batched over points.
    @param partials: Output rows and operand columns of the selected linearisation.
    """

    values: list[Value]
    partials: list[list[Value]]

    def __post_init__(self) -> None:
        """
        @brief Require a rectangular linearisation with one row per output.
        """
        if not self.values or len(self.values) != len(self.partials):
            raise ValueError("a local evaluation needs one partial row per output")
        count = len(self.partials[0])
        if any(len(row) != count for row in self.partials):
            raise ValueError("local partial rows must have the same operand count")


@runtime_checkable
class LocalLaw(Protocol[Value]):
    """
    @brief A local map with a selected linearisation and an exact joint finite action.
    @details Selected partials are ordinary derivatives on a regular branch. Their transpose
        at a switch is a named branch choice, not an exact basic coderivative. Exact local
        reverse relations require an explicit additional law capability. Singular or unsupported
        actions raise a response error rather than manufacture a finite branch gradient.
    """

    @abstractmethod
    def evaluate(self, inputs: list[Value]) -> LocalEvaluation[Value]:
        """
        @brief Evaluate every output and selected partial from the same input values.
        """
        ...

    @abstractmethod
    def directional(self, inputs: list[Value], directions: list[Value | None]) -> list[Value]:
        """
        @brief Apply all active perturbations together, preserving frozen operands.
        """
        ...


class ArrayLaw(LocalLaw[NDArray[np.float64]]):
    """
    @brief A numerical local law evaluated on real arrays, including scalar arrays.
    """


class ScalarKinkLaw(ArrayLaw):
    """
    @brief A continuous scalar-input restriction with at most two finite smooth incident jets.
    @details With all other law parameters fixed, every output is continuous in the scalar
        coordinate and continuously differentiable on each incident side. Finite limiting
        slopes give the exact local basic coderivative through the two-sided scalar rule.
        This capability does not cover joint parameter covectors or composition of reverse sets.
    """


class SymbolicLaw(LocalLaw[Expr]):
    """
    @brief A native symbolic local law whose selected action remains differentiable.
    """


@overload
def join(
    margin: Expr,
    negative: Expr | float,
    positive: Expr | float,
    direction: Expr | float | None = None,
) -> Expr: ...


@overload
def join(
    margin: ArrayValue,
    negative: ArrayValue,
    positive: ArrayValue,
    direction: ArrayValue | None = None,
) -> NDArray[np.float64]: ...


def join(
    margin: Expr | ArrayValue,
    negative: Expr | ArrayValue,
    positive: Expr | ArrayValue,
    direction: Expr | ArrayValue | None = None,
) -> Expr | NDArray[np.float64]:
    """
    @brief Select a local branch for values, incident actions, or interval masks.
    @details The positive branch is selected at equality without a direction. A supplied margin
        action selects the negative side at equality only when it is negative. The action must
        include every moving operand of the margin. A law assembling a continuous join must
        supply matching branch values and finite action formulas that agree on tangent directions.
        Singular and discontinuous joins require their own local response contract.
    @param margin: Signed switching function, positive on the positive branch.
    @param negative: Value or action on the negative branch.
    @param positive: Value or action on the positive branch.
    @param direction: Joint action of the switching function, absent for selected evaluation.
    @return The selected branch in the native representation.
    @pre The assembled law establishes continuity and directional validity at its joins.
    """
    if isinstance(margin, Expr):
        if (
            isinstance(negative, np.ndarray)
            or isinstance(positive, np.ndarray)
            or isinstance(direction, np.ndarray)
        ):
            raise TypeError("symbolic joins require symbolic or scalar branch operands")
        selected = (
            margin >= 0.0
            if direction is None
            else Or(margin > 0.0, And(eq(margin, 0.0), as_ufl(direction) >= 0.0))
        )
        return conditional(selected, positive, negative)
    if isinstance(negative, Expr) or isinstance(positive, Expr) or isinstance(direction, Expr):
        raise TypeError("numerical joins require numerical branch operands")
    point = np.asarray(margin, dtype=np.float64)
    selected = (
        point >= 0.0
        if direction is None
        else (point > 0.0) | ((point == 0.0) & (np.asarray(direction) >= 0.0))
    )
    return np.asarray(np.where(selected, positive, negative), dtype=np.float64)


def ordered_branch(
    breakpoints: NDArray[np.float64],
    points: ArrayValue,
    direction: ArrayValue | None = None,
) -> NDArray[np.int64]:
    """
    @brief Select smooth intervals separated by ordered finite breakpoints.
    @param breakpoints: Strictly increasing internal switching coordinates.
    @param points: Input coordinates, including exact knot values.
    @param direction: Input actions, absent for right-selected evaluation.
    @return Interval indices from zero to the number of breakpoints.
    """
    knots = np.asarray(breakpoints, dtype=np.float64)
    coordinates = np.asarray(points, dtype=np.float64)
    if knots.ndim != 1 or not np.isfinite(knots).all() or np.any(np.diff(knots) <= 0.0):
        raise ValueError("local breakpoints must be finite and strictly increasing")
    if not np.isfinite(coordinates).all():
        raise ValueError("local branch coordinates must be finite")
    right = np.searchsorted(knots, coordinates, side="right")
    if direction is None:
        return np.asarray(right, dtype=np.int64)
    rates = np.asarray(direction, dtype=np.float64)
    if not np.isfinite(rates).all():
        raise ValueError("local branch directions must be finite")
    left = np.searchsorted(knots, coordinates, side="left")
    return np.asarray(np.where(rates < 0.0, left, right), dtype=np.int64)
