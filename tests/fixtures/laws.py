# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/fixtures/laws.py
@brief Analytical local laws for scalar, field, and symbolic consumers.
@details Sum, product, and square laws have independent closed-form first derivatives.
"""

import numpy as np
from numpy.typing import NDArray
from ufl import as_ufl
from ufl.core.expr import Expr

from ufl_nonsmooth.laws import ArrayLaw, LocalEvaluation, SymbolicLaw


class CoupledLaw(ArrayLaw):
    """
    @brief Sum and product of a field operand and a shared scalar.
    """

    def evaluate(self, inputs: list[NDArray[np.float64]]) -> LocalEvaluation[NDArray[np.float64]]:
        """
        @brief Evaluate two outputs and their exact partial derivatives.
        """
        first, second = inputs
        return LocalEvaluation(
            [first + second, first * second],
            [[np.ones_like(first), np.ones_like(first)], [second, first]],
        )

    def directional(
        self, inputs: list[NDArray[np.float64]], directions: list[NDArray[np.float64] | None]
    ) -> list[NDArray[np.float64]]:
        """
        @brief Apply the analytical sum and product rules jointly.
        """
        first, second = inputs
        left = np.zeros_like(first) if directions[0] is None else directions[0]
        right = np.zeros_like(second) if directions[1] is None else directions[1]
        return [left + right, second * left + first * right]


class SquareLaw(SymbolicLaw):
    """
    @brief Square of one native scalar operand.
    """

    def evaluate(self, inputs: list[Expr]) -> LocalEvaluation[Expr]:
        """
        @brief Evaluate the square and its analytical derivative.
        """
        point = inputs[0]
        return LocalEvaluation([point**2], [[2.0 * point]])

    def directional(self, inputs: list[Expr], directions: list[Expr | None]) -> list[Expr]:
        """
        @brief Apply the smooth square derivative to the active direction.
        """
        rate = as_ufl(0.0) if directions[0] is None else directions[0]
        return [2.0 * inputs[0] * rate]
