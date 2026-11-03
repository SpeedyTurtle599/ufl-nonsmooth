# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/scalar.py
@brief Scalar native tape application of a shared numerical local law.
@details One retained block records all scalar operands and outputs. Forward responses use
    the law's joint action, while reverse responses contract its selected partial derivatives.
    Replay evaluates the same law at saved operands.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pyadjoint import AdjFloat, Block, annotate_tape, get_working_tape, no_annotations

from ufl_nonsmooth.laws import ArrayLaw, LocalEvaluation, ScalarKinkLaw
from ufl_nonsmooth.primitives import ScalarGraphResponse, kink_coderivative, masked_pullback

if TYPE_CHECKING:
    from pyadjoint.block_variable import BlockVariable


class _ScalarLawBlock(Block):
    """
    @brief Retain every scalar operand and output of one numerical local law.
    """

    law: ArrayLaw

    def __init__(self, law: ArrayLaw, inputs: list[float]) -> None:
        """
        @brief Retain the law and every scalar operand in its original position.
        """
        super().__init__()
        self.law = law
        for value in inputs:
            self.add_dependency(value if isinstance(value, AdjFloat) else AdjFloat(value))

    @no_annotations
    def prepare_evaluate_tlm(
        self,
        inputs: list[float],
        tlm_inputs: list[float | None],
        relevant_outputs: list[tuple[int, BlockVariable]],
    ) -> list[NDArray[np.float64]]:
        """
        @brief Evaluate all output actions jointly with absent operand directions retained.
        """
        return self.law.directional(
            [np.asarray(value, dtype=np.float64) for value in inputs],
            [
                None if value is None else np.asarray(value, dtype=np.float64)
                for value in tlm_inputs
            ],
        )

    @no_annotations
    def evaluate_tlm_component(
        self,
        inputs: list[float],
        tlm_inputs: list[float | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> float:
        """
        @brief Read one scalar component of the prepared joint action.
        """
        if not isinstance(prepared, list):
            raise TypeError("a scalar local action requires its prepared output responses")
        return float(prepared[idx])

    @no_annotations
    def prepare_evaluate_adj(
        self,
        inputs: list[float],
        adj_inputs: list[float | None],
        relevant_dependencies: list[tuple[int, BlockVariable]],
    ) -> LocalEvaluation[NDArray[np.float64]]:
        """
        @brief Read all selected partials at the saved operands.
        """
        return self.law.evaluate([np.asarray(value, dtype=np.float64) for value in inputs])

    @no_annotations
    def evaluate_adj_component(
        self,
        inputs: list[float],
        adj_inputs: list[float | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> float | None:
        """
        @brief Contract every observed output seed with one selected operand partial.
        """
        if not isinstance(prepared, LocalEvaluation):
            raise TypeError("a scalar local pullback requires its prepared selected partials")
        if all(seed is None for seed in adj_inputs):
            return None
        pullback = sum(
            float(masked_pullback(seed, partials[idx]))
            for seed, partials in zip(adj_inputs, prepared.partials, strict=True)
            if seed is not None
        )
        if not np.isfinite(pullback):
            raise ValueError("the local law has no finite selected pullback for this operand")
        return pullback

    @no_annotations
    def prepare_recompute_component(
        self, inputs: list[float], relevant_outputs: list[tuple[int, BlockVariable]]
    ) -> LocalEvaluation[NDArray[np.float64]]:
        """
        @brief Evaluate all physical outputs once at the replayed operands.
        """
        return self.law.evaluate([np.asarray(value, dtype=np.float64) for value in inputs])

    @no_annotations
    def recompute_component(
        self,
        inputs: list[float],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> float:
        """
        @brief Read one replayed scalar output.
        """
        if not isinstance(prepared, LocalEvaluation):
            raise TypeError("a scalar local replay requires its prepared output values")
        return float(prepared.values[idx])


def evaluate_scalar(law: ArrayLaw, inputs: list[float]) -> list[float]:
    """
    @brief Apply a numerical local law to scalars on the native adjoint tape.
    @param law: Shared numerical value, selected partials, and joint directional action.
    @param inputs: Scalar operands, including native adjoint scalar controls.
    @return Scalar outputs, retained as adjoint scalars when annotation is enabled.
    """
    block = _ScalarLawBlock(law, inputs) if annotate_tape() else None
    evaluation = law.evaluate([np.asarray(value, dtype=np.float64) for value in inputs])
    if any(value.shape for value in evaluation.values):
        raise ValueError("scalar local application requires scalar outputs")
    values = [float(value) for value in evaluation.values]
    if block is None:
        return values
    results: list[float] = []
    get_working_tape().add_block(block)
    for value in values:
        result = AdjFloat(value)
        block.add_output(result.block_variable)
        results.append(result)
    return results


def scalar_coderivative(
    law: ScalarKinkLaw, point: float, seeds: list[float]
) -> ScalarGraphResponse:
    """
    @brief Exact local scalar-input basic coderivative of a qualified finite two-sided law.
    @details The law has one varying scalar input, fixed remaining parameters, and continuous
        smooth incident branches with finite limiting derivatives. Output covectors contract
        the two branch jets before the scalar basic-normal rule is applied. This is a local
        reverse relation and does not assert an arbitrary programme composition theorem.
    @param law: Local law declaring the finite two-sided scalar-input capability.
    @param point: Scalar graph coordinate.
    @param seeds: One covector per output of the law.
    @return Exact scalar reverse set, including disconnected limiting branch values.
    """
    if not isinstance(law, ScalarKinkLaw):
        raise TypeError("an exact scalar reverse query requires a declared finite two-sided law")
    if not np.isfinite(point) or not np.isfinite(seeds).all():
        raise ValueError("scalar reverse coordinates and covectors must be finite")
    coordinate = [np.asarray(point, dtype=np.float64)]
    left = law.directional(coordinate, [np.asarray(-1.0)])
    right = law.directional(coordinate, [np.asarray(1.0)])
    if len(seeds) != len(left) or len(left) != len(right):
        raise ValueError("a local reverse query needs one covector per output")
    negative = sum(-seed * float(value) for seed, value in zip(seeds, left, strict=True))
    positive = sum(seed * float(value) for seed, value in zip(seeds, right, strict=True))
    return kink_coderivative(negative, positive, 1.0)
