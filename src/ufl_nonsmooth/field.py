# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/field.py
@brief Recorded numerical local laws on native scalar fields.
@details Joint field and real-space operands share one value, action, selected pullback, and
    replay implementation. Real-space covectors sum their contributions over the mesh communicator.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from firedrake import Cofunction, Function
from mpi4py import MPI
from numpy.typing import NDArray
from pyadjoint import Block, annotate_tape, get_working_tape, no_annotations

from ufl_nonsmooth.laws import ArrayLaw, LocalEvaluation
from ufl_nonsmooth.primitives import masked_pullback

if TYPE_CHECKING:
    from pyadjoint.block_variable import BlockVariable


class _FieldLawBlock(Block):
    """
    @brief A numerical local law with native checkpoint and derivative storage.
    """

    def __init__(self, law: ArrayLaw, inputs: list[Function], outputs: list[Function]) -> None:
        """
        @brief Retain the law and all operand dependencies in their declared order.
        """
        super().__init__()
        self.law = law
        self.spaces = [output.function_space() for output in outputs]
        for value in inputs:
            self.add_dependency(value, no_duplicates=False)

    @no_annotations
    def prepare_evaluate_adj(
        self,
        inputs: list[Function],
        adj_inputs: list[Cofunction | None],
        relevant_dependencies: list[tuple[int, BlockVariable]],
    ) -> LocalEvaluation[NDArray[np.float64]]:
        """
        @brief Evaluate all selected partials once at the saved primal operands.
        """
        values = field_values(inputs)
        return self.law.evaluate(values)

    @no_annotations
    def evaluate_adj_component(
        self,
        inputs: list[Function],
        adj_inputs: list[Cofunction | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Cofunction | None:
        """
        @brief Pull back every observed output and reduce a shared scalar operand globally.
        """
        if not any(seed is not None for seed in adj_inputs):
            return None
        if not isinstance(prepared, LocalEvaluation):
            raise TypeError("a numerical law pullback requires its saved local evaluation")
        result = Cofunction(inputs[idx].function_space().dual())
        comm = inputs[idx].function_space().mesh().comm
        observed = next(seed for seed in adj_inputs if seed is not None)
        pullback = np.zeros_like(observed.dat.data_ro)
        for seed, partials in zip(adj_inputs, prepared.partials, strict=True):
            if seed is not None:
                pullback += masked_pullback(seed.dat.data_ro, partials[idx])
        if not comm.allreduce(bool(np.isfinite(pullback).all()), op=MPI.LAND):
            raise ValueError("the local law has no finite selected pullback for this operand")
        if (
            inputs[idx].ufl_element().family() == "Real"
            and self.spaces[0].ufl_element().family() != "Real"
        ):
            result.dat.data[:] = comm.allreduce(float(pullback.sum()), op=MPI.SUM)
        else:
            result.dat.data[:] = pullback
        return result

    @no_annotations
    def prepare_evaluate_tlm(
        self,
        inputs: list[Function],
        tlm_inputs: list[Function | None],
        relevant_outputs: list[tuple[int, BlockVariable]],
    ) -> list[NDArray[np.float64]]:
        """
        @brief Apply one joint direction with explicit frozen operands at the saved values.
        """
        values = field_values(inputs)
        directions = [
            None if direction is None else field_values([direction])[0] for direction in tlm_inputs
        ]
        return self.law.directional(values, directions)

    @no_annotations
    def evaluate_tlm_component(
        self,
        inputs: list[Function],
        tlm_inputs: list[Function | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Function:
        """
        @brief Store the requested local action in its native output space.
        """
        if not isinstance(prepared, list):
            raise TypeError("a numerical law action requires its joint prepared response")
        result = Function(self.spaces[idx])
        result.dat.data[:] = prepared[idx]
        return result

    @no_annotations
    def prepare_recompute_component(
        self, inputs: list[Function], relevant_outputs: list[tuple[int, BlockVariable]]
    ) -> list[NDArray[np.float64]]:
        """
        @brief Evaluate every output once from replayed operand values.
        """
        evaluation = self.law.evaluate(field_values(inputs))
        return evaluation.values

    @no_annotations
    def recompute_component(
        self,
        inputs: list[Function],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Function:
        """
        @brief Place a replayed value into its declared native field space.
        """
        if not isinstance(prepared, list):
            raise TypeError("a numerical law replay requires its prepared values")
        result = Function(self.spaces[idx])
        result.dat.data[:] = prepared[idx]
        return result


def field_values(inputs: list[Function]) -> list[NDArray[np.float64]]:
    """
    @brief Read scalar field samples, representing shared real-space operands as zero-dimensional arrays.
    """
    return [
        np.asarray(float(value.dat.data_ro[0]))
        if value.ufl_element().family() == "Real"
        else value.dat.data_ro
        for value in inputs
    ]


def apply_fields(law: ArrayLaw, inputs: list[Function], outputs: list[Function]) -> None:
    """
    @brief Apply a numerical law into preallocated scalar fields and retain its native tape operation.
    """
    if not inputs or not outputs:
        raise ValueError("a local law needs operands and preallocated outputs")
    space = outputs[0].function_space()
    if not space.ufl_element().pullback.is_identity:
        raise ValueError("numerical local laws require an identity output pullback")
    if any(value.function_space().mesh() is not space.mesh() for value in inputs):
        raise ValueError("a local field law requires every operand on the output mesh")
    if any(value.ufl_shape for value in [*inputs, *outputs]):
        raise ValueError("numerical field laws require scalar operands and outputs")
    if any(output.function_space() != space for output in outputs):
        raise ValueError("numerical law outputs must share one native scalar space")
    if any(
        value.ufl_element().family() != "Real" and value.function_space() != space
        for value in inputs
    ):
        raise ValueError("numerical law operands must use the output space or its real space")
    block = _FieldLawBlock(law, inputs, outputs) if annotate_tape() else None
    evaluation = law.evaluate(field_values(inputs))
    if len(evaluation.values) != len(outputs):
        raise ValueError("a local law needs one preallocated field for every output")
    if any(len(row) != len(inputs) for row in evaluation.partials):
        raise ValueError("a local law needs one selected partial for every operand")
    values = [
        np.broadcast_to(value, output.dat.data_ro.shape)
        for output, value in zip(outputs, evaluation.values, strict=True)
    ]
    for output, value in zip(outputs, values, strict=True):
        output.dat.data[:] = value
    if block is not None:
        get_working_tape().add_block(block)
        for output in outputs:
            block.add_output(output.create_block_variable())
