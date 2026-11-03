# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/tape.py
@brief Joint directional responses on the native computation tape.
@details Assembly, interpolation, expression assignment, and implicit solves evaluate each
    operation with all of its incoming directions together. Other operations use their native
    tangent hooks. Local laws use the same traversal through their shared native adapters.
    Responses occupy the tape's tangent values without changing primal values,
    checkpoints, adjoints, or the recorded sequence of operations.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from firedrake import (
    Cofunction,
    Constant,
    Function,
    MeshGeometry,
    SpatialCoordinate,
    TrialFunction,
    assemble,
    derivative,
)
from firedrake.adjoint_utils.blocks.assembly import AssembleBlock
from firedrake.adjoint_utils.blocks.function import FunctionAssignBlock
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock
from firedrake.matrix import MatrixBase
from pyadjoint import (
    Block,
    annotate_tape,
    get_working_tape,
    no_annotations,
    stop_annotating,
)
from ufl import BaseForm, ZeroBaseForm, replace
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_base_form_operators, extract_coefficients
from ufl.classes import (
    ExternalOperator,
    Interpolate,
)
from ufl.core.expr import Expr

from ufl_nonsmooth.implicit import evaluate_solve
from ufl_nonsmooth.primitives import (
    directional,
    selected_directional,
)
from ufl_nonsmooth.sampling import operation_directions, sampled_stratum

if TYPE_CHECKING:
    from pyadjoint import Control, Tape
    from pyadjoint.block_variable import BlockVariable


Direction = Function | Cofunction | Constant | float


class _OperatorBlock(AssembleBlock):
    """
    @brief Native opaque-operator assembly with an explicit selected coordinate transpose.
    @details Coefficient adjoints retain native assembly hooks. The mesh covector transposes
        the exact native coordinate jet with the native selected primitive branches, including
        the right operand of a maximum at equality. This is a selected linearisation and does
        not identify the complete reverse relation at a non-smooth point.
    """

    @no_annotations
    def evaluate_adj_component(
        self,
        inputs: list[object],
        adj_inputs: list[Cofunction | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Cofunction | None:
        """
        @brief Transpose the exact selected coordinate jet or use the native coefficient pullback.
        """
        if not isinstance(block_variable.output, MeshGeometry):
            return super().evaluate_adj_component(inputs, adj_inputs, block_variable, idx, prepared)
        seed = adj_inputs[0]
        if seed is None:
            return None
        if not isinstance(prepared, (Expr, BaseForm)):
            raise NotImplementedError("an operator geometry pullback requires its saved expression")
        mesh = block_variable.saved_output
        direction = Function(mesh.coordinates.function_space())
        response = selected_directional(prepared, [SpatialCoordinate(mesh)], [direction])
        for operator in extract_base_form_operators(response):
            if isinstance(operator, ExternalOperator) and any(
                direction in extract_coefficients(slot)
                for slot in (*operator.ufl_operands, *operator.argument_slots())
            ):
                raise NotImplementedError(
                    "an opaque coordinate jet must expose its symbolic direction before transpose"
                )
        response = _materialise_operators(response)
        jacobian = expand_derivatives(
            derivative(response, direction, TrialFunction(direction.function_space()))
        )
        if direction in extract_coefficients(jacobian):
            raise NotImplementedError(
                "a nonlinear coordinate jet requires a named reverse selection"
            )
        result = Cofunction(direction.function_space().dual())
        matrix = assemble(jacobian)
        if not isinstance(matrix, MatrixBase):
            if isinstance(matrix, ZeroBaseForm) or (
                isinstance(matrix, (int, float)) and matrix == 0.0
            ):
                return result
            raise NotImplementedError("an operator coordinate jet must assemble to a native matrix")
        with seed.dat.vec_ro as incoming, result.dat.vec as outgoing:
            matrix.petscmat.multHermitian(incoming, outgoing)
        return result


def assemble_operator(expression: Expr) -> Function:
    """
    @brief Assemble a native opaque scalar operator with complete coordinate tape dependencies.
    @details The pointwise operator must provide an exact shared intrinsic jet on its active
        domain. Its specialised native block retains coefficient derivatives and adds the
        selected coordinate transpose. The coordinate jet must expose its symbolic direction
        and become linear after the native primitive branches are selected.
    """
    block = _OperatorBlock(expression) if annotate_tape() else None
    with stop_annotating():
        result = assemble(expression)
    if not isinstance(result, Function):
        raise NotImplementedError("an opaque pointwise operator must assemble to a native field")
    if block is not None:
        get_working_tape().add_block(block)
        block.add_output(result.create_block_variable())
    return result


def _causal_blocks(
    tape: Tape, cut: set[BlockVariable], outputs: list[BlockVariable]
) -> list[Block]:
    """
    @brief Retain the native causal slice from independent cut nodes to requested outputs.
    """
    needed = set(outputs)
    blocks = []
    for block in reversed(list(tape.get_blocks())):
        if any(node in needed and node not in cut for node in block.get_outputs()):
            blocks.append(block)
            needed.update(block.get_dependencies())
    blocks.reverse()
    return blocks


def evaluate_directional(
    tape: Tape,
    controls: list[Control],
    directions: list[Direction],
    outputs: list[BlockVariable] | None = None,
) -> None:
    """
    @brief Propagate one joint direction through the recorded forward computation.
    @param tape: Native tape with available primal checkpoints.
    @param controls: Distinct tape nodes at which to prescribe the perturbation.
    @param directions: Perturbations in the corresponding control spaces, including explicit zeros.
    @param outputs: Requested observable nodes, absent to traverse every native operation.
    @details Each traversed output's response is available as its block variable's ``tlm_value``.
        An absent value on the requested causal slice means independence of the controls.
        Unvisited nodes have no computed response. Existing tangent values are reset.
        Control nodes inside the tape cut dependence on their own producing operations.
        Implicit state equations require a regular branch with a linear response equation.
        Active state kinks require an additional implicit uniqueness contract.
        Boundary conditions and nullspaces retain the forward solve's native treatment.
    @throws ValueError: Controls are empty, repeated, or differ in number from the directions.
    @throws NotImplementedError: An operation has no native response hook or the expression has
        no supported finite directional action.
    """
    if not controls or len(controls) != len(directions):
        raise ValueError("provide one direction for every control and at least one control")
    if len({id(control.block_variable) for control in controls}) != len(controls):
        raise ValueError("directional controls must refer to distinct tape nodes")
    seeds = {
        control.block_variable: direction
        for control, direction in zip(controls, directions, strict=True)
    }
    blocks = (
        list(tape.get_blocks()) if outputs is None else _causal_blocks(tape, set(seeds), outputs)
    )
    _propagate_directional(tape, seeds, blocks)


def _propagate_directional(
    tape: Tape,
    directions: dict[BlockVariable, Direction],
    blocks: list[Block],
) -> None:
    """
    @brief Traverse native operations while keeping all independent cut-node directions fixed.
    """
    with stop_annotating():
        tape.reset_variables(types=("tlm",))
        seeds = {
            node: _normalise_direction(node.output, direction)
            for node, direction in directions.items()
        }
        for node, direction in seeds.items():
            node.tlm_value = direction
        for block in blocks:
            if not any(dependency.tlm_value is not None for dependency in block.get_dependencies()):
                continue
            outputs = [output for output in block.get_outputs() if output not in seeds]
            if not outputs:
                continue
            if isinstance(block, GenericSolveBlock):
                evaluate_solve(block, outputs)
            elif isinstance(block, AssembleBlock):
                _evaluate_assembly(block, outputs)
            elif isinstance(block, FunctionAssignBlock) and block.expr is not None:
                _evaluate_assignment(block, outputs)
            else:
                block.evaluate_tlm()
            for output in outputs:
                output.tlm_value = _normalise_direction(output.output, output.tlm_value)
            for node, direction in seeds.items():
                node.tlm_value = direction


def _normalise_direction(primal: object, value: object) -> object:
    """
    @brief Retain native primal field types across scalar and dual-valued assignment hooks.
    @details A scalar assigned to a field broadcasts in the field's space. A dual-valued
        assignment supplies its coefficient array through the native Euclidean Riesz map.
        Other native tangent values retain their own types.
    """
    if isinstance(primal, Function):
        if isinstance(value, (float, int, np.floating)):
            return Function(primal.function_space()).assign(float(value))
        if isinstance(value, Cofunction):
            return value.riesz_representation("l2")
    return value


def _materialise_operators(expression: Expr | BaseForm) -> Expr | BaseForm:
    """
    @brief Evaluate native external operators before pointwise expression compilation.
    @details Operators are assembled from their operands in dependency order. Their native
        derivative actions remain on the causal tape when annotation is enabled, and scalar
        expression consumers receive ordinary coefficient fields.
    """
    operators = [
        operator
        for operator in extract_base_form_operators(expression)
        if isinstance(operator, ExternalOperator)
    ]
    while operators:
        operator = next(
            candidate
            for candidate in operators
            if sum(
                isinstance(dependency, ExternalOperator)
                for dependency in extract_base_form_operators(candidate)
            )
            == 1
        )
        value = assemble_operator(operator)
        if not isinstance(value, (Expr, BaseForm)):
            raise NotImplementedError("an external expression operator must assemble to a field")
        expression = replace(expression, {operator: value})
        operators = [
            remaining
            for remaining in extract_base_form_operators(expression)
            if isinstance(remaining, ExternalOperator)
        ]
    return expression


def _evaluate_assembly(block: AssembleBlock, outputs: list[BlockVariable]) -> None:
    """
    @brief Assemble the joint response of an integral or interpolation operation.
    """
    dependencies = block.get_dependencies()
    prepared = block.prepare_evaluate_tlm(
        [dependency.saved_output for dependency in dependencies],
        [dependency.tlm_value for dependency in dependencies],
        list(enumerate(block.get_outputs())),
    )
    coefficients, directions = operation_directions(block)
    if not isinstance(prepared, BaseForm):
        raise NotImplementedError("directional assembly requires a native symbolic form")
    prepared = sampled_stratum(prepared, coefficients)
    response = directional(prepared, coefficients, directions)
    if len(outputs) != 1:
        raise NotImplementedError("directional assembly requires one output")
    if isinstance(response, Interpolate):
        response = _materialise_operators(response)
    outputs[0].add_tlm_output(assemble(response))


def _evaluate_assignment(block: FunctionAssignBlock, outputs: list[BlockVariable]) -> None:
    """
    @brief Assign the joint response of a pointwise coefficient expression.
    """
    dependencies = block.get_dependencies()
    prepared = block.prepare_evaluate_tlm(
        [dependency.saved_output for dependency in dependencies],
        [dependency.tlm_value for dependency in dependencies],
        list(enumerate(block.get_outputs())),
    )
    if not isinstance(prepared, Expr):
        raise NotImplementedError("directional expression assignment requires a native expression")
    coefficients, directions = operation_directions(block)
    prepared = sampled_stratum(prepared, coefficients)
    response = directional(prepared, coefficients, directions)
    response = _materialise_operators(response)
    for output in outputs:
        result = Function(output.output.function_space())
        result.assign(response)
        output.add_tlm_output(result)
