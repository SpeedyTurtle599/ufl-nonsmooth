# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/symbolic.py
@brief Native external operators for symbolic local laws.
@details One adapter registers values, selected linearisations, transposes, and joint actions.
    Every local output retains the same operands and its declared native function space.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from firedrake import Argument, Cofunction, Function, Interpolate, action, adjoint, assemble
from firedrake.external_operators import AbstractExternalOperator, assemble_method
from firedrake.matrix import AssembledMatrix, Matrix, MatrixBase
from numpy.typing import NDArray
from petsc4py import PETSc
from ufl import as_ufl, replace
from ufl.algorithms.analysis import extract_arguments
from ufl.core.expr import Expr
from ufl.form import BaseForm

from ufl_nonsmooth.laws import SymbolicLaw

if TYPE_CHECKING:
    from firedrake.functionspaceimpl import WithGeometry


class NativeLaw(AbstractExternalOperator):
    """
    @brief One output of a symbolic law represented in a native scalar space.
    """

    law: SymbolicLaw
    output_index: int

    def __init__(
        self,
        *operands: Expr,
        function_space: WithGeometry,
        derivatives: tuple[int, ...] | None = None,
        argument_slots: tuple[Expr, ...] = (),
        operator_data: dict[str, object] | None = None,
    ) -> None:
        """
        @brief Bind a local law and output index throughout native symbolic reconstruction.
        """
        if operator_data is None:
            raise ValueError("a native law requires its model and output index")
        law, output = operator_data.get("law"), operator_data.get("output")
        if not isinstance(law, SymbolicLaw) or not isinstance(output, int):
            raise TypeError("a native law requires a symbolic model and an integer output index")
        if any(operand.ufl_shape for operand in operands):
            raise ValueError("native local laws require scalar operands")
        self.law, self.output_index = law, output
        super().__init__(
            *operands,
            function_space=function_space,
            derivatives=derivatives,
            argument_slots=argument_slots,
            operator_data=operator_data,
        )
        if not self.ufl_element().pullback.is_identity:
            raise ValueError("native local laws require an identity output pullback")

    @assemble_method(0, (0,))  # ty: ignore[invalid-argument-type]
    def assemble_value(self, assembly_opts: dict[str, object] | None = None) -> Function:
        """
        @brief Interpolate the local value in the declared native output space.
        """
        evaluation = self.law.evaluate(list(self.ufl_operands))
        return assemble(Interpolate(evaluation.values[self.output_index], self.function_space()))

    def selected_action(self, operand_actions: list[Expr | None]) -> Expr:
        """
        @brief Apply the selected partials linearly to the supplied operand actions.
        """
        partials = self.law.evaluate(list(self.ufl_operands)).partials[self.output_index]
        value = sum(
            (
                partial * direction
                for partial, direction in zip(partials, operand_actions, strict=True)
                if direction is not None
            ),
            as_ufl(0.0),
        )
        return Interpolate(value, self.function_space())

    def directional_action(self, operand_actions: list[Expr | None]) -> Expr:
        """
        @brief Apply the complete joint local action while retaining symbolic perturbations.
        """
        values = self.law.directional(list(self.ufl_operands), operand_actions)
        return Interpolate(values[self.output_index], self.function_space())

    def _selected_partial(self) -> Expr:
        """
        @brief Read one validated first-order operand partial of the selected local law.
        """
        derivatives = self.derivatives
        if derivatives is None or sum(derivatives) != 1:
            raise NotImplementedError("native local-law assembly supports first-order derivatives")
        index = derivatives.index(1)
        return self.law.evaluate(list(self.ufl_operands)).partials[self.output_index][index]

    def jacobian_form(self) -> BaseForm:
        """
        @brief Build the selected input partial in native canonical argument order.
        """
        partial = self._selected_partial()
        slot = self.argument_slots()[-1]
        arguments = {
            argument: Argument(argument.ufl_function_space(), 1)
            for argument in extract_arguments(slot)
        }
        return Interpolate(partial * replace(slot, arguments), self.function_space())

    @assemble_method(1, (0, 1))  # ty: ignore[invalid-argument-type]
    def assemble_jacobian(self, assembly_opts: dict[str, object] | None = None) -> MatrixBase:
        """
        @brief Assemble the selected local Jacobian in the native operand space.
        """
        form = self.jacobian_form()
        return _assemble_selected_matrix(form)

    @assemble_method(1, (1, 0))  # ty: ignore[invalid-argument-type]
    def assemble_adjoint(self, assembly_opts: dict[str, object] | None = None) -> MatrixBase:
        """
        @brief Assemble the transpose of the same selected local Jacobian.
        """
        form = self.jacobian_form()
        matrix = _assemble_selected_matrix(form)
        return Matrix(adjoint(form), matrix.petscmat.hermitianTranspose())

    @assemble_method(1, (0, None))  # ty: ignore[invalid-argument-type]
    def assemble_jvp(self, assembly_opts: dict[str, object] | None = None) -> Function:
        """
        @brief Apply one selected local partial to its native input perturbation.
        """
        partial = self._selected_partial()
        return assemble(Interpolate(partial * self.argument_slots()[-1], self.function_space()))

    @assemble_method(1, (None, 0))  # ty: ignore[invalid-argument-type]
    def assemble_vjp(self, assembly_opts: dict[str, object] | None = None) -> Cofunction:
        """
        @brief Apply the selected transpose to the supplied native output covector.
        """
        form = action(adjoint(self.jacobian_form()), self.argument_slots()[0])
        return assemble(form)

    @assemble_method(1, (None, None))  # ty: ignore[invalid-argument-type]
    def assemble_pairing(
        self, assembly_opts: dict[str, object] | None = None
    ) -> float | Cofunction:
        """
        @brief Pair the selected local action with a native output covector.
        """
        if extract_arguments(self.argument_slots()[-1]):
            return assemble(action(adjoint(self.jacobian_form()), self.argument_slots()[0]))
        partial = self._selected_partial()
        response = assemble(Interpolate(partial * self.argument_slots()[-1], self.function_space()))
        with response.dat.vec_ro as vector, self.argument_slots()[0].dat.vec_ro as covector:
            return covector.dot(vector)


def _assemble_selected_matrix(form: BaseForm) -> Matrix:
    """
    @brief Assemble selected native Jacobians with explicit one-column shared-scalar storage.
    @details A shared real operand has one global degree of freedom. One native action reads
        its complete column, which a sparse matrix stores for composition with other operators.
    """
    native = assemble(form)
    if not isinstance(native, (Matrix, AssembledMatrix)):
        raise NotImplementedError("a local-law Jacobian requires an assembled native matrix")
    matrix = native.petscmat
    if matrix.getType() != "python" or matrix.getSize()[1] != 1:
        return Matrix(form, matrix)
    direction, response = matrix.createVecs()
    direction.set(1.0)
    matrix.mult(direction, response)
    explicit = PETSc.Mat().createAIJ(size=matrix.getSizes(), nnz=1, comm=matrix.comm)
    start, stop = explicit.getOwnershipRange()
    values: NDArray[np.float64] = np.asarray(response.array_r, dtype=np.float64)
    explicit.setValues(list(range(start, stop)), [0], values.tolist())
    explicit.assemble()
    return Matrix(form, explicit)
