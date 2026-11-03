# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/application.py
@brief One application boundary for numerical and native symbolic local laws.
@details Numerical laws record scalar or field operands. Symbolic laws produce native external
    operators for expressions, or assign those operators into preallocated output fields.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from typing import overload

from firedrake import Function
from firedrake.functionspaceimpl import WithGeometry
from ufl.core.expr import Expr
from ufl.domain import extract_unique_domain

from ufl_nonsmooth.field import apply_fields
from ufl_nonsmooth.laws import ArrayLaw, SymbolicLaw
from ufl_nonsmooth.scalar import evaluate_scalar
from ufl_nonsmooth.symbolic import NativeLaw
from ufl_nonsmooth.tape import assemble_operator


@overload
def apply_law(law: ArrayLaw, inputs: list[float]) -> list[float]: ...


@overload
def apply_law(
    law: ArrayLaw | SymbolicLaw, inputs: list[Function], outputs: list[Function]
) -> None: ...


@overload
def apply_law(
    law: SymbolicLaw, inputs: list[Expr] | list[Function], *, function_space: WithGeometry
) -> list[Expr]: ...


def apply_law(
    law: ArrayLaw | SymbolicLaw,
    inputs: list[float] | list[Function] | list[Expr],
    outputs: list[Function] | None = None,
    *,
    function_space: WithGeometry | None = None,
) -> list[float] | list[Expr] | None:
    """
    @brief Apply a local law to scalars, preallocated fields, or a native expression space.
    @param law: Numerical or symbolic law supplying values and both first-order responses.
    @param inputs: Ordered scalar or native operands.
    @param outputs: Preallocated native output fields for field evaluation.
    @param function_space: Native scalar output space when constructing symbolic expressions.
    @return Scalar outputs, native expressions, or no value when assigning output fields.
    """
    if not inputs:
        raise ValueError("a local law requires at least one operand")
    if isinstance(law, ArrayLaw):
        if function_space is not None:
            raise TypeError("a numerical law requires scalar values or preallocated fields")
        if outputs is None:
            if not all(isinstance(value, (float, int)) for value in inputs):
                raise TypeError("numerical field operands require preallocated outputs")
            scalars = [value for value in inputs if isinstance(value, (float, int))]
            return evaluate_scalar(law, scalars)
        fields = [value for value in inputs if isinstance(value, Function)]
        if len(fields) != len(inputs):
            raise TypeError("numerical field application requires native field operands")
        apply_fields(law, fields, outputs)
        return None
    operands = [value for value in inputs if isinstance(value, Expr)]
    if len(operands) != len(inputs):
        raise TypeError("symbolic local laws require native expression operands")
    if outputs is not None:
        if function_space is not None or not outputs:
            raise ValueError("provide preallocated outputs or a symbolic output space")
        function_space = outputs[0].function_space()
        if any(output.function_space() != function_space for output in outputs):
            raise ValueError("a symbolic law's outputs must share one native scalar space")
    if function_space is None:
        raise ValueError("symbolic local application requires an output function space")
    if any(
        extract_unique_domain(value) is not None
        and extract_unique_domain(value) != function_space.mesh()
        for value in operands
    ):
        raise ValueError("a symbolic local law requires operands on its output mesh")
    if function_space.value_size != 1:
        raise ValueError("local laws require scalar output function spaces")
    count = len(law.evaluate(operands).values)
    expressions: list[Expr] = [
        NativeLaw(
            *operands, function_space=function_space, operator_data={"law": law, "output": index}
        )
        for index in range(count)
    ]
    if outputs is None:
        return expressions
    if len(outputs) != count:
        raise ValueError("a local law needs one preallocated field for every output")
    for output, expression in zip(outputs, expressions, strict=True):
        output.assign(assemble_operator(expression))
    return None
