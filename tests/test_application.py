# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_application.py
@brief The common scalar, field, and symbolic local-law application boundary.
@details Analytical sum, product, and square laws exercise native values, output storage,
    operand types, scalar spaces, and mesh ownership.
"""

import numpy as np
import pytest
from firedrake import BoxMesh, Function, FunctionSpace, MeshGeometry

from tests.fixtures.laws import CoupledLaw, SquareLaw
from ufl_nonsmooth.application import apply_law


def test_scalar_application_retains_all_outputs() -> None:
    """
    @brief Two scalar operands supply both analytical sum and product outputs.
    """
    assert apply_law(CoupledLaw(), [2.0, 3.0]) == pytest.approx([5.0, 6.0])


def test_field_application_uses_preallocated_outputs(box_mesh: MeshGeometry) -> None:
    """
    @brief Numerical and symbolic laws assign their analytical values into native fields.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first = Function(space).assign(2.0)
    second = Function(space).assign(3.0)
    outputs = [Function(space), Function(space)]
    apply_law(CoupledLaw(), [first, second], outputs)
    assert np.allclose(outputs[0].dat.data_ro, 5.0)
    assert np.allclose(outputs[1].dat.data_ro, 6.0)
    apply_law(SquareLaw(), [first], [outputs[0]])
    assert np.allclose(outputs[0].dat.data_ro, 4.0)


def test_application_requires_operands_and_native_output_storage(box_mesh: MeshGeometry) -> None:
    """
    @brief The common entry point rejects incomplete or incompatible storage contracts.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    field = Function(space)
    with pytest.raises(ValueError, match="at least one operand"):
        apply_law(CoupledLaw(), [])
    with pytest.raises(TypeError, match="preallocated outputs"):
        apply_law(CoupledLaw(), [field, field])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="output function space"):
        apply_law(SquareLaw(), [field])  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="every output"):
        apply_law(SquareLaw(), [field], [Function(space), Function(space)])


def test_symbolic_application_preserves_mesh_ownership(box_mesh: MeshGeometry) -> None:
    """
    @brief Native symbolic operands must belong to the output mesh.
    """
    other = BoxMesh(1, 1, 1, 1.0, 1.0, 1.0)
    field = Function(FunctionSpace(box_mesh, "DG", 0))
    with pytest.raises(ValueError, match="output mesh"):
        apply_law(SquareLaw(), [field], function_space=FunctionSpace(other, "DG", 0))
