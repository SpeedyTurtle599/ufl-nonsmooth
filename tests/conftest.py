# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/conftest.py
@brief Native mesh fixture for the calculus test suite.
@details A translated Cartesian box resolves scalar, vector, facet, and coordinate responses.
"""

import pytest
from firedrake import BoxMesh, MeshGeometry


@pytest.fixture(scope="session")
def box_mesh() -> MeshGeometry:
    """
    @brief A unit-length box with half-unit width and a thickness of 0.02.
    """
    mesh = BoxMesh(4, 2, 2, 1.0, 0.5, 0.02)
    mesh.coordinates.dat.data[:] -= [1.0, 0.25, 0.01]
    return mesh
