# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_implicit.py
@brief Native implicit recording, checkpoint replay, and transpose pairings.
@details Residuals retain predicate-only inputs, supplied Jacobians, scalar Constants, and
    nonzero boundary data through native replay and derivative queries.
"""

import numpy as np
import pytest
from firedrake import (
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    MeshGeometry,
    NonlinearVariationalProblem,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    assemble,
    dx,
    grad,
    inner,
)
from firedrake.adjoint import (
    Control,
    ReducedFunctional,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
)
from ufl import as_vector, conditional, max_value

from ufl_nonsmooth.implicit import ImplicitSolver
from ufl_nonsmooth.tape import (
    evaluate_directional,
)


def test_implicit_predicate_only_dependency_retains_zero_pullback_and_replay(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A marked selector has a zero local pullback while replay still changes its branch.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    selector = Function(space).assign(1.0)
    source = Function(space).assign(3.0)
    state = Function(space)
    test = TestFunction(space)
    residual = (state - conditional(selector > 0.5, source, 2.0 * source)) * test * dx(degree=0)
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(selector), Control(source)]
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        reduced = ReducedFunctional(functional, controls)
        gradients = reduced.derivative()
        assert np.allclose(gradients[0].dat.data_ro, 0.0)
        assert box_mesh.comm.allreduce(float(gradients[1].dat.data_ro.sum())) == pytest.approx(
            volume
        )
        tape.reset_tlm_values()
        selector.block_variable.tlm_value = Function(space).assign(1.0)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 0.0)
        evaluate_directional(
            tape, controls, [Function(space).assign(0.2), Function(space).assign(1.0)]
        )
        assert float(functional.block_variable.tlm_value) == pytest.approx(volume)
        assert float(reduced([Function(space).assign(0.0), source])) == pytest.approx(6.0 * volume)
        assert float(reduced([Function(space).assign(1.0), source])) == pytest.approx(3.0 * volume)


@pytest.mark.parametrize("compiler_parameters", [None, {"quadrature_degree": 0}])
def test_implicit_supplied_jacobian_dependencies_replay_with_saved_coefficients(
    box_mesh: MeshGeometry, compiler_parameters: dict[str, int] | None
) -> None:
    """
    @brief A Jacobian-only coefficient remains replayed while its exact solution pullback is zero.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    scale = Function(space).assign(1.0)
    state = Function(space).assign(1.0)
    test = TestFunction(space)
    residual = (state**2 - 4.0) * test * dx
    jacobian = 2.0 * scale * state * TrialFunction(space) * test * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(
            residual, state, J=jacobian, form_compiler_parameters=compiler_parameters
        ),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "snes_max_it": 100},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(scale)
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        reduced = ReducedFunctional(functional, control)
        gradient = reduced.derivative()
        assert np.allclose(gradient.dat.data_ro, 0.0)
        assert float(reduced(Function(space).assign(1.25))) == pytest.approx(2.0 * volume, rel=1e-7)


def test_implicit_nonzero_boundary_data_retains_replay_and_transpose(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Recorded native Dirichlet data retain harmonic values, responses, and replay.
    """
    space = FunctionSpace(box_mesh, "CG", 1)
    boundary_data = Function(space).assign(2.0)
    state = Function(space)
    test = TestFunction(space)
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(boundary_data)
        boundary = DirichletBC(space, boundary_data, 6)
        solver = ImplicitSolver(
            NonlinearVariationalProblem(inner(grad(state), grad(test)) * dx, state, bcs=boundary),
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        assert float(functional) == pytest.approx(2.0 * volume)
        boundary_data.block_variable.tlm_value = Function(space).assign(1.0)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.0)
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.0)
        assert float(functional.block_variable.tlm_value) == pytest.approx(volume)
        reduced = ReducedFunctional(functional, control)
        gradient = reduced.derivative()
        assert box_mesh.comm.allreduce(float(gradient.dat.data_ro.sum())) == pytest.approx(volume)
        assert reduced(Function(space).assign(3.0)) == pytest.approx(3.0 * volume)
        assert reduced(Function(space).assign(2.0)) == pytest.approx(2.0 * volume)


def test_implicit_scalar_constant_retains_saved_pullback_response_and_replay(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A scalar Constant has its analytical solution derivative at the recorded checkpoint.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Constant(2.0)
    state = Function(space).assign(1.0)
    residual = (state**2 - source**2) * TestFunction(space) * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        functional = assemble(state**2 * dx)
        pause_annotation()
        source.assign(7.0)
        reduced = ReducedFunctional(functional, control)
        gradient = reduced.derivative()
        assert float(gradient.dat.data_ro[0]) == pytest.approx(4.0 * volume)
        control.tlm_value = Constant(1.5)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.5)
        assert float(functional.block_variable.tlm_value) == pytest.approx(6.0 * volume)
        evaluate_directional(tape, [control], [1.5])
        assert float(functional.block_variable.tlm_value) == pytest.approx(6.0 * volume)
        assert float(reduced(Constant(3.0))) == pytest.approx(9.0 * volume)
        assert float(reduced(Constant(2.0))) == pytest.approx(4.0 * volume)


def test_implicit_constant_predicate_retains_zero_pullback_and_branch_replay(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Predicate-only Constants remain recorded despite their zero selected pullback.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    selector = Constant(1.0)
    source = Function(space).assign(3.0)
    state = Function(space)
    residual = (
        (state - conditional(selector > 0.5, source, 2.0 * source))
        * TestFunction(space)
        * dx(degree=0)
    )
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(selector), Control(source)]
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        selector.assign(0.0)
        reduced = ReducedFunctional(functional, controls)
        gradients = reduced.derivative()
        assert float(gradients[0].dat.data_ro[0]) == 0.0
        assert box_mesh.comm.allreduce(float(gradients[1].dat.data_ro.sum())) == pytest.approx(
            volume
        )
        controls[0].tlm_value = Constant(0.2)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 0.0)
        evaluate_directional(tape, controls, [0.2, Function(space).assign(1.0)])
        assert float(functional.block_variable.tlm_value) == pytest.approx(volume)
        assert float(reduced([Constant(0.0), source])) == pytest.approx(6.0 * volume)
        assert float(reduced([Constant(1.0), source])) == pytest.approx(3.0 * volume)


def test_implicit_constant_only_in_jacobian_retains_its_saved_value(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A Jacobian-only Constant affects Newton replay and has a zero exact solution response.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    scale = Constant(1.0)
    state = Function(space).assign(1.0)
    test = TestFunction(space)
    residual = (state**2 - 4.0) * test * dx
    jacobian = 2.0 * scale * state * TrialFunction(space) * test * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state, J=jacobian),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "snes_max_it": 100},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(scale)
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        scale.assign(0.5)
        reduced = ReducedFunctional(functional, control)
        gradient = reduced.derivative()
        assert float(gradient.dat.data_ro[0]) == 0.0
        control.tlm_value = Constant(1.0)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 0.0)
        evaluate_directional(tape, [control], [1.0])
        assert float(functional.block_variable.tlm_value) == 0.0
        assert float(reduced(Constant(1.25))) == pytest.approx(2.0 * volume, rel=1e-7)


def test_implicit_joint_constant_switch_uses_one_joint_action(box_mesh: MeshGeometry) -> None:
    """
    @brief Equal rising scalar Constants raise their implicit maximum once.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first, second = Constant(0.0), Constant(0.0)
    state = Function(space)
    residual = (state - max_value(first, second)) * TestFunction(space) * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(first), Control(second)]
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        for first_direction, second_direction, expected in [
            (1.0, 1.0, 1.0),
            (1.0, -2.0, 1.0),
            (-2.0, -3.0, -2.0),
        ]:
            evaluate_directional(tape, controls, [first_direction, second_direction])
            assert np.allclose(state.block_variable.tlm_value.dat.data_ro, expected)
            assert float(functional.block_variable.tlm_value) == pytest.approx(expected * volume)


def test_implicit_vector_constant_requires_a_native_response_contract(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Vector Constants are refused instead of silently dropping their dependency.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Constant([1.0, 2.0, 3.0])
    state = Function(space)
    residual = (state - inner(source, as_vector((1.0, 0.0, 0.0)))) * TestFunction(space) * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        try:
            with pytest.raises(NotImplementedError, match="scalar native Constants"):
                solver.solve()
        finally:
            pause_annotation()


@pytest.mark.parametrize("iteration_scale", [1.0, 1.5])
def test_residual_response_preserves_sampling_and_ignores_iteration_scaling(
    box_mesh: MeshGeometry, iteration_scale: float
) -> None:
    """
    @brief Selected and joint responses follow the sampled residual rather than its iteration matrix.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(2.0)
    state = Function(space)
    test = TestFunction(space)
    coordinate = SpatialCoordinate(box_mesh)
    expected = Function(space).interpolate(coordinate[0] ** 2)
    residual = (state - source * coordinate[0] ** 2) * test * dx
    jacobian = iteration_scale * TrialFunction(space) * test * dx
    solver = ImplicitSolver(
        NonlinearVariationalProblem(
            residual, state, J=jacobian, form_compiler_parameters={"quadrature_degree": 0}
        ),
        solver_parameters={
            "ksp_type": "preonly",
            "pc_type": "lu",
            "snes_max_it": 80,
            "snes_atol": 1e-12,
            "snes_rtol": 1e-12,
        },
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        assert np.allclose(state.dat.data_ro, 2.0 * expected.dat.data_ro)
        control.tlm_value = Function(space).assign(1.0)
        tape.evaluate_tlm()
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, expected.dat.data_ro)
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, expected.dat.data_ro)
        expected_response = assemble(expected * dx)
        assert float(functional.block_variable.tlm_value) == pytest.approx(expected_response)
        reduced = ReducedFunctional(functional, control)
        derivative = reduced.derivative()
        assert box_mesh.comm.allreduce(float(derivative.dat.data_ro.sum())) == pytest.approx(
            expected_response
        )
        assert reduced(Function(space).assign(3.0)) == pytest.approx(3.0 * expected_response)
