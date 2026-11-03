# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file tests/test_tape.py
@brief Joint directional propagation through native recorded operations.
@details Analytical switch responses, implicit state-kink refusals, replay, and smooth
    transpose pairings exercise a single causal computation tape.
"""

import numpy as np
import pytest
from firedrake import (
    DirichletBC,
    Function,
    FunctionSpace,
    MeshGeometry,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    assemble,
    dS,
    dx,
    grad,
    inner,
    solve,
)
from firedrake.adjoint import (
    AdjFloat,
    Control,
    ReducedFunctional,
    Tape,
    continue_annotation,
    pause_annotation,
    set_working_tape,
    stop_annotating,
)
from ufl import as_ufl, as_vector, conditional, max_value

from tests.fixtures.laws import SquareLaw
from ufl_nonsmooth.application import apply_law
from ufl_nonsmooth.implicit import ImplicitSolver
from ufl_nonsmooth.piecewise import PiecewiseLinear
from ufl_nonsmooth.tape import assemble_operator, evaluate_directional


def test_joint_interpolation_and_assembly(box_mesh: MeshGeometry) -> None:
    """
    @brief Raising both equal arguments by one raises their maximum by one throughout the tape.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first, second = Function(space), Function(space)
    direction = Function(space).assign(1.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(first), Control(second)]
        result = Function(space).interpolate(max_value(first, second))
        functional = assemble(result * dx)
        pause_annotation()
        block_count = len(tape.get_blocks())
        evaluate_directional(tape, controls, [direction, direction])
        assert np.allclose(result.block_variable.tlm_value.dat.data_ro, 1.0)
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            assemble(1.0 * dx(domain=box_mesh))
        )
        assert np.allclose(result.dat.data_ro, 0.0)
        assert len(tape.get_blocks()) == block_count
        evaluate_directional(tape, controls, [direction, Function(space).assign(-1.0)])
        assert np.allclose(result.block_variable.tlm_value.dat.data_ro, 1.0)


def test_implicit_material_knot_requires_a_uniqueness_contract(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A state-dependent constitutive kink needs an explicit implicit response contract.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, state = Function(space), Function(space)
    material = PiecewiseLinear([(-2.0, 0.0), (0.0, 0.0), (2.0, 2.0)])
    residual = (state + material.expression(state) - source) * TestFunction(space) * dx(degree=0)
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "mat_type": "aij"},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        pause_annotation()
        with pytest.raises(NotImplementedError, match="implicit state kink"):
            evaluate_directional(tape, [control], [Function(space).assign(1.0)])


def test_implicit_strict_state_branch_retains_its_linear_response(box_mesh: MeshGeometry) -> None:
    """
    @brief A separated discrete state branch retains its native linear implicit action.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, state = Function(space).assign(2.0), Function(space).assign(1.0)
    residual = (state + max_value(state, 0.0) - source) * TestFunction(space) * dx(degree=0)
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        objective = assemble(state * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(2.0)])
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.0)
        gradient = ReducedFunctional(objective, control).derivative()
        pairing = box_mesh.comm.allreduce(
            float(gradient.dat.data_ro @ np.full_like(source.dat.data_ro, 2.0))
        )
        assert float(objective.block_variable.tlm_value) == pytest.approx(pairing)


def test_cancelled_state_kink_retains_the_complete_regular_first_jet(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief The square of an absolute value has zero first sensitivity to its tied primitive.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, state = Function(space), Function(space)
    residual = (state + abs(as_ufl(state)) ** 2 - source) * TestFunction(space) * dx(degree=0)
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu", "snes_atol": 1e-13},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        objective = assemble(state * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.0)
        reduced = ReducedFunctional(objective, control)
        gradient = reduced.derivative()
        expected = float(assemble(1.0 * dx(domain=box_mesh)))
        assert float(objective.block_variable.tlm_value) == pytest.approx(expected)
        pairing = box_mesh.comm.allreduce(float(np.sum(gradient.dat.data_ro)))
        assert pairing == pytest.approx(expected)
        plus = float(reduced(Function(space).assign(1e-5)))
        minus = float(reduced(Function(space).assign(-1e-5)))
        assert (plus - minus) / 2e-5 == pytest.approx(expected, rel=1e-7)


def test_nested_state_kinks_cannot_certify_each_other_by_selected_slopes(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Joint primitive lifting retains the nonzero sensitivity of an outer active maximum.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, state = Function(space), Function(space)
    residual = (
        (state + max_value(abs(as_ufl(state)), 0.0) - source) * TestFunction(space) * dx(degree=0)
    )
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        pause_annotation()
        with pytest.raises(NotImplementedError, match="implicit state kink"):
            evaluate_directional(tape, [control], [Function(space).assign(1.0)])


def test_joint_assembled_maximum(box_mesh: MeshGeometry) -> None:
    """
    @brief Direct weak-form assembly keeps simultaneous inputs at the same switch coupled.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first, second = Function(space), Function(space)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(first), Control(second)]
        functional = assemble(max_value(first, second) * dx)
        pause_annotation()
        direction = Function(space).assign(1.0)
        evaluate_directional(tape, controls, [direction, direction])
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            assemble(1.0 * dx(domain=box_mesh))
        )


@pytest.mark.parametrize("solver_type", [NonlinearVariationalSolver, ImplicitSolver])
def test_smooth_response_pairs_with_native_adjoint(
    box_mesh: MeshGeometry, solver_type: type[NonlinearVariationalSolver]
) -> None:
    """
    @brief Smooth diffusion with homogeneous boundary data retains the native transpose pairing.
    """
    space = FunctionSpace(box_mesh, "CG", 1)
    source = Function(space).assign(2.0)
    state = Function(space)
    test = TestFunction(space)
    residual = (inner(grad(state), grad(test)) + (state - source) * test) * dx
    boundary = DirichletBC(space, 0.0, 6)
    solver = solver_type(
        NonlinearVariationalProblem(residual, state, bcs=boundary),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    direction = Function(space).interpolate(1.0 + SpatialCoordinate(box_mesh)[0])
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        functional = assemble(state**2 * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [direction])
        response = float(functional.block_variable.tlm_value)
        assert response > 0.0
        gradient = ReducedFunctional(functional, control).derivative()
        pairing = box_mesh.comm.allreduce(float(gradient.dat.data_ro @ direction.dat.data_ro))
        assert response == pytest.approx(pairing, rel=1.0e-9, abs=1.0e-12)


def test_control_nodes_cut_their_producing_operation(box_mesh: MeshGeometry) -> None:
    """
    @brief A control introduced inside the tape supplies its prescribed independent response.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first = Function(space).assign(2.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        first_control = Control(first)
        intermediate = Function(space).interpolate(first**2)
        second_control = Control(intermediate)
        functional = assemble(intermediate * dx)
        pause_annotation()
        evaluate_directional(
            tape, [first_control, second_control], [Function(space).assign(1.0), Function(space)]
        )
        assert float(functional.block_variable.tlm_value) == 0.0


def test_boundary_control_is_propagated_to_the_directional_solve(box_mesh: MeshGeometry) -> None:
    """
    @brief A constant perturbation of Dirichlet data produces the constant harmonic response.
    """
    space = FunctionSpace(box_mesh, "CG", 1)
    boundary_data = Function(space)
    state = Function(space)
    test = TestFunction(space)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(boundary_data)
        boundary = DirichletBC(space, boundary_data, 6)
        solver = NonlinearVariationalSolver(
            NonlinearVariationalProblem(inner(grad(state), grad(test)) * dx, state, bcs=boundary),
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
        solver.solve()
        functional = assemble(state * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert np.allclose(state.block_variable.tlm_value.dat.data_ro, 1.0)
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            assemble(1.0 * dx(domain=box_mesh))
        )


def test_joint_geometry_and_coefficients_share_one_assembly_response(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief Equal rising coefficients and an axial mesh stretch contribute jointly to an integral.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    first, second = Function(space).assign(1.0), Function(space).assign(1.0)
    field_direction = Function(space).assign(1.0)
    vector_space = VectorFunctionSpace(box_mesh, "CG", 1)
    coordinates = SpatialCoordinate(box_mesh)
    shape_direction = Function(vector_space).interpolate(as_vector((coordinates[0], 0.0, 0.0)))
    volume = assemble(1.0 * dx(domain=box_mesh))
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        controls = [Control(first), Control(second), Control(box_mesh)]
        functional = assemble(max_value(first, second) * dx)
        pause_annotation()
        evaluate_directional(tape, controls, [field_direction, field_direction, shape_direction])
        assert float(functional.block_variable.tlm_value) == pytest.approx(2.0 * volume)


@pytest.mark.parametrize("point,expected", [(-1.0, -2.0), (1.0, 3.0)])
@pytest.mark.parametrize("operation", ["form", "interpolation"])
def test_separated_sampled_predicate_keeps_its_local_stratum(
    box_mesh: MeshGeometry, point: float, expected: float, operation: str
) -> None:
    """
    @brief A strict sampled guard uses its local branch without losing payload dependence.
    """
    space = (
        FunctionSpace(box_mesh, "CG", 1)
        if operation == "interpolation"
        else FunctionSpace(box_mesh, "DG", 0)
    )
    source = Function(space).assign(point)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        expression = conditional(source < 0.0, source**2, 3.0 * source)
        if operation == "interpolation":
            result = Function(space).interpolate(expression)
            functional = assemble(result * dx)
        else:
            functional = assemble(expression * dx(degree=0))
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            expected * assemble(1.0 * dx(domain=box_mesh))
        )


def test_active_sampled_predicate_requires_an_event_response(box_mesh: MeshGeometry) -> None:
    """
    @brief A sampled unit-step event cannot silently return a selected finite gradient.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        functional = assemble(conditional(source >= 0.0, 1.0, 0.0) * dx(degree=0))
        pause_annotation()
        with pytest.raises(NotImplementedError, match="active sampled predicate"):
            evaluate_directional(tape, [control], [Function(space).assign(-1.0)])
        assert functional.block_variable.tlm_value is None


def test_inferred_quadrature_does_not_certify_a_moving_predicate(box_mesh: MeshGeometry) -> None:
    """
    @brief A guard without an explicit sampling contract retains the general refusal boundary.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(1.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        assemble(conditional(source >= 0.0, source, -source) * dx)
        pause_annotation()
        with pytest.raises(NotImplementedError, match="declared degree-zero sampling"):
            evaluate_directional(tape, [control], [Function(space).assign(1.0)])


def test_native_external_operator_retains_its_jet_through_a_pointwise_consumer(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief An opaque native smooth jet remains differentiable through a pointwise consumer.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    mach = Function(space).assign(6.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(mach)
        angle = assemble_operator(apply_law(SquareLaw(), [mach], function_space=space)[0])
        result = Function(space).interpolate(2.0 * angle + mach**2)
        functional = assemble(result * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        slope = 12.0
        expected = (2.0 * slope + 12.0) * assemble(1.0 * dx(domain=box_mesh))
        assert float(functional.block_variable.tlm_value) == pytest.approx(expected)


def test_scalar_tape_direction_retains_the_assigned_field_type(box_mesh: MeshGeometry) -> None:
    """
    @brief A scalar trajectory control reaches field consumers as a native coefficient direction.
    """
    space = FunctionSpace(box_mesh, "R", 0)
    parameter = AdjFloat(2.0)
    field = Function(space)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(parameter)
        field.assign(parameter)
        functional = assemble(field**2 * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [3.0])
        assert isinstance(field.block_variable.tlm_value, Function)
        assert np.allclose(field.block_variable.tlm_value.dat.data_ro, 3.0)
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            12.0 * assemble(1.0 * dx(domain=box_mesh))
        )


def test_implicit_custom_application_context_requires_a_response_contract(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A solver's custom primal application context cannot silently enter its response solve.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(1.0)
    state = Function(space)
    test = TestFunction(space)
    solver = NonlinearVariationalSolver(
        NonlinearVariationalProblem((state - source) * test * dx, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        appctx={"source": source},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        pause_annotation()
        with pytest.raises(NotImplementedError, match="response contract for appctx"):
            evaluate_directional(tape, [control], [Function(space).assign(1.0)])


@pytest.mark.parametrize("operand", ["spatial", "reference", "native_tie"])
def test_opaque_operator_coordinate_transpose_retains_the_native_selected_jet(
    box_mesh: MeshGeometry, operand: str
) -> None:
    """
    @brief Exact coordinate motion and reference-fixed samples obey their selected transpose.
    @details At an equal maximum, the native reverse selects the right reference field. Its
        zero coordinate response differs from the graphical action towards the left operand.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    coordinate = SpatialCoordinate(box_mesh)[0]
    field = Function(FunctionSpace(box_mesh, "CG", 1)).interpolate(coordinate)
    if operand == "spatial":
        mach = 2.0 + coordinate
    elif operand == "reference":
        mach = Function(space).assign(2.0)
    else:
        mach = 2.0 + max_value(coordinate, field)
    direction = Function(box_mesh.coordinates.function_space()).interpolate(
        as_vector((1.0, 0.0, 0.0))
    )
    spatial_response = assemble(2.0 * (2.0 + coordinate) * dx(degree=0))
    expected = spatial_response if operand == "spatial" else 0.0
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(box_mesh)
        result = assemble_operator(apply_law(SquareLaw(), [mach], function_space=space)[0])
        functional = assemble(result * dx(degree=0))
        pause_annotation()
        evaluate_directional(tape, [control], [direction])
        graphical = spatial_response if operand == "native_tie" else expected
        assert float(functional.block_variable.tlm_value) == pytest.approx(graphical, abs=1e-12)
        tape.reset_variables(types=("adjoint",))
        functional.block_variable.adj_value = 1.0
        with stop_annotating():
            tape.evaluate_adj()
        gradient = control.adj_value
        with gradient.dat.vec_ro as covector, direction.dat.vec_ro as tangent:
            pairing = covector.dot(tangent)
        assert pairing == pytest.approx(expected, abs=1e-12)


def test_linear_solve_retains_an_old_value_on_its_right_hand_side(box_mesh: MeshGeometry) -> None:
    """
    @brief A native in-place linear update differentiates its saved right-hand-side state.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    state = Function(space).assign(2.0)
    trial, test = TrialFunction(space), TestFunction(space)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(state)
        solve(
            trial * test * dx == 2.0 * state * test * dx,
            state,
            solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
        )
        functional = assemble(state * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [Function(space).assign(1.0)])
        assert np.allclose(state.dat.data_ro, 4.0)
        assert float(functional.block_variable.tlm_value) == pytest.approx(
            2.0 * assemble(1.0 * dx(domain=box_mesh))
        )


def test_sampled_facet_predicate_retains_its_enclosing_restriction(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A strict facet predicate and its directional payload use the same restricted trace.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source = Function(space).assign(2.0)
    direction = Function(space).assign(3.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        expression = conditional(source > 0.0, max_value(source, 0.0), source**2)
        functional = assemble(expression("+") * dS(degree=0))
        pause_annotation()
        evaluate_directional(tape, [control], [direction])
        expected = 3.0 * assemble(1.0 * dS(domain=box_mesh, degree=0))
        assert float(functional.block_variable.tlm_value) == pytest.approx(expected)


def test_requested_physical_outputs_exclude_an_unconnected_active_jump(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A physical observable traverses only its own retained causal source programme.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, jump, smooth = Function(space), Function(space), Function(space)
    direction = Function(space).assign(1.0)
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        jump.interpolate(conditional(source > 0.0, 1.0, 0.0))
        smooth.interpolate(source**2)
        functional = assemble(smooth * dx)
        pause_annotation()
        evaluate_directional(tape, [control], [direction], [functional.block_variable])
        assert float(functional.block_variable.tlm_value) == 0.0
        assert jump.block_variable.tlm_value is None
        with pytest.raises(NotImplementedError, match="graphical event"):
            evaluate_directional(tape, [control], [direction], [jump.block_variable])


def test_state_sampling_refuses_spatially_varying_primitive_operands(
    box_mesh: MeshGeometry,
) -> None:
    """
    @brief A constant unknown cannot qualify a primitive that also varies within each cell.
    """
    space = FunctionSpace(box_mesh, "DG", 0)
    source, state = Function(space).assign(4.0), Function(space).assign(1.0)
    coordinate = SpatialCoordinate(box_mesh)[0]
    residual = (
        (state + max_value(state + coordinate, 0.0) ** 2 - source)
        * TestFunction(space)
        * dx(degree=0)
    )
    solver = ImplicitSolver(
        NonlinearVariationalProblem(residual, state),
        solver_parameters={"ksp_type": "preonly", "pc_type": "lu"},
    )
    tape = Tape()
    with set_working_tape(tape):
        continue_annotation()
        control = Control(source)
        solver.solve()
        pause_annotation()
        with pytest.raises(NotImplementedError, match="cellwise-constant primitive"):
            evaluate_directional(tape, [control], [Function(space).assign(1.0)])
