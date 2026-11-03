# SPDX-FileCopyrightText: 2026 Benjamin Coveler
# SPDX-License-Identifier: LGPL-3.0-or-later

"""
@file ufl_nonsmooth/implicit.py
@brief Native variational-solver recording on the adjoint tape.
@details Converged residuals, scalar Constants, structural zero forms, supplied Jacobians,
    and boundary conditions retain their native pullbacks and replay values.
@author Benjamin Coveler
@date October 2026
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from firedrake import (
    Cofunction,
    Constant,
    DirichletBC,
    Function,
    FunctionSpace,
    MeshGeometry,
    NonlinearVariationalProblem,
    NonlinearVariationalSolver,
    SpatialCoordinate,
    TrialFunction,
    action,
    adjoint,
    assemble,
    derivative,
    solve,
)
from firedrake.adjoint_utils.blocks.solving import GenericSolveBlock, SolveVarFormBlock
from firedrake.matrix import MatrixBase
from mpi4py import MPI
from pyadjoint import annotate_tape, get_working_tape, no_annotations, stop_annotating
from ufl import BaseForm, Form, ZeroBaseForm, as_ufl, conditional, eq, replace, zero
from ufl.algorithms import expand_derivatives
from ufl.algorithms.analysis import extract_coefficients, extract_type
from ufl.checks import is_cellwise_constant
from ufl.classes import Abs, Conditional, MaxValue, MinValue, Restricted
from ufl.core.expr import Expr
from ufl.corealg.traversal import unique_post_traversal
from ufl.domain import extract_unique_domain

from ufl_nonsmooth.primitives import directional
from ufl_nonsmooth.sampling import (
    active_dependencies,
    operation_directions,
    sampled_stratum,
    validate_predicate_sample,
)

if TYPE_CHECKING:
    from pyadjoint.block_variable import BlockVariable


@dataclass(frozen=True)
class _PreparedResidual:
    """
    @brief Saved residual data and one native linear response solve.
    @details The exact state Jacobian is differentiated from the converged residual. A supplied
        primal iteration Jacobian remains a replay dependency and does not define this response.
    """

    form: Form
    jacobian: BaseForm
    solution: Function
    boundaries: list[DirichletBC]
    assemble_options: dict[str, object]
    solve_options: dict[str, object]

    @classmethod
    def from_block(
        cls, block: GenericSolveBlock, output: BlockVariable, prepared: object = None
    ) -> _PreparedResidual:
        """
        @brief Collect exact checkpoint forms, boundary directions, and native solve settings.
        """
        if prepared is None:
            dependencies = block.get_dependencies()
            prepared = block.prepare_evaluate_tlm(
                [dependency.saved_output for dependency in dependencies],
                [dependency.tlm_value for dependency in dependencies],
                list(enumerate(block.get_outputs())),
            )
        if (
            not isinstance(prepared, dict)
            or not isinstance(prepared.get("form"), Form)
            or not isinstance(prepared.get("dFdu"), BaseForm)
        ):
            raise NotImplementedError(
                "an implicit response requires its saved residual and Jacobian"
            )
        if not isinstance(output.saved_output, Function):
            raise NotImplementedError("an implicit response requires a native solution field")
        boundaries = []
        for dependency in block.get_dependencies():
            if not isinstance(dependency.output, DirichletBC):
                continue
            boundary = (
                dependency.saved_output.reconstruct(g=0)
                if dependency.tlm_value is None
                else dependency.tlm_value
            )
            if not isinstance(boundary, DirichletBC):
                raise NotImplementedError(
                    "an implicit response requires native Dirichlet directions"
                )
            boundaries.append(boundary)
        assemble_options = dict(block.assemble_kwargs)
        solve_options = dict(block.adj_kwargs)
        compiler_parameters = solve_options.pop(
            "form_compiler_parameters", block.forward_kwargs.get("form_compiler_parameters")
        )
        if compiler_parameters is not None:
            assemble_options["form_compiler_parameters"] = compiler_parameters
        solve_options.pop("J", None)
        solve_options.pop("Jp", None)
        return cls(
            prepared["form"],
            prepared["dFdu"],
            output.saved_output,
            boundaries,
            assemble_options,
            solve_options,
        )

    def solve(self, forcing: BaseForm, jacobian: BaseForm | None = None) -> Function:
        """
        @brief Assemble and solve one linear response with native boundary and zero-load handling.
        """
        result = Function(self.solution.function_space())
        matrix = assemble(
            self.jacobian if jacobian is None else jacobian,
            bcs=self.boundaries,
            **self.assemble_options,
        )
        forcing = expand_derivatives(forcing)
        load = (
            Cofunction(result.function_space().dual())
            if isinstance(forcing, ZeroBaseForm)
            or isinstance(forcing, Form)
            and not forcing.integrals()
            else assemble(forcing, **self.assemble_options)
        )
        solve(matrix, result, load, **self.solve_options)
        return result


class _ImplicitSolveBlock(SolveVarFormBlock):
    """
    @brief A native residual solve with structural-zero coefficient pullbacks.
    @details The block retains every residual and supplied-Jacobian dependency, including scalar
        native Constants. Predicate-only inputs have a zero weak derivative on the selected stratum.
        Native preparation,
        adjoint solves, boundary rows, and replay remain those of a variational solve.
    """

    residual: Form
    jacobian: BaseForm

    def __init__(self, problem: NonlinearVariationalProblem, options: dict[str, object]) -> None:
        """
        @brief Record the native residual, boundary conditions, Jacobian, and solver options.
        """
        self.residual = problem.F
        self.jacobian = problem.J
        constants = extract_type(self.residual, Constant) | extract_type(self.jacobian, Constant)
        if any(constant.ufl_shape for constant in constants):
            raise NotImplementedError("implicit recording supports scalar native Constants")
        super().__init__(self.residual == 0, problem.u, problem.bcs, **options)
        for constant in sorted(constants, key=Constant.count):
            self.add_dependency(constant, no_duplicates=True)
        for coefficient in self.jacobian.coefficients():
            self.add_dependency(coefficient, no_duplicates=True)
        self.forward_kwargs["J"] = self.jacobian
        compiler_parameters = self.adj_kwargs.pop("form_compiler_parameters", None)
        if compiler_parameters is not None:
            self.assemble_kwargs["form_compiler_parameters"] = compiler_parameters

    def _checkpoint_residual(self) -> dict[Constant, Constant]:
        """
        @brief Bind native scalar Constants to their current saved values in the residual.
        """
        replacements = {
            dependency.output: dependency.saved_output
            for dependency in self.get_dependencies()
            if isinstance(dependency.output, Constant)
        }
        self.lhs = replace(self.residual, replacements)
        return replacements

    @no_annotations
    def prepare_evaluate_adj(
        self,
        inputs: list[object],
        adj_inputs: list[Cofunction | None],
        relevant_dependencies: list[tuple[int, BlockVariable]],
    ) -> dict[str, object]:
        """
        @brief Prepare the native adjoint from the saved residual and scalar Constant values.
        """
        self._checkpoint_residual()
        return super().prepare_evaluate_adj(inputs, adj_inputs, relevant_dependencies)

    @no_annotations
    def prepare_evaluate_tlm(
        self,
        inputs: list[object],
        tlm_inputs: list[object],
        relevant_outputs: list[tuple[int, BlockVariable]],
    ) -> dict[str, object]:
        """
        @brief Prepare the native state Jacobian at the saved scalar Constant values.
        """
        self._checkpoint_residual()
        return super().prepare_evaluate_tlm(inputs, tlm_inputs, relevant_outputs)

    @no_annotations
    def evaluate_adj_component(
        self,
        inputs: list[object],
        adj_inputs: list[Cofunction | None],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Cofunction | float | list[DirichletBC] | None:
        """
        @brief Assemble the native coefficient transpose with an explicit zero-form identity.
        """
        coefficient = block_variable.output
        if not isinstance(coefficient, (Function, Constant)):
            return super().evaluate_adj_component(inputs, adj_inputs, block_variable, idx, prepared)
        if not self.linear and coefficient is self.func:
            return None
        if not isinstance(prepared, dict) or not isinstance(prepared.get("form"), Form):
            raise NotImplementedError(
                "an implicit coefficient pullback requires its saved residual"
            )
        if isinstance(coefficient, Constant):
            variation = expand_derivatives(
                derivative(
                    -action(prepared["form"], prepared["adj_sol"]),
                    block_variable.saved_output,
                    as_ufl(1.0),
                )
            )
            if isinstance(variation, ZeroBaseForm) or (
                isinstance(variation, Form) and not variation.integrals()
            ):
                return 0.0
            return float(assemble(variation, **self.assemble_kwargs))
        space = coefficient.function_space()
        variation = expand_derivatives(
            derivative(
                -prepared["form"],
                block_variable.saved_output,
                TrialFunction(space),
            )
        )
        if isinstance(variation, ZeroBaseForm) or (
            isinstance(variation, Form) and not variation.integrals()
        ):
            result = Cofunction(space.dual())
        else:
            result = assemble(
                action(adjoint(variation), prepared["adj_sol"]), **self.assemble_kwargs
            )
        return result

    @no_annotations
    def evaluate_tlm_component(
        self,
        inputs: list[object],
        tlm_inputs: list[object],
        block_variable: BlockVariable,
        idx: int,
        prepared: object = None,
    ) -> Function:
        """
        @brief Assemble the native selected tangent load and solve with native boundary rows.
        """
        residual = _PreparedResidual.from_block(self, block_variable, prepared)
        variation = Form([])
        for dependency in active_dependencies(self, None if self.linear else self.func):
            target = dependency.saved_output
            if isinstance(target, MeshGeometry):
                target = SpatialCoordinate(target)
            variation += derivative(-residual.form, target, as_ufl(dependency.tlm_value))
        return residual.solve(variation)

    @no_annotations
    def prepare_recompute_component(
        self, inputs: list[object], relevant_outputs: list[tuple[int, BlockVariable]]
    ) -> tuple[Form, int, Function, list[DirichletBC]]:
        """
        @brief Rebind the supplied Jacobian to the same checkpoints and unknown as the residual.
        """
        constants = self._checkpoint_residual()
        prepared = super().prepare_recompute_component(inputs, relevant_outputs)
        replacements = {
            dependency.output: dependency.saved_output
            for dependency in self.get_dependencies()
            if dependency.output in self.jacobian.coefficients()
        }
        replacements.update(constants)
        replacements[self.func] = prepared[2]
        self.forward_kwargs["J"] = replace(self.jacobian, replacements)
        return prepared


class ImplicitSolver(NonlinearVariationalSolver):
    """
    @brief Native nonlinear variational solves recorded through the shared residual boundary.
    @details Ordinary Dirichlet conditions, scalar Constants, and symbolic Jacobians retain
        their native values. Vector Constants require a native derivative contract.
        Restricted-space remaps, separate preconditioner forms, objectives, nullspaces, custom
        application contexts, and solver callbacks require their own recording contract.
    """

    problem: NonlinearVariationalProblem
    recording_options: dict[str, object]

    def __init__(self, problem: NonlinearVariationalProblem, **options: object) -> None:
        """
        @brief Retain the public problem and configure its native forward solver.
        """
        if problem.restrict or problem.u_restrict is not problem.u:
            raise NotImplementedError("implicit recording requires an unrestricted native problem")
        if not isinstance(problem.F, Form) or not isinstance(problem.J, BaseForm):
            raise NotImplementedError(
                "implicit recording requires symbolic native residual and Jacobian forms"
            )
        if problem.Jp is not None or problem.E is not None:
            raise NotImplementedError(
                "implicit recording requires a contract for preconditioners and objectives"
            )
        if any(not isinstance(boundary, DirichletBC) for boundary in problem.bcs):
            raise NotImplementedError(
                "implicit recording supports ordinary Dirichlet boundary conditions"
            )
        unsupported = [
            key
            for key, value in options.items()
            if value is not None
            and (
                key in ("appctx", "nullspace", "transpose_nullspace", "near_nullspace")
                or key.endswith("callback")
            )
        ]
        if unsupported:
            raise NotImplementedError(
                "implicit recording requires a contract for " + ", ".join(unsupported)
            )
        self.problem = problem
        self.recording_options = dict(options)
        super().__init__(problem, **options)
        self.recording_options["solver_parameters"] = dict(self.parameters)
        if problem.form_compiler_parameters is not None:
            self.recording_options["form_compiler_parameters"] = problem.form_compiler_parameters

    def solve(self, bounds: tuple[Function, Function] | None = None) -> None:
        """
        @brief Run the native forward solve and retain its complete residual relation on the tape.
        """
        recording = annotate_tape()
        if recording and bounds is not None:
            raise NotImplementedError(
                "implicit recording requires a variational-inequality response contract"
            )
        block = _ImplicitSolveBlock(self.problem, self.recording_options) if recording else None
        if block is not None:
            get_working_tape().add_block(block)
        with stop_annotating():
            super().solve(bounds=bounds)
        if block is not None:
            block.add_output(self.problem.u.create_block_variable())


def evaluate_solve(block: GenericSolveBlock, outputs: list[BlockVariable]) -> None:
    """
    @brief Solve a regular-state residual action using a native variational solver.
    @details The unknown response and all parameter directions enter one residual action.
        The state response must enter linearly. Parameter switches retain their joint action,
        while active state kinks require an additional implicit uniqueness contract. The solve
        retains the primal linear solver configuration and nullspaces.
    """
    unsupported = [
        key
        for key, value in block.forward_kwargs.items()
        if value is not None and (key in ("Jp", "appctx") or key.endswith("callback"))
    ]
    if unsupported:
        raise NotImplementedError(
            "directional implicit solves require a response contract for " + ", ".join(unsupported)
        )
    if len(outputs) != 1:
        raise NotImplementedError("directional implicit solves require one solution output")
    output = outputs[0]
    prepared = _PreparedResidual.from_block(block, output)
    dependencies = block.get_dependencies()
    solution = prepared.solution
    response = Function(solution.function_space())
    form = prepared.form
    constants: dict[Constant, Function] = {}
    for dependency in dependencies:
        constant = dependency.saved_output
        if dependency.tlm_value is not None and isinstance(constant, Constant):
            if constant.ufl_shape:
                raise NotImplementedError("implicit responses support scalar native Constants")
            mesh = extract_unique_domain(form)
            if mesh is None:
                raise NotImplementedError("a scalar Constant response requires one native mesh")
            constants[constant] = Function(FunctionSpace(mesh, "R", 0)).assign(float(constant))
    if constants:
        form = replace(form, constants)
    form = _regular_state_form(form, solution)
    coefficients, directions = operation_directions(
        block, constants, None if block.linear else block.func
    )
    coefficients.append(solution)
    directions.append(response)
    form = sampled_stratum(form, coefficients)
    residual = directional(form, coefficients, directions)
    if not isinstance(residual, Form):
        raise NotImplementedError("implicit responses require a nonzero native state equation")
    for integral in residual.integrals():
        for node in unique_post_traversal(integral.integrand()):
            if isinstance(node, (Abs, MaxValue, MinValue)):
                tested = node
            elif isinstance(node, Conditional):
                tested = node.ufl_operands[0]
            else:
                continue
            if response in extract_coefficients(tested):
                raise NotImplementedError(
                    "an active implicit state kink requires a uniqueness contract"
                )
    jacobian = expand_derivatives(derivative(residual, response))
    if response in extract_coefficients(jacobian):
        raise NotImplementedError("a nonlinear implicit response requires a uniqueness contract")
    forcing = -replace(residual, {response: zero(response.ufl_shape)})
    output.add_tlm_output(prepared.solve(forcing, jacobian))


def _regular_state_form(form: Form, solution: Function) -> Form:
    """
    @brief Freeze regular or first-order cancelled scalar state branches in a sampled programme.
    @details Piecewise-constant state primitives require explicit degree-zero quadrature.
        All tied absolute values, minima, and maxima are lifted simultaneously to independent
        value-preserving scalar fields. Their complete native residual sensitivities must
        vanish. The remaining composition then has the same first jet for every incident
        branch. Nonzero sensitivities require an implicit uniqueness contract.
    """
    sampled = sampled_stratum(form, [solution])
    if not isinstance(sampled, Form):
        raise NotImplementedError("state branch sampling requires a native residual form")
    form = sampled
    snapshot = solution.copy(deepcopy=True)
    mesh = extract_unique_domain(form)
    if mesh is None:
        raise NotImplementedError("state branch sampling requires one native mesh")
    space = FunctionSpace(mesh, "DG", 0)
    selected: dict[Expr, Expr] = {}
    lifted: dict[Expr, Expr] = {}
    proxies = []
    for integral in form.integrals():
        for node in unique_post_traversal(integral.integrand()):
            if not isinstance(
                node, (Abs, MaxValue, MinValue)
            ) or solution not in extract_coefficients(node):
                continue
            if (
                integral.metadata().get("quadrature_degree") != 0
                or solution.ufl_element().degree() != 0
            ):
                raise NotImplementedError(
                    "implicit state branches require an explicit degree-zero sampling contract"
                )
            if node in selected:
                continue
            restrictions = extract_type(node, Restricted)
            sides = {restriction.side() for restriction in restrictions}
            left = node.ufl_operands[0]
            right = as_ufl(0.0) if isinstance(node, Abs) else node.ufl_operands[1]
            if len(sides) > 1:
                try:
                    validate_predicate_sample(eq(left, right), integral)
                except NotImplementedError as error:
                    raise NotImplementedError(
                        "a mixed-side active state primitive needs its own sampling contract"
                    ) from error
                predicate = replace(
                    left < right if isinstance(node, MinValue) else left > right,
                    {solution: snapshot},
                )
                tied = None
            else:
                local = replace(
                    node, {restriction: restriction.ufl_operands[0] for restriction in restrictions}
                )
                if not is_cellwise_constant(local):
                    raise NotImplementedError(
                        "state branch sampling requires cellwise-constant primitive operands"
                    )
                local_right = as_ufl(0.0) if isinstance(node, Abs) else local.ufl_operands[1]
                sign = Function(space).interpolate(
                    conditional(
                        local.ufl_operands[0] < local_right
                        if isinstance(node, MinValue)
                        else local.ufl_operands[0] > local_right,
                        1.0,
                        0.0,
                    )
                )
                predicate = (sign if not sides else sign(next(iter(sides)))) > 0.5
                tied = Function(space).interpolate(
                    conditional(eq(local.ufl_operands[0], local_right), 1.0, 0.0)
                )
            selected_left, selected_right = replace(left, selected), replace(right, selected)
            lifted_left, lifted_right = replace(left, lifted), replace(right, lifted)
            selected[node] = conditional(
                predicate,
                selected_left,
                -selected_left if isinstance(node, Abs) else selected_right,
            )
            lifted[node] = conditional(
                predicate, lifted_left, -lifted_left if isinstance(node, Abs) else lifted_right
            )
            if tied is None or not space.mesh().comm.allreduce(
                bool(np.any(tied.dat.data_ro != 0.0)), op=MPI.LOR
            ):
                continue
            proxy = Function(space).interpolate(local)
            proxies.append(proxy)
            value = proxy if not sides else proxy(next(iter(sides)))
            equality = (tied if not sides else tied(next(iter(sides)))) > 0.5
            lifted[node] = conditional(equality, value, lifted[node])
    prepared = replace(form, lifted)
    for proxy in proxies:
        variation = expand_derivatives(derivative(prepared, proxy))
        if (
            isinstance(variation, ZeroBaseForm)
            or isinstance(variation, Form)
            and not variation.integrals()
        ):
            continue
        matrix = assemble(variation, mat_type="aij")
        if not isinstance(matrix, MatrixBase):
            raise NotImplementedError(
                "cancelled state branches require a native residual sensitivity matrix"
            )
        entries = matrix.petscmat.getValuesCSR()[2]
        if space.mesh().comm.allreduce(bool(np.any(entries != 0.0)), op=MPI.LOR):
            raise NotImplementedError(
                "an active implicit state kink requires a uniqueness contract"
            )
    return replace(form, selected)
