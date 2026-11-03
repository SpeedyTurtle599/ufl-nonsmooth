# Native non-smooth calculus


The toolkit supports local piecewise-smooth laws, continuous piecewise-linear tables, and native
Firedrake directional propagation. It does not construct general selection graphs or differentiate
changes of mesh topology, eigenspace selection, numerical rank, or active history branches.
A consumer with an unsupported active event raises an error naming that event. Import toolkit
objects from their defining modules. The package overview does not re-export them.

Local law field outputs use scalar spaces with identity pullbacks. A mapped output space
requires its own coefficient and coordinate action and is refused at application. Generic native
interpolation retains its separate pullback support.

`ufl_nonsmooth/laws.py` defines the common local-law contract. An `ArrayLaw` uses NumPy arrays,
including zero-dimensional scalar arrays. A `SymbolicLaw` uses native Unified Form Language
expressions. Both supply `evaluate(inputs)` and `directional(inputs, directions)` with every
operand declared in order. Evaluation returns `LocalEvaluation(values, partials)` with one value
and one partial row per output, and one partial column per input. Selected partials describe a
linear branch convention. The joint directional action can be nonlinear in its perturbations.
An absent direction freezes an operand, while an explicit zero keeps it active.

`apply_law` in `ufl_nonsmooth/application.py` is the common application entry point. Numerical
scalar operands return scalar outputs and retain their native adjoint dependencies when recording:

```python
from ufl_nonsmooth.application import apply_law
from ufl_nonsmooth.piecewise import PiecewiseLinear

chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)])
value = apply_law(chart, [coordinate])[0]
```

Numerical field operands write into preallocated output fields. Each operand uses the output's
scalar space or a shared real space. Symbolic laws can write into preallocated fields through
the same call, or return native expressions when given `function_space`:

```python
apply_law(numerical_law, [first_field, shared_scalar], [output_field])
expressions = apply_law(symbolic_law, [first_operand, second_operand], function_space=space)
```

`SymbolicLaw.evaluate` also supplies native expressions for composing an implicit residual.
Its `LocalEvaluation` provides both the value and selected partial needed by that residual.
Use `apply_law` for explicit recorded scalar, field, or native expression outputs. A symbolic
residual can therefore use the same physical law directly without introducing an external
operator into its nonlinear solve. External-operator assembly inside nonlinear residuals needs
its own native solver support.

Downstream models own their physical formulas and domain limits. The shared scalar, field, and symbolic
backends own recording, replay, selected transposes, and native operator registration. Continuous joins follow their complete moving
switch action through `join`. `ordered_branch` selects intervals at exact numerical breakpoints.
The local law must establish continuity, finite branch actions, and agreement on tangent
perturbations. A branch selector alone is not an event-response contract.

`ufl_nonsmooth/primitives.py` supplies `clip`, `absolute`, and `oriented_flux` for native
expressions.
`maximum_values`, `absolute_values`, and `clip_values` supply numerical values with their selected
branch slopes. `maximum_directional` and `clip_directional` apply joint numerical perturbations,
including moving clipping bounds. `masked_pullback` applies selected slopes without evaluating an
unobserved singular contribution.

For a joint native directional action, provide every varying coefficient together:

```python
from ufl_nonsmooth.primitives import directional

response = directional(expression, [first, second], [first_direction, second_direction])
```

The result follows the supplied direction at max, min, and absolute-value ties. These actions
compose through supported continuous operations and continuously differentiable stages.
Transposing one chosen branch does not supply the full reverse relation.

`maximum_coderivative`, `minimum_coderivative`, `absolute_coderivative`, and `clip_coderivative`
return exact local scalar basic coderivatives with fixed thresholds and clipping bounds.
`kink_coderivative` supplies their shared finite two-sided scalar rule. A continuous scalar-input
law can declare `ScalarKinkLaw` when all its other parameters are fixed and its outputs have at
most two smooth incident branches with finite limiting derivatives. The shared reverse query
contracts output seeds with both branch jets before applying that rule:

```python
from ufl_nonsmooth.scalar import scalar_coderivative

reverse = scalar_coderivative(chart, 1.0, [1.0])
assert reverse.contains(1.0)
assert reverse.contains(2.0)
assert not reverse.contains(1.5)
```

This capability covers fixed-row piecewise-linear charts and qualified downstream laws.
It does not follow from an arbitrary law's directional action. `ScalarGraphResponse` stores a
union of intervals of finite scalar responses. Empty relations, unique values, disconnected
sets, and unbounded sets have distinct meanings. `value()` succeeds only for a unique finite
value. Unsupported operations raise errors instead of returning an empty relation.

`ufl_nonsmooth/piecewise.py` owns chart values, native expressions, and joint actions. Numerical
chart operands are the query coordinate alone, or that coordinate followed by every alternating
knot and ordinate. Native `expression` accepts the corresponding symbolic rows. Ordered knots
and a fixed row count define the supported domain. `coderivative_contains` batches fixed-row
scalar membership queries, while `joint_coderivative_contains` also includes knot and ordinate
covectors. Array queries are independent scalar queries and do not combine seeds across outputs
sharing the same table parameters.

`ufl_nonsmooth/tape.py` propagates joint native directions through a recorded run:

```python
from ufl_nonsmooth.tape import evaluate_directional

evaluate_directional(tape, controls, directions, outputs=[objective.block_variable])
response = objective.block_variable.tlm_value
```

Controls are native `Control` objects. Directions are matching fields or scalar values. The
optional outputs list selects their causal slice. Without it, the traversal visits the whole
tape. It resets tangent values and retains primal checkpoints, adjoints, and operation order.
An absent tangent on the requested slice means independence from the supplied controls.
An unvisited node has no computed response.

Assembly, interpolation, expression assignment, and regular implicit solves receive joint
actions. Other operations use their native tangent hooks. `ufl_nonsmooth/sampling.py` retains
active operands and admits explicitly sampled weak-form and scalar interpolation predicates
only on separated numerical branches. Weak-form predicates need degree-zero active fields and
explicit degree-zero quadrature. An active sampled event requires a model outside this toolkit.
These responses describe the finite quadrature or interpolation programme. A continuum moving
interface needs its own topology and interface terms.

`ImplicitSolver` in `ufl_nonsmooth/implicit.py` records the converged residual, boundary
conditions, and supplied primal iteration Jacobian. Derivatives use the exact state Jacobian
of that residual, independently of the iteration matrix used to reach its solution. Selected
native tangent and joint directional solves share the same response kernel, boundary handling,
structural-zero loads, and compiler options. The supplied iteration matrix remains a dependency
for forward replay.

The state-response equation must remain linear on the admitted branch. Native first-sensitivity
checks admit sampled state kinks whose complete first-order contribution cancels. All tied
scalar primitives are checked together with cellwise-constant state and primitive operands in
the explicit degree-zero programme. Nonzero sensitivities and nonlinear response equations need
an additional uniqueness contract. Parameter kinks can supply a joint forcing when the state
equation remains regular. Unsupported solve options are reported explicitly.

Native opaque operators declare separate selected and joint actions through
`NativeOperatorResponse`. The shared symbolic backend supplies these hooks for local laws.
An opaque operator with explicit fixed-topology geometry dependence also supplies
`NativeCoordinateResponse`. Coordinate actions retain native reference-cell fields, geometric
factors, normals, and measures. A selected coordinate transpose requires a jet linear in its
perturbation.

Native pyadjoint pullbacks follow the executed branch convention. They represent ordinary
physical derivatives on a persistent regular branch. Qualified local coderivative queries
provide complete reverse sets only for their documented restrictions. The toolkit does not assemble
an exact global coderivative by independently choosing primitive reverse values. The definitions
and scope are in [the mathematical scope](theory.md).
