# UFL Non-smooth

Native Unified Form Language (UFL) and Firedrake calculus for local piecewise-smooth laws.
A law declares its values, selected linearisation, and joint directional action once.
The shared application boundary retains native scalar, field, symbolic, and adjoint operations.

## Installation

Install [Firedrake](https://www.firedrakeproject.org/install.html) with its native PETSc and MPI
prerequisites, then install the package into that environment:

```bash
python -m pip install ufl-nonsmooth
```

Version 2026.2 targets Firedrake 2026.10.x, UFL 2026.2.x, pyadjoint 2026.10.x, and Python 3.11 or later.
The dependency bounds describe the supported release series. The tests exercise the public
contracts on the declared environment, including native shape actions and two-rank MPI responses.
Firedrake natively supports Ubuntu and ARM macOS. Windows users can use WSL.
The package contains Python source and is distributed as one platform-independent wheel.
Its compiled dependencies retain their own installation requirements.

## Use

```python
from ufl_nonsmooth.application import apply_law
from ufl_nonsmooth.piecewise import PiecewiseLinear
from ufl_nonsmooth.primitives import directional
from ufl_nonsmooth.scalar import scalar_coderivative

chart = PiecewiseLinear([(0.0, 0.0), (1.0, 2.0), (2.0, 3.0)])
value = apply_law(chart, [0.5])[0]
assert value == 1.0
reverse = scalar_coderivative(chart, 1.0, [2.0])
assert reverse.contains(2.0)
assert reverse.contains(4.0)
assert not reverse.contains(3.0)
```

For native expressions, call `directional(expression, coefficients, directions)` with all
varying operands together. For a recorded computation, use
`ufl_nonsmooth.tape.evaluate_directional`. Import each API from its defining module.

A selected linearisation at a kink has a linear transpose. The joint directional response may
be nonlinear. Qualified local coderivative queries return their actual reverse sets, which can
be disconnected. General graph composition, active history changes, topology changes,
rank changes, and degenerate implicit response equations require additional models outside
this toolkit. Unsupported responses raise an error.

## Documentation

- [Native API and examples](docs/api.md)
- [Mathematical scope](docs/theory.md)
- [Development and release](docs/development.md)

## Licence

Copyright 2026 Benjamin Coveler.
Licensed under the GNU Lesser General Public License, version 3 or later
(`LGPL-3.0-or-later`). See [COPYING.LESSER](COPYING.LESSER) and [COPYING](COPYING).
