# Development and release

Use a supported Firedrake environment. Install the development tools and the editable package:

```bash
python -m pip install -e ".[dev]"
python -m ruff format src tests
python -m ruff check --fix src tests
python -m ty check
python -m pytest -n auto --dist worksteal
mpiexec -n 2 python -m pytest
python -m build
python -m twine check --strict dist/*
```

Run the native tests and type checks locally in an existing Firedrake environment before
pushing. GitHub Actions runs Ruff and distribution checks on Linux and macOS.

The suite uses native Cartesian meshes and analytical laws. It does not require MARLIN,
a physical model, input files, or mesh assets from a downstream project. MARLIN retains its
own atmosphere, cone, wedge, thermal, and coupled integration regressions.

Test both the wheel and the source distribution in a supported environment. From a directory
outside this checkout, verify the installed `ufl_nonsmooth` path and run the packaged tests.
The wheel contains `py.typed` and the LGPL licence texts.

## PyPI trusted publishing

The GitHub repository is `SpeedyTurtle599/ufl-nonsmooth`. Configure a PyPI pending publisher with:

| Field | Value |
|---|---|
| PyPI project name | `ufl-nonsmooth` |
| Owner | `SpeedyTurtle599` |
| Repository name | `ufl-nonsmooth` |
| Workflow filename | `publish.yml` |
| Environment name | `pypi` |

Create the GitHub Actions environment `pypi` in the repository settings. The release workflow
builds and checks both distributions, verifies that the release tag matches the package version,
and publishes the checked artefacts through OpenID Connect. It requires no stored PyPI token.
Publish a GitHub release tagged `2026.2` or `v2026.2` for the first release.
For an existing release, run the `Publish` workflow manually and supply its tag. Both paths
build the tagged source and require its package version to match the release tag.

## Downstream development

Install this checkout into the downstream project's Firedrake environment before installing
that project. For sibling checkouts:

```bash
python -m pip install -e ../ufl-nonsmooth
python -m pip install -e .
```

Downstream metadata depends on the released version range. It contains no local filesystem URL
or bundled copy of the calculus implementation.
