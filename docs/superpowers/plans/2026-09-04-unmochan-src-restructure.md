# Unmochan Source-Layout Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the dual `unfoldlab`/`unmochan` package arrangement with one real `src/unmochan` package, a curated public API, one CLI identity, and consolidated documentation without changing scientific behavior.

**Architecture:** Move the existing implementation as one Git-traceable tree while retaining the `core`, `io`, `workflows`, `twist`, and `cli` boundaries. First make the source-layout cutover pass the unchanged scientific suite, then narrow the root API, verify built distributions and CI, and finally consolidate documentation and remove superseded history files.

**Tech Stack:** Python 3.10+, Hatchling, Typer, NumPy, pytest, Ruff, mypy, `build`, GitHub Actions, PowerShell-compatible local commands.

**Spec:** `docs/superpowers/specs/2026-09-04-unmochan-src-restructure-design.md`

## Global Constraints

- The migration is an intentional breaking change: do not create an `unfoldlab` package, import hook, alias module, warning shim, or console command.
- Keep Python support at `>=3.10`; do not upgrade dependencies or alter optional dependency groups.
- Do not change numerical algorithms, scientific conventions, exception semantics, array shapes, or calculation outputs.
- Preserve the `core`, `io`, `workflows`, `twist`, and `cli` package boundaries; do not split large modules in this plan.
- Use absolute `unmochan.*` imports; do not introduce relative imports as part of the move.
- Preserve the on-disk schema identifiers `unfoldlab.effective_band_structure` and `unfoldlab.run_manifest`; they identify version-1 data formats and are not import namespaces.
- Keep `graphify-out/` local and untracked; commit its durable conclusions only through `docs/architecture.md`.
- Do not delete or weaken scientific tests. Namespace-only test edits must retain their existing assertions.
- Run Windows pytest with `--basetemp .tmp-pytest -p no:cacheprovider` to avoid cache and temporary-directory lock failures.

---

## File Structure and Responsibility Map

**New implementation tree**

- `src/unmochan/__init__.py`: version plus the approved curated root API.
- `src/unmochan/core/`: the existing 34 core files, moved without algorithm changes.
- `src/unmochan/io/`: the existing 14 adapters and serializers, moved without format changes.
- `src/unmochan/workflows/`: the existing 6 orchestration files.
- `src/unmochan/twist/`: the existing 4 twist and layered-structure files.
- `src/unmochan/cli/`: the existing 12 CLI files, with only the legacy entry function removed.

**Contract tests**

- `tests/test_package_layout.py`: physical `src` layout and absence of the legacy import package.
- `tests/test_public_api.py`: exact root export contract and exclusion of specialized symbols.
- `tests/test_unmochan_branding.py`: sole package/CLI identity and help behavior.
- `tests/test_serialization.py`: stable legacy data-schema identifiers despite the Python rename.

**Build and documentation**

- `pyproject.toml`: sole package and sole console entry point.
- `.github/workflows/tests.yml`: `src`-aware type/compile gates and clean-wheel smoke checks.
- `.gitignore`: generated Graphify output.
- `docs/architecture.md`: durable Graphify-backed package map and dependency rules.
- `README.md`, `docs/testing.md`: current paths and development commands.
- `docs/roadmap.md`: the only future-facing project document.
- `docs/formal-model.md`, `docs/benchmarks.md`, `docs/examples.md`, `docs/projections.md`: maintained technical documentation with the new namespace.

---

### Task 1: Atomic Source-Tree and Namespace Cutover

**Files:**

- Create: `tests/test_package_layout.py`
- Delete: `unmochan/__init__.py`
- Delete: `unmochan/cli/__init__.py`
- Delete: `unmochan/cli/main.py`
- Move: `unfoldlab/__init__.py` → `src/unmochan/__init__.py`
- Move: `unfoldlab/cli/*.py` → `src/unmochan/cli/*.py` (12 files)
- Move: `unfoldlab/core/*.py` → `src/unmochan/core/*.py` (34 files)
- Move: `unfoldlab/io/*.py` → `src/unmochan/io/*.py` (14 files)
- Move: `unfoldlab/twist/*.py` → `src/unmochan/twist/*.py` (4 files)
- Move: `unfoldlab/workflows/*.py` → `src/unmochan/workflows/*.py` (6 files)
- Modify: all Python files under `src/unmochan/`, `tests/`, and `examples/` that contain `unfoldlab`
- Modify: `src/unmochan/cli/main.py`
- Modify: `src/unmochan/io/serialization.py`
- Modify: `tests/test_unmochan_branding.py`
- Modify: `tests/test_cli_structure.py`
- Modify: `tests/test_model_cli.py`
- Modify: `tests/test_serialization.py`
- Modify: `pyproject.toml`

**Interfaces:**

- Consumes: the existing `unfoldlab.*` module tree and its 934-test documented baseline.
- Produces: importable `unmochan.*` modules from `src/unmochan`, `main(argv: list[str] | None = None, *, prog_name: str = "unmochan") -> int`, and no `unfoldlab` Python package or CLI function.
- Preserves: `EBS_SCHEMA == "unfoldlab.effective_band_structure"` and run-manifest schema `"unfoldlab.run_manifest"`.

- [ ] **Step 1: Record the unchanged baseline before moving files**

Run:

```powershell
python -m pip install -e ".[dev,io,plot]"
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy unfoldlab unmochan
```

Expected: every command exits `0`; pytest should report the documented baseline of 934 passing tests. If the count or status differs, stop and preserve the exact output before changing the tree.

- [ ] **Step 2: Add the failing physical-layout and legacy-absence contract**

Create `tests/test_package_layout.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import unmochan


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_unmochan_is_loaded_from_the_src_tree() -> None:
    package_dir = Path(unmochan.__file__).resolve().parent

    assert package_dir == REPOSITORY_ROOT / "src" / "unmochan"


def test_legacy_unfoldlab_package_is_absent() -> None:
    assert not (REPOSITORY_ROOT / "unfoldlab").exists()
    assert importlib.util.find_spec("unfoldlab") is None
```

- [ ] **Step 3: Run the layout contract to verify the expected red state**

Run:

```powershell
python -m pytest tests/test_package_layout.py -v --basetemp .tmp-pytest -p no:cacheprovider
```

Expected: both tests fail because `unmochan` currently loads from the repository-root forwarding shell and `unfoldlab` still exists.

- [ ] **Step 4: Remove the forwarding shell and move the implementation with Git history**

Run:

```powershell
git rm -r -- unmochan
New-Item -ItemType Directory -Path src -Force | Out-Null
git mv unfoldlab src/unmochan
```

Expected: `git status --short` shows the three forwarding files deleted and all 71 implementation files as renames under `src/unmochan/`.

- [ ] **Step 5: Mechanically rewrite the Python namespace**

Run this scoped bulk rewrite only across implementation, tests, and examples:

```powershell
$rewritePaths = git grep -l "unfoldlab" -- "src/unmochan/**/*.py" "tests/**/*.py" "examples/**/*.py"
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($path in $rewritePaths) {
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
    $updated = $text.Replace("unfoldlab", "unmochan")
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $path), $updated, $utf8NoBom)
}
```

Then restore the two stable data-format identifiers in `src/unmochan/io/serialization.py`:

```python
# These are stable version-1 data-format identifiers, not Python import paths.
EBS_SCHEMA = "unfoldlab.effective_band_structure"
RUN_MANIFEST_SCHEMA = "unfoldlab.run_manifest"
```

Use `RUN_MANIFEST_SCHEMA` in `RunManifest.to_dict()`:

```python
return {
    "schema": RUN_MANIFEST_SCHEMA,
    "schema_version": MANIFEST_SCHEMA_VERSION,
    # existing fields remain unchanged
}
```

Do not modify the remaining serialization payload keys or versions.

- [ ] **Step 6: Add a regression test pinning the intentionally stable schemas**

In `tests/test_serialization.py`, import `EBS_SCHEMA` and `RUN_MANIFEST_SCHEMA`, then add:

```python
def test_package_rename_preserves_version_one_schema_identifiers():
    manifest = build_run_manifest(code="vasp", weight_mode="plane-wave projection")

    assert EBS_SCHEMA == "unfoldlab.effective_band_structure"
    assert RUN_MANIFEST_SCHEMA == "unfoldlab.run_manifest"
    assert manifest.to_dict()["schema"] == RUN_MANIFEST_SCHEMA
```

In `tests/test_model_cli.py`, import `EBS_SCHEMA` from `unmochan.io.serialization` and change the existing JSON assertion to:

```python
assert stored["schema"] == EBS_SCHEMA
```

- [ ] **Step 7: Remove the legacy CLI entry function**

In `src/unmochan/cli/main.py`:

```python
"""Entry point and command registration for the `unmochan` executable."""

from __future__ import annotations

import typer

from unmochan.cli import analysis as _analysis  # noqa: F401  (registers commands)
from unmochan.cli import backends as _backends  # noqa: F401  (registers commands)
from unmochan.cli import commands as _commands  # noqa: F401  (registers commands)
from unmochan.cli import diagnostics as _diagnostics  # noqa: F401  (registers commands)
from unmochan.cli import symmetry as _symmetry  # noqa: F401  (registers commands)
from unmochan.cli.app import app, qe_app, vasp_app
from unmochan.cli.common import click_exception_classes, console

__all__ = ["app", "main", "qe_app", "vasp_app"]


def main(argv: list[str] | None = None, *, prog_name: str = "unmochan") -> int:
    command = typer.main.get_command(app)
    click_exceptions = click_exception_classes("ClickException")
    exit_exceptions = click_exception_classes("Exit")
    abort_exceptions = click_exception_classes("Abort")
    try:
        command(args=argv, prog_name=prog_name, standalone_mode=False)
    except click_exceptions as exc:
        exc.show()
        return int(exc.exit_code)
    except exit_exceptions as exc:
        return int(exc.exit_code)
    except abort_exceptions:
        console.print("Aborted.")
        return 1
    except (ValueError, OSError) as exc:
        console.print(f"Error: {exc}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

This removes `unfoldlab_main()` while preserving `main()`'s exception behavior.

- [ ] **Step 8: Rewrite the branding and CLI structure contracts**

Replace `tests/test_unmochan_branding.py` with:

```python
from typer.testing import CliRunner

import unmochan
from unmochan.cli.main import app, main


runner = CliRunner()


def test_unmochan_import_exposes_the_project_version() -> None:
    assert unmochan.__version__ == "0.1.0"


def test_primary_cli_name_is_unmochan() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.output
    assert "Usage: unmochan" in result.output
    assert "UNfolding MOmentum-space Crystal Hamiltonian ANalysis" in result.output


def test_console_entrypoint_program_name_is_unmochan() -> None:
    assert main(["--help"]) == 0
```

In `tests/test_cli_structure.py`, update every module name to `unmochan.cli.*`, import `unmochan.cli`, and derive the length-check directory from `unmochan.cli.__path__`. Do not alter `TOP_LEVEL_COMMANDS`, `GROUP_COMMANDS`, or the 1000-line limit.

- [ ] **Step 9: Update package discovery and the sole console script**

Replace the relevant `pyproject.toml` sections with:

```toml
[project.scripts]
unmochan = "unmochan.cli.main:main"

[tool.hatch.build.targets.wheel]
packages = ["src/unmochan"]
```

Leave the project name, version, Python floor, dependencies, optional dependencies, pytest markers, Ruff configuration, and mypy configuration unchanged.

- [ ] **Step 10: Install the moved package and run focused migration tests**

Run:

```powershell
python -m pip install -e ".[dev,io,plot]"
python -m pytest tests/test_package_layout.py tests/test_unmochan_branding.py tests/test_cli_structure.py tests/test_serialization.py tests/test_model_cli.py -v --basetemp .tmp-pytest -p no:cacheprovider
```

Expected: all focused tests pass, including the two new layout tests and one new schema-stability test.

- [ ] **Step 11: Verify the mechanical cutover did not alter behavior**

Run:

```powershell
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy src/unmochan
```

Expected: pytest reports the baseline test total plus three new tests, and Ruff and mypy exit `0`.

Run the scoped legacy-name audit:

```powershell
rg -n "unfoldlab" src tests examples
```

Expected: matches are limited to the two stable schema literals in `src/unmochan/io/serialization.py` and the schema assertions in `tests/test_serialization.py`; there are no import paths, CLI names, or source-tree paths.

- [ ] **Step 12: Commit the atomic namespace migration**

```powershell
git add pyproject.toml src tests examples
git commit -m "refactor: move implementation to src unmochan package"
```

---

### Task 2: Curate the Root Public API

**Files:**

- Create: `tests/test_public_api.py`
- Modify: `src/unmochan/__init__.py`
- Modify: `tests/test_augmentation.py`
- Modify: `tests/test_degeneracy.py`
- Modify: `tests/test_shared_weights.py`

**Interfaces:**

- Consumes: concrete types and workflows from their existing modules under `unmochan.*`.
- Produces: the exact 15-name `unmochan.__all__` contract listed below.
- Preserves: specialized imports through `unmochan.core`, `unmochan.io`, `unmochan.workflows`, and `unmochan.twist`.

- [ ] **Step 1: Add the failing exact-export contract**

Create `tests/test_public_api.py`:

```python
from __future__ import annotations

import unmochan


EXPECTED_PUBLIC_API = {
    "EffectiveBandStructure",
    "ProjectionSelector",
    "QEUnfoldResult",
    "Structure",
    "TransformationMatrix",
    "TwistedUnfoldingProblem",
    "UnfoldingProblem",
    "VaspUnfoldResult",
    "__version__",
    "build_qe_effective_band_structure",
    "build_vasp_effective_band_structure",
    "compute_backend_weights",
    "unfold_backend_bands",
    "unfold_qe_bands",
    "unfold_vasp_bands",
}


def test_root_public_api_is_deliberately_curated() -> None:
    assert set(unmochan.__all__) == EXPECTED_PUBLIC_API
    assert all(hasattr(unmochan, name) for name in EXPECTED_PUBLIC_API)


def test_specialized_algorithms_are_not_root_exports() -> None:
    assert not hasattr(unmochan, "diagnose_augmentation")
    assert not hasattr(unmochan, "shared_weights_from_coefficients")
    assert not hasattr(unmochan, "degeneracy_report")
```

- [ ] **Step 2: Run the API contract to verify the expected red state**

Run:

```powershell
python -m pytest tests/test_public_api.py -v --basetemp .tmp-pytest -p no:cacheprovider
```

Expected: both tests fail because the moved root `__init__.py` still exports the historical broad surface.

- [ ] **Step 3: Replace the root initializer with the approved API**

Replace `src/unmochan/__init__.py` with:

```python
"""Public API for Unmochan."""

from unmochan.core.projections import ProjectionSelector
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.structures import Structure
from unmochan.core.transformations import TransformationMatrix
from unmochan.twist.problem import TwistedUnfoldingProblem
from unmochan.workflows.backend import compute_backend_weights, unfold_backend_bands
from unmochan.workflows.problem import UnfoldingProblem
from unmochan.workflows.qe import (
    QEUnfoldResult,
    build_qe_effective_band_structure,
    unfold_qe_bands,
)
from unmochan.workflows.vasp import (
    VaspUnfoldResult,
    build_vasp_effective_band_structure,
    unfold_vasp_bands,
)

__all__ = [
    "EffectiveBandStructure",
    "ProjectionSelector",
    "QEUnfoldResult",
    "Structure",
    "TransformationMatrix",
    "TwistedUnfoldingProblem",
    "UnfoldingProblem",
    "VaspUnfoldResult",
    "__version__",
    "build_qe_effective_band_structure",
    "build_vasp_effective_band_structure",
    "compute_backend_weights",
    "unfold_backend_bands",
    "unfold_qe_bands",
    "unfold_vasp_bands",
]

__version__ = "0.1.0"
```

- [ ] **Step 4: Move specialized test imports to their owning APIs**

Make only these contract changes:

- `tests/test_degeneracy.py`: import all degeneracy symbols from `unmochan.core.degeneracy` rather than the package root.
- `tests/test_augmentation.py`: rename `test_the_module_is_exported` to `test_the_module_is_exported_from_core`, import `unmochan.core as core`, and assert `core.diagnose_augmentation` and `core.augmented_weights` identity.
- `tests/test_shared_weights.py`: rename the root-export test to `test_the_batched_kernel_is_exported_from_core`, import `unmochan.core as core`, and assert the same three symbol identities against `core`.

Do not change the numerical assertions in these files.

- [ ] **Step 5: Run focused API and specialized-module tests**

Run:

```powershell
python -m pytest tests/test_public_api.py tests/test_augmentation.py tests/test_degeneracy.py tests/test_shared_weights.py -v --basetemp .tmp-pytest -p no:cacheprovider
```

Expected: all tests pass, including both new public-API tests.

- [ ] **Step 6: Verify no test still expects a broad root API**

Run:

```powershell
rg -n "from unmochan import|import unmochan$" tests --glob "*.py"
```

Expected: direct root use is limited to `tests/test_package_layout.py`, `tests/test_public_api.py`, and the version check in `tests/test_unmochan_branding.py`; specialized tests import their owning module or subpackage.

- [ ] **Step 7: Commit the curated API**

```powershell
git add src/unmochan/__init__.py tests/test_public_api.py tests/test_augmentation.py tests/test_degeneracy.py tests/test_shared_weights.py
git commit -m "refactor: curate the Unmochan public API"
```

---

### Task 3: Enforce Distribution and CI Contracts

**Files:**

- Modify: `.github/workflows/tests.yml`
- Verify: `pyproject.toml`
- Verify: built `dist/*.whl` and `dist/*.tar.gz` artifacts (do not commit)

**Interfaces:**

- Consumes: the `src/unmochan` package and `unmochan.cli.main:main` entry point from Task 1.
- Produces: CI gates that type-check and compile the `src` package, build a wheel, prove the wheel contains `unmochan` but no `unfoldlab`, and smoke-test only the `unmochan` executable.

- [ ] **Step 1: Update source paths in the matrix job**

In `.github/workflows/tests.yml`, replace the type and compile steps with:

```yaml
      - name: Mypy
        run: python -m mypy src/unmochan

      - name: Compile
        run: python -m compileall src/unmochan tests examples
```

Keep the Python/OS matrix, install command, pytest selection, Ruff gate, dependency check, and optional real-I/O job unchanged.

- [ ] **Step 2: Strengthen the package-smoke job**

Replace its build/install commands with:

```yaml
      - name: Build and inspect distributions
        run: |
          python -m pip install build
          python -m build
          python -c "from pathlib import Path; from zipfile import ZipFile; wheel = next(Path('dist').glob('*.whl')); names = ZipFile(wheel).namelist(); assert any(name.startswith('unmochan/') for name in names); assert not any(name.startswith('unfoldlab/') for name in names)"

      - name: Install wheel and smoke-test CLI
        run: |
          python -m venv .venv-smoke
          . .venv-smoke/bin/activate
          python -m pip install dist/*.whl
          python -c "import unmochan; assert unmochan.__version__ == '0.1.0'"
          python -c "import importlib.util; assert importlib.util.find_spec('unfoldlab') is None"
          unmochan --help
          test ! -e .venv-smoke/bin/unfoldlab
```

- [ ] **Step 3: Run local static and compile equivalents**

Run:

```powershell
python -m ruff check .
python -m mypy src/unmochan
python -m compileall -q src/unmochan tests examples
```

Expected: all commands exit `0`.

- [ ] **Step 4: Build and inspect the wheel locally**

Run:

```powershell
python -m build
python -c "from pathlib import Path; from zipfile import ZipFile; wheel=next(Path('dist').glob('*.whl')); names=ZipFile(wheel).namelist(); assert any(name.startswith('unmochan/') for name in names); assert not any(name.startswith('unfoldlab/') for name in names); print(wheel)"
```

Expected: one wheel path is printed after both package-content assertions pass.

- [ ] **Step 5: Install the wheel into an isolated environment**

Run:

```powershell
python -m venv tmp-pytest\wheel-smoke
$wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
& .\tmp-pytest\wheel-smoke\Scripts\python.exe -m pip install $wheel
& .\tmp-pytest\wheel-smoke\Scripts\python.exe -c "import unmochan; assert unmochan.__version__ == '0.1.0'"
& .\tmp-pytest\wheel-smoke\Scripts\python.exe -c "import importlib.util; assert importlib.util.find_spec('unfoldlab') is None"
& .\tmp-pytest\wheel-smoke\Scripts\unmochan.exe --help
if (Test-Path -LiteralPath '.\tmp-pytest\wheel-smoke\Scripts\unfoldlab.exe') { throw 'legacy unfoldlab executable was installed' }
```

Expected: installation and `unmochan --help` exit `0`, both Python assertions pass, and the final executable check does not throw.

- [ ] **Step 6: Commit the CI contract**

```powershell
git add .github/workflows/tests.yml
git commit -m "ci: verify the src package distribution"
```

---

### Task 4: Document the New Architecture and Developer Workflow

**Files:**

- Create: `docs/architecture.md`
- Modify: `README.md`
- Modify: `docs/testing.md`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: the final package tree and CI commands from Tasks 1–3 plus the Graphify evidence recorded in the spec.
- Produces: one durable architecture map, current installation/development instructions, and an ignored local graph-output directory.

- [ ] **Step 1: Write the durable architecture map**

Create `docs/architecture.md` with this content:

````markdown
# Unmochan architecture

Unmochan is a material-agnostic Python toolkit for band unfolding and related
supercell analysis. The installable source lives under `src/unmochan`.

## Package map

| Package | Responsibility | May depend on |
| --- | --- | --- |
| `unmochan.core` | Numerical kernels, physical data models, transformations, symmetry, projections, and spectral analysis | NumPy and other core modules |
| `unmochan.io` | VASP, Quantum ESPRESSO, Wannier90, force-constant, serialization, and plotting adapters | `unmochan.core` |
| `unmochan.workflows` | Backend-neutral and backend-specific orchestration | `unmochan.core`, `unmochan.io` |
| `unmochan.twist` | Commensurate twist, layer geometry, and moire workflows | `unmochan.core`, `unmochan.io`, `unmochan.workflows` |
| `unmochan.cli` | Typer applications, parsing, command registration, and guided flows | All lower layers |

## Dependency direction

```text
cli ───────► workflows ───────► io ───────► core
 │               │              │            ▲
 └───────────────┴──────────────┴────────────┘

twist ─────► workflows / io / core
```

`core` must not import `cli` or `workflows`. `io` must not import CLI modules.
Workflows coordinate core algorithms and I/O adapters; they do not own
backend-specific parsing.

## Central abstractions

The repository knowledge graph identifies `EffectiveBandStructure`,
`Structure`, and `TransformationMatrix` as the main cross-community objects.
They are deliberately small data/contract boundaries shared by calculations,
adapters, serialization, plotting, and workflows. Changes to any of these
types require full-suite and backend-parity verification.

## Public imports

The package root exposes only the primary models and workflows documented in
`unmochan.__all__`. Specialized algorithms belong to the subpackage or module
that owns them; for example, use `unmochan.core.degeneracy` for degeneracy
analysis and `unmochan.io.vasp_wfc` for WAVECAR operations.

## Command registration

`unmochan.cli.main` imports the command modules that register commands on the
Typer application. The deferred guided-mode import in `unmochan.cli.app`
prevents the registration cycle pinned by `tests/test_cli_structure.py`.

## Later decomposition

The source-layout migration deliberately leaves large modules intact. The
highest-priority candidates for later responsibility-based splits are
`cli/analysis.py`, `core/spacegroup.py`, `core/tight_binding.py`,
`io/vasp_wfc.py`, `core/lcao.py`, `cli/commands.py`, `core/unfolding.py`,
`io/qe_wfc.py`, `io/qe.py`, and `cli/diagnostics.py`.

## Graph evidence

The 2026-09-02 Graphify pass covered 166 supported files and produced 3,328
nodes, 10,228 edges, and 115 communities. It found no import cycles. Generated
`graphify-out` artifacts are local inspection aids; this document is the
version-controlled architectural summary.
````

- [ ] **Step 2: Rewrite the README repository map and identity**

Replace the repository table with:

```markdown
| Path | Contents |
| --- | --- |
| `src/unmochan/core/` | unfolding kernels, physical data models, symmetry, spectral analysis, tight binding, LCAO, phonons, and disorder ensembles |
| `src/unmochan/io/` | VASP, Quantum ESPRESSO, Wannier90, serialization, and plotting adapters |
| `src/unmochan/twist/` | commensurate twist angles, moire cells, and stack diagnostics |
| `src/unmochan/workflows/` | backend-neutral and backend-specific workflows |
| `src/unmochan/cli/` | implementation of the `unmochan` command line |
| `tests/` | synthetic and contract test suite |
| `docs/` | architecture, theory, benchmarks, examples, testing, projections, and roadmap |
| `examples/` | runnable toy models and benchmarks that need no DFT run |
```

Replace the historical compatibility paragraph with:

```markdown
## Project

Unmochan is the active package and command name. The implementation is
developed and distributed solely from `src/unmochan`.

Thanks to Ritam Chakraborty for contributions to the project direction and
implementation, and to Prajwal Souza for the initial codebase from which the
project developed.
```

Replace the development-gate block with:

```bash
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy src/unmochan
python -m compileall -q src/unmochan tests examples
```

Link `docs/architecture.md`, `docs/testing.md`, and `docs/roadmap.md` beneath the gate. Remove references to the deleted progress and findings logs.

- [ ] **Step 3: Update the testing guide for the src package**

In `docs/testing.md`, make the default gate exactly:

```bash
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy src/unmochan
python -m compileall -q src/unmochan tests examples
```

Update the packaging smoke section to require:

```bash
python -m build
python -m venv .venv-smoke
. .venv-smoke/bin/activate
python -m pip install dist/*.whl
python -c "import unmochan"
unmochan --help
```

State directly beneath the block that a release wheel must contain no `unfoldlab` package or console script, while the version-1 serialized schema identifiers remain stable.

- [ ] **Step 4: Ignore generated Graphify output**

Append this entry to `.gitignore`:

```gitignore
graphify-out/
```

- [ ] **Step 5: Verify documentation paths and ignored output**

Run:

```powershell
rg -n "unfoldlab/|mypy unfoldlab|compileall unfoldlab|legacy command|compatibility alias" README.md docs/architecture.md docs/testing.md
git check-ignore graphify-out/graph.json
```

Expected: the text search returns no matches and `git check-ignore` prints `graphify-out/graph.json`.

- [ ] **Step 6: Commit the architecture documentation**

```powershell
git add .gitignore README.md docs/architecture.md docs/testing.md
git commit -m "docs: describe the Unmochan src architecture"
```

---

### Task 5: Consolidate Durable Documentation and Remove History Logs

**Files:**

- Modify: `docs/roadmap.md`
- Modify: `docs/formal-model.md`
- Modify: `docs/benchmarks.md`
- Modify: `docs/examples.md`
- Modify: `docs/projections.md`
- Delete: `docs/findings.md`
- Delete: `docs/progress.md`
- Delete: `docs/future-implementation.md`

**Interfaces:**

- Consumes: completed/current distinctions in the three superseded documents and the new namespace from Task 1.
- Produces: one current roadmap and maintained technical documentation using `unmochan.*` paths exclusively.

- [ ] **Step 1: Replace the roadmap with current, non-duplicated priorities**

Rewrite `docs/roadmap.md` around these exact sections and open outcomes:

```markdown
# Unmochan roadmap

The implemented scientific surface is documented by the test suite,
`formal-model.md`, and `benchmarks.md`. This roadmap lists only work that is
still open; completed pass logs are intentionally not retained here.

## 1. Real-backend validation

- Build a small redistributable fixture corpus containing equivalent VASP and
  Quantum ESPRESSO toy systems.
- Validate WAVECAR parsing for multiple VASP versions, precision modes,
  collinear spin, and SOC/noncollinear calculations.
- Validate Quantum ESPRESSO HDF5 and Fortran-binary readers against real
  `pw.x` outputs in collected and distributed layouts.
- Publish a parity report comparing effective band structures from equivalent
  VASP and Quantum ESPRESSO calculations.

## 2. Production I/O hardening

- Resolve Quantum ESPRESSO save directories from `prefix`, `outdir`, XML
  metadata, and collected/distributed wavefunction layouts.
- Expand POSCAR and Quantum ESPRESSO structure parsing across common input
  variants.
- Add explicit tested errors for unsupported wavefunction storage conventions.
- Compare plane-wave and PROCAR/projector-assisted approximations and document
  when their weights are physically comparable.

## 3. Examples and release readiness

- Provide one redistributable VASP tutorial, one Quantum ESPRESSO tutorial,
  and one backend-parity tutorial.
- Separate production-supported, experimental, and unsupported behavior in
  release notes.
- Require the full gate in `testing.md` before a version tag.

## 4. Later scientific extensions

- Generic HDF5 backends beyond Quantum ESPRESSO wavefunction layouts.
- Wannier90 `_centres.xyz` and disentanglement metadata.
- Exciton, electron-phonon, superconducting-gap, and ARPES-oriented workflows.
- Textbook-level theory chapters and expanded closed-form benchmarks.

## 5. Later code organization

Split the large modules listed in `architecture.md` only through focused,
behavior-preserving plans with direct contract tests.
```

- [ ] **Step 2: Rewrite namespace references in maintained technical documents**

Run the scoped rewrite:

```powershell
$docs = @(
    'docs\formal-model.md',
    'docs\benchmarks.md',
    'docs\examples.md',
    'docs\projections.md'
)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
foreach ($path in $docs) {
    $text = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
    $updated = $text.Replace('unfoldlab', 'unmochan').Replace('UnfoldLab', 'Unmochan')
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $path), $updated, $utf8NoBom)
}
```

Review the diff to ensure this changes only Python/CLI names and prose branding, not Lean theorem identifiers. If a Lean namespace is actually named `UnfoldLab`, retain that exact theorem namespace and explain it as a stable formal-model identifier.

- [ ] **Step 3: Confirm durable material is already represented before deletion**

Use the following checklist against the maintained files:

```text
findings mathematical invariants      -> docs/formal-model.md
findings closed-form reference cases  -> docs/benchmarks.md
findings verification policy          -> docs/testing.md
findings package/dependency structure -> docs/architecture.md
future real-backend work              -> docs/roadmap.md section 1
future I/O hardening                   -> docs/roadmap.md section 2
future release work                    -> docs/roadmap.md section 3
future advanced physics               -> docs/roadmap.md section 4
```

Verify each destination section exists:

```powershell
rg -n "^## " docs/formal-model.md docs/benchmarks.md docs/testing.md docs/architecture.md docs/roadmap.md
```

Expected: every destination named in the checklist has at least one matching section; do not delete a source document if its destination is missing.

- [ ] **Step 4: Delete the superseded history/status documents**

Run:

```powershell
git rm -- docs/findings.md docs/progress.md docs/future-implementation.md
```

- [ ] **Step 5: Verify maintained documentation has no stale links or executable names**

Run:

```powershell
rg -n "docs/(findings|progress|future-implementation)\.md|unfoldlab (vasp|qe|model|unfold|weights|norms|strain|dispersion|shadow|dos|mesh|layers)|unfoldlab\.core|unfoldlab\.io|unfoldlab\.workflows|unfoldlab\.twist|unfoldlab\.cli" README.md docs --glob "!superpowers/**"
```

Expected: no matches. Stable `UnfoldLab.*` Lean theorem namespaces are allowed only in `docs/formal-model.md`; stable lowercase serialization schema identifiers do not belong in these documentation files except the explicit note in `docs/testing.md`.

- [ ] **Step 6: Commit the documentation consolidation**

```powershell
git add docs README.md
git commit -m "docs: consolidate project guidance"
```

---

### Task 6: Final Repository and Distribution Verification

**Files:**

- Verify: all tracked changes since `main`
- Verify: `src/unmochan/`, `tests/`, `examples/`, `.github/workflows/tests.yml`, `pyproject.toml`, and maintained documentation
- Do not create or commit generated test, build, environment, or Graphify artifacts.

**Interfaces:**

- Consumes: the completed commits from Tasks 1–5.
- Produces: evidence that the clean cutover satisfies every acceptance criterion without numerical changes.

- [ ] **Step 1: Verify repository shape and legacy-package absence**

Run:

```powershell
git ls-files src/unmochan
git ls-files unfoldlab unmochan
git status --short
```

Expected: the first command lists the 71 moved implementation files; the second prints nothing; status contains no generated `graphify-out`, `dist`, build, environment, cache, or temporary files.

- [ ] **Step 2: Verify stale-name allowlisting**

Run:

```powershell
rg -n "unfoldlab" src tests examples pyproject.toml .github README.md docs --glob "!superpowers/**"
```

Expected: lowercase matches are limited to the stable schema constants and their explicit schema-stability tests/documentation note. There are no imports, source paths, console commands, or compatibility claims.

- [ ] **Step 3: Run the complete Python gate**

Run:

```powershell
python -m pip install -e ".[dev,io,plot]"
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy src/unmochan
python -m compileall -q src/unmochan tests examples
python -m pip check
```

Expected: every command exits `0`; pytest reports the recorded baseline total plus the five tests added in Tasks 1 and 2.

- [ ] **Step 4: Rebuild distributions from the verified tree**

Run:

```powershell
python -m build
python -c "from pathlib import Path; from zipfile import ZipFile; wheel=next(Path('dist').glob('*.whl')); names=ZipFile(wheel).namelist(); assert any(name.startswith('unmochan/') for name in names); assert not any(name.startswith('unfoldlab/') for name in names); print(f'{wheel}: {len(names)} entries')"
```

Expected: build exits `0`; the wheel summary prints only after both content assertions pass.

- [ ] **Step 5: Repeat the isolated wheel smoke test**

Run:

```powershell
python -m venv tmp-pytest\final-wheel-smoke
$wheel = (Get-ChildItem dist\*.whl | Select-Object -First 1).FullName
& .\tmp-pytest\final-wheel-smoke\Scripts\python.exe -m pip install $wheel
& .\tmp-pytest\final-wheel-smoke\Scripts\python.exe -c "import unmochan; assert set(unmochan.__all__) == {'EffectiveBandStructure','ProjectionSelector','QEUnfoldResult','Structure','TransformationMatrix','TwistedUnfoldingProblem','UnfoldingProblem','VaspUnfoldResult','__version__','build_qe_effective_band_structure','build_vasp_effective_band_structure','compute_backend_weights','unfold_backend_bands','unfold_qe_bands','unfold_vasp_bands'}"
& .\tmp-pytest\final-wheel-smoke\Scripts\python.exe -c "import importlib.util; assert importlib.util.find_spec('unfoldlab') is None"
& .\tmp-pytest\final-wheel-smoke\Scripts\unmochan.exe --help
if (Test-Path -LiteralPath '.\tmp-pytest\final-wheel-smoke\Scripts\unfoldlab.exe') { throw 'legacy unfoldlab executable was installed' }
```

Expected: all commands exit `0` and the legacy executable check does not throw.

- [ ] **Step 6: Review the diff for move purity and intended deletions**

Run:

```powershell
git diff --find-renames=90% --summary main...HEAD
git diff --check main...HEAD
git log --oneline --decorate main..HEAD
```

Expected: Git recognizes the implementation as renames plus namespace edits; deleted source files are limited to the forwarding shell and the old package locations represented by renames; deleted documents are exactly `findings.md`, `progress.md`, and `future-implementation.md`; `git diff --check` prints nothing.

- [ ] **Step 7: Record the final verification in the task handoff**

Report the exact pytest count, Ruff/mypy/build results, wheel filename, isolated-install results, commit list, deleted files, and any intentionally retained `unfoldlab` schema literals. Do not create a success commit when verification itself changes no tracked files.
