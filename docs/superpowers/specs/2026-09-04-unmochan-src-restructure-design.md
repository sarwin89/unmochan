# Unmochan Source-Layout Restructure Design

## Objective

Replace the repository's inverted dual-package arrangement with one real
`unmochan` implementation under a standard `src` layout. The migration is an
intentional breaking change: the historical `unfoldlab` Python package and
command will be removed rather than deprecated.

This phase changes package layout, imports, packaging metadata, public exports,
tests, examples, and documentation. It does not change numerical algorithms or
split large implementation modules.

## Current-State Evidence

The Graphify inventory covered 166 supported files and about 195,639 words. It
produced 3,328 nodes, 10,228 edges, and 115 communities with no detected import
cycles. The most connected abstractions were `EffectiveBandStructure`,
`Structure`, and `TransformationMatrix`, which confirms that the present
`core`, `io`, `workflows`, `twist`, and `cli` boundaries are meaningful enough
to retain during the namespace migration.

The current layout has two contradictory identities:

- `unfoldlab/` contains the complete implementation and almost all imports.
- `unmochan/` contains three forwarding files that re-export or delegate to
  `unfoldlab`.
- The distribution is named `unmochan`, but Hatch packages both directories.
- The `unmochan` console script enters `unfoldlab.cli.main`, while a second
  `unfoldlab` script preserves the historical command.
- Tests overwhelmingly exercise concrete `unfoldlab.*` paths.

No byte-identical tracked source files were found. Redundancy is structural:
the forwarding package, duplicate command identity, oversized export surfaces,
and overlapping project-history documents.

## Decisions

1. Use a standard `src/unmochan/` package layout.
2. Make a clean breaking cutover with no `unfoldlab` import compatibility.
3. Remove the `unfoldlab` console command.
4. Preserve the current subpackage boundaries during phase one.
5. Curate the root API instead of reproducing the current broad flat API.
6. Perform the namespace migration before splitting oversized modules.
7. Consolidate durable documentation and delete superseded history/status
   documents rather than archiving them.
8. Keep generated Graphify outputs local and out of version control; preserve
   their durable architectural conclusions in `docs/architecture.md`.

## Target Layout

```text
src/
└── unmochan/
    ├── __init__.py
    ├── core/
    ├── io/
    ├── workflows/
    ├── twist/
    └── cli/

tests/
examples/
docs/
pyproject.toml
README.md
```

The existing implementation files move to the corresponding directory under
`src/unmochan/`. Their responsibilities remain unchanged in this phase:

- `core`: numerical kernels, physical data models, transformations, symmetry,
  projections, spectral analysis, and model Hamiltonians.
- `io`: VASP, Quantum ESPRESSO, Wannier90, force-constant, serialization, and
  plotting adapters.
- `workflows`: backend-neutral and backend-specific orchestration.
- `twist`: commensurate twist, layer geometry, and moire workflows.
- `cli`: Typer applications, command registration, parsing, and guided flows.

Internal imports use absolute `unmochan.*` paths. Relative imports will not be
introduced as part of the mechanical migration because doing so would combine
two independent style changes.

## Public API

`src/unmochan/__init__.py` will expose a deliberately small root API. The
approved root exports are:

```python
__version__

Structure
TransformationMatrix
EffectiveBandStructure
ProjectionSelector
UnfoldingProblem
TwistedUnfoldingProblem
QEUnfoldResult
VaspUnfoldResult

compute_backend_weights
unfold_backend_bands
build_qe_effective_band_structure
unfold_qe_bands
build_vasp_effective_band_structure
unfold_vasp_bands
```

Specialized algorithms remain available from their owning subpackages and
modules, such as `unmochan.core`, `unmochan.io`, `unmochan.workflows`, and
`unmochan.twist`. The migration will not duplicate symbols solely to preserve
the former root export list.

Subpackage `__init__.py` files remain explicit API boundaries. They may retain
their current cohesive re-exports, updated to the new namespace, but must not
import from the package root or create circular registration behavior. CLI
command registration remains confined to `unmochan.cli.main` and its command
modules.

## Packaging and Commands

`pyproject.toml` will be updated so Hatch builds only `src/unmochan`. The sole
console script will be:

```toml
[project.scripts]
unmochan = "unmochan.cli.main:main"
```

The project remains named `unmochan`. Versioning, runtime dependencies,
optional dependency groups, Python version support, and test markers remain
unchanged during this phase.

The built wheel and source distribution must contain no `unfoldlab` package,
module path, or console entry point.

## Migration Sequence

1. Record a full test-suite baseline on the unchanged branch.
2. Remove the forwarding-only `unmochan/` shell.
3. Move the implementation from `unfoldlab/` to `src/unmochan/` with Git-aware
   moves so file history remains traceable.
4. Rewrite imports and fully qualified documentation references from
   `unfoldlab` to `unmochan` across source, tests, examples, and maintained
   documentation.
5. Replace the root export surface with the curated API.
6. Update package discovery, console entry points, Ruff/mypy paths, CI smoke
   tests, and developer commands for the `src` layout.
7. Consolidate documentation and remove superseded files.
8. Run focused import and CLI tests, static checks, the complete suite, build
   verification, and clean-environment installation smoke tests.

Each implementation task must leave an independently testable state. Algorithm
changes and large-module splits are excluded even if convenient opportunities
appear during the move.

## Documentation Consolidation

Create `docs/architecture.md` as the durable repository map. It will describe
the five package boundaries, central abstractions, allowed dependency
directions, and the phase-two module-splitting candidates identified by the
graph.

Before deletion, extract still-current content as follows:

- Move mathematical invariants and correctness arguments from `findings.md`
  into `formal-model.md` when formal, `testing.md` when they define a gate, or
  `benchmarks.md` when they define a reference case.
- Move only genuinely pending, current work from `progress.md` and
  `future-implementation.md` into `roadmap.md`.
- Remove dated pass logs, completed-task narration, superseded plans, and
  duplicated verification transcripts.

After extraction, delete:

```text
docs/findings.md
docs/progress.md
docs/future-implementation.md
```

`README.md`, examples, command snippets, docstrings, and maintained documents
must use the `unmochan` name and `src/unmochan` paths exclusively.

`graphify-out/` will be added to `.gitignore`. The generated HTML and JSON
remain useful local inspection artifacts but are not source-of-truth files.

## Dependency Rules

The migration preserves these intended directions:

```text
cli ───────► workflows ───────► io ───────► core
 │               │              │            ▲
 └───────────────┴──────────────┴────────────┘

twist ─────► workflows / io / core
```

`core` must not import `cli` or `workflows`. `io` may depend on core data models
and algorithms but must not depend on CLI modules. `workflows` may coordinate
core and I/O. CLI modules may depend on every lower layer. Existing localized
lazy imports used to avoid optional dependency or registration cycles may
remain, but the namespace migration must not introduce new architectural
cycles.

## Failure Handling and Diagnostics

Because this is a clean break, importing `unfoldlab` or invoking an
`unfoldlab` executable is expected to fail after installation. The project will
not add runtime warnings, alias modules, import hooks, or transitional stubs.

Migration failures will be caught at deterministic boundaries:

- stale internal paths: Ruff, mypy, import sweeps, and repository-wide text
  checks;
- incomplete command registration: existing CLI structure tests and
  `unmochan --help` smoke tests;
- package-discovery errors: wheel/sdist content inspection;
- source-tree-only imports: installation into a clean temporary environment;
- export drift: explicit tests for the curated root `__all__`;
- algorithm regressions: the unchanged scientific test suite.

No exception behavior inside numerical or I/O routines changes in this phase.

## Verification Gates

The implementation is complete only when all applicable gates pass:

1. Baseline test results are recorded before the move.
2. Focused namespace, public-export, CLI, and packaging tests pass.
3. Ruff passes for `src/unmochan`, `tests`, and `examples`.
4. mypy passes for `src/unmochan` using the existing configuration.
5. The complete pytest suite passes.
6. Wheel and source distributions build successfully.
7. A clean temporary environment can install the wheel and run:

   ```text
   import unmochan
   unmochan --help
   ```

8. The clean environment cannot import `unfoldlab` and exposes no `unfoldlab`
   executable.
9. Distribution contents and tracked text contain no stale implementation or
   command references to `unfoldlab`; historical Git commit messages are out of
   scope.
10. The Git diff contains no numerical implementation changes beyond import
    paths and file relocation.

## Phase-Two Candidates

The following files are candidates for later, separately designed splits based
on size and graph centrality:

- `cli/analysis.py`
- `core/spacegroup.py`
- `core/tight_binding.py`
- `io/vasp_wfc.py`
- `core/lcao.py`
- `cli/commands.py`
- `core/unfolding.py`
- `io/qe_wfc.py`
- `io/qe.py`
- `cli/diagnostics.py`

Phase two will group changes by responsibility and direct tests, not by an
arbitrary line-count target.

## Non-Goals

- No compatibility package, import alias, or deprecated command.
- No numerical, scientific, serialization-format, or CLI behavior changes.
- No dependency upgrades.
- No module splitting or feature-first reorganization in phase one.
- No test deletion solely because a test is old; tests may be renamed or
  rewritten only to reflect the new namespace and API.
- No committed generated dependency graph.

## Acceptance Criteria

- The only installable Python package is `unmochan` under `src/`.
- The `unmochan` package contains the real implementation rather than a
  forwarding shell.
- The historical package and command are absent by design.
- Root imports match the curated API exactly.
- Subpackage boundaries and scientific behavior remain intact.
- Maintained source, tests, examples, and documentation use the new namespace.
- Superseded status/history files are removed after durable content extraction.
- All verification gates pass with evidence recorded in the implementation
  work.
