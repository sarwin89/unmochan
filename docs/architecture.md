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
cli ------> workflows ------> io ------> core
 |               |             |          ^
 +---------------+-------------+----------+

twist ----> workflows / io / core
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
