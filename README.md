# UnfoldLab

A material-agnostic Python toolkit for **band unfolding**, spectral analysis and
supercell interpretation, together with a machine-checked formal model of the
mathematics it implements.

Given a supercell calculation and the integer transformation `T` relating the
supercell to a primitive reference cell, UnfoldLab reconstructs the *effective
band structure* in the primitive Brillouin zone: for every supercell eigenstate
it computes how much of its weight belongs to each primitive k-point of the
fiber, and turns those weights into fat bands, spectral functions, densities of
states, Fermi levels, spin textures and effective masses.

## What is in the repository

| Path | Contents |
| --- | --- |
| `unfoldlab/core/` | the unfolding kernels: folding maps, plane-wave matching, weights and sum rules, spectral broadening, DOS, Fermi level, projections, tight binding, LCAO, phonons, symmetry, disorder ensembles |
| `unfoldlab/io/` | readers and writers: VASP `WAVECAR` / `PROCAR` / `EIGENVAL`, Quantum ESPRESSO `.dat`, HDF5 and XML, Wannier90 `_hr.dat` / `_tb.dat`, JSON/HDF5 serialization, plotting |
| `unfoldlab/twist/` | commensurate twist angles, moiré cells and stack diagnostics |
| `unfoldlab/workflows/` | end-to-end problem definitions and backend drivers |
| `unfoldlab/cli/` | the `unfoldlab` command line, scriptable and guided |
| `lean-proofs/RequestProject/` | optional local Lean 4 + Mathlib proof bundle, kept untracked |
| `tests/` | the test suite, including synthetic VASP/QE fixtures |
| `docs/` | review log, formal-model map, roadmap, benchmarks, progress tracker |
| `examples/` | runnable toy models that need no DFT run |

## Installation

```bash
pip install -e ".[io,plot,dev]"
```

The optional extras gate the HDF5 readers (`h5py`), the plotting backends
(`matplotlib`, `plotly`) and the development tools.  Every plot has a
dependency-free SVG fallback, so the core workflows run on `numpy` alone.

## A one-second example

```bash
unfoldlab model unfold --model examples/toy_1d_chain.json \
    --matrix "3 0 0 0 1 0 0 0 1" --kpoint 0.05,0,0

unfoldlab model bands --model examples/toy_1d_chain_defect.json \
    --matrix "3 0 0 0 1 0 0 0 1" --path 0,0,0:0.5,0,0 --points 51 \
    --json chain_ebs.json
```

For the perfect chain each supercell state returns weight one on a single
primitive k-point — band folding undone.  For the defect model the weight
spreads across the fiber and the reported `sum_rules` confirm that it is
conserved.  See `docs/examples.md` for the DFT-backed workflows.

## The formal model

`lean-proofs/RequestProject/` is an optional local Lean 4 development, built
against Mathlib, that states and proves the identities the numerics rely on:
the fiber sum rule, representative independence of the weights, the exact
integer matching criterion, conservation of spectral weight under broadening,
chunking identities, non-orthogonal LCAO sum rules, and machine-checked
counterexamples for known pitfalls.

```bash
cd lean-proofs
lake build
```

`docs/formal-model.md` maps each Lean statement to the Python function it
governs.

## Development gates

```bash
python -m pytest -q          # test suite
ruff check .                 # lint
mypy unfoldlab               # types
cd lean-proofs && lake build # optional local formal model, if present
```

`docs/progress.md` records the state of every area and the backlog;
`docs/findings.md` is the append-only log of what was wrong and how it was
fixed.
