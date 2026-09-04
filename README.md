# Unmochan

**UNfolding MOmentum-space Crystal Hamiltonian ANalysis**: a
material-agnostic Python toolkit for **band unfolding**, spectral analysis and
supercell interpretation, together with a machine-checked formal model of the
mathematics it implements.

Given a supercell calculation and the integer transformation `T` relating the
supercell to a primitive reference cell, Unmochan reconstructs the *effective
band structure* in the primitive Brillouin zone: for every supercell eigenstate
it computes how much of its weight belongs to each primitive k-point of the
fiber, and turns those weights into fat bands, spectral functions, densities of
states, Fermi levels, spin textures and effective masses.

## What is in the repository

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

## Installation

```bash
pip install -e ".[io,plot,dev]"
```

The optional extras gate the HDF5 readers (`h5py`), the plotting backends
(`matplotlib`, `plotly`) and the development tools.  Every plot has a
dependency-free SVG fallback, so the core workflows run on `numpy` alone.

## A one-second example

```bash
unmochan model unfold --model examples/toy_1d_chain.json \
    --matrix "3 0 0 0 1 0 0 0 1" --kpoint 0.05,0,0

unmochan model bands --model examples/toy_1d_chain_defect.json \
    --matrix "3 0 0 0 1 0 0 0 1" --path 0,0,0:0.5,0,0 --points 51 \
    --json chain_ebs.json
```

For the perfect chain each supercell state returns weight one on a single
primitive k-point — band folding undone.  For the defect model the weight
spreads across the fiber and the reported `sum_rules` confirm that it is
conserved.  See `docs/examples.md` for the DFT-backed workflows.

## Project

Unmochan is the active package and command name. The implementation is
developed and distributed solely from `src/unmochan`.

Thanks to Ritam Chakraborty for contributions to the project direction and
implementation, and to Prajwal Souza for the initial codebase from which the
project developed.

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
python -m pytest --basetemp .tmp-pytest -p no:cacheprovider
python -m ruff check .
python -m mypy src/unmochan
python -m compileall -q src/unmochan tests examples
```

See `docs/architecture.md` for package boundaries, `docs/testing.md` for the
verification policy, and `docs/roadmap.md` for open project work.
