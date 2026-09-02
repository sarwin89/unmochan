# Testing UnfoldLab

UnfoldLab's v1 quality bar is backend parity: VASP and QE inputs that describe
the same physical plane-wave problem must normalize into the same internal data
contract and produce the same unfolded weights within numerical tolerance.

## Default Gate

Run this before preparing a commit:

```bash
python -m pytest
python -m ruff check .
python -m compileall unfoldlab tests
python -m pip check
```

The default suite is intentionally lightweight. It uses small synthetic fixtures
so contributors do not need proprietary or large DFT output files.

## Test Markers

- `unit`: fast tests for algorithms, parsers, and data contracts.
- `cli`: command-line and guided-mode behavior.
- `integration`: end-to-end synthetic workflows.
- `parity`: VASP/QE equivalence checks.
- `plot`: plot generation or plot sanity checks.
- `slow`: longer-running tests outside the tight loop.
- `io_real`: optional tests that require real external DFT output fixtures.

Examples:

```bash
python -m pytest -m "unit or parity"
python -m pytest -m "not slow and not io_real"
python -m pytest -m io_real
```

## Required Coverage Areas

- Transform convention: `A_sc = T @ A_pc` and `K_sc = k_pc @ T.T`.
- K-point folding, path interpolation, distance/tick generation, and boundary
  wrapping for diagonal and non-diagonal supercells.
- Plane-wave matching through `(K_sc + G_sc) @ inv(T).T`.
- Spectral weight normalization per `(k-point, band)`, with spinor components
  summed before normalization.
- Gamma-only (time-reversal reduced) wavefunction storage: expansion of the
  stored half basis, both VASP half-space conventions, and the sum rule on the
  expanded basis.
- Unfolding from a symmetry-reduced wavefunction set: reuse of one stored
  k-point across a fiber, point-group images of a stored k-point, and a clear
  error for a k-path the stored set cannot cover.
- Certified minimum-image search: agreement with brute force and a reported
  failure rather than an uncertified answer.
- VASP and QE workflows for `make-kpoints`, `weights`, `unfold`, exports, and
  guided CLI paths.
- Twist/reference workflows from both VASP POSCAR/CONTCAR and QE `pw.x` inputs.

## Optional Real-I/O Fixtures

Real VASP/QE outputs should live behind the `io_real` marker and should not be
required in normal CI. Keep any checked-in fixture tiny, redistributable, and
documented. Larger fixtures should be downloaded manually or stored as release
artifacts rather than added to the repository.

The preferred real fixture matrix is:

- VASP collinear WAVECAR.
- VASP `ISPIN=2` WAVECAR.
- VASP SOC/noncollinear WAVECAR.
- QE HDF5 wavefunction.
- QE Fortran-binary wavefunction.
- One equivalent VASP/QE toy system for parity reporting.

## Packaging Smoke Test

For release candidates, build and install the package in a clean environment:

```bash
python -m build
python -m venv .venv-smoke
.venv-smoke\Scripts\python -m pip install dist/*.whl
.venv-smoke\Scripts\unfoldlab --help
```
