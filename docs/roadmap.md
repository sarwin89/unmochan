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
