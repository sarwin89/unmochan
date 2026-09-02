# Future Implementation Plan

## Phase 1: v1 Stabilization

- Keep `make-kpoints`, `weights`, and `unfold` as the primary backend-neutral
  workflow.
- Document `unmochan` as the primary executable and preserve `unfoldlab vasp ...`
  and `unfoldlab qe ...` as compatibility aliases.
- Add neutral names such as `UnfoldingPath` and `KPointMap` for shared path
  models while keeping QE-prefixed aliases working.
- Keep guided mode focused on DFT workflow steps, not theory pages.
- Document parity assumptions, supported wavefunction formats, unsupported
  gamma-only storage, and expected k-point ordering.

## Phase 2: Real Backend Validation

- Build a small redistributable fixture corpus from equivalent VASP and QE toy
  systems.
- Validate VASP WAVECAR parsing against real collinear, spin-polarized, and
  SOC/noncollinear files.
- Validate QE HDF5 and Fortran-binary readers against real `pw.x` outputs.
- Add a parity report that compares VASP and QE effective band structures for
  equivalent systems.

## Phase 3: Production I/O Hardening

- Expand WAVECAR support across VASP precision modes and version variations.
- Improve QE save-directory discovery using `prefix`, `outdir`, XML metadata,
  and collected/distributed wavefunction layouts.
- Harden QE and POSCAR structure parsing around common DFT input variants.
- Add explicit, tested failure modes for unsupported storage conventions.

## Phase 4: Data Model And Export

- Add stable `EffectiveBandStructure` and diagnostics serialization, with JSON
  first and optional HDF5 later.
- Standardize table headers and metadata blocks across VASP and QE exports.
- Add reproducible run manifests with package version, transform matrix, source
  files, energy reference, spin settings, and weight mode.
- Keep plot visual regression optional, but always smoke-test plot generation.

## Phase 5: Advanced Physics

- Add VASP PROCAR/projector-assisted unfolding as an approximate complementary
  mode.
- Add spin-resolved unfolding and spin texture exports for SOC/noncollinear
  calculations.
- Extend twist/reference workflows to VASP and QE equally, including
  layer/reference-cell diagnostics.
- Consider Wannier, tight-binding, and phonon backends after plane-wave parity
  is stable.
- Add chunked reading, lazy k-point iteration, and optional parallel execution
  for large wavefunction files.

## Phase 6: Release Readiness

- Require the full local gate before tagging v1.
- Publish release notes that separate production-supported, experimental, and
  explicitly unsupported behavior.
- Prepare example datasets and tutorials for one VASP workflow, one QE workflow,
  and one VASP/QE parity comparison.
