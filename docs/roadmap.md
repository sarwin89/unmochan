# UnfoldLab Roadmap

This roadmap condenses the project goal document into implementable milestones.

## Milestone 1: Minimal VASP EBS engine

- Parse `POSCAR`/`CONTCAR`, `KPOINTS`, `EIGENVAL`, and enough `PROCAR` data for
  projection-weight experiments.
- Detect primitive-to-supercell transformation matrices.
- Fold primitive k-path points to supercell k-points and validate whether a
  calculation contains the needed points.
- Store unfolded weights and band energies in a stable `EffectiveBandStructure`
  object.
- Produce initial fat-band and spectral-function plots.

## Milestone 2: K-point generation and diagnostics

- Generate VASP `KPOINTS` files from primitive high-symmetry paths.
- Handle diagonal and non-diagonal supercells.
- Report missing k-points with tolerance-aware diagnostics.
- Add sum-rule and normalization checks for weights.

## Milestone 3: Spectral-function plotting and serialization

- Add Gaussian, Lorentzian, and adaptive broadening.
- Support atom, orbital, layer, and spin filters.
- Serialize large EBS objects to HDF5.
- Add Matplotlib and Plotly frontends.

## Milestone 4: Rigorous VASP wavefunction backend

- Implement or integrate robust `WAVECAR` reading.
- Compute plane-wave coefficient unfolding weights.
- Compare plane-wave and projection-based approximations.

## Milestone 5: Twisted 2D materials

- Detect layers, twist angles, moire reciprocal lattices, and valley mappings.
- Support layer-resolved and valley-resolved unfolding references.
- Build examples around twisted MoS2 and related 2D heterostructures.

## Later milestones

- Wannier90 and tight-binding backends.
- Quantum ESPRESSO and generic HDF5 backends.
- Textbook-level theory documentation and benchmark galleries.
