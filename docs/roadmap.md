# UnfoldLab Roadmap

This roadmap condenses the project goal document into implementable milestones.
The project is material-agnostic: any material-specific tutorial is an instance
of a general method, not a core design assumption.

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

## Milestone 3: Spectral-function plotting, projections, and serialization

- Add Gaussian, Lorentzian, and adaptive broadening.
- Support universal projection selectors for species, atoms, orbitals, layers,
  sublattices, regions, defect shells, surfaces, interfaces, adsorbates,
  substrates, molecule-like components, valleys, and spin channels.
- Add user-defined projection groups through configuration rather than
  hard-coded material aliases.
- Serialize large EBS objects to HDF5.
- Add Matplotlib and Plotly frontends.

## Milestone 4: Rigorous VASP wavefunction backend

- Expand `WAVECAR` validation against real VASP files from multiple versions.
- Add gamma-only reconstruction when a code-independent reconstruction strategy
  is available.
- Compare plane-wave and projection-based approximations.

## Milestone 5: Generic reference-resolved and moire workflows

- Detect layers, relative rotations, moire reciprocal lattices, local stacking,
  local strain, and reference-dependent valley mappings where definitions are
  provided by symmetry detection or user configuration.
- Support unfolding to primitive crystal, conventional-cell, reference-layer,
  substrate, adsorbate, moire mini-zone, local approximate, and user-defined
  reciprocal references.
- Support any number of reference structures for twisted multilayers, rotated
  heterostructures, interfaces, slabs, molecule-on-surface systems, and generic
  layered crystals.

## General example classes

- Toy 1D chain.
- Toy 2D square lattice.
- Toy honeycomb lattice.
- Generic diagonal and non-diagonal supercells.
- Generic vacancy, substitutional, antisite, interstitial, charged, and complex
  defects.
- Generic disordered alloy and special quasirandom structures.
- Generic reconstructed surface and adsorbate-on-surface systems.
- Generic interface and rotated heterostructure systems.
- Generic magnetic, spinor, charge-density-wave, Peierls-distorted, and
  Jahn-Teller-distorted supercells.
- Generic twisted bilayer, twisted multilayer, and moire superlattice systems.
- Generic Wannier, tight-binding, continuum, and Hamiltonian-matrix models.

## General theory chapters

- Band folding and unfolding in arbitrary crystals.
- Defect unfolding in arbitrary crystals.
- Disorder and alloy unfolding.
- Surface, slab, interface, and adsorbate unfolding.
- Magnetic, spinor, and spin-texture unfolding.
- Charge-density-wave, Peierls, Jahn-Teller, and reconstructed-supercell
  unfolding.
- Twisted and moire unfolding for layered and quasi-2D systems.
- Wannier and tight-binding unfolding.
- Phonon, exciton, electron-phonon, superconducting-gap, Fermi-surface, and
  ARPES-like extensions.

## Completed backend foundations

- Packaged Quantum ESPRESSO folded-path generation and plane-wave unfolding
  from saved wavefunctions, external weight tables, or coefficient tables.
- VASP `WAVECAR` plane-wave unfolding through the same matching kernel used by QE.
- Guided CLI mode for menu-driven workflows alongside scriptable subcommands.

## Later milestones

- Wannier90 and tight-binding backends.
- Generic HDF5 backends beyond QE wavefunction layouts.
- Textbook-level theory documentation and benchmark galleries.
