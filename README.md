# UnfoldLab

UnfoldLab is the starting point for a research-grade Python library for band
unfolding, spectral analysis, and supercell interpretation across arbitrary
materials and model systems.

The project is VASP-first for structural mapping workflows and now includes a
packaged Quantum ESPRESSO backend for plane-wave coefficient unfolding. The
layout is designed for multiple backends: projector-based unfolding, plane-wave
coefficient unfolding, Wannier and tight-binding workflows, and generic HDF5
inputs.

The design philosophy is material-agnostic:

```text
This is a general band-unfolding, spectral-analysis, and supercell-interpretation
library for arbitrary materials and model systems.
```

Material-specific workflows belong in optional tutorials or user configuration,
not in core APIs, class names, projection syntax, or algorithmic assumptions.

## Initial scope

- Parse core VASP inputs and outputs: `POSCAR`, `CONTCAR`, `KPOINTS`,
  `EIGENVAL`, and eventually `PROCAR`/`WAVECAR`.
- Generate Quantum ESPRESSO folded supercell k-paths and compute true
  plane-wave unfolding weights from `wfc*.hdf5`, `wfc*.h5`, `wfc*.dat`, or
  exported coefficient tables.
- Detect primitive-to-supercell transformation matrices, including non-diagonal
  integer transforms.
- Fold primitive-path k-points into required supercell k-points.
- Represent effective band structures and broaden them into spectral intensity
  maps.
- Provide clear diagnostics for commensurability, missing k-points, and spectral
  weight normalization.
- Treat arbitrary reference-resolved workflows as first-class design targets:
  primitive cells, conventional cells, layers, substrates, adsorbates,
  interfaces, molecule-like projection bases, moire mini-zones, local
  approximate Brillouin zones, and user-defined reciprocal references.
- Keep projections generic through selectors such as `species:*`, `atom:4`,
  `orbital:d`, `layer:0`, `region:defect_core`, `surface:top`, `interface:A`,
  `valley:example`, and `spin:z`.

## Generic systems in scope

UnfoldLab should support or prepare for standard crystalline supercells,
defects, disorder, alloys, slabs, reconstructed surfaces, adsorbates,
interfaces, magnetic supercells, spinor calculations, lattice instabilities,
layered systems, rotated heterostructures, moire superlattices, Wannier models,
tight-binding models, and generic Hamiltonian matrices.

Specific materials may appear later as optional tutorial instances of these
general methods, never as built-in design targets.

## Quick start from a checkout

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Inspect a primitive/supercell mapping:

```bash
unfoldlab vasp map --primitive primitive/POSCAR --supercell supercell/POSCAR
```

Launch a guided VASPKIT-style menu instead of typing subcommands:

```bash
unfoldlab
```

Fold explicit primitive fractional k-points into the supercell Brillouin zone:

```bash
unfoldlab vasp fold-kpoints \
  --primitive primitive/POSCAR \
  --supercell supercell/POSCAR \
  --kpoint G:0,0,0 \
  --kpoint X:0.5,0,0
```

Parse generic projection selectors:

```python
from unfoldlab import ProjectionSelector

selectors = [
    ProjectionSelector.parse("species:*"),
    ProjectionSelector.parse("orbital:d"),
    ProjectionSelector.parse("surface:top"),
    ProjectionSelector.parse("spin:z"),
]
```

Generate a folded supercell path for either backend:

```bash
unfoldlab make-kpoints --code qe path.json \
  --kpoints qe_kpoints_supercell.in \
  --kmap kmap.tsv \
  --ticks path_ticks.tsv

unfoldlab make-kpoints --code vasp path.json \
  --kpoints KPOINTS \
  --kmap kmap.tsv \
  --ticks path_ticks.tsv
```

Compute and plot QE unfolded bands from saved wavefunctions:

```bash
unfoldlab unfold \
  --code qe \
  --bands bands.dat.gnu \
  --kmap kmap.tsv \
  --qe-save-dir ./qe_tmp/supercell.save \
  --matrix "2 0 0 0 2 0 0 0 1" \
  --fermi 5.43 \
  --out unfolded_bands.dat \
  --plot unfolded_bands.png \
  --write-weights weights.dat
```

Compute and plot VASP unfolded bands from `WAVECAR`:

```bash
unfoldlab unfold \
  --code vasp \
  --kmap kmap.tsv \
  --wavecar WAVECAR \
  --matrix "2 0 0 0 2 0 0 0 1" \
  --fermi 5.43 \
  --out unfolded_bands.dat \
  --plot unfolded_bands.png \
  --write-weights weights.dat
```

## Backend unification

UnfoldLab uses one backend-neutral plane-wave unfolding contract for QE and
VASP `WAVECAR` support:

```text
A_sc = T @ A_pc
K_sc = k_pc @ T.T
(K_sc + G_sc) @ inv(T).T - k_pc must be an integer vector
```

QE and VASP wavefunctions are normalized into `PlaneWaveKPointData` before
weights are computed. If the primitive path, transformation matrix, energies,
G-vectors, and normalized coefficients are equivalent, VASP and QE unfolding
weights should agree within numerical tolerance.

Define material-dependent valleys explicitly:

```python
from unfoldlab import ValleyDefinition

valley = ValleyDefinition(
    label="example_valley",
    center_frac=[0.333333, 0.333333, 0.0],
    radius=0.05,
    reference_bz="layer_1",
)
```

Twist/reference workflows accept loaded `Structure` objects, VASP POSCAR/CONTCAR
files, or QE `pw.x` inputs with `CELL_PARAMETERS` and `ATOMIC_POSITIONS`:

```python
from unfoldlab import TwistedUnfoldingProblem

problem = TwistedUnfoldingProblem.from_qe(
    references={"layer": "primitive.in"},
    supercell="supercell.in",
)
```

## Current status

This repository has tested numerical primitives, VASP structure/EIGENVAL
workflow helpers, VASP `WAVECAR` plane-wave unfolding, and an installable QE
plane-wave unfolding backend.

## License

No public license has been selected yet.
