# UnfoldLab

UnfoldLab is the starting point for a research-grade Python library for band
unfolding, spectral analysis, and supercell interpretation across arbitrary
materials and model systems.

The project is VASP-first, but the package layout is designed for multiple
backends: projector-based unfolding, plane-wave coefficient unfolding, Wannier
and tight-binding workflows, and future Quantum ESPRESSO/generic HDF5 inputs.

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

## Current status

This repository has the first package skeleton and tested numerical primitives.
The unfolding backends are intentionally thin until the parser and data-model
contracts are stable.

## License

No public license has been selected yet.
