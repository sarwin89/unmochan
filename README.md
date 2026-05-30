# UnfoldLab

UnfoldLab is the starting point for a research-grade Python library for band
unfolding, effective band structures, and twisted/moire electronic-structure
analysis.

The project is VASP-first, but the package layout is designed for multiple
backends: projector-based unfolding, plane-wave coefficient unfolding, Wannier
and tight-binding workflows, and future Quantum ESPRESSO/generic HDF5 inputs.

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
- Treat twisted/layer-resolved workflows as first-class design targets rather
  than later plotting conveniences.

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
  --kpoint K:0.3333333333,0.3333333333,0
```

## Current status

This repository has the first package skeleton and tested numerical primitives.
The unfolding backends are intentionally thin until the parser and data-model
contracts are stable.

## License

No public license has been selected yet.
