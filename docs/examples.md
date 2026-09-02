# Example Classes

Examples should be organized around general physical and computational cases:

- `toy_1d_chain/`
- `toy_2d_square_lattice/`
- `toy_honeycomb_lattice/`
- `generic_2x2x2_supercell/`
- `generic_vacancy_defect/`
- `generic_substitution_defect/`
- `generic_antisite_defect/`
- `generic_interstitial_defect/`
- `generic_disordered_alloy/`
- `generic_surface_reconstruction/`
- `generic_adsorbate_on_surface/`
- `generic_interface/`
- `generic_twisted_bilayer/`
- `generic_twisted_multilayer/`
- `generic_rotated_heterostructure/`
- `generic_wannier_model/`
- `generic_tight_binding_model/`

Material-specific tutorials may be added under `material_specific_optional/`
only when they demonstrate a general method without changing core assumptions.

## Runnable model examples

`examples/` ships three model Hamiltonians that need no DFT run at all, so a
workflow can be checked end to end in a second:

- `toy_1d_chain.json` — one orbital per cell, nearest-neighbour hopping.
- `toy_1d_chain_defect.json` — the same chain with one raised on-site energy,
  i.e. a substitutional defect that mixes the fiber.
- `toy_2d_square.json` — an anisotropic square lattice, for non-diagonal
  supercells.

```bash
unmochan model unfold --model examples/toy_1d_chain.json \
    --matrix "3 0 0 0 1 0 0 0 1" --kpoint 0.05,0,0

unmochan model unfold --model examples/toy_2d_square.json \
    --matrix "2 1 0 -1 1 0 0 0 1" --kpoint 0.11,0.23,0 --json ebs.json
```

For the perfect chain every supercell state comes back with weight one on a
single primitive k-point — band folding undone.  For the defect model the
weight spreads over the fiber, and the reported `sum_rules` confirm that it is
conserved: the weights of one state add up to one over the fiber, and the
weights of all states add up to the number of primitive bands at each k-point.

### An effective band structure along a path

`model unfold` asks what one supercell k-point unfolds to; `model bands` asks
the question a plot asks, which is how much weight each *primitive* k-point of a
band path carries:

```bash
unmochan model bands --model examples/toy_1d_chain.json \
    --matrix "3 0 0 0 1 0 0 0 1" --path 0,0,0:0.5,0,0 --points 51 \
    --json chain_ebs.json

unmochan model bands --model examples/toy_1d_chain.json \
    --matrix "3 0 0 0 1 0 0 0 1" --path 0,0,0:0.5,0,0 --points 51 \
    --onsite 1,0,1.5 --json defect_ebs.json
```

The first run reproduces the primitive dispersion with weight one and leaves the
two folded copies at weight zero, which is
`UnfoldLab.tbWeight_blochState` in numerical form.  The second run puts a
defect on one cell of the supercell and the delta peaks broaden into a genuine
spectral function, while the total weight at each k-point is unchanged.

### Reading a Wannier90 Hamiltonian

Any `seedname_hr.dat` written by Wannier90 can be used in place of the JSON
model; the Wigner-Seitz degeneracies are divided out on the way in:

```bash
unmochan model bands --model wannier/silicon_hr.dat --format hr \
    --matrix "2 0 0 0 2 0 0 0 2" --path 0,0,0:0.5,0,0.5 --points 101
```

`--format` defaults to `auto`, which reads `.json` as the model format and
anything else as Wannier90 output.
