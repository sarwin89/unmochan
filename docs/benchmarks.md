# Benchmark gallery

Cases whose unfolded band structure is known in closed form, so that a
disagreement is a bug rather than a difference of convention.  Every case in
this file is executed by `tests/test_benchmarks.py`, so the gallery cannot go
stale: if the code stops reproducing one of these answers, the test suite
fails.

These are *analytic* benchmarks.  They do not reproduce a published figure —
that would need the published input files, which are not part of this
repository — but they pin down exactly the quantities a published figure is
made of: which primitive band the weight lands on, how much of it, and where
the sum rules sit.

## 1. Supercell of a perfect chain

A single-orbital chain has `E(k) = 2 t cos(2 pi k)`.  Build an `N x 1 x 1`
supercell of it and unfold: the supercell spectrum has `N` bands at each
primitive k-point, but only the one at the primitive energy carries any
weight, and it carries all of it.

* checked for `N = 2, 3, 4, 5` in
  `test_perfect_chain_unfolds_to_the_primitive_band`;
* the formal statement is `IsFiberRepr.weight_ideal`.

A caveat the test encodes: two members of a fiber can fold onto *degenerate*
supercell states (for the chain, `k` and `-k` at the zone edge), and a
diagonalizer is then free to return any basis of that eigenspace.  The weight
of an individual eigenvector is not basis independent; the total weight at the
primitive k-point, and the energy of every state carrying any of it, are.  A
benchmark that asserts "exactly one eigenvector has weight one" is testing the
LAPACK build, not the physics.

## 2. Supercell of a perfect square lattice

`E(k) = 2 tx cos(2 pi kx) + 2 ty cos(2 pi ky)` with an anisotropic `2 x 2 x 1`
supercell, checked at Gamma, at `(1/4, 0, 0)`, at `(1/4, 1/4, 0)` and at the
zone corner, in `test_perfect_square_lattice_unfolds_to_the_primitive_band`.
The anisotropy matters: it distinguishes the two axes, so a transposed
transform or a swapped index would show up.

## 3. Dimerized chain (Su–Schrieffer–Heeger)

Alternating hoppings `t1, t2` in a doubled cell give two bands
`± |t1 + t2 exp(2 pi i k)|` and a gap of `2 |t1 - t2|` at the zone boundary.
`test_dimerized_chain_opens_the_expected_gap` checks the gap and that a
`1 x 1 x 1` "supercell" leaves every weight at one — the degenerate case of the
transform being the identity, which is worth having as a fixed point.

## 4. A single plane-wave state — the ideal case, directly

Occupy exactly the plane waves that belong to one member of a fiber, and unfold
against all three members of a `3 x 1 x 1` fiber: weight one at that member,
zero at the other two, sum rule exactly one.  This is the numerical form of
`IsFiberRepr.weight_ideal` and it exercises the matching kernel
(`matching_g_mask`) rather than the tight-binding layer.

## 5. Chain with a defect

An on-site shift on one cell of a `4 x 1 x 1` supercell destroys the ideal case
— the weight spreads over several states, which is the whole point of plotting
an effective band structure — but not the sum rule: at fixed state index the
four fiber members still share a total weight of one
(`IsFiberRepr.sum_weight_eq_one`).  This is checked in
`test_defect_chain_conserves_the_fiber_sum_rule`.

## What is deliberately not here

* A comparison against published unfolded band structures of real materials.
  It needs the original DFT inputs; without them, a "reproduction" would be a
  redrawing.  What the repository does provide instead is `docs/testing.md`,
  which lists the synthetic file fixtures that stand in for real WAVECAR and
  QE `.save` data.
* Absolute spectral-function intensities from a PAW calculation.  The weights
  computed here are pseudo-wavefunction weights; the missing on-site
  augmentation is a property of the input data, not of this code.
