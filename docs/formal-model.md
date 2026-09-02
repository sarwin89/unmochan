# Formal model of band unfolding

The mathematics that UnfoldLab implements is formalized and machine-checked in
Lean 4 (with Mathlib) under the optional local
`lean-proofs/RequestProject/Unfolding/` bundle.  Every statement listed here is
proved without `sorry`; `lean-proofs/RequestProject/Main.lean` imports all
modules, so `cd lean-proofs && lake build` checks the whole development when
the untracked proof bundle is present.

This file is the dictionary between the Lean statements and the Python code.

## Conventions

UnfoldLab stores lattice vectors as **rows** of a 3x3 array, so a supercell is

```text
A_sc = T @ A_pc            (T integral, det T != 0)
```

and fractional reciprocal coordinates transform as `K_sc = k_pc @ T.T`, i.e. as
a *matrix–vector* product with `T` acting on the left of a column vector.  The
Lean model uses the column convention `fold T k = T *ᵥ k`, which is the same
map; `UnfoldLab.fold` is defined that way in `Basic.lean` and the docstring of
`unfoldlab.core.plane_waves` records the correspondence.

Notation used below:

| Lean | meaning |
| --- | --- |
| `ivec m` | an integer vector viewed in `ℝ^n` |
| `CongrModZ x y` | `x - y` is an integer vector (equality of fractional k-points) |
| `rmat T` | the integer matrix `T` viewed over `ℝ` |
| `fold T k` | `rmat T *ᵥ k`, the folding map from primitive to supercell coordinates |
| `FoldsTo T k K` | `fold T k ≡ K (mod ℤ^n)` |
| `PWMatches T k K G` | the supercell plane wave `G` belongs to primitive k-point `k` |

## Lattices and folding — `Unfolding/Basic.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `reciprocalLattice_supercell` | `SupercellTransformation` | the supercell reciprocal lattice is a sublattice of the primitive one, with the same integer matrix `T` |
| `cartesian_fold_eq` | `core.kpoints` | folding in fractional coordinates agrees with the Cartesian identity `k_cart = K_cart`; the fold map is *not* a change of the physical vector, only of its coordinates |
| `fold_congrModZ` | `wrap_fractional` | `k ≡ k' (mod ℤ)` implies `fold T k ≡ fold T k' (mod ℤ)`, so the choice of representative for a primitive k-point is irrelevant |
| `exists_foldsTo` | `core.kpoints` | every supercell k-point has at least one primitive preimage |

## The matching condition — `Unfolding/Matching.lean`

`PWMatches T k K G` is the predicate that decides which supercell plane waves
contribute to the weight of `k`.  It is implemented by
`unfoldlab.core.plane_waves.matching_g_mask`.

| Lean | Python | statement |
| --- | --- | --- |
| `pwMatches_of_add_eq` | `matching_g_mask` | the mask does not depend on the representative chosen for `K`; shifting `K` by an integer vector shifts the matched set consistently |
| `pwMatches_iff_exists` | — | `G` matches `k` iff `G + m₀ ∈ T·ℤ^n` for the offset `m₀` fixing the representatives |
| `mem_range_mulVec_iff_adjugate` | `matching_g_mask`, `integer_adjugate3` | **exactness**: `v ∈ T·ℤ^n` iff `det T` divides every entry of `adj(T)·v`.  This is what allows `matching_g_mask` to be evaluated in exact integer arithmetic instead of by rounding `inv(T) @ v` |
| `existsUnique_primitive_kpoint` | — | each supercell plane wave belongs to exactly one primitive k-point modulo `ℤ^n` |

The exactness lemma is the reason the Python matcher was rewritten: the previous
float test `|round(x) - x| < atol` is replaced by the integer test
`((G + m₀) @ adj(T).T) % det(T) == 0`, which has no tolerance at all.

## G-vector classes — `Unfolding/Classes.lean`

`clsLabel T G = adj(T)·G mod det T` is the *class label* of a G-vector.  It does
not mention the primitive k-point, which is what lets
`shared_weights_from_coefficients` decide the matching test once for all the
k-points a wavefunction serves instead of once per k-point.

| Lean | Python | statement |
| --- | --- | --- |
| `clsLabel` | `_g_class_residues`, `_kpoint_class_residues` | the label itself; the k-point side is the label of `-m₀` |
| `clsLabel_eq_iff_mem` | — | two vectors share a label exactly when they differ by a supercell lattice vector, so the label names the class faithfully and without a tolerance |
| `pwMatches_iff_clsLabel_eq` | `shared_weights_from_coefficients`, `compute_weights_from_coefficient_table` | the matching test is the equality of two labels — one integer comparison, with the G-side independent of `k` |
| `clsLabel_eq_iff_congrModZ` | `_select_classes` | two members of a fiber select the same class only if they are the same primitive k-point; distinct members collect disjoint plane waves, a repeated k-point is served twice from one class |
| `sum_over_matching_eq_sum_over_class` | `_class_sums` | the weight accumulation is a sum over one class, so a single segmented pass serves every k-point at once |

## Fibers — `Unfolding/Fiber.lean`

`IsFiberRepr T K ks` says that the finite set `ks` lists every primitive
k-point folding onto `K` exactly once modulo `ℤ^n` — precisely the input a
correct unfolding k-map must supply.

| Lean | Python | statement |
| --- | --- | --- |
| `card_quotient_imageLattice` | `integer_det3` | `ℤ^n / T·ℤ^n` has `|det T|` elements |
| `IsFiberRepr.existsUnique_pwMatches` | `matching_g_mask` | the masks of a full fiber **partition** the supercell plane-wave basis |
| `IsFiberRepr.card_eq_natAbs_det` | `check_supercell_consistency` | a fiber has exactly `|det T|` primitive k-points |

## Constructing the fiber — `Unfolding/Coset.lean`

`Fiber.lean` characterizes a fiber; this module says how to build one.
`IsCosetRepr T ms` says that `ms` lists every class of `ℤ^n / T·ℤ^n` exactly
once, and the k-points are then recovered by solving `Tᵀ k = K + m`.

| Lean | Python | statement |
| --- | --- | --- |
| `IsCosetRepr.isFiberRepr` | `core.kpoints.fiber_kpoints` | the k-points built from coset representatives are a complete, irredundant fiber over `K` |
| `IsCosetRepr.card_eq_natAbs_det` | `core.lattice_quotient.coset_representatives` | a correct transversal of `ℤ^n / T·ℤ^n` has `|det T|` entries |
| `congrModZ_cosetKPoint_iff` | `core.lattice_quotient` | two representatives give the same k-point exactly when they differ by an element of `T·ℤ^n` |

The representatives themselves come from the Smith normal form `U T V = D`:
since `V` is unimodular, `T·ℤ^n = U^{-1} D ℤ^n`, so the classes are `U^{-1} y`
with `y` in the box `∏ [0, d_i)`.  `core.lattice_quotient.smith_normal_form`
computes `U`, `V`, `D` and `U^{-1}` in exact Python integer arithmetic.

## Weights and the sum rule — `Unfolding/Weights.lean`

`weight T k K Gs g c` models
`unfoldlab.core.plane_waves.weights_from_coefficients`:

```text
w(k) = (Σ_{G matching k} |c_G|²) / (Σ_G |c_G|²)
```

| Lean | Python | statement |
| --- | --- | --- |
| `weight_nonneg`, `weight_le_one` | `diagnose_weights` | weights lie in `[0, 1]` |
| `weight_of_shift` | `matching_g_mask` | the weight is independent of the representative of `K` |
| `IsFiberRepr.sum_partialNorm` | — | the partial norms over a full fiber add up to the total norm |
| `IsFiberRepr.sum_weight_eq_one` | `diagnose_fiber_sum_rule` | **the sum rule**: for a fixed band, the weights of the `|det T|` primitive k-points of a fiber sum to one |
| `IsFiberRepr.sum_weight_subset_le_one` | `diagnose_fiber_sum_rule` (`max_partial_excess`) | **the partial sum rule**: any subset of a fiber sums to *at most* one, which is what a band path can be checked against |

The sum rule is a statement about a *fiber at fixed band index*.  It is **not**
the quantity `max_band_sum` reported by `diagnose_weights`, which sums over
bands at one k-point and has no reason to equal one; see `docs/findings.md`.

The equality needs a complete fiber, which a band-structure path rarely
samples.  The inequality does not, so an incomplete fiber whose weights already
exceed one is a violation the diagnostic can report on any path.

## Γ-only storage — `Unfolding/GammaOnly.lean` and `Unfolding/HalfSpace.lean`

A Γ-only calculation stores one member of every pair `{G, -G}` and reconstructs
the rest by time reversal, `c_{-G} = conj(c_G)`.  `gammaExpand` and
`gammaCoeff` model that reconstruction, which
`unfoldlab.core.gamma.expand_gamma_half_basis` performs for both the VASP and
the QE reader.

| Lean | Python | statement |
| --- | --- | --- |
| `pwMatches_neg` | `expand_gamma_half_basis` | at Γ the plane wave `-G` matches `-k` exactly when `G` matches `k` |
| `totalNorm_gammaExpand` | `weights_from_coefficients` | the expanded basis has the familiar Γ-only norm `|c₀|² + 2 Σ_{G≠0} |c_G|²` |
| `partialNorm_gammaExpand` | `weights_from_coefficients` | the matched norm splits into the stored half at `k` and the time-reversed half at `-k` |
| `sum_weight_gammaExpand_eq_one` | `tests/test_gamma.py` | the sum rule survives the expansion |
| `exists_half_basis_weight_ne` | `io.vasp_wfc`, `io.qe_wfc` | **counterexample**: unfolding the stored half directly gives a different (wrong) weight, so a reader must expand first |
| `inHalfSpace_neg_iff`, `inHalfSpace_or_neg` | `_gamma_half_mask` | the "first nonzero component is positive" rule keeps exactly one member of every nonzero pair |
| `isHalfBasis_filter_inHalfSpace` | `generate_vasp_g_vectors(gamma_half=...)` | filtering a plane-wave sphere with that rule produces a valid half basis |
| `gammaExpand_filter_inHalfSpace` | `WavecarReader` | filtering and re-expanding returns the sphere unchanged — nothing lost, nothing duplicated |

The axis order is a parameter of the Lean statements, so they cover both
half-space conventions a Γ-only VASP build can use (`gamma_half="x"` and
`"z"`).

## Symmetry-reduced k-point meshes — `Unfolding/Symmetry.lean`

A self-consistent run stores wavefunctions only for the irreducible wedge.  The
weights of the whole star follow without touching the coefficients.

| Lean | Python | statement |
| --- | --- | --- |
| `congrModZ_unimodular` | `supercell_operation` | a unimodular integer matrix acts bijectively on the Brillouin-zone torus |
| `pwMatches_symmetry` | `core.symmetry` | the matching mask is equivariant under `S_pc` / `S_sc = T S_pc T⁻¹` |
| `weight_symmetry` | `map_kpoints_to_stored` | hence so are the weights: `w(S_pc k ; S_sc K) = w(k ; K)` with the **unrotated** stored plane-wave list |
| `pwMatches_congr_primitive`, `weight_congr_primitive` | `allclose_mod1` | the weight only sees the primitive k-point modulo a reciprocal-lattice vector |
| `fold_stored_representative` | `map_kpoints_to_stored` | the rotated primitive k-point `S_pc⁻¹ k` folds exactly onto the stored k-point, so the reader's consistency check passes |
| `weight_stored_representative` | `compute_weights_from_wavecar(operations=...)` | the recipe actually implemented: to get `w(k)`, unfold `S_pc⁻¹ k` against the stored file |

## What symmetry reduction assumes — `Unfolding/SymmetryHypothesis.lean`

`Symmetry.lean` is the geometry; this file is the physics the geometry is
attached to.  The kernel sees the two lattices and never the atoms, so it cannot
tell a symmetry of the crystal from a matrix that merely commutes with the
transform.

| Lean | Python | statement |
| --- | --- | --- |
| `weight_symmetry_of_norm_eq` | `map_kpoints_to_stored` | the recipe with its hypothesis exposed: the weights agree as soon as the star member's coefficients have the *same moduli* as the stored ones against the rotated plane-wave list |
| `exists_weight_ne_of_coefficients` | — | that hypothesis is load-bearing: one geometry, two coefficient vectors, weights `1` and `0` |
| `swapAxes_isUnit`, `swapAxes_comm` | `supercell_operation` | the axis exchange passes every test the code makes — unimodular, and integral in supercell coordinates |
| `exists_latticeSymmetry_bandEnergy_ne` | `validate_primitive_operations` | …and is still not a symmetry of an anisotropic Hamiltonian: the two k-points differ in energy by `2 t`, so the stored wavefunction belongs to another state |
| `zeeman_invariant_iff_moment_invariant` | `site_labels`, `space_group_operations(magmoms=...)` | a collinear magnet is invariant exactly under the operations that preserve the *moments*, not merely the species |
| `exists_species_invariant_moment_not` | `detect_primitive_operations(magmoms=...)` | an antiferromagnetic pair has such an operation, so a symmetry search that ignores the moments returns too large a group |

## Minimum image on the k-point torus — `Unfolding/MinimumImage.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `sqLen_mulVec_le` | — | the Frobenius bound `‖A x‖ ≤ ‖A‖_F ‖x‖` |
| `abs_det_mul_euclidLen_le` | `lattice_growth_constant` | `|det A| ‖x‖ ≤ ‖adj A‖_F ‖A x‖`, so `σ = |det A| / ‖adj A‖_F` is a certified growth constant, computable without an eigenvalue solver |
| `minimum_image_box_exact`, `minimum_image_box_exact_frac` | `minimum_image_distance` | the stopping certificate: once the best distance found in a box of radius `r` satisfies `v ≤ σ (r + 1 - ‖δ‖)`, no translation outside the box can be closer |
| `isLeast_minimum_image` | `minimum_image_distance` | the certified answer really is the minimum |

The search enlarges the box until the certificate holds and raises rather than
returning an uncertified value, so the result is exact rather than heuristic.

## The first Brillouin zone — `Unfolding/BrillouinZone.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `IsBZRepr` | `reduce_to_first_bz` | the predicate: no reciprocal-lattice translate of the k-point is shorter in Cartesian length |
| `isBZRepr_iff_bragg` | `zone_boundary_distance`, `is_in_first_bz` | the zone is the intersection of the Bragg half spaces `\|2 ⟪k, g⟫\| ≤ ‖g‖²`, so membership needs no minimization |
| `isBZRepr_of_two_mul_lt` | `inscribed_radius` | a k-point closer to Γ than half the shortest lattice vector is in the zone for free |
| `sqLen_eq_of_isBZRepr`, `bragg_eq_of_isBZRepr`, `eq_of_isBZRepr_of_strict` | — | uniqueness up to the boundary: two zone representatives of one coset have equal length and both lie on the Bragg plane of their difference |
| `isBZRepr_of_rounded_of_orthogonal` | `has_orthogonal_reciprocal_basis` | for an orthogonal reciprocal basis, componentwise rounding *is* the zone reduction |
| `exists_rounded_not_isBZRepr` | `diagnose_bz_reduction` | and for a skewed one it is not: a rounded representative five times longer in squared length |
| `isBZRepr_of_box_certificate` | `reduce_to_first_bz`, `certified_minimum_image_shift` | the certified growing-box search returns a genuine zone representative |
| `foldsTo_add_ivec` | — | zone reduction moves a k-point by a lattice vector, so it changes no fiber, weight or sum rule |

## Moiré cells — `Unfolding/Moire.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `det_sub_eq_zero_of_row_eq` | — | matrices agreeing in one row have a singular difference |
| `moire_reciprocal_singular_of_shared_axis` | `build_moire_reciprocal_lattice` | **the full `B_a - B_b` of a layered stack is always singular**, because the layers share their stacking vector; the real-space moiré cell must come from the transverse block |
| `reciprocalLattice_moire` | `TwistedUnfoldingProblem.moire_lattice` | inverting a nonsingular moiré reciprocal lattice returns a cell with exactly that reciprocal lattice |

## Numerical tolerances — `Unfolding/Numerics.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `congrModZ_iff_wrapFrac_eq` | `wrap_fractional` | two k-points are equal modulo `ℤ^n` iff their wrapped representatives in `[0,1)` coincide — the justification for comparing k-points after wrapping |
| `roundTest_of_half_le` | `check_atol`, `MAX_MEANINGFUL_ATOL` | for `atol ≥ 1/2` the test `|round(x) - x| ≤ atol` is **vacuously true for every `x`**, so such a tolerance silently accepts everything.  `check_atol` now rejects it |
| `roundTest_iff` | — | for `atol < 1/2` the test holds iff `x` is within `atol` of an integer |
| `roundTest_zero_iff` | — | at `atol = 0` the test is exactly integrality |

## Broadening — `Unfolding/Spectral.lean`

`spectralFunction b w e E = Σ_j w_j · kernel(E - e_j)` models
`unfoldlab.core.spectral.spectral_function`.

| Lean | Python | statement |
| --- | --- | --- |
| `integral_gaussianKernel` | `spectral_function(kernel="gaussian")` | the Gaussian kernel integrates to 1 |
| `integral_lorentzianKernel` | `spectral_function(kernel="lorentzian")` | the Lorentzian kernel integrates to 1 |
| `integral_spectralFunction` | `EffectiveBandStructure` | `∫ A(E) dE = Σ_j w_j`: broadening conserves total spectral weight, for any normalized kernel and any broadening width |
| `integral_adaptiveSpectralFunction` | `spectral_function(broadening=widths)`, `adaptive_widths` | the same identity when *every state carries its own width*: adaptive broadening is a broadening, not a reweighting |
| `adaptiveSpectralFunction_const` | — | a fixed width is the special case of an adaptive one |

`tests/test_spectral.py` checks the last identities numerically for both kernels.

## Spin-resolved unfolding — `Unfolding/Spin.lean`

For a noncollinear (spinor) state each plane wave carries two coefficients, and
`spinVec a b = (2 Re(ā b), 2 Im(ā b), ‖a‖² − ‖b‖²)` is its Bloch vector.  The
unfolded texture `spinTexture` sums those vectors over the matching plane waves
and normalizes by the norm of the state; it models
`unfoldlab.core.plane_waves.spin_texture_from_coefficients`.

| Lean | Python | statement |
| --- | --- | --- |
| `norm_spinVec` | — | a single plane-wave spinor is a *pure* state: the length of its Bloch vector is exactly its norm |
| `spinPartial_apply_two` | `weights_from_coefficients(component_resolved=True)` | the `z` component of the texture is the difference of the two component-resolved weights — the consistency check between the two code paths |
| `norm_spinTexture_le_pairWeight` | `diagnose_spin_textures` | `‖S(k)‖ ≤ w(k)`: the texture never exceeds the spectral weight, with equality exactly when every matching plane wave carries the same spin direction |
| `norm_spinTexture_le_one` | — | hence `‖S(k)‖ ≤ 1` |
| `spinPartial_of_shift` | `spin_texture_from_coefficients` | the texture depends only on the physical plane waves `K + G`, not on the k-point representative |
| `IsFiberRepr.sum_spinTexture` | `spin_expectation_from_coefficients` | **spin sum rule**: the textures of the `|det T|` primitive k-points of a fiber add up to the spin expectation value of the supercell state — a vector identity, so it constrains the transverse components too |

The transverse components are the reason this is not just a relabelling of the
component-resolved weights: `|c_up|²` and `|c_down|²` fix only `S_z`, while `S_x`
and `S_y` come from the off-diagonal product `conj(c_up) c_down`.

A point-group operation leaves the *weight* invariant
(`weight_symmetry`), which is what makes unfolding from an irreducible wedge
legitimate, but the spin is a pseudovector and rotates with the operation.  The
readers therefore refuse to produce a texture in symmetry-reduced mode.

## Supercell detection — `Unfolding/Supercell.lean`

| Lean | Python | statement |
| --- | --- | --- |
| `transform_unique` | `SupercellTransformation.from_lattices` | the integer matrix relating a primitive and a supercell lattice is unique |
| `abs_det_supercell` | `check_supercell_consistency` | `|det A_sc| = |det T| · |det A_pc|`, the volume ratio |
| `card_fiber_eq_volume_ratio` | `check_supercell_consistency` | the number of primitive k-points folding onto one supercell k-point equals the cell-volume ratio, hence also the expected site-count ratio |

## Counterexamples — `Unfolding/Pitfalls.lean`

Three negative results, each recording a bug that was found in the numerical
code (see `docs/findings.md`).

| Lean | Python | statement |
| --- | --- | --- |
| `exists_pwMatches_representative_mismatch` | `io.vasp_wfc`, `io.qe_wfc` | keeping the `G` list while changing the representative of `K` changes the matched set: the readers must use the k-point stored in the wavefunction file |
| `exists_shorter_image_than_reduced` | `minimum_image_distance` | for a hexagonal reciprocal lattice, componentwise rounding does not give the minimum image |
| `exists_equal_fractional_distinct_cartesian` | `interpolate_path` | equal fractional norms can correspond to different physical distances |

## Projected unfolding — `Unfolding/Projection.lean`

`projectedWeight T k K Gs g c P p S = maskFraction P p S * weight T k K Gs g c`
models `unfoldlab.workflows.projected`: the plane-wave weight times the fraction
of the band's site/orbital character (`P` is the whole `(site, orbital)` table,
`p` its entries, `S` the selection) that the selectors keep.

| Lean | Python | statement |
| --- | --- | --- |
| `maskFraction_nonneg`, `maskFraction_le_one` | `projection_fractions` | a fraction lies in `[0, 1]`, including the degenerate empty-table case where the convention `f = 0` is what preserves the bound |
| `maskFraction_union_of_disjoint` | `resolve_site_orbital_mask` | fractions are additive over disjoint selections |
| `sum_maskFraction_of_partition` | `projection_fractions` | the fractions of a partition of the table sum to one |
| `projectedWeight_le`, `projectedWeight_nonneg` | `apply_projection_fractions` | a projection never creates spectral weight |
| `sum_projectedWeight_of_partition` | `apply_projection_fractions` | summing over a partition of the table returns the plain weight: a projection only redistributes |
| `IsFiberRepr.sum_projectedWeight` | `diagnose_fiber_sum_rule` | for one group the fiber sum is the group's fraction, not one — that is the number a projected sum-rule check must compare against |
| `sum_fiber_projected_partition` | — | summing over the fiber *and* over a full partition gives one: the sum rule survives projection |
| `exists_partial_normalization_sum_ne_one` | `projection_fractions` | normalizing by a partial total (a `tot` column that omits part of the table) breaks the partition identity, so the code always divides by the sum over the whole table |
| `exists_signed_table_fraction_gt_one` | `projection_fractions` | a signed table (a magnetization column) can give a "fraction" above one, so such tables are rejected |

What the Lean file deliberately does *not* claim is that the projected weight is
the true `k`-resolved projected spectral function: the character tables are not
resolved in the primitive k-point, so the same fraction multiplies every k-point
of a fiber.  That approximation is documented on the Python side.

## The point group of a lattice — `Unfolding/PointGroup.lean`

In the fractional (column) convention an integer matrix `M` is a symmetry of the
lattice exactly when it preserves the metric tensor `G = A Aᵀ`, that is
`Mᵀ G M = G` (`MetricPreserving`).  `unfoldlab.core.spacegroup` detects these by
enumerating candidate columns in a finite box; the file proves that box is wide
enough.

| Lean | Python | statement |
| --- | --- | --- |
| `biForm_sq_le` | — | Cauchy–Schwarz for the positive semidefinite form `vᵀ G v` |
| `sq_component_le_of_quadForm` | `_vectors_of_norm` | `(v k)² ≤ (G⁻¹)ₖₖ · vᵀ G v`: the search box for lattice vectors of a given length |
| `quadForm_col_of_metricPreserving` | `lattice_point_group` | column `i` of a lattice symmetry has exactly the length of basis vector `i` |
| `sq_entry_le_of_metricPreserving` | `lattice_point_group` | consequently `(M k i)² ≤ (G⁻¹)ₖₖ Gᵢᵢ`, so the enumeration is exhaustive and no operation is missed |
| `MetricPreserving.one`, `.mul`, `.inv` | `test_point_group_is_closed_under_multiplication_and_inverse` | the operations form a group |
| `MetricPreserving.det_sq_eq_one`, `intMetricPreserving_det_eq_one_or_neg_one` | `reciprocal_operation` | a lattice symmetry is unimodular, which is why `(M⁻¹)ᵀ` is again integral |
| `metricPreserving_reciprocal` | `reciprocal_operation` | the reciprocal-space form `S = (M⁻¹)ᵀ` preserves the reciprocal metric `G⁻¹` |

The decoration of the cell (which site carries which species) is not modelled in
Lean: `space_group_operations` filters the lattice symmetries by an explicit
site-permutation check, which is a finite verification rather than a theorem.

## QE `xk`, `alat` and units — `Unfolding/Units.lean`

A Quantum ESPRESSO wavefunction file records its k-point as `xk`: a *Cartesian*
vector in units of `2π/alat`.  `unfoldlab.io.qe_wfc.fractional_kpoint_from_xk`
turns it into fractional reciprocal coordinates with QE's `at` matrix (the
direct lattice in units of `alat`), and
`unfoldlab.io.qe_xml.read_qe_xml` now reads `at` out of the calculation's own
`data-file-schema.xml` instead of asking the user for it.

| Lean | Python | statement |
| --- | --- | --- |
| `xkToFrac_of_reciprocal` | `fractional_kpoint_from_xk` | the conversion is correct: with `bg` dual to `at`, a k-point written as `xk = f @ bg` comes back as `f` |
| `latticeAlat_unit_invariant`, `xkToFrac_unit_invariant` | `read_qe_xml` | `at = A / alat` is dimensionless, so reading the cell and `alat` from the XML in bohr gives the same matrix as any other length unit |
| `xkToFrac_injective` | `check_xml_kpoint_order` | for an invertible cell distinct `xk` values are distinct k-points, so a mismatch is real |
| `xkToFrac_congrModZ_iff` | `compute_weights_from_qe_save` | two `xk` values are the same k-point of the torus exactly when they differ by a reciprocal-lattice vector — the precise content of the "same k-point mod 1" cross-check |

## Tight-binding, Wannier and model-Hamiltonian unfolding — `Unfolding/TightBinding.lean`

A model-Hamiltonian supercell state is not a list of plane-wave coefficients:
it carries one amplitude per *(orbital, primitive cell)* pair, and its
unfolding weight is the squared norm of its discrete Fourier component,
`A(k, a) = Σ_R exp(-2π i ⟪k, R⟫) c(a, R)`.  The sum rule is therefore a
different theorem from the plane-wave one — it is discrete Fourier duality
between the cell classes `ℤ³ / ℤ³ @ T` inside the supercell and the reciprocal
classes `ℤ³ / T·ℤ³` that label the fiber.  `unfoldlab.core.tight_binding`
implements it and `unfoldlab model unfold` exposes it on the command line.

| Lean | Python | statement |
| --- | --- | --- |
| `dualPairing`, `dualChar` | `tight_binding_weights` | the pairing `⟪m, R⟫ = (T⁻¹ m) ⬝ R` between the two quotients and its character |
| `sum_dualChar` | `supercell_cells`, `fiber_kpoints` | **character orthogonality**: summed over a transversal of `ℤⁿ/T·ℤⁿ` the character gives `\|det T\|` on the trivial cell class and `0` on every other, so the two finite groups are dual — this is why the recipe is exact for a *non-diagonal* supercell |
| `sum_phase_pair` | `tight_binding_weights` | two different cells of the supercell are orthogonal after summing over the fiber |
| `sum_phase_pair_cells` | `tight_binding_weights` | two different fiber members are orthogonal after summing over the cells: the unfolding transform is unitary |
| `sum_normSq_tbAmplitude` | `unfold_tight_binding` | **Parseval**: the fiber sum of `Σ_a \|A(k, a)\|²` is `\|det T\|` times the squared norm of the coefficients |
| `sum_tbWeight_eq_one` | `diagnose_tight_binding_weights` (`max_fiber_deviation`) | the weights of one state over one fiber add up to one |
| `sum_tbWeight_states` | `diagnose_tight_binding_weights` (`max_band_deviation`) | at a fixed primitive k-point the weights of a complete orthonormal set of supercell states add up to the number of orbitals per primitive cell |
| `tbWeight_nonneg`, `tbWeight_le_one` | `tight_binding_weights` | the weights are probabilities |
| `tbAmplitude_congr_left` | `tight_binding_weights` | the weights do not depend on how the primitive k-points are wrapped |
| `tbWeight_orbital_phase` | `tight_binding_weights` | a per-orbital phase — i.e. the convention for where a Wannier function sits inside the cell — leaves every weight unchanged |
| `blochState`, `tbWeight_blochState` | `unfold_tight_binding_path`, `unfoldlab model bands` | **band folding undone**: a supercell state that is a primitive Bloch state unfolds to weight exactly one at its own fiber member and exactly zero at all the others |
| `tbOrbitalWeight` | `tight_binding_orbital_weights` | the fat-band weight: the unfolding weight restricted to a group of primitive-cell orbitals, normalized by the *full* state norm |
| `tbOrbitalWeight_union_of_disjoint` | `unfold_tight_binding_path_projected` | disjoint orbital groups add, so a partition of the orbitals reproduces the total weight |
| `tbOrbitalWeight_le_tbWeight`, `tbOrbitalWeight_mono`, `tbOrbitalWeight_nonneg` | `tight_binding_orbital_weights` | a group weight lies between zero and the total, and grows with the group |
| `sum_tbOrbitalWeight_eq_norm_share` | `tight_binding_orbital_weights` | over a complete fiber a group's weights add up to the group's share of the state norm, which is one for the full orbital set |

The hypothesis `IsCosetRepr Tᵀ Rs` — the cells are a complete, irredundant list
of the primitive cells of the supercell — is checked at runtime by
`validate_supercell_cells`, using the same exact integer adjugate criterion as
the plane-wave matching test.

## Blocked / chunked evaluation — `Unfolding/Chunking.lean`

Large wavefunction files are read a band block at a time, and in principle a
plane-wave block at a time, so the weights are *accumulated* rather than
computed in one pass.  These statements say that the accumulation is exact:
splitting the work changes nothing but the peak memory.

| Lean | Python | statement |
| --- | --- | --- |
| `partialNorm_biUnion`, `totalNorm_biUnion` | `compute_plane_wave_unfolding_weights_chunked` | the matched and the total norm are additive over a disjoint partition of the plane-wave basis |
| `weight_biUnion` | `compute_plane_wave_unfolding_weights_chunked` | **plane-wave chunking is exact**: accumulating the two norms over the blocks and dividing at the end reproduces the weight of the whole basis, including the `0/0 ↦ 0` convention |
| `weight_append`, `weight_of_accumulate` | `compute_plane_wave_unfolding_weights_chunked` | the two-block and the fold-over-a-list forms of the same identity, matching the shape of the Python accumulator |
| `bandWeights_flatten`, `bandWeights_flatten_length` | `compute_weights_from_wavecar(..., band_chunk=m)`, `compute_weights_from_qe_save(..., band_chunk=m)`, CLI `--band-chunk` | **band chunking is exact**: weights computed per band block and concatenated equal the single-pass weight table, and the table has the expected length |
| `spinPartial_biUnion`, `pairPartial_biUnion`, `pairNorm_biUnion` | `read_kpoint_wavefunction(..., band_start=, band_count=)` on noncollinear files | the spinor components and the spin-texture cross terms are additive over blocks too |

Numerically the chunked and unchunked paths agree to rounding, not bit for bit:
summing in blocks reassociates the floating-point additions.
`tests/test_band_chunking.py` asserts agreement to `1e-12` and checks that only
one block is resident at a time.

## The ideal supercell — `Unfolding/Ideal.lean`

The benchmark case behind `docs/benchmarks.md`: a supercell state whose plane
waves *all* match one primitive k-point of the fiber.  This is what an
unperturbed supercell calculation produces, and it is the case where the
unfolded band structure must be exactly the primitive one.

| Lean | Python | statement |
| --- | --- | --- |
| `CarriedBy` | `tests/test_benchmarks.py` | the hypothesis: every plane wave with a nonzero coefficient matches the primitive k-point `k₀` |
| `partialNorm_eq_totalNorm_of_carriedBy` | `compute_plane_wave_unfolding_weights` | under that hypothesis the matched norm is the whole norm |
| `weight_eq_one_of_supported` | `compute_plane_wave_unfolding_weights` | so the weight at `k₀` is exactly one |
| `weight_eq_zero_of_not_matching` | `compute_plane_wave_unfolding_weights` | at a fiber member distinct from `k₀` no plane wave matches, so the weight is exactly zero |
| `IsFiberRepr.weight_ideal` | `unfold_wavefunction`, `tests/test_benchmarks.py` | **folding undone**: over the whole fiber the weight is `1` at `k₀` and `0` everywhere else |

A caveat the numerical benchmarks have to respect: when a fiber member carries
*degenerate* supercell states the split of the weight between the individual
eigenvectors is basis dependent, so the tests assert the total weight and the
energies that carry it, never "exactly one eigenvector".

## Configurational averaging — `Unfolding/Ensemble.lean`

A disordered supercell — an alloy snapshot, a special quasirandom structure, an
ensemble of inequivalent defect placements — has no single band structure.  Each
configuration is unfolded separately and the results are averaged with the
probability of each configuration.  `unfoldlab.core.ensemble` implements the
average and the `unfoldlab ensemble` command exposes it.

| Lean | Python | statement |
| --- | --- | --- |
| `mixture` | `stack_configurations` | the configurational average `⟨A⟩ = Σ_i p_i A_i` |
| `mixture_nonneg`, `mixture_le` | `ensemble_spectral_function` | the average is non-negative, and bounded by whatever bounds every configuration |
| `integral_mixture_spectralFunction` | `stack_configurations` | **averaging conserves spectral weight**: the averaged map integrates to the average of the configurations' total weights |
| `integral_mixture_spectralFunction_one` | `unfoldlab ensemble` | so with the unfolding sum rule holding configuration by configuration, it holds for the average |
| `specCentre`, `specSpread` | `band_moments` | the centre and the squared width of a weighted spectrum at one k-point |
| `sum_weighted_deviation`, `sum_weighted_sq_shift` | `band_moments` | a normalized spectrum has no first moment about its own centre (parallel-axis theorem) |
| `ensembleSpread_decomposition` | `disorder_broadening` | **law of total variance**: the squared width of the averaged band is the average of the configurations' own squared widths plus the variance of their centres — the second term is the disorder broadening |
| `le_ensembleSpread` | `DisorderBroadening.disorder_width` | disorder never sharpens the average |
| `ensembleSpread_const` | `disorder_broadening` | an ensemble of identical configurations has zero disorder broadening |

The decomposition assumes each configuration carries the same total weight at
the k-point, which the unfolding sum rule guarantees; the Python side does not
assume it but reports the residual of the identity through
`DisorderBroadening.residual`.

## Energy windows and constant-energy cuts — `Unfolding/Windows.lean`

`unfoldlab.core.windows` answers the questions that follow a heat map: what is
`A(k, E₀)` along the path (the unfolded Fermi surface, or an ARPES-like cut),
and how much unfolded weight lies between two energies (the occupied weight, or
the weight a broadened plot leaks into a gap).  Both are evaluated in closed
form from the discrete spectrum — an error function for the Gaussian kernel, an
arctangent for the Lorentzian — so neither depends on an energy grid.

| Lean | Python | statement |
| --- | --- | --- |
| `windowWeight` | `energy_window_weight` | the unfolded weight of a k-point inside `[lo, hi]` |
| `windowWeight_add_adjacent`, `windowWeight_split_at` | `band_filling` | **adjacent windows add**, so the occupied and the empty part add back up to the sum rule |
| `windowWeight_nonneg` | `energy_window_weight` | a window carries non-negative weight |
| `windowWeight_le_total` | `energy_window_weight` | and never more than all of the k-point's weight |
| `windowWeight_mono` | `energy_window_weight` | widening a window cannot lose weight |
| `windowWeight_self` | `energy_window_weight` | an empty window carries none |
| `integrable_spectralFunction` (in `Spectral.lean`) | — | every window is well defined, because each kernel is integrable |

`constant_energy_cut` is one column of the spectral function evaluated directly
rather than interpolated off a grid; `Spectral.lean`'s `spectralFunction_nonneg`
and `spectralFunction_le` already bound it.

## Degenerate multiplets and gauge invariance — `Unfolding/Degeneracy.lean`

A diagonalizer returns an arbitrary orthonormal basis of each eigenspace, so the
unfolded weight of an *individual* band inside a degenerate multiplet is not a
physical number: a unitary mixing of the multiplet redistributes it.  What is
physical is the weight of the whole multiplet.  `unfoldlab.core.degeneracy`
groups the bands by energy and reports it, and `unfoldlab degeneracy` exposes
that from the command line.

| Lean | Python | statement |
| --- | --- | --- |
| `mixStates` | — | the gauge freedom: the members of the multiplet mixed by a unitary matrix |
| `sum_normSq_mulVec_of_unitary` | — | a unitary matrix preserves the intensity the multiplet puts on each plane wave |
| `subspaceWeight` | `subspace_weights` | the unfolded weight of a whole multiplet |
| `subspacePartialNorm_mix`, `subspaceTotalNorm_mix` | — | matched and total norm are gauge invariant |
| `subspaceWeight_mix` | `degeneracy_averaged_weights` | **gauge invariance**: the multiplet weight does not depend on the basis inside the multiplet |
| `subspaceWeight_nonneg`, `subspaceWeight_le_one` | `degeneracy_report` | it still lies in `[0, 1]` |
| `IsFiberRepr.sum_subspaceWeight_eq_one` | `degeneracy_averaged_weights` | and still obeys the fiber sum rule, so grouping costs no diagnostic power |
| `subspaceWeight_eq_average_of_normalized` | `degeneracy_averaged_weights` | for normalized states it is the average of the member weights — which is what the code writes back into each band |

`DegeneracyReport.max_spread` is the numerical companion: the largest deviation
of a per-band weight from its multiplet average, i.e. how much of a reported
per-band weight is gauge artefact.

## Time reversal — `Unfolding/TimeReversal.lean`

A plane-wave code reduces its k-point mesh by `k ↦ -k` for any nonmagnetic
system, centrosymmetric or not, because `ψ_{-K} = conj ψ_K`.  The spatial point
group detected from the structure contains `-1` only for a centrosymmetric
crystal, so `detect_primitive_operations` needs to adjoin time reversal
explicitly; `time_reversal=True` (CLI `detect-symmetry --time-reversal`) does.

| Lean | Python | statement |
| --- | --- | --- |
| `congrModZ_neg` | — | negation is a bijection of the Brillouin-zone torus |
| `pwMatches_timeReversal` | `matching_g_mask` | negating `k`, `K` and `G` together leaves the matching condition unchanged (`GammaOnly.pwMatches_neg` is its `K = 0` case) |
| `foldsTo_neg` | `_folding_offset` | `-k` folds onto `-K`, so the reader's consistency check still passes |
| `weight_timeReversal` | `weights_from_coefficients` | the time-reversed state carries the same weight: conjugation is invisible to `‖c‖` |
| `weight_timeReversal_stored` | `map_kpoints_to_stored`, `detect_primitive_operations` | **the recipe**: to unfold `k` against the time-reversal partner of a stored file, unfold `-k` against the file as stored |

The caveat is the same one that already applies to point-group reduction, and
for the same reason: a spin texture is odd under time reversal, so it must not
be reconstructed this way.  A magnetic calculation is not time-reversal
symmetric at all and the flag must not be used for one.

## Normalization, truncation and perturbation — `Unfolding/Normalization.lean`

The weight is the ratio of the matched squared norm to the total squared norm of
the coefficients *the file contains*.  This module says what that does and does
not depend on.

| Lean | Python | statement |
| --- | --- | --- |
| `partialNorm_smul`, `totalNorm_smul` | — | both norms scale by `‖z‖²` |
| `weight_smul` | `weights_from_coefficients` | **scale invariance**: only the ray of the state matters, so any normalization convention gives the same weights |
| `partialNorm_union_sdiff`, `totalNorm_union_sdiff` | — | the norms split over a subset and its complement |
| `weight_truncation_bound` | `truncation_weight_error_bound`, `StateNormDiagnostics.weight_error_bound` | **truncation bound**: a missing norm fraction `δ` moves any weight by at most `δ / (1 - δ)` |
| `abs_sub_normSq_le` | — | `\|‖a‖² - ‖b‖²\| ≤ ‖a - b‖ (‖a‖ + ‖b‖)`, the per-component step |
| `weight_perturbation_bound` | — | **stability**: normalized states at `ℓ²` distance `ε` give weights within `2 ε` |

The truncation bound is what makes the stored-norm diagnostic actionable: a PAW
or ultrasoft run stores only the pseudo part of each state, and the shortfall of
its norm below one is precisely the `δ` of the bound.
`unfoldlab.io.vasp_wfc.state_norms_from_wavecar`,
`unfoldlab.io.qe_wfc.state_norms_from_qe_save` and the `unfoldlab norms` command
measure it.

## Phonon masses and isotope defects — `Unfolding/Phonons.lean`

Phonon unfolding reuses the tight-binding kernel on the *mass-weighted* force
constants, so `TightBinding.lean` already covers the weights and the sum rules.
What it does not cover is the masses, and there the code had a real gap: the
only supercell defect it could express was an additive on-site term, which is a
force-constant defect, never a mass defect.

| Lean | Python | statement |
| --- | --- | --- |
| `dynMatrix` | `ForceConstantModel.mass_weighted_model`, `supercell_dynamical_matrix` | the dynamical matrix `D_ab = Φ_ab / √(m_a m_b)` |
| `massDisplacement` | — | the displacement pattern `u = M^{-1/2} e` of an eigenvector `e` |
| `dynMatrix_isHermitian` | `supercell_dynamical_matrix` | mass weighting — including a site-dependent one — preserves Hermiticity, so `eigh` stays legitimate |
| `dynMatrix_mulVec_iff` | `unfold_phonon_path`, `unfold_phonon_fiber` | `D e = ω² e` *is* the generalized problem `Φ u = ω² M u`: diagonalizing the mass-weighted matrix solves the physical one |
| `dynMatrix_massScale` | `mass_site_scaling`, `_apply_site_scaling` | **the formula**: changing the masses by a factor `s` is the congruence `D ↦ S D S`, `S = diag(1/√s)` |
| `mass_change_not_diagonal_shift` | `supercell_dynamical_matrix` | **the bug it fixes**: no on-site term reproduces a mass defect once the substituted site is coupled to anything, because an off-diagonal entry moves |
| `acoustic_zero_mode` | `ForceConstantModel.acoustic_sum_rule_deviation` | with the acoustic sum rule satisfied, a rigid translation is an exact zero mode *for any masses*, so an acoustic branch that misses zero at Γ always indicts the force constants |
| `massDisplacement_inner` | `unfold_phonon_*` | displacements are orthonormal only in the mass-weighted inner product, which is why the *eigenvector* is what gets unfolded |
| `massDisplacement_not_normalized` | — | the counterexample: unfolding the displacement instead would break the sum rules by a factor `1/m` |

`supercell_site_masses` builds the `(|det T|, n_atoms)` mass table from
`(cell, atom, mass)` substitutions, the JSON reader takes a `site_masses` block,
and `unfoldlab phonon bands --site-mass cell,atom,mass` exposes it.  Because the
congruence is real and diagonal, the eigenvectors stay orthonormal and both sum
rules survive unchanged — which the tests check numerically.

## Non-orthogonal (LCAO) bases — `Unfolding/Overlap.lean`

`TightBinding.lean` assumes the orbitals are orthonormal, which is true of
Wannier functions and of a model tight-binding basis and false of every LCAO
code (SIESTA, CP2K, OpenMX, FHI-aims), where the eigenproblem is the generalized
one `H c = E S c` and `⟨ψ|ψ⟩ = c† S c` rather than `∑ |c|²`.  This module carries
the overlap through the whole derivation.

The object it adds is the **Bloch-twisted overlap kernel** at the supercell
k-point `K`, `S α β D = ∑_v e^{2πi K·v} S^∞ α β (D + Tᵀv)`, which obeys
`S α β (D + Tᵀv) = e^{-2πi K·v} S α β D` (`IsBlochOverlap`) and is therefore
determined by its values on the `|det T|` cells of the supercell.

| Lean | Python | statement |
| --- | --- | --- |
| `IsBlochOverlap` | `bloch_overlap_kernel`, `supercell_overlap_matrix` | the twisted periodicity that lets a kernel stored on one transversal be evaluated at any cell separation |
| `IsOrthonormalOn` | `orthonormal_overlap_kernel` | the orthonormal special case, `S α β W = δ_{αβ} δ_{W,0}` |
| `ovKernel` | `reciprocal_overlap` | the reciprocal-space overlap `S̃(k)_{αβ}` |
| `ovNorm` | `lcao_norms` | the true squared norm `c† S c` |
| `ovNumer`, `ovWeight` | `lcao_weights`, `unfold_lcao` | the weight `A(k)† S̃(k) A(k) / (\|det T\| c† S c)` |
| `sum_ovNumer` | — | **Parseval with an overlap**: the numerators of one fiber add up to `\|det T\|` times `c† S c` |
| `sum_ovWeight_eq_one` | `unfold_lcao`, `unfold_lcao_model` | consequently the fiber weights add up to one for every state of nonzero norm |
| `ovWeight_orthonormal` | `orthonormal_overlap_kernel` in `tests/test_lcao.py` | with an orthonormal kernel the weight is *exactly* `tbWeight`, so the new formula is a strict generalization |
| `ovNumer_sub_tbNumer` | `diagnose_overlap_neglect` | the discrepancy is `∑_{αβ} conj(A_α) A_β (S̃_{αβ} - δ_{αβ})`: first order in `S̃ - 1`, and not cancelling between numerator and denominator |
| `ovWeight_smul` | `lcao_weights` | scale invariance, as in the orthonormal case |

`solve_generalized_eigenproblem` solves `H c = E S c` by Cholesky (numpy only,
no new dependency) and `unfold_lcao_model` runs the whole pipeline from a
primitive Hamiltonian and a primitive overlap table.

## Completeness with an overlap — `Unfolding/OverlapCompleteness.lean`

`Overlap.lean` proves the sum rule *across the fiber* (one state, the `|det T|`
primitive k-points it can unfold onto).  This module proves the complementary
sum rule *across the bands*: at one primitive k-point, a complete set of
supercell states.

In an orthonormal basis that is `sum_tbWeight_states`, and its proof only needs
the completeness relation `∑_s c_s c_s† = 1`.  In a non-orthogonal basis the
eigenvectors of `H c = E S c` are orthonormal *in the metric `S`*, so
completeness reads `∑_s c_s c_s† = S⁻¹` — a different matrix at every k-point —
and it is not obvious that the total is unchanged.  It is, and the reason is a
convolution theorem: the reciprocal-space overlap of the inverse kernel is the
inverse of the reciprocal-space overlap, so the two factors cancel inside the
trace.

| Lean | Python | statement |
| --- | --- | --- |
| `ovTerm`, `ovTerm_congr` | — | the summand of `ovKernel` is a class function of the cell: the Bloch twist of the kernel is exactly cancelled by the phase |
| `sum_ovTerm_shift`, `sum_ovTerm_reflect` | — | consequently its sum over a transversal survives translating and reflecting the argument, which is what turns a double sum over two cells into a sum over their difference |
| `ovKernel_mul_eq_convolution` | — | the product of two reciprocal-space overlaps as a single twisted convolution |
| `ovKernel_mul_of_inverse` | `numpy.linalg.inv` of `reciprocal_overlap` | **the convolution theorem**: if `G` inverts `S` in real space then `G̃(k)` inverts `S̃(k)` at every k-point |
| `sum_states_amplitude_product` | — | the state sum of the amplitude products is `\|det T\|` times `G̃(k)`, i.e. where completeness enters |
| `sum_ovWeight_states`, `sum_ovWeight_states_of_inverse` | `tests/test_lcao.py::test_band_sum_rule_survives_the_overlap` | **the across-bands sum rule**: the overlap-aware weights of a complete `S`-orthonormal set add up to the number of orbitals per primitive cell, exactly as in the orthonormal case |

`sum_ovWeight_states` takes the reciprocal-space hypothesis (`S̃(k) G̃(k) = 1`)
and `sum_ovWeight_states_of_inverse` the real-space one it follows from, so a
caller may supply whichever it has.

## Bloch-sum conventions — `Unfolding/Convention.lean`

A coefficient table `c[a, R]` only means something once the Bloch-sum convention
is fixed, and three are in circulation: the phase on the cell alone (what this
package assumes), the phase on the cell *and* the orbital position (Wannier90's
and several LCAO codes'), and the amplitudes of the full wavefunction rather
than of its supercell-periodic part.  The first two differ by a diagonal phase
that is constant over the cells; the third differs by one that is constant over
the orbitals.  Only one of the two matters.

| Lean | Python | statement |
| --- | --- | --- |
| `gaugeOrbital`, `gaugeKernel` | `apply_orbital_phase`, `gauge_overlap_kernel` | a per-orbital phase, and the matching `S ↦ diag(u) S diag(u)†` on the overlap kernel |
| `gaugeShift`, `gaugeCell` | `apply_cell_phase`, `cell_phase_factors` | a per-cell phase, and its special case where the shift is a difference of fiber members |
| `tbWeight_gaugeOrbital` | `tests/test_convention.py` | **the atomic-versus-cell convention changes nothing**: the weight is exactly invariant |
| `tbOrbitalWeight_gaugeOrbital` | `tests/test_convention.py` | and neither does any fat band |
| `tbWeight_gaugeShift` | `diagnose_bloch_convention` | **a per-cell phase translates the whole distribution in k**, reporting the weight of `k + x` at `k` — with the fiber sum rule intact, so no runtime check sees it |
| `tbWeight_gaugeCell` | `fiber_shift_permutation` | when the shift is a difference of fiber members the translation is a permutation of the fiber |
| `IsOrthonormalOn.gaugeKernel` | `gauge_overlap_kernel` | an orthonormal kernel is regauged to itself, which is *why* the per-orbital phase is invisible in an orthonormal basis |
| `ovWeight_gauge` | `tests/test_convention.py` | in a non-orthogonal basis the per-orbital phase is a gauge **only** if the kernel is regauged with the state |
| `ovNumer_gaugeOrbital_untransformed` | `gauge_overlap_kernel` | the exact expression for the numerator one gets by forgetting to |

`tbAmplitudeAt` and `tbWeightAt` are the amplitude and weight at an *arbitrary*
k-point, which is what the translated distribution lives on: the shifted fiber
is generally not a fiber.

## Strained and approximately commensurate supercells — `Unfolding/Strain.lean`

Everything above assumes an exact `A_sc = T A_pc`.  Relaxed defect cells,
strained heterostructure layers and commensurate approximants of twisted stacks
miss it by a little, and the package used to answer only "no".  These
statements say what the mismatch costs.

| Lean | Python | statement |
| --- | --- | --- |
| `latticeSpan`, `mem_latticeSpan` | — | the lattice spanned over ℤ by the rows of a cell matrix |
| `latticeSpan_subset_iff_intMatrix` | `detect_transformation`, `diagnose_commensurability` | **commensurability is integrality**: the supercell translations are primitive translations exactly when `M = A_sc A_pc⁻¹` is an integer matrix — there is no approximate version, only a quantified failure |
| `det_ratio_eq_det` | `check_supercell_consistency` | the cell-volume ratio is `det M`, so a non-integral ratio already rules out an integer transform |
| `reciprocalLattice_deform` | `deformation_gradient` | deforming a lattice by the Cartesian gradient `D` (rows `A ↦ A Dᵀ`) multiplies its reciprocal lattice by `D⁻¹` |
| `cartesian_strain_shift` | `kpoint_shift` | the Cartesian point a strained supercell k-point really denotes is the reported one times `D⁻¹` |
| `strain_kshift` | `kpoint_shift`, `tests/test_strain.py` | **the k error is linear in k**: it is `q (1 − D⁻¹)`, zero at Γ and largest at the zone boundary, so a strained reference stretches a band path rather than shifting it |
| `strain_kshift_bound` | `CommensurabilityReport.max_kpoint_shift` | an entrywise bound on that error, which is why sampling the corners of the fractional cube bounds it over the zone |
| `strain_kshift_eq_zero_iff` | `CommensurabilityReport.summary` | **no integer transform repairs a strained reference**: the assignment is right for every k only when `D = 1` |
| `abs_sub_round_le_half`, `round_matrix_optimal` | `nearest_integer_transformation` | entrywise rounding is the best integer approximation and its residual never exceeds `1/2`, so a residual near `1/2` means the transform is genuinely ambiguous (`CommensurabilityReport.ambiguous`) |
| `commensurate_of_round_exact` | `detect_transformation` | conversely, an exact rounding certifies commensurability |

## Reading a dispersion off a spectral function — `Unfolding/Dispersion.lean`

An unfolded band structure is `A(k, E)`, not a set of bands.  Every number one
quotes from it — a band energy, a gap, an effective mass — comes from locating
peaks and differentiating their positions, and each step has a systematic error.

| Lean | Python | statement |
| --- | --- | --- |
| `parabolicVertex` | `parabolic_vertex` | the three-point vertex estimate used to refine a grid peak |
| `parabolic_vertex_exact` | `tests/test_dispersion.py` | **the refinement is exact** for a locally quadratic peak, whatever the offset of the grid |
| `cubicBand`, `central_second_difference_cubic` | `second_derivative`, `effective_mass` | **the symmetric second difference is exact for a cubic**, so the leading error of an effective mass is quartic in the k-spacing; a one-sided difference is not |
| `deriv_deriv_cubicBand` | — | the second derivative it is compared against |
| `gaussPair` | `EffectiveBandStructure.spectral_function` | two Gaussians of width `sigma` at `±d` |
| `gaussPair_merged` | `unresolvable_pairs` | **two states closer than the width merge**: the spectrum is higher *between* them than at either, so the single reported peak lies where no state is |
| `gaussPair_resolved` | `unresolvable_pairs` | at `d ≥ 2 sigma` each state outranks the midpoint, so the two are genuinely resolved |

## Conditioning of the overlap matrix — `Unfolding/Conditioning.lean`

An LCAO unfolding weight is `c* S K c / c* S c`, and both the solver and the
weight assume `S` is positive definite.  A realistic basis makes that fragile:
near-linear dependence gives `S` eigenvalues at rounding-error level, the
Cholesky factorization the plain solver uses fails, and the weight becomes
`0/0`.  Canonical orthogonalization replaces `S` by an isometry onto its
well-conditioned subspace; these statements say what that preserves.

| Lean | Python | statement |
| --- | --- | --- |
| `sNorm` | `overlap_weight` denominator in `core/lcao.py` | the `S`-inner product `c* S c`, the squared norm of a state in a non-orthogonal basis |
| `congr_mulVec_eq` | — | with `Xᴴ S X = 1` the map `X` carries the standard inner product to the `S`-inner product |
| `standard_of_generalized`, `generalized_of_standard` | `solve_generalized_eigenproblem`, `solve_generalized_eigenproblem_truncated` | **the two solvers solve the same problem**: the equivalence holds for *any* `X` with `Xᴴ S X = 1`, so Cholesky and canonical orthogonalization are two choices of one construction |
| `isHermitian_congr` | `solve_generalized_eigenproblem_truncated` | the transformed Hamiltonian stays Hermitian, so the energies stay real and `eigh` is the right routine |
| `sNorm_transformed` | `canonical_orthogonalization`, `tests/test_conditioning.py` | a normalized solution of the standard problem pulls back to an `S`-normalized state — the weight's denominator is one by construction |
| `sNorm_eq_zero_of_mulVec_eq_zero` | `canonical_orthogonalization` | **a null direction of `S` is not a state**: its `S`-norm is zero and its weight is `0/0`, so it must be projected out rather than normalized |
| `sNorm_ge_of_diagonally_dominant` | `diagonal_dominance_bound` | **the cheap certificate**: a unit-diagonal Hermitian `S` whose off-diagonal row sums are at most `r` satisfies `c* S c ≥ (1 − r) ‖c‖²`, an `O(n²)` proof of positive definiteness and a lower bound on the smallest eigenvalue |

The criterion is sufficient and not necessary: a positive `diagonal_dominance_bound`
settles the question, a negative one settles nothing and the eigenvalues have to
be computed (`diagnose_overlap_conditioning`).

## The Fermi level and the energy reference — `Unfolding/Fermi.lean`

Every number quoted from an unfolded band structure is measured from an energy
reference, and the reference is fixed by the electron count.  At finite smearing
the occupied weight is `filling w e T E = ∑ w i · fermiDirac ((e i − E) / T)`,
and these statements say that solving `filling ... E = N` is a well-posed
problem whose answer transforms correctly.

| Lean | Python | statement |
| --- | --- | --- |
| `fermiDirac` | `fermi_dirac_occupation` | the occupation `1 / (1 + exp x)` of a reduced energy |
| `fermiDirac_add_neg` | `tests/test_fermi.py` | particle–hole symmetry, `f(x) + f(−x) = 1` |
| `filling` | `electron_count` | the occupied weight at a chemical potential |
| `filling_add_empty` | `tests/test_fermi.py` | **occupied plus empty is the total**, so the two may be reported separately |
| `filling_strictMono` | `find_fermi_level` | the count increases strictly with the chemical potential — the root is isolated |
| `tendsto_filling_atBot`, `tendsto_filling_atTop` | `find_fermi_level` | it runs from `0` to the total weight, which is why the bracket expansion terminates |
| `existsUnique_fermiLevel` | `find_fermi_level`, `total_weight` | **exactly one Fermi level** for any count strictly between zero and the total weight; a count outside that range is rejected rather than approximated |
| `filling_shift`, `fermiLevel_shift` | `align_reference` | **the Fermi level is equivariant under a change of reference**: shifting every eigenvalue by `d` shifts the level by `d` and changes no occupation |
| `abs_filling_sub_le` | `FermiLevelReport.sensitivity` | the count is Lipschitz with constant `(∑ w) / (4 T)`: cold smearing sharpens the root and tightens the bracket |

Only Fermi–Dirac smearing is offered on the Python side, because that is the
occupation function the formal development covers.  The electron count is *per
primitive cell* — `electrons_per_primitive_cell` performs and checks the
division by `|det T|`.

## The unfolded density of states — `Unfolding/Dos.lean`

Every other diagnostic in the package is local to one k-point.  The density of
states is the one curve an unfolded run can be checked against without referring
to the unfolding at all: the supercell calculation already has one, and
unfolding must reproduce it.  Because the weights of a band sum to one over a
complete fiber, the unfolded spectral functions of the fiber add up to the
undecomposed spectrum of the supercell k-point.

| Lean | Python | statement |
| --- | --- | --- |
| `bandDos` | `supercell_dos` | the broadened spectrum of a k-point with every band counted once |
| `fiberDos` | `unfolded_dos` | the unfolded spectral functions of the primitive k-points of a fiber, added up |
| `IsFiberRepr.fiberDos_eq_bandDos` | `diagnose_dos_conservation` | **unfolding conserves the density of states**: over a complete fiber the unfolded curve *is* the supercell curve, at every energy and for any broadening |
| `IsFiberRepr.fiberDos_div_card` | `unfolded_dos(..., multiplicity=...)` | averaged over the fiber — per primitive cell — it is that curve divided by `|det T|` |
| `integral_bandDos`, `IsFiberRepr.integral_fiberDos` | `DensityOfStates.states`, `DosConservation.expected_states` | both integrate to the number of bands, so the unfolded curve carries `n_bands / |det T|` states per primitive cell whatever the broadening |
| `IsFiberRepr.fiberDos_subset_le` | `DosConservation.max_fiber_deficit` | an incomplete fiber — what a band path samples — can only undercount |
| `fiberDos_nonneg`, `meshDos_nonneg` | `unfolded_dos` | a non-negative kernel and non-negative weights give a non-negative curve |
| `meshDos` | `unfolded_dos(..., kpoint_weights=...)` | the zone average with the mesh weights, which a high-symmetry path does not provide |
| `integrable_meshDos`, `integral_meshDos` | `DensityOfStates.states` | the mesh curve integrates to the total weight — the exact number `states` reports, independently of the energy grid |
| `meshDos_le` | `DosConservation.max_excess` | **weights in `[0, 1]` cannot make the unfolded curve exceed the unweighted one**, whatever the k-point sample; this is the one check that is meaningful on a path, and a positive `max_excess` is a bug |

The integrated curve is evaluated in closed form, like the energy windows of
`Unfolding/Windows.lean`, so the state count does not depend on the resolution
of the energy grid; `DensityOfStates.tail_loss` reports what the grid misses,
which for a Lorentzian is not negligible.

## Tracking peaks into branches — `Unfolding/Tracking.lean`

`Dispersion.lean` locates the peaks of `A(k, ·)` one k-point at a time.  Turning
a peak table into the *branches* one plots means deciding which peak continues
which, and that is a matching problem.  The package uses the matching of least
total squared energy jump, and these statements say why that choice is both
cheap and defensible — and where it is undecidable.

| Lean | Python | statement |
| --- | --- | --- |
| `matchCost` | `unfoldlab.core.tracking._align` | the total squared energy jump of a pairing of two peak lists |
| `monovary_of_monotone` | — | two increasing lists monovary, the hypothesis of the rearrangement inequality |
| `matchCost_id_le` | `track_branches` | **sorting is the assignment algorithm**: for peak lists in increasing order the order-preserving pairing is optimal among *all* permutations, so no assignment problem has to be solved |
| `matchCost_greedy_gt` | `tests/test_tracking.py` | the nearest-free-partner rule is strictly worse on an explicit pair of k-points — greedy tracking is not merely a different convention |
| `exists_matchCost_tie` | `Crossing` | **a degeneracy leaves the branch labels undetermined**: two matchings then cost exactly the same, so a crossing cannot be told from an anticrossing and the ambiguity is reported rather than resolved |
| `abs_pair_exchange` | — | the same exchange inequality for the absolute cost: ordered pairs are matched more cheaply in order than crossed |
| `le_max_jump` | `Branch.max_jump` | each individual squared jump is bounded by the total cost |

When two k-points hold different numbers of peaks the Python side aligns them
with a gap penalty, which is the order-preserving matching of least cost; the
theorem above says that for equal counts nothing outside that class does better.

## Slabs and partial periodicity — `Unfolding/Slab.lean`

A slab is periodic in two directions only; the third holds vacuum, whose
thickness is a convergence parameter rather than a period.  The transform must
be trivial along that axis, and once it is, two things follow that the numerical
code can use: the fiber does not disperse along the vacuum direction, and the
perpendicular plane-wave index never enters the matching test.

| Lean | Python | statement |
| --- | --- | --- |
| `IsSlabTransform` | `is_slab_transform`, `diagnose_slab_transform` | row `p` and column `p` of `T` are `e_p`: the cell is not multiplied along the non-periodic axis `p`, nor is `p` mixed with the periodic directions |
| `IsSlabTransform.fold_perp` | — | folding leaves the perpendicular component of a k-point alone |
| `IsSlabTransform.mulVec_perp` | — | an integer vector along `p` is fixed by `T` |
| `IsSlabTransform.foldsTo_perp` | `flat_kpoint_deviation` | **a slab fiber is flat**: every primitive k-point folding onto `K` shares its perpendicular component with `K` modulo an integer, so an unfolding path that disperses along the vacuum direction asks for weights that are all zero |
| `IsSlabTransform.pwMatches_perp` | — | a matched plane wave pins that component too: the matching test carries no perpendicular information |
| `IsSlabTransform.pwMatches_perp_free` | `collapse_perpendicular` | **the perpendicular index of `G` is irrelevant**: two stored plane waves differing only along the vacuum direction are matched or rejected together |
| `IsSlabTransform.partialNorm_perp_gauge` | — | the matched norm is unchanged by any relabelling of those components |
| `IsSlabTransform.weight_perp_gauge` | `slab_weights_from_coefficients` | consequently `|c_G|²` may be summed over the perpendicular index *before* the matching test — an exact optimization, not an approximation |
| `weight_vacuum_fold_half`, `weight_vacuum_fold_zero` | `diagnose_slab_transform`, `tests/test_slab.py` | the pitfall: with a transform that folds along the vacuum direction, a single vacuum-direction plane wave receives *all* of its weight at `k_perp = 1/2` and none at `k_perp = 0` — a band whose only parameter is the vacuum thickness |

The last two are stated in one dimension with `T = (2)`, which is the smallest
setting in which a "supercell" of nothing but vacuum can be written down; the
three-dimensional version is checked numerically in `tests/test_slab.py`.

## Layer-resolved weights — `Unfolding/Layers.lean`

Unfolding splits a state by k-point, which is a partition of the plane-wave
index.  A slab invites a second split, by layer, which is a partition of *real
space* and therefore a full matrix in the plane-wave basis.  These statements
separate what the second split can and cannot claim.

| Lean | Python | statement |
| --- | --- | --- |
| `regionForm`, `regionCharge` | `layer_charges` | the charge of a state in the region described by an operator, written in the coefficient basis |
| `blockRestrict` | `matching_g_mask` | the coefficients one primitive k-point of the fiber claims |
| `crossForm` | `LayerConservation.cross_term` | the off-block part of the charge |
| `sum_regionCharge_layers` | `layer_resolved_weights` | **at a fixed k-point the layer split is exact**: layer operators summing to the identity split the matched norm exactly, whatever the layer boundaries |
| `regionCharge_eq_blockSum_add_crossForm` | `diagnose_layer_conservation` | **across k-points it is not**: the charge of a region is the sum of the per-k-point charges *plus* a cross term between the blocks |
| `crossForm_eq_zero_of_diagonal`, `sum_regionCharge_blocks_of_diagonal` | `tight_binding_orbital_weights` | the cross term vanishes exactly for an operator diagonal in the same basis — a site or orbital projection in a tight-binding or LCAO basis, which is why those fat bands do carry a fiber sum rule |
| `regionCharge_blockSum_lt` | `tests/test_layers.py` | an explicit two-dimensional region whose per-k-point charges recover only half the true charge: the gap is not a small correction |

The Python side computes the planar-averaged density from its Fourier
coefficients, so a layer charge is a closed-form integral of the density rather
than a sum over a real-space mesh.

## Brillouin-zone meshes — `Unfolding/Mesh.lean`

A supercell run on a k-mesh samples the primitive zone only at the points the
mesh and the transform allow.  A *generalized regular mesh* is generated by an
integer matrix `M` and a fractional shift `s` (`M k = m + s`); the Gamma-centred
Monkhorst--Pack mesh with divisions `n` is `M = diag(n)`, `s = 0`.

| Lean | Python | statement |
| --- | --- | --- |
| `Mesh` | `MeshSpec`, `MeshSpec.contains` | the mesh generated by `M` with shift `s` |
| `Mesh_eq_setOf_foldsTo` | `MeshSpec.points` | **a mesh is a fiber**: it is the set of k-points `M` folds onto `s`, so it has `|det M|` points and the fiber enumeration produces them |
| `unfoldedMesh_eq` | `unfolded_mesh`, `unfolded_mesh_points` | **the primitive k-points a supercell mesh resolves are the mesh generated by `M @ T`**, with the same shift |
| `card_unfoldedMesh_repr` | `MeshReport.n_primitive_kpoints` | there are `|det M| . |det T|` of them |
| `IsFiberRepr.mesh_fiber_complete` | `mesh_kpoint_mappings` | unfolding a complete mesh covers every fiber, so the sum rules are equalities rather than bounds |
| `zero_mem_Mesh_iff`, `zero_not_mem_Mesh_of_shift` | `MeshSpec.is_gamma_centred`, `MeshReport.gamma_resolved` | Gamma is sampled exactly when the shift is integral — a half-shifted supercell mesh misses it for every transform |
| `zoneBoundary_mem_iff` | `MeshReport.zone_boundary_resolved` | `1/2 e_i` is sampled exactly when the division along `i` is even |
| `mesh_resolution_bound` | `mesh_distance`, `nearest_mesh_point` | a k-point at half the mesh spacing is that far from every mesh point: the resolution is `1 / (2 N_i)` |
| `exists_unfoldedMesh_not_product` | `MeshSpec.is_product_mesh`, `axis_divisions` | the unfolded mesh is generally *not* a Monkhorst--Pack mesh, so its k-points must be enumerated and cannot be quoted as three divisions |

## Twisted and multi-layer stacks — `Unfolding/Twist.lean`

A stack has *two* primitive references, one per layer, and one supercell.  Each
is a legitimate unfolding target, which is exactly what makes the layer question
subtle.

| Lean | Python | statement |
| --- | --- | --- |
| `IsCommensurateStack` | `diagnose_stack`, `StackReport` | one supercell that is an integer multiple of both layers: `Am = Ta Aa = Tb Ab` |
| `IsCommensurateStack.det_relation` | `StackReport.multiplicities` | the two multiplicities are fixed by the cell volumes: `|det Ta| |det Aa| = |det Tb| |det Ab|` |
| `transferK`, `foldsTo_transferK` | `transfer_kpoint`, `StackReport.transfer` | re-expressing a layer-A k-point in layer B lands in the fiber of the same supercell k-point |
| `IsCommensurateStack.transfer_cartesian_eq` | `transfer_kpoint` | the transfer is the identity on Cartesian reciprocal space: only the coordinates change |
| `IsCommensurateStack.sum_weight_both_layers` | `StackReport.summary` | **the pitfall**: a state has total weight one in *both* layers' zones, so spectral weight says nothing about which layer a state lives on |
| `layerWeight`, `IsFiberRepr.sum_layerWeight_eq_fraction` | `layer_projected_weights` | the layer-projected weight sums over a fiber to the *fraction* of the state on that layer — the honest layer question, exact only in a site basis |
| `layerWeight_eq_zero_of_support` | `layer_projected_weights` | a state avoiding a layer has no projected weight anywhere in the fiber |
| `hexVec_normSq` | `moire_cell_size` | `|m a1 + n a2|² = a² (m² - m n + n²)`, the norm form of the Eisenstein integers |
| `cosAngle_hexVec_swap` | `CommensurateTwist.cos_angle` | the commensurate twist angle: `cos θ = (4 m n - m² - n²) / (2 (m² - m n + n²))` |
| `det_moireBasis`, `moireBasis_col_eq_rotation` | `commensurate_stack_transforms` | the moiré cell holds `Q = m² - m n + n²` primitive cells of each layer, and its second vector is the 60° rotation of the first |
| `hexRotation_pow_six`, `det_hexRotation` | — | the sixfold rotation in hexagonal coordinates is an integer unimodular matrix |

## Shadow bands of a weak perturbation — `Unfolding/Perturbation.lean`

A charge-density wave, a Peierls dimerization or a Jahn–Teller distortion mixes
the primitive states `|k⟩` and `|k - Q⟩` through one matrix element.  Unfolded,
that shows up as a faint replica of the primitive band, and the whole content of
the picture is in *how* faint.  The two-level problem is solved exactly here —
diagonalized rather than asserted — with `mean ± detuning` the unperturbed
energies and `coupling` the matrix element.

| Lean | Python | statement |
| --- | --- | --- |
| `Shadow.shadowHamiltonian`, `Shadow.mulVec_upperVector`, `Shadow.mulVec_lowerVector` | `two_level_hamiltonian`, `two_level_solution` | the energies and weights used are the eigenpairs of the 2×2 Hamiltonian |
| `Shadow.upperVector_orthogonal`, `Shadow.upperVector_normSq` | — | the two eigenvectors are orthogonal and share the norm `2R(R+d)` |
| `Shadow.upperWeight_eq_component`, `Shadow.lowerWeight_eq_component` | `TwoLevelSolution` | the weights are the squared `\|k⟩` components of those eigenvectors |
| `Shadow.upperWeight_add_lowerWeight` | `two_level_solution` | **the fiber sum rule**: main band and shadow share the primitive character of `\|k⟩` and exhaust it |
| `Shadow.upperWeight_nonneg`, `Shadow.lowerWeight_le_one` | `two_level_solution` | both weights lie in `[0, 1]` |
| `Shadow.lowerWeight_eq_sq_div` | `shadow_weight_bounds` | the closed form `v² / (2R(R+d))`, manifestly `O(v²)` |
| `Shadow.lowerWeight_le_quadratic`, `Shadow.lowerWeight_ge` | `shadow_weight_bounds` | **the shadow is second order in the perturbation**: `v²/(2(d+\|v\|)(2d+\|v\|)) ≤ w ≤ v²/(4d²)` |
| `Shadow.lowerWeight_anti` | — | the shadow fades as the two unperturbed levels separate |
| `Shadow.splitting_ge`, `Shadow.splitting_at_degeneracy` | `TwoLevelSolution.splitting` | level repulsion: the splitting is never below `2\|v\|`, with equality at the degeneracy |
| `Shadow.upperWeight_at_degeneracy` | `two_level_solution` | at the degeneracy the weight splits evenly, so the two branches of an avoided crossing cannot be told apart there |
| `Shadow.upperWeight_neg` | `two_level_solution` | only the modulus of the coupling enters: a phase on the matrix element is a gauge choice |
| `Shadow.abs_coupling_eq`, `Shadow.detuning_eq` | `coupling_from_peaks`, `diagnose_shadow_bands` | **the inversion**: `\|v\| = ΔE √(w₊ w₋)` and `d = (ΔE/2)(w₊ - w₋)` recover the perturbation from a measured pair |

## PAW / ultrasoft augmentation — `Unfolding/Augmentation.lean`

The plane-wave coefficients in a `WAVECAR` or a QE `wfc` file are the *pseudo*
wavefunction: for a PAW or ultrasoft calculation the augmentation charge inside
the spheres is not there, so the weights are the pseudo weights.  This module
says how far off they are, and — the uncomfortable part — that the package's
headline diagnostic cannot tell.

| Lean | Python | statement |
| --- | --- | --- |
| `structureFactor`, `structureFactor_eq` | `structure_factor` | the structure factor of an ideal supercell is `\|det T\|` on the primitive reciprocal lattice and zero off it |
| `exists_structureFactor_ne_zero_of_not_mem` | `structure_factor` | an unreplicated site — a vacancy, an adatom — has structure factor one everywhere, so a defect supercell couples the fiber through the augmentation |
| `IsFiberBlockDiagonal`, `augMatrix_isFiberBlockDiagonal` | `is_fiber_block_diagonal` | an ideal supercell's augmentation never mixes different members of the fiber |
| `augWeight` | `augmented_weights` | the augmentation-aware weight `(a + p) / (N + q)` |
| `IsFiberRepr.sum_augWeight_eq_one` | `augmented_weights` | **the sum rule is blind to the error**: the true weights add to one over a fiber, and so do the pseudo weights |
| `augWeight_sub_weight`, `augWeight_eq_weight_iff` | — | the exact discrepancy, and the condition for the pseudo weight to be right: the augmentation must be spread over the fiber in the same proportion as the plane-wave norm |
| `abs_augWeight_sub_weight_le` | `weight_error_bound`, `diagnose_augmentation`, `unfoldlab norms` | **the error bound**: the pseudo weight is within `q / (N + q)` of the true one, and for an `S`-normalized state that is exactly the norm deficit the readers already report |
