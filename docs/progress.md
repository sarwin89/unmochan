# UnfoldLab progress tracker

A living record of *where the project stands* and *what is being worked on*, so
that work is not repeated and the next pass can start from a known state.

* `docs/findings.md` — the narrative review log (what was wrong, why, and how it
  was fixed).  Append-only, organized by pass.
* `docs/roadmap.md`, `docs/future-implementation.md` — the intended scope.
* `docs/benchmarks.md` — the closed-form cases the code is checked against.
* **this file** — the checklist view: done / in progress / open, with the
  verification state of the repository at the top.

## Repository state

Last updated: local integration pass, 2026-09-02.

| Gate | Command | State |
| --- | --- | --- |
| Python tests | `F:\miniconda3\python.exe -m pytest --basetemp .tmp-pytest -p no:cacheprovider` | 931 passed, 0 failed |
| Lint | `F:\miniconda3\python.exe -m ruff check .` | clean |
| Compile | `PYTHONPYCACHEPREFIX=.tmp-pycache F:\miniconda3\python.exe -m compileall unfoldlab tests` | clean |
| Dependencies | `F:\miniconda3\python.exe -m pip check` | clean |
| Formal model | `lake build` | not run locally; `lake` was not installed |
| Types | `mypy unfoldlab` | not run locally; `mypy` was not installed in the active interpreter |
| Coverage | `python -m pytest -q --cov=unfoldlab` | not run locally in this integration pass |

The local integration pass imported the `output-final_aristotle` tree into this
repository, excluding generated caches and coverage artifacts.  Two local
fixes were needed after import: the spectral plotter now forces the
non-interactive Matplotlib backend before falling back to SVG, and the synthetic
spinor WAVECAR test uses a full-sphere WAVECAR record tag rather than a
gamma-only tag.

Pass 36 received the repository with its directory structure flattened — every
Lean module, Python module, test, document and example in one directory, and the
root `README.md` overwritten by pytest's cache README.  The layout was rebuilt
from the imports (`lean-proofs/RequestProject/`, `unfoldlab/`, `tests/`, `docs/`,
`examples/`), the committed build caches removed and ignored, and the README
rewritten.  If the tree ever arrives flat again, that is the first thing to do:
nothing else can be run until it is.

Pass 17 made `mypy unfoldlab` a gate.  Run all four at the start of a pass.
Pass 33 added a fifth check, run at least once per pass:
`python -m pytest -q --cov=unfoldlab --cov-report=term-missing`.  It is not a
hard gate — the interactive `cli/guided.py` is not worth pinning line by line —
but a module at *zero* percent means nothing imports it, and the other four
gates are all happy with that.  That is how `core/augmentation.py` was found
orphaned after having been written, formalized and forgotten.


Pass 19 found the Lean development carrying four `sorry`s on arrival, in
`Overlap.lean`, a module `Main.lean` did not import — so the default build
target never saw it and only the *warnings* of a full `lake build` revealed it.
Read the warnings, not just the exit status, and check that every module is
reachable from `Main.lean`.

Pass 16 found the Lean build *broken* on arrival (a `ring` in `Truncation.lean`
left with no goals after `field_simp` grew stronger).  Run every gate at the
start of a pass; a tactic proof can stop compiling with nobody touching it.

Environment note: the Python gates need the package installed with its optional
extras, `pip install -e ".[io,plot,dev]"` (`h5py` in particular gates the QE
HDF5 tests, which skip silently without it).

## Component status

| Area | Module(s) | State |
| --- | --- | --- |
| Folding / fiber enumeration | `core/kpoints.py`, `core/lattice_quotient.py` | done, exact integer arithmetic, formalized |
| Brillouin-zone meshes | `core/mesh.py` | the resolvable primitive k-points of a supercell mesh (the mesh of `M @ T`), the shift and even/odd pitfalls, the resolution error, and the mesh-to-fiber k-point list; CLI `mesh`; formalized in `Mesh.lean` |
| Weak perturbations and shadow bands | `core/perturbation.py` | the exact two-level model of a charge-density wave, Peierls or Jahn–Teller distortion, the `O(v^2)` shadow intensity with proved bounds, and the inversion that reads `\|v\|` back off a measured pair; CLI `shadow`; formalized in `Perturbation.lean` |
| PAW / ultrasoft augmentation | `core/augmentation.py` | the error the plane-wave weight makes on a pseudo wavefunction: the structure factor of an ideal supercell, the fiber-block-diagonal test, the augmentation-aware weight, the blindness of the fiber sum rule to all of it, and the bound by the norm deficit; reported by `unfoldlab norms`; formalized in `Augmentation.lean` |
| Plane-wave matching kernel | `core/plane_waves.py` | done, tolerance-free, formalized; the G-vectors carry a k-independent class label (`Classes.lean`) so `shared_weights_from_coefficients` runs the matching test and squares the coefficient array once for a whole fiber or symmetry star, whatever its size |
| Weights and sum rules | `core/unfolding.py` | done, both sum rules checked at runtime; `SharedWavefunctionGroup` / `compute_plane_wave_unfolding_weights_shared` batch the k-points that share one stored wavefunction |
| Stored-state completeness | `core/unfolding.py`, `io/vasp_wfc.py`, `io/qe_wfc.py` | stored plane-wave norms, empty-state count and the `delta / (1 - delta)` weight-error bound; CLI `norms`; formalized |
| Spectral function | `core/spectral.py` | Gaussian, Lorentzian, per-state adaptive widths; formalized |
| Peak tracking, branches | `core/tracking.py` | order-preserving least-cost matching of peaks between neighbouring k-points, with the crossings and termini it cannot resolve reported; CLI `branches`; formalized in `Tracking.lean` |
| Density of states | `core/dos.py` | unfolded and supercell curves, closed-form state count, the conservation report; CLI `dos`; formalized in `Dos.lean` |
| Peak dispersion, effective mass | `core/dispersion.py` | peak extraction with exact parabolic refinement, the broadening resolution limit, curvature masses; CLI `dispersion`; formalized in `Dispersion.lean` |
| Degeneracy / gauge invariance | `core/degeneracy.py` | multiplet grouping, gauge-invariant subspace weights, gauge-sensitivity report; CLI `degeneracy`; formalized |
| Spin texture | `core/plane_waves.py`, `io/vasp_wfc.py`, `io/qe_wfc.py` | noncollinear textures with the Cauchy–Schwarz bound and fiber sum rule; both readers; CLI `--spin-texture` |
| Projections / selectors | `core/projections.py`, `core/site_projection.py` | done for species/atom/orbital/layer/region/valley |
| Valleys, minimum image | `core/valleys.py` | Cartesian minimum image with a certified growing-box search; `certified_minimum_image_shift` returns the winning translation as well as the distance, so the same search serves the zone reduction; formalized in `MinimumImage.lean` |
| First Brillouin zone | `core/brillouin.py` | Wigner-Seitz representative of a k-point (certified), the signed distance to the nearest Bragg plane, the inscribed radius, the orthogonality test that says when `rint` reduction is already exact, and the discrepancy report for a k-point list; CLI `bz`; formalized in `BrillouinZone.lean` |
| Slabs, wires, partial periodicity | `core/slab.py` | vacuum-gap detection, the check that the transform is trivial along the non-periodic axes, the flat-fiber check on a k-path, and the exact collapse of the plane-wave list over the perpendicular index; CLI `slab`; formalized in `Slab.lean` |
| Layer resolution of a slab | `core/layers.py` | planar-averaged density from its Fourier coefficients, closed-form layer charges, the layer split of an unfolded weight, and the cross-term diagnostic that says how far a k-summed layer weight is from a layer charge; CLI `layers`; formalized in `Layers.lean` |
| Commensurability / strain | `core/strain.py`, `core/transformations.py` | exact integer transform required; the mismatch of an approximately commensurate pair quantified (residual, strain, rotation, k error) by `diagnose_commensurability` and CLI `strain`; formalized in `Strain.lean` |
| Symmetry | `core/symmetry.py`, `core/spacegroup.py` | operations supplied or detected in-house (no spglib dependency: the lattice point group is enumerated with a proved Cauchy–Schwarz bound, `PointGroup.lean`); reduced-set unfolding done for VASP and QE; time reversal (`k -> -k`) adjoinable for nonmagnetic systems; magnetic detection with `magmoms`, collinear as a site decoration and noncollinear as an axial vector; supplied operations audited against the crystal by `validate_primitive_operations` and `unfoldlab check-symmetry`; formalized in `Symmetry.lean` and `SymmetryHypothesis.lean` |
| VASP `WAVECAR` | `io/vasp_wfc.py` | collinear, spin-polarized, Γ-only, noncollinear spinors |
| VASP `PROCAR`/`EIGENVAL` | `io/vasp.py`, `io/procar.py` | done (approximate projection mode) |
| Quantum ESPRESSO | `io/qe.py`, `io/qe_wfc.py`, `io/qe_xml.py` | `.dat` and HDF5 wavefunctions, XML metadata |
| Tight binding / Wannier | `core/tight_binding.py`, `io/wannier90.py` | JSON models, `seedname_hr.dat` and `seedname_tb.dat`; fiber and path unfolding; orbital-projected fat bands |
| Non-orthogonal (LCAO) bases | `core/lcao.py` | overlap-aware weights, generalized eigenproblem, neglect diagnostic; formalized in `Overlap.lean` (fiber sum rule) and `OverlapCompleteness.lean` (across-bands sum rule) |
| Overlap conditioning | `core/conditioning.py` | eigenvalue range and condition number of `S`, an `O(n^2)` diagonal-dominance certificate of positive definiteness, canonical orthogonalization and a truncated generalized solver; CLI `overlap`; formalized in `Conditioning.lean` |
| Bloch-sum conventions | `core/convention.py` | conversion between the cell / atomic / full-wavefunction conventions, kernel regauging, and the misassignment diagnostic; formalized in `Convention.lean` |
| Phonons | `core/phonons.py`, `io/force_constants.py` | force-constant models, on-site (force-constant) defects and site-dependent masses (isotopes, substitutions); masses formalized in `Phonons.lean` |
| Twist / moire | `twist/`, `twist/commensurate.py` | commensurate hexagonal twist angles and moire cell sizes, the two layer transforms, the exact k transfer between the two primitive zones, the per-layer stack diagnosis, and layer-projected tight-binding weights; CLI `twist`; formalized in `Twist.lean` |
| Fermi level, energy reference | `core/fermi.py` | Fermi–Dirac electron counting, a bracketed bisection whose root is proved unique, the per-primitive-cell division by `|det T|`, and reference alignment; CLI `fermi`; formalized in `Fermi.lean` |
| Cuts and windows | `core/windows.py` | closed-form constant-energy cuts, energy-window weights and band filling; CLI `cut`; formalized |
| Disorder / alloys | `core/ensemble.py` | configurational averaging, band moments, law-of-total-variance disorder broadening; CLI `ensemble`; formalized |
| Serialization | `io/serialization.py` | JSON + run manifest + HDF5 |
| Figures | `io/plot_bands.py`, `io/plot_spectral.py` | the per-state scatter plot and the broadened `A(k, E)` heat map, each with a matplotlib renderer and a standard-library-only fallback that clips to the energy window and survives non-finite input |
| CLI | `cli/` | scriptable subcommands and guided mode |

## Backlog

Ordered by value.  `[x]` done, `[~]` in progress, `[ ]` open.

### Physics gaps

- [x] Component-resolved (spin-projected) weights for noncollinear files.
- [x] **Spin texture** for noncollinear unfolding: the off-diagonal spinor
      cross terms, i.e. ⟨σx⟩ and ⟨σy⟩, which component-resolved weights cannot
      express.  Done in pass 10 with the Cauchy–Schwarz bound and the fiber sum
      rule proved in `Spin.lean`.
- [x] **Adaptive broadening**: per-state widths (band- and k-dependent), with
      weight conservation proved rather than assumed.  Done in pass 10.
- [x] Unfolding from a symmetry-reduced QE wavefunction set (the roadmap listed
      this as open; it has in fact been implemented for both readers, see
      `tests/test_qe_symmetry_workflow.py`).
- [x] Wannier90 `seedname_tb.dat` (carries the lattice and the position
      operator).  Done in pass 11: `read_wannier90_tb`, Wannier centres, and
      `centre_structure()` so `detect_layers` can build orbital groups.
      Interpolation onto a k-path was already available through
      `interpolate_segment` + `unfold_tight_binding_path`.
- [x] Spin texture for the QE reader (the VASP WAVECAR path was already done).
- [x] **Orbital-projected (fat band) tight-binding unfolding**: which orbital,
      sublattice or layer carries the unfolded weight.  Done in pass 11 with
      additivity, the bound by the total weight and the projected fiber sum
      rule proved in `TightBinding.lean`.

- [x] **Configurational averaging** for disorder, alloys and defect ensembles.
      Done in pass 13: `core/ensemble.py` merges configurations with their
      probabilities (`stack_configurations`), averages their spectral functions
      with the configuration-to-configuration scatter
      (`ensemble_spectral_function`), and splits the width of the averaged band
      into an intrinsic and a disorder part (`disorder_broadening`), with the
      sum rule and the law of total variance proved in `Ensemble.lean`.
      `unfoldlab ensemble --input ... --report ...` exposes it.

- [x] **Constant-energy cuts and energy windows** (Fermi surface, ARPES-like
      cuts, occupied weight, gap leakage).  Done in pass 13: `core/windows.py`
      evaluates `A(k, E0)` directly and integrates the broadened spectrum over
      an energy window in closed form, with additivity, the bound by the total
      weight and monotonicity proved in `Windows.lean`.  `unfoldlab cut
      --energy ...` / `--window lo,hi` exposes it.

- [x] **Gauge invariance of degenerate multiplets**: the weight of an
      individual band inside a degenerate multiplet depends on the basis the
      diagonalizer returned.  Done in pass 14: `core/degeneracy.py` groups the
      bands, reports the gauge sensitivity and produces the invariant multiplet
      weight, with the invariance and the surviving sum rule proved in
      `Degeneracy.lean`.  `unfoldlab degeneracy --input ...` exposes it.

- [x] **Time-reversal reduction of the k-point mesh**: a nonmagnetic
      calculation stores only one of each pair `+/-K`, whether or not the
      crystal is centrosymmetric, and the detected spatial point group cannot
      express that.  Done in pass 14: `detect_primitive_operations(...,
      time_reversal=True)` and `detect-symmetry --time-reversal` adjoin
      `k -> -k`, with the recipe proved in `TimeReversal.lean`.

- [x] **Completeness of the stored wavefunction (the PAW caveat)**: the weight
      divides by the norm of what the file holds, which for a PAW or ultrasoft
      run is only the pseudo part of the state.  Done in pass 15: the weight is
      proved to be scale invariant (`weight_smul`), the error caused by a
      missing norm fraction `delta` is bounded by `delta / (1 - delta)`
      (`weight_truncation_bound`) and the stability of a weight under a
      perturbation of the state by `eps` is `2 eps`
      (`weight_perturbation_bound`).  `state_norms_from_wavecar`,
      `state_norms_from_qe_save`, `diagnose_state_norms` and the `unfoldlab
      norms` command measure `delta` and report the bound.  Recovering the
      augmentation charge itself would need the PAW projectors and is out of
      scope for a wavefunction-file reader.

- [x] **Mass (isotope) defects in a phonon supercell.**  Done in pass 16: the
      only supercell defect the phonon layer could express was an additive
      on-site term, which is a *force-constant* defect; a mass defect is the
      congruence `D -> S D S` and no on-site term reproduces it
      (`Phonons.lean`: `dynMatrix_massScale`,
      `mass_change_not_diagonal_shift`).  `site_scaling` in the tight-binding
      kernel, `supercell_site_masses` / `mass_site_scaling` /
      `supercell_dynamical_matrix` in `core/phonons.py`, a `site_masses` block
      in the force-constant JSON and `phonon bands --site-mass` expose it.  The
      same file also proves that diagonalizing the mass-weighted matrix solves
      the generalized problem, that the acoustic zero mode survives any masses,
      and why the eigenvector rather than the displacement is unfolded.

- [x] **Unfolding from a non-orthogonal (LCAO) basis.**  Done in pass 19: the
      tight-binding weight assumes orthonormal orbitals, which is false for
      SIESTA, CP2K, OpenMX and FHI-aims, where `<psi|psi> = c* S c` and the
      eigenproblem is `H c = E S c`.  `core/lcao.py` carries the overlap
      through numerator and denominator, and `Overlap.lean` proves the fiber
      sum rule with the overlap (`sum_ovWeight_eq_one`), that the orthonormal
      case is recovered exactly (`ovWeight_orthonormal`), the size of the error
      made by ignoring the overlap (`ovNumer_sub_tbNumer`) and scale invariance
      (`ovWeight_smul`).

- [x] **Bloch-sum conventions.**  Done in pass 21: a coefficient table means
      nothing until the convention is fixed, and the package had no statement
      about it.  `Convention.lean` shows that the atomic-versus-cell
      distinction (a per-orbital phase) is a *gauge* of the weight and of every
      fat band, while the full-wavefunction reading (a per-cell phase)
      translates the whole distribution in k — with the fiber sum rule intact,
      so no runtime check catches it — and that in a non-orthogonal basis the
      overlap kernel must be regauged with the state.  `core/convention.py`
      converts between the three conventions and reports the misplaced weight.

- [x] **Strained and approximately commensurate supercells.**  Done in pass 22:
      a relaxed defect cell, a strained heterostructure layer or a commensurate
      approximant of a twisted stack is not an integer multiple of the
      reference, and the package's only answer was a hard error.
      `Strain.lean` proves that commensurability *is* integrality of
      `A_sc A_pc⁻¹` (`latticeSpan_subset_iff_intMatrix`, so there is no
      approximate version), that residual strain mislabels k by `q (1 − D⁻¹)`
      — linear in k, invisible at Γ, worst at the zone boundary
      (`strain_kshift`) — that no integer transform removes it
      (`strain_kshift_eq_zero_iff`), and that entrywise rounding is already the
      best integer approximation (`round_matrix_optimal`).  `core/strain.py`
      and `unfoldlab strain` report the residual, the strain tensor
      (infinitesimal or Green–Lagrange), the rigid rotation from the polar
      decomposition and the worst-case k error.

- [x] **Quoting numbers from a spectral function.**  Done in pass 23: the
      package drew `A(k, E)` but never turned it back into a band energy or an
      effective mass.  `Dispersion.lean` proves the three-point parabolic
      refinement exact for a locally quadratic peak, the symmetric second
      difference exact for a cubic (so a mass is quartically accurate in the
      k-spacing), and — the physically important one — that two states closer
      than the broadening width `sigma` produce a *single* peak lying between
      them where no state is (`gaussPair_merged`), while at `2 sigma` they are
      resolved (`gaussPair_resolved`).  `core/dispersion.py` and `unfoldlab
      dispersion` extract the peak table, the curvature masses and the list of
      pairs the broadening merges.

- [x] **Charge-density-wave, Peierls and Jahn–Teller supercells**: what a *weak*
      superlattice perturbation looks like unfolded, and what can be measured
      from it.  Done in pass 32: `Unfolding/Perturbation.lean` diagonalizes the
      two-level problem exactly (the eigenvalue equations, not a posited
      formula), proves the two weights sum to one, that the shadow intensity is
      second order in the coupling with matching upper and lower bounds, that
      the splitting never falls below `2|v|`, and that `|v| = dE sqrt(w+ w-)`
      inverts the picture; `core/perturbation.py` and `unfoldlab shadow` expose
      it, with the pair-selection heuristic and its failure mode documented and
      pinned by a test.

- [x] **The symmetry operations nobody checked, and magnetism.**  Done in pass
      36.  Symmetry-reduced unfolding is the one place the code serves a stored
      wavefunction for a k-point it was not computed at, and the kernel can only
      test that the operation is unimodular and compatible with the two
      lattices — it never sees the atoms.  A matrix that passes those tests but
      is not a symmetry of the crystal gives the weights of an unrelated state,
      with every sum rule still satisfied.  `SymmetryHypothesis.lean` exposes the
      hypothesis (`weight_symmetry_of_norm_eq`), shows it is load-bearing
      (`exists_weight_ne_of_coefficients`), and exhibits the pitfall: for a
      square lattice with anisotropic hoppings the axis exchange is unimodular,
      commutes with every diagonal transform, and identifies two k-points `2 t`
      apart in energy (`exists_latticeSymmetry_bandEnergy_ne`).  The magnetic
      half is the same gap: the detected group was that of the nonmagnetic
      skeleton (`zeeman_invariant_iff_moment_invariant`,
      `exists_species_invariant_moment_not`).  `core/spacegroup.py` gained
      `site_labels`, `cartesian_rotation`, `magmoms` on the two detection
      entry points and `validate_primitive_operations`; the CLI gained
      `check-symmetry`, `detect-symmetry --magmom`, and validation of a
      `--symmetry` file inside `weights --structure` and `unfold --supercell`.
      `tests/test_symmetry_validation.py`, `tests/test_symmetry_cli.py`;
      `docs/findings.md` §38.

- [x] **PAW / ultrasoft augmentation.**  The mathematics (`Augmentation.lean`)
      and the numerics (`core/augmentation.py`) existed but were never
      connected: pass 33 found the module at 0% coverage, unexported,
      untested, uncalled and undocumented.  It is now exported, tested against
      explicit matrices (including the fact that a fiber-block-diagonal
      augmentation leaves the sum rule intact and a cross term breaks it), and
      reported by `unfoldlab norms` alongside the truncation bound.

### Numerics and performance

- [x] Exact integer matching, vectorized weights, cached supercell hoppings.
- [x] **Take the k-point out of the matching test.**  Done in pass 37.  The
      batched kernel of pass 34 still called `matching_g_mask` once per
      primitive k-point and contracted against a dense `(n_g, n_kpoints)` mask,
      so both the integer test and the accumulation carried a factor of
      `|det T|` for a complete fiber.  A G-vector now carries the class label
      `G @ adj(T).T mod |det T|`, which does not mention the k-point at all;
      the k-point selects one class, the class sums are accumulated in a single
      segmented pass, and a repeated or unreachable k-point falls out as a
      lookup.  `Unfolding/Classes.lean` proves the label names the class and the
      k-point faithfully, so the rewrite is exact, and it is: the weights are
      bit-for-bit those of the two older routes.  On 35 937 G-vectors, 60 bands
      and a 64-point fiber, 2.150 s (per k-point) → 0.104 s (pass 34) →
      0.035 s.  `compute_weights_from_coefficient_table` lost its k-point loop
      and its `np.add.at` calls the same way.  `tests/test_gvector_classes.py`,
      `examples/benchmark_weights.py`, `docs/findings.md` §39.
- [x] **Measure before rewriting the tight-binding path.**  Done in pass 37 and
      the answer was *leave it alone*: for a 64-cell, 4-orbital supercell over
      200 k-points, `unfold_tight_binding_path` spends 3.54 s of 3.79 s inside
      `numpy.linalg.eigh`.  The `np.add.at` in
      `_assemble_supercell_hamiltonian` is a genuinely slow numpy path and 0.2 %
      of the runtime; batching the assembly or the diagonalization moves the
      total by under a percent.  See `docs/findings.md` §39.5 before spending a
      pass on it.
- [x] **The plot nobody without matplotlib had ever seen.**  Done in pass 35.
      The per-state band scatter plot lived in `io/qe.py` — a Quantum ESPRESSO
      *reader*, which the VASP workflow was importing a plotter out of — and
      its dependency-free SVG renderer, the one an install without the optional
      `plot` extra actually uses, was at zero coverage and had two defects that
      only appear on real data: states outside `--emin/--emax` were drawn
      *outside* the axes box, over the tick labels and the title, where
      matplotlib clips them; and a single non-finite energy set the whole
      y-axis to `nan`, destroying the figure rather than costing one marker.
      The code is now `io/plot_bands.py`, backend-neutral (it asks only for a
      `distances` array), re-exported from `io/qe.py` so the old import path
      keeps working, with both defects fixed, shapes validated, and the marker
      *area* — not its radius — made affine in the weight so that the two
      renderers show the same picture.  `tests/test_plot_bands.py`.

- [x] **Batch the k-points that share one wavefunction.**  Done in pass 34.
      `weights_from_coefficients` spends nearly all its time forming
      `re**2 + im**2` over the whole `(n_bands, n_components, n_g)` array; the
      matching mask is an order of magnitude cheaper.  Whenever one stored
      wavefunction serves several primitive k-points — a complete fiber, or a
      band path that revisits an irreducible k-point under symmetry — that
      dominant cost was paid once per k-point.
      `core/plane_waves.shared_weights_from_coefficients` now takes an
      `(n_kpoints, 3)` array and contracts the squared moduli against an
      `(n_g, n_kpoints)` mask matrix in one product, and
      `weights_from_coefficients` is the one-k-point case of it, so the two
      cannot drift.  `core/unfolding.SharedWavefunctionGroup` and
      `compute_plane_wave_unfolding_weights_shared` are the streaming driver,
      and both symmetry-reduced readers
      (`io/vasp_wfc._compute_weights_from_wavecar_symmetry`,
      `io/qe_wfc._compute_symmetry_reduced_weights`) were rewired onto it.
      Measured on 64 bands x 200 000 G-vectors and an eight-point fiber: 2.37 s
      to 0.35 s, with a maximum weight difference of `3e-16`.
      `tests/test_shared_weights.py` checks the batched kernel against a
      Python-loop transcription of the definition rather than against the
      function it replaces.

- [x] Chunked / lazy reading of large wavefunction files.  Done in pass 12 for
      the VASP `WAVECAR`: `WavecarReader.read_kpoint_wavefunction` takes a band
      block, `compute_weights_from_wavecar(..., band_chunk=m)` and the CLI's
      `--band-chunk` accumulate the weights block by block, and
      `Chunking.lean` proves that blocking over bands *or* over plane waves
      returns the unblocked answer.
      The QE readers take the same `band_chunk`: `read_wfc_hdf5` slices the
      `evc` dataset and `read_wfc_dat` keeps only the records of the requested
      block.

- [x] **Make `mypy unfoldlab` clean and add it to the gate.**  Done in pass 17:
      71 errors in 19 files went to zero, fixed at the root rather than
      silenced.  The three structural changes were `TransformLike` (an
      `int64`-or-`float64` transform, since every consumer rounds anyway) and
      `OrbitalGroups` (a group is a sequence *or* an index array) as named
      aliases, and `default_weights` / `default_distances` in `core/spectral.py`
      so `EffectiveBandStructure.weights` and `.distances` are genuinely
      non-optional after construction.  `TransformationMatrix.from_values` is
      the normalizing constructor for callers that only have floats.  The
      exercise turned up one real portability bug — `datetime.UTC` in
      `io/serialization.py` does not exist on Python 3.10, which the package
      advertises as its minimum — and one latent `None` dereference in
      `unfoldlab norms`, where the exclusive `--wavecar` / `--qe-save-dir`
      check was written so that neither option also fell through to the QE
      branch.  `tests/test_typing_contracts.py` pins the changed behaviour.

- [x] **Split `cli/main.py`.**  Done in pass 18: 1766 lines became six modules,
      none over 600.  `cli/app.py` holds the Typer application and its root
      callback; `cli/common.py` the console, the prompts and the parsers;
      `cli/commands.py` the pipeline (`make-kpoints`, `weights`, `unfold`,
      `qe-info`); `cli/diagnostics.py` the questions asked *about* a run
      (`fiber`, `cut`, `ensemble`, `norms`, `degeneracy`, `detect-symmetry`);
      `cli/backends.py` the `vasp`/`qe` subgroups; `cli/guided.py` the
      interactive menu.  `cli/main.py` is now just the entry point and still
      exports `app` and `main`.  The one cycle -- the root callback launches the
      guided menu, which drives the commands, which register on the app -- is
      broken by a deferred import inside the callback, and
      `tests/test_cli_structure.py` asserts every module imports on its own, the
      full command set is registered, and no module is back over 1000 lines.

- [x] **Conditioning of the LCAO overlap.**  Done in pass 24.  The generalized
      solver added in pass 19 assumed a positive definite `S` and reported a
      bare Cholesky failure when it was not.  `core/conditioning.py` now gives
      the eigenvalue range, the condition number, an `O(n^2)`
      diagonal-dominance certificate of positive definiteness
      (`sNorm_ge_of_diagonally_dominant`), canonical orthogonalization
      (`X* S X = 1`) and a truncated solver that drops the near-null directions
      instead of failing on them; `unfoldlab overlap` is the CLI surface.
      Truncation is deliberately *not* threaded through `unfold_lcao_path`:
      a k-dependent number of states is not a rectangular band structure.

- [x] **Split `cli/diagnostics.py`.**  Done in pass 25, before adding the
      `fermi` command would have pushed it over the 1000-line limit that
      `tests/test_cli_structure.py` enforces.  `cli/diagnostics.py` (505 lines)
      keeps the questions asked about an unfolded run -- `fiber`, `cut`,
      `ensemble`, `norms`, `degeneracy` -- and `cli/analysis.py` holds the
      commands that interrogate the inputs of a run or extract physical numbers
      from a finished one: `detect-symmetry`, `strain`, `dispersion`,
      `overlap`, `fermi`.

- [x] **Find the Fermi level, and pin the energy reference.**  Done in pass 25.
      `core/windows.py` could integrate the weight below a given energy but
      nothing could find that energy, and an unfolded run referenced to the
      wrong number looks entirely normal.  `core/fermi.py` counts electrons per
      *primitive* cell (dividing by `|det T|` and checking the division),
      brackets and bisects for the level -- unique by
      `existsUnique_fermiLevel` -- reports the Lipschitz sensitivity of the
      count, and realigns the reference; `unfoldlab fermi` is the CLI surface.
      Only Fermi--Dirac smearing is offered: the Methfessel--Paxton occupation
      is not monotone, so the uniqueness theorem simply fails for it.

- [x] **The unfolded density of states.**  Done in pass 26.  Every diagnostic
      the package had was local to one k-point and used quantities the
      unfolding itself produced; nothing checked the run against the supercell
      calculation's own spectrum.  `Dos.lean` proves that summing the unfolded
      spectral functions over a complete fiber returns the supercell density of
      states (`IsFiberRepr.fiberDos_eq_bandDos`), that per primitive cell it is
      that curve divided by `|det T|` (`fiberDos_div_card`), that both
      integrate to the band count (`integral_bandDos`,
      `IsFiberRepr.integral_fiberDos`), and the two one-sided statements that
      survive an imperfect sample: an incomplete fiber undercounts
      (`fiberDos_subset_le`) and weights in `[0, 1]` cannot exceed the
      unweighted curve (`meshDos_le`) — the latter being the only check that is
      meaningful on a band path.  `core/dos.py` and `unfoldlab dos` report all
      of it, with the state count in closed form so it does not depend on the
      energy grid.

- [x] **Tracking peaks into branches.**  Done in pass 27.  The package could
      find the peaks of `A(k, E)` but not say which peak at one k-point
      continues which at the next, so nothing it produced could be plotted or
      differentiated as a band.  `Tracking.lean` proves that for peak lists in
      increasing order the order-preserving matching minimizes the total
      squared energy jump among *all* permutations (`matchCost_id_le`, the
      rearrangement inequality), that the nearest-free-partner rule is strictly
      worse on an explicit example (`matchCost_greedy_gt`), and that at a
      degeneracy two matchings tie (`exists_matchCost_tie`) so a crossing
      cannot be told from an anticrossing.  `core/tracking.py` and `unfoldlab
      branches` build the branches and *report* the ambiguities rather than
      inventing a continuation.

- [x] **Slabs and partial periodicity.**  Done in pass 28.  Every path assumed
      all three cell directions are real periods, so a transform that folds
      along the vacuum direction of a slab was accepted and produced a band
      dispersing purely because of the vacuum thickness.  `Slab.lean` proves
      that for a transform trivial along the non-periodic axis the fiber is
      flat (`IsSlabTransform.foldsTo_perp`), that the perpendicular plane-wave
      index never enters the matching test
      (`IsSlabTransform.pwMatches_perp_free`) so it may be summed over first
      (`IsSlabTransform.weight_perp_gauge`), and exhibits the pitfall
      (`weight_vacuum_fold_half` / `weight_vacuum_fold_zero`).  `core/slab.py`
      and `unfoldlab slab` diagnose the transform and implement the collapse.

- [x] **Layer-resolved unfolding of an asymmetric slab.**  Done in pass 29.
      `Layers.lean` settles what such a number may claim: at a fixed k-point the
      layer split of the weight is exact for any boundaries
      (`sum_regionCharge_layers`), but summing a layer's share over the fiber does
      *not* return the layer's charge, because a layer is diagonal in real space
      and hence a full matrix in the plane-wave basis
      (`regionCharge_eq_blockSum_add_crossForm`); the cross term vanishes precisely
      for a site or orbital projection in a tight-binding or LCAO basis
      (`sum_regionCharge_blocks_of_diagonal`), and an explicit region loses half its
      charge to it (`regionCharge_blockSum_lt`).  `core/layers.py` and `unfoldlab
      layers` compute the layer charges in closed form and report the gap.

- [x] **Which primitive k-points a supercell mesh resolves.**  Done in pass 30.
      The two zone-integrating diagnostics added in passes 25 and 26 (the Fermi
      level and the density of states) both take a *mesh* of supercell
      k-points, and nothing said which primitive k-points such a run can be
      unfolded onto.  `Mesh.lean` shows that a mesh is a fiber
      (`Mesh_eq_setOf_foldsTo`), that the resolvable primitive k-points are
      exactly the mesh generated by `M @ T` with the same shift
      (`unfoldedMesh_eq`), `|det M| . |det T|` of them and one full fiber above
      each supercell k-point (`IsFiberRepr.mesh_fiber_complete`), that a
      half-shifted supercell mesh misses Gamma for *every* transform
      (`zero_mem_Mesh_iff`), that an odd division misses the zone boundary
      (`zoneBoundary_mem_iff`), that the resolution is half a mesh spacing
      (`mesh_resolution_bound`), and — the reason the module is needed at all —
      that the unfolded mesh is generally not a Monkhorst--Pack mesh
      (`exists_unfoldedMesh_not_product`), so its k-points must be enumerated.
      `core/mesh.py` and `unfoldlab mesh` expose it; `--first-bz` adds the
      Wigner--Seitz representative of each resolvable k-point (pass 38).

- [x] **Twisted and multi-layer stacks.**  Done in pass 31.  The twist area
      could describe a stack but could not say which twist angles are
      commensurate, nor what unfolding a stack onto one of its two references
      means.  `Twist.lean` settles both: `IsCommensurateStack` ties the two
      transforms to one supercell and `det_relation` fixes their multiplicities
      by the cell volumes; `transferK` moves a k-point between the two
      primitive zones without moving it in Cartesian reciprocal space
      (`transfer_cartesian_eq`) or out of its fiber (`foldsTo_transferK`);
      `sum_weight_both_layers` is the pitfall — a state has total weight one in
      *both* layers' zones, so spectral weight says nothing about which layer it
      lives on, and only the layer-projected `layerWeight`, whose fiber sum is
      the layer's *share* of the state norm, answers that question, exactly and
      only in a site basis; and `hexVec_normSq` / `cosAngle_hexVec_swap` /
      `det_moireBasis` give the commensurate hexagonal angles and the
      `m^2 - m n + n^2` cell.  `twist/commensurate.py` and `unfoldlab twist`
      expose it.

### Data model

- [x] Run manifest with versions, transform, source digests.
- [x] Optional HDF5 export of large `EffectiveBandStructure` objects.  Done in
      pass 12: `write_ebs_hdf5` / `read_ebs_hdf5` store the four arrays as
      native `float64` datasets with the manifest in the attributes, and
      `write_ebs` / `read_ebs` dispatch on the suffix, so `unfoldlab unfold
      --json run.h5` writes HDF5.

### Documentation

- [x] `docs/formal-model.md` maps every Lean statement to the Python function
      it governs; keep it in step with new modules.
- [x] Benchmark gallery.  Done in pass 12 as `docs/benchmarks.md` plus
      `tests/test_benchmarks.py`: five cases with closed-form answers (perfect
      chain and square-lattice supercells, the dimerized chain gap, a single
      plane-wave state, a defect chain), backed by
      `UnfoldLab.IsFiberRepr.weight_ideal` in `Ideal.lean`.  A comparison
      against *published* figures is deliberately out of scope; see the last
      section of that document.

## Pass log

| Pass | Focus |
| --- | --- |
| 1–9 | see `docs/findings.md` sections 1–12 |
| 38 | which k-point is actually being reported: the package reduced k-points only with `rint`, the parallelepiped representative, which is not the first Brillouin zone for a non-orthogonal cell — `Unfolding/BrillouinZone.lean` (the zone as an intersection of Bragg half spaces, the inscribed-sphere sufficient test, uniqueness up to the boundary, rounding proved exact for an orthogonal basis and a counterexample five times too long for a skewed one, the box certificate, and the proof that reduction changes no fiber or weight), `core/brillouin.py`, `unfoldlab bz`, one shared certified search in `core/valleys.py` instead of two; `tests/test_brillouin.py`; `docs/findings.md` §40 |
| 37 | performance, measured rather than guessed: the plane-wave matching test no longer runs once per k-point — a G-vector carries the k-independent class label `G @ adj(T).T mod \|det T\|` (`Unfolding/Classes.lean`: the label names the class and the k-point faithfully, matching is label equality, the accumulation is a class sum), the fiber weights are one segmented pass, `compute_weights_from_coefficient_table` is loop-free, 2.150 s → 0.035 s on a 64-point fiber with the weights bit-for-bit unchanged; the tight-binding path measured and deliberately left alone (93 % of it is `eigh`); `tests/test_gvector_classes.py`, `examples/benchmark_weights.py`, `docs/findings.md` §39 |
| 36 | the unchecked hypothesis of symmetry-reduced unfolding: `Unfolding/SymmetryHypothesis.lean` (the recipe with its physical hypothesis exposed, a proof that the hypothesis is load-bearing, the anisotropic-lattice pitfall, and the magnetic condition), `validate_primitive_operations` and magnetic detection (`magmoms`, collinear as a decoration and noncollinear as an axial vector) in `core/spacegroup.py`, `unfoldlab check-symmetry`, `--magmom`, and validation wired into `weights` / `unfold`; `cli/symmetry.py` split out of `cli/analysis.py`; repository layout and README restored; `docs/findings.md` §38 |
| 35 | the band scatter plot moved out of the QE reader into `io/plot_bands.py` and its standard-library-only renderer repaired: out-of-window states were drawn outside the axes box and one non-finite energy destroyed the whole figure; marker area (not radius) made affine in the weight so the two renderers agree; `tests/test_plot_bands.py`; `docs/findings.md` §37 |
| 34 | batched fiber weights: `shared_weights_from_coefficients` computes the squared moduli once for all the primitive k-points that share a stored wavefunction, `SharedWavefunctionGroup` / `compute_plane_wave_unfolding_weights_shared` stream them, and the symmetry-reduced VASP and QE readers use it (6.8x on an eight-point fiber, weights unchanged to `3e-16`); `tests/test_shared_weights.py`; `docs/findings.md` §36 |
| 33 | connected the PAW/ultrasoft augmentation work that an earlier pass left orphaned: `core/augmentation.py` exported, `tests/test_augmentation.py`, the PAW weight bound in `unfoldlab norms`, `Augmentation.lean` documented in `docs/formal-model.md`, and a dropped normalization warning in `AugmentationReport.summary` fixed; coverage added to the review checklist; `docs/findings.md` §35 |
| 32 | weak superlattice perturbations and shadow bands: `Unfolding/Perturbation.lean` (the 2x2 problem diagonalized, the fiber sum rule for a main band and its shadow, the `O(v^2)` shadow intensity with proved bounds, level repulsion, and the exact inversion `|v| = dE sqrt(w+ w-)`), `core/perturbation.py`, `unfoldlab shadow`, `tests/test_perturbation.py`; `docs/findings.md` §34 |
| 10 | spin texture for noncollinear unfolding; adaptive broadening; this tracker |
| 11 | spin texture for the QE reader; orbital-projected fat bands for tight-binding/Wannier models (Lean + Python + CLI); `seedname_tb.dat` reader; repository hygiene (`.gitignore`, untracked bytecode) |
| 13 | configurational averaging for disordered supercells (`Ensemble.lean`, `core/ensemble.py`, `unfoldlab ensemble`); constant-energy cuts and closed-form energy windows (`Windows.lean`, `core/windows.py`, `unfoldlab cut`); pass-12 documentation closed out (`docs/findings.md` §15, `docs/formal-model.md` entries for `Chunking.lean` and `Ideal.lean`) |
| 12 | band-chunked WAVECAR reading with `Chunking.lean` behind it; k-map parsing errors reported with a line number; HDF5 export of an `EffectiveBandStructure` |
| 14 | degenerate multiplets and the gauge dependence of per-band weights (`Degeneracy.lean`, `core/degeneracy.py`, `unfoldlab degeneracy`); time reversal as an unfolding operation (`TimeReversal.lean`, `--time-reversal`), which also subsumes the Γ-only `K = 0` lemma; `docs/findings.md` §16 |
| 16 | mass defects in a phonon supercell: `Phonons.lean`, `site_scaling` in the tight-binding kernel, `site_masses` in the phonon API, reader and CLI; repaired the Lean build (`Truncation.lean`); `docs/findings.md` §18 |
| 15 | completeness of the stored wavefunction: scale invariance, truncation and perturbation bounds for the weight (`Normalization.lean`), stored-state norm diagnostics for both readers and `unfoldlab norms`; `docs/findings.md` §17 |
| 17 | type-cleaned the whole package: `mypy unfoldlab` from 71 errors to zero and promoted to a gate; `TransformLike` / `OrbitalGroups` / `BroadeningKind` aliases threaded through the readers, kernels and CLI; `TransformationMatrix.from_values`; `default_weights` / `default_distances`; fixed `datetime.UTC` on Python 3.10 and a fall-through in `unfoldlab norms`; `tests/test_typing_contracts.py`; `docs/findings.md` §19 |
| 19 | non-orthogonal (LCAO) unfolding: closed the four `sorry`s in `Overlap.lean` and wired it into `Main.lean`; `core/lcao.py` (overlap kernel, overlap-aware weights, generalized eigenproblem, `diagnose_overlap_neglect`); `tests/test_lcao.py` with the closed-form non-orthogonal chain; `docs/findings.md` §21 |
| 21 | Bloch-sum conventions: `Unfolding/Convention.lean` (a per-orbital phase is a gauge of the weight and of every fat band; a per-cell phase translates the distribution in k with the fiber sum rule intact; the LCAO kernel must be regauged with the state), `core/convention.py`, `tests/test_convention.py`; `docs/findings.md` §23 |
| 20 | the across-bands sum rule in a non-orthogonal basis: `Unfolding/OverlapCompleteness.lean` (class-function property of the `ovKernel` summand, `IsCosetRepr.sum_neg`, the twisted convolution theorem `ovKernel_mul_of_inverse`, and `sum_ovWeight_states`), imported from `Main.lean`; `docs/findings.md` §22 |
| 18 | split the 1766-line `cli/main.py` into `app` / `common` / `commands` / `diagnostics` / `backends` / `guided`, with `tests/test_cli_structure.py` pinning the command surface, the absence of import cycles and the module-length limit; `docs/findings.md` §20 |
| 22 | strained / approximately commensurate supercells: `Unfolding/Strain.lean` (commensurability is integrality; the k error of residual strain is linear in k and no integer transform removes it; rounding is optimal), `core/strain.py`, `unfoldlab strain`, `tests/test_strain.py`; `docs/findings.md` §24 |
| 23 | reading numbers off a spectral function: `Unfolding/Dispersion.lean` (exact parabolic peak refinement, exact cubic second difference, the merge/resolve sandwich at one and two broadening widths), `core/dispersion.py`, `unfoldlab dispersion`, `tests/test_dispersion.py`; `docs/findings.md` §25 |
| 24 | conditioning of the LCAO overlap: `Unfolding/Conditioning.lean` (the standard and generalized problems are equivalent for any `X` with `Xᴴ S X = 1`; a null direction of `S` has zero `S`-norm and no weight; an `O(n²)` diagonal-dominance bound on the smallest eigenvalue), `core/conditioning.py`, `unfoldlab overlap`, `tests/test_conditioning.py`; `docs/findings.md` §26 |
| 27 | tracking peaks into branches: `Unfolding/Tracking.lean` (sorting is the optimal assignment; greedy nearest-partner tracking is strictly worse; a degeneracy leaves the branch labels undetermined), `core/tracking.py`, `unfoldlab branches`, `tests/test_tracking.py`; `docs/findings.md` §29 |
| 29 | layer-resolved unfolding: `Unfolding/Layers.lean` (the layer split of a weight is exact at fixed k, acquires a cross term when summed over k, and is exact for a projection diagonal in the same basis), `core/layers.py` with the planar-averaged density from its Fourier coefficients, `unfoldlab layers`, `tests/test_layers.py`; `docs/findings.md` §31 |
| 30 | Brillouin-zone meshes: `Unfolding/Mesh.lean` (a mesh is a fiber; the unfolded mesh is the mesh of `M @ T` with the same shift; a shifted mesh misses Gamma and an odd division the zone boundary; the resolution is half a spacing; the unfolded mesh is generally not a product mesh), `core/mesh.py`, `unfoldlab mesh`, `tests/test_mesh.py`; `docs/findings.md` §32 |
| 31 | twisted and multi-layer stacks: `Unfolding/Twist.lean` (a stack is two folding problems sharing one supercell; the multiplicities are fixed by the cell volumes; the transfer between the two primitive zones is the identity in Cartesian reciprocal space; a state has total weight one in *both* layers' zones, so only a layer-projected weight answers the layer question; the commensurate hexagonal angles and the `m^2 - m n + n^2` moire cell), `twist/commensurate.py`, `unfoldlab twist`, `tests/test_twist_commensurate.py`; `docs/findings.md` §33 |
| 28 | slabs, wires and partial periodicity: `Unfolding/Slab.lean` (a slab fiber is flat; the perpendicular plane-wave index does not enter the matching test, so collapsing the list over it is exact; folding along the vacuum direction invents a band whose only parameter is the vacuum thickness), `core/slab.py`, `unfoldlab slab`, `tests/test_slab.py`; `docs/findings.md` §30 |
| 26 | the unfolded density of states: `Unfolding/Dos.lean` (unfolding conserves the density of states over a complete fiber; per primitive cell it is the supercell curve divided by `|det T|`; both integrate to the band count; an incomplete fiber undercounts and weights in `[0, 1]` cannot exceed the unweighted curve), `core/dos.py`, `unfoldlab dos`, `tests/test_dos.py`; `normalized_kpoint_weights` shared with `core/fermi.py` and `broadening_table` / `cumulative_kernel` shared out of `core/windows.py`; `docs/findings.md` §28 |
| 25 | the Fermi level and the energy reference: `Unfolding/Fermi.lean` (the occupied weight is strictly monotone in the chemical potential and sweeps `(0, ∑ w)`, so the Fermi level exists and is unique; it is equivariant under a change of reference; the count is Lipschitz with constant `(∑ w)/(4T)`), `core/fermi.py`, `unfoldlab fermi`, `tests/test_fermi.py`; the `cli/diagnostics.py` split into `diagnostics` + `analysis`; `docs/findings.md` §27 |
