"""Unfolding of tight-binding, Wannier, and model-Hamiltonian supercells.

Why this exists
---------------

Every backend in the package so far unfolds a *plane-wave* state: a list of
coefficients indexed by supercell reciprocal-lattice vectors.  A tight-binding,
Wannier, or model-Hamiltonian supercell state is not of that form.  It is given
by one amplitude per ``(orbital, primitive cell)`` pair, the primitive cells
being the ``|det T|`` translates that build the supercell, and its unfolding
weight is the squared norm of its discrete Fourier component at the primitive
k-point,

``A(k, a) = sum_R exp(-2 pi i k . R) c(a, R)``,
``W(k) = sum_a |A(k, a)|^2 / (|det T| sum_{a,R} |c(a, R)|^2)``.

Conventions
-----------

``R`` is a primitive lattice translation written in *primitive fractional*
coordinates, so it is an integer vector, and ``k`` is a primitive k-point in
fractional reciprocal coordinates; ``k . R`` is then the ordinary dot product
and the Cartesian phase is ``exp(-i k_cart . R_cart) = exp(-2 pi i k . R)``.

With row-vector lattices and ``A_sc = T @ A_pc``, a supercell lattice vector in
primitive fractional coordinates is ``n @ T`` for integer ``n``, i.e. the
supercell lattice is ``Z^3 @ T``.  The primitive cells inside the supercell are
therefore the classes of ``Z^3 / Z^3 @ T``, enumerated by
:func:`supercell_cells`.

Exactness and correctness
-------------------------

The theory is formalized in ``RequestProject/Unfolding/TightBinding.lean``:

* ``UnfoldLab.sum_dualChar`` — the cells ``Z^3 / Z^3 @ T`` and the reciprocal
  classes that label a fiber are dual finite abelian groups.  This is why the
  fiber phases are an orthogonal basis and why the recipe below is exact for a
  *non-diagonal* supercell, not only for a diagonal one.
* ``UnfoldLab.sum_normSq_tbAmplitude`` — Parseval for the supercell.
* ``UnfoldLab.sum_tbWeight_eq_one`` — consequently the weights of one complete
  fiber add up to one for every state, which
  :func:`tight_binding_weights` reproduces numerically.
* ``UnfoldLab.tbAmplitude_congr_left`` — the weights do not depend on how the
  primitive k-points are wrapped into the Brillouin zone.

Membership of the supercell lattice is tested with the exact integer adjugate
criterion, so :func:`validate_supercell_cells` never depends on a tolerance.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.kpoints import fiber_kpoints
from unmochan.core.lattice_quotient import coset_representatives
from unmochan.core.numerics import as_int_array
from unmochan.core.transformations import TransformationMatrix

OrbitalGroups: TypeAlias = "Sequence[Sequence[int] | NDArray[np.int64]]"
"""Orbital index sets for projected ("fat band") weights.

Each group is either a plain sequence of orbital indices or an integer array,
so the normalized output of :func:`_validate_orbital_groups` can be fed straight
back into the weight kernels.
"""

__all__ = [
    "TightBindingSumRules",
    "cell_residue_keys",
    "validate_site_scaling",
    "TightBindingModel",
    "diagnose_tight_binding_weights",
    "supercell_cells",
    "supercell_bloch_hamiltonian",
    "tight_binding_orbital_weights",
    "tight_binding_weights",
    "unfold_tight_binding",
    "unfold_tight_binding_path_projected",
    "unfold_tight_binding_model",
    "unfold_tight_binding_path",
    "validate_supercell_cells",
]


def supercell_cells(transform: TransformationMatrix) -> NDArray[np.int64]:
    """The ``|det T|`` primitive cells that build the supercell.

    Returns integer primitive-fractional translations, one per class of
    ``Z^3 / Z^3 @ T``, each class exactly once and with the origin first.  These
    are the cells that index a tight-binding supercell state.
    """

    return coset_representatives(transform.matrix.T)


def _in_supercell_lattice(
    vectors: NDArray[np.int64], transform: TransformationMatrix
) -> NDArray[np.bool_]:
    """Exact test of ``v in Z^3 @ T`` for each row of ``vectors``.

    ``v = n @ T`` has the integer solution ``n = v @ adj(T) / det(T)`` exactly
    when ``det(T)`` divides every component of ``v @ adj(T)`` (Cramer's rule),
    which is the transpose of the criterion proved as
    ``UnfoldLab.mem_range_mulVec_iff_adjugate``.
    """

    projected = np.asarray(vectors, dtype=np.int64) @ transform.adjugate
    return np.all(projected % transform.determinant == 0, axis=-1)


def validate_supercell_cells(
    cells: ArrayLike, transform: TransformationMatrix
) -> NDArray[np.int64]:
    """Check that ``cells`` is a complete, irredundant list of supercell cells.

    A tight-binding supercell state must carry exactly one amplitude per
    primitive cell of the supercell.  The check is the numerical counterpart of
    the hypothesis ``IsCosetRepr Tᵀ Rs`` of ``UnfoldLab.sum_tbWeight_eq_one``:
    ``|det T|`` cells, pairwise inequivalent modulo the supercell lattice.
    """

    array = as_int_array(np.asarray(cells), name="cells")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    expected = transform.multiplicity
    if array.shape[0] != expected:
        raise ValueError(f"expected {expected} = |det T| supercell cells, got {array.shape[0]}")
    differences = array[:, np.newaxis, :] - array[np.newaxis, :, :]
    equivalent = _in_supercell_lattice(differences.reshape(-1, 3), transform)
    equivalent = equivalent.reshape(expected, expected)
    np.fill_diagonal(equivalent, False)
    if bool(np.any(equivalent)):
        first = np.argwhere(equivalent)[0]
        raise ValueError(
            "cells must be pairwise inequivalent modulo the supercell lattice: "
            f"rows {int(first[0])} and {int(first[1])} describe the same cell"
        )
    return array


def tight_binding_weights(
    cells: ArrayLike,
    coefficients: ArrayLike,
    kpoints: ArrayLike,
    *,
    multiplicity: int | None = None,
) -> NDArray[np.float64]:
    """Unfolding weights of tight-binding supercell states.

    ``coefficients`` has shape ``(n_states, n_orbitals, n_cells)`` and holds the
    amplitude of each state on each orbital of each primitive cell of the
    supercell; ``cells`` are the corresponding ``(n_cells, 3)`` integer
    translations, and ``kpoints`` are ``(n_kpoints, 3)`` primitive k-points in
    fractional coordinates.

    Returns a ``(n_kpoints, n_states)`` array of weights in ``[0, 1]``.  When
    the k-points form a complete fiber the weights of each state add up to one
    (``UnfoldLab.sum_tbWeight_eq_one``).

    ``multiplicity`` defaults to the number of cells, which is the right
    normalization whenever ``cells`` is a complete list; pass it explicitly only
    when deliberately working with a subset.
    """

    cell_array = as_int_array(np.asarray(cells), name="cells")
    if cell_array.ndim != 2 or cell_array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_states, n_orbitals, n_cells)")
    if coeffs.shape[2] != cell_array.shape[0]:
        raise ValueError("coefficients must have one entry per cell")
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")

    count = int(multiplicity) if multiplicity is not None else cell_array.shape[0]
    if count <= 0:
        raise ValueError("multiplicity must be positive")

    # (n_kpoints, n_cells) phases, then one contraction over the cell axis.
    phases = np.exp(-2j * np.pi * (points @ cell_array.T.astype(float)))
    amplitudes = np.einsum("kR,soR->kso", phases, coeffs, optimize=True)
    partial = (amplitudes.real**2 + amplitudes.imag**2).sum(axis=2)
    total = (coeffs.real**2 + coeffs.imag**2).sum(axis=(1, 2))
    safe = np.where(total > 0.0, total, 1.0)
    weights = partial / (float(count) * safe[np.newaxis, :])
    return np.where(total[np.newaxis, :] > 0.0, weights, 0.0)


def _validate_orbital_groups(groups: OrbitalGroups, n_orbitals: int) -> list[NDArray[np.int64]]:
    """Normalize orbital groups to arrays of distinct in-range indices."""

    normalized: list[NDArray[np.int64]] = []
    for position, group in enumerate(groups):
        indices = np.asarray(list(group), dtype=np.int64).reshape(-1)
        if indices.size and (int(indices.min()) < 0 or int(indices.max()) >= n_orbitals):
            raise ValueError(
                f"orbital group {position} refers to an orbital outside 0..{n_orbitals - 1}"
            )
        if np.unique(indices).size != indices.size:
            raise ValueError(f"orbital group {position} repeats an orbital")
        normalized.append(indices)
    return normalized


def tight_binding_orbital_weights(
    cells: ArrayLike,
    coefficients: ArrayLike,
    kpoints: ArrayLike,
    groups: OrbitalGroups,
    *,
    multiplicity: int | None = None,
) -> NDArray[np.float64]:
    """Orbital-projected ("fat band") tight-binding unfolding weights.

    Same inputs as :func:`tight_binding_weights` plus ``groups``, a list of
    orbital index sets.  The numerator is restricted to the orbitals of a group
    while the denominator keeps the *full* state norm, so a group's weight is
    its share of the total weight rather than a renormalized quantity of its
    own.  This is ``UnfoldLab.tbOrbitalWeight``, and the guarantees it comes
    with are proved there:

    * disjoint groups add (``tbOrbitalWeight_union_of_disjoint``), so a
      partition of the orbitals reproduces :func:`tight_binding_weights`;
    * a group weight never exceeds the total (``tbOrbitalWeight_le_tbWeight``);
    * over a complete fiber a group's weights add up to the group's share of
      the state norm (``sum_tbOrbitalWeight_eq_norm_share``), which is one for
      the full orbital set.

    Returns an array of shape ``(n_groups, n_kpoints, n_states)``.
    """

    cell_array = as_int_array(np.asarray(cells), name="cells")
    if cell_array.ndim != 2 or cell_array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_states, n_orbitals, n_cells)")
    if coeffs.shape[2] != cell_array.shape[0]:
        raise ValueError("coefficients must have one entry per cell")
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")

    count = int(multiplicity) if multiplicity is not None else cell_array.shape[0]
    if count <= 0:
        raise ValueError("multiplicity must be positive")
    index_groups = _validate_orbital_groups(groups, coeffs.shape[1])

    phases = np.exp(-2j * np.pi * (points @ cell_array.T.astype(float)))
    amplitudes = np.einsum("kR,soR->kso", phases, coeffs, optimize=True)
    resolved = amplitudes.real**2 + amplitudes.imag**2
    total = (coeffs.real**2 + coeffs.imag**2).sum(axis=(1, 2))
    safe = np.where(total > 0.0, total, 1.0)
    keep = total[np.newaxis, :] > 0.0

    projected = np.empty((len(index_groups), points.shape[0], coeffs.shape[0]), dtype=float)
    for position, indices in enumerate(index_groups):
        partial = resolved[:, :, indices].sum(axis=2)
        weights = partial / (float(count) * safe[np.newaxis, :])
        projected[position] = np.where(keep, weights, 0.0)
    return projected


def unfold_tight_binding(
    cells: ArrayLike,
    coefficients: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unfold supercell states onto the complete fiber over one k-point.

    Returns ``(kpoints, weights)`` with ``kpoints`` of shape ``(|det T|, 3)``
    from :func:`unmochan.core.kpoints.fiber_kpoints` and ``weights`` of shape
    ``(|det T|, n_states)`` summing to one along the k-point axis.
    """

    validate_supercell_cells(cells, transform)
    points = fiber_kpoints(supercell_kpoint, transform)
    weights = tight_binding_weights(
        cells, coefficients, points, multiplicity=transform.multiplicity
    )
    return points, weights


@dataclass(frozen=True)
class TightBindingSumRules:
    """Both sum rules obeyed by a table of tight-binding unfolding weights.

    ``max_fiber_deviation`` is the largest ``|sum_k W(k, s) - 1|`` over the
    states ``s``: the weights of one state must spread over the fiber without
    creating or destroying spectral weight
    (``UnfoldLab.sum_tbWeight_eq_one``).

    ``max_band_deviation`` is the largest ``|sum_s W(k, s) - n_orbitals|`` over
    the k-points, and is only meaningful when the states form a complete
    orthonormal set, i.e. when there are ``|det T| * n_orbitals`` of them
    (``UnfoldLab.sum_tbWeight_states``).  It is ``None`` otherwise, because a
    truncated band window legitimately carries less weight.
    """

    max_fiber_deviation: float
    max_band_deviation: float | None
    n_kpoints: int
    n_states: int
    n_orbitals: int | None

    def violated(self, atol: float = 1e-8) -> bool:
        if self.max_fiber_deviation > atol:
            return True
        return self.max_band_deviation is not None and self.max_band_deviation > atol

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "max_fiber_deviation": self.max_fiber_deviation,
            "max_band_deviation": self.max_band_deviation,
            "n_kpoints": self.n_kpoints,
            "n_states": self.n_states,
            "n_orbitals": self.n_orbitals,
        }


def diagnose_tight_binding_weights(
    weights: ArrayLike, *, n_orbitals: int | None = None
) -> TightBindingSumRules:
    """Check the two sum rules on a ``(n_kpoints, n_states)`` weight table.

    The k-points are assumed to be one complete fiber, as produced by
    :func:`unfold_tight_binding`.  Pass ``n_orbitals`` to enable the
    across-bands check, which additionally requires the state list to be the
    complete set of ``n_kpoints * n_orbitals`` supercell eigenstates.
    """

    table = np.asarray(weights, dtype=float)
    if table.ndim != 2:
        raise ValueError("weights must have shape (n_kpoints, n_states)")
    n_kpoints, n_states = table.shape
    fiber = float(np.max(np.abs(table.sum(axis=0) - 1.0))) if table.size else 0.0
    band: float | None = None
    if n_orbitals is not None and n_states == n_kpoints * int(n_orbitals):
        band = float(np.max(np.abs(table.sum(axis=1) - float(n_orbitals))))
    return TightBindingSumRules(
        max_fiber_deviation=fiber,
        max_band_deviation=band,
        n_kpoints=int(n_kpoints),
        n_states=int(n_states),
        n_orbitals=None if n_orbitals is None else int(n_orbitals),
    )


@dataclass(frozen=True)
class TightBindingModel:
    """A primitive-cell tight-binding model.

    ``hoppings`` maps an integer primitive translation ``D`` to the matrix
    ``t(D)`` with ``t(D)[a, b] = <0 a| H |D b>``.  Hermiticity of the
    Hamiltonian requires ``t(-D) == t(D).conj().T``; :meth:`hermitized` returns
    a model in which that holds, so only one member of each pair has to be
    supplied.

    The hoppings are also kept as the packed arrays :attr:`hopping_cells` and
    :attr:`hopping_blocks`, so that every Bloch Hamiltonian is one contraction
    rather than a Python loop over the dictionary.
    """

    n_orbitals: int
    hoppings: dict[tuple[int, int, int], NDArray[np.complex128]] = field(default_factory=dict)
    _packed_cells: NDArray[np.int64] = field(
        init=False, repr=False, compare=False, default=cast("NDArray[np.int64]", None)
    )
    _packed_blocks: NDArray[np.complex128] = field(
        init=False, repr=False, compare=False, default=cast("NDArray[np.complex128]", None)
    )

    def __post_init__(self) -> None:
        if self.n_orbitals <= 0:
            raise ValueError("n_orbitals must be positive")
        normalized: dict[tuple[int, int, int], NDArray[np.complex128]] = {}
        for cell, matrix in self.hoppings.items():
            key = tuple(int(component) for component in cell)
            if len(key) != 3:
                raise ValueError("hopping keys must be integer 3-vectors")
            block = np.asarray(matrix, dtype=np.complex128)
            if block.shape != (self.n_orbitals, self.n_orbitals):
                raise ValueError("hopping blocks must have shape (n_orbitals, n_orbitals)")
            normalized[key] = normalized.get(key, 0) + block  # type: ignore[assignment]
        object.__setattr__(self, "hoppings", normalized)
        cells = np.array(list(normalized.keys()), dtype=np.int64).reshape(-1, 3)
        blocks = np.array(list(normalized.values()), dtype=np.complex128).reshape(
            -1, self.n_orbitals, self.n_orbitals
        )
        object.__setattr__(self, "_packed_cells", cells)
        object.__setattr__(self, "_packed_blocks", blocks)

    @property
    def hopping_cells(self) -> NDArray[np.int64]:
        """The ``(n_hoppings, 3)`` integer translations, in dictionary order."""

        return self._packed_cells

    @property
    def hopping_blocks(self) -> NDArray[np.complex128]:
        """The ``(n_hoppings, n_orbitals, n_orbitals)`` amplitudes ``t(D)``."""

        return self._packed_blocks

    def hermitized(self) -> TightBindingModel:
        """Return the model completed so that ``t(-D) == t(D)^dagger``.

        A hopping listed in only one direction has its conjugate partner filled
        in with the *same* amplitude, so ``t(D) = -1`` means a hopping of
        strength one in both directions rather than one half.  A pair that is
        listed in both directions is replaced by its Hermitian part, which
        leaves an already-consistent pair untouched; the on-site block is
        likewise replaced by its Hermitian part.  The operation is idempotent.
        """

        blocks: dict[tuple[int, int, int], NDArray[np.complex128]] = {
            cell: np.array(matrix, dtype=np.complex128) for cell, matrix in self.hoppings.items()
        }
        for cell, matrix in self.hoppings.items():
            opposite = (-cell[0], -cell[1], -cell[2])
            if opposite not in blocks:
                blocks[opposite] = matrix.conj().T
        symmetrized = {
            cell: 0.5 * (matrix + blocks[(-cell[0], -cell[1], -cell[2])].conj().T)
            for cell, matrix in blocks.items()
        }
        return TightBindingModel(self.n_orbitals, symmetrized)

    def bloch_hamiltonian(self, kpoint: ArrayLike) -> NDArray[np.complex128]:
        """``H(k) = sum_D exp(2 pi i k . D) t(D)`` in fractional coordinates."""

        k = np.asarray(kpoint, dtype=float)
        if k.shape != (3,):
            raise ValueError("kpoint must have shape (3,)")
        return self.bloch_hamiltonians(k[np.newaxis, :])[0]

    def bloch_hamiltonians(self, kpoints: ArrayLike) -> NDArray[np.complex128]:
        """``H(k)`` for many k-points at once, shape ``(n_kpoints, n, n)``.

        One contraction over the hoppings replaces a Python loop per k-point,
        which is what makes a band path over a Wannier model cheap.
        """

        points = np.asarray(kpoints, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("kpoints must have shape (n_kpoints, 3)")
        if self._packed_cells.shape[0] == 0:
            return np.zeros(
                (points.shape[0], self.n_orbitals, self.n_orbitals),
                dtype=np.complex128,
            )
        phases = np.exp(2j * np.pi * (points @ self._packed_cells.T.astype(float)))
        return np.tensordot(phases, self._packed_blocks, axes=(1, 0))

    def bands(self, kpoints: ArrayLike) -> NDArray[np.float64]:
        """Eigenvalues of the primitive Bloch Hamiltonian, sorted ascending."""

        return np.asarray(np.linalg.eigvalsh(self.bloch_hamiltonians(kpoints)), dtype=np.float64)


def cell_residue_keys(
    vectors: NDArray[np.int64], transform: TransformationMatrix
) -> NDArray[np.int64]:
    """A canonical label of each vector's class in ``Z^3 / Z^3 @ T``.

    ``v @ adj(T) mod |det T|`` is a group homomorphism whose kernel is exactly
    the supercell lattice — that is the content of the adjugate criterion used
    by :func:`_in_supercell_lattice` — so it separates the classes exactly, and
    two cells are the same cell precisely when their labels agree.  Hashing
    these labels turns the "which cell is ``R + D``?" lookup from a scan over
    all cells into a dictionary hit.
    """

    modulus = abs(transform.determinant)
    return (np.asarray(vectors, dtype=np.int64) @ transform.adjugate) % modulus


@dataclass(frozen=True)
class _SupercellCouplings:
    """The q-independent skeleton of a supercell Bloch Hamiltonian.

    For every hopping ``D`` and every cell ``R`` there is exactly *one* cell
    ``R'`` with ``R' = R + D`` modulo the supercell lattice, so the supercell
    Hamiltonian has ``n_hoppings * n_cells`` non-zero blocks, not
    ``n_cells**2``.  Their positions and the translations ``L = R + D - R'``
    that carry the k-dependence are the same for every k-point, so they are
    computed once and reused.
    """

    pairs: NDArray[np.int64]
    blocks: NDArray[np.int64]
    translations: NDArray[np.float64]
    n_cells: int
    n_orbitals: int


def _supercell_couplings(
    model: TightBindingModel,
    transform: TransformationMatrix,
    cells: NDArray[np.int64],
) -> _SupercellCouplings:
    n_cells = cells.shape[0]
    hopping_cells = model.hopping_cells
    n_hoppings = hopping_cells.shape[0]

    index_of: dict[bytes, int] = {
        key.tobytes(): index for index, key in enumerate(cell_residue_keys(cells, transform))
    }
    if len(index_of) != n_cells:
        raise ValueError("cells must be pairwise inequivalent modulo the supercell")

    # targets[d, r] = R_r + D_d, one row per (hopping, cell) pair.
    targets = (cells[np.newaxis, :, :] + hopping_cells[:, np.newaxis, :]).reshape(-1, 3)
    keys = cell_residue_keys(targets, transform)
    columns = np.empty(targets.shape[0], dtype=np.int64)
    for position, key in enumerate(keys):
        found = index_of.get(key.tobytes())
        if found is None:  # pragma: no cover - impossible for a complete transversal
            raise ValueError("cells are not a complete set of supercell cells")
        columns[position] = found

    rows = np.tile(np.arange(n_cells, dtype=np.int64), n_hoppings)
    blocks = np.repeat(np.arange(n_hoppings, dtype=np.int64), n_cells)
    translations = (targets - cells[columns]).astype(float)
    return _SupercellCouplings(
        pairs=rows * n_cells + columns,
        blocks=blocks,
        translations=translations,
        n_cells=n_cells,
        n_orbitals=model.n_orbitals,
    )


def _assemble_supercell_hamiltonian(
    couplings: _SupercellCouplings,
    blocks: NDArray[np.complex128],
    supercell_kpoint: NDArray[np.float64],
) -> NDArray[np.complex128]:
    n_cells = couplings.n_cells
    n_orbitals = couplings.n_orbitals
    stacked = np.zeros((n_cells * n_cells, n_orbitals, n_orbitals), dtype=np.complex128)
    if couplings.pairs.size:
        phases = np.exp(2j * np.pi * (couplings.translations @ supercell_kpoint))
        contributions = phases[:, np.newaxis, np.newaxis] * blocks[couplings.blocks]
        np.add.at(stacked, couplings.pairs, contributions)
    return (
        stacked.reshape(n_cells, n_cells, n_orbitals, n_orbitals)
        .transpose(0, 2, 1, 3)
        .reshape(n_cells * n_orbitals, n_cells * n_orbitals)
    )


def validate_site_scaling(
    scaling: ArrayLike | None, n_cells: int, n_orbitals: int
) -> NDArray[np.float64] | None:
    """Normalize a ``(n_cells, n_orbitals)`` table of positive site factors.

    ``None`` passes through unchanged, which is the "no rescaling" case.  The
    entries must be strictly positive because the factors act as a *congruence*
    ``H -> S H S`` (see :func:`_apply_site_scaling`), and a vanishing or negative
    factor would either delete a site from the basis or flip the sign of half of
    its couplings.
    """

    if scaling is None:
        return None
    array = np.asarray(scaling, dtype=float)
    if array.shape != (n_cells, n_orbitals):
        raise ValueError(
            f"site_scaling must have shape ({n_cells}, {n_orbitals}), got {array.shape}"
        )
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise ValueError("site_scaling entries must be finite and positive")
    return array


def _apply_site_scaling(
    matrix: NDArray[np.complex128], scaling: NDArray[np.float64] | None
) -> NDArray[np.complex128]:
    """Apply the congruence ``H -> S H S`` with ``S = diag(scaling)``.

    This is *not* the same as adding on-site terms: a factor at one site
    rescales a whole row and a whole column, so it changes the couplings of that
    site to its neighbours as well as its own diagonal entry.  A mass defect in
    a phonon supercell is exactly of this form, with
    ``scaling = sqrt(m_reference / m_site)``, and
    ``UnfoldLab.mass_change_not_diagonal_shift`` proves that no on-site term can
    imitate it.  The congruence is real and diagonal, so it preserves
    Hermiticity (``UnfoldLab.dynMatrix_isHermitian``).
    """

    if scaling is None:
        return matrix
    flat = scaling.reshape(-1)
    matrix *= np.outer(flat, flat)
    return matrix


def _apply_perturbations(
    matrix: NDArray[np.complex128],
    n_orbitals: int,
    n_cells: int,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None,
) -> None:
    for cell_index, orbital, value in perturbations or ():
        cell = int(cell_index)
        band = int(orbital)
        if not (0 <= cell < n_cells):
            raise ValueError(f"perturbation cell index {cell} is outside the supercell")
        if not (0 <= band < n_orbitals):
            raise ValueError(f"perturbation orbital {band} is outside the basis")
        index = cell * n_orbitals + band
        shift = np.asarray(value, dtype=np.complex128)
        if shift.size != 1:
            raise ValueError(
                "a perturbation is a single on-site energy shift for one "
                f"(cell, orbital) pair; got an array of shape {shift.shape} for "
                f"cell {cell}, orbital {band}"
            )
        matrix[index, index] += complex(shift.reshape(()))


def supercell_bloch_hamiltonian(
    model: TightBindingModel,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    cells: ArrayLike | None = None,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
    site_scaling: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Bloch Hamiltonian of the supercell built from a primitive model.

    The basis is ``|q; R a> = sum_L exp(2 pi i q . L) |R + L, a>`` with ``L``
    running over the supercell lattice, ordered as ``(cell, orbital)``, and

    ``H[(R a), (R' b)](q) = sum_L exp(2 pi i q . L) t_{a b}(R' + L - R)``.

    ``supercell_kpoint`` is the primitive-fractional coordinate ``q`` of the
    k-point (any representative of the supercell k-point works, because the
    phases only see ``q`` on the supercell lattice).

    ``perturbations`` adds ``(cell_index, orbital, value)`` on-site terms, which
    is how a defect, a distortion, or an alloy configuration is expressed in a
    supercell that has no primitive counterpart.

    ``site_scaling`` is an optional ``(n_cells, n_orbitals)`` table of positive
    factors applied as the congruence ``H -> S H S`` *after* the perturbations.
    Unlike an on-site term it rescales the couplings of a site as well as its
    own energy, which is what a mass defect does to a phonon supercell; see
    :func:`unmochan.core.phonons.supercell_dynamical_matrix`.

    Returns the matrix and the cell list it was built with.
    """

    cell_array = (
        supercell_cells(transform) if cells is None else validate_supercell_cells(cells, transform)
    )
    q = np.asarray(supercell_kpoint, dtype=float)
    if q.shape != (3,):
        raise ValueError("supercell_kpoint must have shape (3,)")

    couplings = _supercell_couplings(model, transform, cell_array)
    scaling = validate_site_scaling(site_scaling, cell_array.shape[0], model.n_orbitals)
    matrix = _assemble_supercell_hamiltonian(couplings, model.hopping_blocks, q)
    _apply_perturbations(matrix, model.n_orbitals, cell_array.shape[0], perturbations)
    matrix = _apply_site_scaling(matrix, scaling)
    return matrix, cell_array


def unfold_tight_binding_model(
    model: TightBindingModel,
    transform: TransformationMatrix,
    supercell_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
    site_scaling: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Diagonalize the supercell model and unfold it onto primitive k-points.

    For each supercell k-point the supercell Bloch Hamiltonian is built and
    diagonalized, and every eigenstate is unfolded onto the ``|det T|``
    primitive k-points of the fiber.  The k-independent structure of the
    supercell Hamiltonian is built once and shared by all the k-points.

    Returns ``(kpoints, energies, weights)`` where ``kpoints`` has shape
    ``(n_supercell_kpoints * |det T|, 3)``, and ``energies`` and ``weights``
    both have shape ``(n_supercell_kpoints * |det T|, n_states)``.  For a
    supercell of a perfect crystal with no perturbations the weights are ``1``
    on exactly one fiber member per band, which is band folding undone.

    ``site_scaling`` is the congruence described in
    :func:`supercell_bloch_hamiltonian`; the eigenvectors of the scaled matrix
    are still orthonormal, so both sum rules survive it unchanged.
    """

    points = np.asarray(supercell_kpoints, dtype=float).reshape(-1, 3)
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    couplings = _supercell_couplings(model, transform, cell_array)
    perturbation_list = list(perturbations or ())
    scaling = validate_site_scaling(site_scaling, n_cells, model.n_orbitals)

    all_kpoints: list[NDArray[np.float64]] = []
    all_energies: list[NDArray[np.float64]] = []
    all_weights: list[NDArray[np.float64]] = []
    for q in points:
        matrix = _assemble_supercell_hamiltonian(couplings, model.hopping_blocks, q)
        _apply_perturbations(matrix, model.n_orbitals, n_cells, perturbation_list)
        matrix = _apply_site_scaling(matrix, scaling)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        # eigenvectors[:, s] is state s in the (cell, orbital) basis.
        coefficients = eigenvectors.T.reshape(-1, n_cells, model.n_orbitals)
        coefficients = np.swapaxes(coefficients, 1, 2)
        fiber = fiber_kpoints(q @ transform.matrix.T, transform)
        weights = tight_binding_weights(cell_array, coefficients, fiber, multiplicity=n_cells)
        all_kpoints.append(fiber)
        all_energies.append(np.tile(eigenvalues, (fiber.shape[0], 1)))
        all_weights.append(weights)

    return (
        np.concatenate(all_kpoints, axis=0),
        np.concatenate(all_energies, axis=0),
        np.concatenate(all_weights, axis=0),
    )


def unfold_tight_binding_path(
    model: TightBindingModel,
    transform: TransformationMatrix,
    primitive_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
    site_scaling: ArrayLike | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unfold a supercell model along a path in the *primitive* Brillouin zone.

    This is the effective band structure a user actually plots: for each
    requested primitive k-point the supercell Hamiltonian is built at the
    supercell k-point that ``k`` folds onto — which is the same matrix for every
    member of a fiber — diagonalized, and each eigenstate is given its weight at
    ``k`` alone rather than over the whole fiber.

    A primitive k-point in fractional coordinates *is* a valid representative of
    the supercell k-point it folds onto, so no wrapping is needed here; that the
    answer does not depend on the representative chosen is
    ``UnfoldLab.tbAmplitude_congr_left``.

    Returns ``(energies, weights)``, both of shape
    ``(n_kpoints, |det T| * n_orbitals)``.  At each k-point the weights of all
    the states add up to ``n_orbitals`` (``UnfoldLab.sum_tbWeight_states``), and
    for a perfect crystal each primitive band shows up with weight one.

    ``site_scaling`` is the congruence described in
    :func:`supercell_bloch_hamiltonian`, used by the phonon layer to express a
    mass defect.
    """

    points = np.asarray(primitive_kpoints, dtype=float).reshape(-1, 3)
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    couplings = _supercell_couplings(model, transform, cell_array)
    perturbation_list = list(perturbations or ())
    scaling = validate_site_scaling(site_scaling, n_cells, model.n_orbitals)

    energies = np.empty((points.shape[0], n_cells * model.n_orbitals), dtype=float)
    weights = np.empty_like(energies)
    for index, k in enumerate(points):
        matrix = _assemble_supercell_hamiltonian(couplings, model.hopping_blocks, k)
        _apply_perturbations(matrix, model.n_orbitals, n_cells, perturbation_list)
        matrix = _apply_site_scaling(matrix, scaling)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        coefficients = np.swapaxes(eigenvectors.T.reshape(-1, n_cells, model.n_orbitals), 1, 2)
        energies[index] = eigenvalues
        weights[index] = tight_binding_weights(
            cell_array, coefficients, k[np.newaxis, :], multiplicity=n_cells
        )[0]
    return energies, weights


def unfold_tight_binding_path_projected(
    model: TightBindingModel,
    transform: TransformationMatrix,
    primitive_kpoints: Sequence[ArrayLike] | ArrayLike,
    groups: OrbitalGroups,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """:func:`unfold_tight_binding_path` with orbital-projected fat bands.

    ``groups`` lists sets of *primitive-cell orbital* indices — a sublattice, a
    layer, an angular-momentum channel, or the Wannier functions whose centres
    lie in a region of the cell.  Every cell of the supercell contributes the
    orbitals of its group, so a group selects ``|det T|`` supercell basis
    functions.

    Returns ``(energies, weights, projected)`` where ``energies`` and
    ``weights`` are as in :func:`unfold_tight_binding_path` and ``projected``
    has shape ``(n_groups, n_kpoints, |det T| * n_orbitals)``.  A partition of
    the orbitals satisfies ``projected.sum(axis=0) == weights`` exactly (up to
    floating point), which is ``UnfoldLab.tbOrbitalWeight_union_of_disjoint``.
    """

    points = np.asarray(primitive_kpoints, dtype=float).reshape(-1, 3)
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    couplings = _supercell_couplings(model, transform, cell_array)
    perturbation_list = list(perturbations or ())
    index_groups = _validate_orbital_groups(groups, model.n_orbitals)

    n_states = n_cells * model.n_orbitals
    energies = np.empty((points.shape[0], n_states), dtype=float)
    weights = np.empty_like(energies)
    projected = np.empty((len(index_groups), points.shape[0], n_states), dtype=float)
    for index, k in enumerate(points):
        matrix = _assemble_supercell_hamiltonian(couplings, model.hopping_blocks, k)
        _apply_perturbations(matrix, model.n_orbitals, n_cells, perturbation_list)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        coefficients = np.swapaxes(eigenvectors.T.reshape(-1, n_cells, model.n_orbitals), 1, 2)
        energies[index] = eigenvalues
        weights[index] = tight_binding_weights(
            cell_array, coefficients, k[np.newaxis, :], multiplicity=n_cells
        )[0]
        projected[:, index, :] = tight_binding_orbital_weights(
            cell_array,
            coefficients,
            k[np.newaxis, :],
            index_groups,
            multiplicity=n_cells,
        )[:, 0, :]
    return energies, weights, projected
