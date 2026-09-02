"""Unfolding of supercell states given in a non-orthogonal (LCAO) basis.

Why this exists
---------------

:mod:`unfoldlab.core.tight_binding` unfolds a supercell state given by one
amplitude per ``(orbital, primitive cell)`` pair with

``A(k, a) = sum_R exp(-2 pi i k . R) c(a, R)``,
``W(k) = sum_a |A(k, a)|^2 / (|det T| sum_{a,R} |c(a, R)|^2)``,

which assumes the orbitals are **orthonormal**.  That is true of Wannier
functions and of a model tight-binding basis, and false of every LCAO code --
SIESTA, CP2K, OpenMX, FHI-aims -- where the basis functions overlap, the
eigenproblem is the generalized one ``H c = E S c`` and ``<psi|psi> = c* S c``
rather than ``sum |c|^2``.  Feeding such coefficients to the orthonormal
formula is wrong in the numerator *and* in the denominator, and the error is
first order in ``S - 1``: it does not cancel between the two.

The formula implemented here is

``W(k) = A(k)* S~(k) A(k) / (|det T| c* S c)``,

with ``S~(k)_{ab} = sum_W exp(2 pi i k . W) S_{ab}(W)`` the reciprocal-space
overlap, ``W`` running over the primitive cells of the supercell.

Conventions
-----------

``S_{ab}(D) = <q; 0 a | q; D b>`` is the *Bloch-twisted* overlap kernel at the
supercell k-point: the primitive-cell overlap summed over supercell images,

``S_{ab}(D) = sum_L exp(2 pi i q . L) S^inf_{ab}(D + L)``,

with ``L`` running over the supercell lattice ``Z^3 @ T`` in primitive
fractional coordinates.  The sum therefore obeys the twisted periodicity

``S_{ab}(D + L) = exp(-2 pi i q . L) S_{ab}(D)``,

so it is determined by its values on the ``|det T|`` cells of the supercell,
which is how it is stored here: an array of shape
``(n_orbitals, n_orbitals, n_cells)``.  This matches the basis
``|q; R a> = sum_L exp(2 pi i q . L) |R + L, a>`` used by
:func:`unfoldlab.core.tight_binding.supercell_bloch_hamiltonian`, so a
Hamiltonian and an overlap built from the same ``q`` and the same cell list are
consistent with each other.

Exactness and correctness
-------------------------

The theory is formalized in ``RequestProject/Unfolding/Overlap.lean``:

* ``UnfoldLab.IsBlochOverlap`` — the twisted periodicity above.
* ``UnfoldLab.sum_ovNumer`` / ``UnfoldLab.sum_ovWeight_eq_one`` — Parseval and
  the fiber sum rule for a non-orthogonal basis, which
  :func:`lcao_weights` reproduces numerically.
* ``UnfoldLab.ovWeight_orthonormal`` — with an orthonormal kernel the weight is
  exactly the orthonormal tight-binding weight, so this module is a strict
  generalization of :func:`unfoldlab.core.tight_binding.tight_binding_weights`.
* ``UnfoldLab.ovNumer_sub_tbNumer`` — the discrepancy between the two
  numerators, which :func:`diagnose_overlap_neglect` measures.
* ``UnfoldLab.ovWeight_smul`` — the weight is invariant under rescaling the
  state, so an eigenvector normalization convention cannot change it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.conditioning import solve_generalized_eigenproblem_truncated
from unfoldlab.core.kpoints import fiber_kpoints
from unfoldlab.core.numerics import as_int_array
from unfoldlab.core.tight_binding import (
    TightBindingModel,
    cell_residue_keys,
    supercell_bloch_hamiltonian,
    supercell_cells,
    tight_binding_weights,
    validate_supercell_cells,
)
from unfoldlab.core.transformations import TransformationMatrix

__all__ = [
    "OverlapNeglect",
    "bloch_overlap_kernel",
    "diagnose_overlap_neglect",
    "lcao_norms",
    "lcao_weights",
    "orthonormal_overlap_kernel",
    "overlap_kernel_from_matrix",
    "overlap_neglect_from_model",
    "reciprocal_overlap",
    "solve_generalized_eigenproblem",
    "supercell_overlap_matrix",
    "unfold_lcao",
    "unfold_lcao_model",
    "unfold_lcao_path",
    "validate_overlap_kernel",
]


def _check_cells(cells: ArrayLike) -> NDArray[np.int64]:
    array = as_int_array(np.asarray(cells), name="cells")
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError("cells must have shape (n_cells, 3)")
    return array


def _check_kernel(overlaps: ArrayLike, n_cells: int) -> NDArray[np.complex128]:
    kernel = np.asarray(overlaps, dtype=np.complex128)
    if kernel.ndim != 3 or kernel.shape[0] != kernel.shape[1]:
        raise ValueError("overlaps must have shape (n_orbitals, n_orbitals, n_cells)")
    if kernel.shape[2] != n_cells:
        raise ValueError("overlaps must have one entry per cell")
    return kernel


def _check_kpoint(kpoint: ArrayLike, name: str) -> NDArray[np.float64]:
    point = np.asarray(kpoint, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must have shape (3,)")
    return point


def _cell_lookup(cells: NDArray[np.int64], transform: TransformationMatrix) -> dict[bytes, int]:
    """Map the residue label of each cell to its index in ``cells``."""

    lookup = {key.tobytes(): index for index, key in enumerate(cell_residue_keys(cells, transform))}
    if len(lookup) != cells.shape[0]:
        raise ValueError("cells must be pairwise inequivalent modulo the supercell")
    return lookup


def orthonormal_overlap_kernel(n_orbitals: int, cells: ArrayLike) -> NDArray[np.complex128]:
    """The kernel of an orthonormal basis, ``S_{ab}(W) = delta_ab delta_{W,0}``.

    This is the ``IsOrthonormalOn`` case of the Lean model; feeding it to
    :func:`lcao_weights` reproduces
    :func:`unfoldlab.core.tight_binding.tight_binding_weights` exactly
    (``UnfoldLab.ovWeight_orthonormal``).
    """

    if n_orbitals <= 0:
        raise ValueError("n_orbitals must be positive")
    cell_array = _check_cells(cells)
    origin = np.flatnonzero(np.all(cell_array == 0, axis=1))
    if origin.size != 1:
        raise ValueError("cells must contain the origin exactly once")
    kernel = np.zeros((n_orbitals, n_orbitals, cell_array.shape[0]), dtype=np.complex128)
    kernel[np.arange(n_orbitals), np.arange(n_orbitals), int(origin[0])] = 1.0
    return kernel


def bloch_overlap_kernel(
    overlaps: Mapping[tuple[int, int, int], ArrayLike],
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    cells: ArrayLike | None = None,
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Fold a primitive-cell overlap table into the supercell kernel.

    ``overlaps`` maps an integer primitive translation ``D`` to the block
    ``S^inf(D)[a, b] = <phi_{a,0} | phi_{b,D}>``, the same layout the hoppings
    of :class:`~unfoldlab.core.tight_binding.TightBindingModel` use.  The
    supercell kernel is the image sum

    ``S_{ab}(W) = sum_{D = W mod supercell} exp(2 pi i q . (D - W)) S^inf_{ab}(D)``.

    Returns ``(kernel, cells)`` with ``kernel`` of shape
    ``(n_orbitals, n_orbitals, n_cells)``.
    """

    q = _check_kpoint(supercell_kpoint, "supercell_kpoint")
    cell_array = (
        supercell_cells(transform) if cells is None else validate_supercell_cells(cells, transform)
    )
    lookup = _cell_lookup(cell_array, transform)

    blocks = {
        tuple(int(component) for component in key): np.asarray(value, dtype=np.complex128)
        for key, value in overlaps.items()
    }
    if not blocks:
        raise ValueError("overlaps must contain at least one block")
    shapes = {block.shape for block in blocks.values()}
    if len(shapes) != 1:
        raise ValueError("every overlap block must have the same shape")
    shape = shapes.pop()
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError("overlap blocks must be square (n_orbitals, n_orbitals)")
    n_orbitals = int(shape[0])

    kernel = np.zeros((n_orbitals, n_orbitals, cell_array.shape[0]), dtype=np.complex128)
    translations = np.array(list(blocks.keys()), dtype=np.int64).reshape(-1, 3)
    keys = cell_residue_keys(translations, transform)
    for position, (translation, block) in enumerate(blocks.items()):
        index = lookup.get(keys[position].tobytes())
        if index is None:  # pragma: no cover - a complete transversal has no gaps
            raise ValueError("cells are not a complete set of supercell cells")
        shift = np.asarray(translation, dtype=float) - cell_array[index].astype(float)
        kernel[:, :, index] += np.exp(2j * np.pi * float(shift @ q)) * block
    return kernel, cell_array


def reciprocal_overlap(
    cells: ArrayLike, overlaps: ArrayLike, kpoints: ArrayLike
) -> NDArray[np.complex128]:
    """``S~(k)_{ab} = sum_W exp(2 pi i k . W) S_{ab}(W)``, shape ``(n_k, n, n)``.

    This is ``UnfoldLab.ovKernel``.  For an orthonormal basis it is the identity
    at every k-point, and how far it is from the identity is exactly how wrong
    the orthonormal weight formula is (``UnfoldLab.ovNumer_sub_tbNumer``).
    """

    cell_array = _check_cells(cells)
    kernel = _check_kernel(overlaps, cell_array.shape[0])
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")
    phases = np.exp(2j * np.pi * (points @ cell_array.T.astype(float)))
    return np.asarray(np.einsum("kW,abW->kab", phases, kernel, optimize=True), dtype=np.complex128)


def supercell_overlap_matrix(
    cells: ArrayLike,
    overlaps: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
) -> NDArray[np.complex128]:
    """The dense supercell overlap in the ``(cell, orbital)`` basis.

    ``M[(R a), (R' b)] = S_{ab}(R' - R)``, the value at a general separation
    being recovered from the stored transversal by the twisted periodicity
    ``S(D) = exp(-2 pi i q . (D - W)) S(W)`` with ``W`` the representative of
    ``D`` (``UnfoldLab.IsBlochOverlap``).  The ordering matches
    :func:`unfoldlab.core.tight_binding.supercell_bloch_hamiltonian`, so ``H``
    and ``M`` can be handed to :func:`solve_generalized_eigenproblem` together.
    """

    cell_array = validate_supercell_cells(cells, transform)
    kernel = _check_kernel(overlaps, cell_array.shape[0])
    q = _check_kpoint(supercell_kpoint, "supercell_kpoint")
    n_cells = cell_array.shape[0]
    n_orbitals = kernel.shape[0]
    lookup = _cell_lookup(cell_array, transform)

    separations = (cell_array[np.newaxis, :, :] - cell_array[:, np.newaxis, :]).reshape(-1, 3)
    keys = cell_residue_keys(separations, transform)
    indices = np.empty(separations.shape[0], dtype=np.int64)
    for position, key in enumerate(keys):
        found = lookup.get(key.tobytes())
        if found is None:  # pragma: no cover - impossible for a complete transversal
            raise ValueError("cells are not a complete set of supercell cells")
        indices[position] = found
    shifts = separations.astype(float) - cell_array[indices].astype(float)
    phases = np.exp(-2j * np.pi * (shifts @ q))

    blocks = kernel[:, :, indices]  # (n_orbitals, n_orbitals, n_cells**2)
    blocks = blocks * phases[np.newaxis, np.newaxis, :]
    matrix = blocks.reshape(n_orbitals, n_orbitals, n_cells, n_cells)
    return np.asarray(
        matrix.transpose(2, 0, 3, 1).reshape(n_cells * n_orbitals, n_cells * n_orbitals),
        dtype=np.complex128,
    )


def validate_overlap_kernel(
    cells: ArrayLike,
    overlaps: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    atol: float = 1e-8,
    require_positive_definite: bool = False,
) -> NDArray[np.complex128]:
    """Check that a kernel describes a genuine inner product, and return it.

    An overlap matrix is Hermitian, and positive definite whenever the basis is
    linearly independent.  Hermiticity is checked on the dense supercell matrix,
    which is where the twisted periodicity has to be consistent with
    ``S_{ab}(D)* = S_{ba}(-D)``; positive definiteness is optional because it
    costs a Cholesky factorization and a diagnostic caller may legitimately want
    to inspect a kernel that fails it.
    """

    matrix = supercell_overlap_matrix(cells, overlaps, transform, supercell_kpoint)
    deviation = float(np.max(np.abs(matrix - matrix.conj().T))) if matrix.size else 0.0
    if deviation > atol:
        raise ValueError(
            "overlap kernel is not Hermitian on the supercell: "
            f"max |S - S^dagger| = {deviation:.3e} exceeds atol={atol:.3e}"
        )
    if require_positive_definite:
        try:
            np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError as error:  # pragma: no cover - message only
            raise ValueError(
                "overlap kernel is not positive definite; the basis it describes "
                "is linearly dependent at this k-point"
            ) from error
    return matrix


def overlap_kernel_from_matrix(
    matrix: ArrayLike, n_cells: int, n_orbitals: int, *, origin: int = 0
) -> NDArray[np.complex128]:
    """Read the kernel off a dense supercell overlap matrix.

    ``S_{ab}(W) = M[(origin a), (W b)]`` once ``cells[origin]`` is the zero
    cell, which :func:`unfoldlab.core.tight_binding.supercell_cells` guarantees
    for ``origin = 0``.  This is the entry point for an LCAO code that hands out
    the supercell overlap matrix rather than a real-space table.
    """

    dense = np.asarray(matrix, dtype=np.complex128)
    size = int(n_cells) * int(n_orbitals)
    if dense.shape != (size, size):
        raise ValueError(f"matrix must have shape ({size}, {size})")
    if not (0 <= int(origin) < int(n_cells)):
        raise ValueError("origin must be a cell index")
    rows = dense[int(origin) * n_orbitals : (int(origin) + 1) * n_orbitals, :]
    reshaped = rows.reshape(n_orbitals, n_cells, n_orbitals)
    return np.asarray(reshaped.transpose(0, 2, 1), dtype=np.complex128)


def _amplitudes(
    cells: NDArray[np.int64],
    coefficients: NDArray[np.complex128],
    points: NDArray[np.float64],
) -> NDArray[np.complex128]:
    phases = np.exp(-2j * np.pi * (points @ cells.T.astype(float)))
    return np.asarray(
        np.einsum("kR,soR->kso", phases, coefficients, optimize=True),
        dtype=np.complex128,
    )


def _check_coefficients(
    coefficients: ArrayLike, n_cells: int, n_orbitals: int
) -> NDArray[np.complex128]:
    coeffs = np.asarray(coefficients, dtype=np.complex128)
    if coeffs.ndim != 3:
        raise ValueError("coefficients must have shape (n_states, n_orbitals, n_cells)")
    if coeffs.shape[2] != n_cells:
        raise ValueError("coefficients must have one entry per cell")
    if coeffs.shape[1] != n_orbitals:
        raise ValueError("coefficients must have one entry per orbital")
    return coeffs


def lcao_norms(
    cells: ArrayLike,
    coefficients: ArrayLike,
    overlaps: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
) -> NDArray[np.float64]:
    """The true squared norms ``c* S c`` of the states, shape ``(n_states,)``.

    This is ``UnfoldLab.ovNorm``.  For an orthonormal kernel it reduces to
    ``sum_{a,R} |c(a, R)|^2``, the denominator of the orthonormal formula.
    """

    cell_array = validate_supercell_cells(cells, transform)
    kernel = _check_kernel(overlaps, cell_array.shape[0])
    coeffs = _check_coefficients(coefficients, cell_array.shape[0], kernel.shape[0])
    matrix = supercell_overlap_matrix(cell_array, kernel, transform, supercell_kpoint)
    # (n_states, n_cells * n_orbitals) in the (cell, orbital) ordering.
    flat = np.swapaxes(coeffs, 1, 2).reshape(coeffs.shape[0], -1)
    products = flat.conj() * (flat @ matrix.T)
    return np.asarray(products.sum(axis=1).real, dtype=np.float64)


def lcao_weights(
    cells: ArrayLike,
    coefficients: ArrayLike,
    overlaps: ArrayLike,
    kpoints: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    multiplicity: int | None = None,
) -> NDArray[np.float64]:
    """Unfolding weights of supercell states in a non-orthogonal basis.

    ``coefficients`` has shape ``(n_states, n_orbitals, n_cells)`` exactly as in
    :func:`unfoldlab.core.tight_binding.tight_binding_weights`; ``overlaps`` is
    the Bloch-twisted kernel at ``supercell_kpoint``.  Returns a
    ``(n_kpoints, n_states)`` table which, over a complete fiber, adds up to one
    per state (``UnfoldLab.sum_ovWeight_eq_one``).

    States of zero norm get weight zero rather than a division by zero.
    """

    cell_array = validate_supercell_cells(cells, transform)
    kernel = _check_kernel(overlaps, cell_array.shape[0])
    coeffs = _check_coefficients(coefficients, cell_array.shape[0], kernel.shape[0])
    points = np.asarray(kpoints, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_kpoints, 3)")
    count = int(multiplicity) if multiplicity is not None else cell_array.shape[0]
    if count <= 0:
        raise ValueError("multiplicity must be positive")

    amplitudes = _amplitudes(cell_array, coeffs, points)
    reciprocal = reciprocal_overlap(cell_array, kernel, points)
    numerator = np.einsum(
        "ksa,kab,ksb->ks", amplitudes.conj(), reciprocal, amplitudes, optimize=True
    ).real
    norms = lcao_norms(cell_array, coeffs, kernel, transform, supercell_kpoint)
    safe = np.where(norms > 0.0, norms, 1.0)
    weights = numerator / (float(count) * safe[np.newaxis, :])
    return np.asarray(np.where(norms[np.newaxis, :] > 0.0, weights, 0.0), dtype=np.float64)


def unfold_lcao(
    cells: ArrayLike,
    coefficients: ArrayLike,
    overlaps: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unfold non-orthogonal supercell states onto the complete fiber.

    Returns ``(kpoints, weights)`` with ``kpoints`` of shape ``(|det T|, 3)``
    and ``weights`` of shape ``(|det T|, n_states)`` summing to one along the
    k-point axis.
    """

    cell_array = validate_supercell_cells(cells, transform)
    q = _check_kpoint(supercell_kpoint, "supercell_kpoint")
    points = fiber_kpoints(q @ transform.matrix.T, transform)
    weights = lcao_weights(
        cell_array,
        coefficients,
        overlaps,
        points,
        transform,
        q,
        multiplicity=transform.multiplicity,
    )
    return points, weights


@dataclass(frozen=True)
class OverlapNeglect:
    """How much is lost by using the orthonormal formula on LCAO output.

    ``max_kernel_deviation`` is ``max_k ||S~(k) - 1||_max``: the orthonormal
    formula is exact precisely when this vanishes
    (``UnfoldLab.ovNumer_sub_tbNumer``).

    ``max_norm_ratio_deviation`` is ``max_s |c* S c / sum|c|^2 - 1|``, the
    denominator's share of the error.

    ``max_weight_difference`` is the largest absolute difference between the two
    weight tables, and ``max_fiber_total_variation`` is
    ``max_s (1/2) sum_k |W_overlap(k, s) - W_orthonormal(k, s)|``: both formulas
    put a total weight of one on the fiber -- the orthonormal one does so for
    *any* coefficients, by Parseval -- so what the overlap changes is how that
    weight is *distributed*, and the total variation is the fraction of it that
    ends up on the wrong primitive k-point.
    """

    max_kernel_deviation: float
    max_norm_ratio_deviation: float
    max_weight_difference: float
    max_fiber_total_variation: float
    n_kpoints: int
    n_states: int

    def significant(self, atol: float = 1e-8) -> bool:
        return (
            self.max_kernel_deviation > atol
            or self.max_norm_ratio_deviation > atol
            or self.max_weight_difference > atol
        )

    def to_dict(self) -> dict[str, float | int]:
        return {
            "max_kernel_deviation": self.max_kernel_deviation,
            "max_norm_ratio_deviation": self.max_norm_ratio_deviation,
            "max_weight_difference": self.max_weight_difference,
            "max_fiber_total_variation": self.max_fiber_total_variation,
            "n_kpoints": self.n_kpoints,
            "n_states": self.n_states,
        }


def diagnose_overlap_neglect(
    cells: ArrayLike,
    coefficients: ArrayLike,
    overlaps: ArrayLike,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    kpoints: ArrayLike | None = None,
) -> OverlapNeglect:
    """Compare the overlap-aware weights with the orthonormal ones.

    ``kpoints`` defaults to the complete fiber over ``supercell_kpoint``, which
    is the only case in which ``max_fiber_total_variation`` is meaningful.
    """

    cell_array = validate_supercell_cells(cells, transform)
    kernel = _check_kernel(overlaps, cell_array.shape[0])
    coeffs = _check_coefficients(coefficients, cell_array.shape[0], kernel.shape[0])
    q = _check_kpoint(supercell_kpoint, "supercell_kpoint")
    points = (
        fiber_kpoints(q @ transform.matrix.T, transform)
        if kpoints is None
        else np.asarray(kpoints, dtype=float).reshape(-1, 3)
    )

    reciprocal = reciprocal_overlap(cell_array, kernel, points)
    identity = np.eye(kernel.shape[0], dtype=np.complex128)
    kernel_deviation = float(np.max(np.abs(reciprocal - identity[np.newaxis, :, :])))

    exact = lcao_norms(cell_array, coeffs, kernel, transform, q)
    plain = (coeffs.real**2 + coeffs.imag**2).sum(axis=(1, 2))
    safe = np.where(plain > 0.0, plain, 1.0)
    norm_deviation = float(np.max(np.abs(exact / safe - 1.0))) if plain.size else 0.0

    overlap_weights = lcao_weights(
        cell_array,
        coeffs,
        kernel,
        points,
        transform,
        q,
        multiplicity=transform.multiplicity,
    )
    orthonormal_weights = tight_binding_weights(
        cell_array, coeffs, points, multiplicity=transform.multiplicity
    )
    difference = (
        float(np.max(np.abs(overlap_weights - orthonormal_weights)))
        if overlap_weights.size
        else 0.0
    )
    variation = (
        float(np.max(0.5 * np.abs(overlap_weights - orthonormal_weights).sum(axis=0)))
        if overlap_weights.size
        else 0.0
    )
    return OverlapNeglect(
        max_kernel_deviation=kernel_deviation,
        max_norm_ratio_deviation=norm_deviation,
        max_weight_difference=difference,
        max_fiber_total_variation=variation,
        n_kpoints=int(points.shape[0]),
        n_states=int(coeffs.shape[0]),
    )


def overlap_neglect_from_model(
    model: TightBindingModel,
    overlap: TightBindingModel,
    transform: TransformationMatrix,
    supercell_kpoint: ArrayLike,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> OverlapNeglect:
    """:func:`diagnose_overlap_neglect` for a model, at one supercell k-point.

    Builds the supercell, solves ``H c = E S c``, and compares the weights its
    eigenstates get from the overlap-aware formula with the ones the orthonormal
    formula would have given.  This is the number to look at before deciding
    that an LCAO basis is "nearly orthonormal enough".
    """

    if overlap.n_orbitals != model.n_orbitals:
        raise ValueError("model and overlap must have the same number of orbitals")
    q = _check_kpoint(supercell_kpoint, "supercell_kpoint")
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    matrix, _ = supercell_bloch_hamiltonian(
        model, transform, q, cells=cell_array, perturbations=list(perturbations or ())
    )
    metric, _ = supercell_bloch_hamiltonian(overlap, transform, q, cells=cell_array)
    _, vectors = solve_generalized_eigenproblem(matrix, metric)
    coefficients = np.swapaxes(vectors.T.reshape(-1, n_cells, model.n_orbitals), 1, 2)
    kernel = overlap_kernel_from_matrix(metric, n_cells, model.n_orbitals)
    return diagnose_overlap_neglect(cell_array, coefficients, kernel, transform, q)


def solve_generalized_eigenproblem(
    hamiltonian: ArrayLike,
    overlap: ArrayLike,
    *,
    threshold: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.complex128]]:
    """Solve ``H c = E S c`` for a Hermitian ``H`` and a positive definite ``S``.

    The Cholesky factorization ``S = L L*`` turns the problem into the standard
    one for ``L^-1 H L^-*``; the returned eigenvectors are the back-transformed
    ``c = L^-* y``, so they are ``S``-orthonormal, ``c* S c = 1``.  Only numpy is
    used, so this adds no dependency.

    Returns ``(energies, vectors)`` with ``vectors[:, s]`` the state ``s``.

    With ``threshold`` the near-null directions of ``S`` are dropped instead
    (canonical orthogonalization, see
    :mod:`unfoldlab.core.conditioning`), which tolerates a rank-deficient
    overlap at the price of returning fewer states than the basis dimension.
    """

    if threshold is not None:
        return solve_generalized_eigenproblem_truncated(hamiltonian, overlap, threshold=threshold)

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    metric = np.asarray(overlap, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square")
    if metric.shape != matrix.shape:
        raise ValueError("overlap must have the same shape as the hamiltonian")
    try:
        lower = np.linalg.cholesky(metric)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "overlap matrix is not positive definite; the LCAO basis is "
            "linearly dependent at this k-point. Pass a threshold to drop the "
            "near-null directions (canonical orthogonalization), or call "
            "unfoldlab.diagnose_overlap_conditioning to see how close to "
            "singular it is."
        ) from error
    reduced = np.linalg.solve(lower, matrix)
    reduced = np.linalg.solve(lower, reduced.conj().T).conj().T
    energies, vectors = np.linalg.eigh(0.5 * (reduced + reduced.conj().T))
    back = np.linalg.solve(lower.conj().T, vectors)
    return np.asarray(energies, dtype=np.float64), np.asarray(back, dtype=np.complex128)


def unfold_lcao_path(
    model: TightBindingModel,
    overlap: TightBindingModel,
    transform: TransformationMatrix,
    primitive_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Unfold a non-orthogonal supercell model along a primitive band path.

    The counterpart of
    :func:`unfoldlab.core.tight_binding.unfold_tight_binding_path`: at each
    requested primitive k-point the supercell is built at the supercell k-point
    that ``k`` folds onto -- ``k`` itself being a valid representative -- the
    generalized problem ``H c = E S c`` is solved, and every eigenstate is given
    its overlap-aware weight at ``k`` alone.

    ``perturbations`` are the on-site Hamiltonian terms of
    :func:`unfoldlab.core.tight_binding.supercell_bloch_hamiltonian`; they act on
    ``H`` only, since a defect that changed the *basis* would change ``S`` too
    and would have to be given as a different overlap table.

    Returns ``(energies, weights)``, both of shape
    ``(n_kpoints, |det T| * n_orbitals)``.
    """

    if overlap.n_orbitals != model.n_orbitals:
        raise ValueError("model and overlap must have the same number of orbitals")
    points = np.asarray(primitive_kpoints, dtype=float).reshape(-1, 3)
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    n_states = n_cells * model.n_orbitals
    perturbation_list = list(perturbations or ())

    energies = np.empty((points.shape[0], n_states), dtype=float)
    weights = np.empty_like(energies)
    for index, k in enumerate(points):
        matrix, _ = supercell_bloch_hamiltonian(
            model, transform, k, cells=cell_array, perturbations=perturbation_list
        )
        metric, _ = supercell_bloch_hamiltonian(overlap, transform, k, cells=cell_array)
        eigenvalues, vectors = solve_generalized_eigenproblem(matrix, metric)
        coefficients = np.swapaxes(vectors.T.reshape(-1, n_cells, model.n_orbitals), 1, 2)
        kernel = overlap_kernel_from_matrix(metric, n_cells, model.n_orbitals)
        energies[index] = eigenvalues
        weights[index] = lcao_weights(
            cell_array,
            coefficients,
            kernel,
            k[np.newaxis, :],
            transform,
            k,
            multiplicity=n_cells,
        )[0]
    return energies, weights


def unfold_lcao_model(
    model: TightBindingModel,
    overlap: TightBindingModel,
    transform: TransformationMatrix,
    supercell_kpoints: Sequence[ArrayLike] | ArrayLike,
    *,
    perturbations: Iterable[tuple[int, int, ArrayLike]] | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Diagonalize a non-orthogonal supercell model and unfold its states.

    ``model`` carries ``<0 a| H |D b>`` and ``overlap`` carries
    ``<phi_{a,0} | phi_{b,D}>`` in the same real-space layout; both are folded
    into the supercell with the same conventions, so the generalized
    eigenproblem ``H c = E S c`` is solved consistently and the eigenvectors are
    unfolded with the overlap-aware weight.

    Returns ``(kpoints, energies, weights)`` shaped as in
    :func:`unfoldlab.core.tight_binding.unfold_tight_binding_model`.
    """

    if overlap.n_orbitals != model.n_orbitals:
        raise ValueError("model and overlap must have the same number of orbitals")
    points = np.asarray(supercell_kpoints, dtype=float).reshape(-1, 3)
    cell_array = supercell_cells(transform)
    n_cells = cell_array.shape[0]
    perturbation_list = list(perturbations or ())

    all_kpoints: list[NDArray[np.float64]] = []
    all_energies: list[NDArray[np.float64]] = []
    all_weights: list[NDArray[np.float64]] = []
    for q in points:
        matrix, _ = supercell_bloch_hamiltonian(
            model, transform, q, cells=cell_array, perturbations=perturbation_list
        )
        metric, _ = supercell_bloch_hamiltonian(overlap, transform, q, cells=cell_array)
        energies, vectors = solve_generalized_eigenproblem(matrix, metric)
        coefficients = vectors.T.reshape(-1, n_cells, model.n_orbitals)
        coefficients = np.swapaxes(coefficients, 1, 2)
        kernel = overlap_kernel_from_matrix(metric, n_cells, model.n_orbitals)
        fiber, weights = unfold_lcao(cell_array, coefficients, kernel, transform, q)
        all_kpoints.append(fiber)
        all_energies.append(np.tile(energies, (fiber.shape[0], 1)))
        all_weights.append(weights)

    return (
        np.concatenate(all_kpoints, axis=0),
        np.concatenate(all_energies, axis=0),
        np.concatenate(all_weights, axis=0),
    )
