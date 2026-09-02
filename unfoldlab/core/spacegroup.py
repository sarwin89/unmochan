"""Detection of the symmetry operations of a lattice and of a structure.

Symmetry-reduced unfolding (:mod:`unfoldlab.core.symmetry`) needs the
point-group operations of the cell, and until now the user had to supply them
by hand.  This module derives them from the structure itself.

Conventions follow :class:`~unfoldlab.core.structures.Structure`: the lattice
matrix ``A`` holds one real-space lattice vector per *row*, and a fractional
position is a row vector, so the Cartesian position is ``x @ A``.  A symmetry
operation is written in the column convention, ``x -> M x + t``, which in row
form reads ``x -> x M^T + t``.

**Which integer matrices are lattice symmetries.**  ``M`` maps the lattice onto
itself and preserves all distances exactly when it preserves the metric tensor
``G = A A^T``:

``M^T G M = G``.

The columns of such an ``M`` are the images of the three basis vectors, so each
column ``v`` satisfies ``v^T G v = G_ii``: it is a lattice vector of the same
length as the basis vector it replaces.  Only finitely many integer vectors
have a given length, and Cauchy-Schwarz in the inner product defined by ``G``
bounds their components,

``v_k^2 <= (G^-1)_kk (v^T G v)``

(``UnfoldLab.sq_component_le_of_quadForm_le`` in
``RequestProject/Unfolding/PointGroup.lean``).  The enumeration below uses
exactly that bound, so it is *complete*: no lattice symmetry can be missed.

**Reciprocal space.**  A weight-unfolding operation acts on fractional
*reciprocal* coordinates.  If ``M`` is the operation on fractional positions,
the matching reciprocal operation is ``S = (M^-1)^T``, which is again integral
because a lattice symmetry is unimodular.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import integer_adjugate3, integer_det3
from unfoldlab.core.structures import Structure
from unfoldlab.core.symmetry import primitive_operation, supercell_operation

__all__ = [
    "OperationCheck",
    "OperationValidation",
    "SymmetryOperation",
    "cartesian_rotation",
    "detect_primitive_operations",
    "lattice_point_group",
    "metric_tensor",
    "reciprocal_operation",
    "site_labels",
    "space_group_operations",
    "validate_primitive_operations",
]


def metric_tensor(lattice: ArrayLike) -> NDArray[np.float64]:
    """Return the metric tensor ``G = A A^T`` of a row-vector lattice matrix."""

    a = np.asarray(lattice, dtype=float)
    if a.shape != (3, 3):
        raise ValueError(f"lattice must have shape (3, 3), got {a.shape}")
    if abs(float(np.linalg.det(a))) < 1e-12:
        raise ValueError("lattice must be non-degenerate")
    return a @ a.T


def _vectors_of_norm(
    metric: NDArray[np.float64],
    target: float,
    tol: float,
) -> list[NDArray[np.int64]]:
    """All integer vectors ``v`` with ``v^T G v`` within ``tol`` of ``target``.

    The search box comes from the Cauchy-Schwarz bound quoted in the module
    docstring, so the list is exhaustive.
    """

    inverse = np.linalg.inv(metric)
    limit = target + tol
    bounds = [int(np.floor(np.sqrt(max(limit * inverse[k, k], 0.0)) + 1e-9)) for k in range(3)]
    found: list[NDArray[np.int64]] = []
    for coeffs in product(*(range(-b, b + 1) for b in bounds)):
        v = np.array(coeffs, dtype=np.int64)
        if abs(float(v @ metric @ v) - target) <= tol:
            found.append(v)
    return found


def lattice_point_group(
    lattice: ArrayLike,
    *,
    tol: float = 1e-5,
) -> list[NDArray[np.int64]]:
    """Return every integer matrix ``M`` with ``M^T G M = G``.

    These are the point-group operations of the *lattice* (the holohedry), in
    the column convention on fractional coordinates.  The list always contains
    the identity and ``-I``, and its length is even and at most 48 in three
    dimensions.

    ``tol`` is an absolute tolerance on the entries of the metric tensor, so it
    scales as a squared length; the default suits lattice vectors given in
    angstrom to the usual six digits.
    """

    metric = metric_tensor(lattice)
    columns = [_vectors_of_norm(metric, float(metric[i, i]), tol) for i in range(3)]
    operations: list[NDArray[np.int64]] = []
    for candidate in product(*columns):
        m = np.stack(candidate, axis=1)
        if np.allclose(m.T @ metric @ m, metric, atol=tol, rtol=0.0):
            operations.append(m.astype(np.int64))
    return operations


def reciprocal_operation(operation: ArrayLike) -> NDArray[np.int64]:
    """Convert a fractional-position operation ``M`` into ``S = (M^-1)^T``.

    ``S`` is the same symmetry acting on fractional *reciprocal* coordinates,
    which is the form the unfolding routines take.
    """

    m = np.asarray(operation, dtype=np.int64)
    if m.shape != (3, 3):
        raise ValueError(f"operation must have shape (3, 3), got {m.shape}")
    determinant = integer_det3(m)
    if abs(determinant) != 1:
        raise ValueError(
            f"a lattice symmetry must be unimodular (|det| = 1), got det = {determinant}"
        )
    return (integer_adjugate3(m) // determinant).T.astype(np.int64)


@dataclass(frozen=True)
class SymmetryOperation:
    """One space-group operation ``x -> M x + t`` on fractional coordinates.

    ``rotation`` is the integer matrix ``M`` in the column convention and
    ``translation`` the fractional part ``t``, reduced into ``[0, 1)``.  An
    operation with a nonzero ``translation`` that is not a lattice vector is
    non-symmorphic; only ``rotation`` matters for unfolding weights, because a
    translation multiplies a plane-wave coefficient by a phase and the weights
    depend on ``|c|^2`` alone.
    """

    rotation: NDArray[np.int64]
    translation: NDArray[np.float64]

    @property
    def is_symmorphic(self) -> bool:
        residual = self.translation - np.round(self.translation)
        return bool(np.allclose(residual, 0.0, atol=1e-8))

    @property
    def determinant(self) -> int:
        return integer_det3(np.asarray(self.rotation, dtype=np.int64))


def _reduce(coords: NDArray[np.float64]) -> NDArray[np.float64]:
    return coords - np.floor(coords)


def site_labels(
    structure: Structure,
    magmoms: ArrayLike | None = None,
    *,
    magmom_tol: float = 1e-3,
) -> tuple[str, ...]:
    """Decorate the sites of ``structure`` with their species and moment.

    Without ``magmoms`` the label of a site is its species, which is the right
    decoration for a nonmagnetic calculation.  With ``magmoms`` -- one
    *collinear* moment per site, shape ``(n_sites,)`` -- two sites of the same
    species carry the same label only when their moments agree to
    ``magmom_tol``.  Noncollinear moments are not labels: the operation rotates
    them, so they are handled by :func:`space_group_operations` itself.

    This is what makes the detected group the group of the *magnetic* crystal.
    A spin-polarized calculation solves one Hamiltonian per spin channel, and a
    channel is invariant only under the operations that leave the moment pattern
    alone: an operation exchanging an up site with a down site maps the
    majority-spin problem to the minority-spin problem, not to itself
    (``UnfoldLab.zeeman_invariant_iff_moment_invariant`` in
    ``RequestProject/Unfolding/SymmetryHypothesis.lean``).  Operations that flip
    the spin as well -- the anti-unitary elements of the magnetic group -- are
    deliberately excluded: they do not relate two states of one channel.
    """

    species = tuple(structure.species)
    if magmoms is None:
        return species

    moments = np.asarray(magmoms, dtype=float)
    if moments.ndim != 1 or moments.shape[0] != len(species):
        raise ValueError(
            "collinear magmoms must have shape (n_sites,), got "
            f"{np.shape(magmoms)} for {len(species)} sites"
        )
    moments = moments[:, np.newaxis]

    buckets: dict[str, list[NDArray[np.float64]]] = {}
    labels: list[str] = []
    for name, moment in zip(species, moments, strict=True):
        bucket = buckets.setdefault(name, [])
        for index, value in enumerate(bucket):
            if float(np.linalg.norm(value - moment)) <= magmom_tol:
                labels.append(f"{name}:{index}")
                break
        else:
            bucket.append(np.asarray(moment, dtype=float))
            labels.append(f"{name}:{len(bucket) - 1}")
    return tuple(labels)


def cartesian_rotation(
    lattice: ArrayLike,
    rotation: ArrayLike,
) -> NDArray[np.float64]:
    """Cartesian form ``R = A^T M A^-T`` of a fractional rotation ``M``.

    With row-vector lattices a Cartesian column vector is ``r = A^T x``, so the
    fractional operation ``x -> M x`` is ``r -> A^T M A^-T r``.  ``R`` is
    orthogonal exactly when ``M`` preserves the metric tensor.
    """

    a = np.asarray(lattice, dtype=float)
    m = np.asarray(rotation, dtype=float)
    return a.T @ m @ np.linalg.inv(a.T)


def _label_codes(labels: Sequence[str]) -> NDArray[np.int64]:
    """Number the distinct decorations, so sites compare as integers."""

    numbering: dict[str, int] = {}
    return np.array(
        [numbering.setdefault(label, len(numbering)) for label in labels],
        dtype=np.int64,
    )


def _feasible_partners(
    frac: NDArray[np.float64],
    images: NDArray[np.float64],
    codes: NDArray[np.int64],
    image_codes: NDArray[np.int64],
    tol: float,
    rotated_moments: NDArray[np.float64] | None,
    moments: NDArray[np.float64] | None,
    magmom_tol: float,
) -> NDArray[np.bool_]:
    """``(n_images, n_sites)`` table of the sites each image may be assigned to.

    A site is a candidate when it is within ``tol`` of the image in every
    fractional component (measured on the torus, so the cell boundary is not a
    barrier), carries the same decoration, and -- in the noncollinear case --
    carries the *rotated* moment ``det(R) R m``, a magnetization being an axial
    vector.
    """

    delta = images[:, np.newaxis, :] - frac[np.newaxis, :, :]
    delta -= np.round(delta)
    close = np.all(np.abs(delta) <= tol, axis=2)
    close &= image_codes[:, np.newaxis] == codes[np.newaxis, :]
    if moments is not None and rotated_moments is not None:
        gaps = np.linalg.norm(moments[np.newaxis, :, :] - rotated_moments[:, np.newaxis, :], axis=2)
        close &= gaps <= magmom_tol
    return close


def _perfect_matching(feasible: NDArray[np.bool_]) -> NDArray[np.int64] | None:
    """Assign every image one of its candidate sites bijectively, or fail.

    Deciding this greedily -- walk the images in order, take the first free
    candidate -- is wrong: an image can consume the only site another image
    could have used, and the operation is then rejected although an assignment
    exists.  That happens whenever two sites lie within the tolerance of each
    other, which is exactly the regime a tolerance is there to handle.  The
    fallback below is the standard augmenting-path (Kuhn) algorithm, which is
    exact; the search is iterative so that a long augmenting path cannot
    overflow the Python stack.

    Returns the site assigned to each image, or ``None`` when no assignment
    exists.
    """

    n_images, n_sites = feasible.shape
    if n_images != n_sites:  # pragma: no cover - the caller always squares it
        return None
    counts = feasible.sum(axis=1)
    if not counts.all():
        return None
    partners = np.argmax(feasible, axis=1)
    if int(counts.max()) == 1:
        # No image has a choice, so the assignment is forced and the only
        # question is whether it is injective.  This is the case for every
        # structure whose sites are further apart than the tolerance, which is
        # to say all of them bar the pathological ones, and it stays in numpy.
        return partners if np.unique(partners).size == n_sites else None
    candidates = [np.flatnonzero(row).tolist() for row in feasible]
    owner = [-1] * n_sites
    for start in range(n_images):
        seen = bytearray(n_sites)
        stack = [start]
        iterators = {start: iter(candidates[start])}
        entry: dict[int, int] = {start: -1}
        augmented = False
        while stack:
            row = stack[-1]
            site = next(iterators[row], None)
            if site is None:
                stack.pop()
                continue
            if seen[site]:
                continue
            seen[site] = 1
            if owner[site] == -1:
                while stack:
                    current = stack.pop()
                    owner[site] = current
                    site = entry[current]
                augmented = True
                break
            following = owner[site]
            entry[following] = site
            iterators[following] = iter(candidates[following])
            stack.append(following)
        if not augmented:
            return None
    for site, image in enumerate(owner):
        partners[image] = site
    return partners


_PROBE_LIMIT = 4
"""How many images are tested on their own before the full table is built.

Most candidate translations fail at the very first site, and finding that out
costs ``O(n_sites)`` instead of the ``O(n_sites^2)`` of the full table.
"""


def _site_assignment(
    frac: NDArray[np.float64],
    codes: NDArray[np.int64],
    rotation: NDArray[np.int64],
    translation: NDArray[np.float64],
    tol: float,
    rotated_moments: NDArray[np.float64] | None = None,
    moments: NDArray[np.float64] | None = None,
    magmom_tol: float = 1e-3,
) -> NDArray[np.int64] | None:
    """How ``x -> M x + t`` permutes the decorated sites, or ``None``.

    The operation is a symmetry when the images of the sites can be assigned to
    the sites *bijectively*, every image landing within ``tol`` of the site it
    is assigned to; :func:`_perfect_matching` decides that exactly and returns
    the assignment, which the caller reuses to verify composite operations
    cheaply.

    With ``moments`` and ``rotated_moments`` given -- the noncollinear case --
    the site that receives the image of site ``i`` must also carry the rotated
    moment ``det(R) R m_i``, because a magnetization is an axial vector.
    """

    images = _reduce(frac @ rotation.T + translation)
    probe = min(_PROBE_LIMIT, images.shape[0])
    if probe and not np.all(
        _feasible_partners(
            frac,
            images[:probe],
            codes,
            codes[:probe],
            tol,
            None if rotated_moments is None else rotated_moments[:probe],
            moments,
            magmom_tol,
        ).any(axis=1)
    ):
        return None
    feasible = _feasible_partners(
        frac, images, codes, codes, tol, rotated_moments, moments, magmom_tol
    )
    return _perfect_matching(feasible)


def _assignment_holds(
    frac: NDArray[np.float64],
    codes: NDArray[np.int64],
    rotation: NDArray[np.int64],
    translation: NDArray[np.float64],
    partners: NDArray[np.int64],
    tol: float,
    rotated_moments: NDArray[np.float64] | None,
    moments: NDArray[np.float64] | None,
    magmom_tol: float,
) -> bool:
    """Check one *given* assignment, in ``O(n_sites)`` rather than ``O(n^2)``.

    Used for the operations obtained by composition (see
    :func:`space_group_operations`): the permutation is already known, so no
    search is needed and the tolerance is still applied in full.
    """

    images = _reduce(frac @ rotation.T + translation)
    delta = images - frac[partners]
    delta -= np.round(delta)
    if not bool(np.all(np.abs(delta) <= tol)):
        return False
    if not bool(np.array_equal(codes[partners], codes)):
        return False
    if moments is not None and rotated_moments is not None:
        gaps = np.linalg.norm(moments[partners] - rotated_moments, axis=1)
        if not bool(np.all(gaps <= magmom_tol)):
            return False
    return True


def _unique_translations(candidates: NDArray[np.float64], tol: float) -> list[NDArray[np.float64]]:
    """Drop candidate translations that repeat one already in the list.

    Two translations differing by a lattice vector are the same operation, so
    the comparison is made on the torus; comparing the raw numbers instead
    reports ``0.999999`` and ``0.000001`` as different operations.
    """

    kept: list[NDArray[np.float64]] = []
    for candidate in candidates:
        if kept:
            delta = np.asarray(kept) - candidate
            delta -= np.round(delta)
            if bool(np.any(np.all(np.abs(delta) <= tol, axis=1))):
                continue
        kept.append(candidate)
    return kept


def _candidate_translations(
    frac: NDArray[np.float64],
    anchors: Sequence[int],
    reference: int,
    rotation: NDArray[np.int64],
    tol: float,
) -> list[NDArray[np.float64]]:
    """Every translation that could complete ``rotation`` into a symmetry.

    A symmetry must carry the reference site onto a site carrying the same
    decoration, and ``anchors`` lists those, so the translation is determined
    by which anchor receives the reference.
    """

    rotated_reference = frac[reference] @ rotation.T
    return _unique_translations(_reduce(frac[anchors] - rotated_reference), tol)


def space_group_operations(
    structure: Structure,
    *,
    symprec: float = 1e-5,
    lattice_tol: float | None = None,
    magmoms: ArrayLike | None = None,
    magmom_tol: float = 1e-3,
) -> list[SymmetryOperation]:
    """Return the space-group operations of ``structure``.

    Every lattice symmetry ``M`` (:func:`lattice_point_group`) is combined with
    the candidate translations that carry one chosen reference site onto a site
    of the same species; an operation is kept when it permutes all decorated
    sites.  ``symprec`` is a tolerance on fractional coordinates, ``lattice_tol``
    an absolute tolerance on the metric tensor (defaulting to ``symprec`` times
    the largest metric entry, so it scales like a squared length).

    ``magmoms`` returns the group of the magnetic crystal, which is what a
    spin-polarized calculation is invariant under.  Without it a ferrimagnetic
    or antiferromagnetic cell reports the operations of its nonmagnetic
    skeleton, several of which relate states of *different* Hamiltonians.  A
    collinear moment per site, shape ``(n_sites,)``, decorates the sites (see
    :func:`site_labels`): spin and space are decoupled in a non-relativistic
    spin-polarized run, so the operation must map every site onto a site of the
    same species *and* the same moment.  A noncollinear moment per site, shape
    ``(n_sites, 3)``, is instead rotated with the operation, as an axial vector
    ``m -> det(R) R m``, which is the condition with spin-orbit coupling.

    An empty structure has only its lattice symmetries, each with zero
    translation.

    **Cost.**  Testing every (rotation, translation) pair from scratch is cubic
    in the number of sites, and a supercell -- the very thing this package
    unfolds -- is where the count is large *and* nearly every translation is a
    symmetry.  The group structure removes the extra factor: the translations
    that complete a given rotation form a coset of the pure-translation
    subgroup, so once one of them is found the rest are obtained by composing
    with the subgroup, and the permutation of a composite is the composition of
    the two permutations.  Verifying a *known* permutation is linear, so the
    detection is quadratic overall.  Nothing is accepted on the strength of the
    group argument alone: every operation reported has been checked against
    ``symprec`` site by site.
    """

    metric = metric_tensor(structure.lattice)
    if lattice_tol is None:
        lattice_tol = float(symprec * max(1.0, np.abs(metric).max()))
    rotations = lattice_point_group(structure.lattice, tol=lattice_tol)

    frac = _reduce(np.asarray(structure.frac_coords, dtype=float))
    vector_moments: NDArray[np.float64] | None = None
    if magmoms is not None and np.asarray(magmoms, dtype=float).ndim == 2:
        vector_moments = np.asarray(magmoms, dtype=float)
        if vector_moments.shape != (len(structure.species), 3):
            raise ValueError(
                "noncollinear magmoms must have shape (n_sites, 3), got "
                f"{vector_moments.shape} for {len(structure.species)} sites"
            )
        species = tuple(structure.species)
    else:
        species = site_labels(structure, magmoms, magmom_tol=magmom_tol)
    if len(species) == 0:
        return [SymmetryOperation(rotation=m, translation=np.zeros(3)) for m in rotations]

    counts = Counter(species)
    rarest = min(counts, key=lambda name: (counts[name], name))
    anchors = [i for i, name in enumerate(species) if name == rarest]
    reference = anchors[0]
    codes = _label_codes(species)

    identity = np.eye(3, dtype=np.int64)
    subgroup: list[tuple[NDArray[np.float64], NDArray[np.int64]]] = []
    for shift in _candidate_translations(frac, anchors, reference, identity, symprec):
        permutation = _site_assignment(
            frac,
            codes,
            identity,
            shift,
            symprec,
            rotated_moments=vector_moments,
            moments=vector_moments,
            magmom_tol=magmom_tol,
        )
        if permutation is not None:
            subgroup.append((shift, permutation))
    shifts = np.asarray([shift for shift, _ in subgroup])

    operations: list[SymmetryOperation] = []
    for rotation in rotations:
        rotated_moments: NDArray[np.float64] | None = None
        if vector_moments is not None:
            cartesian = cartesian_rotation(structure.lattice, rotation)
            sign = float(np.sign(np.linalg.det(cartesian)))
            rotated_moments = sign * (vector_moments @ cartesian.T)
        candidates = _candidate_translations(frac, anchors, reference, rotation, symprec)
        seed = -1
        seed_permutation: NDArray[np.int64] | None = None
        for index, translation in enumerate(candidates):
            seed_permutation = _site_assignment(
                frac,
                codes,
                rotation,
                translation,
                symprec,
                rotated_moments=rotated_moments,
                moments=vector_moments,
                magmom_tol=magmom_tol,
            )
            if seed_permutation is not None:
                seed = index
                break
        if seed_permutation is None:
            continue
        operations.append(SymmetryOperation(rotation=rotation, translation=candidates[seed]))
        for translation in candidates[seed + 1 :]:
            delta = shifts - (translation - candidates[seed])
            delta -= np.round(delta)
            hits = np.flatnonzero(np.all(np.abs(delta) <= symprec, axis=1))
            if hits.size == 0:
                continue
            permutation = subgroup[int(hits[0])][1][seed_permutation]
            if _assignment_holds(
                frac,
                codes,
                rotation,
                translation,
                permutation,
                symprec,
                rotated_moments,
                vector_moments,
                magmom_tol,
            ):
                operations.append(SymmetryOperation(rotation=rotation, translation=translation))
    return operations
    return operations


def detect_primitive_operations(
    structure: Structure,
    transform: ArrayLike,
    *,
    symprec: float = 1e-5,
    time_reversal: bool = False,
    magmoms: ArrayLike | None = None,
    magmom_tol: float = 1e-3,
) -> NDArray[np.int64]:
    """Point-group operations of a supercell, expressed for unfolding.

    ``structure`` is the *supercell* and ``transform`` the supercell matrix
    ``T``.  Each space-group operation of the supercell is converted to
    fractional reciprocal supercell coordinates, ``S_sc = (M^-1)^T``, and then
    to primitive coordinates, ``S_pc = T^-1 S_sc T``; the operations for which
    that is not integral are not symmetries of the primitive cell and are
    dropped.

    The result is the array of distinct ``S_pc`` matrices, shape
    ``(n_operations, 3, 3)``, ready to be passed as ``operations`` to
    :func:`~unfoldlab.core.symmetry.map_kpoints_to_stored` or written out for
    the ``--symmetry`` option.  The identity is always present.

    ``time_reversal=True`` additionally includes ``-S_pc`` for every retained
    operation, i.e. it adjoins ``k -> -k``.  A plane-wave code reduces its mesh
    by time reversal whether or not the crystal is centrosymmetric, because for
    a nonmagnetic calculation the state at ``-K`` is the complex conjugate of
    the state at ``K``; the spatial point group detected here contains ``-1``
    only for a centrosymmetric structure, so without this flag a
    non-centrosymmetric crystal's stored wavefunction set looks incomplete when
    it is not.  The weight is unaffected by the conjugation, which is the
    content of ``UnfoldLab.weight_timeReversal_stored`` in
    ``RequestProject/Unfolding/TimeReversal.lean``.

    Use it only for a time-reversal symmetric calculation: not for a magnetic
    one, and never to reconstruct a spin texture, which is odd under time
    reversal.

    ``magmoms`` restricts the search to the operations of the *magnetic*
    structure; see :func:`site_labels`.  Pass it for every spin-polarized run,
    and then leave ``time_reversal`` off.
    """

    unique: list[NDArray[np.int64]] = []
    for operation in space_group_operations(
        structure, symprec=symprec, magmoms=magmoms, magmom_tol=magmom_tol
    ):
        s_sc = reciprocal_operation(operation.rotation)
        try:
            s_pc = primitive_operation(s_sc, transform)
        except ValueError:
            continue
        if not any(np.array_equal(s_pc, known) for known in unique):
            unique.append(s_pc)
    if not unique:  # pragma: no cover - the identity always survives
        unique = [np.eye(3, dtype=np.int64)]
    if time_reversal:
        for s_pc in list(unique):
            reversed_op = (-s_pc).astype(np.int64)
            if not any(np.array_equal(reversed_op, known) for known in unique):
                unique.append(reversed_op)
    return np.stack(unique).astype(np.int64)


@dataclass(frozen=True)
class OperationCheck:
    """The verdict on one supplied symmetry operation.

    ``operation`` is the matrix as supplied, in primitive fractional reciprocal
    coordinates.  ``lattice_compatible`` says whether it passes the two tests
    the unfolding kernel itself makes (unimodular, and ``T S T^-1`` integral);
    ``crystal_symmetry`` whether it is in fact an operation of the decorated
    supercell, which is the hypothesis the weights rely on and which the kernel
    cannot check.  ``time_reversal`` marks an operation that is one only after
    adjoining ``k -> -k``.
    """

    operation: NDArray[np.int64]
    lattice_compatible: bool
    crystal_symmetry: bool
    time_reversal: bool
    reason: str

    @property
    def ok(self) -> bool:
        return self.lattice_compatible and self.crystal_symmetry


@dataclass(frozen=True)
class OperationValidation:
    """Result of :func:`validate_primitive_operations`."""

    checks: tuple[OperationCheck, ...]
    detected: NDArray[np.int64]

    @property
    def ok(self) -> bool:
        return all(check.ok for check in self.checks)

    @property
    def invalid(self) -> tuple[int, ...]:
        return tuple(index for index, check in enumerate(self.checks) if not check.ok)

    def summary(self) -> list[str]:
        """One human-readable line per supplied operation, plus a headline."""

        lines = [
            f"{len(self.checks)} operation(s) supplied, "
            f"{len(self.detected)} detected for this structure and transform"
        ]
        for index, check in enumerate(self.checks):
            flat = " ".join(str(int(value)) for value in check.operation.reshape(-1))
            mark = "ok" if check.ok else "REJECTED"
            lines.append(f"[{index}] {flat}: {mark} -- {check.reason}")
        return lines


def validate_primitive_operations(
    structure: Structure,
    operations: ArrayLike,
    transform: ArrayLike,
    *,
    symprec: float = 1e-5,
    magmoms: ArrayLike | None = None,
    magmom_tol: float = 1e-3,
    time_reversal: bool = False,
) -> OperationValidation:
    """Check supplied unfolding operations against the supercell structure.

    :func:`~unfoldlab.core.symmetry.map_kpoints_to_stored` verifies only that
    each operation is unimodular and compatible with the two lattices.  That is
    all it can do: it never sees the atoms.  The identity the weights rest on --
    that the wavefunction stored at ``K_f`` is the symmetry image of the state
    at ``S_sc K_f`` -- additionally requires the operation to be a symmetry of
    the crystal, and, for a spin-polarized run, of its moment pattern.  When it
    is not, the computed weights are those of an unrelated state and nothing
    downstream complains: the sum rule is satisfied by any state
    (``UnfoldLab.exists_weight_ne_of_coefficients`` and
    ``UnfoldLab.exists_latticeSymmetry_bandEnergy_ne`` in
    ``RequestProject/Unfolding/SymmetryHypothesis.lean``).

    This function performs that missing check.  ``structure`` is the supercell,
    ``operations`` an ``(n, 3, 3)`` array in primitive fractional reciprocal
    coordinates -- exactly what ``unfoldlab detect-symmetry --matrix`` writes and
    ``--symmetry`` reads -- and ``transform`` the supercell matrix ``T``.

    ``time_reversal=True`` also accepts an operation that is a crystal symmetry
    composed with ``k -> -k``, which a nonmagnetic plane-wave calculation may
    legitimately have used to reduce its mesh.  Do not combine it with
    ``magmoms``.
    """

    if time_reversal and magmoms is not None:
        raise ValueError(
            "time_reversal is not valid for a magnetic structure: the state at -K is the "
            "conjugate of the state at K only when the two spin channels are degenerate"
        )

    supplied = np.asarray(operations, dtype=float).reshape(-1, 3, 3)
    detected = detect_primitive_operations(
        structure,
        transform,
        symprec=symprec,
        magmoms=magmoms,
        magmom_tol=magmom_tol,
    )

    checks: list[OperationCheck] = []
    for raw in supplied:
        rounded = np.rint(raw).astype(np.int64)
        if not np.allclose(raw, rounded, atol=1e-8):
            checks.append(
                OperationCheck(
                    operation=rounded,
                    lattice_compatible=False,
                    crystal_symmetry=False,
                    time_reversal=False,
                    reason="not an integer matrix",
                )
            )
            continue
        try:
            determinant = integer_det3(rounded)
            if abs(determinant) != 1:
                raise ValueError(
                    f"a lattice symmetry must be unimodular (|det| = 1), got det = {determinant}"
                )
            supercell_operation(rounded, transform)
        except ValueError as error:
            checks.append(
                OperationCheck(
                    operation=rounded,
                    lattice_compatible=False,
                    crystal_symmetry=False,
                    time_reversal=False,
                    reason=str(error),
                )
            )
            continue

        if any(np.array_equal(rounded, known) for known in detected):
            reason = "a symmetry of the supercell"
            checks.append(
                OperationCheck(
                    operation=rounded,
                    lattice_compatible=True,
                    crystal_symmetry=True,
                    time_reversal=False,
                    reason=reason,
                )
            )
            continue

        negated = (-rounded).astype(np.int64)
        if any(np.array_equal(negated, known) for known in detected):
            reason = (
                "a symmetry of the supercell composed with time reversal (k -> -k)"
                if time_reversal
                else "only a symmetry once k -> -k is adjoined; pass time_reversal=True "
                "if the calculation is nonmagnetic"
            )
            checks.append(
                OperationCheck(
                    operation=rounded,
                    lattice_compatible=True,
                    crystal_symmetry=time_reversal,
                    time_reversal=True,
                    reason=reason,
                )
            )
            continue

        checks.append(
            OperationCheck(
                operation=rounded,
                lattice_compatible=True,
                crystal_symmetry=False,
                time_reversal=False,
                reason=(
                    "compatible with the lattices but not a symmetry of the "
                    + ("magnetic " if magmoms is not None else "")
                    + "supercell: the stored wavefunction it would reuse belongs to a "
                    "different state"
                ),
            )
        )

    return OperationValidation(checks=tuple(checks), detected=detected)
