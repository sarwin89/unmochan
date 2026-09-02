"""Exact enumeration of the finite quotient ``Z^3 / M Z^3``.

Why this exists
---------------

The unfolding theory says that exactly ``|det T|`` primitive k-points fold onto
each supercell k-point (``UnfoldLab.IsFiberRepr.card_eq_natAbs_det``), but it
does not say how to find them.  A user who has run a supercell calculation on a
k-mesh and wants the unfolded band structure needs precisely that list.

The construction is arithmetic.  With column-vector conventions the folding
relation ``k_pc @ T.T = K_sc + m`` reads ``T @ k_pc = K_sc + m`` with
``m`` an integer vector, and two ``m``'s give k-points differing by a primitive
reciprocal lattice vector exactly when they differ by an element of ``T Z^3``.
So the fiber is indexed by the finite group ``Z^3 / T Z^3``, whose order is
``|det T|``.

Representatives of that group come from the Smith normal form ``U M V = D``
with ``U``, ``V`` unimodular and ``D`` diagonal: since ``V`` is invertible over
the integers, ``M Z^3 = U^-1 D Z^3``, so the classes are ``U^-1 y`` for
``y`` running over ``prod_i [0, d_i)``.

Everything here is done in exact Python integer arithmetic; no floating-point
rounding enters, and the result is checked against ``|det M|`` before it is
returned.  The correctness of feeding these representatives to the unfolding is
formalized as ``UnfoldLab.IsCosetRepr.isFiberRepr`` and
``UnfoldLab.IsCosetRepr.card_eq_natAbs_det`` in
``RequestProject/Unfolding/Coset.lean``.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unfoldlab.core.numerics import as_int_array, integer_det3

__all__ = [
    "SmithNormalForm",
    "coset_representatives",
    "smith_normal_form",
]


@dataclass(frozen=True)
class SmithNormalForm:
    """Result of :func:`smith_normal_form`.

    Satisfies ``left @ matrix @ right == diagonal`` with ``left`` and ``right``
    unimodular (determinant ``+-1``), and ``left_inverse @ left == I``.
    """

    diagonal: NDArray[np.int64]
    left: NDArray[np.int64]
    right: NDArray[np.int64]
    left_inverse: NDArray[np.int64]

    @property
    def invariant_factors(self) -> tuple[int, ...]:
        """The diagonal entries ``d_1 | d_2 | ...``, as Python integers."""

        return tuple(int(x) for x in np.diagonal(self.diagonal))


def _row_add(a: list[list[int]], target: int, source: int, factor: int) -> None:
    """``row[target] += factor * row[source]``."""

    if factor == 0:
        return
    row_s = a[source]
    row_t = a[target]
    for j, value in enumerate(row_s):
        row_t[j] += factor * value


def _col_add(a: list[list[int]], target: int, source: int, factor: int) -> None:
    """``col[target] += factor * col[source]``."""

    if factor == 0:
        return
    for row in a:
        row[target] += factor * row[source]


def _row_swap(a: list[list[int]], i: int, j: int) -> None:
    if i != j:
        a[i], a[j] = a[j], a[i]


def _col_swap(a: list[list[int]], i: int, j: int) -> None:
    if i != j:
        for row in a:
            row[i], row[j] = row[j], row[i]


def _row_negate(a: list[list[int]], i: int) -> None:
    a[i] = [-value for value in a[i]]


def _identity(n: int) -> list[list[int]]:
    return [[1 if i == j else 0 for j in range(n)] for i in range(n)]


def smith_normal_form(matrix: ArrayLike, *, atol: float = 1e-6) -> SmithNormalForm:
    """Exact Smith normal form of a square integer matrix.

    Returns ``U``, ``V`` unimodular and a diagonal ``D`` with ``U M V == D``,
    non-negative diagonal entries each dividing the next, together with the
    integer inverse of ``U``.

    The computation is carried out in Python integers, so it is exact for
    matrices of any magnitude and never depends on ``atol`` beyond checking
    that the input really is integral.
    """

    array = as_int_array(np.asarray(matrix), name="matrix", atol=atol)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"matrix must be square, got shape {array.shape}")
    n = int(array.shape[0])

    work = [[int(value) for value in row] for row in array]
    left = _identity(n)
    left_inverse = _identity(n)
    right = _identity(n)

    def swap_rows(i: int, j: int) -> None:
        _row_swap(work, i, j)
        _row_swap(left, i, j)
        _col_swap(left_inverse, i, j)

    def add_rows(target: int, source: int, factor: int) -> None:
        # row[target] += factor * row[source]  =>  L <- E L, L^-1 <- L^-1 E^-1
        _row_add(work, target, source, factor)
        _row_add(left, target, source, factor)
        _col_add(left_inverse, source, target, -factor)

    def negate_row(i: int) -> None:
        _row_negate(work, i)
        _row_negate(left, i)
        for row in left_inverse:
            row[i] = -row[i]

    for t in range(n):
        while True:
            pivot = None
            best = 0
            for i in range(t, n):
                for j in range(t, n):
                    value = abs(work[i][j])
                    if value != 0 and (pivot is None or value < best):
                        pivot = (i, j)
                        best = value
            if pivot is None:
                break
            swap_rows(t, pivot[0])
            _col_swap(work, t, pivot[1])
            _col_swap(right, t, pivot[1])

            cleared = True
            head = work[t][t]
            for i in range(t + 1, n):
                if work[i][t] != 0:
                    add_rows(i, t, -(work[i][t] // head))
                    if work[i][t] != 0:
                        cleared = False
            for j in range(t + 1, n):
                if work[t][j] != 0:
                    factor = -(work[t][j] // head)
                    _col_add(work, j, t, factor)
                    _col_add(right, j, t, factor)
                    if work[t][j] != 0:
                        cleared = False
            if not cleared:
                continue

            # The pivot must divide the rest of the block, or the diagonal will
            # not satisfy d_t | d_{t+1}.  Folding an offending row into the
            # pivot row makes the pivot strictly smaller, so this terminates.
            offender = None
            for i in range(t + 1, n):
                for j in range(t + 1, n):
                    if work[i][j] % work[t][t] != 0:
                        offender = i
                        break
                if offender is not None:
                    break
            if offender is None:
                break
            add_rows(t, offender, 1)

        if work[t][t] < 0:
            negate_row(t)

    diagonal = np.array(work, dtype=np.int64)
    return SmithNormalForm(
        diagonal=diagonal,
        left=np.array(left, dtype=np.int64),
        right=np.array(right, dtype=np.int64),
        left_inverse=np.array(left_inverse, dtype=np.int64),
    )


def coset_representatives(matrix: ArrayLike, *, atol: float = 1e-6) -> NDArray[np.int64]:
    """Representatives of ``Z^n / M Z^n`` for a non-singular integer ``M``.

    The result has shape ``(|det M|, n)``: one representative per class, each
    class exactly once.  The zero vector is always the first row, so the
    identity class comes first.

    Raises ``ValueError`` when ``M`` is singular, in which case the quotient is
    infinite.
    """

    snf = smith_normal_form(matrix, atol=atol)
    factors = snf.invariant_factors
    if any(factor == 0 for factor in factors):
        raise ValueError("matrix is singular: Z^n / M Z^n is infinite")

    n = len(factors)
    ranges = [range(factor) for factor in factors]
    # ``U^-1 y`` with ``y`` in the box is a transversal of ``Z^n / M Z^n``.
    box = np.array(list(product(*ranges)), dtype=np.int64).reshape(-1, n)
    representatives = box @ snf.left_inverse.T

    expected = int(np.prod(factors))
    if representatives.shape[0] != expected:
        raise AssertionError("coset enumeration produced the wrong number of classes")
    # ``|det M| == prod d_i`` because U and V are unimodular; assert it so a
    # bug in the Smith reduction cannot silently produce a wrong fiber.
    array = as_int_array(np.asarray(matrix), name="matrix", atol=atol)
    if n == 3 and abs(integer_det3(array)) != expected:
        raise AssertionError("coset enumeration disagrees with |det M|")
    return representatives
