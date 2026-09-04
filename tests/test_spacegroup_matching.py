"""How the symmetry finder assigns sites to their images.

Two things are pinned here.  First, that the assignment is decided *exactly*:
when the tolerance is wide enough for a site to be near two candidates, taking
the first free one in index order can consume a partner another image needs and
reject a symmetry that exists.  Second, that the group shortcut used to keep the
cost quadratic -- the translations completing a rotation form a coset of the
pure-translation subgroup, and the permutation of a composite is the composition
of the permutations -- reports exactly the operations a brute-force search over
every (rotation, translation) pair reports, with each one still verified against
``symprec`` site by site.
"""

from __future__ import annotations

import numpy as np

from unmochan.core.spacegroup import (
    _assignment_holds,
    _candidate_translations,
    _feasible_partners,
    _label_codes,
    _reduce,
    _site_assignment,
    _unique_translations,
    lattice_point_group,
    site_labels,
    space_group_operations,
)
from unmochan.core.structures import Structure

INVERSION = -np.eye(3, dtype=np.int64)
IDENTITY = np.eye(3, dtype=np.int64)


def two_close_sites() -> tuple[np.ndarray, np.ndarray]:
    """Two sites 0.1 apart, which a tolerance of 0.06 nearly merges."""

    return np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0]]), np.array([0, 0])


def test_the_ambiguous_case_really_is_ambiguous() -> None:
    frac, codes = two_close_sites()
    images = _reduce(frac @ INVERSION.T + np.array([0.05, 0.0, 0.0]))
    feasible = _feasible_partners(frac, images, codes, codes, 0.06, None, None, 1e-3)
    # The image of the first site is within tolerance of *both* sites.
    assert feasible[0].sum() == 2
    # The image of the second is within tolerance of the first site only.
    assert feasible[1].tolist() == [True, False]


def test_a_greedy_assignment_would_reject_a_symmetry_that_exists() -> None:
    frac, codes = two_close_sites()
    partners = _site_assignment(frac, codes, INVERSION, np.array([0.05, 0.0, 0.0]), 0.06)
    assert partners is not None
    assert sorted(partners.tolist()) == [0, 1]
    # Greedy in index order takes site 0 for the first image and then has
    # nothing left for the second.
    assert partners[0] == 1


def test_a_longer_augmenting_path_is_followed() -> None:
    """Three images whose candidate sets force the search to back up twice."""

    frac = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    codes = np.array([0, 0, 0])
    partners = _site_assignment(frac, codes, IDENTITY, np.array([0.1, 0.0, 0.0]), 0.11)
    assert partners is not None
    assert sorted(partners.tolist()) == [0, 1, 2]


def test_a_site_with_no_partner_at_all_fails() -> None:
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    codes = np.array([0, 1])
    assert _site_assignment(frac, codes, IDENTITY, np.array([0.25, 0.0, 0.0]), 1e-5) is None


def test_a_decoration_mismatch_fails_even_when_the_positions_line_up() -> None:
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    codes = np.array([0, 1])
    assert _site_assignment(frac, codes, IDENTITY, np.array([0.5, 0.0, 0.0]), 1e-5) is None


def test_translations_are_compared_on_the_torus() -> None:
    candidates = np.array([[0.0, 0.0, 0.0], [1.0 - 1e-9, 0.0, 0.0], [0.5, 0.0, 0.0]])
    kept = _unique_translations(candidates, 1e-5)
    assert len(kept) == 2


def test_a_known_permutation_is_re_verified_against_the_tolerance() -> None:
    frac = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    codes = np.array([0, 0])
    translation = np.array([0.5, 0.0, 0.0])
    right = np.array([1, 0])
    wrong = np.array([0, 1])
    assert _assignment_holds(frac, codes, IDENTITY, translation, right, 1e-5, None, None, 1e-3)
    assert not _assignment_holds(frac, codes, IDENTITY, translation, wrong, 1e-5, None, None, 1e-3)


def brute_force_operations(
    structure: Structure, *, symprec: float = 1e-5
) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    """Every (rotation, translation) pair tested on its own, no group theory."""

    frac = _reduce(np.asarray(structure.frac_coords, dtype=float))
    labels = site_labels(structure, None)
    codes = _label_codes(labels)
    metric_tol = float(
        symprec
        * max(1.0, np.abs(np.asarray(structure.lattice) @ np.asarray(structure.lattice).T).max())
    )
    found = []
    for rotation in lattice_point_group(structure.lattice, tol=metric_tol):
        for translation in _candidate_translations(
            frac, list(range(len(labels))), 0, rotation, symprec
        ):
            if _site_assignment(frac, codes, rotation, translation, symprec) is not None:
                found.append(
                    (
                        tuple(int(v) for v in rotation.ravel()),
                        tuple(np.round(_reduce(translation), 9)),
                    )
                )
    return sorted(found)


def as_keys(operations) -> list[tuple[tuple[int, ...], tuple[float, ...]]]:
    return sorted(
        (
            tuple(int(v) for v in op.rotation.ravel()),
            tuple(np.round(_reduce(np.asarray(op.translation)), 9)),
        )
        for op in operations
    )


def cubic_supercell(n: int) -> Structure:
    axis = np.arange(n) / n
    coords = np.array([(i, j, k) for i in axis for j in axis for k in axis])
    return Structure(
        lattice=np.eye(3) * n,
        species=("Si",) * len(coords),
        frac_coords=coords,
    )


def test_the_coset_shortcut_reproduces_the_brute_force_search() -> None:
    structure = cubic_supercell(2)
    assert as_keys(space_group_operations(structure)) == brute_force_operations(structure)


def test_the_coset_shortcut_reproduces_it_for_a_decorated_supercell() -> None:
    structure = cubic_supercell(2)
    species = list(structure.species)
    species[0] = "Ge"
    decorated = Structure(
        lattice=structure.lattice,
        species=tuple(species),
        frac_coords=structure.frac_coords,
    )
    assert as_keys(space_group_operations(decorated)) == brute_force_operations(decorated)


def test_a_supercell_reports_one_operation_per_cell_and_rotation() -> None:
    """`|det T|` pure translations times the 48 rotations of the cubic lattice."""

    operations = space_group_operations(cubic_supercell(3))
    assert len(operations) == 48 * 27
    rotations = {tuple(int(v) for v in op.rotation.ravel()) for op in operations}
    assert len(rotations) == 48
