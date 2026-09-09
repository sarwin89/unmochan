"""Symmetry operations must belong to the crystal, not just to the lattice.

`unmochan.core.symmetry.map_kpoints_to_stored` can only test that a supplied
operation is unimodular and compatible with the two lattices; whether the state
stored at the representative k-point really is the symmetry image of the
requested one is a fact about the crystal.  The mathematics is in
``RequestProject/Unfolding/SymmetryHypothesis.lean``:
``UnfoldLab.weight_symmetry_of_norm_eq`` (the hypothesis stated),
``UnfoldLab.exists_weight_ne_of_coefficients`` (it is load-bearing),
``UnfoldLab.exists_latticeSymmetry_bandEnergy_ne`` (a lattice symmetry need not
be a Hamiltonian symmetry) and
``UnfoldLab.zeeman_invariant_iff_moment_invariant`` (the magnetic case).
"""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core.spacegroup import (
    cartesian_rotation,
    detect_primitive_operations,
    site_labels,
    space_group_operations,
    validate_primitive_operations,
)
from unmochan.core.structures import Structure
from unmochan.core.symmetry import map_kpoints_to_stored

pytestmark = pytest.mark.unit


def _cubic_atom() -> Structure:
    return Structure(np.eye(3) * 3.0, ("Fe",), np.array([[0.0, 0.0, 0.0]]))


def _ferrimagnetic_chain() -> Structure:
    """Three equally spaced sites of one species along z."""

    return Structure(
        np.diag([3.0, 3.0, 9.0]),
        ("Fe", "Fe", "Fe"),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1 / 3], [0.0, 0.0, 2 / 3]]),
    )


def _tetragonal_pair() -> Structure:
    """Two sites of one species, in a cell whose two in-plane axes differ."""

    return Structure(
        np.diag([3.0, 4.0, 10.0]),
        ("Fe", "Fe"),
        np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.0]]),
    )


def test_site_labels_separate_moments() -> None:
    structure = _ferrimagnetic_chain()
    assert site_labels(structure) == ("Fe", "Fe", "Fe")
    assert site_labels(structure, [1.0, 1.0, -1.0]) == ("Fe:0", "Fe:0", "Fe:1")
    # Moments within the tolerance are one label.
    assert site_labels(structure, [1.0, 1.0 + 1e-6, -1.0]) == ("Fe:0", "Fe:0", "Fe:1")


def test_site_labels_reject_wrong_length() -> None:
    with pytest.raises(ValueError, match="collinear magmoms"):
        site_labels(_ferrimagnetic_chain(), [1.0, -1.0])


def test_collinear_moments_shrink_the_group() -> None:
    """A ferrimagnetic pattern removes rotations that the atoms alone allow."""

    structure = _ferrimagnetic_chain()
    transform = np.diag([1, 1, 3])
    nonmagnetic = detect_primitive_operations(structure, transform)
    magnetic = detect_primitive_operations(structure, transform, magmoms=[1.0, 2.0, 3.0])
    assert len(magnetic) < len(nonmagnetic)
    # Everything the magnetic search keeps was already there.
    for operation in magnetic:
        assert any(np.array_equal(operation, other) for other in nonmagnetic)


def test_antiferromagnetic_translation_does_not_change_the_rotations() -> None:
    """An operation surviving with *some* translation still serves the weights.

    An antiferromagnetic doubling loses half of its space-group operations to
    the moments, but every rotation survives paired with a different
    translation, and a translation only multiplies a coefficient by a phase.
    So the unfolding operation list is unchanged -- which is the correct
    answer, not an oversight.
    """

    structure = _ferrimagnetic_chain()
    transform = np.diag([1, 1, 3])
    assert len(space_group_operations(structure, magmoms=[1.0, -1.0, 1.0])) < len(
        space_group_operations(structure)
    )
    assert len(detect_primitive_operations(structure, transform, magmoms=[1.0, -1.0, 1.0])) == len(
        detect_primitive_operations(structure, transform)
    )


def test_noncollinear_moment_rotates_as_an_axial_vector() -> None:
    """A ferromagnet along z in a cubic cell keeps only the operations about z."""

    structure = _cubic_atom()
    full = space_group_operations(structure)
    along_z = space_group_operations(structure, magmoms=np.array([[0.0, 0.0, 2.0]]))
    assert len(full) == 48
    assert len(along_z) == 8  # the unitary part of 4/mm'm'
    for operation in along_z:
        cartesian = cartesian_rotation(structure.lattice, operation.rotation)
        image = float(np.sign(np.linalg.det(cartesian))) * (cartesian @ np.array([0.0, 0.0, 2.0]))
        assert np.allclose(image, [0.0, 0.0, 2.0], atol=1e-9)


def test_noncollinear_moments_reject_wrong_shape() -> None:
    with pytest.raises(ValueError, match="noncollinear magmoms"):
        space_group_operations(_cubic_atom(), magmoms=np.zeros((2, 3)))


def test_validation_accepts_the_detected_operations() -> None:
    structure = _tetragonal_pair()
    transform = np.diag([1, 1, 1])
    operations = detect_primitive_operations(structure, transform)
    validation = validate_primitive_operations(structure, operations, transform)
    assert validation.ok
    assert validation.invalid == ()
    assert all(check.crystal_symmetry for check in validation.checks)
    assert len(validation.summary()) == len(operations) + 1


def test_validation_rejects_an_axis_exchange_of_a_tetragonal_cell() -> None:
    """The pitfall of ``exists_latticeSymmetry_bandEnergy_ne``, in code.

    Exchanging x and y is unimodular and commutes with the identity transform,
    so ``map_kpoints_to_stored`` accepts it and happily serves one k-point's
    wavefunction for another -- even though the two axes of this cell have
    different lengths and the two k-points carry different energies.
    """

    structure = _tetragonal_pair()
    transform = np.eye(3, dtype=int)
    swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=int)

    # The kernel is perfectly happy.
    matches = map_kpoints_to_stored(
        np.array([[0.5, 0.0, 0.0]]),
        np.array([[0.0, 0.5, 0.0]]),
        transform,
        operations=np.stack([swap]),
    )
    assert len(matches) == 1

    validation = validate_primitive_operations(structure, np.stack([swap]), transform)
    assert not validation.ok
    assert validation.invalid == (0,)
    check = validation.checks[0]
    assert check.lattice_compatible
    assert not check.crystal_symmetry
    assert "not a symmetry" in check.reason


def test_validation_flags_a_time_reversal_image() -> None:
    """Inversion of a non-centrosymmetric cell is legitimate only with k -> -k."""

    structure = Structure(
        np.diag([3.0, 3.0, 9.0]),
        ("Fe", "O"),
        np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.3]]),
    )
    transform = np.diag([1, 1, 2])
    inversion = -np.eye(3, dtype=int)

    strict = validate_primitive_operations(structure, np.stack([inversion]), transform)
    relaxed = validate_primitive_operations(
        structure, np.stack([inversion]), transform, time_reversal=True
    )
    assert not strict.ok
    assert strict.checks[0].time_reversal
    assert "k -> -k" in strict.checks[0].reason
    assert relaxed.ok


def test_validation_rejects_a_non_integer_and_an_incompatible_matrix() -> None:
    structure = _tetragonal_pair()
    transform = np.diag([1, 1, 2])
    operations = np.stack([np.eye(3) * 0.5, np.eye(3) * 2])
    validation = validate_primitive_operations(structure, operations, transform)
    assert validation.invalid == (0, 1)
    assert validation.checks[0].reason == "not an integer matrix"
    assert not validation.checks[1].lattice_compatible
    assert "unimodular" in validation.checks[1].reason


def test_validation_uses_the_moments() -> None:
    """An operation of the atoms alone is rejected once the moments are given."""

    structure = _ferrimagnetic_chain()
    transform = np.diag([1, 1, 3])
    nonmagnetic = detect_primitive_operations(structure, transform)
    magnetic = detect_primitive_operations(structure, transform, magmoms=[1.0, 2.0, 3.0])
    surplus = np.stack(
        [
            operation
            for operation in nonmagnetic
            if not any(np.array_equal(operation, kept) for kept in magnetic)
        ]
    )
    assert len(surplus) > 0

    assert validate_primitive_operations(structure, surplus, transform).ok
    magnetic_validation = validate_primitive_operations(
        structure, surplus, transform, magmoms=[1.0, 2.0, 3.0]
    )
    assert not magnetic_validation.ok
    assert "magnetic" in magnetic_validation.checks[0].reason


def test_time_reversal_is_refused_for_a_magnetic_structure() -> None:
    structure = _ferrimagnetic_chain()
    with pytest.raises(ValueError, match="not valid for a magnetic structure"):
        validate_primitive_operations(
            structure,
            np.stack([np.eye(3, dtype=int)]),
            np.diag([1, 1, 3]),
            magmoms=[1.0, 2.0, 3.0],
            time_reversal=True,
        )
