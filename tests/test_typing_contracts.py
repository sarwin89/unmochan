"""Regression tests for the constructors and guards added while type-cleaning.

The type annotations of the package are now checked (``mypy unmochan`` is
clean), and making them honest changed a handful of runtime paths: new
normalizing constructors, stricter argument validation in the CLI, and one
genuine Python 3.10 incompatibility.  Each of those changes is pinned here so a
later refactor cannot quietly undo it.
"""

from __future__ import annotations

import numpy as np
import pytest
import typer

from unmochan.cli.common import parse_broadening_kind, parse_code, parse_kpoint
from unmochan.core.kpoints import KPoint, fiber_kpoint_mappings, fold_kpoints_to_supercell
from unmochan.core.numerics import TransformLike  # noqa: F401  (public alias)
from unmochan.core.tight_binding import tight_binding_orbital_weights
from unmochan.core.transformations import TransformationMatrix


def test_transformation_matrix_from_values_rounds_floats() -> None:
    """A float matrix that is integral to within rounding is accepted."""

    values = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    transform = TransformationMatrix.from_values(values)
    assert transform.matrix.dtype == np.int64
    assert transform.determinant == 2
    assert transform.residual == 0.0


def test_transformation_matrix_from_values_rejects_non_integer() -> None:
    with pytest.raises(ValueError):
        TransformationMatrix.from_values(np.eye(3) * 1.5)


def test_transformation_matrix_from_values_matches_direct_construction() -> None:
    matrix = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 1]], dtype=np.int64)
    assert np.array_equal(
        TransformationMatrix.from_values(matrix.astype(float)).matrix,
        TransformationMatrix(matrix).matrix,
    )


def test_fold_kpoints_accepts_plain_sequences() -> None:
    """``fold_kpoints_to_supercell`` normalizes array-likes into k-points."""

    transform = TransformationMatrix(np.diag([2, 1, 1]).astype(np.int64))
    mappings = fold_kpoints_to_supercell([[0.5, 0.0, 0.0], KPoint(np.zeros(3))], transform)
    assert len(mappings) == 2
    for mapping in mappings:
        assert mapping.primitive.fractional.dtype == np.float64
        assert mapping.supercell.fractional.dtype == np.float64
    assert np.allclose(mappings[0].supercell.fractional, np.zeros(3))


def test_fiber_kpoint_mappings_accept_plain_sequences() -> None:
    transform = TransformationMatrix(np.diag([2, 1, 1]).astype(np.int64))
    mappings = fiber_kpoint_mappings([[0.0, 0.0, 0.0]], transform)
    assert len(mappings) == 2
    assert sum(mapping.primitive.weight for mapping in mappings) == pytest.approx(1.0)


def test_orbital_weights_accept_integer_array_groups() -> None:
    """Groups may be integer arrays, not only plain lists of ``int``."""

    cells = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.int64)
    coefficients = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex128)
    kpoints = np.zeros((1, 3))
    from_lists = tight_binding_orbital_weights(cells, coefficients, kpoints, [[0], [1]])
    from_arrays = tight_binding_orbital_weights(
        cells,
        coefficients,
        kpoints,
        [np.array([0], dtype=np.int64), np.array([1], dtype=np.int64)],
    )
    assert np.allclose(from_lists, from_arrays)


def test_parse_code_returns_canonical_backend() -> None:
    assert parse_code(" VASP ") == "vasp"
    assert parse_code("Qe") == "qe"
    with pytest.raises(typer.BadParameter):
        parse_code("abinit")


def test_parse_broadening_kind_returns_canonical_kind() -> None:
    assert parse_broadening_kind("Gaussian") == "gaussian"
    assert parse_broadening_kind(" lorentzian") == "lorentzian"
    with pytest.raises(typer.BadParameter):
        parse_broadening_kind("voigt")


def test_parse_kpoint_produces_float_array() -> None:
    point = parse_kpoint("G:0,0,0")
    assert point.label == "G"
    assert point.fractional.dtype == np.float64
    with pytest.raises(typer.BadParameter):
        parse_kpoint("0,0")


def test_serialization_timestamp_uses_stdlib_utc() -> None:
    """``datetime.UTC`` is 3.11+, but the package advertises ``>=3.10``."""

    import unmochan.io.serialization as serialization

    assert not hasattr(serialization, "UTC")
    assert serialization.datetime.now(serialization.timezone.utc).tzinfo is not None
