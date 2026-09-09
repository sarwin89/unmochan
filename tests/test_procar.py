"""PROCAR parsing and generic selector resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from synthetic_procar import write_synthetic_procar

from unmochan.core.projections import parse_projection_selectors
from unmochan.core.site_projection import (
    detect_layers,
    orbital_group_mask,
    projection_fractions,
    resolve_site_mask,
    resolve_site_orbital_mask,
)
from unmochan.core.structures import Structure
from unmochan.io.procar import read_procar

ORBITALS = ("s", "py", "pz", "px")


def _projections(n_spin: int = 1, n_k: int = 2, n_b: int = 2, n_ions: int = 2) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 1.0, size=(n_spin, n_k, n_b, n_ions, len(ORBITALS)))


def _write(tmp_path: Path, **kwargs) -> Path:
    projections = kwargs.pop("projections")
    n_spin, n_k, n_b = projections.shape[:3]
    return write_synthetic_procar(
        tmp_path / "PROCAR",
        kpoints=kwargs.pop("kpoints", np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])[:n_k]),
        kpoint_weights=np.full(n_k, 1.0 / n_k),
        energies=np.arange(n_spin * n_k * n_b, dtype=float).reshape(n_spin, n_k, n_b),
        occupations=np.ones((n_spin, n_k, n_b)),
        projections=projections,
        orbital_labels=ORBITALS,
        **kwargs,
    )


def test_read_procar_roundtrips_a_collinear_file(tmp_path: Path):
    projections = _projections()
    path = _write(tmp_path, projections=projections)

    data = read_procar(path)

    assert data.orbital_labels == ORBITALS
    assert data.n_spin == 1
    assert (data.n_kpoints, data.n_bands, data.n_ions) == (2, 2, 2)
    assert not data.noncollinear
    assert np.allclose(data.projections, projections, atol=1e-3)
    assert np.allclose(data.kpoints[1], [0.5, 0.0, 0.0])
    assert np.allclose(data.kpoint_weights, 0.5)


def test_read_procar_keeps_both_spin_channels(tmp_path: Path):
    projections = _projections(n_spin=2)
    path = _write(tmp_path, projections=projections)

    data = read_procar(path)

    assert data.n_spin == 2
    assert np.allclose(data.projections, projections, atol=1e-3)


def test_read_procar_separates_the_noncollinear_magnetization(tmp_path: Path):
    projections = _projections()
    magnetization = np.linspace(-1.0, 1.0, projections[0].size).reshape(projections[0].shape)
    magnetization = np.stack([magnetization, -magnetization, 0.5 * magnetization], axis=0)
    path = _write(tmp_path, projections=projections, magnetization=magnetization)

    data = read_procar(path)

    assert data.noncollinear
    assert data.n_spin == 1
    assert np.allclose(data.projections, projections, atol=1e-3)
    assert data.magnetization is not None
    assert np.allclose(data.magnetization, magnetization, atol=1e-3)


def test_read_procar_skips_phase_tables(tmp_path: Path):
    projections = _projections()
    path = _write(tmp_path, projections=projections, phase=True)

    data = read_procar(path)

    assert not data.noncollinear
    assert np.allclose(data.projections, projections, atol=1e-3)


def test_read_procar_rejects_a_truncated_file(tmp_path: Path):
    projections = _projections()
    path = _write(tmp_path, projections=projections)
    lines = path.read_text().splitlines()
    cut = next(index for index, line in enumerate(lines) if line.startswith("band     2"))
    path.write_text("\n".join(lines[:cut]) + "\n")

    with pytest.raises(ValueError, match="missing at least one"):
        read_procar(path)


def test_projection_fractions_of_a_partition_sum_to_one(tmp_path: Path):
    projections = _projections()
    data = read_procar(_write(tmp_path, projections=projections))
    first = np.zeros((2, len(ORBITALS)), dtype=bool)
    first[0] = True
    second = ~first

    total = projection_fractions(data.projections[0], first) + projection_fractions(
        data.projections[0], second
    )

    assert np.allclose(total, 1.0)


def test_projection_fractions_are_zero_for_an_empty_table():
    values = np.zeros((3, 2, 4))
    mask = np.ones((2, 4), dtype=bool)

    assert np.allclose(projection_fractions(values, mask), 0.0)


def test_projection_fractions_reject_signed_tables():
    values = np.array([[[0.5, -0.5]]])
    mask = np.ones((1, 2), dtype=bool)

    with pytest.raises(ValueError, match="nonnegative"):
        projection_fractions(values, mask)


def test_orbital_group_mask_matches_shells_and_labels():
    assert list(orbital_group_mask(ORBITALS, "p")) == [False, True, True, True]
    assert list(orbital_group_mask(ORBITALS, "px")) == [False, False, False, True]
    assert list(orbital_group_mask(ORBITALS, "*")) == [True] * 4
    with pytest.raises(ValueError, match="no orbital"):
        orbital_group_mask(ORBITALS, "dxy")


def test_orbital_group_mask_normalizes_dx2_labels():
    labels = ("dxy", "dx2-y2", "dz2")
    assert list(orbital_group_mask(labels, "dx2")) == [False, True, False]
    assert list(orbital_group_mask(labels, "d")) == [True, True, True]


def _slab() -> Structure:
    lattice = np.diag([1.0, 1.0, 12.0])
    frac = np.array(
        [
            [0.0, 0.0, 0.10],
            [0.5, 0.5, 0.10],
            [0.0, 0.0, 0.30],
            [0.5, 0.5, 0.30],
        ]
    )
    return Structure(lattice=lattice, species=("A", "B", "A", "B"), frac_coords=frac)


def test_detect_layers_finds_the_slab_layers():
    layers = detect_layers(_slab(), axis=2, tol=0.5)

    assert layers.n_layers == 2
    assert list(layers.index) == [0, 0, 1, 1]


def test_detect_layers_uses_the_interplane_normal_for_a_sheared_cell():
    lattice = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [6.0, 0.0, 12.0]])
    frac = np.array([[0.0, 0.0, 0.1], [0.0, 0.0, 0.3]])
    structure = Structure(lattice=lattice, species=("A", "A"), frac_coords=frac)

    layers = detect_layers(structure, axis=2, tol=0.5)

    # The in-plane shear of the stacking vector must not split or merge layers.
    assert layers.n_layers == 2
    assert np.allclose(layers.heights, [1.2, 3.6])


def test_resolve_site_mask_combines_namespaces():
    structure = _slab()
    selectors = parse_projection_selectors(["species:A", "layer:1"])

    mask = resolve_site_mask(selectors, structure)

    assert list(mask) == [False, False, True, False]


def test_resolve_site_mask_unions_within_a_namespace():
    structure = _slab()
    selectors = parse_projection_selectors(["species:A", "species:B", "layer:0"])

    mask = resolve_site_mask(selectors, structure)

    assert list(mask) == [True, True, False, False]


def test_resolve_site_mask_handles_surfaces_and_wildcards():
    structure = _slab()

    top = resolve_site_mask(parse_projection_selectors(["surface:top"]), structure)
    bottom = resolve_site_mask(parse_projection_selectors(["surface:bottom"]), structure)
    everything = resolve_site_mask(parse_projection_selectors(["species:*"]), structure)

    assert list(top) == [False, False, True, True]
    assert list(bottom) == [True, True, False, False]
    assert list(everything) == [True] * 4


def test_resolve_site_mask_requires_site_groups_for_user_conventions():
    structure = _slab()
    selectors = parse_projection_selectors(["region:defect"])

    with pytest.raises(ValueError, match="user convention"):
        resolve_site_mask(selectors, structure)

    mask = resolve_site_mask(selectors, structure, site_groups={"region": {"defect": [1, 3]}})
    assert list(mask) == [False, True, False, True]


def test_resolve_site_mask_uses_the_minimum_image_for_defect_shells():
    structure = _slab()
    selectors = parse_projection_selectors(["distance_from_defect:1.3"])

    mask = resolve_site_mask(selectors, structure, defect_center=np.array([0.0, 0.0, 0.10]))

    # Only the two sites within 1.3 angstrom of the defect centre; the periodic
    # image along x/y is what makes the in-plane neighbours 0.707 away.
    assert list(mask) == [True, True, False, False]


def test_resolve_site_mask_rejects_spin_selectors():
    with pytest.raises(ValueError, match="spin selectors"):
        resolve_site_mask(parse_projection_selectors(["spin:z"]), _slab())


def test_resolve_site_orbital_mask_is_the_product_of_the_two_masks():
    structure = _slab()
    selectors = parse_projection_selectors(["species:A", "orbital:p"])

    mask = resolve_site_orbital_mask(selectors, structure, ORBITALS)

    expected = np.zeros((4, 4), dtype=bool)
    expected[np.ix_([0, 2], [1, 2, 3])] = True
    assert np.array_equal(mask, expected)
