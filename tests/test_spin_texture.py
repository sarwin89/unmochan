"""Spin-resolved unfolding of noncollinear (spinor) states."""

from __future__ import annotations

import numpy as np
import pytest

from unmochan.core import (
    PlaneWaveKPointData,
    TransformationMatrix,
    compute_plane_wave_spin_textures,
    compute_plane_wave_unfolding_weights,
    diagnose_spin_textures,
    fiber_kpoints,
    spin_expectation_from_coefficients,
    spin_texture_from_coefficients,
    weights_from_coefficients,
)

pytestmark = pytest.mark.unit


def _g_grid(radius: int = 2) -> np.ndarray:
    span = range(-radius, radius + 1)
    return np.array([[i, j, k] for i in span for j in span for k in span], dtype=int)


def _random_spinor(rng: np.random.Generator, n_bands: int, n_g: int) -> np.ndarray:
    real = rng.normal(size=(n_bands, 2, n_g))
    imag = rng.normal(size=(n_bands, 2, n_g))
    return real + 1j * imag


def test_pure_spinor_saturates_the_bound():
    """A state with one common spin direction has |S| equal to its weight."""

    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    rng = np.random.default_rng(0)
    amplitude = rng.normal(size=g_vectors.shape[0]) + 1j * rng.normal(size=g_vectors.shape[0])
    # Same spinor direction (1, i)/sqrt(2) -- pointing along +y -- on every G.
    direction = np.array([1.0, 1.0j]) / np.sqrt(2.0)
    coefficients = (direction[None, :, None] * amplitude[None, None, :]).astype(complex)

    primitive = np.array([0.25, 0.0, 0.0])
    folded = primitive @ transform.T
    texture = spin_texture_from_coefficients(g_vectors, coefficients, primitive, folded, transform)
    weight = weights_from_coefficients(g_vectors, coefficients, primitive, folded, transform)

    assert np.allclose(np.linalg.norm(texture, axis=1), weight)
    # The direction is +y, so only the transverse component is non-zero.
    assert np.allclose(texture[:, 0], 0.0, atol=1e-12)
    assert np.allclose(texture[:, 2], 0.0, atol=1e-12)
    assert texture[0, 1] > 0.0


def test_component_resolved_weights_only_see_the_z_component():
    """Squaring the components separately loses the transverse spin."""

    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    rng = np.random.default_rng(1)
    coefficients = _random_spinor(rng, 3, g_vectors.shape[0])
    primitive = np.array([0.25, 0.0, 0.0])
    folded = primitive @ transform.T

    texture = spin_texture_from_coefficients(g_vectors, coefficients, primitive, folded, transform)
    components = weights_from_coefficients(
        g_vectors, coefficients, primitive, folded, transform, component_resolved=True
    )
    assert np.allclose(texture[:, 2], components[:, 0] - components[:, 1])
    # A random state has genuine transverse weight that the component split misses.
    assert np.max(np.abs(texture[:, :2])) > 1e-3


def test_texture_never_exceeds_the_weight():
    g_vectors = _g_grid(2)
    transform = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    rng = np.random.default_rng(2)
    coefficients = _random_spinor(rng, 4, g_vectors.shape[0])
    folded = np.array([0.0, 0.0, 0.0])
    for primitive in fiber_kpoints(folded, TransformationMatrix(transform)):
        texture = spin_texture_from_coefficients(
            g_vectors, coefficients, primitive, folded, transform
        )
        weight = weights_from_coefficients(g_vectors, coefficients, primitive, folded, transform)
        assert np.all(np.linalg.norm(texture, axis=1) <= weight + 1e-12)
        assert np.all(np.linalg.norm(texture, axis=1) <= 1.0 + 1e-12)


def test_fiber_spin_sum_rule():
    """Textures over a complete fiber add up to the spin of the supercell state."""

    g_vectors = _g_grid(2)
    transform = np.array([[2.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 3.0]])
    rng = np.random.default_rng(3)
    coefficients = _random_spinor(rng, 4, g_vectors.shape[0])
    supercell_kpoint = np.array([0.0, 0.0, 0.0])

    total = np.zeros((coefficients.shape[0], 3))
    for primitive in fiber_kpoints(supercell_kpoint, TransformationMatrix(transform)):
        total += spin_texture_from_coefficients(
            g_vectors, coefficients, primitive, supercell_kpoint, transform
        )
    assert np.allclose(total, spin_expectation_from_coefficients(coefficients))


def test_texture_is_representative_independent():
    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    rng = np.random.default_rng(4)
    coefficients = _random_spinor(rng, 2, g_vectors.shape[0])
    primitive = np.array([0.25, 0.0, 0.0])
    folded = primitive @ transform.T

    shift = np.array([1, 0, 0])
    shifted = spin_texture_from_coefficients(
        g_vectors - shift[None, :],
        coefficients,
        primitive,
        folded + shift,
        transform,
    )
    reference = spin_texture_from_coefficients(
        g_vectors, coefficients, primitive, folded, transform
    )
    assert np.allclose(shifted, reference)


def test_backend_neutral_textures_and_diagnostics():
    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    rng = np.random.default_rng(5)
    entries = []
    for fraction in (0.0, 0.25, 0.5):
        primitive = np.array([fraction, 0.0, 0.0])
        entries.append(
            PlaneWaveKPointData(
                primitive_kpoint=primitive,
                folded_supercell_kpoint=primitive @ transform.T,
                g_supercell=g_vectors,
                coefficients=_random_spinor(rng, 3, g_vectors.shape[0]),
            )
        )
    textures = compute_plane_wave_spin_textures(entries, transform)
    weights = compute_plane_wave_unfolding_weights(entries, transform)
    assert textures.shape == (3, 3, 3)
    diagnostics = diagnose_spin_textures(textures, weights)
    assert diagnostics.max_texture_excess == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.max_norm <= 1.0 + 1e-12
    assert diagnostics.to_dict()["n_bands"] == 3


def test_collinear_input_is_rejected():
    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    coefficients = np.ones((2, 1, g_vectors.shape[0]), dtype=complex)
    primitive = np.zeros(3)
    with pytest.raises(ValueError, match="exactly two spinor components"):
        spin_texture_from_coefficients(
            g_vectors, coefficients, primitive, primitive @ transform.T, transform
        )


def test_zero_state_has_zero_texture():
    g_vectors = _g_grid(1)
    transform = np.diag([2, 1, 1]).astype(float)
    coefficients = np.zeros((2, 2, g_vectors.shape[0]), dtype=complex)
    primitive = np.zeros(3)
    texture = spin_texture_from_coefficients(
        g_vectors, coefficients, primitive, primitive @ transform.T, transform
    )
    assert np.allclose(texture, 0.0)
    assert np.allclose(spin_expectation_from_coefficients(coefficients), 0.0)


def _write_spinor_wavecar(path, *, lattice, encut, kpoints, up, down, energies):
    """Write a synthetic noncollinear WAVECAR with one band per k-point."""

    from synthetic_wavecar import write_synthetic_wavecar

    from unmochan.io.vasp_wfc import generate_vasp_g_vectors

    n_g = None
    blocks = []
    for kpoint in kpoints:
        g_vectors = generate_vasp_g_vectors(lattice, kpoint, encut)
        n_g = len(g_vectors)
        block = np.zeros((1, 2, n_g), dtype=np.complex64)
        block[0, 0, :] = up
        block[0, 1, :] = down
        blocks.append(block)
    write_synthetic_wavecar(
        path,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.asarray(kpoints, dtype=float),
        energies=np.asarray(energies, dtype=float),
        occupations=np.ones_like(np.asarray(energies, dtype=float)),
        coefficients=np.stack(blocks),
        n_plane_waves=2 * n_g,
    )
    return n_g


def test_spin_texture_from_a_noncollinear_wavecar(tmp_path):
    """A state polarized along +y unfolds to a texture of length = its weight."""

    from unmochan.io.vasp_wfc import (
        compute_spin_texture_from_wavecar,
        generate_vasp_g_vectors,
    )

    wavecar = tmp_path / "WAVECAR.spinor"
    lattice = np.eye(3)
    encut = 200.0
    _write_spinor_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        kpoints=[[0.0, 0.0, 0.0]],
        up=1.0,
        down=1.0j,
        energies=[[0.4]],
    )

    transform = np.diag([2, 1, 1])
    _kpoints, energies, weights, textures = compute_spin_texture_from_wavecar(
        wavecar,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        transform,
    )

    g_vectors = generate_vasp_g_vectors(lattice, np.zeros(3), encut)
    expected = np.count_nonzero(g_vectors[:, 0] % 2 == 0) / len(g_vectors)
    assert np.allclose(energies, [[0.4]])
    assert np.allclose(weights, [[expected]])
    assert textures.shape == (1, 1, 3)
    assert np.allclose(textures[0, 0], [0.0, expected, 0.0])


def test_collinear_wavecar_has_no_spin_texture(tmp_path):
    from synthetic_wavecar import write_synthetic_wavecar

    from unmochan.io.vasp_wfc import (
        compute_spin_texture_from_wavecar,
        generate_vasp_g_vectors,
    )

    wavecar = tmp_path / "WAVECAR.collinear"
    lattice = np.eye(3)
    encut = 200.0
    g_vectors = generate_vasp_g_vectors(lattice, np.zeros(3), encut)
    write_synthetic_wavecar(
        wavecar,
        lattice=lattice,
        encut=encut,
        rtag=45200,
        kpoints=np.array([[0.0, 0.0, 0.0]]),
        energies=np.array([[-1.0]]),
        occupations=np.array([[1.0]]),
        coefficients=np.ones((1, 1, 1, len(g_vectors)), dtype=np.complex64),
    )

    with pytest.raises(ValueError, match="exactly two spinor components"):
        compute_spin_texture_from_wavecar(
            wavecar,
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.diag([2, 1, 1]),
        )


def test_weights_cli_writes_a_spin_texture_table(tmp_path):
    from typer.testing import CliRunner

    from unmochan.cli.main import app
    from unmochan.io.qe import read_spin_texture_table
    from unmochan.io.vasp import write_vasp_path_files

    path_json = tmp_path / "path.json"
    kpoints_file = tmp_path / "KPOINTS"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"G","k":[0,0,0]}]}'
    )
    write_vasp_path_files(path_json, kpoints=kpoints_file, kmap=kmap, ticks=ticks)

    wavecar = tmp_path / "WAVECAR"
    _write_spinor_wavecar(
        wavecar,
        lattice=np.eye(3),
        encut=200.0,
        kpoints=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        up=1.0,
        down=0.0,
        energies=[[-1.0], [-0.5]],
    )

    weights_file = tmp_path / "weights.dat"
    texture_file = tmp_path / "texture.dat"
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "weights",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--out",
            str(weights_file),
            "--spin-texture",
            str(texture_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert texture_file.exists()
    textures = read_spin_texture_table(texture_file, 2, 1)
    # Pure spin-up: the texture points along +z with length equal to the weight.
    assert np.allclose(textures[:, :, 0], 0.0)
    assert np.allclose(textures[:, :, 1], 0.0)
    assert np.all(textures[:, :, 2] > 0.0)


def test_spin_texture_is_refused_for_symmetry_reduced_input(tmp_path):
    from typer.testing import CliRunner

    from unmochan.cli.main import app
    from unmochan.io.vasp import write_vasp_path_files

    path_json = tmp_path / "path.json"
    kpoints_file = tmp_path / "KPOINTS"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"G","k":[0,0,0]}]}'
    )
    write_vasp_path_files(path_json, kpoints=kpoints_file, kmap=kmap, ticks=ticks)

    wavecar = tmp_path / "WAVECAR"
    _write_spinor_wavecar(
        wavecar,
        lattice=np.eye(3),
        encut=200.0,
        kpoints=[[0.0, 0.0, 0.0]],
        up=1.0,
        down=0.0,
        energies=[[-1.0]],
    )

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "weights",
            "--code",
            "vasp",
            "--kmap",
            str(kmap),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--wavecar",
            str(wavecar),
            "--out",
            str(tmp_path / "weights.dat"),
            "--spin-texture",
            str(tmp_path / "texture.dat"),
            "--reuse-kpoints",
        ],
    )

    assert result.exit_code != 0
    assert "pseudovector" in result.output


def _write_spinor_wfc_dat(path, xk, miller, up, down):
    """Write a minimal noncollinear (``npol = 2``) QE ``wfc*.dat`` file."""

    import struct

    def record(payload: bytes) -> bytes:
        marker = struct.pack("<i", len(payload))
        return marker + payload + marker

    igwx = int(miller.shape[0])
    n_bands = int(up.shape[0])
    chunks = [
        record(struct.pack("<i3diid", 1, xk[0], xk[1], xk[2], 1, 0, 1.0)),
        record(struct.pack("<4i", igwx, igwx, 2, n_bands)),
        record(struct.pack("<9d", *np.eye(3).reshape(-1))),
        record(np.ascontiguousarray(miller, dtype="<i4").tobytes()),
    ]
    for band in range(n_bands):
        spinor = np.concatenate([up[band], down[band]])
        chunks.append(record(np.asarray(spinor, dtype="<c16").tobytes()))
    path.write_bytes(b"".join(chunks))


def test_spin_texture_from_a_noncollinear_qe_save(tmp_path):
    from unmochan.io.qe_wfc import compute_spin_texture_from_qe_save

    save_dir = tmp_path / "pwscf.save"
    save_dir.mkdir()
    miller = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    up = np.ones((1, 3), dtype=complex)
    down = 1j * np.ones((1, 3), dtype=complex)
    _write_spinor_wfc_dat(save_dir / "wfc1.dat", (0.0, 0.0, 0.0), miller, up, down)

    transform = np.diag([2, 1, 1])
    weights, textures = compute_spin_texture_from_qe_save(
        save_dir,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        transform,
        1,
        use_xml_metadata=False,
    )

    # Two of the three plane waves have even G_x, so they unfold onto k = 0.
    assert np.allclose(weights, [[2.0 / 3.0]])
    assert textures.shape == (1, 1, 3)
    assert np.allclose(textures[0, 0], [0.0, 2.0 / 3.0, 0.0])
    assert np.linalg.norm(textures[0, 0]) == pytest.approx(weights[0, 0])
