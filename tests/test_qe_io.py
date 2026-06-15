from pathlib import Path

import numpy as np
import pytest

from unfoldlab.io.qe import (
    PathNode,
    build_qe_path,
    interpolate_path,
    qe_effective_band_structure,
    read_gnu_blocks,
    read_kmap,
    read_ticks,
    read_weight_table,
    weights_from_coefficient_table,
    write_kmap,
    write_qe_kpoints,
    write_ticks,
    write_unfolded,
)
from unfoldlab.io.qe_wfc import compute_weights_from_qe_save


def test_qe_path_interpolation_and_folding(tmp_path: Path):
    path_json = tmp_path / "path.json"
    path_json.write_text(
        """
{
  "transformation_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
  "path": [
    {"label": "G", "k": [0.0, 0.0, 0.0], "n": 3},
    {"label": "X", "k": [0.5, 0.0, 0.0], "n": 2},
    {"label": "M", "k": [0.5, 0.5, 0.0]}
  ]
}
""".strip()
    )

    qe_path = build_qe_path(path_json)

    assert qe_path.kpoints.shape == (4, 3)
    assert qe_path.labels == ("G", "", "X", "M")
    assert np.allclose(qe_path.supercell_folded[2], [0.0, 0.0, 0.0])


def test_qe_kmap_ticks_and_blocks_roundtrip(tmp_path: Path):
    qe_path = interpolate_path(
        [
            PathNode("G", np.array([0.0, 0.0, 0.0]), 2),
            PathNode("X", np.array([0.5, 0.0, 0.0]), 1),
        ],
        np.diag([2, 1, 1]),
    )
    kpoints_file = tmp_path / "qe_kpoints.in"
    kmap_file = tmp_path / "kmap.tsv"
    ticks_file = tmp_path / "ticks.tsv"
    write_qe_kpoints(kpoints_file, qe_path.supercell_folded)
    write_kmap(kmap_file, qe_path)
    write_ticks(ticks_file, qe_path)

    kmap = read_kmap(kmap_file)
    tick_x, tick_labels = read_ticks(ticks_file, kmap)

    assert "K_POINTS crystal" in kpoints_file.read_text()
    assert np.allclose(kmap.primitive_kpoints[-1], [0.5, 0.0, 0.0])
    assert tick_labels == ["G", "X"]
    assert np.allclose(tick_x, [0.0, 0.5])


def test_read_gnu_blocks_and_weight_table(tmp_path: Path):
    bands = tmp_path / "bands.dat.gnu"
    bands.write_text("0.0 -1.0\n1.0 -0.5\n\n0.0 2.0\n1.0 2.5\n")
    weights_file = tmp_path / "weights.dat"
    weights_file.write_text("# ik band weight\n1 1 0.25\n1 2 0.75\n2 1 0.5\n2 2 1.0\n")

    x_values, energies = read_gnu_blocks(bands)
    weights = read_weight_table(weights_file, 2, 2)

    assert np.allclose(x_values, [0.0, 1.0])
    assert np.allclose(energies, [[-1.0, 2.0], [-0.5, 2.5]])
    assert np.allclose(weights, [[0.25, 0.75], [0.5, 1.0]])


def test_coefficient_table_weights_are_normalized(tmp_path: Path):
    qe_path = interpolate_path(
        [
            PathNode("G", np.array([0.0, 0.0, 0.0]), 2),
            PathNode("X", np.array([0.5, 0.0, 0.0]), 1),
        ],
        np.diag([2, 1, 1]),
    )
    kmap_file = tmp_path / "kmap.tsv"
    coeff_file = tmp_path / "coefficients.dat"
    write_kmap(kmap_file, qe_path)
    kmap = read_kmap(kmap_file)
    coeff_file.write_text(
        "\n".join(
            [
                "# ik ib G1 G2 G3 Re Im",
                "1 1 0 0 0 1.0 0.0",
                "1 1 1 0 0 1.0 0.0",
                "1 2 1 0 0 1.0 0.0",
                "2 1 0 0 0 1.0 0.0",
                "2 1 1 0 0 1.0 0.0",
                "2 2 0 0 0 0.0 0.0",
                "2 2 1 0 0 2.0 0.0",
            ]
        )
        + "\n"
    )

    weights = weights_from_coefficient_table(coeff_file, kmap, np.diag([2, 1, 1]), 2)

    assert np.allclose(weights, [[0.5, 0.0], [0.5, 1.0]])


def test_unfolded_table_and_ebs_output(tmp_path: Path):
    kmap = read_kmap(_write_minimal_kmap(tmp_path))
    energies = np.array([[-1.0, 1.0], [-0.5, 1.5]])
    weights = np.array([[1.0, 0.25], [0.5, 0.75]])
    out = tmp_path / "unfolded.dat"

    write_unfolded(out, kmap, energies, weights)
    ebs = qe_effective_band_structure(kmap, energies, weights, reference_energy=0.2)

    assert "spectral_weight" in out.read_text()
    assert ebs.n_kpoints == 2
    assert ebs.n_bands == 2
    assert np.allclose(ebs.distances, [0.0, 0.5])
    assert ebs.reference_energy == 0.2


def test_compute_weights_from_synthetic_hdf5(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    save_dir = tmp_path / "qe.save"
    save_dir.mkdir()
    path = save_dir / "wfc1.hdf5"
    with h5py.File(path, "w") as handle:
        handle.attrs["ik"] = 1
        handle.attrs["ispin"] = 1
        handle.attrs["gamma_only"] = ".FALSE."
        handle.attrs["npol"] = 1
        handle.create_dataset("MillerIndices", data=np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]]))
        handle.create_dataset(
            "evc",
            data=np.array(
                [
                    [1.0, 0.0, 1.0, 0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
        )

    weights = compute_weights_from_qe_save(
        save_dir,
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.diag([2, 1, 1]),
        2,
        file_format="hdf5",
    )

    assert np.allclose(weights, [[5.0 / 6.0, 0.0]])


def _write_minimal_kmap(tmp_path: Path) -> Path:
    qe_path = interpolate_path(
        [
            PathNode("G", np.array([0.0, 0.0, 0.0]), 2),
            PathNode("X", np.array([0.5, 0.0, 0.0]), 1),
        ],
        np.diag([2, 1, 1]),
    )
    path = tmp_path / "kmap.tsv"
    write_kmap(path, qe_path)
    return path
