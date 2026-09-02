import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.io.serialization import (
    EBS_SCHEMA_VERSION,
    build_run_manifest,
    read_ebs,
    read_ebs_hdf5,
    read_ebs_json,
    read_run_manifest,
    write_ebs,
    write_ebs_hdf5,
    write_ebs_json,
    write_run_manifest,
)

runner = CliRunner()


def _ebs() -> EffectiveBandStructure:
    return EffectiveBandStructure(
        kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        energies=np.array([[-1.0, 1.0], [-0.5, 1.5]]),
        weights=np.array([[1.0, 0.25], [0.5, 0.75]]),
        reference_energy=0.5,
        metadata={"mode": "test"},
    )


def test_ebs_json_round_trip(tmp_path: Path):
    original = _ebs()
    path = write_ebs_json(tmp_path / "ebs.json", original)

    restored, manifest = read_ebs_json(path)

    assert manifest is None
    np.testing.assert_allclose(restored.kpoints, original.kpoints)
    np.testing.assert_allclose(restored.energies, original.energies)
    np.testing.assert_allclose(restored.weights, original.weights)
    np.testing.assert_allclose(restored.distances, original.distances)
    assert restored.reference_energy == pytest.approx(original.reference_energy)
    assert restored.metadata == original.metadata


def test_ebs_json_carries_the_manifest(tmp_path: Path):
    source = tmp_path / "WAVECAR"
    source.write_bytes(b"not really a wavecar")
    manifest = build_run_manifest(
        code="vasp",
        weight_mode="plane-wave projection",
        transformation=[[2, 0, 0], [0, 1, 0], [0, 0, 1]],
        reference_energy=1.25,
        spin_channel=1,
        sources={"wavecar": source, "missing": None},
    )

    path = write_ebs_json(tmp_path / "ebs.json", _ebs(), manifest=manifest)
    _, restored = read_ebs_json(path)

    assert restored is not None
    assert restored.code == "vasp"
    assert restored.weight_mode == "plane-wave projection"
    assert restored.transformation == [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert restored.reference_energy == pytest.approx(1.25)
    assert [item.role for item in restored.sources] == ["wavecar"]
    assert restored.sources[0].size == source.stat().st_size
    assert len(restored.sources[0].sha256) == 64


def test_manifest_records_a_stable_digest(tmp_path: Path):
    source = tmp_path / "kmap.tsv"
    source.write_text("1 0 0\n")
    first = build_run_manifest(code="qe", weight_mode="read", sources={"kmap": source})
    second = build_run_manifest(code="qe", weight_mode="read", sources={"kmap": source})

    assert first.sources[0].sha256 == second.sources[0].sha256

    source.write_text("1 0 1\n")
    third = build_run_manifest(code="qe", weight_mode="read", sources={"kmap": source})

    assert third.sources[0].sha256 != first.sources[0].sha256


def test_manifest_can_skip_hashing(tmp_path: Path):
    source = tmp_path / "kmap.tsv"
    source.write_text("1 0 0\n")

    manifest = build_run_manifest(
        code="qe", weight_mode="read", sources={"kmap": source}, digest=False
    )

    assert manifest.sources[0].sha256 is None
    assert manifest.sources[0].size == source.stat().st_size


def test_manifest_round_trip(tmp_path: Path):
    manifest = build_run_manifest(
        code="vasp",
        weight_mode="plane-wave projection",
        projection="species:A",
        extras={"note": "hello"},
    )
    path = write_run_manifest(tmp_path / "manifest.json", manifest)

    assert read_run_manifest(path) == manifest


def test_read_ebs_json_rejects_a_foreign_document(tmp_path: Path):
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"schema": "something.else"}))

    with pytest.raises(ValueError, match="not an unfoldlab effective band structure"):
        read_ebs_json(path)


def test_read_ebs_json_rejects_a_newer_schema(tmp_path: Path):
    path = write_ebs_json(tmp_path / "ebs.json", _ebs())
    payload = json.loads(path.read_text())
    payload["schema_version"] = EBS_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="schema version"):
        read_ebs_json(path)


def test_unfold_cli_writes_json_and_manifest(tmp_path: Path):
    kmap = _write_kmap(tmp_path)
    bands = tmp_path / "bands.dat.gnu"
    bands.write_text("0 -1\n1 -0.5\n\n0 1\n1 1.5\n")
    weights = tmp_path / "weights.dat"
    weights.write_text("1 1 1.0\n1 2 0.5\n2 1 0.25\n2 2 0.75\n")

    out = tmp_path / "unfolded.dat"
    plot = tmp_path / "plot.svg"
    json_out = tmp_path / "ebs.json"
    manifest_out = tmp_path / "manifest.json"

    result = runner.invoke(
        app,
        [
            "unfold",
            "--code",
            "qe",
            "--bands",
            str(bands),
            "--kmap",
            str(kmap),
            "--weights",
            str(weights),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--out",
            str(out),
            "--plot",
            str(plot),
            "--json",
            str(json_out),
            "--manifest",
            str(manifest_out),
        ],
    )

    assert result.exit_code == 0, result.output
    assert json_out.exists()
    assert manifest_out.exists()

    ebs, manifest = read_ebs_json(json_out)
    assert ebs.energies.shape == (2, 2)
    assert manifest is not None
    assert manifest.code == "qe"
    assert manifest.transformation == [[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

    roles = {item.role for item in manifest.sources}
    assert {"bands", "kmap", "weights"} <= roles

    stored = read_run_manifest(manifest_out)
    assert stored.code == manifest.code
    assert stored.sources == manifest.sources


def _write_kmap(tmp_path: Path) -> Path:
    path_json = tmp_path / "path.json"
    qek = tmp_path / "qe_kpoints.in"
    kmap = tmp_path / "kmap.tsv"
    ticks = tmp_path / "ticks.tsv"
    path_json.write_text(
        '{"transformation_matrix": [[2,0,0],[0,1,0],[0,0,1]], '
        '"path": [{"label":"G","k":[0,0,0],"n":2},{"label":"X","k":[0.5,0,0]}]}'
    )
    result = runner.invoke(
        app,
        [
            "qe",
            "make-kpoints",
            str(path_json),
            "--qe-kpoints",
            str(qek),
            "--kmap",
            str(kmap),
            "--ticks",
            str(ticks),
        ],
    )
    assert result.exit_code == 0, result.output
    return kmap


def test_hdf5_round_trip_preserves_everything(tmp_path: Path):
    """HDF5 keeps the arrays bit for bit, unlike a text format."""

    h5py = pytest.importorskip("h5py")
    rng = np.random.default_rng(7)
    original = EffectiveBandStructure(
        kpoints=rng.normal(size=(5, 3)),
        energies=rng.normal(size=(5, 4)),
        weights=rng.random(size=(5, 4)),
        distances=np.cumsum(rng.random(size=5)),
        reference_energy=-1.2345678901234567,
        metadata={"source_code": "vasp", "transformation": [[2, 0, 0], [0, 1, 0], [0, 0, 1]]},
    )
    manifest = build_run_manifest(code="vasp", weight_mode="plane-wave", digest=False)

    path = write_ebs_hdf5(tmp_path / "ebs.h5", original, manifest=manifest)
    restored, restored_manifest = read_ebs_hdf5(path)

    assert np.array_equal(restored.kpoints, original.kpoints)
    assert np.array_equal(restored.energies, original.energies)
    assert np.array_equal(restored.weights, original.weights)
    assert np.array_equal(restored.distances, original.distances)
    assert restored.reference_energy == original.reference_energy
    assert restored.metadata == original.metadata
    assert restored_manifest is not None
    assert restored_manifest.code == "vasp"

    with h5py.File(path, "r") as handle:
        assert set(handle.keys()) == {"distances", "energies", "kpoints", "weights"}
        assert handle["energies"].dtype == np.float64


def test_read_ebs_dispatches_on_the_suffix(tmp_path: Path):
    pytest.importorskip("h5py")
    ebs = EffectiveBandStructure(
        kpoints=np.zeros((2, 3)), energies=np.array([[0.0, 1.0], [0.5, 1.5]])
    )
    json_path = write_ebs(tmp_path / "ebs.json", ebs)
    hdf5_path = write_ebs(tmp_path / "ebs.hdf5", ebs)

    for path in (json_path, hdf5_path):
        restored, manifest = read_ebs(path)
        assert manifest is None
        assert np.allclose(restored.energies, ebs.energies)


def test_read_ebs_hdf5_rejects_a_foreign_file(tmp_path: Path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "other.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("energies", data=np.zeros(3))
    with pytest.raises(ValueError, match="not an unfoldlab effective band structure"):
        read_ebs_hdf5(path)
