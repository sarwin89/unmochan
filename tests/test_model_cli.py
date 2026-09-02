"""The model-Hamiltonian CLI backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from unfoldlab.cli.main import app
from unfoldlab.io.model_hamiltonian import read_tight_binding_model

runner = CliRunner()

CHAIN = {
    "n_orbitals": 1,
    "hoppings": [
        {"cell": [0, 0, 0], "matrix": [[0.2]]},
        {"cell": [1, 0, 0], "matrix": [[-1.0]]},
    ],
}


def _write(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload))
    return path


def test_model_unfold_cli_reports_the_sum_rules(tmp_path: Path) -> None:
    model = _write(tmp_path, CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(model),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0.05,0,0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["multiplicity"] == 3
    assert len(payload["kpoints"]) == 3
    assert payload["sum_rules"]["max_fiber_deviation"] < 1e-10
    assert payload["sum_rules"]["max_band_deviation"] < 1e-10
    weights = np.array(payload["weights"])
    assert np.allclose(weights.sum(axis=0), 1.0)


def test_model_unfold_cli_writes_an_ebs_json(tmp_path: Path) -> None:
    model = _write(tmp_path, {**CHAIN, "perturbations": [{"cell": 1, "orbital": 0, "value": 0.6}]})
    output = tmp_path / "ebs.json"
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(model),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0.1,0,0",
            "--json",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    stored = json.loads(output.read_text())
    assert stored["schema"] == "unfoldlab.effective_band_structure"
    assert len(stored["kpoints"]) == 2


def test_model_unfold_cli_rejects_a_bad_matrix(tmp_path: Path) -> None:
    model = _write(tmp_path, CHAIN)
    result = runner.invoke(
        app,
        ["model", "unfold", "--model", str(model), "--matrix", "1 2 3", "--kpoint", "0,0,0"],
    )
    assert result.exit_code != 0
    assert "nine numbers" in result.output


@pytest.mark.parametrize(
    ("name", "matrix"),
    [
        ("toy_1d_chain.json", "3 0 0 0 1 0 0 0 1"),
        ("toy_1d_chain_defect.json", "3 0 0 0 1 0 0 0 1"),
        ("toy_2d_square.json", "2 1 0 -1 1 0 0 0 1"),
    ],
)
def test_shipped_examples_unfold_and_conserve_weight(name: str, matrix: str) -> None:
    """The example models in `examples/` must run exactly as documented."""

    model = Path(__file__).resolve().parents[1] / "examples" / name
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(model),
            "--matrix",
            matrix,
            "--kpoint",
            "0.11,0.23,0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["sum_rules"]["max_fiber_deviation"] < 1e-10
    assert payload["sum_rules"]["max_band_deviation"] < 1e-10


def test_reader_hermitizes_and_reads_complex_entries(tmp_path: Path) -> None:
    payload = {
        "n_orbitals": 1,
        "hoppings": [{"cell": [1, 0, 0], "matrix": [[[0.0, 0.5]]]}],
    }
    model, perturbations = read_tight_binding_model(_write(tmp_path, payload))
    assert perturbations == []
    assert set(model.hoppings) == {(1, 0, 0), (-1, 0, 0)}
    assert model.hoppings[(1, 0, 0)] == pytest.approx(np.array([[0.5j]]))
    matrix = model.bloch_hamiltonian([0.21, 0.0, 0.0])
    assert np.allclose(matrix, matrix.conj().T)


def test_reader_rejects_a_malformed_model(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="n_orbitals"):
        read_tight_binding_model(_write(tmp_path, {"hoppings": []}))
    too_small = {"n_orbitals": 2, "hoppings": [{"cell": [0, 0, 0], "matrix": [[1.0]]}]}
    with pytest.raises(ValueError, match="must be 2x2"):
        read_tight_binding_model(_write(tmp_path, too_small))
    with pytest.raises(ValueError, match="JSON object"):
        read_tight_binding_model(_write(tmp_path, [1, 2, 3]))  # type: ignore[arg-type]


HR_CHAIN = """written by the test suite
1
3
1 1 1
0 0 0 1 1 0.2 0.0
1 0 0 1 1 -1.0 0.0
-1 0 0 1 1 -1.0 0.0
"""


def test_model_unfold_cli_reads_a_wannier90_file(tmp_path: Path) -> None:
    path = tmp_path / "chain_hr.dat"
    path.write_text(HR_CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(path),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0.05,0,0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_orbitals"] == 1
    assert payload["sum_rules"]["max_fiber_deviation"] < 1e-10
    energies = np.array(payload["energies"])
    weights = np.array(payload["weights"])
    for index, point in enumerate(payload["kpoints"]):
        expected = 0.2 - 2.0 * np.cos(2.0 * np.pi * point[0])
        assert energies[index][int(np.argmax(weights[index]))] == pytest.approx(expected)


def test_model_unfold_cli_honours_an_explicit_format(tmp_path: Path) -> None:
    """`--format hr` must win over the `.json` extension, and vice versa."""

    disguised = tmp_path / "chain.json"
    disguised.write_text(HR_CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(disguised),
            "--format",
            "hr",
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0,0,0",
        ],
    )
    assert result.exit_code == 0, result.output

    bad = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(disguised),
            "--format",
            "toml",
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0,0,0",
        ],
    )
    assert bad.exit_code != 0
    assert "auto, json, hr, or tb" in bad.output


def test_model_unfold_cli_adds_onsite_perturbations(tmp_path: Path) -> None:
    model = _write(tmp_path, CHAIN)
    common = [
        "model",
        "unfold",
        "--model",
        str(model),
        "--matrix",
        "3 0 0 0 1 0 0 0 1",
        "--kpoint",
        "0.05,0,0",
    ]

    clean = json.loads(runner.invoke(app, common).output)
    result = runner.invoke(app, [*common, "--onsite", "1,0,0.5"])
    assert result.exit_code == 0, result.output
    perturbed = json.loads(result.output)

    # A defect breaks the perfect-crystal delta weights but keeps the sum rules,
    # and it raises the trace of the supercell Hamiltonian by exactly 0.5.
    assert perturbed["sum_rules"]["max_fiber_deviation"] < 1e-10
    assert perturbed["sum_rules"]["max_band_deviation"] < 1e-10
    clean_trace = np.array(clean["energies"])[0].sum()
    assert np.array(perturbed["energies"])[0].sum() == pytest.approx(clean_trace + 0.5)
    assert np.max(np.array(perturbed["weights"])) < 1.0 - 1e-6

    bad = runner.invoke(app, [*common, "--onsite", "1,0"])
    assert bad.exit_code != 0
    assert "cell,orbital,value" in bad.output


def test_model_bands_cli_unfolds_a_primitive_path(tmp_path: Path) -> None:
    model = _write(tmp_path, CHAIN)
    output = tmp_path / "path.json"
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "6",
            "--json",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert output.exists()
    stored = json.loads(output.read_text())
    assert len(stored["kpoints"]) == 6
    energies = np.array(stored["energies"])
    weights = np.array(stored["weights"])
    assert energies.shape == (6, 3)
    # A perfect chain: weight one on the primitive band, zero on the folded copies.
    assert weights.sum(axis=1) == pytest.approx(np.ones(6))
    for index, point in enumerate(stored["kpoints"]):
        expected = 0.2 - 2.0 * np.cos(2.0 * np.pi * point[0])
        assert energies[index][int(np.argmax(weights[index]))] == pytest.approx(expected)
    assert np.array(stored["distances"])[0] == pytest.approx(0.0)


def test_model_bands_cli_joins_segments_without_duplicating_corners(tmp_path: Path) -> None:
    model = Path(__file__).resolve().parents[1] / "examples" / "toy_2d_square.json"
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "2 1 0 -1 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0:0.5,0.5,0",
            "--points",
            "4",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_kpoints"] == 7  # 4 + 4 - 1 shared corner
    assert payload["max_band_deviation"] < 1e-10


def test_model_bands_cli_rejects_a_one_corner_path(tmp_path: Path) -> None:
    model = _write(tmp_path, CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0",
        ],
    )
    assert result.exit_code != 0
    assert "at least two corners" in result.output


TWO_ORBITAL = {
    "n_orbitals": 2,
    "hoppings": [
        {"cell": [0, 0, 0], "matrix": [[-1.5, 0.1], [0.1, 1.5]]},
        {"cell": [1, 0, 0], "matrix": [[-0.4, 0.0], [0.0, 0.3]]},
    ],
}


def test_model_bands_cli_reports_orbital_group_shares(tmp_path: Path) -> None:
    model = _write(tmp_path, TWO_ORBITAL)
    output = tmp_path / "fat.json"
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "4",
            "--group",
            "0",
            "--group",
            "1",
            "--json",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output.split("Wrote")[0])
    assert payload["orbital_groups"] == [[0], [1]]
    assert sum(payload["group_shares"]) == pytest.approx(1.0)

    stored = json.loads(output.read_text())
    projected = np.array(stored["metadata"]["projected_weights"])
    weights = np.array(stored["weights"])
    assert projected.shape == (2, *weights.shape)
    assert np.allclose(projected.sum(axis=0), weights)


def test_model_bands_cli_rejects_an_out_of_range_orbital_group(tmp_path: Path) -> None:
    model = _write(tmp_path, TWO_ORBITAL)
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "2 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "3",
            "--group",
            "5",
        ],
    )

    assert result.exit_code != 0
    assert "outside" in result.output


def test_model_bands_cli_reads_a_wannier90_tb_file(tmp_path: Path) -> None:
    path = tmp_path / "chain_tb.dat"
    path.write_text(
        "\n".join(
            [
                "written by the test suite",
                "2.5 0.0 0.0",
                "0.0 6.0 0.0",
                "0.0 0.0 8.0",
                "1",
                "3",
                "1 1 1",
                "",
                "0 0 0",
                "1 1 0.2 0.0",
                "",
                "1 0 0",
                "1 1 -1.0 0.0",
                "",
                "-1 0 0",
                "1 1 -1.0 0.0",
            ]
        )
        + "\n"
    )
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(path),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["n_orbitals"] == 1
    assert payload["multiplicity"] == 3
    assert payload["max_band_deviation"] < 1e-10


LCAO_CHAIN = {
    "n_orbitals": 1,
    "hoppings": [{"cell": [1, 0, 0], "matrix": [[-1.0]]}],
    "overlaps": [
        {"cell": [0, 0, 0], "matrix": [[1.0]]},
        {"cell": [1, 0, 0], "matrix": [[0.25]]},
    ],
}


def test_overlaps_block_is_optional_and_hermitized(tmp_path: Path) -> None:
    from unfoldlab.io.model_hamiltonian import read_overlap_model

    assert read_overlap_model(_write(tmp_path, CHAIN)) is None
    overlap = read_overlap_model(_write(tmp_path, LCAO_CHAIN))
    assert overlap is not None
    assert overlap.n_orbitals == 1
    assert np.isclose(overlap.hoppings[(-1, 0, 0)][0, 0], 0.25)


def test_overlaps_block_requires_the_on_site_cell(tmp_path: Path) -> None:
    from unfoldlab.io.model_hamiltonian import read_overlap_model

    broken = {**LCAO_CHAIN, "overlaps": [{"cell": [1, 0, 0], "matrix": [[0.25]]}]}
    with pytest.raises(ValueError, match=r"\[0, 0, 0\]"):
        read_overlap_model(_write(tmp_path, broken))


def test_model_unfold_cli_uses_the_non_orthogonal_formula(tmp_path: Path) -> None:
    model = _write(tmp_path, LCAO_CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "unfold",
            "--model",
            str(model),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--kpoint",
            "0.07,0,0",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["basis"] == "non-orthogonal (LCAO)"
    assert payload["sum_rules"]["max_fiber_deviation"] < 1e-10
    # The overlap is real: the reciprocal-space kernel is not the identity.
    assert payload["overlap_neglect"]["max_kernel_deviation"] > 0.1
    weights = np.array(payload["weights"])
    assert np.allclose(weights.sum(axis=0), 1.0)
    # A perfect crystal still folds back exactly, one k-point per band.
    assert np.allclose(np.sort(weights, axis=0)[-1], 1.0, atol=1e-10)
    # The generalized eigenvalues follow the non-orthogonal dispersion.
    energies = np.array(payload["energies"])
    kpoints = np.array(payload["kpoints"])
    carrier = np.argmax(weights, axis=0)
    cosine = np.cos(2.0 * np.pi * kpoints[carrier, 0])
    exact = -2.0 * cosine / (1.0 + 0.5 * cosine)
    assert np.allclose(energies[carrier, np.arange(energies.shape[1])], exact, atol=1e-9)


def test_model_bands_cli_reports_the_basis(tmp_path: Path) -> None:
    model = _write(tmp_path, LCAO_CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--points",
            "5",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["basis"] == "non-orthogonal (LCAO)"
    assert summary["max_band_deviation"] < 1e-10


def test_model_bands_cli_refuses_fat_bands_with_an_overlap(tmp_path: Path) -> None:
    model = _write(tmp_path, LCAO_CHAIN)
    result = runner.invoke(
        app,
        [
            "model",
            "bands",
            "--model",
            str(model),
            "--matrix",
            "3 0 0 0 1 0 0 0 1",
            "--path",
            "0,0,0:0.5,0,0",
            "--group",
            "0",
        ],
    )

    assert result.exit_code != 0
    assert "non-orthogonal" in result.output
