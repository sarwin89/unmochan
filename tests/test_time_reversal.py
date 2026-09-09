"""Time reversal: the ``k -> -k`` half of a reduced wavefunction set."""

from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from unmochan.cli.main import app
from unmochan.core.plane_waves import weights_from_coefficients
from unmochan.core.spacegroup import detect_primitive_operations
from unmochan.core.structures import Structure
from unmochan.core.symmetry import map_kpoints_to_stored, supercell_operation

runner = CliRunner()

INVERSION = -np.eye(3, dtype=np.int64)

# A cell with no inversion centre: the two species sit at 0 and 0.3, and
# inversion would need a B at -0.3.
NONCENTROSYMMETRIC = Structure(
    lattice=np.diag([2.0, 1.0, 1.0]),
    species=("A", "B"),
    frac_coords=np.array([[0.0, 0.0, 0.0], [0.3, 0.0, 0.0]]),
)
TRANSFORM = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])


def _contains(operations: np.ndarray, target: np.ndarray) -> bool:
    return any(np.array_equal(op, target) for op in operations)


class TestDetection:
    def test_a_noncentrosymmetric_cell_has_no_inversion(self) -> None:
        operations = detect_primitive_operations(NONCENTROSYMMETRIC, TRANSFORM)
        assert _contains(operations, np.eye(3, dtype=np.int64))
        assert not _contains(operations, INVERSION)

    def test_time_reversal_adjoins_k_to_minus_k(self) -> None:
        plain = detect_primitive_operations(NONCENTROSYMMETRIC, TRANSFORM)
        reversed_set = detect_primitive_operations(
            NONCENTROSYMMETRIC, TRANSFORM, time_reversal=True
        )
        assert _contains(reversed_set, INVERSION)
        assert len(reversed_set) == 2 * len(plain)
        # Still a usable operation set: every element is integral in supercell
        # coordinates, and the set is closed under negation.
        for op in reversed_set:
            supercell_operation(op, TRANSFORM)
            assert _contains(reversed_set, -op)

    def test_a_centrosymmetric_cell_gains_nothing(self) -> None:
        centrosymmetric = Structure(
            lattice=np.diag([2.0, 1.0, 1.0]),
            species=("A", "A"),
            frac_coords=np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]),
        )
        plain = detect_primitive_operations(centrosymmetric, TRANSFORM)
        with_tr = detect_primitive_operations(centrosymmetric, TRANSFORM, time_reversal=True)
        assert _contains(plain, INVERSION)
        assert len(with_tr) == len(plain)


class TestKPointMapping:
    def test_the_missing_half_of_the_mesh_is_reached_only_with_time_reversal(self) -> None:
        stored = np.array([[0.25, 0.0, 0.0]])  # the file at K_f
        # This primitive k-point folds to -K_f, which is not K_f modulo 1.
        requested = np.array([[-0.125, 0.0, 0.0]])
        folded = requested @ TRANSFORM.T
        assert np.allclose(folded, -stored)

        plain = detect_primitive_operations(NONCENTROSYMMETRIC, TRANSFORM)
        with pytest.raises(ValueError, match="does not cover"):
            map_kpoints_to_stored(requested, stored, TRANSFORM, plain)

        with_tr = detect_primitive_operations(NONCENTROSYMMETRIC, TRANSFORM, time_reversal=True)
        matches = map_kpoints_to_stored(requested, stored, TRANSFORM, with_tr)
        assert len(matches) == 1
        assert matches[0].index == 0
        # The recipe is "unfold -k against the stored file".
        assert np.allclose(
            matches[0].effective_primitive_kpoint @ matches[0].operation.T, requested[0]
        )


class TestWeights:
    """The numerical counterpart of ``UnfoldLab.weight_timeReversal_stored``."""

    @staticmethod
    def _weights(g_vectors, coefficients, primitive, folded):
        return weights_from_coefficients(
            g_vectors,
            coefficients,
            primitive_kpoint=np.asarray(primitive, dtype=float),
            folded_supercell_kpoint=np.asarray(folded, dtype=float),
            transform=TRANSFORM.astype(float),
        )

    def test_the_time_reversed_state_gives_the_weight_of_the_negated_kpoint(self) -> None:
        rng = np.random.default_rng(4242)
        g_vectors = np.array([[m, 0, 0] for m in range(-3, 4)], dtype=int)
        coefficients = rng.normal(size=(2, 1, len(g_vectors))) + 1j * rng.normal(
            size=(2, 1, len(g_vectors))
        )
        stored_k = np.array([0.25, 0.0, 0.0])
        primitive = np.array([-0.125, 0.0, 0.0])  # folds to -K_f

        # The state stored in the file, unfolded at -k.
        from_stored = self._weights(g_vectors, coefficients, -primitive, stored_k)
        # Its time-reversal partner: k-point, plane waves and phases all flipped.
        from_partner = self._weights(-g_vectors, np.conjugate(coefficients), primitive, -stored_k)
        assert from_partner == pytest.approx(from_stored)

    def test_the_two_fiber_members_carry_different_weights(self) -> None:
        """Otherwise the previous test would compare two equal numbers."""

        rng = np.random.default_rng(11)
        g_vectors = np.array([[m, 0, 0] for m in range(-3, 4)], dtype=int)
        coefficients = rng.normal(size=(1, 1, len(g_vectors))) + 1j * rng.normal(
            size=(1, 1, len(g_vectors))
        )
        stored_k = np.array([0.25, 0.0, 0.0])
        assert not np.allclose(
            self._weights(g_vectors, coefficients, np.array([0.125, 0.0, 0.0]), stored_k),
            self._weights(g_vectors, coefficients, np.array([0.625, 0.0, 0.0]), stored_k),
        )

    def test_the_fiber_sum_rule_holds_for_the_time_reversed_state(self) -> None:
        rng = np.random.default_rng(2024)
        g_vectors = np.array([[m, 0, 0] for m in range(-3, 4)], dtype=int)
        coefficients = rng.normal(size=(2, 1, len(g_vectors))) + 1j * rng.normal(
            size=(2, 1, len(g_vectors))
        )
        stored_k = np.array([0.25, 0.0, 0.0])
        fiber = [np.array([-0.125, 0.0, 0.0]), np.array([0.375, 0.0, 0.0])]
        total = sum(
            self._weights(-g_vectors, np.conjugate(coefficients), k, -stored_k) for k in fiber
        )
        assert total == pytest.approx(np.ones(2))


class TestCLI:
    @staticmethod
    def _poscar(tmp_path):
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "\n".join(
                [
                    "noncentrosymmetric supercell",
                    "1.0",
                    "2.0 0.0 0.0",
                    "0.0 1.0 0.0",
                    "0.0 0.0 1.0",
                    "A B",
                    "1 1",
                    "Direct",
                    "0.0 0.0 0.0",
                    "0.3 0.0 0.0",
                    "",
                ]
            )
        )
        return poscar

    def test_cli_writes_the_time_reversed_operation_set(self, tmp_path) -> None:
        poscar = self._poscar(tmp_path)
        plain_out = tmp_path / "plain.txt"
        tr_out = tmp_path / "tr.txt"
        base = ["detect-symmetry", "--structure", str(poscar), "--matrix", "2 0 0 0 1 0 0 0 1"]

        plain = runner.invoke(app, [*base, "--out", str(plain_out)])
        assert plain.exit_code == 0, plain.output
        with_tr = runner.invoke(app, [*base, "--time-reversal", "--out", str(tr_out)])
        assert with_tr.exit_code == 0, with_tr.output
        assert "Time reversal" in with_tr.output

        plain_rows = [line for line in plain_out.read_text().splitlines() if line.strip()]
        tr_rows = [line for line in tr_out.read_text().splitlines() if line.strip()]
        assert len(tr_rows) == 2 * len(plain_rows)
        assert "-1 0 0 0 -1 0 0 0 -1" in tr_rows

    def test_cli_rejects_time_reversal_without_a_matrix(self, tmp_path) -> None:
        poscar = self._poscar(tmp_path)
        result = runner.invoke(
            app, ["detect-symmetry", "--structure", str(poscar), "--time-reversal"]
        )
        assert result.exit_code != 0


def test_qe_workflow_serves_the_time_reversal_partner_of_a_stored_kpoint(tmp_path):
    """End to end: a k-map row whose fold is -K_f, and a save directory at +K_f."""

    from test_qe_wfc_kpoints import write_synthetic_wfc_dat

    from unmochan.workflows.backend import compute_backend_weights

    header = (
        "ik\ts_pc\tkpc_1\tkpc_2\tkpc_3\t"
        "Ksc_unfold_1\tKsc_unfold_2\tKsc_unfold_3\t"
        "Ksc_fold_1\tKsc_fold_2\tKsc_fold_3\tlabel"
    )
    row = "\t".join(
        ["1", "0.0", "-0.125", "0.0", "0.0"]
        + ["-0.25", "0.0", "0.0"]
        + ["-0.25", "0.0", "0.0"]
        + ["X"]
    )
    kmap = tmp_path / "kmap.tsv"
    kmap.write_text(f"{header}\n{row}\n")

    save_dir = tmp_path / "pwscf.save"
    save_dir.mkdir()
    miller = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    coefficients = np.array([[1.0, 1.0, 2.0], [0.0, 1.0, 0.0]], dtype=complex)
    write_synthetic_wfc_dat(save_dir / "wfc1.dat", (0.25, 0.0, 0.0), miller, coefficients)

    kwargs = dict(
        kmap=kmap,
        transform=np.diag([2.0, 1.0, 1.0]),
        out=tmp_path / "weights.dat",
        nbnd=2,
        qe_save_dir=save_dir,
        qe_lattice_alat=np.eye(3),
    )

    # The stored k-point is +K_f, so without time reversal nothing matches it.
    with pytest.raises(ValueError):
        compute_backend_weights("qe", operations=np.zeros((0, 3, 3), dtype=int), **kwargs)

    diagnostics = compute_backend_weights("qe", operations=INVERSION.reshape(1, 3, 3), **kwargs)
    assert diagnostics.n_kpoints == 1
    assert diagnostics.n_bands == 2
    assert 0.0 <= diagnostics.min_weight <= diagnostics.max_weight <= 1.0
