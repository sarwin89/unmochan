"""Synthetic VASP PROCAR writer shared by the tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def write_synthetic_procar(
    path: Path,
    *,
    kpoints: np.ndarray,
    kpoint_weights: np.ndarray,
    energies: np.ndarray,
    occupations: np.ndarray,
    projections: np.ndarray,
    orbital_labels: tuple[str, ...],
    magnetization: np.ndarray | None = None,
    phase: bool = False,
) -> Path:
    """Write a PROCAR with ``projections`` of shape ``(n_spin, n_k, n_b, n_ion, n_orb)``.

    ``magnetization`` of shape ``(3, n_k, n_b, n_ion, n_orb)`` produces the
    noncollinear layout (four tables per band).  ``phase=True`` adds the extra
    ``LORBIT = 12`` phase table, whose rows carry real and imaginary parts.
    """

    projections = np.asarray(projections, dtype=float)
    n_spin, n_kpoints, n_bands, n_ions, n_orbitals = projections.shape
    lines: list[str] = ["PROCAR lm decomposed" + (" + phase" if phase else "")]
    header = (
        f"# of k-points:  {n_kpoints}         # of bands:  {n_bands}         # of ions:  {n_ions}"
    )
    for ispin in range(n_spin):
        lines.append(header)
        lines.append("")
        for ik in range(n_kpoints):
            coordinates = " ".join(f"{value: .8f}" for value in kpoints[ik])
            lines.append(
                f" k-point {ik + 1:5d} : {coordinates}     weight = {kpoint_weights[ik]:.8f}"
            )
            lines.append("")
            for ib in range(n_bands):
                lines.append(
                    f"band {ib + 1:5d} # energy {energies[ispin, ik, ib]:14.8f} "
                    f"# occ. {occupations[ispin, ik, ib]:12.8f}"
                )
                lines.append("")
                tables = [projections[ispin, ik, ib]]
                if magnetization is not None:
                    tables.extend(magnetization[component, ik, ib] for component in range(3))
                for table in tables:
                    lines.extend(_table_lines(table, orbital_labels))
                if phase:
                    lines.extend(_phase_lines(projections[ispin, ik, ib], orbital_labels))
                lines.append("")
    path.write_text("\n".join(lines) + "\n")
    return path


def _table_lines(table: np.ndarray, orbital_labels: tuple[str, ...]) -> list[str]:
    lines = ["ion      " + "  ".join(orbital_labels) + "  tot"]
    for index, row in enumerate(table):
        values = "  ".join(f"{value:.3f}" for value in row)
        lines.append(f"{index + 1:4d}  {values}  {row.sum():.3f}")
    totals = table.sum(axis=0)
    lines.append("tot   " + "  ".join(f"{value:.3f}" for value in totals) + f"  {table.sum():.3f}")
    return lines


def _phase_lines(table: np.ndarray, orbital_labels: tuple[str, ...]) -> list[str]:
    lines = ["ion      " + "  ".join(orbital_labels) + "  tot"]
    for index, row in enumerate(table):
        parts = []
        for value in row:
            parts.append(f"{np.sqrt(value):.3f}")
            parts.append("0.000")
        lines.append(f"{index + 1:4d}  " + "  ".join(parts))
    return lines
