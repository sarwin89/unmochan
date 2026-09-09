"""VASP file readers used by the initial workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unmochan.core.structures import Structure
from unmochan.io.qe import QEPath


@dataclass(frozen=True)
class EigenvalData:
    """Parsed VASP EIGENVAL data.

    ``energies`` has shape ``(n_kpoints, n_bands)`` for non-spin-polarized files
    and ``(n_kpoints, n_bands, n_channels)`` when multiple energy channels are
    present.
    """

    kpoints: NDArray[np.float64]
    kpoint_weights: NDArray[np.float64]
    energies: NDArray[np.float64]
    occupations: NDArray[np.float64]
    nelect: int | None = None


def read_poscar(path: str | Path) -> Structure:
    """Read a VASP POSCAR/CONTCAR into a :class:`Structure`."""

    path = Path(path)
    raw_lines = path.read_text().splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    if len(lines) < 8:
        raise ValueError(f"{path} is too short to be a POSCAR")

    name = lines[0]
    scale_tokens = [float(token) for token in lines[1].split()]
    if len(scale_tokens) != 1:
        raise ValueError("only scalar POSCAR scale factors are currently supported")
    scale = scale_tokens[0]
    lattice = np.array([[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)], dtype=float)
    lattice *= scale

    cursor = 5
    first = lines[cursor].split()
    if _all_ints(first):
        species_names = [f"X{i + 1}" for i in range(len(first))]
        counts = [int(x) for x in first]
        cursor += 1
    else:
        species_names = first
        cursor += 1
        counts = [int(x) for x in lines[cursor].split()]
        cursor += 1

    if len(species_names) != len(counts):
        raise ValueError("number of POSCAR species labels does not match counts")

    if lines[cursor].lower().startswith("s"):
        cursor += 1
    coord_mode = lines[cursor].lower()
    cursor += 1

    n_sites = sum(counts)
    coord_rows = []
    for line in lines[cursor : cursor + n_sites]:
        coord_rows.append([float(x) for x in line.split()[:3]])
    coords = np.array(coord_rows, dtype=float)

    if coord_mode.startswith(("d", "f")):
        frac_coords = coords
    elif coord_mode.startswith(("c", "k")):
        frac_coords = coords @ np.linalg.inv(lattice)
    else:
        raise ValueError(f"unknown POSCAR coordinate mode: {coord_mode}")

    species: list[str] = []
    for symbol, count in zip(species_names, counts, strict=True):
        species.extend([symbol] * count)
    return Structure(lattice=lattice, species=tuple(species), frac_coords=frac_coords, name=name)


def read_eigenval(path: str | Path) -> EigenvalData:
    """Read the common VASP EIGENVAL layout.

    The parser intentionally preserves multiple spin channels when present
    instead of collapsing them.
    """

    path = Path(path)
    lines = path.read_text().splitlines()
    if len(lines) < 7:
        raise ValueError(f"{path} is too short to be an EIGENVAL")

    header_tokens = lines[5].split()
    if len(header_tokens) < 3:
        raise ValueError("EIGENVAL line 6 must contain electron, k-point, and band counts")
    nelect = int(float(header_tokens[-3]))
    n_kpoints = int(float(header_tokens[-2]))
    n_bands = int(float(header_tokens[-1]))

    cursor = 6
    kpoints = np.zeros((n_kpoints, 3), dtype=float)
    kpoint_weights = np.zeros(n_kpoints, dtype=float)
    energy_rows: list[list[list[float]]] = []
    occupation_rows: list[list[list[float]]] = []

    for ik in range(n_kpoints):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            raise ValueError(f"missing k-point block {ik + 1}")

        krow = [float(x) for x in lines[cursor].split()[:4]]
        if len(krow) != 4:
            raise ValueError(f"invalid k-point row at line {cursor + 1}")
        kpoints[ik] = krow[:3]
        kpoint_weights[ik] = krow[3]
        cursor += 1

        band_energies: list[list[float]] = []
        band_occupations: list[list[float]] = []
        for _ in range(n_bands):
            tokens = [float(x) for x in lines[cursor].split()]
            if len(tokens) < 3:
                raise ValueError(f"invalid band row at line {cursor + 1}")
            values = tokens[1:]
            midpoint = len(values) // 2
            if len(values) % 2 == 0 and len(values) > 2:
                band_energies.append(values[:midpoint])
                band_occupations.append(values[midpoint:])
            else:
                band_energies.append([values[0]])
                band_occupations.append([values[-1]])
            cursor += 1
        energy_rows.append(band_energies)
        occupation_rows.append(band_occupations)

    energies = _squeeze_last_channel(np.array(energy_rows, dtype=float))
    occupations = _squeeze_last_channel(np.array(occupation_rows, dtype=float))
    return EigenvalData(
        kpoints=kpoints,
        kpoint_weights=kpoint_weights,
        energies=energies,
        occupations=occupations,
        nelect=nelect,
    )


def write_vasp_kpoints(path: str | Path, folded_kpoints: NDArray[np.float64]) -> None:
    """Write an explicit VASP reciprocal-space KPOINTS file."""

    rows = ["UnfoldLab folded supercell path", str(len(folded_kpoints)), "Reciprocal"]
    for kpoint in np.asarray(folded_kpoints, dtype=float):
        rows.append(f"{kpoint[0]: .12f} {kpoint[1]: .12f} {kpoint[2]: .12f} 1.0")
    Path(path).write_text("\n".join(rows) + "\n")


def write_vasp_path_files(
    path_json: str | Path,
    *,
    kpoints: str | Path,
    kmap: str | Path,
    ticks: str | Path,
) -> QEPath:
    """Generate VASP KPOINTS plus the shared kmap/tick files from a path JSON."""

    from unmochan.io.qe import build_qe_path, write_kmap, write_ticks

    path_data = build_qe_path(path_json)
    write_vasp_kpoints(kpoints, path_data.supercell_folded)
    write_kmap(kmap, path_data)
    write_ticks(ticks, path_data)
    return path_data


def _all_ints(tokens: list[str]) -> bool:
    try:
        for token in tokens:
            int(token)
    except ValueError:
        return False
    return True


def _squeeze_last_channel(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.shape[-1] == 1:
        return values[..., 0]
    return values
