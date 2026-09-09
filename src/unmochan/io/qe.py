"""Quantum ESPRESSO text I/O and effective-band helpers.

The band scatter plot used to live here, which was the wrong home for it: it is
backend-neutral and the VASP workflow imported it out of the QE reader.  It is
now :mod:`unmochan.io.plot_bands`, re-exported below so that the old import
path keeps working.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unmochan.core.kpoints import interpolate_segment
from unmochan.core.numerics import TransformLike, as_matrix3, wrap_fractional
from unmochan.core.plane_waves import compute_weights_from_coefficient_table
from unmochan.core.spectral import EffectiveBandStructure
from unmochan.core.structures import Structure
from unmochan.core.unfolding import BandUnfoldingData
from unmochan.io.plot_bands import color_from_weight as color_from_weight
from unmochan.io.plot_bands import plot_unfolded as plot_unfolded
from unmochan.io.plot_bands import plot_unfolded_svg as plot_unfolded_svg

BOHR_TO_ANGSTROM = 0.529177210903


@dataclass(frozen=True)
class PathNode:
    label: str
    kpoint: NDArray[np.float64]
    n_to_next: int


@dataclass(frozen=True)
class QEPath:
    transform: NDArray[np.int64]
    kpoints: NDArray[np.float64]
    distances: NDArray[np.float64]
    labels: tuple[str, ...]

    @property
    def supercell_unfolded(self) -> NDArray[np.float64]:
        return self.kpoints @ self.transform.T

    @property
    def supercell_folded(self) -> NDArray[np.float64]:
        return wrap_fractional(self.supercell_unfolded)


@dataclass(frozen=True)
class QEKMap:
    distances: NDArray[np.float64]
    primitive_kpoints: NDArray[np.float64]
    supercell_unfolded_kpoints: NDArray[np.float64]
    supercell_folded_kpoints: NDArray[np.float64]
    labels: tuple[str, ...]

    @property
    def n_kpoints(self) -> int:
        return int(self.primitive_kpoints.shape[0])


def read_path_json(
    path: str | Path,
) -> tuple[NDArray[np.int64], list[PathNode], NDArray[np.float64] | None]:
    """Read a path definition.

    An optional ``"primitive_lattice"`` entry (3x3 real-space lattice rows, in
    angstrom) makes the path coordinate a true reciprocal-space distance; see
    :func:`interpolate_path`.
    """

    data = json.loads(Path(path).read_text())
    matrix = np.asarray(data["transformation_matrix"], dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("transformation_matrix must be a 3x3 matrix")
    rounded = np.rint(matrix).astype(int)
    if not np.allclose(matrix, rounded):
        raise ValueError("transformation_matrix must contain integer entries")

    raw_points = data["path"]
    if len(raw_points) < 2:
        raise ValueError("path must contain at least two points")
    nodes: list[PathNode] = []
    for idx, item in enumerate(raw_points):
        n_default = 1 if idx == len(raw_points) - 1 else 40
        kpoint = np.asarray(item["k"], dtype=float)
        if kpoint.shape != (3,):
            raise ValueError("each path k entry must contain three coordinates")
        nodes.append(
            PathNode(
                label=str(item.get("label", "")),
                kpoint=kpoint,
                n_to_next=int(item.get("n", n_default)),
            )
        )

    primitive_lattice = data.get("primitive_lattice")
    lattice_arr: NDArray[np.float64] | None = None
    if primitive_lattice is not None:
        lattice_arr = as_matrix3(primitive_lattice, name="primitive_lattice")

    return rounded, nodes, lattice_arr


def read_pw_input_structure(path: str | Path) -> Structure:
    """Read a QE ``pw.x`` input with ``ibrav=0`` style cell/positions blocks."""

    path = Path(path)
    raw_lines = path.read_text().splitlines()
    lines = [_strip_qe_comment(line).strip() for line in raw_lines]
    lines = [line for line in lines if line]
    cell_index = _find_qe_card(lines, "CELL_PARAMETERS")
    positions_index = _find_qe_card(lines, "ATOMIC_POSITIONS")
    if cell_index is None or positions_index is None:
        raise ValueError(f"{path} must contain CELL_PARAMETERS and ATOMIC_POSITIONS cards")

    cell_unit = _qe_card_unit(lines[cell_index], default="angstrom")
    lattice_rows = [
        [float(value) for value in lines[cell_index + offset].split()[:3]] for offset in range(1, 4)
    ]
    lattice = np.array(lattice_rows, dtype=float)
    lattice *= _qe_length_scale(cell_unit, lines)

    position_unit = _qe_card_unit(lines[positions_index], default="crystal")
    species: list[str] = []
    coords: list[list[float]] = []
    cursor = positions_index + 1
    while cursor < len(lines) and not _looks_like_qe_card_or_namelist(lines[cursor]):
        parts = lines[cursor].split()
        if len(parts) >= 4:
            species.append(parts[0])
            coords.append([float(value) for value in parts[1:4]])
        cursor += 1
    if not species:
        raise ValueError(f"{path} has no ATOMIC_POSITIONS rows")
    coord_array = np.asarray(coords, dtype=float)
    if position_unit in {"crystal", "crystal_sg"}:
        frac_coords = coord_array
    elif position_unit == "angstrom":
        frac_coords = coord_array @ np.linalg.inv(lattice)
    elif position_unit == "bohr":
        frac_coords = coord_array * BOHR_TO_ANGSTROM @ np.linalg.inv(lattice)
    elif position_unit == "alat":
        frac_coords = coord_array * _qe_alat_angstrom(lines) @ np.linalg.inv(lattice)
    else:
        raise ValueError(f"unsupported QE ATOMIC_POSITIONS unit: {position_unit}")
    return Structure(
        lattice=lattice,
        species=tuple(species),
        frac_coords=frac_coords,
        name=path.stem,
    )


def interpolate_path(
    nodes: list[PathNode],
    transform: NDArray[np.integer],
    *,
    primitive_lattice: NDArray[np.float64] | None = None,
) -> QEPath:
    """Interpolate a primitive-cell k-path and accumulate its path coordinate.

    When ``primitive_lattice`` is given, consecutive k-points are separated by
    the Cartesian reciprocal-space distance ``|Δk_frac @ B|`` with
    ``B = 2π inv(A).T``, which is the physically meaningful abscissa of a band
    plot.  Without it the fractional-coordinate norm is used, which distorts the
    relative length of path segments for every cell whose reciprocal lattice is
    not cubic — pass the lattice whenever band velocities are to be read off the
    plot.
    """

    if len(nodes) < 2:
        raise ValueError("at least two path nodes are required")

    metric: NDArray[np.float64] | None = None
    if primitive_lattice is not None:
        metric = np.asarray(
            2.0 * np.pi * np.linalg.inv(as_matrix3(primitive_lattice, name="lattice")).T,
            dtype=np.float64,
        )

    kpoints: list[NDArray[np.float64]] = []
    labels: list[str] = []
    distances: list[float] = []
    cumulative = 0.0

    for segment_index, (start, end) in enumerate(zip(nodes[:-1], nodes[1:], strict=True)):
        segment = interpolate_segment(
            start.kpoint,
            end.kpoint,
            n_points=max(2, start.n_to_next),
            start_label=start.label,
            end_label=end.label,
        )
        if segment_index > 0:
            segment = segment[1:]
        for point in segment:
            if kpoints:
                delta = point.fractional - kpoints[-1]
                if metric is not None:
                    delta = delta @ metric
                cumulative += float(np.linalg.norm(delta))
            kpoints.append(point.fractional)
            labels.append(point.label or "")
            distances.append(cumulative)

    return QEPath(
        transform=np.asarray(transform, dtype=int),
        kpoints=np.asarray(kpoints, dtype=float),
        distances=np.asarray(distances, dtype=float),
        labels=tuple(labels),
    )


def build_qe_path(
    path_json: str | Path,
    *,
    primitive_lattice: NDArray[np.float64] | None = None,
) -> QEPath:
    matrix, nodes, json_lattice = read_path_json(path_json)
    return interpolate_path(
        nodes,
        matrix,
        primitive_lattice=primitive_lattice if primitive_lattice is not None else json_lattice,
    )


def _strip_qe_comment(line: str) -> str:
    return line.split("!", 1)[0]


def _find_qe_card(lines: list[str], card: str) -> int | None:
    card_upper = card.upper()
    for index, line in enumerate(lines):
        if line.upper().startswith(card_upper):
            return index
    return None


def _qe_card_unit(line: str, *, default: str) -> str:
    parts = line.replace("{", " ").replace("}", " ").replace("(", " ").replace(")", " ").split()
    if len(parts) == 1:
        return default
    return parts[1].strip().lower()


def _qe_length_scale(unit: str, lines: list[str]) -> float:
    unit_lower = unit.lower()
    if unit_lower in {"angstrom", "ang"}:
        return 1.0
    if unit_lower == "bohr":
        return BOHR_TO_ANGSTROM
    if unit_lower == "alat":
        return _qe_alat_angstrom(lines)
    raise ValueError(f"unsupported QE CELL_PARAMETERS unit {unit!r}; use angstrom or bohr")


def _qe_alat_angstrom(lines: list[str]) -> float:
    text = "\n".join(lines)
    celldm_match = re.search(
        r"celldm\s*\(\s*1\s*\)\s*=\s*([0-9.eEdD+-]+)",
        text,
        re.IGNORECASE,
    )
    if celldm_match is not None:
        return float(celldm_match.group(1).replace("D", "E").replace("d", "e")) * BOHR_TO_ANGSTROM
    a_match = re.search(r"\bA\s*=\s*([0-9.eEdD+-]+)", text)
    if a_match is not None:
        return float(a_match.group(1).replace("D", "E").replace("d", "e"))
    raise ValueError("QE alat units require celldm(1) in bohr or A in angstrom")


def _looks_like_qe_card_or_namelist(line: str) -> bool:
    upper = line.upper()
    return (
        line.startswith("&")
        or upper
        in {
            "K_POINTS",
            "CELL_PARAMETERS",
            "ATOMIC_SPECIES",
            "ATOMIC_POSITIONS",
            "CONSTRAINTS",
            "OCCUPATIONS",
            "ATOMIC_FORCES",
        }
        or upper.startswith(
            (
                "K_POINTS ",
                "CELL_PARAMETERS ",
                "ATOMIC_SPECIES ",
                "ATOMIC_POSITIONS ",
                "CONSTRAINTS ",
                "OCCUPATIONS ",
                "ATOMIC_FORCES ",
            )
        )
    )


def write_qe_kpoints(path: str | Path, k_sc_folded: NDArray[np.float64]) -> None:
    rows = ["K_POINTS crystal", f"{len(k_sc_folded)}"]
    for kpoint in np.asarray(k_sc_folded, dtype=float):
        rows.append(f"{kpoint[0]: .12f} {kpoint[1]: .12f} {kpoint[2]: .12f} 1.0")
    Path(path).write_text("\n".join(rows) + "\n")


def write_kmap(path: str | Path, qe_path: QEPath) -> None:
    header = (
        "ik\ts_pc\tkpc_1\tkpc_2\tkpc_3\t"
        "Ksc_unfold_1\tKsc_unfold_2\tKsc_unfold_3\t"
        "Ksc_fold_1\tKsc_fold_2\tKsc_fold_3\tlabel"
    )
    rows = [header]
    for idx, (kp, ksu, ksf, dist, label) in enumerate(
        zip(
            qe_path.kpoints,
            qe_path.supercell_unfolded,
            qe_path.supercell_folded,
            qe_path.distances,
            qe_path.labels,
            strict=True,
        ),
        start=1,
    ):
        rows.append(
            "\t".join(
                [
                    str(idx),
                    f"{dist:.12f}",
                    *(f"{x:.12f}" for x in kp),
                    *(f"{x:.12f}" for x in ksu),
                    *(f"{x:.12f}" for x in ksf),
                    label,
                ]
            )
        )
    Path(path).write_text("\n".join(rows) + "\n")


#: Columns of a k-map row, in order.
KMAP_COLUMNS = "ik s_pc k_pc(3) K_sc_unfolded(3) K_sc_folded(3) [label]"
_KMAP_MIN_FIELDS = 11


def read_kmap(path: str | Path) -> QEKMap:
    """Read a k-map table written by :func:`write_kmap`.

    Rows are tab-separated; a whitespace-separated file is accepted as well,
    since that is what a hand-written k-map usually looks like.  A row with the
    wrong number of fields, or a field that is not a number, is reported with
    its line number instead of surfacing as an ``IndexError`` or a bare
    ``ValueError`` from ``float`` -- these files are typically written by hand
    or by a user's script.
    """

    rows: list[list[str]] = []
    line_numbers: list[int] = []
    for number, raw in enumerate(Path(path).read_text().splitlines(), start=1):
        if not raw.strip() or raw.startswith("ik"):
            continue
        fields = raw.split("\t")
        if len(fields) < _KMAP_MIN_FIELDS:
            # A hand-written k-map is usually space-separated.  A label may not
            # contain whitespace in that case, which is why the tab form is
            # tried first.
            fields = raw.split()
        if len(fields) < _KMAP_MIN_FIELDS:
            raise ValueError(
                f"{path}, line {number}: a k-map row needs at least "
                f"{_KMAP_MIN_FIELDS} fields ({KMAP_COLUMNS}), got {len(fields)}"
            )
        rows.append(fields)
        line_numbers.append(number)
    if not rows:
        raise ValueError(f"no k-point rows found in {path}")

    def _floats(row: list[str], number: int, start: int, stop: int) -> list[float]:
        try:
            return [float(value) for value in row[start:stop]]
        except ValueError as error:
            raise ValueError(
                f"{path}, line {number}: k-map fields {start + 1}..{stop} must be "
                f"numbers ({KMAP_COLUMNS}), got {row[start:stop]}"
            ) from error

    return QEKMap(
        distances=np.array(
            [_floats(row, number, 1, 2)[0] for row, number in zip(rows, line_numbers, strict=True)],
            dtype=float,
        ),
        primitive_kpoints=np.array(
            [_floats(row, number, 2, 5) for row, number in zip(rows, line_numbers, strict=True)],
            dtype=float,
        ),
        supercell_unfolded_kpoints=np.array(
            [_floats(row, number, 5, 8) for row, number in zip(rows, line_numbers, strict=True)],
            dtype=float,
        ),
        supercell_folded_kpoints=np.array(
            [_floats(row, number, 8, 11) for row, number in zip(rows, line_numbers, strict=True)],
            dtype=float,
        ),
        labels=tuple(row[11] if len(row) > 11 else "" for row in rows),
    )


def write_ticks(path: str | Path, qe_path: QEPath) -> None:
    rows = ["s_pc\tlabel"]
    for distance, label in zip(qe_path.distances, qe_path.labels, strict=True):
        if label:
            rows.append(f"{distance:.12f}\t{label}")
    Path(path).write_text("\n".join(rows) + "\n")


def read_ticks(path: str | Path | None, kmap: QEKMap) -> tuple[list[float], list[str]]:
    if path is None:
        pairs = [
            (float(s), label) for s, label in zip(kmap.distances, kmap.labels, strict=True) if label
        ]
        return [x for x, _ in pairs], [label for _, label in pairs]
    ticks: list[float] = []
    labels: list[str] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("s_pc"):
            continue
        x_value, label = line.split("\t", 1)
        ticks.append(float(x_value))
        labels.append(label)
    return ticks, labels


def read_gnu_blocks(path: str | Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    blocks: list[list[tuple[float, float]]] = []
    block: list[tuple[float, float]] = []
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line:
            if block:
                blocks.append(block)
                block = []
            continue
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            block.append((float(parts[0]), float(parts[1])))
    if block:
        blocks.append(block)
    if not blocks:
        raise ValueError(f"no band blocks found in {path}")

    n_kpoints = len(blocks[0])
    for idx, item in enumerate(blocks, start=1):
        if len(item) != n_kpoints:
            raise ValueError(f"band block {idx} has {len(item)} k-points; expected {n_kpoints}")
    x_values = np.array([pair[0] for pair in blocks[0]], dtype=float)
    energies = np.array([[pair[1] for pair in block] for block in blocks], dtype=float).T
    return x_values, energies


def read_weight_table(path: str | Path, n_kpoints: int, n_bands: int) -> NDArray[np.float64]:
    explicit_rows = False
    weights = np.zeros((n_kpoints, n_bands), dtype=float)
    seen = np.zeros((n_kpoints, n_bands), dtype=bool)
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 3:
            continue
        explicit_rows = True
        ik = int(parts[0]) - 1
        ib = int(parts[1]) - 1
        weights[ik, ib] = float(parts[2])
        seen[ik, ib] = True

    if explicit_rows and np.all(seen):
        return weights
    if explicit_rows:
        missing = int(np.size(seen) - np.count_nonzero(seen))
        raise ValueError(f"weight table is missing {missing} ik/ib entries")

    _, gnu_weights = read_gnu_blocks(path)
    if gnu_weights.shape != (n_kpoints, n_bands):
        raise ValueError(
            f"GNU-style weight blocks have shape {gnu_weights.shape}; "
            f"expected {(n_kpoints, n_bands)}"
        )
    return gnu_weights


def write_weight_table(path: str | Path, weights: NDArray[np.float64]) -> None:
    rows = ["# ik band spectral_weight"]
    for ik in range(weights.shape[0]):
        for ib in range(weights.shape[1]):
            rows.append(f"{ik + 1:d} {ib + 1:d} {weights[ik, ib]:.12f}")
    Path(path).write_text("\n".join(rows) + "\n")


def write_spin_texture_table(path: str | Path, textures: NDArray[np.float64]) -> None:
    """Write an unfolded spin texture as rows ``ik band Sx Sy Sz |S|``.

    Backend-neutral, like :func:`write_weight_table`: the table is indexed by
    the k-map row and the band, and the length ``|S|`` is written out because
    it is the quantity that must not exceed the spectral weight.
    """

    arr = np.asarray(textures, dtype=float)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("textures must have shape (n_kpoints, n_bands, 3)")
    rows = ["# ik band spin_x spin_y spin_z spin_norm"]
    for ik in range(arr.shape[0]):
        for ib in range(arr.shape[1]):
            sx, sy, sz = arr[ik, ib]
            norm = float(np.linalg.norm(arr[ik, ib]))
            rows.append(f"{ik + 1:d} {ib + 1:d} {sx:.12f} {sy:.12f} {sz:.12f} {norm:.12f}")
    Path(path).write_text("\n".join(rows) + "\n")


def read_spin_texture_table(
    path: str | Path,
    n_kpoints: int,
    n_bands: int,
) -> NDArray[np.float64]:
    """Read back a table written by :func:`write_spin_texture_table`."""

    textures = np.zeros((n_kpoints, n_bands, 3), dtype=float)
    seen = np.zeros((n_kpoints, n_bands), dtype=bool)
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 5:
            raise ValueError("spin texture rows must be: ik band Sx Sy Sz [|S|]")
        ik = int(parts[0]) - 1
        ib = int(parts[1]) - 1
        if not (0 <= ik < n_kpoints and 0 <= ib < n_bands):
            raise ValueError(f"spin texture index ({ik + 1}, {ib + 1}) outside the band grid")
        textures[ik, ib] = [float(value) for value in parts[2:5]]
        seen[ik, ib] = True
    if not seen.all():
        raise ValueError("spin texture table does not cover every (k-point, band)")
    return textures


def write_unfolded(
    path: str | Path,
    kmap: QEKMap,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> None:
    rows = ["# ik band s_pc kpc_1 kpc_2 kpc_3 energy_eV spectral_weight"]
    for ik, distance in enumerate(kmap.distances, start=1):
        kpoint = kmap.primitive_kpoints[ik - 1]
        for ib in range(energies.shape[1]):
            rows.append(
                f"{ik:d} {ib + 1:d} {distance:.12f} "
                f"{kpoint[0]:.12f} {kpoint[1]:.12f} {kpoint[2]:.12f} "
                f"{energies[ik - 1, ib]:.12f} {weights[ik - 1, ib]:.12f}"
            )
    Path(path).write_text("\n".join(rows) + "\n")


def qe_effective_band_structure(
    kmap: QEKMap,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64] | None = None,
    *,
    reference_energy: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> EffectiveBandStructure:
    if energies.shape[0] != kmap.n_kpoints:
        raise ValueError("energies and kmap have different k-point counts")
    if weights is not None and weights.shape != energies.shape:
        raise ValueError("weights must have the same shape as energies")
    return BandUnfoldingData(
        kpoints=kmap.primitive_kpoints,
        energies=energies,
        weights=weights,
        distances=kmap.distances,
        reference_energy=reference_energy,
        source_code="qe",
        metadata=metadata or {},
    ).to_effective_band_structure()


def weights_from_coefficient_table(
    coeff_path: str | Path,
    kmap: QEKMap,
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float = 1e-6,
) -> NDArray[np.float64]:
    return compute_weights_from_coefficient_table(
        coeff_path,
        kmap.primitive_kpoints,
        kmap.supercell_folded_kpoints,
        transform,
        n_bands,
        tol=tol,
    )


def write_qe_path_files(
    path_json: str | Path,
    *,
    qe_kpoints: str | Path,
    kmap: str | Path,
    ticks: str | Path,
) -> QEPath:
    qe_path = build_qe_path(path_json)
    write_qe_kpoints(qe_kpoints, qe_path.supercell_folded)
    write_kmap(kmap, qe_path)
    write_ticks(ticks, qe_path)
    return qe_path
