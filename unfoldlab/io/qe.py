"""Quantum ESPRESSO text I/O and effective-band helpers."""

from __future__ import annotations

import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.kpoints import interpolate_segment
from unfoldlab.core.numerics import wrap_fractional
from unfoldlab.core.plane_waves import compute_weights_from_coefficient_table
from unfoldlab.core.spectral import EffectiveBandStructure
from unfoldlab.core.structures import Structure
from unfoldlab.core.unfolding import BandUnfoldingData

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


def read_path_json(path: str | Path) -> tuple[NDArray[np.int64], list[PathNode]]:
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
    return rounded, nodes


def read_pw_input_structure(path: str | Path) -> Structure:
    """Read a QE ``pw.x`` input with ``ibrav=0`` style cell/positions blocks."""

    path = Path(path)
    raw_lines = path.read_text().splitlines()
    lines = [_strip_qe_comment(line).strip() for line in raw_lines]
    lines = [line for line in lines if line]
    cell_index = _find_qe_card(lines, "CELL_PARAMETERS")
    positions_index = _find_qe_card(lines, "ATOMIC_POSITIONS")
    if cell_index is None or positions_index is None:
        raise ValueError(
            f"{path} must contain CELL_PARAMETERS and ATOMIC_POSITIONS cards"
        )

    cell_unit = _qe_card_unit(lines[cell_index], default="angstrom")
    lattice_rows = [
        [float(value) for value in lines[cell_index + offset].split()[:3]]
        for offset in range(1, 4)
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


def interpolate_path(nodes: list[PathNode], transform: NDArray[np.integer]) -> QEPath:
    if len(nodes) < 2:
        raise ValueError("at least two path nodes are required")

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
                cumulative += float(np.linalg.norm(point.fractional - kpoints[-1]))
            kpoints.append(point.fractional)
            labels.append(point.label or "")
            distances.append(cumulative)

    return QEPath(
        transform=np.asarray(transform, dtype=int),
        kpoints=np.asarray(kpoints, dtype=float),
        distances=np.asarray(distances, dtype=float),
        labels=tuple(labels),
    )


def build_qe_path(path_json: str | Path) -> QEPath:
    matrix, nodes = read_path_json(path_json)
    return interpolate_path(nodes, matrix)


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
    raise ValueError(
        f"unsupported QE CELL_PARAMETERS unit {unit!r}; use angstrom or bohr"
    )


def _qe_alat_angstrom(lines: list[str]) -> float:
    text = "\n".join(lines)
    celldm_match = re.search(
        r"celldm\s*\(\s*1\s*\)\s*=\s*([0-9.eEdD+-]+)",
        text,
        re.IGNORECASE,
    )
    if celldm_match is not None:
        return (
            float(celldm_match.group(1).replace("D", "E").replace("d", "e"))
            * BOHR_TO_ANGSTROM
        )
    a_match = re.search(r"\bA\s*=\s*([0-9.eEdD+-]+)", text)
    if a_match is not None:
        return float(a_match.group(1).replace("D", "E").replace("d", "e"))
    raise ValueError("QE alat units require celldm(1) in bohr or A in angstrom")


def _looks_like_qe_card_or_namelist(line: str) -> bool:
    upper = line.upper()
    return line.startswith("&") or upper in {
        "K_POINTS",
        "CELL_PARAMETERS",
        "ATOMIC_SPECIES",
        "ATOMIC_POSITIONS",
        "CONSTRAINTS",
        "OCCUPATIONS",
        "ATOMIC_FORCES",
    } or upper.startswith(
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


def read_kmap(path: str | Path) -> QEKMap:
    rows: list[list[str]] = []
    for raw in Path(path).read_text().splitlines():
        if not raw.strip() or raw.startswith("ik"):
            continue
        rows.append(raw.split("\t"))
    if not rows:
        raise ValueError(f"no k-point rows found in {path}")
    return QEKMap(
        distances=np.array([float(row[1]) for row in rows], dtype=float),
        primitive_kpoints=np.array([[float(x) for x in row[2:5]] for row in rows], dtype=float),
        supercell_unfolded_kpoints=np.array(
            [[float(x) for x in row[5:8]] for row in rows], dtype=float
        ),
        supercell_folded_kpoints=np.array(
            [[float(x) for x in row[8:11]] for row in rows], dtype=float
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
        ticks = [
            (float(s), label)
            for s, label in zip(kmap.distances, kmap.labels, strict=True)
            if label
        ]
        return [x for x, _ in ticks], [label for _, label in ticks]
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
    transform: NDArray[np.float64],
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


def color_from_weight(weight: float) -> str:
    stops = np.array(
        [[68, 1, 84], [49, 104, 142], [53, 183, 121], [253, 231, 37]],
        dtype=float,
    )
    value = float(np.clip(weight, 0.0, 1.0))
    scaled = value * (len(stops) - 1)
    idx = min(int(np.floor(scaled)), len(stops) - 2)
    frac = scaled - idx
    rgb = (1.0 - frac) * stops[idx] + frac * stops[idx + 1]
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in rgb)


def plot_unfolded(
    path: str | Path,
    kmap: QEKMap,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    ticks: tuple[list[float], list[str]],
    *,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
) -> Path:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return plot_unfolded_svg(
            path,
            kmap,
            energies,
            weights,
            ticks,
            fermi=fermi,
            emin=emin,
            emax=emax,
            marker_scale=marker_scale,
        )

    out_path = Path(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    x_values = np.repeat(kmap.distances, energies.shape[1])
    y_values = (energies - fermi).reshape(-1)
    clipped_weights = np.clip(weights.reshape(-1), 0.0, 1.0)
    scatter = ax.scatter(
        x_values,
        y_values,
        s=3.0 + marker_scale * clipped_weights,
        c=clipped_weights,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        alpha=0.85,
    )
    tick_x, tick_labels = ticks
    for xpos in tick_x:
        ax.axvline(xpos, color="0.75", lw=0.7, zorder=0)
    if tick_x:
        ax.set_xticks(tick_x, tick_labels)
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    x_min = float(kmap.distances.min())
    x_max = float(kmap.distances.max())
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    ax.set_xlim(x_min, x_max)
    if emin is not None or emax is not None:
        ax.set_ylim(emin, emax)
    ax.set_ylabel("Energy - Ef (eV)" if fermi else "Energy (eV)")
    ax.set_xlabel("Primitive-cell k-path")
    fig.colorbar(scatter, ax=ax, pad=0.02).set_label("Spectral weight")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_unfolded_svg(
    path: str | Path,
    kmap: QEKMap,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    ticks: tuple[list[float], list[str]],
    *,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
) -> Path:
    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg":
        svg_path = svg_path.with_suffix(".svg")
    width, height = 900, 600
    left, right, top, bottom = 82, 28, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom
    x_min = float(kmap.distances.min())
    x_max = float(kmap.distances.max())
    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    y_values = energies - fermi
    y_min = float(np.min(y_values) if emin is None else emin)
    y_max = float(np.max(y_values) if emax is None else emax)
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    tick_x, tick_labels = ticks
    rows = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
            'fill="white" stroke="#222" stroke-width="1"/>'
        ),
    ]
    for xpos, label in zip(tick_x, tick_labels, strict=True):
        px = sx(xpos)
        rows.append(
            
                f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" '
                f'y2="{top + plot_h}" stroke="#c8c8c8" stroke-width="1"/>'
            
        )
        rows.append(
            
                f'<text x="{px:.2f}" y="{height - 36}" '
                'font-family="Arial, sans-serif" font-size="15" '
                f'text-anchor="middle">{html.escape(label)}</text>'
            
        )
    if y_min <= 0.0 <= y_max:
        y0 = sy(0.0)
        rows.append(
            
                f'<line x1="{left}" y1="{y0:.2f}" x2="{left + plot_w}" '
                f'y2="{y0:.2f}" stroke="#666" stroke-width="1" '
                'stroke-dasharray="5,5"/>'
            
        )
    for tick in np.linspace(y_min, y_max, 6):
        py = sy(float(tick))
        rows.append(
            
                f'<line x1="{left - 5}" y1="{py:.2f}" x2="{left}" '
                f'y2="{py:.2f}" stroke="#222" stroke-width="1"/>'
            
        )
        rows.append(
            
                f'<text x="{left - 10}" y="{py + 5:.2f}" '
                'font-family="Arial, sans-serif" font-size="13" '
                f'text-anchor="end">{tick:.2f}</text>'
            
        )
    max_radius = max(2.0, 1.5 + marker_scale * 0.12)
    for ik, distance in enumerate(kmap.distances):
        px = sx(float(distance))
        for ib in range(energies.shape[1]):
            weight = float(np.clip(weights[ik, ib], 0.0, 1.0))
            py = sy(float(y_values[ik, ib]))
            radius = 1.2 + max_radius * weight
            color = color_from_weight(weight)
            rows.append(
                
                    f'<circle cx="{px:.2f}" cy="{py:.2f}" r="{radius:.2f}" '
                    f'fill="{color}" fill-opacity="0.82"/>'
                
            )
    ylabel = "Energy - Ef (eV)" if fermi else "Energy (eV)"
    rows.extend(
        [
            (
                f'<text x="{left + plot_w / 2:.2f}" y="{height - 12}" '
                'font-family="Arial, sans-serif" font-size="16" '
                'text-anchor="middle">Primitive-cell k-path</text>'
            ),
            (
                f'<text x="22" y="{top + plot_h / 2:.2f}" '
                'font-family="Arial, sans-serif" font-size="16" '
                'text-anchor="middle" '
                f'transform="rotate(-90 22 {top + plot_h / 2:.2f})">'
                f"{html.escape(ylabel)}</text>"
            ),
            (
                '<text x="730" y="32" font-family="Arial, sans-serif" '
                'font-size="13">color/size = spectral weight</text>'
            ),
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(rows) + "\n")
    return svg_path
