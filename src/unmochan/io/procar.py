"""Reader for VASP ``PROCAR`` site- and orbital-projected character tables.

The file is a sequence of blocks

``k-point`` header -> ``band`` header -> ``ion`` table -> ...

with one table per band for a collinear calculation, four tables per band for a
noncollinear one (charge, then the three magnetization components), and the
whole thing repeated once per spin channel for a spin-polarized run.  Files
written with ``LORBIT = 12`` interleave an extra table of complex phase factors
per band; those rows carry twice as many numbers as an occupation row and are
skipped here.

The projections are the usual VASP sphere projections: they are *not*
normalized (their sum over ions and orbitals is below one by the amount of
charge outside the projection spheres and in the interstitial region), which is
why :func:`unmochan.core.projections.projection_fractions` always divides by
the per-band total instead of trusting the ``tot`` column.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

__all__ = ["ProcarData", "read_procar"]


@dataclass(frozen=True)
class ProcarData:
    """Parsed ``PROCAR`` content.

    ``projections`` has shape ``(n_spin, n_kpoints, n_bands, n_ions,
    n_orbitals)``.  For a noncollinear file ``n_spin`` is 1 and the three
    magnetization tables are kept separately in :attr:`magnetization` with shape
    ``(3, n_kpoints, n_bands, n_ions, n_orbitals)``.
    """

    kpoints: NDArray[np.float64]
    kpoint_weights: NDArray[np.float64]
    energies: NDArray[np.float64]
    occupations: NDArray[np.float64]
    projections: NDArray[np.float64]
    orbital_labels: tuple[str, ...]
    magnetization: NDArray[np.float64] | None = None

    def __post_init__(self) -> None:
        projections = np.asarray(self.projections, dtype=float)
        if projections.ndim != 5:
            raise ValueError("projections must have shape (n_spin, n_k, n_bands, n_ions, n_orb)")
        if projections.shape[-1] != len(self.orbital_labels):
            raise ValueError("orbital_labels length must match the projection table width")
        if self.energies.shape != projections.shape[:3]:
            raise ValueError("energies must have shape (n_spin, n_kpoints, n_bands)")
        if self.magnetization is not None:
            magnetization = np.asarray(self.magnetization, dtype=float)
            if magnetization.shape != (3, *projections.shape[1:]):
                raise ValueError("magnetization must have shape (3, n_k, n_bands, n_ions, n_orb)")
            object.__setattr__(self, "magnetization", magnetization)
        object.__setattr__(self, "projections", projections)
        object.__setattr__(self, "orbital_labels", tuple(self.orbital_labels))

    @property
    def n_spin(self) -> int:
        return int(self.projections.shape[0])

    @property
    def n_kpoints(self) -> int:
        return int(self.projections.shape[1])

    @property
    def n_bands(self) -> int:
        return int(self.projections.shape[2])

    @property
    def n_ions(self) -> int:
        return int(self.projections.shape[3])

    @property
    def noncollinear(self) -> bool:
        return self.magnetization is not None


def read_procar(path: str | Path) -> ProcarData:
    """Read a VASP ``PROCAR`` file."""

    text = Path(path).read_text()
    lines = text.splitlines()
    if not lines:
        raise ValueError(f"{path} is empty")

    blocks = list(_split_spin_blocks(lines))
    if not blocks:
        raise ValueError(f"{path} contains no '# of k-points' header")

    parsed = [_parse_spin_block(block) for block in blocks]
    first = parsed[0]
    for other in parsed[1:]:
        if other.orbital_labels != first.orbital_labels:
            raise ValueError("spin channels disagree on the orbital labels")
        if other.charge.shape != first.charge.shape:
            raise ValueError("spin channels disagree on the table shape")

    noncollinear = first.n_tables == 4
    if noncollinear and len(parsed) != 1:
        raise ValueError("a noncollinear PROCAR must contain a single spin block")

    projections = np.stack([item.charge for item in parsed], axis=0)
    energies = np.stack([item.energies for item in parsed], axis=0)
    occupations = np.stack([item.occupations for item in parsed], axis=0)
    magnetization = first.magnetization if noncollinear else None
    return ProcarData(
        kpoints=first.kpoints,
        kpoint_weights=first.kpoint_weights,
        energies=energies,
        occupations=occupations,
        projections=projections,
        orbital_labels=first.orbital_labels,
        magnetization=magnetization,
    )


@dataclass
class _SpinBlock:
    kpoints: NDArray[np.float64]
    kpoint_weights: NDArray[np.float64]
    energies: NDArray[np.float64]
    occupations: NDArray[np.float64]
    charge: NDArray[np.float64]
    magnetization: NDArray[np.float64] | None
    orbital_labels: tuple[str, ...]
    n_tables: int


def _split_spin_blocks(lines: Sequence[str]) -> Iterator[Sequence[str]]:
    starts = [index for index, line in enumerate(lines) if "# of k-points" in line]
    for position, start in enumerate(starts):
        stop = starts[position + 1] if position + 1 < len(starts) else len(lines)
        yield lines[start:stop]


def _parse_counts(line: str) -> tuple[int, int, int]:
    tokens = line.replace(":", " ").split()
    numbers = [int(token) for token in tokens if _is_int(token)]
    if len(numbers) < 3:
        raise ValueError(f"cannot read k-point/band/ion counts from {line!r}")
    return numbers[0], numbers[1], numbers[2]


def _parse_spin_block(lines: Sequence[str]) -> _SpinBlock:
    n_kpoints, n_bands, n_ions = _parse_counts(lines[0])
    kpoints = np.zeros((n_kpoints, 3), dtype=float)
    kpoint_weights = np.zeros(n_kpoints, dtype=float)
    energies = np.zeros((n_kpoints, n_bands), dtype=float)
    occupations = np.zeros((n_kpoints, n_bands), dtype=float)
    tables: list[list[NDArray[np.float64]]] = []
    orbital_labels: tuple[str, ...] | None = None

    index = 1
    ik = -1
    ib = -1
    seen = np.zeros((n_kpoints, n_bands), dtype=bool)
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        lowered = stripped.lower()
        if lowered.startswith("k-point"):
            ik, kpoint, weight = _parse_kpoint_line(stripped, n_kpoints)
            kpoints[ik] = kpoint
            kpoint_weights[ik] = weight
            index += 1
            continue
        if lowered.startswith("band"):
            ib, energy, occupation = _parse_band_line(stripped, n_bands)
            if ik < 0:
                raise ValueError("a band header appeared before any k-point header")
            energies[ik, ib] = energy
            occupations[ik, ib] = occupation
            if seen[ik, ib]:
                raise ValueError(f"duplicate band {ib + 1} at k-point {ik + 1}")
            seen[ik, ib] = True
            tables.append([])
            index += 1
            continue
        if lowered.startswith("ion"):
            labels = tuple(token for token in stripped.split()[1:] if token.lower() != "tot")
            if orbital_labels is None:
                orbital_labels = labels
            table, index = _parse_ion_table(lines, index + 1, n_ions, len(labels))
            if table is not None:
                if not tables:
                    raise ValueError("an ion table appeared before any band header")
                tables[-1].append(table)
            continue
        index += 1

    if orbital_labels is None:
        raise ValueError("no ion table was found")
    if not seen.all():
        raise ValueError("PROCAR is missing at least one (k-point, band) entry")
    if len(tables) != n_kpoints * n_bands:
        raise ValueError(f"expected {n_kpoints * n_bands} band tables, found {len(tables)}")
    n_tables = len(tables[0])
    if n_tables not in (1, 4):
        raise ValueError(f"expected 1 or 4 projection tables per band, found {n_tables}")
    if any(len(item) != n_tables for item in tables):
        raise ValueError("inconsistent number of projection tables across bands")

    stacked = np.array(tables, dtype=float)
    stacked = stacked.reshape(n_kpoints, n_bands, n_tables, n_ions, len(orbital_labels))
    charge = stacked[:, :, 0]
    magnetization = np.moveaxis(stacked[:, :, 1:], 2, 0) if n_tables == 4 else None
    return _SpinBlock(
        kpoints=kpoints,
        kpoint_weights=kpoint_weights,
        energies=energies,
        occupations=occupations,
        charge=charge,
        magnetization=magnetization,
        orbital_labels=orbital_labels,
        n_tables=n_tables,
    )


def _parse_kpoint_line(line: str, n_kpoints: int) -> tuple[int, NDArray[np.float64], float]:
    head, _, tail = line.partition("weight")
    weight = float(tail.replace("=", " ").split()[0]) if tail.strip() else 0.0
    body = head[len("k-point") :].replace(":", " ")
    # VASP writes coordinates without separating blanks for large negative
    # values, e.g. "0.50000000-0.50000000"; re-insert the missing separator.
    body = body.replace("-", " -")
    tokens = body.split()
    if len(tokens) < 4:
        raise ValueError(f"cannot read k-point header {line!r}")
    index = int(tokens[0]) - 1
    if not 0 <= index < n_kpoints:
        raise ValueError(f"k-point index out of range in {line!r}")
    coordinates = np.array([float(token) for token in tokens[1:4]], dtype=float)
    return index, coordinates, weight


def _parse_band_line(line: str, n_bands: int) -> tuple[int, float, float]:
    tokens = line.replace("#", " ").split()
    if len(tokens) < 2:
        raise ValueError(f"cannot read band header {line!r}")
    index = int(tokens[1]) - 1
    if not 0 <= index < n_bands:
        raise ValueError(f"band index out of range in {line!r}")
    numbers = [float(token) for token in tokens[2:] if _is_float(token)]
    energy = numbers[0] if numbers else 0.0
    occupation = numbers[1] if len(numbers) > 1 else 0.0
    return index, energy, occupation


def _parse_ion_table(
    lines: Sequence[str],
    start: int,
    n_ions: int,
    n_orbitals: int,
) -> tuple[NDArray[np.float64] | None, int]:
    """Read ``n_ions`` rows of ``n_orbitals`` numbers starting at ``start``.

    Returns ``(None, next_index)`` for a phase-factor table, whose rows carry
    real and imaginary parts and therefore twice as many numbers.
    """

    rows: list[list[float]] = []
    index = start
    is_phase = False
    while index < len(lines) and len(rows) < n_ions:
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        tokens = stripped.split()
        if not _is_int(tokens[0]):
            break
        values = [float(token) for token in tokens[1:]]
        if len(values) >= 2 * n_orbitals:
            is_phase = True
        elif len(values) < n_orbitals:
            raise ValueError(f"projection row is too short: {stripped!r}")
        rows.append(values[:n_orbitals])
        index += 1

    if len(rows) != n_ions:
        raise ValueError(f"expected {n_ions} ion rows, found {len(rows)}")
    # Skip the trailing "tot" summary row when present.
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.lower().startswith("tot"):
            index += 1
        break
    if is_phase:
        return None, index
    return np.array(rows, dtype=float), index


def _is_int(token: str) -> bool:
    try:
        int(token)
    except ValueError:
        return False
    return True


def _is_float(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True
