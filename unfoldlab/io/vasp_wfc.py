"""VASP WAVECAR readers and plane-wave unfolding adapters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.unfolding import PlaneWaveKPointData, compute_plane_wave_unfolding_weights

RYDBERG_TO_EV = 13.605693122994
HSQDTM = 3.80998212  # hbar^2 / (2 m_e) in eV Angstrom^2


@dataclass(frozen=True)
class WavecarHeader:
    record_length: int
    n_spin: int
    rtag: int
    n_kpoints: int
    n_bands: int
    encut: float
    lattice: NDArray[np.float64]
    efermi: float
    coefficient_dtype: np.dtype

    @property
    def spinor_components(self) -> int:
        return 2 if self.rtag == 53300 else 1


@dataclass(frozen=True)
class WavecarKPoint:
    spin_index: int
    kpoint_index: int
    n_plane_waves: int
    kpoint: NDArray[np.float64]
    energies: NDArray[np.float64]
    occupations: NDArray[np.float64]
    g_vectors: NDArray[np.int64]


class WavecarReader:
    """Read a standard VASP WAVECAR.

    The supported tags are the common VASP complex64, complex128, and spinor
    layouts. Gamma-only reduced storage is intentionally unsupported because it
    needs backend-specific reconstruction to be comparable with QE.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = self.path.open("rb")
        self.header = self._read_header()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> WavecarReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _read_record_floats(
        self,
        record_index: int,
        count: int | None = None,
    ) -> NDArray[np.float64]:
        self._handle.seek((record_index - 1) * self.header_record_length)
        data = self._handle.read(self.header_record_length)
        values = np.frombuffer(data, dtype=np.float64)
        return values if count is None else values[:count]

    @property
    def header_record_length(self) -> int:
        if hasattr(self, "_record_length"):
            return int(self._record_length)
        return 24

    def _read_header(self) -> WavecarHeader:
        self._handle.seek(0)
        first = np.frombuffer(self._handle.read(24), dtype=np.float64)
        if first.size < 3:
            raise ValueError(f"{self.path} is too short to be a WAVECAR")
        record_length = int(round(first[0]))
        if record_length <= 0:
            raise ValueError(f"invalid WAVECAR record length: {record_length}")
        object.__setattr__(self, "_record_length", record_length)
        n_spin = int(round(first[1]))
        rtag = int(round(first[2]))
        if rtag == 45200:
            coeff_dtype = np.dtype(np.complex64)
        elif rtag == 45210:
            coeff_dtype = np.dtype(np.complex128)
        elif rtag == 53300:
            coeff_dtype = np.dtype(np.complex64)
        else:
            raise ValueError(
                f"unsupported WAVECAR RTAG {rtag}; gamma-only/special storage is not supported"
            )

        second = self._read_record_floats(2, 13)
        if second.size < 12:
            raise ValueError(f"{self.path} has an incomplete WAVECAR header record")
        n_kpoints = int(round(second[0]))
        n_bands = int(round(second[1]))
        encut = float(second[2])
        lattice = second[3:12].reshape(3, 3)
        efermi = float(second[12]) if second.size > 12 else 0.0
        return WavecarHeader(
            record_length=record_length,
            n_spin=n_spin,
            rtag=rtag,
            n_kpoints=n_kpoints,
            n_bands=n_bands,
            encut=encut,
            lattice=lattice,
            efermi=efermi,
            coefficient_dtype=coeff_dtype,
        )

    def _kpoint_record_index(self, spin: int, kpoint_index: int) -> int:
        if spin < 1 or spin > self.header.n_spin:
            raise ValueError(f"spin must be in 1..{self.header.n_spin}")
        if kpoint_index < 1 or kpoint_index > self.header.n_kpoints:
            raise ValueError(f"kpoint_index must be in 1..{self.header.n_kpoints}")
        block = (spin - 1) * self.header.n_kpoints + (kpoint_index - 1)
        return 3 + block * (self.header.n_bands + 1)

    def read_kpoint_header(self, spin: int, kpoint_index: int) -> WavecarKPoint:
        values = self._read_record_floats(self._kpoint_record_index(spin, kpoint_index))
        min_values = 4 + self.header.n_bands * 3
        if values.size < min_values:
            raise ValueError(
                f"WAVECAR k-point header {spin}/{kpoint_index} has {values.size} values; "
                f"expected at least {min_values}"
            )
        n_plane_waves = int(round(values[0]))
        kpoint = values[1:4].astype(float)
        band_data = values[4 : 4 + self.header.n_bands * 3].reshape(self.header.n_bands, 3)
        energies = band_data[:, 0].astype(float)
        occupations = band_data[:, 2].astype(float)
        g_vectors = generate_vasp_g_vectors(
            self.header.lattice,
            kpoint,
            self.header.encut,
            n_expected=n_plane_waves,
        )
        return WavecarKPoint(
            spin_index=spin,
            kpoint_index=kpoint_index,
            n_plane_waves=n_plane_waves,
            kpoint=kpoint,
            energies=energies,
            occupations=occupations,
            g_vectors=g_vectors,
        )

    def read_coefficients(
        self,
        spin: int,
        kpoint_index: int,
        band_index: int,
    ) -> NDArray[np.complex128]:
        if band_index < 1 or band_index > self.header.n_bands:
            raise ValueError(f"band_index must be in 1..{self.header.n_bands}")
        k_header = self.read_kpoint_header(spin, kpoint_index)
        record_index = self._kpoint_record_index(spin, kpoint_index) + band_index
        self._handle.seek((record_index - 1) * self.header.record_length)
        data = self._handle.read(self.header.record_length)
        raw = np.frombuffer(data, dtype=self.header.coefficient_dtype)
        n_total = k_header.n_plane_waves * self.header.spinor_components
        if raw.size < n_total:
            raise ValueError(
                f"WAVECAR coefficient record has {raw.size} values; expected {n_total}"
            )
        coeffs = raw[:n_total].astype(np.complex128, copy=False)
        if self.header.spinor_components == 1:
            return coeffs.reshape(1, k_header.n_plane_waves)
        return coeffs.reshape(self.header.spinor_components, k_header.n_plane_waves)

    def read_kpoint_wavefunction(
        self,
        spin: int,
        kpoint_index: int,
    ) -> tuple[WavecarKPoint, NDArray[np.complex128]]:
        k_header = self.read_kpoint_header(spin, kpoint_index)
        coeffs = np.zeros(
            (self.header.n_bands, self.header.spinor_components, k_header.n_plane_waves),
            dtype=np.complex128,
        )
        for band in range(1, self.header.n_bands + 1):
            coeffs[band - 1] = self.read_coefficients(spin, kpoint_index, band)
        return k_header, coeffs

    def read_all_energies(self, spin: int = 1) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        kpoints = np.zeros((self.header.n_kpoints, 3), dtype=float)
        energies = np.zeros((self.header.n_kpoints, self.header.n_bands), dtype=float)
        for ik in range(1, self.header.n_kpoints + 1):
            k_header = self.read_kpoint_header(spin, ik)
            kpoints[ik - 1] = k_header.kpoint
            energies[ik - 1] = k_header.energies
        return kpoints, energies

    def plane_wave_kpoint_data(
        self,
        primitive_kpoint: NDArray[np.float64],
        folded_supercell_kpoint: NDArray[np.float64],
        *,
        spin: int,
        kpoint_index: int,
    ) -> PlaneWaveKPointData:
        k_header, coeffs = self.read_kpoint_wavefunction(spin, kpoint_index)
        return PlaneWaveKPointData(
            primitive_kpoint=primitive_kpoint,
            folded_supercell_kpoint=folded_supercell_kpoint,
            g_supercell=k_header.g_vectors,
            coefficients=coeffs,
        )


def read_wavecar_header(path: str | Path) -> WavecarHeader:
    with WavecarReader(path) as reader:
        return reader.header


def generate_vasp_g_vectors(
    lattice: NDArray[np.float64],
    kpoint: NDArray[np.float64],
    encut: float,
    *,
    n_expected: int | None = None,
) -> NDArray[np.int64]:
    """Generate VASP plane-wave G-vectors in fractional reciprocal coordinates."""

    lattice_arr = np.asarray(lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_arr).T
    reciprocal_lengths = np.linalg.norm(reciprocal, axis=1)
    max_index = np.ceil(np.sqrt(float(encut) / HSQDTM) / reciprocal_lengths).astype(int) + 1
    k_arr = np.asarray(kpoint, dtype=float)
    vectors: list[tuple[int, int, int]] = []
    energies: list[float] = []
    for n1 in range(-max_index[0], max_index[0] + 1):
        for n2 in range(-max_index[1], max_index[1] + 1):
            for n3 in range(-max_index[2], max_index[2] + 1):
                g = np.array([n1, n2, n3], dtype=float)
                cart = (g + k_arr) @ reciprocal
                kinetic = HSQDTM * float(np.dot(cart, cart))
                if kinetic <= encut + 1e-10:
                    vectors.append((n1, n2, n3))
                    energies.append(kinetic)
    if not vectors:
        raise ValueError("no VASP G-vectors generated; check lattice, k-point, and encut")
    order = np.lexsort(
        (
            np.array([v[2] for v in vectors]),
            np.array([v[1] for v in vectors]),
            np.array([v[0] for v in vectors]),
            np.array(energies),
        )
    )
    g_vectors = np.array([vectors[index] for index in order], dtype=int)
    if n_expected is not None and g_vectors.shape[0] != n_expected:
        if g_vectors.shape[0] < n_expected:
            raise ValueError(
                f"generated {g_vectors.shape[0]} VASP G-vectors, WAVECAR expects {n_expected}"
            )
        g_vectors = g_vectors[:n_expected]
    return g_vectors


def compute_weights_from_wavecar(
    wavecar: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: NDArray[np.float64],
    *,
    spin: int = 1,
    tol: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(kpoints, energies, weights)`` from a VASP WAVECAR."""

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    with WavecarReader(wavecar) as reader:
        if primitive.shape[0] != reader.header.n_kpoints:
            raise ValueError(
                f"kmap has {primitive.shape[0]} k-points but WAVECAR has "
                f"{reader.header.n_kpoints}"
            )
        kpoints = np.zeros((reader.header.n_kpoints, 3), dtype=float)
        energies = np.zeros((reader.header.n_kpoints, reader.header.n_bands), dtype=float)
        data: list[PlaneWaveKPointData] = []
        for ik in range(1, reader.header.n_kpoints + 1):
            k_header, coeffs = reader.read_kpoint_wavefunction(spin, ik)
            kpoints[ik - 1] = k_header.kpoint
            energies[ik - 1] = k_header.energies
            data.append(
                PlaneWaveKPointData(
                    primitive_kpoint=primitive[ik - 1],
                    folded_supercell_kpoint=folded[ik - 1],
                    g_supercell=k_header.g_vectors,
                    coefficients=coeffs,
                )
            )
        weights = compute_plane_wave_unfolding_weights(data, transform, tol=tol)
    return kpoints, energies, weights
