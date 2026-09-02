"""VASP WAVECAR readers and plane-wave unfolding adapters."""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.gamma import expand_gamma_half_basis
from unfoldlab.core.numerics import TransformLike, allclose_mod1
from unfoldlab.core.plane_waves import (
    spin_texture_from_coefficients,
    state_norms_from_coefficients,
    weights_from_coefficients,
)
from unfoldlab.core.symmetry import map_kpoints_to_stored
from unfoldlab.core.unfolding import (
    PlaneWaveKPointData,
    SharedWavefunctionGroup,
    compute_plane_wave_unfolding_weights,
    compute_plane_wave_unfolding_weights_chunked,
    compute_plane_wave_unfolding_weights_shared,
)

RYDBERG_TO_EV = 13.605693122994
HSQDTM = 3.80998212  # hbar^2 / (2 m_e) in eV Angstrom^2

#: WAVECAR record tags.  45200/45210 hold the full plane-wave sphere in single
#: and double precision; 53300/53310 are the reduced storage written by a
#: gamma-only binary, which keeps one member of every ``{G, -G}`` pair.
RTAG_COMPLEX_SINGLE = 45200
RTAG_COMPLEX_DOUBLE = 45210
RTAG_GAMMA_SINGLE = 53300
RTAG_GAMMA_DOUBLE = 53310

#: The two half-space conventions a gamma-only VASP run can use.  Which one a
#: file uses is decided by matching the number of stored plane waves.
GAMMA_HALF_CONVENTIONS = ("x", "z")


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
    def gamma_only(self) -> bool:
        """Whether the file uses time-reversal reduced (gamma-only) storage."""

        return self.rtag in {RTAG_GAMMA_SINGLE, RTAG_GAMMA_DOUBLE}


@dataclass(frozen=True)
class WavecarKPoint:
    spin_index: int
    kpoint_index: int
    #: Number of coefficients stored per band, as recorded in the file.  For a
    #: noncollinear (spinor) calculation this is twice the number of
    #: G-vectors.
    n_plane_waves: int
    kpoint: NDArray[np.float64]
    energies: NDArray[np.float64]
    occupations: NDArray[np.float64]
    #: Plane-wave basis used for unfolding: the stored list for an ordinary
    #: file, the time-reversal expanded list for a gamma-only file.
    g_vectors: NDArray[np.int64]
    #: Number of polarization/spinor components per G-vector.
    n_components: int = 1
    #: Half-space convention detected for a gamma-only file, else ``None``.
    gamma_half: str | None = None
    #: The G-vectors as *stored* in the file: the same list as ``g_vectors``
    #: for an ordinary file, the un-expanded half sphere for a gamma-only one.
    #: Kept so the expansion does not have to regenerate it.
    g_stored: NDArray[np.int64] | None = None

    @property
    def n_stored_g(self) -> int:
        """Number of G-vectors actually stored in the file."""

        return self.n_plane_waves // self.n_components


class WavecarReader:
    """Read a VASP WAVECAR.

    Supported layouts are the full-sphere single- and double-precision tags
    (45200/45210), noncollinear spinor files (detected from the number of
    stored coefficients rather than from the tag), and the gamma-only reduced
    tags (53300/53310), whose half basis is expanded with ``c(-G) = conj(c(G))``
    before the coefficients are handed to the unfolding kernels.
    """

    def __init__(self, path: str | Path, *, gamma_half: str = "x", cache_size: int = 2):
        if gamma_half not in GAMMA_HALF_CONVENTIONS:
            raise ValueError(f"gamma_half must be one of {GAMMA_HALF_CONVENTIONS}")
        if cache_size < 1:
            raise ValueError("cache_size must be at least 1")
        self.path = Path(path)
        self.gamma_half = gamma_half
        self._handle = self.path.open("rb")
        self.header = self._read_header()
        # Deriving a k-point header means generating the whole plane-wave
        # sphere, which costs far more than the record read itself.  Reading a
        # band re-derives it, so without a cache a k-point with ``n_bands``
        # bands generated the sphere ``n_bands + 1`` times.  Access is
        # sequential, so a two-entry cache is enough; the entries are large
        # (three integers per G-vector), which is why it is bounded.
        self._cache_size = int(cache_size)
        self._kpoint_cache: OrderedDict[tuple[int, int], WavecarKPoint] = OrderedDict()

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
        coeff_dtype: np.dtype[np.complexfloating]
        if rtag in {RTAG_COMPLEX_SINGLE, RTAG_GAMMA_SINGLE}:
            coeff_dtype = np.dtype(np.complex64)
        elif rtag in {RTAG_COMPLEX_DOUBLE, RTAG_GAMMA_DOUBLE}:
            coeff_dtype = np.dtype(np.complex128)
        else:
            raise ValueError(f"unsupported WAVECAR RTAG {rtag}")

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
        key = (spin, kpoint_index)
        cached = self._kpoint_cache.get(key)
        if cached is not None:
            self._kpoint_cache.move_to_end(key)
            return cached
        result = self._read_kpoint_header_uncached(spin, kpoint_index)
        self._kpoint_cache[key] = result
        while len(self._kpoint_cache) > self._cache_size:
            self._kpoint_cache.popitem(last=False)
        return result

    def _read_kpoint_header_uncached(self, spin: int, kpoint_index: int) -> WavecarKPoint:
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
        g_stored, n_components, gamma_half = self._resolve_basis(kpoint, n_plane_waves)
        if gamma_half is None:
            g_vectors = g_stored
        else:
            g_vectors, _ = expand_gamma_half_basis(
                g_stored, np.zeros((1, 1, g_stored.shape[0]), dtype=np.complex128)
            )
        return WavecarKPoint(
            spin_index=spin,
            kpoint_index=kpoint_index,
            n_plane_waves=n_plane_waves,
            kpoint=kpoint,
            energies=energies,
            occupations=occupations,
            g_vectors=g_vectors,
            n_components=n_components,
            gamma_half=gamma_half,
            g_stored=g_stored,
        )

    def _resolve_basis(
        self,
        kpoint: NDArray[np.float64],
        n_plane_waves: int,
    ) -> tuple[NDArray[np.int64], int, str | None]:
        """Return ``(stored G-vectors, n_components, gamma_half)``.

        The number of coefficients stored per band decides the layout:

        * ``n_plane_waves == n_g``: one component per G-vector;
        * ``n_plane_waves == 2 * n_g``: a noncollinear spinor calculation.  The
          record tag does not distinguish this case -- VASP writes spinor
          wavefunctions with the ordinary complex tags -- so it has to be
          detected here.

        For a gamma-only file the stored list is a half sphere.  Which of the
        two half-space conventions was used is a property of the VASP build, not
        of the file, so the reader's ``gamma_half`` (default ``"x"``) is tried
        first and the other convention only if the stored count rules the
        preferred one out.  For a cell whose two halves have the same size the
        count cannot discriminate; pass ``gamma_half="z"`` explicitly if the
        weights come out wrong for a gamma-only run.
        """

        if self.header.gamma_only:
            preference = [self.gamma_half] + [
                half for half in GAMMA_HALF_CONVENTIONS if half != self.gamma_half
            ]
            candidates = [
                (
                    half,
                    generate_vasp_g_vectors(
                        self.header.lattice, kpoint, self.header.encut, gamma_half=half
                    ),
                )
                for half in preference
            ]
            for half, g_stored in candidates:
                for components in (1, 2):
                    if n_plane_waves == components * g_stored.shape[0]:
                        return g_stored, components, half
            counts = {half: int(g.shape[0]) for half, g in candidates}
            raise ValueError(
                f"gamma-only WAVECAR stores {n_plane_waves} coefficients per band, "
                f"which matches neither half-space convention (sizes {counts}); "
                "the plane-wave basis cannot be reconstructed"
            )

        g_full = generate_vasp_g_vectors(self.header.lattice, kpoint, self.header.encut)
        n_g = g_full.shape[0]
        if n_plane_waves == n_g:
            return g_full, 1, None
        if n_plane_waves == 2 * n_g:
            return g_full, 2, None
        raise ValueError(
            f"WAVECAR stores {n_plane_waves} coefficients per band but the plane-wave "
            f"basis generated from ENCUT={self.header.encut:g} eV has {n_g} G-vectors "
            "(and a spinor file would have twice that); the G-vector list and the "
            "coefficients cannot be paired up"
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
        n_total = k_header.n_plane_waves
        if raw.size < n_total:
            raise ValueError(
                f"WAVECAR coefficient record has {raw.size} values; expected {n_total}"
            )
        coeffs = raw[:n_total].astype(np.complex128, copy=False)
        return coeffs.reshape(k_header.n_components, k_header.n_stored_g)

    def read_kpoint_wavefunction(
        self,
        spin: int,
        kpoint_index: int,
        *,
        band_start: int = 1,
        band_count: int | None = None,
    ) -> tuple[WavecarKPoint, NDArray[np.complex128]]:
        """Read the coefficients of one k-point, optionally only a band block.

        ``band_start`` is one-based and ``band_count`` defaults to all
        remaining bands.  Reading in blocks bounds the resident array at
        ``band_count`` bands instead of the whole k-point, and the weights
        computed block by block are the same as the ones computed in one pass
        (``UnfoldLab.bandWeights_flatten``).
        """

        k_header = self.read_kpoint_header(spin, kpoint_index)
        if band_start < 1 or band_start > self.header.n_bands:
            raise ValueError(f"band_start must be in 1..{self.header.n_bands}")
        remaining = self.header.n_bands - band_start + 1
        n_read = remaining if band_count is None else int(band_count)
        if n_read < 1 or n_read > remaining:
            raise ValueError(f"band_count must be in 1..{remaining} for band_start={band_start}")
        coeffs = np.zeros(
            (n_read, k_header.n_components, k_header.n_stored_g),
            dtype=np.complex128,
        )
        for offset in range(n_read):
            coeffs[offset] = self.read_coefficients(spin, kpoint_index, band_start + offset)
        if k_header.gamma_half is not None:
            g_stored = k_header.g_stored
            if g_stored is None:  # pragma: no cover - always set by the reader
                g_stored = generate_vasp_g_vectors(
                    self.header.lattice,
                    k_header.kpoint,
                    self.header.encut,
                    gamma_half=k_header.gamma_half,
                )
            _, coeffs = expand_gamma_half_basis(g_stored, coeffs)
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
        band_start: int = 1,
        band_count: int | None = None,
    ) -> PlaneWaveKPointData:
        k_header, coeffs = self.read_kpoint_wavefunction(
            spin, kpoint_index, band_start=band_start, band_count=band_count
        )
        return PlaneWaveKPointData(
            primitive_kpoint=primitive_kpoint,
            folded_supercell_kpoint=folded_supercell_kpoint,
            g_supercell=k_header.g_vectors,
            coefficients=coeffs,
        )


def read_wavecar_header(path: str | Path) -> WavecarHeader:
    with WavecarReader(path) as reader:
        return reader.header


def _fft_index_order(max_index: int) -> NDArray[np.int64]:
    """FFT index sequence ``0, 1, ..., M, -M, ..., -1`` of a wrapped axis."""

    return np.concatenate(
        [np.arange(0, max_index + 1, dtype=np.int64), np.arange(-max_index, 0, dtype=np.int64)]
    )


def _gamma_half_mask(g_vectors: NDArray[np.int64], convention: str) -> NDArray[np.bool_]:
    """Half-space selection used by a gamma-only calculation.

    One member of every pair ``{G, -G}`` is kept: the one that is positive in
    the leading axis of the convention, with ties broken by the remaining axes
    and ``G = 0`` kept.
    """

    if convention == "x":
        first, second, third = g_vectors[:, 0], g_vectors[:, 1], g_vectors[:, 2]
    elif convention == "z":
        first, second, third = g_vectors[:, 2], g_vectors[:, 1], g_vectors[:, 0]
    else:
        raise ValueError(f"gamma_half must be one of {GAMMA_HALF_CONVENTIONS}, got {convention!r}")
    return (
        (first > 0) | ((first == 0) & (second > 0)) | ((first == 0) & (second == 0) & (third >= 0))
    )


def generate_vasp_g_vectors(
    lattice: NDArray[np.float64],
    kpoint: NDArray[np.float64],
    encut: float,
    *,
    n_expected: int | None = None,
    gamma_half: str | None = None,
) -> NDArray[np.int64]:
    """Generate VASP plane-wave G-vectors in fractional reciprocal coordinates.

    The *order* matters as much as the set: WAVECAR stores the coefficients in
    the order in which VASP scans its FFT grid, namely the third axis outermost
    and the first axis innermost, each axis running over the wrapped index
    sequence ``0, 1, ..., M, -M, ..., -1``.  Sorting the list any other way
    (by kinetic energy, say) pairs every coefficient with the wrong G-vector
    and silently produces meaningless weights.  Because the surviving vectors
    keep their relative order when the scan range grows, the generated order
    does not depend on the exact grid size used here.

    ``gamma_half`` selects one member of every ``{G, -G}`` pair, as a gamma-only
    run does; the caller then expands the list with
    :func:`unfoldlab.core.gamma.expand_gamma_half_basis`.
    """

    lattice_arr = np.asarray(lattice, dtype=float)
    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice_arr).T
    reciprocal_lengths = np.linalg.norm(reciprocal, axis=1)
    max_index = np.ceil(np.sqrt(float(encut) / HSQDTM) / reciprocal_lengths).astype(int) + 1
    k_arr = np.asarray(kpoint, dtype=float)

    axis1 = _fft_index_order(int(max_index[0]))
    axis2 = _fft_index_order(int(max_index[1]))
    axis3 = _fft_index_order(int(max_index[2]))
    # ``indexing="ij"`` with the third axis first reproduces VASP's scan order:
    # n3 outermost, n1 innermost.
    grid3, grid2, grid1 = np.meshgrid(axis3, axis2, axis1, indexing="ij")
    vectors = np.stack([grid1.ravel(), grid2.ravel(), grid3.ravel()], axis=1)

    cartesian = (vectors + k_arr) @ reciprocal
    kinetic = HSQDTM * np.einsum("ij,ij->i", cartesian, cartesian)
    keep = kinetic <= float(encut) + 1e-10
    if gamma_half is not None:
        keep &= _gamma_half_mask(vectors, gamma_half)
    g_vectors = vectors[keep].astype(np.int64, copy=False)
    if g_vectors.shape[0] == 0:
        raise ValueError("no VASP G-vectors generated; check lattice, k-point, and encut")

    if n_expected is not None and g_vectors.shape[0] != n_expected:
        raise ValueError(
            f"generated {g_vectors.shape[0]} VASP G-vectors, WAVECAR expects {n_expected}; "
            "the lattice, ENCUT, or k-point passed here do not describe this file"
        )
    return g_vectors


def compute_weights_from_wavecar(
    wavecar: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    *,
    spin: int = 1,
    tol: float = 1e-6,
    gamma_half: str = "x",
    operations: NDArray[np.integer] | None = None,
    band_chunk: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(kpoints, energies, weights)`` from a VASP WAVECAR.

    ``gamma_half`` picks the half-space convention assumed for a gamma-only
    (RTAG 53300/53310) file; it is ignored for ordinary files.

    By default the WAVECAR is required to hold exactly the k-points of the
    k-map, in the same order.  Passing ``operations`` -- a sequence of integer
    ``3 x 3`` point-group matrices acting on *primitive* fractional reciprocal
    coordinates, the identity being implicit -- switches to symmetry-reduced
    mode: each requested primitive k-point is matched to whichever stored
    k-point reaches it under one of the operations, and unfolded against that
    file after the corresponding rotation of the primitive k-point.  That is
    what makes a WAVECAR written on an irreducible wedge usable for an
    arbitrary primitive band path; the identity behind it is
    ``UnfoldLab.weight_stored_representative``.  Pass an empty sequence to allow
    reordering and reuse of stored k-points without any rotation.

    ``band_chunk`` bounds how many bands of a k-point are held in memory at
    once.  The default reads a whole k-point, which costs
    ``n_bands * n_components * n_g`` complex numbers -- gigabytes for a large
    supercell.  With ``band_chunk=m`` the weights are accumulated over blocks
    of ``m`` bands, which is exact: a band's weight depends only on that band's
    coefficients (``UnfoldLab.bandWeights_flatten``).  It costs one extra file
    seek per block and nothing else.
    """

    if band_chunk is not None and band_chunk < 1:
        raise ValueError("band_chunk must be a positive number of bands")

    if operations is not None:
        return _compute_weights_from_wavecar_symmetry(
            wavecar,
            primitive_kpoints,
            transform,
            spin=spin,
            tol=tol,
            gamma_half=gamma_half,
            operations=operations,
        )

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    with WavecarReader(wavecar, gamma_half=gamma_half) as reader:
        if band_chunk is None:
            kpoints, energies, stream = _aligned_wavecar_stream(reader, primitive, folded, spin)
            weights = compute_plane_wave_unfolding_weights(stream(), transform, tol=tol)
        else:
            kpoints, energies, chunked = _chunked_wavecar_stream(
                reader, primitive, folded, spin, band_chunk
            )
            weights = compute_plane_wave_unfolding_weights_chunked(chunked(), transform, tol=tol)
    return kpoints, energies, weights


def state_norms_from_wavecar(
    wavecar: str | Path,
    *,
    spin: int = 1,
    gamma_half: str = "x",
    band_chunk: int | None = None,
) -> NDArray[np.float64]:
    """Squared norms of the stored states of a WAVECAR, shape ``(nk, nbnd)``.

    The weights are ratios and do not use these numbers
    (``UnfoldLab.weight_smul``); they diagnose how complete the file is.  For a
    PAW run they fall short of one by the augmentation charge, and
    :class:`unfoldlab.core.unfolding.StateNormDiagnostics` turns the shortfall
    into a bound on the weights.

    A gamma-only file is expanded to the full ``{G, -G}`` basis first, so the
    number reported is the norm of the physical state rather than of the stored
    half.

    ``band_chunk`` bounds the number of bands held in memory at once, as for
    :func:`compute_weights_from_wavecar`.
    """

    if band_chunk is not None and band_chunk < 1:
        raise ValueError("band_chunk must be a positive number of bands")
    with WavecarReader(wavecar, gamma_half=gamma_half) as reader:
        n_kpoints = reader.header.n_kpoints
        n_bands = reader.header.n_bands
        norms = np.zeros((n_kpoints, n_bands), dtype=float)
        step = n_bands if band_chunk is None else int(band_chunk)
        for ik in range(1, n_kpoints + 1):
            for start in range(0, n_bands, step):
                count = min(step, n_bands - start)
                _k_header, coeffs = reader.read_kpoint_wavefunction(
                    spin, ik, band_start=start + 1, band_count=count
                )
                norms[ik - 1, start : start + count] = state_norms_from_coefficients(coeffs)
    return norms


def _aligned_wavecar_stream(
    reader: WavecarReader,
    primitive: NDArray[np.float64],
    folded: NDArray[np.float64],
    spin: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], Callable[[], Iterator[PlaneWaveKPointData]]]:
    """K-points, energies and a one-at-a-time stream for an aligned WAVECAR.

    "Aligned" means the file holds exactly the k-points of the k-map, in the
    same order.  The arrays are filled as the stream is consumed.
    """

    _check_aligned(reader, primitive)
    kpoints = np.zeros((reader.header.n_kpoints, 3), dtype=float)
    energies = np.zeros((reader.header.n_kpoints, reader.header.n_bands), dtype=float)

    def stream() -> Iterator[PlaneWaveKPointData]:
        """Yield one k-point at a time so only one wavefunction is resident."""

        for ik in range(1, reader.header.n_kpoints + 1):
            k_header, coeffs = reader.read_kpoint_wavefunction(spin, ik)
            kpoints[ik - 1] = k_header.kpoint
            energies[ik - 1] = k_header.energies
            _check_kpoint_matches(ik, k_header.kpoint, folded[ik - 1])
            yield PlaneWaveKPointData(
                primitive_kpoint=primitive[ik - 1],
                folded_supercell_kpoint=k_header.kpoint,
                g_supercell=k_header.g_vectors,
                coefficients=coeffs,
            )

    return kpoints, energies, stream


def _check_aligned(reader: WavecarReader, primitive: NDArray[np.float64]) -> None:
    """Require the file to hold exactly the k-points of the k-map."""

    if primitive.shape[0] != reader.header.n_kpoints:
        raise ValueError(
            f"kmap has {primitive.shape[0]} k-points but WAVECAR has "
            f"{reader.header.n_kpoints}; pass operations=[] (the CLI's "
            "--reuse-kpoints) if one stored k-point is meant to serve "
            "several k-map rows, as it does for a fiber k-map"
        )


def _check_kpoint_matches(
    ik: int,
    stored: NDArray[np.float64],
    expected: NDArray[np.float64],
) -> None:
    """Require the stored k-point and the k-map row to agree modulo ``ℤ³``.

    The G-vector list is generated for the k-point representative stored in
    the WAVECAR, and the matching test depends on the physical vectors
    ``K + G`` rather than on ``G`` alone (``UnfoldLab.pwMatches_of_add_eq``).
    Using the k-map representative instead would select a shifted set of plane
    waves whenever the two differ by a reciprocal lattice vector, so the file's
    own k-point is used and the two are only required to agree modulo one.
    """

    if not allclose_mod1(stored, expected):
        raise ValueError(
            f"WAVECAR k-point {ik} is {stored.tolist()} but the "
            f"k-map expects {expected.tolist()} (mod 1); the k-path "
            "and the WAVECAR are not the same calculation or are ordered "
            "differently"
        )


def _chunked_wavecar_stream(
    reader: WavecarReader,
    primitive: NDArray[np.float64],
    folded: NDArray[np.float64],
    spin: int,
    band_chunk: int,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    Callable[[], Iterator[Iterator[PlaneWaveKPointData]]],
]:
    """Like :func:`_aligned_wavecar_stream`, but band block by band block.

    The outer stream yields one k-point at a time and the inner stream yields
    consecutive blocks of at most ``band_chunk`` bands, so at most one block is
    ever resident.
    """

    _check_aligned(reader, primitive)
    kpoints = np.zeros((reader.header.n_kpoints, 3), dtype=float)
    energies = np.zeros((reader.header.n_kpoints, reader.header.n_bands), dtype=float)
    n_bands = reader.header.n_bands

    def blocks(ik: int, k_header: WavecarKPoint) -> Iterator[PlaneWaveKPointData]:
        for start in range(1, n_bands + 1, band_chunk):
            count = min(band_chunk, n_bands - start + 1)
            _, coeffs = reader.read_kpoint_wavefunction(
                spin, ik, band_start=start, band_count=count
            )
            yield PlaneWaveKPointData(
                primitive_kpoint=primitive[ik - 1],
                folded_supercell_kpoint=k_header.kpoint,
                g_supercell=k_header.g_vectors,
                coefficients=coeffs,
            )

    def stream() -> Iterator[Iterator[PlaneWaveKPointData]]:
        for ik in range(1, reader.header.n_kpoints + 1):
            k_header = reader.read_kpoint_header(spin, ik)
            kpoints[ik - 1] = k_header.kpoint
            energies[ik - 1] = k_header.energies
            _check_kpoint_matches(ik, k_header.kpoint, folded[ik - 1])
            yield blocks(ik, k_header)

    return kpoints, energies, stream


def compute_spin_texture_from_wavecar(
    wavecar: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    *,
    spin: int = 1,
    tol: float = 1e-6,
    gamma_half: str = "x",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(kpoints, energies, weights, textures)`` for a noncollinear WAVECAR.

    ``textures`` has shape ``(n_kpoints, n_bands, 3)`` and holds the unfolded
    spin expectation value at each primitive k-point; ``weights`` is the
    ordinary unfolding weight, and ``|texture| <= weight`` holds state by state
    (``UnfoldLab.norm_spinTexture_le_pairWeight``).

    The file must be a spinor calculation: a collinear or spin-polarized
    WAVECAR stores one component per G-vector and carries no transverse spin.

    There is deliberately no symmetry-reduced mode.  Weights are invariant
    under a point-group operation, which is what makes unfolding from an
    irreducible wedge legitimate, but the spin is a pseudovector: at ``S k``
    the texture is the *rotated* stored texture, and for a spinor wavefunction
    the rotation is fixed only up to the choice of SU(2) lift and is reversed
    by time reversal.  Reusing a stored texture unrotated would silently point
    the spin the wrong way, so the caller must supply an explicit wavefunction
    for every k-point of the path.
    """

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    weight_rows: list[NDArray[np.float64]] = []
    texture_rows: list[NDArray[np.float64]] = []
    with WavecarReader(wavecar, gamma_half=gamma_half) as reader:
        kpoints, energies, stream = _aligned_wavecar_stream(reader, primitive, folded, spin)
        # One pass over the file: each wavefunction is large, so weights and
        # textures are taken from the same resident copy.
        for item in stream():
            weight_rows.append(
                weights_from_coefficients(
                    item.g_supercell,
                    item.coefficients,
                    item.primitive_kpoint,
                    item.folded_supercell_kpoint,
                    transform,
                    tol=tol,
                )
            )
            texture_rows.append(
                spin_texture_from_coefficients(
                    item.g_supercell,
                    item.coefficients,
                    item.primitive_kpoint,
                    item.folded_supercell_kpoint,
                    transform,
                    tol=tol,
                )
            )
    return kpoints, energies, np.stack(weight_rows), np.stack(texture_rows)


def _compute_weights_from_wavecar_symmetry(
    wavecar: str | Path,
    primitive_kpoints: NDArray[np.float64],
    transform: TransformLike,
    *,
    spin: int,
    tol: float,
    gamma_half: str,
    operations: NDArray[np.integer],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Symmetry-reduced variant of :func:`compute_weights_from_wavecar`.

    The returned ``kpoints`` are the stored representatives actually used, one
    row per requested primitive k-point, and the energies are those of the
    stored k-point -- which are the energies of the whole star.
    """

    primitive = np.asarray(primitive_kpoints, dtype=float)
    with WavecarReader(wavecar, gamma_half=gamma_half) as reader:
        stored_kpoints, stored_energies = reader.read_all_energies(spin)
        matches = map_kpoints_to_stored(
            primitive,
            stored_kpoints,
            transform,
            operations,
            atol=tol,
        )
        kpoints = np.zeros_like(primitive)
        energies = np.zeros((primitive.shape[0], reader.header.n_bands), dtype=float)
        for row, match in enumerate(matches):
            kpoints[row] = stored_kpoints[match.index]
            energies[row] = stored_energies[match.index]

        # A band path typically revisits the same irreducible k-point many
        # times.  Grouping the requested rows by stored k-point reads every
        # wavefunction exactly once *and* keeps only one of them resident, which
        # an index-keyed cache would not: the cache grows to the whole
        # irreducible wedge.
        groups: dict[int, list[int]] = defaultdict(list)
        for row, match in enumerate(matches):
            groups[match.index].append(row)
        order = sorted(groups)
        grouped_rows = [row for index in order for row in groups[index]]

        # Every row of a group shares the wavefunction, so the squared moduli
        # are formed once per stored k-point rather than once per path point.
        def stream() -> Iterator[SharedWavefunctionGroup]:
            for index in order:
                k_header, coeffs = reader.read_kpoint_wavefunction(spin, index + 1)
                yield SharedWavefunctionGroup(
                    primitive_kpoints=np.stack(
                        [matches[row].effective_primitive_kpoint for row in groups[index]]
                    ),
                    folded_supercell_kpoint=stored_kpoints[index],
                    g_supercell=k_header.g_vectors,
                    coefficients=coeffs,
                )

        grouped = compute_plane_wave_unfolding_weights_shared(stream(), transform, tol=tol)
        weights = np.empty_like(grouped)
        weights[np.asarray(grouped_rows, dtype=np.int64)] = grouped
    return kpoints, energies, weights
