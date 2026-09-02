"""Quantum ESPRESSO wavefunction *files*: records, headers and readers.

This is the file-format layer of the QE backend -- Fortran record framing,
binary headers, HDF5 datasets, the k-point a file records, and the readers that
turn one file into ``(g_vectors, coefficients)``.  The unfolding computations
built on top live in :mod:`unfoldlab.io.qe_wfc`, which re-exports everything
here so that the historical import path keeps working.
"""

from __future__ import annotations

import re
import struct
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.gamma import expand_gamma_half_basis
from unfoldlab.core.numerics import as_matrix3

#: A reader of a wavefunction file's own k-point, in Cartesian ``2 pi / alat``
#: units.  ``None`` means the file does not record one.
XkReader = Callable[["str | Path"], "NDArray[np.float64] | None"]


@dataclass(frozen=True)
class WfcInfo:
    path: Path
    ik: int
    ispin: int


@dataclass(frozen=True)
class SymmetryReducedWeights:
    """Result of unfolding against a symmetry-reduced wavefunction set.

    ``weights`` has one row per *requested* primitive k-point.  ``stored_indices``
    says which wavefunction file served each row (a 0-based index into the
    sorted file list, i.e. into a per-file energy table), ``stored_kpoints``
    holds the fractional supercell k-point of that file, and ``n_stored`` is the
    number of files available.
    """

    weights: NDArray[np.float64]
    stored_indices: NDArray[np.int64]
    stored_kpoints: NDArray[np.float64]
    n_stored: int


@dataclass(frozen=True)
class BinaryHeader:
    ik: int
    ispin: int
    #: k-point of this file, Cartesian, in units of ``2*pi/alat``.
    xk: tuple[float, float, float]
    gamma_only: bool
    ngw: int
    igwx: int
    npol: int
    nbnd: int
    endian: str
    marker_size: int


class FortranRecordReader:
    """Read unformatted Fortran records with 4- or 8-byte markers."""

    def __init__(self, path: Path, endian: str = "<", marker_size: int = 4):
        self.handle = path.open("rb")
        self.path = path
        self.endian = endian
        self.marker_size = marker_size
        self.marker_fmt = endian + ("i" if marker_size == 4 else "q")

    def close(self) -> None:
        self.handle.close()

    def __enter__(self) -> FortranRecordReader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def read_record(self) -> bytes:
        head = self.handle.read(self.marker_size)
        if not head:
            raise EOFError(f"unexpected end of file in {self.path}")
        if len(head) != self.marker_size:
            raise ValueError(f"truncated Fortran record marker in {self.path}")
        (length,) = struct.unpack(self.marker_fmt, head)
        if length <= 0 or length > 2_000_000_000:
            raise ValueError(f"invalid Fortran record length {length} in {self.path}")
        payload = self.handle.read(length)
        tail = self.handle.read(self.marker_size)
        if len(payload) != length or len(tail) != self.marker_size:
            raise ValueError(f"truncated Fortran record in {self.path}")
        (tail_length,) = struct.unpack(self.marker_fmt, tail)
        if tail_length != length:
            raise ValueError(
                f"Fortran record marker mismatch in {self.path}: {length} != {tail_length}"
            )
        return payload


def open_fortran_reader(path: str | Path) -> FortranRecordReader:
    path = Path(path)
    with path.open("rb") as handle:
        first = handle.read(8)
    for endian in ("<", ">"):
        for marker_size, fmt in ((4, "i"), (8, "q")):
            if len(first) < marker_size:
                continue
            (length,) = struct.unpack(endian + fmt, first[:marker_size])
            if length in {44, 48} or 32 <= length <= 128:
                return FortranRecordReader(path, endian=endian, marker_size=marker_size)
    raise ValueError(f"could not detect Fortran record markers for {path}")


def import_h5py() -> Any:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "reading QE HDF5 wavefunctions requires h5py; install unfoldlab[io] "
            "or export a coefficient/weight table with another tool"
        ) from exc
    return h5py


def attr_scalar(value: Any, default: int | float | str | None = None) -> Any:
    if value is None:
        return default
    arr = np.asarray(value)
    item = arr.item() if arr.shape == () else arr.flat[0]
    if isinstance(item, bytes):
        return item.decode().strip()
    return item


def gamma_only_value(value: Any) -> bool:
    text = str(attr_scalar(value, ".FALSE.")).strip().upper()
    return text in {"T", "TRUE", ".TRUE.", "1"}


def parse_ik_from_name(path: str | Path) -> int:
    match = re.search(r"(\d+)(?=\.(?:hdf5|h5)$)", Path(path).name, re.IGNORECASE)
    if match is None:
        raise ValueError(f"could not infer k-point index from {Path(path).name}")
    return int(match.group(1))


def parse_binary_header(path: str | Path) -> BinaryHeader:
    path = Path(path)
    with open_fortran_reader(path) as reader:
        rec1 = reader.read_record()
        if len(rec1) == 44:
            ik, x1, x2, x3, ispin, gamma_i, _scalef = struct.unpack(reader.endian + "i3diid", rec1)
        elif len(rec1) == 48:
            ik, x1, x2, x3, ispin, gamma_i, _pad, _scalef = struct.unpack(
                reader.endian + "i3diiid", rec1
            )
        else:
            raise ValueError(f"unsupported first-record size {len(rec1)} in {path}")
        rec2 = reader.read_record()
        if len(rec2) != 16:
            raise ValueError(f"unsupported second-record size {len(rec2)} in {path}")
        ngw, igwx, npol, nbnd = struct.unpack(reader.endian + "4i", rec2)
    return BinaryHeader(
        ik=int(ik),
        ispin=int(ispin),
        xk=(float(x1), float(x2), float(x3)),
        gamma_only=bool(gamma_i),
        ngw=int(ngw),
        igwx=int(igwx),
        npol=int(npol),
        nbnd=int(nbnd),
        endian=reader.endian,
        marker_size=reader.marker_size,
    )


def fractional_kpoint_from_xk(
    xk: NDArray[np.float64] | tuple[float, float, float],
    lattice_alat: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a QE ``xk`` to fractional reciprocal coordinates.

    QE stores ``xk`` in Cartesian coordinates in units of ``2*pi/alat``.  With
    ``at`` the direct lattice in units of ``alat`` (QE's own ``at`` matrix, rows
    = lattice vectors, i.e. ``A / alat`` for a lattice ``A`` in any length unit)
    the fractional coordinates are ``xk @ at.T``.
    """

    matrix = as_matrix3(lattice_alat, name="lattice_alat")
    return np.asarray(xk, dtype=float) @ matrix.T


def read_xk_dat(path: str | Path) -> NDArray[np.float64]:
    """Cartesian ``xk`` (units of ``2*pi/alat``) stored in a QE ``wfc*.dat``."""

    return np.asarray(parse_binary_header(path).xk, dtype=float)


def read_xk_hdf5(path: str | Path) -> NDArray[np.float64] | None:
    """Cartesian ``xk`` of a QE ``wfc*.hdf5``, or ``None`` if not recorded."""

    h5py = import_h5py()
    with h5py.File(Path(path), "r") as handle:
        raw = handle.attrs.get("xk")
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != 3:
        return None
    return arr


def find_dataset(group: Any, aliases: set[str]) -> Any | None:
    wanted = {name.lower() for name in aliases}
    found = None

    def visitor(name: str, obj: Any) -> None:
        nonlocal found
        if found is None and hasattr(obj, "shape") and name.split("/")[-1].lower() in wanted:
            found = obj

    group.visititems(visitor)
    return found


def list_wfc_hdf5_files(
    save_dir: str | Path,
    *,
    pattern: str = "wfc*.hdf5",
    spin: int | None = None,
) -> list[WfcInfo]:
    h5py = import_h5py()
    save_path = Path(save_dir)
    files = sorted(save_path.glob(pattern))
    if not files and pattern == "wfc*.hdf5":
        files = sorted(save_path.glob("wfc*.h5"))
    if not files:
        raise FileNotFoundError(
            f"no QE HDF5 wavefunction files matching {pattern!r} found in {save_path}"
        )

    infos: list[WfcInfo] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            ik = int(attr_scalar(handle.attrs.get("ik"), parse_ik_from_name(path)))
            ispin = int(attr_scalar(handle.attrs.get("ispin"), 1))
        if spin is None or ispin == spin:
            infos.append(WfcInfo(path=path, ik=ik, ispin=ispin))
    return _order_unique_wfc_infos(infos, spin=spin, kind="HDF5")


def list_wfc_dat_files(
    save_dir: str | Path,
    *,
    pattern: str = "wfc*.dat",
    spin: int | None = None,
) -> list[WfcInfo]:
    save_path = Path(save_dir)
    files = sorted(save_path.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"no QE binary wavefunction files matching {pattern!r} found in {save_path}"
        )
    infos = []
    for path in files:
        header = parse_binary_header(path)
        if spin is None or header.ispin == spin:
            infos.append(WfcInfo(path=path, ik=header.ik, ispin=header.ispin))
    return _order_unique_wfc_infos(infos, spin=spin, kind="binary")


def _order_unique_wfc_infos(
    infos: list[WfcInfo],
    *,
    spin: int | None,
    kind: str,
) -> list[WfcInfo]:
    if not infos:
        raise ValueError(f"no {kind} wavefunction files left after applying spin={spin}")
    seen: dict[int, WfcInfo] = {}
    for info in infos:
        if info.ik in seen:
            raise ValueError(
                f"multiple {kind} wavefunction files share k-point index {info.ik}; "
                "for spin-polarized calculations pass --spin 1 or --spin 2"
            )
        seen[info.ik] = info
    return [seen[idx] for idx in sorted(seen)]


def evc_to_complex(evc: NDArray[Any]) -> NDArray[np.complex128]:
    arr = np.asarray(evc)
    if np.iscomplexobj(arr):
        return arr.astype(np.complex128, copy=False)
    if arr.ndim == 2 and arr.shape[1] % 2 == 0:
        return (arr[:, 0::2] + 1j * arr[:, 1::2]).astype(np.complex128)
    if arr.ndim == 3 and arr.shape[-1] == 2:
        return (arr[..., 0] + 1j * arr[..., 1]).astype(np.complex128)
    raise ValueError(
        "unsupported QE evc dataset shape; expected complex data, "
        "(nbnd, 2*npw), or (..., 2) real/imag storage"
    )


def coefficients_by_component(
    coeffs: NDArray[np.complexfloating],
    n_g: int,
    n_bands: int,
    npol: int,
) -> NDArray[np.complex128]:
    if coeffs.shape[0] < n_bands:
        raise ValueError(
            f"wavefunction file contains {coeffs.shape[0]} bands but {n_bands} are required"
        )
    selected = coeffs[:n_bands]
    n_coeff = selected.shape[1]
    if n_coeff == n_g:
        return selected[:, np.newaxis, :].astype(np.complex128, copy=False)
    if npol > 1 and n_coeff >= n_g * npol:
        return np.asarray(
            selected[:, : n_g * npol].reshape(n_bands, npol, n_g), dtype=np.complex128
        )
    if n_coeff % n_g == 0:
        n_comp = n_coeff // n_g
        return np.asarray(
            selected[:, : n_g * n_comp].reshape(n_bands, n_comp, n_g), dtype=np.complex128
        )
    raise ValueError(f"cannot map {n_coeff} complex coefficients onto {n_g} G-vectors")


def band_block(n_bands: int, band_start: int, band_count: int | None) -> tuple[int, int]:
    """Validate a one-based band block and return it as a ``[start, stop)`` pair.

    Reading a k-point in band blocks keeps the resident array at ``band_count``
    bands; the weights are unaffected (``UnfoldLab.bandWeights_flatten``).
    """

    if band_start < 1 or band_start > n_bands:
        raise ValueError(f"band_start must be in 1..{n_bands}")
    remaining = n_bands - band_start + 1
    count = remaining if band_count is None else int(band_count)
    if count < 1 or count > remaining:
        raise ValueError(f"band_count must be in 1..{remaining} for band_start={band_start}")
    return band_start - 1, band_start - 1 + count


def read_wfc_hdf5(
    path: str | Path,
    n_bands: int,
    *,
    band_start: int = 1,
    band_count: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Miller indices and coefficients of one QE HDF5 wavefunction.

    ``band_start`` (one-based) and ``band_count`` select a block of the first
    ``n_bands`` bands; only that block is read out of the ``evc`` dataset.
    """

    h5py = import_h5py()
    path = Path(path)
    start, stop = band_block(n_bands, band_start, band_count)
    with h5py.File(path, "r") as handle:
        gamma_only = gamma_only_value(handle.attrs.get("gamma_only"))
        miller_dataset = find_dataset(handle, {"MillerIndices", "Miller_Indices"})
        evc_dataset = find_dataset(handle, {"evc"})
        if miller_dataset is None or evc_dataset is None:
            raise ValueError(f"{path} does not contain both MillerIndices and evc datasets")
        miller = np.asarray(miller_dataset, dtype=int)
        stored_bands = int(np.asarray(evc_dataset.shape)[0])
        if stored_bands < n_bands:
            raise ValueError(
                f"wavefunction file contains {stored_bands} bands but {n_bands} are required"
            )
        # Slice the dataset itself: an HDF5 read of a band block never
        # materializes the whole k-point.
        coeffs = evc_to_complex(np.asarray(evc_dataset[start:stop]))
        npol = int(attr_scalar(handle.attrs.get("npol"), 1))

    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError(f"MillerIndices in {path} must have shape (npw, 3)")
    n_g = min(miller.shape[0], coeffs.shape[-1])
    g_vectors = miller[:n_g]
    components = coefficients_by_component(coeffs, n_g, stop - start, npol)
    if gamma_only:
        # Only half of each pair {G, -G} is stored; the partner coefficient is
        # the complex conjugate.  The two members belong to different primitive
        # k-points of the fiber, so the basis must be expanded before matching.
        return expand_gamma_half_basis(g_vectors, components)
    return g_vectors, components


def read_wfc_dat(
    path: str | Path,
    n_bands: int,
    *,
    band_start: int = 1,
    band_count: int | None = None,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    """Miller indices and coefficients of one QE Fortran ``.dat`` wavefunction.

    ``band_start`` (one-based) and ``band_count`` select a block of the first
    ``n_bands`` bands.  The records of the skipped bands still have to be
    walked past -- the format is sequential -- but they are not kept.
    """

    path = Path(path)
    header = parse_binary_header(path)
    if n_bands > header.nbnd:
        raise ValueError(f"{path} contains {header.nbnd} bands but {n_bands} are required")
    start, stop = band_block(n_bands, band_start, band_count)

    with FortranRecordReader(path, header.endian, header.marker_size) as reader:
        reader.read_record()
        reader.read_record()
        reader.read_record()
        miller_record = reader.read_record()
        expected_miller_bytes = 3 * header.igwx * 4
        if len(miller_record) != expected_miller_bytes:
            raise ValueError(
                f"Miller-index record in {path} has {len(miller_record)} bytes; "
                f"expected {expected_miller_bytes}"
            )
        miller_raw = np.frombuffer(miller_record, dtype=np.dtype(header.endian + "i4"))
        miller = miller_raw.reshape((3, header.igwx), order="F").T

        coeffs = np.zeros((stop - start, header.npol, header.igwx), dtype=np.complex128)
        expected_evc_bytes = header.npol * header.igwx * 16
        complex_dtype = np.dtype(header.endian + "c16")
        for ib in range(stop):
            evc_record = reader.read_record()
            if len(evc_record) != expected_evc_bytes:
                raise ValueError(
                    f"band {ib + 1} in {path} has {len(evc_record)} bytes; "
                    f"expected {expected_evc_bytes}"
                )
            if ib < start:
                continue
            evc = np.frombuffer(evc_record, dtype=complex_dtype)
            coeffs[ib - start] = evc.reshape((header.npol, header.igwx))
    g_vectors = miller[: header.igwx]
    components = coeffs[:, :, : header.igwx]
    if header.gamma_only:
        # See ``read_wfc_hdf5``: expand the time-reversal reduced half basis.
        return expand_gamma_half_basis(g_vectors, components)
    return g_vectors, components
