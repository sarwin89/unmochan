"""Quantum ESPRESSO wavefunction readers for plane-wave unfolding."""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.unfolding import PlaneWaveKPointData, compute_plane_wave_unfolding_weights


@dataclass(frozen=True)
class WfcInfo:
    path: Path
    ik: int
    ispin: int


@dataclass(frozen=True)
class BinaryHeader:
    ik: int
    ispin: int
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
            ik, _x1, _x2, _x3, ispin, gamma_i, _scalef = struct.unpack(
                reader.endian + "i3diid", rec1
            )
        elif len(rec1) == 48:
            ik, _x1, _x2, _x3, ispin, gamma_i, _pad, _scalef = struct.unpack(
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
        gamma_only=bool(gamma_i),
        ngw=int(ngw),
        igwx=int(igwx),
        npol=int(npol),
        nbnd=int(nbnd),
        endian=reader.endian,
        marker_size=reader.marker_size,
    )


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
        return selected[:, : n_g * npol].reshape(n_bands, npol, n_g)
    if n_coeff % n_g == 0:
        n_comp = n_coeff // n_g
        return selected[:, : n_g * n_comp].reshape(n_bands, n_comp, n_g)
    raise ValueError(
        f"cannot map {n_coeff} complex coefficients onto {n_g} G-vectors"
    )


def read_wfc_hdf5(
    path: str | Path,
    n_bands: int,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    h5py = import_h5py()
    path = Path(path)
    with h5py.File(path, "r") as handle:
        if gamma_only_value(handle.attrs.get("gamma_only")):
            raise NotImplementedError(
                "gamma-only QE HDF5 wavefunctions are not supported for unfolding; "
                "run a regular k-point bands calculation"
            )
        miller_dataset = find_dataset(handle, {"MillerIndices", "Miller_Indices"})
        evc_dataset = find_dataset(handle, {"evc"})
        if miller_dataset is None or evc_dataset is None:
            raise ValueError(f"{path} does not contain both MillerIndices and evc datasets")
        miller = np.asarray(miller_dataset, dtype=int)
        coeffs = evc_to_complex(np.asarray(evc_dataset))
        npol = int(attr_scalar(handle.attrs.get("npol"), 1))

    if miller.ndim != 2 or miller.shape[1] != 3:
        raise ValueError(f"MillerIndices in {path} must have shape (npw, 3)")
    n_g = min(miller.shape[0], coeffs.shape[-1])
    return miller[:n_g], coefficients_by_component(coeffs, n_g, n_bands, npol)


def read_wfc_dat(
    path: str | Path,
    n_bands: int,
) -> tuple[NDArray[np.int64], NDArray[np.complex128]]:
    path = Path(path)
    header = parse_binary_header(path)
    if header.gamma_only:
        raise NotImplementedError(
            "gamma-only QE binary wavefunctions are not supported for unfolding; "
            "run a regular k-point bands calculation"
        )
    if n_bands > header.nbnd:
        raise ValueError(f"{path} contains {header.nbnd} bands but {n_bands} are required")

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

        coeffs = np.zeros((n_bands, header.npol, header.igwx), dtype=np.complex128)
        expected_evc_bytes = header.npol * header.igwx * 16
        complex_dtype = np.dtype(header.endian + "c16")
        for ib in range(n_bands):
            evc_record = reader.read_record()
            if len(evc_record) != expected_evc_bytes:
                raise ValueError(
                    f"band {ib + 1} in {path} has {len(evc_record)} bytes; "
                    f"expected {expected_evc_bytes}"
                )
            evc = np.frombuffer(evc_record, dtype=complex_dtype)
            coeffs[ib] = evc.reshape((header.npol, header.igwx))
    return miller[: header.igwx], coeffs[:, :, : header.igwx]


def compute_weights_from_qe_hdf5(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: NDArray[np.float64],
    n_bands: int,
    *,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str = "wfc*.hdf5",
) -> NDArray[np.float64]:
    save_path = Path(save_dir)
    if not save_path.is_dir():
        raise NotADirectoryError(f"QE save directory not found: {save_path}")
    infos = list_wfc_hdf5_files(save_path, pattern=pattern, spin=spin)
    return _compute_weights_from_infos(
        infos,
        read_wfc_hdf5,
        primitive_kpoints,
        folded_supercell_kpoints,
        transform,
        n_bands,
        tol=tol,
    )


def compute_weights_from_qe_save(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: NDArray[np.float64],
    n_bands: int,
    *,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str | None = None,
    file_format: str = "auto",
) -> NDArray[np.float64]:
    save_path = Path(save_dir)
    if file_format not in {"auto", "hdf5", "dat"}:
        raise ValueError("file_format must be one of: auto, hdf5, dat")

    use_format = file_format
    if file_format == "auto":
        if list(save_path.glob(pattern or "wfc*.hdf5")) or list(save_path.glob("wfc*.h5")):
            use_format = "hdf5"
        elif list(save_path.glob(pattern or "wfc*.dat")):
            use_format = "dat"
        else:
            raise FileNotFoundError(
                f"no QE wfc*.hdf5, wfc*.h5, or wfc*.dat files found in {save_path}"
            )

    if use_format == "hdf5":
        return compute_weights_from_qe_hdf5(
            save_path,
            primitive_kpoints,
            folded_supercell_kpoints,
            transform,
            n_bands,
            tol=tol,
            spin=spin,
            pattern=pattern or "wfc*.hdf5",
        )

    infos = list_wfc_dat_files(save_path, pattern=pattern or "wfc*.dat", spin=spin)
    return _compute_weights_from_infos(
        infos,
        read_wfc_dat,
        primitive_kpoints,
        folded_supercell_kpoints,
        transform,
        n_bands,
        tol=tol,
    )


def _compute_weights_from_infos(
    infos: list[WfcInfo],
    reader: Any,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: NDArray[np.float64],
    n_bands: int,
    *,
    tol: float,
) -> NDArray[np.float64]:
    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    if len(infos) != len(primitive):
        raise ValueError(
            f"found {len(infos)} wavefunction files but kmap has {len(primitive)} k-points"
        )

    weights = np.zeros((len(primitive), n_bands), dtype=float)
    for out_idx, info in enumerate(infos):
        expected_ik = out_idx + 1
        if info.ik != expected_ik:
            raise ValueError(
                f"wavefunction k-point order mismatch: expected ik={expected_ik}, "
                f"found ik={info.ik} in {info.path.name}"
            )
        g_supercell, coefficients = reader(info.path, n_bands)
        weights[out_idx, :] = compute_plane_wave_unfolding_weights(
            [
                PlaneWaveKPointData(
                    primitive_kpoint=primitive[out_idx],
                    folded_supercell_kpoint=folded[out_idx],
                    g_supercell=g_supercell,
                    coefficients=coefficients,
                )
            ],
            transform,
            tol=tol,
        )[0]
    return weights
