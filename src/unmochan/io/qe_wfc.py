"""Plane-wave unfolding from Quantum ESPRESSO wavefunctions.

The file-format layer -- Fortran records, binary headers, HDF5 datasets and the
per-file readers -- lives in :mod:`unmochan.io.qe_wfc_files` and is re-exported
here, so ``from unmochan.io.qe_wfc import read_wfc_dat`` keeps working.  What
remains in this module is the physics: turning stored coefficients into
unfolding weights, spin textures and state norms.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from unmochan.core.numerics import TransformLike, allclose_mod1
from unmochan.core.plane_waves import (
    spin_texture_from_coefficients,
    state_norms_from_coefficients,
)
from unmochan.core.symmetry import map_kpoints_to_stored
from unmochan.core.unfolding import (
    PlaneWaveKPointData,
    SharedWavefunctionGroup,
    compute_plane_wave_unfolding_weights,
    compute_plane_wave_unfolding_weights_chunked,
    compute_plane_wave_unfolding_weights_shared,
)
from unmochan.io.qe_wfc_files import (
    BinaryHeader,
    FortranRecordReader,
    SymmetryReducedWeights,
    WfcInfo,
    XkReader,
    attr_scalar,
    band_block,
    coefficients_by_component,
    evc_to_complex,
    find_dataset,
    fractional_kpoint_from_xk,
    gamma_only_value,
    import_h5py,
    list_wfc_dat_files,
    list_wfc_hdf5_files,
    open_fortran_reader,
    parse_binary_header,
    parse_ik_from_name,
    read_wfc_dat,
    read_wfc_hdf5,
    read_xk_dat,
    read_xk_hdf5,
)
from unmochan.io.qe_xml import (
    QESaveMetadata,
    try_read_qe_metadata,
    warn_about_pseudopotentials,
)

__all__ = [
    "BinaryHeader",
    "FortranRecordReader",
    "SymmetryReducedWeights",
    "WfcInfo",
    "XkReader",
    "attr_scalar",
    "band_block",
    "coefficients_by_component",
    "compute_spin_texture_from_qe_save",
    "compute_symmetry_weights_from_qe_save",
    "compute_weights_from_qe_hdf5",
    "compute_weights_from_qe_save",
    "evc_to_complex",
    "find_dataset",
    "fractional_kpoint_from_xk",
    "gamma_only_value",
    "import_h5py",
    "list_wfc_dat_files",
    "list_wfc_hdf5_files",
    "open_fortran_reader",
    "parse_binary_header",
    "parse_ik_from_name",
    "read_wfc_dat",
    "read_wfc_hdf5",
    "read_xk_dat",
    "read_xk_hdf5",
    "resolve_qe_metadata",
    "resolve_wfc_format",
    "state_norms_from_qe_save",
]


def compute_weights_from_qe_hdf5(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str = "wfc*.hdf5",
    lattice_alat: NDArray[np.float64] | None = None,
    operations: NDArray[np.integer] | None = None,
    use_xml_metadata: bool = True,
    band_chunk: int | None = None,
) -> NDArray[np.float64]:
    save_path = Path(save_dir)
    if not save_path.is_dir():
        raise NotADirectoryError(f"QE save directory not found: {save_path}")
    infos = list_wfc_hdf5_files(save_path, pattern=pattern, spin=spin)
    lattice_alat, metadata = resolve_qe_metadata(
        save_path, lattice_alat, use_xml_metadata=use_xml_metadata
    )
    xk_hdf5 = _xk_reader_with_xml(read_xk_hdf5, infos, metadata)
    if operations is not None:
        return _compute_symmetry_weights_from_infos(
            infos,
            read_wfc_hdf5,
            xk_hdf5,
            primitive_kpoints,
            transform,
            n_bands,
            tol=tol,
            lattice_alat=lattice_alat,
            operations=operations,
        ).weights
    return _compute_weights_from_infos(
        infos,
        read_wfc_hdf5,
        primitive_kpoints,
        folded_supercell_kpoints,
        transform,
        n_bands,
        tol=tol,
        xk_reader=xk_hdf5,
        lattice_alat=lattice_alat,
        band_chunk=band_chunk,
    )[0]


def compute_weights_from_qe_save(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str | None = None,
    file_format: str = "auto",
    lattice_alat: NDArray[np.float64] | None = None,
    operations: NDArray[np.integer] | None = None,
    use_xml_metadata: bool = True,
    band_chunk: int | None = None,
) -> NDArray[np.float64]:
    """Unfolding weights from a QE ``.save`` directory.

    ``lattice_alat`` is the supercell direct lattice in units of ``alat`` (QE's
    ``at`` matrix).  When it is supplied the k-point recorded in each
    wavefunction file is converted to fractional coordinates and cross-checked
    against ``folded_supercell_kpoints`` modulo a reciprocal lattice vector, and
    the file's own representative is then used for the plane-wave matching --
    the Miller indices are indexed against that representative, so using a
    different one would select a shifted set of plane waves.  Without it no
    such check is possible and the caller is trusted to supply k-points in the
    same order and gauge as the wavefunction files.  When ``use_xml_metadata``
    is left on and the directory contains a ``data-file-schema.xml``, ``at`` is
    read from it, so the check is on by default and the caller no longer has to
    supply the matrix (``UnfoldLab.xkToFrac_of_reciprocal``); the XML k-point
    list is also used to check the file order and, for HDF5 files that do not
    record ``xk``, to supply it.

    ``operations`` switches to symmetry-reduced mode; see
    :func:`compute_symmetry_weights_from_qe_save`.

    ``band_chunk`` reads each wavefunction in blocks of that many bands rather
    than whole, which bounds the memory of a large supercell without changing
    the weights (``UnfoldLab.bandWeights_flatten``).
    """

    save_path = Path(save_dir)
    use_format = resolve_wfc_format(save_path, pattern=pattern, file_format=file_format)

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
            lattice_alat=lattice_alat,
            operations=operations,
            use_xml_metadata=use_xml_metadata,
            band_chunk=band_chunk,
        )

    infos = list_wfc_dat_files(save_path, pattern=pattern or "wfc*.dat", spin=spin)
    lattice_alat, metadata = resolve_qe_metadata(
        save_path, lattice_alat, use_xml_metadata=use_xml_metadata
    )
    xk_dat = _xk_reader_with_xml(read_xk_dat, infos, metadata)
    if operations is not None:
        return _compute_symmetry_weights_from_infos(
            infos,
            read_wfc_dat,
            xk_dat,
            primitive_kpoints,
            transform,
            n_bands,
            tol=tol,
            lattice_alat=lattice_alat,
            operations=operations,
        ).weights
    return _compute_weights_from_infos(
        infos,
        read_wfc_dat,
        primitive_kpoints,
        folded_supercell_kpoints,
        transform,
        n_bands,
        tol=tol,
        xk_reader=xk_dat,
        lattice_alat=lattice_alat,
        band_chunk=band_chunk,
    )[0]


def state_norms_from_qe_save(
    save_dir: str | Path,
    n_bands: int,
    *,
    spin: int | None = None,
    pattern: str | None = None,
    file_format: str = "auto",
    band_chunk: int | None = None,
) -> NDArray[np.float64]:
    """Squared norms of the stored QE states, shape ``(n_kpoints, n_bands)``.

    The k-points are the wavefunction files of the ``.save`` directory in their
    own order; no k-map is needed, because the norm of a state does not depend
    on the unfolding at all.  See
    :func:`unmochan.io.vasp_wfc.state_norms_from_wavecar` for what the numbers
    mean; a gamma-only file is expanded to the full ``{G, -G}`` basis first.
    """

    if band_chunk is not None and band_chunk < 1:
        raise ValueError("band_chunk must be a positive number of bands")
    save_path = Path(save_dir)
    use_format = resolve_wfc_format(save_path, pattern=pattern, file_format=file_format)
    if use_format == "hdf5":
        infos = list_wfc_hdf5_files(save_path, pattern=pattern or "wfc*.hdf5", spin=spin)
        reader = read_wfc_hdf5
    else:
        infos = list_wfc_dat_files(save_path, pattern=pattern or "wfc*.dat", spin=spin)
        reader = read_wfc_dat

    norms = np.zeros((len(infos), n_bands), dtype=float)
    step = n_bands if band_chunk is None else int(band_chunk)
    for out_idx, info in enumerate(infos):
        for start in range(0, n_bands, step):
            count = min(step, n_bands - start)
            _g, coefficients = reader(info.path, n_bands, band_start=start + 1, band_count=count)
            norms[out_idx, start : start + count] = state_norms_from_coefficients(coefficients)
    return norms


def compute_spin_texture_from_qe_save(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str | None = None,
    file_format: str = "auto",
    lattice_alat: NDArray[np.float64] | None = None,
    use_xml_metadata: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(weights, textures)`` for a noncollinear QE ``.save`` directory.

    ``textures`` has shape ``(n_kpoints, n_bands, 3)``.  The wavefunctions must
    have been written by a noncollinear run (``npol = 2``); a scalar-relativistic
    or collinear run stores one component per G-vector and carries no transverse
    spin.

    As for the VASP reader there is deliberately no symmetry-reduced mode: the
    weight is invariant under a point-group operation but the spin is a
    pseudovector and rotates with it, so a stored texture cannot be reused for
    another member of the star.
    """

    save_path = Path(save_dir)
    use_format = resolve_wfc_format(save_path, pattern=pattern, file_format=file_format)
    if use_format == "hdf5":
        if not save_path.is_dir():
            raise NotADirectoryError(f"QE save directory not found: {save_path}")
        infos = list_wfc_hdf5_files(save_path, pattern=pattern or "wfc*.hdf5", spin=spin)
        reader = read_wfc_hdf5
        xk_base = read_xk_hdf5
    else:
        infos = list_wfc_dat_files(save_path, pattern=pattern or "wfc*.dat", spin=spin)
        reader = read_wfc_dat
        xk_base = read_xk_dat
    lattice_alat, metadata = resolve_qe_metadata(
        save_path, lattice_alat, use_xml_metadata=use_xml_metadata
    )
    weights, textures = _compute_weights_from_infos(
        infos,
        reader,
        primitive_kpoints,
        folded_supercell_kpoints,
        transform,
        n_bands,
        tol=tol,
        xk_reader=_xk_reader_with_xml(xk_base, infos, metadata),
        lattice_alat=lattice_alat,
        with_spin_texture=True,
    )
    assert textures is not None
    return weights, textures


def resolve_wfc_format(
    save_dir: str | Path,
    *,
    pattern: str | None = None,
    file_format: str = "auto",
) -> str:
    """Decide whether a QE ``.save`` directory holds HDF5 or binary wavefunctions."""

    save_path = Path(save_dir)
    if file_format not in {"auto", "hdf5", "dat"}:
        raise ValueError("file_format must be one of: auto, hdf5, dat")
    if file_format != "auto":
        return file_format
    if list(save_path.glob(pattern or "wfc*.hdf5")) or list(save_path.glob("wfc*.h5")):
        return "hdf5"
    if list(save_path.glob(pattern or "wfc*.dat")):
        return "dat"
    raise FileNotFoundError(f"no QE wfc*.hdf5, wfc*.h5, or wfc*.dat files found in {save_path}")


def compute_symmetry_weights_from_qe_save(
    save_dir: str | Path,
    primitive_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    lattice_alat: NDArray[np.float64] | None = None,
    operations: NDArray[np.integer] | None = None,
    tol: float = 1e-6,
    spin: int | None = None,
    pattern: str | None = None,
    file_format: str = "auto",
    use_xml_metadata: bool = True,
) -> SymmetryReducedWeights:
    """Unfold a primitive k-path against a symmetry-reduced QE ``.save``.

    The wavefunction files need neither be as many as the requested k-points
    nor be in the same order: each requested primitive k-point ``k`` is folded
    to ``K = k T^T`` and matched to a stored k-point ``K_f`` and an operation
    ``S_pc`` with ``K = S_sc K_f`` modulo a reciprocal-lattice vector, where
    ``S_sc = T S_pc T^-1``.  Unfolding ``S_pc^-1 k`` against the stored,
    unrotated file then gives the weight of ``k``
    (``UnfoldLab.weight_stored_representative``).

    ``lattice_alat`` (QE's ``at`` matrix for the *supercell*) is required: the
    k-point of each file is only recorded in Cartesian ``2*pi/alat`` units, and
    without it the stored fractional k-points -- and hence the matching --
    cannot be determined.  It is read from ``data-file-schema.xml`` when the
    caller does not supply it and ``use_xml_metadata`` is on.  ``operations``
    is a sequence of integer ``3 x 3`` matrices in primitive fractional
    reciprocal coordinates; the identity is implicit, so an empty sequence
    merely allows reordering and reuse of the stored k-points.
    """

    save_path = Path(save_dir)
    if not save_path.is_dir():
        raise NotADirectoryError(f"QE save directory not found: {save_path}")
    use_format = resolve_wfc_format(save_path, pattern=pattern, file_format=file_format)
    if use_format == "hdf5":
        infos = list_wfc_hdf5_files(save_path, pattern=pattern or "wfc*.hdf5", spin=spin)
        reader, base_xk_reader = read_wfc_hdf5, read_xk_hdf5
    else:
        infos = list_wfc_dat_files(save_path, pattern=pattern or "wfc*.dat", spin=spin)
        reader, base_xk_reader = read_wfc_dat, read_xk_dat
    lattice_alat, metadata = resolve_qe_metadata(
        save_path, lattice_alat, use_xml_metadata=use_xml_metadata
    )
    xk_reader = _xk_reader_with_xml(base_xk_reader, infos, metadata)
    return _compute_symmetry_weights_from_infos(
        infos,
        reader,
        xk_reader,
        primitive_kpoints,
        transform,
        n_bands,
        tol=tol,
        lattice_alat=lattice_alat,
        operations=operations,
    )


def resolve_qe_metadata(
    save_dir: str | Path,
    lattice_alat: NDArray[np.float64] | None,
    *,
    use_xml_metadata: bool = True,
    warn_pseudo: bool = True,
) -> tuple[NDArray[np.float64] | None, QESaveMetadata | None]:
    """Fill in what the caller did not supply from ``data-file-schema.xml``.

    Returns the ``at`` matrix to use -- the caller's if given, else the one the
    XML records -- together with the parsed metadata, or ``None`` when there is
    no readable XML.  A PAW/ultrasoft calculation is warned about here, because
    that is the one place every ``.save``-based entry point passes through.
    """

    if not use_xml_metadata:
        return lattice_alat, None
    metadata = try_read_qe_metadata(save_dir)
    if metadata is None:
        return lattice_alat, None
    if warn_pseudo:
        warn_about_pseudopotentials(metadata)
    if lattice_alat is None:
        return metadata.lattice_alat, metadata
    return lattice_alat, metadata


def _xk_reader_with_xml(
    xk_reader: XkReader | None,
    infos: list[WfcInfo],
    metadata: QESaveMetadata | None,
    *,
    atol: float = 1e-6,
) -> XkReader | None:
    """Back a file's own ``xk`` with the XML k-point list.

    QE HDF5 wavefunctions do not always record ``xk``; when the XML does, its
    entry for the file's ``ik`` is used instead, which makes the k-point
    cross-check possible for those files too.  When both are available they
    must agree, otherwise the file order does not match the calculation.
    """

    if metadata is None or metadata.n_kpoints == 0:
        return xk_reader
    table = {
        info.path: metadata.kpoints_cart_alat[info.ik - 1]
        for info in infos
        if 1 <= info.ik <= metadata.n_kpoints
    }
    if not table:
        return xk_reader

    def reader(path: str | Path) -> NDArray[np.float64] | None:
        reference = table.get(Path(path))
        stored = None if xk_reader is None else xk_reader(path)
        if stored is None:
            return reference
        stored = np.asarray(stored, dtype=float)
        if reference is not None and not np.allclose(stored, reference, atol=atol):
            raise ValueError(
                f"{Path(path).name} records xk {np.round(stored, 8).tolist()} but "
                f"{metadata.path.name} lists {np.round(reference, 8).tolist()} at "
                "that k-point index; the wavefunction files and the XML are not "
                "from the same calculation"
            )
        return stored

    return reader


def _stored_fractional_kpoints(
    infos: list[WfcInfo],
    xk_reader: Any,
    lattice_alat: NDArray[np.float64],
) -> NDArray[np.float64]:
    stored = np.zeros((len(infos), 3), dtype=float)
    for index, info in enumerate(infos):
        xk = xk_reader(info.path)
        if xk is None:
            raise ValueError(
                f"{info.path.name} does not record its k-point (xk); symmetry-reduced "
                "unfolding cannot tell which stored k-point it holds"
            )
        stored[index] = fractional_kpoint_from_xk(xk, lattice_alat)
    return stored


def _compute_symmetry_weights_from_infos(
    infos: list[WfcInfo],
    reader: Any,
    xk_reader: Any,
    primitive_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float,
    lattice_alat: NDArray[np.float64] | None,
    operations: NDArray[np.integer] | None,
) -> SymmetryReducedWeights:
    if lattice_alat is None:
        raise ValueError(
            "symmetry-reduced QE unfolding requires lattice_alat (the supercell 'at' "
            "matrix in units of alat) to convert the stored Cartesian k-points to "
            "fractional coordinates"
        )
    primitive = np.asarray(primitive_kpoints, dtype=float)
    stored = _stored_fractional_kpoints(infos, xk_reader, lattice_alat)
    matches = map_kpoints_to_stored(primitive, stored, transform, operations, atol=tol)

    # A band path revisits the same irreducible k-point many times.  Grouping
    # the requested rows by stored k-point reads every wavefunction file exactly
    # once and keeps only one of them in memory; an index-keyed cache would
    # instead grow to hold the whole irreducible wedge.
    groups: dict[int, list[int]] = defaultdict(list)
    for row, match in enumerate(matches):
        groups[match.index].append(row)
    order = sorted(groups)
    grouped_rows = [row for index in order for row in groups[index]]

    # Every row of a group shares the wavefunction, so the squared moduli are
    # formed once per stored k-point rather than once per path point.
    def stream() -> Iterator[SharedWavefunctionGroup]:
        for index in order:
            g_supercell, coefficients = reader(infos[index].path, n_bands)
            yield SharedWavefunctionGroup(
                primitive_kpoints=np.stack(
                    [matches[row].effective_primitive_kpoint for row in groups[index]]
                ),
                folded_supercell_kpoint=stored[index],
                g_supercell=g_supercell,
                coefficients=coefficients,
            )

    grouped = compute_plane_wave_unfolding_weights_shared(stream(), transform, tol=tol)
    weights = np.empty_like(grouped)
    weights[np.asarray(grouped_rows, dtype=np.int64)] = grouped
    indices = np.array([match.index for match in matches], dtype=np.int64)
    return SymmetryReducedWeights(
        weights=weights,
        stored_indices=indices,
        stored_kpoints=stored[indices] if indices.size else stored[:0],
        n_stored=len(infos),
    )


def _compute_weights_from_infos(
    infos: list[WfcInfo],
    reader: Any,
    primitive_kpoints: NDArray[np.float64],
    folded_supercell_kpoints: NDArray[np.float64],
    transform: TransformLike,
    n_bands: int,
    *,
    tol: float,
    xk_reader: Any | None = None,
    lattice_alat: NDArray[np.float64] | None = None,
    with_spin_texture: bool = False,
    band_chunk: int | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Weights, and optionally spin textures, for an aligned set of QE files.

    The wavefunction files must be the k-points of the k-map, in the same
    order.  With ``with_spin_texture`` the same resident copy of each
    wavefunction is used for both tables, so the files are read once.

    ``band_chunk`` reads each k-point in blocks of that many bands instead of
    whole, which bounds the memory a large supercell needs and leaves the
    weights unchanged (``UnfoldLab.bandWeights_flatten``).  It is not available
    together with a spin texture, whose Bloch vectors are accumulated from the
    same resident copy.
    """

    if band_chunk is not None:
        if band_chunk < 1:
            raise ValueError("band_chunk must be a positive number of bands")
        if with_spin_texture:
            raise ValueError("band_chunk is not supported together with a spin texture")

    primitive = np.asarray(primitive_kpoints, dtype=float)
    folded = np.asarray(folded_supercell_kpoints, dtype=float)
    if len(infos) != len(primitive):
        raise ValueError(
            f"found {len(infos)} wavefunction files but kmap has {len(primitive)} k-points"
        )

    weights = np.zeros((len(primitive), n_bands), dtype=float)
    textures = np.zeros((len(primitive), n_bands, 3), dtype=float) if with_spin_texture else None
    for out_idx, info in enumerate(infos):
        expected_ik = out_idx + 1
        if info.ik != expected_ik:
            raise ValueError(
                f"wavefunction k-point order mismatch: expected ik={expected_ik}, "
                f"found ik={info.ik} in {info.path.name}"
            )
        folded_kpoint = folded[out_idx]
        if lattice_alat is not None and xk_reader is not None:
            xk = xk_reader(info.path)
            if xk is not None:
                file_kpoint = fractional_kpoint_from_xk(xk, lattice_alat)
                if not allclose_mod1(file_kpoint, folded[out_idx], atol=1e-6):
                    raise ValueError(
                        f"QE wavefunction {info.path.name} is at k-point "
                        f"{np.round(file_kpoint, 8).tolist()} but the k-map expects "
                        f"{folded[out_idx].tolist()} (mod 1); the k-path and the "
                        "wavefunctions are not the same calculation or are ordered "
                        "differently"
                    )
                folded_kpoint = file_kpoint
        if band_chunk is not None:
            weights[out_idx, :] = compute_plane_wave_unfolding_weights_chunked(
                [
                    (
                        _qe_band_block_data(
                            reader,
                            info,
                            n_bands,
                            start,
                            min(band_chunk, n_bands - start + 1),
                            primitive[out_idx],
                            folded_kpoint,
                        )
                        for start in range(1, n_bands + 1, band_chunk)
                    )
                ],
                transform,
                tol=tol,
            )[0]
            continue
        g_supercell, coefficients = reader(info.path, n_bands)
        weights[out_idx, :] = compute_plane_wave_unfolding_weights(
            [
                PlaneWaveKPointData(
                    primitive_kpoint=primitive[out_idx],
                    folded_supercell_kpoint=folded_kpoint,
                    g_supercell=g_supercell,
                    coefficients=coefficients,
                )
            ],
            transform,
            tol=tol,
        )[0]
        if textures is not None:
            textures[out_idx] = spin_texture_from_coefficients(
                g_supercell,
                coefficients,
                primitive[out_idx],
                folded_kpoint,
                transform,
                tol=tol,
            )
    return weights, textures


def _qe_band_block_data(
    reader: Any,
    info: WfcInfo,
    n_bands: int,
    band_start: int,
    band_count: int,
    primitive_kpoint: NDArray[np.float64],
    folded_kpoint: NDArray[np.float64],
) -> PlaneWaveKPointData:
    """One band block of a QE wavefunction, ready for the matching kernel."""

    g_supercell, coefficients = reader(
        info.path, n_bands, band_start=band_start, band_count=band_count
    )
    return PlaneWaveKPointData(
        primitive_kpoint=primitive_kpoint,
        folded_supercell_kpoint=folded_kpoint,
        g_supercell=g_supercell,
        coefficients=coefficients,
    )
