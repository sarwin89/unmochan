"""Backend-neutral workflow dispatchers for VASP and QE."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from unfoldlab.core.numerics import TransformLike
from unfoldlab.core.projections import ProjectionSelector
from unfoldlab.core.structures import Structure
from unfoldlab.core.unfolding import (
    SpinTextureDiagnostics,
    WeightDiagnostics,
    diagnose_fiber_sum_rule,
    diagnose_spin_textures,
    diagnose_weights,
)
from unfoldlab.io.qe import (
    read_gnu_blocks,
    read_kmap,
    write_qe_path_files,
    write_spin_texture_table,
    write_weight_table,
)
from unfoldlab.io.qe_wfc import compute_spin_texture_from_qe_save, compute_weights_from_qe_save
from unfoldlab.io.vasp import write_vasp_path_files
from unfoldlab.io.vasp_wfc import compute_spin_texture_from_wavecar, compute_weights_from_wavecar
from unfoldlab.workflows.qe import QEUnfoldResult, unfold_qe_bands
from unfoldlab.workflows.vasp import VaspUnfoldResult, unfold_vasp_bands

BackendCode = Literal["vasp", "qe"]


def write_backend_path_files(
    code: BackendCode,
    path_json: str | Path,
    *,
    kpoints: str | Path,
    kmap: str | Path,
    ticks: str | Path,
):
    if code == "qe":
        return write_qe_path_files(path_json, qe_kpoints=kpoints, kmap=kmap, ticks=ticks)
    if code == "vasp":
        return write_vasp_path_files(path_json, kpoints=kpoints, kmap=kmap, ticks=ticks)
    raise ValueError("code must be vasp or qe")


def compute_backend_weights(
    code: BackendCode,
    *,
    kmap: str | Path,
    transform: TransformLike,
    out: str | Path,
    bands: str | Path | None = None,
    nbnd: int | None = None,
    coefficients: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    wavecar: str | Path | None = None,
    spin: int = 1,
    wfc_format: str = "auto",
    wfc_pattern: str | None = None,
    tol: float = 1e-6,
    qe_lattice_alat: NDArray[np.float64] | None = None,
    operations: NDArray[np.integer] | None = None,
    spin_texture_out: str | Path | None = None,
    band_chunk: int | None = None,
) -> WeightDiagnostics:
    """Compute and write unfolding weights for one backend.

    ``spin_texture_out`` additionally writes the unfolded spin texture of a
    noncollinear wavefunction set -- a VASP ``WAVECAR`` or a QE ``.save``
    directory; see
    :func:`unfoldlab.io.vasp_wfc.compute_spin_texture_from_wavecar` and
    :func:`unfoldlab.io.qe_wfc.compute_spin_texture_from_qe_save`.

    ``qe_lattice_alat`` is the supercell direct lattice in units of ``alat``
    (Quantum ESPRESSO's ``at`` matrix).  Supplying it enables a cross-check of
    the k-point stored in each QE wavefunction file against the k-map; see
    :func:`unfoldlab.io.qe_wfc.compute_weights_from_qe_save`.

    ``operations`` enables symmetry-reduced unfolding for the VASP backend; see
    :func:`unfoldlab.io.vasp_wfc.compute_weights_from_wavecar`.

    ``band_chunk`` bounds the number of bands of a wavefunction held in memory
    at once, for both backends reading wavefunctions; the weights are unchanged
    (``UnfoldLab.bandWeights_flatten``).
    """

    mapping = read_kmap(kmap)
    if spin_texture_out is not None and operations is not None:
        raise ValueError(
            "a spin texture cannot be reconstructed from a symmetry-reduced "
            "wavefunction set: the weight is invariant under a point-group "
            "operation but the spin is a pseudovector and rotates with it"
        )
    if code == "qe":
        if spin_texture_out is not None and qe_save_dir is None:
            raise ValueError(
                "a QE spin texture needs the wavefunctions themselves; a coefficient "
                "or weight table carries no spinor components"
            )
        if operations is not None and coefficients is not None:
            raise ValueError(
                "symmetry-reduced unfolding needs the wavefunctions themselves; a "
                "coefficient table is indexed against the k-points it was written for"
            )
        if bands is not None:
            _, energies = read_gnu_blocks(bands)
            n_bands = energies.shape[1]
        elif nbnd is not None:
            n_bands = nbnd
        elif qe_save_dir is not None:
            n_bands = _qe_band_count_from_xml(qe_save_dir, spin)
        else:
            raise ValueError("QE weights require bands or nbnd")
        if coefficients is not None:
            from unfoldlab.io.qe import weights_from_coefficient_table

            weights = weights_from_coefficient_table(
                coefficients,
                mapping,
                transform,
                n_bands,
                tol=tol,
            )
        elif qe_save_dir is not None:
            if spin_texture_out is not None:
                weights, textures = compute_spin_texture_from_qe_save(
                    qe_save_dir,
                    mapping.primitive_kpoints,
                    mapping.supercell_folded_kpoints,
                    transform,
                    n_bands,
                    tol=tol,
                    spin=spin,
                    pattern=wfc_pattern,
                    file_format=wfc_format,
                    lattice_alat=qe_lattice_alat,
                )
            else:
                weights = compute_weights_from_qe_save(
                    qe_save_dir,
                    mapping.primitive_kpoints,
                    mapping.supercell_folded_kpoints,
                    transform,
                    n_bands,
                    tol=tol,
                    spin=spin,
                    pattern=wfc_pattern,
                    file_format=wfc_format,
                    lattice_alat=qe_lattice_alat,
                    operations=operations,
                    band_chunk=band_chunk,
                )
        else:
            raise ValueError("QE weights require qe_save_dir or coefficients")
    elif code == "vasp":
        if wavecar is None:
            raise ValueError("VASP weights require wavecar")
        if spin_texture_out is not None:
            _energies, weights, textures = compute_spin_texture_from_wavecar(
                wavecar,
                mapping.primitive_kpoints,
                mapping.supercell_folded_kpoints,
                transform,
                spin=spin,
                tol=tol,
            )[1:]
        else:
            _energies, weights = compute_weights_from_wavecar(
                wavecar,
                mapping.primitive_kpoints,
                mapping.supercell_folded_kpoints,
                transform,
                spin=spin,
                tol=tol,
                operations=operations,
                band_chunk=band_chunk,
            )[1:]
    else:
        raise ValueError("code must be vasp or qe")

    if spin_texture_out is not None:
        write_spin_texture_table(spin_texture_out, textures)
        _warn_on_spin_texture_violation(diagnose_spin_textures(textures, weights))

    write_weight_table(out, weights)
    _warn_on_sum_rule_violation(weights, mapping.primitive_kpoints, transform)
    return diagnose_weights(weights)


def _warn_on_spin_texture_violation(
    report: SpinTextureDiagnostics,
    *,
    atol: float = 1e-8,
) -> None:
    """Warn when a texture is longer than the weight it belongs to.

    The bound is exact mathematics (``UnfoldLab.norm_spinTexture_le_pairWeight``),
    so an excess means the two tables were not produced from the same state.
    """

    if report.max_texture_excess > atol:
        warnings.warn(
            "spin texture exceeds the spectral weight by "
            f"{report.max_texture_excess:.3e}; the texture and the weights do not "
            "come from the same wavefunction",
            RuntimeWarning,
            stacklevel=2,
        )


def _qe_band_count_from_xml(save_dir: str | Path, spin: int) -> int:
    """Band count from the ``.save`` XML, so ``--nbnd`` can be omitted."""

    from unfoldlab.io.qe_xml import eigenvalues_for_spin, try_read_qe_metadata

    metadata = try_read_qe_metadata(save_dir)
    if metadata is None:
        raise ValueError(
            "QE weights require bands or nbnd; the save directory has no readable "
            "data-file-schema.xml to take the band count from"
        )
    return int(eigenvalues_for_spin(metadata, spin).shape[1])


def _warn_on_sum_rule_violation(
    weights: NDArray[np.float64],
    primitive_kpoints: NDArray[np.float64],
    transform: TransformLike,
    *,
    atol: float = 1e-6,
) -> None:
    """Warn when the unfolding sum rule is violated.

    A complete fiber must sum to exactly one; an incomplete one -- which is what
    a band-structure path usually samples -- must still sum to at most one, so
    an excess above one is reported as well.
    """

    try:
        report = diagnose_fiber_sum_rule(weights, primitive_kpoints, transform)
    except ValueError:
        return
    if report.max_deviation is not None and report.max_deviation > atol:
        warnings.warn(
            "unfolding sum rule violated: the weights of a complete fiber of "
            f"{report.multiplicity} primitive k-points deviate from 1 by "
            f"{report.max_deviation:.3e} (worst fiber {report.worst_fiber}); "
            "check the transformation matrix and the k-point/G-vector conventions",
            RuntimeWarning,
            stacklevel=2,
        )
    if report.max_partial_excess is not None and report.max_partial_excess > atol:
        warnings.warn(
            "unfolding sum rule violated: an incompletely sampled fiber already "
            f"carries {1.0 + report.max_partial_excess:.6f} > 1 of weight "
            f"(worst fiber {report.worst_partial_fiber}); check the "
            "transformation matrix and the k-point/G-vector conventions",
            RuntimeWarning,
            stacklevel=2,
        )


def unfold_backend_bands(
    code: BackendCode,
    *,
    kmap: str | Path,
    transform: TransformLike | None = None,
    ticks: str | Path | None = None,
    bands: str | Path | None = None,
    weights: str | Path | None = None,
    qe_save_dir: str | Path | None = None,
    coefficients: str | Path | None = None,
    wavecar: str | Path | None = None,
    eigenval: str | Path | None = None,
    fermi: float = 0.0,
    out: str | Path | None = None,
    plot: str | Path | None = None,
    write_weights: str | Path | None = None,
    spin: int = 1,
    tol: float = 1e-6,
    wfc_format: str = "auto",
    wfc_pattern: str | None = None,
    qe_lattice_alat: NDArray[np.float64] | None = None,
    operations: NDArray[np.integer] | None = None,
    projections: Sequence[str | ProjectionSelector] = (),
    procar: str | Path | None = None,
    supercell: str | Path | Structure | None = None,
    site_groups: Mapping[str, Mapping[object, Sequence[int]]] | None = None,
    layer_axis: int = 2,
    layer_tol: float = 0.5,
) -> QEUnfoldResult | VaspUnfoldResult:
    """Unfold bands for one backend.

    ``projections`` enables the approximate projector-assisted mode and needs
    ``procar`` plus the supercell structure; it is available for the VASP
    backend only, since it is driven by a VASP ``PROCAR``.

    ``qe_lattice_alat`` is QE's supercell ``at`` matrix; it enables the k-point
    cross-check and is required together with ``operations`` for the QE backend.
    """

    if code == "qe":
        if projections:
            raise ValueError(
                "projected unfolding is driven by a VASP PROCAR and is not available "
                "for the QE backend"
            )
        if bands is None and qe_save_dir is None:
            raise ValueError(
                "QE unfold requires bands, or a save directory whose XML records the eigenvalues"
            )
        return unfold_qe_bands(
            bands=bands,
            kmap=kmap,
            ticks=ticks,
            weights=weights,
            qe_save_dir=qe_save_dir,
            coefficients=coefficients,
            transform=transform,
            fermi=fermi,
            out=out,
            plot=plot,
            write_weights=write_weights,
            tol=tol,
            spin=spin,
            wfc_pattern=wfc_pattern,
            wfc_format=wfc_format,
            lattice_alat=qe_lattice_alat,
            operations=operations,
        )
    if code == "vasp":
        return unfold_vasp_bands(
            kmap=kmap,
            ticks=ticks,
            wavecar=wavecar,
            eigenval=eigenval,
            weights=weights,
            transform=transform,
            fermi=fermi,
            out=out,
            plot=plot,
            write_weights=write_weights,
            spin=spin,
            tol=tol,
            operations=operations,
            projections=projections,
            procar=procar,
            supercell=supercell,
            site_groups=site_groups,
            layer_axis=layer_axis,
            layer_tol=layer_tol,
        )
    raise ValueError("code must be vasp or qe")
