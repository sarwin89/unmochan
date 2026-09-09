"""Backend-neutral UnfoldLab commands.

These are the workflows that do not belong to a single code: building a folded
k-point path, computing weights, unfolding, and the diagnostics (`fiber`, `cut`,
`ensemble`, `norms`, `degeneracy`, `detect-symmetry`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table

from unmochan.cli.app import app
from unmochan.cli.common import (
    check_symmetry_against_structure,
    console,
    parse_code,
    parse_kpoint,
    parse_matrix,
    resolve_operations,
    resolve_qe_save_dir,
)
from unmochan.core.brillouin import reduce_kpoints_to_first_bz
from unmochan.core.mesh import MeshSpec, diagnose_mesh, mesh_kpoint_mappings
from unmochan.io.qe_xml import read_qe_xml
from unmochan.io.serialization import (
    build_run_manifest,
    write_ebs,
    write_run_manifest,
)
from unmochan.io.vasp import read_poscar
from unmochan.twist.commensurate import diagnose_stack, hexagonal_commensurate_twists
from unmochan.workflows.backend import (
    compute_backend_weights,
    unfold_backend_bands,
    write_backend_path_files,
)


@app.command("make-kpoints")
def make_kpoints(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    path_json: Annotated[Path, typer.Argument(exists=True, readable=True)],
    kpoints: Annotated[Path | None, typer.Option("--kpoints")] = None,
    kmap: Annotated[Path, typer.Option("--kmap")] = Path("kmap.tsv"),
    ticks: Annotated[Path, typer.Option("--ticks")] = Path("path_ticks.tsv"),
) -> None:
    """Generate folded-path k-points for VASP or QE."""

    backend = parse_code(code)
    output_kpoints = kpoints or Path("KPOINTS" if backend == "vasp" else "qe_kpoints_supercell.in")
    path_data = write_backend_path_files(
        backend,
        path_json,
        kpoints=output_kpoints,
        kmap=kmap,
        ticks=ticks,
    )
    console.print(f"Wrote {len(path_data.kpoints)} {backend.upper()} k-points to {output_kpoints}")
    console.print(f"Wrote k-point map to {kmap}")
    console.print(f"Wrote plot ticks to {ticks}")


@app.command("weights")
def weights_command(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help='Nine numbers, e.g. "2 0 0 0 2 0 0 0 1".'),
    ],
    out: Annotated[Path, typer.Option("--out")] = Path("weights.dat"),
    bands: Annotated[Path | None, typer.Option("--bands", exists=True, readable=True)] = None,
    nbnd: Annotated[int | None, typer.Option("--nbnd")] = None,
    coefficients: Annotated[
        Path | None,
        typer.Option("--coefficients", exists=True, readable=True),
    ] = None,
    qe_save_dir: Annotated[
        Path | None,
        typer.Option("--qe-save-dir", "--save-dir", exists=True, file_okay=False),
    ] = None,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir")] = None,
    wavecar: Annotated[Path | None, typer.Option("--wavecar", exists=True, readable=True)] = None,
    spin: Annotated[int, typer.Option("--spin", min=1)] = 1,
    wfc_format: Annotated[str, typer.Option("--wfc-format")] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
    symmetry: Annotated[
        Path | None,
        typer.Option(
            "--symmetry",
            exists=True,
            readable=True,
            help=(
                "Point-group operations in primitive reciprocal coordinates, nine "
                "integers per line; enables unfolding from a symmetry-reduced "
                "wavefunction set."
            ),
        ),
    ] = None,
    reuse_kpoints: Annotated[
        bool,
        typer.Option(
            "--reuse-kpoints/--no-reuse-kpoints",
            help=(
                "Allow one stored wavefunction to serve several k-map rows, which "
                "is what a fiber k-map produced by `unmochan fiber` needs."
            ),
        ),
    ] = False,
    lattice_alat: Annotated[
        str | None,
        typer.Option(
            "--lattice-alat",
            help=(
                "Supercell direct lattice in units of alat (QE's 'at' matrix), nine "
                "numbers. Enables the cross-check of the k-point stored in each QE "
                "wavefunction file, and is required by --symmetry for QE."
            ),
        ),
    ] = None,
    spin_texture: Annotated[
        Path | None,
        typer.Option(
            "--spin-texture",
            help=(
                "Also write the unfolded spin texture (ik band Sx Sy Sz |S|) of a "
                "noncollinear VASP WAVECAR to this file. Not available with "
                "--symmetry: the spin rotates with the point-group operation."
            ),
        ),
    ] = None,
    band_chunk: Annotated[
        int | None,
        typer.Option(
            "--band-chunk",
            min=1,
            help=(
                "Read each wavefunction in blocks of this many bands instead "
                "of all at once. Bounds the memory a large supercell needs; "
                "the weights are unchanged."
            ),
        ),
    ] = None,
    structure: Annotated[
        Path | None,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help=(
                "Supercell POSCAR. When given together with --symmetry, the "
                "operations are checked against the crystal, which is the one "
                "hypothesis of symmetry-reduced unfolding the kernel cannot test."
            ),
        ),
    ] = None,
    magmom: Annotated[
        str | None,
        typer.Option(
            "--magmom",
            help=(
                "Local moments for the --structure symmetry check, one per site "
                "(collinear) or three per site (noncollinear). Required for a "
                "spin-polarized run: a moment pattern breaks symmetries that the "
                "atoms alone do not."
            ),
        ),
    ] = None,
) -> None:
    """Compute spectral weights for VASP or QE through the common kernel."""

    operations = resolve_operations(symmetry, reuse_kpoints)
    if symmetry is not None and structure is not None and operations is not None:
        check_symmetry_against_structure(structure, operations, parse_matrix(matrix), magmom)

    if spin_texture is not None and (symmetry is not None or reuse_kpoints):
        raise typer.BadParameter(
            "--spin-texture cannot be combined with --symmetry or --reuse-kpoints: "
            "the spectral weight is invariant under a point-group operation but the "
            "spin is a pseudovector and rotates with it, so a stored texture cannot "
            "be reused for another member of the star"
        )

    diagnostics = compute_backend_weights(
        parse_code(code),
        kmap=kmap,
        transform=parse_matrix(matrix),
        out=out,
        bands=bands,
        nbnd=nbnd,
        coefficients=coefficients,
        qe_save_dir=resolve_qe_save_dir(qe_save_dir, outdir, prefix),
        wavecar=wavecar,
        spin=spin,
        wfc_format=wfc_format,
        wfc_pattern=wfc_pattern,
        tol=tol,
        qe_lattice_alat=None if lattice_alat is None else parse_matrix(lattice_alat),
        operations=operations,
        spin_texture_out=spin_texture,
        band_chunk=band_chunk,
    )
    console.print(f"Wrote {out}")
    if spin_texture is not None:
        console.print(f"Wrote {spin_texture}")
    console.print_json(json.dumps(diagnostics.to_dict()))


@app.command("unfold")
def unfold_command(
    code: Annotated[str, typer.Option("--code", help="Backend: vasp or qe")],
    kmap: Annotated[Path, typer.Option("--kmap", exists=True, readable=True)],
    matrix: Annotated[str | None, typer.Option("--matrix")] = None,
    ticks: Annotated[Path | None, typer.Option("--ticks", exists=True, readable=True)] = None,
    bands: Annotated[Path | None, typer.Option("--bands", exists=True, readable=True)] = None,
    weights: Annotated[Path | None, typer.Option("--weights", exists=True, readable=True)] = None,
    qe_save_dir: Annotated[
        Path | None,
        typer.Option("--qe-save-dir", "--save-dir", exists=True, file_okay=False),
    ] = None,
    prefix: Annotated[str | None, typer.Option("--prefix")] = None,
    outdir: Annotated[Path | None, typer.Option("--outdir")] = None,
    coefficients: Annotated[
        Path | None,
        typer.Option("--coefficients", exists=True, readable=True),
    ] = None,
    wavecar: Annotated[Path | None, typer.Option("--wavecar", exists=True, readable=True)] = None,
    eigenval: Annotated[Path | None, typer.Option("--eigenval", exists=True, readable=True)] = None,
    fermi: Annotated[float, typer.Option("--fermi")] = 0.0,
    out: Annotated[Path, typer.Option("--out")] = Path("unfolded_bands.dat"),
    plot: Annotated[Path, typer.Option("--plot")] = Path("unfolded_bands.png"),
    write_weights: Annotated[Path | None, typer.Option("--write-weights")] = None,
    spin: Annotated[int, typer.Option("--spin", min=1)] = 1,
    tol: Annotated[float, typer.Option("--tol")] = 1e-6,
    wfc_format: Annotated[str, typer.Option("--wfc-format")] = "auto",
    wfc_pattern: Annotated[str | None, typer.Option("--wfc-pattern")] = None,
    symmetry: Annotated[
        Path | None,
        typer.Option(
            "--symmetry",
            exists=True,
            readable=True,
            help=(
                "Point-group operations in primitive reciprocal coordinates, nine "
                "integers per line; enables unfolding from a symmetry-reduced "
                "wavefunction set."
            ),
        ),
    ] = None,
    reuse_kpoints: Annotated[
        bool,
        typer.Option(
            "--reuse-kpoints/--no-reuse-kpoints",
            help=(
                "Allow one stored wavefunction to serve several k-map rows, which "
                "is what a fiber k-map produced by `unmochan fiber` needs."
            ),
        ),
    ] = False,
    lattice_alat: Annotated[
        str | None,
        typer.Option(
            "--lattice-alat",
            help=(
                "Supercell direct lattice in units of alat (QE's 'at' matrix), nine "
                "numbers. Enables the cross-check of the k-point stored in each QE "
                "wavefunction file, and is required by --symmetry for QE."
            ),
        ),
    ] = None,
    projection: Annotated[
        list[str] | None,
        typer.Option(
            "--projection",
            help=(
                "Generic projection selector such as species:A, orbital:d, layer:0 "
                "or surface:top; repeatable. Selectors of the same namespace are "
                "combined with OR, different namespaces with AND. Requires --procar "
                "and --supercell (VASP only)."
            ),
        ),
    ] = None,
    procar: Annotated[
        Path | None,
        typer.Option(
            "--procar",
            exists=True,
            readable=True,
            help="VASP PROCAR supplying the site- and orbital-projected character.",
        ),
    ] = None,
    supercell: Annotated[
        Path | None,
        typer.Option(
            "--supercell",
            exists=True,
            readable=True,
            help="Supercell POSCAR the projection selectors are resolved against.",
        ),
    ] = None,
    layer_axis: Annotated[
        int,
        typer.Option("--layer-axis", min=0, max=2, help="Stacking axis for layer: selectors."),
    ] = 2,
    layer_tol: Annotated[
        float,
        typer.Option("--layer-tol", min=0.0, help="Layer thickness tolerance in angstrom."),
    ] = 0.5,
    json_out: Annotated[
        Path | None,
        typer.Option(
            "--json",
            help=(
                "Also write the effective band structure, manifest included. "
                "A .h5/.hdf5 suffix writes HDF5, anything else JSON."
            ),
        ),
    ] = None,
    manifest: Annotated[
        Path | None,
        typer.Option("--manifest", help="Write a run manifest (provenance) as JSON."),
    ] = None,
    hash_sources: Annotated[
        bool,
        typer.Option(
            "--hash-sources/--no-hash-sources",
            help="Record a SHA-256 digest of every input file in the manifest.",
        ),
    ] = True,
    magmom: Annotated[
        str | None,
        typer.Option(
            "--magmom",
            help=(
                "Local moments for the --supercell symmetry check, one per site "
                "(collinear) or three per site (noncollinear). Required for a "
                "spin-polarized run: a moment pattern breaks symmetries that the "
                "atoms alone do not."
            ),
        ),
    ] = None,
) -> None:
    """Build an unfolded effective band table and plot for VASP or QE."""

    transform = parse_matrix(matrix) if matrix is not None else None
    operations = resolve_operations(symmetry, reuse_kpoints)
    if symmetry is not None and supercell is not None and operations is not None:
        if transform is None:
            raise typer.BadParameter("--symmetry with --supercell also needs --matrix")
        check_symmetry_against_structure(supercell, operations, transform, magmom)
    result = unfold_backend_bands(
        parse_code(code),
        kmap=kmap,
        transform=transform,
        ticks=ticks,
        bands=bands,
        weights=weights,
        qe_save_dir=resolve_qe_save_dir(qe_save_dir, outdir, prefix),
        coefficients=coefficients,
        wavecar=wavecar,
        eigenval=eigenval,
        fermi=fermi,
        out=out,
        plot=plot,
        write_weights=write_weights,
        spin=spin,
        tol=tol,
        wfc_format=wfc_format,
        wfc_pattern=wfc_pattern,
        qe_lattice_alat=None if lattice_alat is None else parse_matrix(lattice_alat),
        operations=operations,
        projections=tuple(projection or ()),
        procar=procar,
        supercell=supercell,
        layer_axis=layer_axis,
        layer_tol=layer_tol,
    )
    console.print(f"Wrote {result.output_path}")
    if result.weights_path is not None:
        console.print(f"Wrote {result.weights_path}")
    if result.plot_path is not None:
        console.print(f"Wrote {result.plot_path}")

    if json_out is not None or manifest is not None:
        run_manifest = build_run_manifest(
            code=parse_code(code),
            weight_mode=result.mode,
            transformation=transform,
            reference_energy=fermi,
            spin_channel=spin,
            projection=" ".join(projection) if projection else None,
            sources={
                "bands": bands,
                "coefficients": coefficients,
                "eigenval": eigenval,
                "kmap": kmap,
                "procar": procar,
                "qe_save_dir": qe_save_dir,
                "supercell": supercell,
                "symmetry": symmetry,
                "ticks": ticks,
                "wavecar": wavecar,
                "weights": weights,
            },
            digest=hash_sources,
        )
        if manifest is not None:
            write_run_manifest(manifest, run_manifest)
            console.print(f"Wrote {manifest}")
        if json_out is not None:
            write_ebs(json_out, result.ebs, manifest=run_manifest)
            console.print(f"Wrote {json_out}")

    console.print(f"Weight mode: {result.mode}")


@app.command("qe-info")
def qe_info_command(
    save: Annotated[
        Path,
        typer.Option(
            "--save",
            exists=True,
            readable=True,
            help=(
                "QE .save directory, an outdir containing one, or the data-file-schema.xml itself."
            ),
        ),
    ],
    kpoints: Annotated[
        bool,
        typer.Option("--kpoints/--no-kpoints", help="List the k-points."),
    ] = False,
) -> None:
    """Report what a QE .save records: cell, at matrix, k-points, pseudo kind."""

    metadata = read_qe_xml(save)
    console.print(f"XML: {metadata.path}")
    console.print(f"alat: {metadata.alat_bohr:.8f} bohr")
    table = Table(title="Direct lattice (rows)")
    for label in ("component", "bohr", "angstrom", "alat units"):
        table.add_column(label)
    for index in range(3):
        table.add_row(
            f"a{index + 1}",
            " ".join(f"{value: .6f}" for value in metadata.lattice_bohr[index]),
            " ".join(f"{value: .6f}" for value in metadata.lattice_angstrom[index]),
            " ".join(f"{value: .6f}" for value in metadata.lattice_alat[index]),
        )
    console.print(table)
    console.print(
        "--lattice-alat " + " ".join(f"{value:.10f}" for value in metadata.lattice_alat.reshape(-1))
    )
    console.print(f"k-points: {metadata.n_kpoints}")
    console.print(f"Bands: {metadata.nbnd}")
    console.print(
        f"Spin: lsda={metadata.lsda} noncolin={metadata.noncolin} spinorbit={metadata.spinorbit}"
    )
    console.print(f"Gamma-only storage: {metadata.gamma_only}")
    if metadata.fermi_energy_ev is not None:
        console.print(f"Fermi/highest occupied level: {metadata.fermi_energy_ev:.6f} eV")
    for name, pseudo in metadata.species:
        console.print(f"Species {name}: {pseudo}")
    warning = metadata.pseudo_warning()
    console.print(
        "Pseudopotentials: norm-conserving (plane-wave weights are exact)"
        if warning is None
        else f"Warning: {warning}"
    )
    if kpoints:
        ktable = Table(title="k-points")
        for label in ("index", "cartesian (2pi/alat)", "fractional", "weight"):
            ktable.add_column(label)
        for index in range(metadata.n_kpoints):
            ktable.add_row(
                str(index + 1),
                " ".join(f"{value: .6f}" for value in metadata.kpoints_cart_alat[index]),
                " ".join(f"{value: .6f}" for value in metadata.kpoints_frac[index]),
                f"{metadata.kpoint_weights[index]:.6f}",
            )
        console.print(ktable)


@app.command("mesh")
def mesh_command(
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Supercell matrix T, nine numbers."),
    ],
    divisions: Annotated[
        str,
        typer.Option(
            "--divisions",
            help="Supercell k-mesh divisions, three numbers, e.g. 4,4,4.",
        ),
    ],
    shift: Annotated[
        str,
        typer.Option(
            "--shift",
            help="Mesh shift in units of one mesh spacing, e.g. 0.5,0.5,0.5.",
        ),
    ] = "0,0,0",
    point: Annotated[
        list[str] | None,
        typer.Option(
            "--point",
            help="Primitive k-point to test for resolvability, repeatable: x,y,z.",
        ),
    ] = None,
    structure: Annotated[
        Path | None,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="Primitive POSCAR, so that distances are Cartesian and not fractional.",
        ),
    ] = None,
    kpoints_out: Annotated[
        Path | None,
        typer.Option("--kpoints", help="Write the resolvable primitive k-points as TSV."),
    ] = None,
    first_bz: Annotated[
        bool,
        typer.Option(
            "--first-bz",
            help="Add first-Brillouin-zone representatives to the k-point file.",
        ),
    ] = False,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the report as JSON."),
    ] = None,
) -> None:
    """Say which primitive k-points a supercell k-mesh can resolve.

    A supercell run on a mesh does not sample the primitive Brillouin zone
    freely: the resolvable primitive k-points are exactly the mesh generated by
    ``M @ T`` with the same shift, ``|det M| * |det T|`` of them.  A shifted
    supercell mesh misses Gamma whatever the transform, an odd division misses
    the zone boundary along that axis, and the unfolded mesh is generally not a
    Monkhorst-Pack mesh at all, so the k-points have to be listed rather than
    specified by three divisions.
    """

    transform = parse_matrix(matrix)
    counts = [int(round(float(token))) for token in divisions.replace(",", " ").split()]
    if len(counts) != 3:
        raise typer.BadParameter("divisions must contain exactly three numbers")
    offsets = [float(token) for token in shift.replace(",", " ").split()]
    if len(offsets) != 3:
        raise typer.BadParameter("shift must contain exactly three numbers")

    requested = [parse_kpoint(raw).fractional for raw in point] if point else None
    reciprocal = read_poscar(structure).reciprocal_lattice if structure is not None else None

    supercell_mesh = MeshSpec.from_divisions(counts, offsets)
    report = diagnose_mesh(
        supercell_mesh,
        transform,
        requested=requested,
        reciprocal_lattice=reciprocal,
    )

    console.print(
        f"Supercell mesh: {report.supercell_mesh.count} k-points; "
        f"transform multiplicity |det T| = {report.multiplicity}; "
        f"resolvable primitive k-points: {report.primitive_mesh.count}"
    )
    console.print(f"Gamma resolved: {report.gamma_resolved}")
    console.print(
        "Zone boundary 1/2 e_i resolved: "
        + ", ".join(f"{axis}={value}" for axis, value in enumerate(report.zone_boundary_resolved))
    )
    console.print(
        f"Distinct values per axis: {list(report.axis_divisions)}; "
        f"a Monkhorst-Pack specification describes this mesh: {report.is_product_mesh}"
    )
    if not report.is_product_mesh:
        console.print(
            "The unfolded mesh is not a product mesh: list its k-points with --kpoints "
            "instead of quoting divisions."
        )

    if report.requested.size:
        units = "Cartesian" if report.cartesian else "fractional"
        table = Table(title=f"Requested primitive k-points ({units} distances)")
        for label in ("k-point", "resolved", "distance"):
            table.add_column(label)
        for kpoint, resolved, distance in zip(
            report.requested, report.resolved, report.distances, strict=True
        ):
            table.add_row(
                " ".join(f"{value: .6f}" for value in kpoint),
                "yes" if resolved else "no",
                f"{distance:.6f}",
            )
        console.print(table)

    if kpoints_out is not None:
        mappings = mesh_kpoint_mappings(supercell_mesh, transform)
        header = "k1\tk2\tk3\tweight\tK1\tK2\tK3"
        reduced: list[list[float]] | None = None
        if first_bz:
            if reciprocal is None:
                console.print(
                    "[yellow]--first-bz without --structure reduces in fractional norm, "
                    "which is the physical one only for a cubic cell.[/yellow]"
                )
            reduced = [
                [float(value) for value in row]
                for row in reduce_kpoints_to_first_bz(
                    [mapping.primitive.fractional for mapping in mappings], reciprocal
                )
            ]
            header += "\tb1\tb2\tb3"
        lines = [header]
        for index, mapping in enumerate(mappings):
            primitive = mapping.primitive
            columns = [
                *(f"{value:.12f}" for value in primitive.fractional),
                f"{primitive.weight:.12f}",
                *(f"{value:.12f}" for value in mapping.supercell.fractional),
            ]
            if reduced is not None:
                columns.extend(f"{value:.12f}" for value in reduced[index])
            lines.append("\t".join(columns))
        kpoints_out.write_text("\n".join(lines) + "\n")
        console.print(f"Wrote {len(mappings)} primitive k-points to {kpoints_out}")

    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")


@app.command("twist")
def twist_command(
    max_index: Annotated[
        int,
        typer.Option(
            "--max-index",
            min=1,
            help="Enumerate commensurate hexagonal twists with indices up to this value.",
        ),
    ] = 8,
    moire: Annotated[
        Path | None,
        typer.Option(
            "--moire",
            exists=True,
            readable=True,
            help="POSCAR of the moire (stack) cell, to diagnose instead of enumerating.",
        ),
    ] = None,
    layer: Annotated[
        list[str] | None,
        typer.Option(
            "--layer",
            help="Layer reference as LABEL=POSCAR; repeat once per layer.",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the result as JSON."),
    ] = None,
) -> None:
    """List commensurate twist angles, or diagnose a stack against its layers.

    With no ``--moire`` this enumerates the commensurate twists of a hexagonal
    bilayer: the angle, the cell size, and the index pair that generates it.
    Only the smallest cell per angle is listed, because the same twist can be
    written in a needlessly larger supercell.

    With ``--moire`` and one ``--layer LABEL=POSCAR`` per layer it reports the
    integer transform, multiplicity and rotation of every layer, and warns
    about the double-counting that makes spectral weight in one layer's zone
    say nothing about which layer a state lives on.
    """

    if moire is None:
        if layer:
            raise typer.BadParameter("--layer requires --moire")
        twists = hexagonal_commensurate_twists(max_index)
        table = Table(title=f"Commensurate hexagonal twists (indices up to {max_index})")
        for label in ("m", "n", "cells", "angle (deg)", "equivalent angle (deg)"):
            table.add_column(label)
        for twist in twists:
            table.add_row(
                str(twist.m),
                str(twist.n),
                str(twist.cells),
                f"{twist.angle_deg:.4f}",
                f"{twist.equivalent_angle_deg:.4f}",
            )
        console.print(table)
        if json_out is not None:
            json_out.write_text(json.dumps([t.to_dict() for t in twists], indent=2))
            console.print(f"Wrote {json_out}")
        return

    if not layer:
        raise typer.BadParameter("--moire requires at least one --layer LABEL=POSCAR")
    layers = {}
    for entry in layer:
        label, separator, path = entry.partition("=")
        if not separator or not label:
            raise typer.BadParameter(f"--layer expects LABEL=POSCAR, got {entry!r}")
        layers[label] = read_poscar(Path(path))
    report = diagnose_stack(read_poscar(moire), layers)
    console.print(report.summary())
    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")
