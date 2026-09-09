"""Analysis commands: `strain`, `slab`, `dispersion`,
`overlap`, `fermi`, `dos`, `branches`, `shadow`, `bz`.

These sit on either side of an unfolded run rather than inside it.  The
symmetry commands live in `unmochan.cli.symmetry`.
`strain` and `slab` interrogate the *inputs* -- the two structures, the
transformation between them, and which of the cell directions are real
periods -- and `dispersion`, `overlap` and `fermi`
extract physical numbers from a finished run: a peak dispersion and effective
mass, the conditioning of a non-orthogonal basis, the energy reference the whole
picture is quoted against, and the density of states -- which is also the one
check of an unfolded run that needs nothing but the supercell eigenvalues.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.table import Table

from unmochan.cli.app import app
from unmochan.cli.common import console, parse_broadening_kind, parse_kpoint, parse_matrix
from unmochan.core.brillouin import (
    diagnose_bz_reduction,
    reduce_to_first_bz,
    zone_boundary_distance,
)
from unmochan.core.conditioning import (
    DEFAULT_OVERLAP_THRESHOLD,
    diagnose_overlap_conditioning,
    solve_generalized_eigenproblem_truncated,
)
from unmochan.core.dispersion import spectral_peaks, unresolvable_pairs
from unmochan.core.dos import diagnose_dos_conservation, supercell_dos, unfolded_dos
from unmochan.core.fermi import (
    align_reference,
    electrons_per_primitive_cell,
    find_fermi_level,
)
from unmochan.core.perturbation import diagnose_shadow_bands
from unmochan.core.slab import (
    VACUUM_THRESHOLD,
    detect_vacuum_axes,
    diagnose_slab_transform,
    vacuum_gap,
)
from unmochan.core.strain import diagnose_commensurability
from unmochan.core.tracking import track_spectral_peaks
from unmochan.io.serialization import read_ebs, write_ebs
from unmochan.io.vasp import read_poscar


@app.command("strain")
def strain_command(
    primitive: Annotated[
        Path,
        typer.Option("--primitive", exists=True, readable=True, help="Primitive POSCAR."),
    ],
    supercell: Annotated[
        Path,
        typer.Option("--supercell", exists=True, readable=True, help="Supercell POSCAR."),
    ],
    atol: Annotated[
        float,
        typer.Option("--atol", min=0.0, help="Residual below which the pair counts as exact."),
    ] = 1e-6,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the report as JSON."),
    ] = None,
) -> None:
    """Measure how far a supercell is from commensurate with its reference.

    Unfolding needs an exact integer relation A_sc = T A_pc.  A relaxed defect
    cell, a strained heterostructure layer or a commensurate approximant of a
    twisted stack misses it, and `weights` or `unfold` then refuse the pair.
    This says by how much, and what it would cost: the residual strain, the
    rigid rotation, and the resulting error in the reported k-vector, which
    grows linearly from zero at Gamma to its largest value at the zone corner.
    """

    report = diagnose_commensurability(
        read_poscar(primitive),
        read_poscar(supercell),
        atol=atol,
    )
    console.print(report.summary())

    table = Table(title="Residual strain tensor (infinitesimal)")
    for axis in ("x", "y", "z"):
        table.add_column(axis, justify="right")
    for row in report.strain:
        table.add_row(*(f"{value:+.3e}" for value in row))
    console.print(table)

    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")


@app.command("slab")
def slab_command(
    structure: Annotated[
        Path,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="Supercell POSCAR of the slab, wire or molecule in a box.",
        ),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Supercell matrix T, nine numbers."),
    ],
    axis: Annotated[
        list[int] | None,
        typer.Option(
            "--axis",
            help=(
                "Non-periodic axis, repeatable. Detected from the vacuum gaps of "
                "the structure when omitted."
            ),
        ),
    ] = None,
    vacuum: Annotated[
        float,
        typer.Option(
            "--vacuum",
            min=0.0,
            help="Gap above which an axis counts as non-periodic, in angstrom.",
        ),
    ] = VACUUM_THRESHOLD,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the report as JSON."),
    ] = None,
) -> None:
    """Check that a partially periodic cell is unfolded along its periodic axes only.

    A slab is periodic in two directions; the third holds vacuum, whose
    thickness is a convergence parameter and not a period.  A transform that
    folds along it produces a band that disperses purely because of that
    thickness, and an unfolding path along it asks for weights that are all
    zero.  This reports which axes carry vacuum, whether the transform leaves
    them alone, and how many perpendicular k-points are artifacts.
    """

    cell = read_poscar(structure)
    gaps = {index: vacuum_gap(cell, index) for index in (0, 1, 2)}
    axes = tuple(axis) if axis else detect_vacuum_axes(cell, threshold=vacuum)

    table = Table(title="Largest gap along each axis")
    table.add_column("axis", justify="right")
    table.add_column("gap (A)", justify="right")
    table.add_column("treated as", justify="left")
    for index, gap in gaps.items():
        table.add_row(
            str(index),
            f"{gap:.3f}",
            "non-periodic" if index in axes else "periodic",
        )
    console.print(table)

    report = diagnose_slab_transform(parse_matrix(matrix), list(axes))
    console.print(report.summary())
    if report.valid and axes:
        console.print(
            "Keep the perpendicular component of the unfolding k-path fixed, and "
            "pass the same axes to slab_weights_from_coefficients to collapse the "
            "plane-wave list over them."
        )

    if json_out is not None:
        payload = report.to_dict()
        payload["vacuum_gaps"] = [gaps[index] for index in (0, 1, 2)]
        json_out.write_text(json.dumps(payload, indent=2))
        console.print(f"Wrote {json_out}")


@app.command("dispersion")
def dispersion_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    broadening: Annotated[
        float,
        typer.Option("--broadening", min=0.0, help="Kernel width in the energy unit of the file."),
    ],
    kind: Annotated[
        str, typer.Option("--kind", help="Broadening kernel: gaussian or lorentzian.")
    ] = "gaussian",
    emin: Annotated[
        float | None,
        typer.Option("--emin", help="Lower end of the energy grid; default the lowest band."),
    ] = None,
    emax: Annotated[
        float | None,
        typer.Option("--emax", help="Upper end of the energy grid; default the highest band."),
    ] = None,
    points: Annotated[int, typer.Option("--points", min=3, help="Number of energy samples.")] = 601,
    min_intensity: Annotated[
        float,
        typer.Option("--min-intensity", min=0.0, help="Ignore peaks below this intensity."),
    ] = 0.0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the peak table."),
    ] = None,
) -> None:
    """Extract a peak dispersion from an unfolded spectral function.

    An unfolded band structure is a spectral function, not a set of bands: this
    locates the maxima of A(k, .) at every k-point and refines each one on the
    parabola through its neighbouring samples.

    The broadening width is a hard resolution limit -- two states closer than
    one width appear as a single peak, at an energy where neither of them is --
    so the number of k-points where that happens is reported alongside.
    """

    if broadening <= 0.0:
        raise typer.BadParameter("--broadening must be positive")
    broadening_kind = parse_broadening_kind(kind)

    structure, _ = read_ebs(input_path)
    energies = structure.shifted_energies()
    lower = float(energies.min() - 5.0 * broadening) if emin is None else emin
    upper = float(energies.max() + 5.0 * broadening) if emax is None else emax
    if not upper > lower:
        raise typer.BadParameter("--emax must exceed --emin")
    grid = np.linspace(lower, upper, points)

    peaks = spectral_peaks(
        structure,
        grid,
        broadening=broadening,
        kind=broadening_kind,
        min_intensity=min_intensity,
    )
    merged = unresolvable_pairs(structure, broadening=broadening)

    console.print(f"Peaks found: {len(peaks)} over {structure.n_kpoints} k-points")
    if merged:
        affected = len({pair.kpoint_index for pair in merged})
        console.print(
            f"[yellow]{len(merged)} state pairs at {affected} k-points are closer than "
            "the broadening width: they merge into a single peak lying between them.[/yellow]"
        )
    else:
        console.print("Every pair of weighted states is separated by more than the width.")

    if out is not None:
        lines = ["# kpoint_index  distance  energy  intensity"]
        lines += [
            f"{peak.kpoint_index:d} {peak.distance:.10g} {peak.energy:.10g} {peak.intensity:.10g}"
            for peak in peaks
        ]
        out.write_text("\n".join(lines) + "\n")
        console.print(f"Wrote {out}")


def _read_matrix_file(path: Path) -> np.ndarray:
    """Read a square matrix from ``.npy`` (any dtype) or whitespace-separated text."""

    if path.suffix == ".npy":
        array = np.asarray(np.load(path))
    else:
        array = np.atleast_2d(np.loadtxt(path))
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise typer.BadParameter(f"{path} does not hold a square matrix")
    return array


@app.command("overlap")
def overlap_command(
    matrix_path: Annotated[
        Path,
        typer.Option(
            "--overlap",
            exists=True,
            readable=True,
            help="Overlap matrix S: a .npy array, or a text file of rows of numbers.",
        ),
    ],
    hamiltonian_path: Annotated[
        Path | None,
        typer.Option(
            "--hamiltonian",
            exists=True,
            readable=True,
            help="Optional H, same shape as S; solves H c = E S c in the retained subspace.",
        ),
    ] = None,
    threshold: Annotated[
        float,
        typer.Option(
            "--threshold",
            min=0.0,
            help="Overlap eigenvalues below this count as null directions.",
        ),
    ] = DEFAULT_OVERLAP_THRESHOLD,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the diagnosis as JSON."),
    ] = None,
) -> None:
    """Report how close an LCAO overlap matrix is to singular.

    An unfolding weight in a non-orthogonal basis is `c* S <kernel> c / c* S c`,
    so a near-null direction of S makes it 0/0.  This prints the eigenvalue
    range and condition number of S, the cheap O(n^2) diagonal-dominance
    certificate of positive definiteness, and how many directions fall below the
    threshold.  With `--hamiltonian` it also solves the generalized problem by
    canonical orthogonalization, which drops those directions instead of
    failing on them.
    """

    if threshold <= 0.0:
        raise typer.BadParameter("--threshold must be positive")
    overlap = _read_matrix_file(matrix_path)
    report = diagnose_overlap_conditioning(overlap, threshold=threshold)
    console.print(report.summary())
    if report.certified_by_dominance:
        console.print(
            f"Diagonal dominance certifies a smallest eigenvalue of at least "
            f"{report.dominance_bound:.3e} without any factorization."
        )
    else:
        console.print(
            "[yellow]Diagonal dominance certifies nothing here; the eigenvalues "
            "above are the answer.[/yellow]"
        )

    payload = report.to_dict()
    if hamiltonian_path is not None:
        hamiltonian = _read_matrix_file(hamiltonian_path)
        if hamiltonian.shape != overlap.shape:
            raise typer.BadParameter("--hamiltonian must have the same shape as --overlap")
        energies, _ = solve_generalized_eigenproblem_truncated(
            hamiltonian, overlap, threshold=threshold
        )
        dropped = overlap.shape[0] - energies.size
        console.print(
            f"Canonical orthogonalization keeps {energies.size} of {overlap.shape[0]} "
            f"directions ({dropped} dropped); energies in "
            f"[{energies[0]:.6g}, {energies[-1]:.6g}]."
        )
        table = Table(title="Eigenvalues of the retained subspace")
        table.add_column("state", justify="right")
        table.add_column("energy", justify="right")
        for index, energy in enumerate(energies):
            table.add_row(str(index), f"{energy:+.6g}")
        console.print(table)
        payload["energies"] = [float(value) for value in energies]
        payload["n_retained"] = int(energies.size)

    if json_out is not None:
        json_out.write_text(json.dumps(payload, indent=2))
        console.print(f"Wrote {json_out}")


@app.command("fermi")
def fermi_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    electrons: Annotated[
        float | None,
        typer.Option("--electrons", help="Electrons per primitive cell."),
    ] = None,
    supercell_electrons: Annotated[
        float | None,
        typer.Option(
            "--supercell-electrons",
            help="Electrons per supercell; divided by --multiplicity to get the above.",
        ),
    ] = None,
    multiplicity: Annotated[
        int,
        typer.Option("--multiplicity", min=1, help="|det T|, the number of primitive cells."),
    ] = 1,
    temperature: Annotated[
        float,
        typer.Option("--temperature", min=0.0, help="Fermi-Dirac smearing width kB*T."),
    ] = 0.025,
    spin_degeneracy: Annotated[
        float,
        typer.Option("--spin-degeneracy", min=0.0, help="2 for a spin-degenerate calculation."),
    ] = 2.0,
    align: Annotated[
        Path | None,
        typer.Option("--align", help="Write a copy of the run with the reference at the level."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the report as JSON."),
    ] = None,
) -> None:
    """Locate the Fermi level of an unfolded run from the electron count.

    The count is per *primitive* cell: an unfolded band structure lives in the
    primitive zone even though the calculation was a supercell one.  Pass
    `--supercell-electrons` with `--multiplicity |det T|` to have the division
    done and checked here.

    The k-points must sample the Brillouin zone with equal weight.  A
    high-symmetry path is not such a sample and the number this returns for one
    is not an electron count.
    """

    if temperature <= 0.0:
        raise typer.BadParameter("--temperature must be positive")
    if (electrons is None) == (supercell_electrons is None):
        raise typer.BadParameter("give exactly one of --electrons and --supercell-electrons")
    if electrons is None:
        assert supercell_electrons is not None
        electrons = electrons_per_primitive_cell(supercell_electrons, multiplicity)
        console.print(
            f"{supercell_electrons:g} electrons over {multiplicity} primitive cells "
            f"= {electrons:g} per primitive cell"
        )

    structure, manifest = read_ebs(input_path)
    report = find_fermi_level(
        structure,
        electrons,
        temperature=temperature,
        spin_degeneracy=spin_degeneracy,
    )
    console.print(report.summary())
    if not report.converged:
        console.print(
            "[yellow]The electron count did not reach the tolerance: the bracket "
            "collapsed to the precision of a double before it did.[/yellow]"
        )

    if align is not None:
        write_ebs(align, align_reference(structure, report.fermi), manifest=manifest)
        console.print(f"Wrote {align} with the reference energy at the Fermi level")

    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")


@app.command("dos")
def dos_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    broadening: Annotated[
        float,
        typer.Option("--broadening", min=0.0, help="Kernel width in the energy unit of the file."),
    ],
    kind: Annotated[
        str, typer.Option("--kind", help="Broadening kernel: gaussian or lorentzian.")
    ] = "gaussian",
    multiplicity: Annotated[
        int,
        typer.Option("--multiplicity", min=1, help="|det T|, the number of primitive cells."),
    ] = 1,
    kpoint_weights: Annotated[
        Path | None,
        typer.Option(
            "--kpoint-weights",
            exists=True,
            readable=True,
            help="Brillouin-zone weights, one per k-point, whitespace separated. Default: uniform.",
        ),
    ] = None,
    points: Annotated[int, typer.Option("--points", min=2, help="Energy grid points.")] = 601,
    emin: Annotated[
        float | None, typer.Option("--emin", help="Lower end of the energy grid.")
    ] = None,
    emax: Annotated[
        float | None, typer.Option("--emax", help="Upper end of the energy grid.")
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write energy, unfolded DOS, integrated DOS and reference."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the conservation report as JSON."),
    ] = None,
) -> None:
    """The unfolded density of states, and the check that it is conserved.

    Unfolding redistributes spectral weight in k but must not create or destroy
    any: summed over a complete fiber the unfolded weight of a band is one, so
    |det T| times the unfolded density of states is the supercell density of
    states, and the unfolded curve integrates to n_bands / |det T| states per
    primitive cell.  Both comparisons are reported.

    A deficit means weight is missing -- an incomplete fiber, a truncated
    plane-wave set, a pseudo-wavefunction norm below one.  An excess cannot
    happen for weights in [0, 1] and is a bug; unlike the state count it is
    meaningful on a band path, which samples fibers only partly.

    The k-points must sample the Brillouin zone; --kpoint-weights supplies the
    mesh weights, and the default is uniform.  On a high-symmetry path the curve
    is a path average and not a density of states.
    """

    if broadening <= 0.0:
        raise typer.BadParameter("--broadening must be positive")
    broadening_kind = parse_broadening_kind(kind)
    structure, _ = read_ebs(input_path)

    weights = None
    if kpoint_weights is not None:
        weights = np.fromstring(kpoint_weights.read_text(encoding="utf-8"), sep=" ")
        if weights.size != structure.n_kpoints:
            raise typer.BadParameter(
                f"--kpoint-weights has {weights.size} entries for {structure.n_kpoints} k-points"
            )

    grid = None
    if (emin is None) != (emax is None):
        raise typer.BadParameter("give both --emin and --emax, or neither")
    if emin is not None and emax is not None:
        if emax <= emin:
            raise typer.BadParameter("--emax must be above --emin")
        grid = np.linspace(emin, emax, points)

    curve = unfolded_dos(
        structure,
        grid,
        broadening=broadening,
        kind=broadening_kind,
        kpoint_weights=weights,
        multiplicity=multiplicity,
        n_points=points,
    )
    reference = supercell_dos(
        structure,
        curve.energies,
        broadening=broadening,
        kind=broadening_kind,
        kpoint_weights=weights,
    )
    report = diagnose_dos_conservation(
        structure,
        broadening=broadening,
        kind=broadening_kind,
        kpoint_weights=weights,
        multiplicity=multiplicity,
        energy_grid=curve.energies,
    )

    table = Table(title="Unfolded density of states")
    table.add_column("quantity")
    table.add_column("value", justify="right")
    table.add_row("states, unfolded (per primitive cell)", f"{report.unfolded_states:.6f}")
    table.add_row("states, expected = n_bands / |det T|", f"{report.expected_states:.6f}")
    table.add_row("states, supercell (per supercell)", f"{report.supercell_states:.6f}")
    table.add_row("state count error", f"{report.states_error:+.3e}")
    table.add_row("max excess over the unweighted curve", f"{report.max_excess:+.3e}")
    table.add_row("max |det T| x DOS above the supercell", f"{report.max_fiber_excess:+.3e}")
    table.add_row("max |det T| x DOS below it", f"{report.max_fiber_deficit:+.3e}")
    table.add_row("states outside the energy grid", f"{curve.tail_loss:.3e}")
    console.print(table)

    if not report.is_bounded():
        console.print(
            "[red]The unfolded density of states exceeds the supercell one: with "
            "weights in [0, 1] that is impossible, so the weights are wrong.[/red]"
        )
    elif not report.is_conserved():
        console.print(
            "[yellow]Weight is missing: an incomplete fiber, a truncated basis or a "
            "pseudo-wavefunction norm below one. `unmochan norms` separates those.[/yellow]"
        )

    if out is not None:
        lines = ["# energy  dos_unfolded  integrated  dos_supercell"]
        for index, energy in enumerate(curve.energies):
            lines.append(
                f"{energy: .8f}  {curve.dos[index]: .8e}  "
                f"{curve.integrated[index]: .8e}  {reference.dos[index]: .8e}"
            )
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"Wrote {out}")

    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")


@app.command("branches")
def branches_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    broadening: Annotated[
        float,
        typer.Option("--broadening", min=0.0, help="Kernel width in the energy unit of the file."),
    ],
    kind: Annotated[
        str, typer.Option("--kind", help="Broadening kernel: gaussian or lorentzian.")
    ] = "gaussian",
    points: Annotated[
        int, typer.Option("--points", min=3, help="Energy grid points for the peak search.")
    ] = 801,
    emin: Annotated[
        float | None, typer.Option("--emin", help="Lower end of the energy grid.")
    ] = None,
    emax: Annotated[
        float | None, typer.Option("--emax", help="Upper end of the energy grid.")
    ] = None,
    min_intensity: Annotated[
        float, typer.Option("--min-intensity", min=0.0, help="Ignore peaks weaker than this.")
    ] = 0.0,
    max_jump: Annotated[
        float | None,
        typer.Option("--max-jump", help="End a branch at an energy step larger than this."),
    ] = None,
    min_length: Annotated[
        int, typer.Option("--min-length", min=1, help="Drop branches shorter than this.")
    ] = 1,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the path abscissa and one column per branch."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the branches and ambiguities as JSON."),
    ] = None,
) -> None:
    """Connect the peaks of A(k, E) into band branches.

    The peaks of one k-point are matched to those of the next by the
    order-preserving assignment of least total squared energy jump, which
    `matchCost_id_le` shows to be the optimal matching outright -- sorting the
    peaks *is* the algorithm, and the nearest-free-partner rule is provably
    worse.

    Where two peaks come within the resolution of the broadening the matching
    has an exact tie: a crossing and an anticrossing look identical in a peak
    table, and the branch labels either side are not determined.  Those
    k-points are listed rather than resolved, together with the places where a
    branch begins or ends inside the path -- a merge, or weight that has gone.
    """

    if broadening <= 0.0:
        raise typer.BadParameter("--broadening must be positive")
    broadening_kind = parse_broadening_kind(kind)
    structure, _ = read_ebs(input_path)

    energies = structure.shifted_energies()
    if (emin is None) != (emax is None):
        raise typer.BadParameter("give both --emin and --emax, or neither")
    if emin is None or emax is None:
        margin = 5.0 * broadening
        lower = float(energies.min()) - margin
        upper = float(energies.max()) + margin
    else:
        lower, upper = emin, emax
    if upper <= lower:
        raise typer.BadParameter("--emax must be above --emin")
    grid = np.linspace(lower, upper, points)

    tracking = track_spectral_peaks(
        structure,
        grid,
        broadening=broadening,
        kind=broadening_kind,
        min_intensity=min_intensity,
        max_jump=math.inf if max_jump is None else max_jump,
        min_length=min_length,
    )

    table = Table(title=f"{tracking.n_branches} branches")
    table.add_column("branch", justify="right")
    table.add_column("k-points", justify="right")
    table.add_column("energy range")
    table.add_column("max jump", justify="right")
    for index, branch in enumerate(tracking.branches):
        values = branch.energies[branch.present]
        table.add_row(
            str(index),
            f"{branch.length}",
            f"{values.min(): .4f} .. {values.max(): .4f}",
            f"{branch.max_jump:.4f}",
        )
    console.print(table)

    degenerate = [c for c in tracking.crossings if c.kind == "degenerate"]
    termini = [c for c in tracking.crossings if c.kind == "terminus"]
    if degenerate:
        console.print(
            f"[yellow]{len(degenerate)} k-points carry peaks closer than the broadening: "
            "a crossing and an anticrossing are indistinguishable there.[/yellow]"
        )
    if termini:
        console.print(
            f"[yellow]{len(termini)} branches begin or end inside the path: either two "
            "peaks merged, or the spectral weight went away.[/yellow]"
        )

    if out is not None:
        lines = ["# distance  " + "  ".join(f"branch{i}" for i in range(tracking.n_branches))]
        energy_table = tracking.energy_table()
        for index, distance in enumerate(tracking.distances):
            row = "  ".join(
                "nan" if np.isnan(value) else f"{value: .8e}" for value in energy_table[:, index]
            )
            lines.append(f"{distance: .8f}  {row}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        console.print(f"Wrote {out}")

    if json_out is not None:
        json_out.write_text(json.dumps(tracking.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")


@app.command("shadow")
def shadow_command(
    input_path: Annotated[
        Path,
        typer.Option(
            "--input",
            exists=True,
            readable=True,
            help="Effective band structure, JSON or HDF5 as written by `unfold --json`.",
        ),
    ],
    window: Annotated[
        float | None,
        typer.Option(
            "--window",
            help="Only consider states within this energy of the reference energy.",
        ),
    ] = None,
    min_pair_fraction: Annotated[
        float,
        typer.Option(
            "--min-pair-fraction",
            min=0.0,
            max=1.0,
            help="Ignore k-points where the pair holds less than this share of the weight.",
        ),
    ] = 0.9,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the per-k-point inversion as JSON."),
    ] = None,
) -> None:
    """Measure a weak superlattice perturbation from its shadow bands.

    A charge-density wave, a Peierls dimerization or an ordered alloy leaves a
    faint replica of the primitive band in the unfolded picture.  Away from
    accidental degeneracies the replica and the main band are the two states of
    a two-level problem, so the coupling that produced them can be read back
    off their splitting and their weights:

    ``|V| = dE sqrt(w+ w-)``  (`UnfoldLab.Shadow.abs_coupling_eq`).

    The inversion is exact for a genuine pair, and meaningless where other
    states carry weight -- so the weight left outside the pair is reported at
    every k-point and k-points below `--min-pair-fraction` are excluded from
    the average.  The pair is chosen by weight, not by physics: use `--window`
    to keep the search near the gap of interest.
    """

    structure, _ = read_ebs(input_path)
    diagnosis = diagnose_shadow_bands(structure, energy_window=window)
    estimate = diagnosis.coupling_estimate(minimum_pair_fraction=min_pair_fraction)
    fractions = diagnosis.pair_fraction
    kept = int(np.count_nonzero(fractions >= min_pair_fraction))

    table = Table(title="Shadow-band inversion")
    table.add_column("distance", justify="right")
    table.add_column("main", justify="right")
    table.add_column("shadow", justify="right")
    table.add_column("w(main)", justify="right")
    table.add_column("w(shadow)", justify="right")
    table.add_column("|V|", justify="right")
    table.add_column("pair", justify="right")
    for index in range(diagnosis.distances.size):
        table.add_row(
            f"{diagnosis.distances[index]: .4f}",
            f"{diagnosis.main_energies[index]: .4f}",
            f"{diagnosis.shadow_energies[index]: .4f}",
            f"{diagnosis.main_weights[index]:.4f}",
            f"{diagnosis.shadow_weights[index]:.4f}",
            f"{diagnosis.couplings[index]:.4f}",
            f"{fractions[index]:.3f}",
        )
    console.print(table)

    if kept == 0:
        console.print(
            "[yellow]No k-point keeps enough of its weight in two states: the "
            "two-level description does not apply here.[/yellow]"
        )
    else:
        console.print(
            f"Mean |V| over {kept} of {diagnosis.distances.size} k-points: {estimate:.6f}"
        )
        spread = float(np.std(diagnosis.couplings[fractions >= min_pair_fraction]))
        if spread > 0.1 * abs(estimate):
            console.print(
                "[yellow]The estimate scatters by more than 10% along the path: a "
                "single matrix element does not describe this perturbation.[/yellow]"
            )

    if json_out is not None:
        payload = diagnosis.to_dict()
        payload["coupling_estimate"] = estimate
        payload["min_pair_fraction"] = float(min_pair_fraction)
        json_out.write_text(json.dumps(payload, indent=2))
        console.print(f"Wrote {json_out}")


def _read_kpoint_table(path: Path) -> list[list[float]]:
    """First three numeric columns of a whitespace-separated k-point file.

    Accepts the TSV that ``unmochan mesh --kpoints`` writes (header line and
    trailing columns included) as well as a bare three-column list.
    """

    points: list[list[float]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        tokens = line.split()
        if not tokens or line.lstrip().startswith("#"):
            continue
        try:
            values = [float(token) for token in tokens[:3]]
        except ValueError:
            if number == 1:
                continue  # a header line
            raise typer.BadParameter(f"{path}:{number}: expected three numbers") from None
        if len(values) != 3:
            raise typer.BadParameter(f"{path}:{number}: expected three numbers")
        points.append(values)
    if not points:
        raise typer.BadParameter(f"{path}: no k-points found")
    return points


@app.command("bz")
def bz_command(
    structure: Annotated[
        Path | None,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="POSCAR whose reciprocal lattice defines the zone.",
        ),
    ] = None,
    point: Annotated[
        list[str] | None,
        typer.Option("--point", help="k-point in fractional coordinates, repeatable: x,y,z."),
    ] = None,
    kpoints_in: Annotated[
        Path | None,
        typer.Option(
            "--kpoints",
            exists=True,
            readable=True,
            help="Read k-points from a file; the first three columns are used.",
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Write the reduced k-points as TSV."),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Write the report as JSON."),
    ] = None,
) -> None:
    """Reduce k-points into the first Brillouin zone and say whether it matters.

    Fractional coordinates name a k-point only modulo the reciprocal lattice.
    Componentwise rounding into ``[-1/2, 1/2]`` -- what ``rint`` does -- is the
    right representative for the matching arithmetic but not for geometry: for a
    non-orthogonal cell it can leave a k-point well outside the zone, and every
    distance, radius and plot then quotes the wrong point.  This command reports
    the Wigner-Seitz representative, its distance to the nearest Bragg plane,
    and how far rounding would have been off.
    """

    reciprocal = read_poscar(structure).reciprocal_lattice if structure is not None else None
    if reciprocal is None:
        console.print(
            "[yellow]No --structure given: lengths are fractional Euclidean norms, "
            "which are only the physical ones for a cubic cell.[/yellow]"
        )

    requested: list[list[float]] = []
    if point:
        requested.extend([float(value) for value in parse_kpoint(raw).fractional] for raw in point)
    if kpoints_in is not None:
        requested.extend(_read_kpoint_table(kpoints_in))
    if not requested:
        raise typer.BadParameter("give at least one --point or a --kpoints file")

    report = diagnose_bz_reduction(requested, reciprocal)

    console.print(
        f"Reciprocal basis orthogonal: {report.orthogonal_basis}; "
        f"inscribed radius (a k-point closer to Gamma than this is in the zone): "
        f"{report.inscribed_radius:.6f}"
    )

    table = Table(title="First Brillouin zone representatives")
    for label in ("input", "reduced", "shift", "|k|", "|k| rounded", "to Bragg plane"):
        table.add_column(label)
    for values in requested:
        reduced = reduce_to_first_bz(values, reciprocal)
        margin = zone_boundary_distance(reduced.fractional, reciprocal)
        table.add_row(
            " ".join(f"{value: .6f}" for value in values),
            " ".join(f"{value: .6f}" for value in reduced.fractional),
            " ".join(f"{value:d}" for value in reduced.shift),
            f"{reduced.length:.6f}",
            f"{reduced.rounded_length:.6f}",
            f"{margin:.6f}",
        )
    console.print(table)

    style = "green" if report.rounding_is_exact else "yellow"
    console.print(f"[{style}]{report.summary()}[/{style}]")

    if out is not None:
        lines = ["k1\tk2\tk3"]
        lines.extend("\t".join(f"{value:.12f}" for value in row) for row in report.reduced)
        out.write_text("\n".join(lines) + "\n")
        console.print(f"Wrote {report.n_points} reduced k-points to {out}")

    if json_out is not None:
        json_out.write_text(json.dumps(report.to_dict(), indent=2))
        console.print(f"Wrote {json_out}")
