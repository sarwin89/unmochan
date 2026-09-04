"""Symmetry commands: `detect-symmetry` and `check-symmetry`.

Both interrogate a structure rather than a finished run.  `detect-symmetry`
derives the operations a symmetry-reduced unfolding may use -- of the crystal,
or, with ``--magmom``, of the magnetic crystal.  `check-symmetry` audits an
operation file that already exists, which is the one hypothesis of
symmetry-reduced unfolding the kernel itself cannot test: it sees the two
lattices but never the atoms, so a matrix that is unimodular and commutes with
the supercell transform is accepted whether or not the crystal has it, and a
wrong one returns the weights of an unrelated state with every sum rule
satisfied.  The mathematics is in
``RequestProject/Unfolding/SymmetryHypothesis.lean``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from unmochan.cli.app import app
from unmochan.cli.common import (
    console,
    parse_magmoms,
    parse_matrix,
    parse_operations,
)
from unmochan.core.spacegroup import (
    detect_primitive_operations,
    space_group_operations,
    validate_primitive_operations,
)
from unmochan.io.vasp import read_poscar


@app.command("detect-symmetry")
def detect_symmetry_command(
    structure: Annotated[
        Path,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="Supercell POSCAR whose symmetry is to be detected.",
        ),
    ],
    matrix: Annotated[
        str | None,
        typer.Option(
            "--matrix",
            help=(
                "Supercell matrix T, nine numbers. When given, the operations are "
                "written in primitive fractional reciprocal coordinates, ready for "
                "--symmetry; otherwise the supercell's own space group is reported."
            ),
        ),
    ] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help=(
                "Write the operations, nine integers per line: the reciprocal "
                "operations S_pc when --matrix is given, the real-space rotations "
                "M otherwise."
            ),
        ),
    ] = None,
    symprec: Annotated[
        float,
        typer.Option("--symprec", min=0.0, help="Tolerance on fractional coordinates."),
    ] = 1e-5,
    time_reversal: Annotated[
        bool,
        typer.Option(
            "--time-reversal/--no-time-reversal",
            help=(
                "Also adjoin k -> -k, which a plane-wave code uses to reduce the "
                "mesh of any nonmagnetic system, centrosymmetric or not. Requires "
                "--matrix; not valid for a magnetic calculation or a spin texture."
            ),
        ),
    ] = False,
    magmom: Annotated[
        str | None,
        typer.Option(
            "--magmom",
            help=(
                "Local moments, one per site (collinear) or three per site "
                "(noncollinear), with VASP's n*value repetition allowed. Restricts "
                "the search to the operations of the magnetic structure, which is "
                "what a spin-polarized run is invariant under."
            ),
        ),
    ] = None,
) -> None:
    """Detect the symmetry operations of a structure for symmetry-reduced unfolding."""

    cell = read_poscar(structure)
    moments = None if magmom is None else parse_magmoms(magmom, cell.n_sites)
    if moments is not None and time_reversal:
        raise typer.BadParameter(
            "--time-reversal is not valid for a magnetic structure: the state at -K is "
            "the conjugate of the state at K only when the two spin channels are degenerate"
        )
    if matrix is None:
        if time_reversal:
            raise typer.BadParameter(
                "--time-reversal acts on reciprocal coordinates and needs --matrix"
            )
        operations = space_group_operations(cell, symprec=symprec, magmoms=moments)
        console.print(f"Space-group operations: {len(operations)}")
        nonsymmorphic = sum(0 if op.is_symmorphic else 1 for op in operations)
        console.print(f"Non-symmorphic among them: {nonsymmorphic}")
        rotations = (
            np.stack([op.rotation for op in operations])
            if operations
            else np.empty((0, 3, 3), dtype=int)
        )
    else:
        rotations = detect_primitive_operations(
            cell,
            parse_matrix(matrix),
            symprec=symprec,
            time_reversal=time_reversal,
            magmoms=moments,
        )
        console.print(
            f"Operations compatible with the supercell matrix: {len(rotations)} "
            "(primitive fractional reciprocal coordinates)"
        )
        if time_reversal:
            console.print("Time reversal (k -> -k) included; not valid for a magnetic system.")

    if out is not None:
        lines = [" ".join(str(int(value)) for value in row.reshape(-1)) for row in rotations]
        out.write_text("\n".join(lines) + ("\n" if lines else ""))
        console.print(f"Wrote {out}")


@app.command("check-symmetry")
def check_symmetry_command(
    structure: Annotated[
        Path,
        typer.Option(
            "--structure",
            exists=True,
            readable=True,
            help="Supercell POSCAR the operations are checked against.",
        ),
    ],
    matrix: Annotated[
        str,
        typer.Option("--matrix", help="Supercell matrix T, nine numbers."),
    ],
    symmetry: Annotated[
        Path,
        typer.Option(
            "--symmetry",
            exists=True,
            readable=True,
            help="The operation file that would be passed to `unfold --symmetry`.",
        ),
    ],
    symprec: Annotated[
        float,
        typer.Option("--symprec", min=0.0, help="Tolerance on fractional coordinates."),
    ] = 1e-5,
    magmom: Annotated[
        str | None,
        typer.Option(
            "--magmom",
            help=(
                "Local moments, one per site (collinear) or three per site "
                "(noncollinear); required for a spin-polarized run."
            ),
        ),
    ] = None,
    time_reversal: Annotated[
        bool,
        typer.Option(
            "--time-reversal/--no-time-reversal",
            help=(
                "Accept an operation that is a symmetry only once k -> -k is "
                "adjoined, which a nonmagnetic run may legitimately have used."
            ),
        ),
    ] = False,
) -> None:
    """Check symmetry operations against the crystal before unfolding with them.

    The unfolding kernel can only test that an operation is unimodular and
    compatible with the two lattices; that the *stored wavefunction* it reuses
    belongs to the state it claims is a physical hypothesis about the crystal,
    and a wrong operation gives weights of an unrelated state with every sum
    rule still satisfied.  This command performs that missing check.
    """

    cell = read_poscar(structure)
    moments = None if magmom is None else parse_magmoms(magmom, cell.n_sites)
    validation = validate_primitive_operations(
        cell,
        parse_operations(symmetry),
        parse_matrix(matrix),
        symprec=symprec,
        magmoms=moments,
        time_reversal=time_reversal,
    )
    for line in validation.summary():
        console.print(line)
    if not validation.ok:
        raise typer.Exit(code=1)
    console.print("[green]All supplied operations are symmetries of this structure.[/green]")
