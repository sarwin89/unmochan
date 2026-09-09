"""Shared console, prompts and argument parsers for the UnfoldLab CLI.

The command modules (`commands`, `backends`, `guided`) are split by workflow;
everything they have in common -- the rich console, the interactive prompts, and
the parsers that turn a command-line string into a validated object -- lives
here so that the split does not duplicate it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import numpy as np
import typer
from rich.console import Console
from rich.prompt import Confirm, Prompt

from unmochan.core.kpoints import KPoint
from unmochan.core.spacegroup import validate_primitive_operations
from unmochan.core.spectral import BroadeningKind
from unmochan.io.qe import read_gnu_blocks
from unmochan.io.vasp import read_poscar
from unmochan.workflows.backend import BackendCode

console = Console()


def prompt_existing_path(label: str) -> Path:
    while True:
        value = Path(Prompt.ask(label)).expanduser()
        if value.exists():
            return value
        console.print(f"[red]Path does not exist:[/red] {value}")


def prompt_optional_existing_path(label: str) -> Path | None:
    value = Prompt.ask(label, default="").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def prompt_output_path(label: str, default: str) -> Path:
    return Path(Prompt.ask(label, default=default)).expanduser()


def prompt_optional_int(label: str) -> int | None:
    value = Prompt.ask(label, default="").strip()
    return int(value) if value else None


def prompt_repeated(label: str) -> list[str]:
    values: list[str] = []
    while True:
        values.append(Prompt.ask(label))
        if not Confirm.ask("Add another?", default=False):
            return values


def parse_kpoint(raw: str) -> KPoint:
    label = None
    coords = raw
    if ":" in raw:
        label, coords = raw.split(":", 1)
    values = [float(part) for part in coords.split(",")]
    if len(values) != 3:
        raise typer.BadParameter("k-point must have three comma-separated coordinates")
    return KPoint(np.asarray(values, dtype=np.float64), label=label or None)


def parse_matrix(raw: str) -> np.ndarray:
    tokens = raw.replace(",", " ").split()
    if len(tokens) != 9:
        raise typer.BadParameter("matrix must contain exactly nine numeric entries")
    return np.array([float(token) for token in tokens], dtype=float).reshape(3, 3)


def parse_magmoms(raw: str, n_sites: int) -> np.ndarray:
    """Parse a ``--magmom`` string into per-site moments.

    Accepts one number per site (collinear, shape ``(n_sites,)``) or three
    numbers per site (noncollinear, shape ``(n_sites, 3)``), separated by spaces
    or commas, with VASP's ``count*value`` repetition allowed, as in
    ``"4*1.0 4*-1.0"``.
    """

    values: list[float] = []
    for token in raw.replace(",", " ").split():
        if "*" in token:
            count_text, _, value_text = token.partition("*")
            try:
                count = int(count_text)
                value = float(value_text)
            except ValueError as error:
                raise typer.BadParameter(f"--magmom: cannot read '{token}'") from error
            if count < 0:
                raise typer.BadParameter(f"--magmom: negative repetition in '{token}'")
            values.extend([value] * count)
            continue
        try:
            values.append(float(token))
        except ValueError as error:
            raise typer.BadParameter(f"--magmom: cannot read '{token}'") from error

    moments = np.array(values, dtype=float)
    if moments.size == n_sites:
        return moments
    if moments.size == 3 * n_sites:
        return moments.reshape(n_sites, 3)
    raise typer.BadParameter(
        f"--magmom must give one value per site ({n_sites}) or three "
        f"({3 * n_sites}), got {moments.size}"
    )


def check_symmetry_against_structure(
    structure_path: Path,
    operations: np.ndarray,
    transform: np.ndarray,
    magmom: str | None = None,
    *,
    symprec: float = 1e-5,
) -> None:
    """Reject supplied symmetry operations that the structure does not have.

    The unfolding kernel checks only that an operation is unimodular and
    compatible with the two lattices; whether the stored wavefunction it reuses
    really is the symmetry image of the requested state depends on the crystal,
    which the kernel never sees.  Whenever the caller has supplied a structure
    the check is made here instead.

    Without ``magmom`` the calculation is assumed nonmagnetic, so an operation
    that is a symmetry once ``k -> -k`` is adjoined is accepted; with moments it
    is not, and the moment pattern must be preserved as well.
    """

    cell = read_poscar(structure_path)
    moments = None if magmom is None else parse_magmoms(magmom, cell.n_sites)
    validation = validate_primitive_operations(
        cell,
        operations,
        transform,
        symprec=symprec,
        magmoms=moments,
        time_reversal=moments is None,
    )
    if validation.ok:
        return
    rejected = "\n".join(validation.summary()[1:])
    raise typer.BadParameter(
        "--symmetry supplies operations that are not symmetries of the structure in "
        f"{structure_path}; unfolding with them would reuse the wavefunction of an "
        "unrelated state and every sum rule would still be satisfied:\n"
        f"{rejected}\n"
        "Regenerate the file with `unmochan detect-symmetry` (adding --magmom for a "
        "spin-polarized run), or drop the structure to skip this check."
    )


def resolve_operations(path: Path | None, reuse_kpoints: bool) -> np.ndarray | None:
    """Turn the ``--symmetry`` / ``--reuse-kpoints`` pair into an operation list.

    ``None`` keeps the strict mode in which the wavefunction file must hold
    exactly the k-map's k-points in order.  An empty array relaxes that to
    matching by k-point, which is what a fiber k-map needs because the same
    supercell k-point then serves ``|det T|`` rows.
    """

    if path is not None:
        if reuse_kpoints:
            raise typer.BadParameter("--reuse-kpoints is implied by --symmetry; pass only one")
        return parse_operations(path)
    if reuse_kpoints:
        return np.zeros((0, 3, 3), dtype=float)
    return None


def parse_operations(path: Path) -> np.ndarray:
    """Read point-group operations, nine integers per non-empty line.

    The matrices act on primitive fractional reciprocal coordinates.  The
    identity is always implied, so a file may be empty; that still enables the
    symmetry-reduced code path, which then only allows stored k-points to be
    reordered and reused.
    """

    rows: list[list[float]] = []
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.replace(",", " ").split()
        if len(tokens) != 9:
            raise typer.BadParameter(
                f"{path}:{lineno}: a symmetry operation must be nine numeric entries"
            )
        rows.append([float(token) for token in tokens])
    return np.array(rows, dtype=float).reshape(-1, 3, 3)


def parse_code(raw: str) -> BackendCode:
    code = raw.strip().lower()
    if code == "vasp":
        return "vasp"
    if code == "qe":
        return "qe"
    raise typer.BadParameter("code must be vasp or qe")


def parse_broadening_kind(raw: str) -> BroadeningKind:
    kind = raw.strip().lower()
    if kind == "gaussian":
        return "gaussian"
    if kind == "lorentzian":
        return "lorentzian"
    raise typer.BadParameter("--kind must be gaussian or lorentzian")


def resolve_qe_save_dir(
    explicit_save_dir: Path | None,
    outdir: Path | None,
    prefix: str | None,
) -> Path | None:
    if explicit_save_dir is not None:
        return explicit_save_dir
    if prefix is None:
        return None
    return (outdir or Path(".")) / f"{prefix}.save"


def infer_n_bands(bands: Path | None, nbnd: int | None) -> int:
    if bands is not None:
        _, energies = read_gnu_blocks(bands)
        return int(energies.shape[1])
    if nbnd is not None:
        if nbnd <= 0:
            raise typer.BadParameter("--nbnd must be positive")
        return int(nbnd)
    raise typer.BadParameter("provide either --bands or --nbnd")


def click_exception_classes(name: str) -> tuple[type[Any], ...]:
    """Collect an exception class from every click implementation in play.

    Recent Typer releases ship a vendored copy of click, so the exceptions
    raised while parsing are not the ones exported by the top-level ``click``
    package.  Catching only ``click.exceptions`` lets a plain usage error escape
    as an unhandled traceback.
    """

    modules = [click.exceptions]
    vendored = getattr(getattr(typer, "_click", None), "exceptions", None)
    if vendored is not None and vendored is not click.exceptions:
        modules.append(vendored)

    classes: list[type[Any]] = []
    for module in modules:
        candidate = getattr(module, name, None)
        if isinstance(candidate, type) and issubclass(candidate, BaseException):
            if candidate not in classes:
                classes.append(candidate)
    return tuple(classes)
