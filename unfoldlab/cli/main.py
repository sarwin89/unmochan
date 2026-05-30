"""Command line interface for early UnfoldLab workflows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from unfoldlab import __version__
from unfoldlab.core.kpoints import KPoint
from unfoldlab.workflows.problem import UnfoldingProblem


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="unfoldlab")
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    subparsers = parser.add_subparsers(dest="namespace")

    vasp = subparsers.add_parser("vasp", help="VASP-oriented workflows.")
    vasp_sub = vasp.add_subparsers(dest="command")

    map_parser = vasp_sub.add_parser("map", help="Detect primitive-to-supercell mapping.")
    map_parser.add_argument("--primitive", required=True, type=Path)
    map_parser.add_argument("--supercell", required=True, type=Path)

    fold_parser = vasp_sub.add_parser(
        "fold-kpoints",
        help="Fold primitive k-points to supercell k-points.",
    )
    fold_parser.add_argument("--primitive", required=True, type=Path)
    fold_parser.add_argument("--supercell", required=True, type=Path)
    fold_parser.add_argument(
        "--kpoint",
        action="append",
        required=True,
        help="Primitive k-point as label:x,y,z or x,y,z. May be repeated.",
    )

    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.namespace == "vasp" and args.command == "map":
        return _vasp_map(args.primitive, args.supercell)
    if args.namespace == "vasp" and args.command == "fold-kpoints":
        return _vasp_fold_kpoints(args.primitive, args.supercell, args.kpoint)
    parser.print_help()
    return 2


def _vasp_map(primitive: Path, supercell: Path) -> int:
    problem = UnfoldingProblem.from_vasp(primitive=primitive, supercell=supercell)
    transform = problem.find_transformation()
    print(json.dumps(transform.to_dict(), indent=2))
    return 0


def _vasp_fold_kpoints(primitive: Path, supercell: Path, raw_kpoints: list[str]) -> int:
    problem = UnfoldingProblem.from_vasp(primitive=primitive, supercell=supercell)
    points = [_parse_kpoint(item) for item in raw_kpoints]
    mappings = problem.generate_supercell_kpoints(points)
    print(json.dumps([mapping.to_dict() for mapping in mappings], indent=2))
    return 0


def _parse_kpoint(raw: str) -> KPoint:
    label = None
    coords = raw
    if ":" in raw:
        label, coords = raw.split(":", 1)
    values = [float(part) for part in coords.split(",")]
    if len(values) != 3:
        raise argparse.ArgumentTypeError("k-point must have three comma-separated coordinates")
    return KPoint(values, label=label or None)


if __name__ == "__main__":
    raise SystemExit(main())
