"""JSON serialization of effective band structures and run manifests.

An unfolded band structure is expensive to produce and easy to mislabel: the
same energies and weights mean different things depending on the transformation
matrix, the spin channel, the energy reference, the weight mode, and which
files they came from.  A run therefore writes, next to its data, a *manifest*
recording exactly that, together with the size and SHA-256 digest of every
input file, so a table can be traced back to the calculation that produced it.

Both the band structure and the manifest are plain JSON: they are meant to be
read without this package.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from unmochan.core.spectral import EffectiveBandStructure

__all__ = [
    "EBS_SCHEMA",
    "RUN_MANIFEST_SCHEMA",
    "RunManifest",
    "SourceFile",
    "build_run_manifest",
    "read_ebs",
    "read_ebs_hdf5",
    "read_ebs_json",
    "read_run_manifest",
    "write_ebs",
    "write_ebs_hdf5",
    "write_ebs_json",
    "write_run_manifest",
]

#: Version of the on-disk layout, bumped when the schema changes incompatibly.
EBS_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1

#: Stable version-1 data-format identifiers, not Python import paths.
EBS_SCHEMA = "unfoldlab.effective_band_structure"
RUN_MANIFEST_SCHEMA = "unfoldlab.run_manifest"

#: Chunk size for hashing, so that a multi-gigabyte WAVECAR is not read at once.
_HASH_CHUNK = 1 << 20


@dataclass(frozen=True)
class SourceFile:
    """Identity of one input file: where it was, how big, and its digest."""

    role: str
    path: str
    size: int | None = None
    sha256: str | None = None

    @classmethod
    def from_path(cls, role: str, path: str | Path, *, digest: bool = True) -> SourceFile:
        resolved = Path(path)
        if not resolved.is_file():
            return cls(role=role, path=str(resolved))
        return cls(
            role=role,
            path=str(resolved),
            size=resolved.stat().st_size,
            sha256=_sha256(resolved) if digest else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> SourceFile:
        return cls(
            role=str(data["role"]),
            path=str(data["path"]),
            size=data.get("size"),
            sha256=data.get("sha256"),
        )


@dataclass(frozen=True)
class RunManifest:
    """Provenance of one unfolding run."""

    code: str
    weight_mode: str
    created: str
    package_version: str
    transformation: list[list[float]] | None = None
    reference_energy: float = 0.0
    spin_channel: int | None = None
    projection: str | None = None
    sources: tuple[SourceFile, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": RUN_MANIFEST_SCHEMA,
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "code": self.code,
            "weight_mode": self.weight_mode,
            "created": self.created,
            "package_version": self.package_version,
            "transformation": self.transformation,
            "reference_energy": self.reference_energy,
            "spin_channel": self.spin_channel,
            "projection": self.projection,
            "sources": [item.to_dict() for item in self.sources],
            "extras": dict(self.extras),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RunManifest:
        return cls(
            code=str(data["code"]),
            weight_mode=str(data["weight_mode"]),
            created=str(data["created"]),
            package_version=str(data["package_version"]),
            transformation=data.get("transformation"),
            reference_energy=float(data.get("reference_energy", 0.0)),
            spin_channel=data.get("spin_channel"),
            projection=data.get("projection"),
            sources=tuple(SourceFile.from_dict(item) for item in data.get("sources", ())),
            extras=dict(data.get("extras", {})),
        )


def build_run_manifest(
    *,
    code: str,
    weight_mode: str,
    transformation: ArrayLike | None = None,
    reference_energy: float = 0.0,
    spin_channel: int | None = None,
    projection: str | None = None,
    sources: Mapping[str, str | Path | None] | None = None,
    extras: Mapping[str, Any] | None = None,
    digest: bool = True,
) -> RunManifest:
    """Collect the provenance of a run.

    ``sources`` maps a role (``"wavecar"``, ``"kmap"``, ...) to a path; entries
    whose path is ``None`` are dropped, so a caller can pass its optional inputs
    unconditionally.  Set ``digest=False`` to skip hashing, which matters for a
    very large wavefunction file.
    """

    from unmochan import __version__

    files = tuple(
        SourceFile.from_path(role, path, digest=digest)
        for role, path in sorted((sources or {}).items())
        if path is not None
    )
    matrix = None
    if transformation is not None:
        matrix = np.asarray(transformation, dtype=float).tolist()
    return RunManifest(
        code=code,
        weight_mode=weight_mode,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        package_version=__version__,
        transformation=matrix,
        reference_energy=float(reference_energy),
        spin_channel=spin_channel,
        projection=projection,
        sources=files,
        extras=dict(extras or {}),
    )


def write_run_manifest(path: str | Path, manifest: RunManifest) -> Path:
    """Write a manifest as JSON."""

    target = Path(path)
    target.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n")
    return target


def read_run_manifest(path: str | Path) -> RunManifest:
    """Read a manifest written by :func:`write_run_manifest`."""

    return RunManifest.from_dict(json.loads(Path(path).read_text()))


def write_ebs_json(
    path: str | Path,
    ebs: EffectiveBandStructure,
    *,
    manifest: RunManifest | None = None,
) -> Path:
    """Write an effective band structure, and optionally its manifest, as JSON."""

    payload: dict[str, Any] = {
        "schema": EBS_SCHEMA,
        "schema_version": EBS_SCHEMA_VERSION,
        **ebs.to_dict(),
    }
    if manifest is not None:
        payload["manifest"] = manifest.to_dict()
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def read_ebs_json(path: str | Path) -> tuple[EffectiveBandStructure, RunManifest | None]:
    """Read a band structure written by :func:`write_ebs_json`.

    Returns the band structure and the manifest that was stored with it, if any.
    """

    data = json.loads(Path(path).read_text())
    schema = data.get("schema")
    if schema != EBS_SCHEMA:
        raise ValueError(f"{path} is not an unmochan effective band structure ({schema!r})")
    version = int(data.get("schema_version", 0))
    if version > EBS_SCHEMA_VERSION:
        raise ValueError(
            f"{path} was written with schema version {version}, which this version of "
            f"unmochan ({EBS_SCHEMA_VERSION}) does not understand"
        )
    manifest = data.get("manifest")
    return (
        EffectiveBandStructure.from_dict(data),
        None if manifest is None else RunManifest.from_dict(manifest),
    )


def write_ebs_hdf5(
    path: str | Path,
    ebs: EffectiveBandStructure,
    *,
    manifest: RunManifest | None = None,
    compression: str | None = "gzip",
) -> Path:
    """Write an effective band structure as HDF5.

    JSON keeps every number as text, which for a dense path -- thousands of
    k-points times hundreds of bands, times energies *and* weights -- is both
    large and slow to parse, and it loses the exact double-precision value.
    The HDF5 form stores the four arrays as native ``float64`` datasets, and
    the scalars, the metadata and the manifest as attributes, so the file is
    readable by any HDF5 tool without this package.

    ``compression`` is passed to ``h5py`` (``None`` disables it).
    """

    h5py = _import_h5py()
    target = Path(path)
    options: dict[str, Any] = {} if compression is None else {"compression": compression}
    with h5py.File(target, "w") as handle:
        handle.attrs["schema"] = EBS_SCHEMA
        handle.attrs["schema_version"] = EBS_SCHEMA_VERSION
        handle.attrs["reference_energy"] = float(ebs.reference_energy)
        handle.attrs["metadata"] = json.dumps(ebs.metadata, sort_keys=True, default=str)
        if manifest is not None:
            handle.attrs["manifest"] = json.dumps(manifest.to_dict(), sort_keys=True)
        for name, array in (
            ("kpoints", ebs.kpoints),
            ("energies", ebs.energies),
            ("weights", ebs.weights),
            ("distances", ebs.distances),
        ):
            handle.create_dataset(name, data=np.asarray(array, dtype=float), **options)
    return target


def read_ebs_hdf5(path: str | Path) -> tuple[EffectiveBandStructure, RunManifest | None]:
    """Read a band structure written by :func:`write_ebs_hdf5`."""

    h5py = _import_h5py()
    with h5py.File(Path(path), "r") as handle:
        schema = handle.attrs.get("schema")
        if isinstance(schema, bytes):
            schema = schema.decode()
        if schema != EBS_SCHEMA:
            raise ValueError(f"{path} is not an unmochan effective band structure ({schema!r})")
        version = int(handle.attrs.get("schema_version", 0))
        if version > EBS_SCHEMA_VERSION:
            raise ValueError(
                f"{path} was written with schema version {version}, which this version of "
                f"unmochan ({EBS_SCHEMA_VERSION}) does not understand"
            )
        ebs = EffectiveBandStructure(
            kpoints=np.asarray(handle["kpoints"], dtype=float),
            energies=np.asarray(handle["energies"], dtype=float),
            weights=np.asarray(handle["weights"], dtype=float),
            distances=np.asarray(handle["distances"], dtype=float),
            reference_energy=float(handle.attrs.get("reference_energy", 0.0)),
            metadata=json.loads(_attr_text(handle.attrs.get("metadata", "{}"))),
        )
        raw_manifest = handle.attrs.get("manifest")
        manifest = (
            None
            if raw_manifest is None
            else RunManifest.from_dict(json.loads(_attr_text(raw_manifest)))
        )
    return ebs, manifest


#: Suffixes read as HDF5 by :func:`read_ebs`.
HDF5_SUFFIXES = (".h5", ".hdf5")


def read_ebs(path: str | Path) -> tuple[EffectiveBandStructure, RunManifest | None]:
    """Read a stored band structure, picking the format from the suffix."""

    resolved = Path(path)
    if resolved.suffix.lower() in HDF5_SUFFIXES:
        return read_ebs_hdf5(resolved)
    return read_ebs_json(resolved)


def write_ebs(
    path: str | Path,
    ebs: EffectiveBandStructure,
    *,
    manifest: RunManifest | None = None,
) -> Path:
    """Write a band structure as HDF5 for ``.h5``/``.hdf5``, else as JSON."""

    resolved = Path(path)
    if resolved.suffix.lower() in HDF5_SUFFIXES:
        return write_ebs_hdf5(resolved, ebs, manifest=manifest)
    return write_ebs_json(resolved, ebs, manifest=manifest)


def _attr_text(value: Any) -> str:
    return value.decode() if isinstance(value, bytes) else str(value)


def _import_h5py() -> Any:
    try:
        import h5py
    except ModuleNotFoundError as error:  # pragma: no cover - depends on the environment
        raise ModuleNotFoundError(
            "HDF5 output needs h5py; install it with `pip install unmochan[io]`"
        ) from error
    return h5py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()
