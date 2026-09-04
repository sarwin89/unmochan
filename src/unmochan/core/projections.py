"""Material-agnostic projection selector model."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

GENERIC_PROJECTION_NAMESPACES = frozenset(
    {
        "species",
        "atom",
        "orbital",
        "layer",
        "sublattice",
        "region",
        "defect_shell",
        "surface",
        "interface",
        "valley",
        "spin",
        "role",
        "distance_from_defect",
    }
)

ROLE_TAGS = frozenset({"adsorbate", "substrate", "molecule"})

CONFIG_KEY_ALIASES = {
    "atoms": "atom",
    "orbitals": "orbital",
    "layers": "layer",
    "sublattices": "sublattice",
    "regions": "region",
    "defect_shells": "defect_shell",
    "surfaces": "surface",
    "interfaces": "interface",
    "valleys": "valley",
    "spins": "spin",
}


@dataclass(frozen=True)
class ProjectionSelector:
    """A single generic projection selector.

    The core parser intentionally recognizes generic namespaces only. Strings
    Material aliases should be represented through user-defined
    :class:`ProjectionGroup` objects, not hard-coded in core.
    """

    namespace: str
    value: str | int | float

    def __post_init__(self) -> None:
        namespace = self.namespace.strip()
        if namespace not in GENERIC_PROJECTION_NAMESPACES:
            raise ValueError(f"unsupported projection namespace: {namespace}")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "value", _coerce_value(self.value))

    @classmethod
    def parse(cls, raw: str) -> ProjectionSelector:
        """Parse a selector such as ``species:*`` or ``orbital:d``."""

        item = raw.strip()
        if item in ROLE_TAGS:
            return cls("role", item)
        if ":" not in item:
            raise ValueError(f"projection selector must use namespace:value syntax: {raw!r}")
        namespace, value = item.split(":", 1)
        return cls(namespace, value)

    @property
    def is_wildcard(self) -> bool:
        return self.value == "*"

    def to_string(self) -> str:
        if self.namespace == "role" and self.value in ROLE_TAGS:
            return str(self.value)
        return f"{self.namespace}:{self.value}"

    def to_dict(self) -> dict[str, str | int | float]:
        return {"namespace": self.namespace, "value": self.value}


@dataclass(frozen=True)
class ProjectionGroup:
    """User-defined group of generic selectors."""

    name: str
    selectors: tuple[ProjectionSelector, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("projection group name must not be empty")
        if not self.selectors:
            raise ValueError("projection group must contain at least one selector")
        object.__setattr__(self, "selectors", tuple(self.selectors))

    @classmethod
    def from_config(cls, name: str, config: Mapping[str, Any]) -> ProjectionGroup:
        """Build a group from a YAML-like mapping.

        The expected input shape is ``{"selector": {"species": ["A"], ...}}``.
        Lists create one selector per value. Scalar values create one selector.
        """

        selector_config = config.get("selector", config)
        if not isinstance(selector_config, Mapping):
            raise ValueError("projection group config must contain a selector mapping")

        selectors: list[ProjectionSelector] = []
        for key, raw_values in selector_config.items():
            namespace = CONFIG_KEY_ALIASES.get(str(key), str(key))
            values = raw_values if _is_sequence(raw_values) else [raw_values]
            for value in values:
                selectors.append(ProjectionSelector(namespace, value))
        return cls(name=name, selectors=tuple(selectors), metadata=dict(config.get("metadata", {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selectors": [selector.to_dict() for selector in self.selectors],
            "metadata": dict(self.metadata),
        }


def parse_projection_selectors(raw_selectors: Iterable[str]) -> tuple[ProjectionSelector, ...]:
    """Parse an iterable of projection selector strings."""

    return tuple(ProjectionSelector.parse(item) for item in raw_selectors)


def _coerce_value(value: Any) -> str | int | float:
    if isinstance(value, int | float):
        return value
    text = str(value).strip()
    if text == "*":
        return text
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, str | bytes | Mapping)
