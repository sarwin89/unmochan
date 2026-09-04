from __future__ import annotations

import importlib.util
from pathlib import Path

import unmochan

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_unmochan_is_loaded_from_the_src_tree() -> None:
    package_dir = Path(unmochan.__file__).resolve().parent

    assert package_dir == REPOSITORY_ROOT / "src" / "unmochan"


def test_legacy_unfoldlab_package_is_absent() -> None:
    assert not (REPOSITORY_ROOT / "unfoldlab").exists()
    assert importlib.util.find_spec("unfoldlab") is None
