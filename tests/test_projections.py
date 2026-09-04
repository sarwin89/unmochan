import pytest

from unmochan.core.projections import (
    ProjectionGroup,
    ProjectionSelector,
    parse_projection_selectors,
)


def test_parse_generic_projection_selectors():
    selectors = parse_projection_selectors(
        ["species:*", "atom:4", "orbital:d", "layer:0", "surface:top", "spin:z"]
    )

    assert [selector.to_string() for selector in selectors] == [
        "species:*",
        "atom:4",
        "orbital:d",
        "layer:0",
        "surface:top",
        "spin:z",
    ]
    assert selectors[1].value == 4


def test_parse_role_projection_tags():
    selector = ProjectionSelector.parse("adsorbate")

    assert selector.namespace == "role"
    assert selector.value == "adsorbate"
    assert selector.to_string() == "adsorbate"


def test_rejects_material_specific_alias_as_core_selector():
    with pytest.raises(ValueError, match="unsupported projection namespace"):
        ProjectionSelector.parse("custom_material:d")


def test_projection_group_from_config_expands_generic_selectors():
    group = ProjectionGroup.from_config(
        "example_group",
        {
            "selector": {
                "species": ["A", "B"],
                "orbitals": ["d"],
                "layer": 0,
                "distance_from_defect": 4.0,
            }
        },
    )

    assert [selector.to_string() for selector in group.selectors] == [
        "species:A",
        "species:B",
        "orbital:d",
        "layer:0",
        "distance_from_defect:4.0",
    ]
