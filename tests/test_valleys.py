from unfoldlab.core.valleys import ValleyDefinition


def test_valley_definition_contains_points_modulo_reciprocal_lattice():
    valley = ValleyDefinition(label="user_defined", center_frac=[0.95, 0.0, 0.0], radius=0.1)

    assert valley.contains([0.01, 0.0, 0.0])
    assert not valley.contains([0.5, 0.0, 0.0])


def test_valley_definition_from_config():
    valley = ValleyDefinition.from_config(
        "example",
        {"center_frac": [1 / 3, 1 / 3, 0.0], "radius": 0.05, "reference_bz": "layer_1"},
    )

    assert valley.label == "example"
    assert valley.reference_bz == "layer_1"
