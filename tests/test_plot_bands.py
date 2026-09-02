"""The band scatter plot, and in particular its dependency-free renderer.

``matplotlib`` is in the optional ``plot`` extra, so :func:`plot_unfolded_svg`
is the renderer an install without that extra actually uses.  It had never been
exercised, and it had two defects that only show up on real data: states
outside the requested energy window were drawn *outside* the axes box, and a
single non-finite energy set the whole y-axis to ``nan``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from unfoldlab.io.plot_bands import (
    color_from_weight,
    marker_area,
    plot_unfolded,
    plot_unfolded_svg,
)

TOP = 30.0
PLOT_HEIGHT = 500.0
BOTTOM = TOP + PLOT_HEIGHT


@dataclass(frozen=True)
class Path1D:
    """The only thing the plotter needs of a k-map."""

    distances: NDArray[np.float64]


def a_path(n_kpoints: int = 3) -> Path1D:
    return Path1D(distances=np.linspace(0.0, 1.0, n_kpoints))


def bands(n_kpoints: int = 3, n_bands: int = 4) -> NDArray[np.float64]:
    return np.linspace(-3.0, 3.0, n_kpoints * n_bands).reshape(n_kpoints, n_bands)


def weights(n_kpoints: int = 3, n_bands: int = 4) -> NDArray[np.float64]:
    return np.linspace(0.0, 1.0, n_kpoints * n_bands).reshape(n_kpoints, n_bands)


def circles(text: str) -> list[tuple[float, float, float]]:
    return [
        (float(cx), float(cy), float(r))
        for cx, cy, r in re.findall(r'<circle cx="([-\d.]+)" cy="([-\d.]+)" r="([-\d.]+)"', text)
    ]


def test_the_svg_has_one_marker_per_state(tmp_path: Path) -> None:
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), bands(), weights(), ([0.0], ["G"]))
    text = out.read_text()
    assert text.startswith("<svg")
    assert text.rstrip().endswith("</svg>")
    assert len(circles(text)) == 12


def test_the_extension_is_forced_to_svg(tmp_path: Path) -> None:
    out = plot_unfolded_svg(tmp_path / "bands.png", a_path(), bands(), weights())
    assert out.suffix == ".svg"
    assert out.exists()


def test_states_outside_the_energy_window_are_not_drawn_outside_the_axes(
    tmp_path: Path,
) -> None:
    """The old renderer scattered them across the labels and the title."""

    energies = np.array([[-10.0, 0.0, 10.0], [-9.0, 0.5, 9.0]])
    out = plot_unfolded_svg(
        tmp_path / "bands.svg",
        Path1D(distances=np.array([0.0, 1.0])),
        energies,
        np.ones_like(energies),
        ([0.0, 1.0], ["G", "X"]),
        emin=-1.0,
        emax=1.0,
    )
    drawn = circles(out.read_text())
    assert len(drawn) == 2  # the two states inside [-1, 1]
    assert all(TOP <= cy <= BOTTOM for _, cy, _ in drawn)


def test_the_window_is_taken_from_the_data_when_it_is_not_given(
    tmp_path: Path,
) -> None:
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), bands(), weights(), ([0.0], ["G"]))
    drawn = circles(out.read_text())
    assert len(drawn) == 12
    assert all(TOP - 1e-6 <= cy <= BOTTOM + 1e-6 for _, cy, _ in drawn)


def test_only_one_end_of_the_window_may_be_given(tmp_path: Path) -> None:
    energies = bands()
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), energies, weights(), emax=0.0)
    assert len(circles(out.read_text())) == int(np.sum(energies <= 0.0))


def test_a_single_nan_energy_costs_one_marker_and_not_the_figure(
    tmp_path: Path,
) -> None:
    energies = bands()
    energies[1, 2] = np.nan
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), energies, weights(), ([0.0], ["G"]))
    text = out.read_text()
    assert "nan" not in text.lower()
    assert len(circles(text)) == 11


def test_a_non_finite_weight_costs_one_marker(tmp_path: Path) -> None:
    w = weights()
    w[0, 1] = np.inf
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), bands(), w)
    assert len(circles(out.read_text())) == 11


def test_a_flat_band_still_gets_a_non_degenerate_axis(tmp_path: Path) -> None:
    energies = np.full((3, 2), 1.25)
    out = plot_unfolded_svg(tmp_path / "bands.svg", a_path(), energies, np.ones_like(energies))
    drawn = circles(out.read_text())
    assert len(drawn) == 6
    assert all(np.isfinite(cy) for _, cy, _ in drawn)


def test_a_single_k_point_still_gets_a_non_degenerate_axis(tmp_path: Path) -> None:
    out = plot_unfolded_svg(
        tmp_path / "bands.svg",
        Path1D(distances=np.array([0.5])),
        np.array([[0.0, 1.0]]),
        np.array([[1.0, 0.5]]),
    )
    drawn = circles(out.read_text())
    assert len(drawn) == 2
    assert all(np.isfinite(cx) for cx, _, _ in drawn)


def test_the_marker_area_is_affine_in_the_weight() -> None:
    areas = marker_area(np.array([0.0, 0.5, 1.0]), 28.0)
    assert np.allclose(areas, [3.0, 17.0, 31.0])
    # ...and the radii the SVG uses are the square roots of exactly that.
    assert np.allclose(np.sqrt(areas), [np.sqrt(3.0), np.sqrt(17.0), np.sqrt(31.0)])


def test_the_marker_area_clips_weights_to_the_unit_interval() -> None:
    assert np.allclose(marker_area(np.array([-0.5, 1.5]), 28.0), [3.0, 31.0])


def test_a_heavier_state_gets_a_bigger_marker(tmp_path: Path) -> None:
    out = plot_unfolded_svg(
        tmp_path / "bands.svg",
        Path1D(distances=np.array([0.0, 1.0])),
        np.array([[0.0], [0.0]]),
        np.array([[0.1], [0.9]]),
    )
    radii = [r for _, _, r in circles(out.read_text())]
    assert radii[1] > radii[0]


def test_the_fermi_shift_moves_the_zero_line_and_relabels_the_axis(
    tmp_path: Path,
) -> None:
    shifted = plot_unfolded_svg(
        tmp_path / "shifted.svg", a_path(), bands(), weights(), fermi=1.0
    ).read_text()
    assert "Energy - Ef (eV)" in shifted
    plain = plot_unfolded_svg(tmp_path / "plain.svg", a_path(), bands(), weights()).read_text()
    assert "Energy (eV)" in plain and "Energy - Ef" not in plain


def test_tick_labels_are_escaped(tmp_path: Path) -> None:
    out = plot_unfolded_svg(
        tmp_path / "bands.svg", a_path(), bands(), weights(), ([0.0], ["<M&M>"])
    )
    text = out.read_text()
    assert "&lt;M&amp;M&gt;" in text
    assert "<M&M>" not in text


def test_the_colour_scale_runs_from_dark_to_bright() -> None:
    assert color_from_weight(0.0) == "#440154"
    assert color_from_weight(1.0) == "#fde725"
    assert color_from_weight(-1.0) == color_from_weight(0.0)
    assert color_from_weight(2.0) == color_from_weight(1.0)
    for weight in np.linspace(0.0, 1.0, 21):
        assert re.fullmatch(r"#[0-9a-f]{6}", color_from_weight(float(weight)))


@pytest.mark.parametrize(
    ("energies", "weight_array", "message"),
    [
        (np.zeros(3), np.zeros(3), "energies must have shape"),
        (np.zeros((3, 4)), np.zeros((3, 5)), "same shape as energies"),
        (np.zeros((2, 4)), np.zeros((2, 4)), "one entry per k-point"),
    ],
)
def test_mismatched_shapes_are_rejected(
    tmp_path: Path,
    energies: NDArray[np.float64],
    weight_array: NDArray[np.float64],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_unfolded_svg(tmp_path / "bands.svg", a_path(), energies, weight_array)


def test_mismatched_tick_lists_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        plot_unfolded_svg(
            tmp_path / "bands.svg",
            a_path(),
            bands(),
            weights(),
            ([0.0, 1.0], ["G"]),
        )


def test_plot_unfolded_falls_back_to_the_svg_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import builtins

    real_import = builtins.__import__

    def no_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib is not installed")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", no_matplotlib)
    out = plot_unfolded(tmp_path / "bands.png", a_path(), bands(), weights(), ([0.0], ["G"]))
    assert out.suffix == ".svg"
    assert out.read_text().startswith("<svg")


def test_plot_unfolded_writes_a_raster_with_matplotlib(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    out = plot_unfolded(
        tmp_path / "bands.png", a_path(), bands(), weights(), ([0.0, 1.0], ["G", "X"])
    )
    assert out.suffix == ".png"
    assert out.stat().st_size > 0


def test_plot_unfolded_rejects_mismatched_shapes_with_matplotlib(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="same shape as energies"):
        plot_unfolded(tmp_path / "bands.png", a_path(), bands(), np.zeros((3, 5)), ([0.0], ["G"]))


def test_the_old_import_path_still_works() -> None:
    from unfoldlab.io import qe

    assert qe.plot_unfolded is plot_unfolded
    assert qe.plot_unfolded_svg is plot_unfolded_svg
    assert qe.color_from_weight is color_from_weight
