"""Heat-map rendering of the unfolded spectral function."""

from __future__ import annotations

import base64
import re
import struct
from pathlib import Path

import numpy as np
import pytest

from unmochan.core.spectral import EffectiveBandStructure
from unmochan.io.plot_spectral import (
    plot_spectral_function,
    plot_spectral_function_svg,
    spectral_energy_grid,
    spectral_intensity_bound,
    spectral_map,
)


def _ebs() -> EffectiveBandStructure:
    distances = np.linspace(0.0, 1.0, 9)
    kpoints = np.stack([distances, np.zeros(9), np.zeros(9)], axis=1)
    energies = np.stack([-1.0 + distances, 1.5 - 0.5 * distances], axis=1)
    weights = np.stack([np.full(9, 0.8), np.full(9, 0.2)], axis=1)
    return EffectiveBandStructure(
        kpoints=kpoints,
        energies=energies,
        weights=weights,
        distances=distances,
    )


def test_the_default_grid_covers_the_bands_with_padding() -> None:
    ebs = _ebs()

    grid = spectral_energy_grid(ebs, n_energies=51)

    assert grid.size == 51
    assert grid.min() < ebs.energies.min()
    assert grid.max() > ebs.energies.max()


def test_an_explicit_window_is_honoured() -> None:
    grid = spectral_energy_grid(_ebs(), n_energies=11, emin=-2.0, emax=2.0)

    assert grid[0] == pytest.approx(-2.0)
    assert grid[-1] == pytest.approx(2.0)


def test_the_map_integrates_to_the_weight_of_each_kpoint() -> None:
    ebs = _ebs()

    grid, intensity = spectral_map(ebs, broadening=0.05, n_energies=4001)

    integral = np.trapezoid(intensity, grid, axis=1)
    assert integral == pytest.approx(ebs.weights.sum(axis=1), abs=1e-4)


def test_the_intensity_never_exceeds_the_proved_bound() -> None:
    ebs = _ebs()

    for kind in ("gaussian", "lorentzian"):
        _grid, intensity = spectral_map(ebs, broadening=0.08, kind=kind, n_energies=801)
        bound = spectral_intensity_bound(ebs, broadening=0.08, kind=kind)
        assert intensity.min() >= 0.0
        assert intensity.max() <= bound + 1e-9


def test_the_bound_matches_the_analytic_peak() -> None:
    ebs = _ebs()

    gaussian = spectral_intensity_bound(ebs, broadening=0.1, kind="gaussian")
    lorentzian = spectral_intensity_bound(ebs, broadening=0.1, kind="lorentzian")

    total = float(ebs.weights.sum(axis=1).max())
    assert gaussian == pytest.approx(total / (0.1 * np.sqrt(2.0 * np.pi)))
    assert lorentzian == pytest.approx(total / (np.pi * 0.1))


def test_an_unknown_kernel_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported broadening kind"):
        spectral_intensity_bound(_ebs(), broadening=0.1, kind="box")  # type: ignore[arg-type]


def test_adaptive_widths_can_drive_the_map() -> None:
    ebs = _ebs()
    widths = ebs.adaptive_widths(scale=1.0, minimum=0.02)

    grid, intensity = spectral_map(ebs, broadening=widths, n_energies=2001)

    integral = np.trapezoid(intensity, grid, axis=1)
    assert integral == pytest.approx(ebs.weights.sum(axis=1), abs=1e-3)


def test_matplotlib_renderer_writes_a_png(tmp_path: Path) -> None:
    out = plot_spectral_function(
        tmp_path / "map.png",
        _ebs(),
        broadening=0.1,
        n_energies=101,
        ticks=([0.0, 1.0], ["G", "X"]),
    )

    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_svg_fallback_embeds_a_valid_png(tmp_path: Path) -> None:
    ebs = _ebs()
    grid, intensity = spectral_map(ebs, broadening=0.1, n_energies=64)

    out = plot_spectral_function_svg(
        tmp_path / "map.svg", ebs, grid, intensity, ticks=([0.0, 1.0], ["G", "X"])
    )

    text = out.read_text()
    assert text.startswith("<svg")
    match = re.search(r"data:image/png;base64,([A-Za-z0-9+/=]+)", text)
    assert match is not None
    payload = base64.b64decode(match.group(1))
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    # The IHDR chunk records the raster size: energies across, k-points down.
    width, height = struct.unpack(">II", payload[16:24])
    assert (width, height) == (ebs.n_kpoints, grid.size)


def test_the_svg_fallback_forces_the_extension(tmp_path: Path) -> None:
    ebs = _ebs()
    grid, intensity = spectral_map(ebs, broadening=0.1, n_energies=16)

    out = plot_spectral_function_svg(tmp_path / "map.png", ebs, grid, intensity)

    assert out.suffix == ".svg"


def test_a_mismatched_intensity_table_is_rejected(tmp_path: Path) -> None:
    ebs = _ebs()
    grid = spectral_energy_grid(ebs, n_energies=16)

    with pytest.raises(ValueError, match="n_energies"):
        plot_spectral_function_svg(tmp_path / "map.svg", ebs, grid, np.zeros((ebs.n_kpoints, 3)))
