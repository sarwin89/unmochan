"""Heat-map plotting of the unfolded spectral function ``A(k, E)``.

The scatter plot in :mod:`unmochan.io.qe` draws one marker per state, sized by
its unfolding weight.  That is faithful but it is not what an ARPES-style figure
looks like, and it hides the broadening entirely: two states half a broadening
apart look like two bands rather than one blurred one.  This module draws the
broadened spectral function itself, on a ``(n_kpoints, n_energies)`` grid.

What the picture is allowed to show is pinned down in
``RequestProject/Unfolding/Spectral.lean``:

* ``integral_spectralFunction`` / ``integral_adaptiveSpectralFunction`` — the
  energy integral of a column is the total unfolding weight of that k-point, so
  broadening moves weight around but never creates or destroys it;
* ``spectralFunction_nonneg`` — the intensity is non-negative, so the colour
  scale legitimately starts at zero;
* ``spectralFunction_le`` with ``gaussianKernel_le_zero`` /
  ``lorentzianKernel_le_zero`` — no column can be brighter than its total weight
  times the peak of the kernel, ``1/(sigma*sqrt(2*pi))`` for a Gaussian and
  ``1/(pi*gamma)`` for a Lorentzian.  :func:`spectral_intensity_bound` returns
  that number, which is the honest top of a shared colour scale.

Both a matplotlib renderer and a dependency-free fallback are provided; the
fallback writes an SVG with an embedded PNG raster, built with nothing but the
standard library.
"""

from __future__ import annotations

import base64
import struct
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike, NDArray

from unmochan.core.spectral import BroadeningKind, EffectiveBandStructure

__all__ = [
    "plot_spectral_function",
    "plot_spectral_function_svg",
    "spectral_energy_grid",
    "spectral_intensity_bound",
    "spectral_map",
]


def spectral_energy_grid(
    ebs: EffectiveBandStructure,
    *,
    n_energies: int = 400,
    padding: float | None = None,
    broadening: float | ArrayLike | None = None,
    emin: float | None = None,
    emax: float | None = None,
) -> NDArray[np.float64]:
    """A uniform energy grid covering the bands, relative to the reference.

    ``padding`` extends the range beyond the extreme band energies so that a
    broadened peak at the edge is not cut in half.  It defaults to five per cent
    of the span (one energy unit for a flat band), widened to five times the
    largest broadening width when ``broadening`` is supplied — which is what
    keeps the weight sum rule visible on the grid rather than spilling off it.
    """

    if n_energies < 2:
        raise ValueError("n_energies must be at least two")
    energies = ebs.shifted_energies()
    low = float(energies.min()) if emin is None else float(emin)
    high = float(energies.max()) if emax is None else float(emax)
    if high < low:
        raise ValueError("emax must not be below emin")
    if padding is None:
        span = high - low
        padding = 0.05 * span if span > 0.0 else 1.0
        if broadening is not None:
            widths = np.asarray(broadening, dtype=float)
            padding = max(padding, 5.0 * float(np.max(widths)))
    if padding < 0.0:
        raise ValueError("padding must be non-negative")
    if emin is None:
        low -= padding
    if emax is None:
        high += padding
    if high == low:
        low -= 0.5
        high += 0.5
    return np.linspace(low, high, int(n_energies))


def spectral_intensity_bound(
    ebs: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
) -> float:
    """The largest value ``A(k, E)`` can take, over all k-points.

    This is ``max_k (sum_n w_kn) * peak``, the bound proved as
    ``UnfoldLab.spectralFunction_le``: all the weight of a k-point piled onto a
    single peak of the narrowest kernel in play.  It is a *bound*, not the
    maximum actually attained, which makes it a stable top for a colour scale
    shared between several figures.
    """

    widths = np.asarray(broadening, dtype=float)
    widths = np.broadcast_to(widths, ebs.energies.shape)
    if not np.all(widths > 0.0):
        raise ValueError("broadening must be positive")
    narrowest = float(widths.min())
    if kind == "gaussian":
        peak = 1.0 / (narrowest * np.sqrt(2.0 * np.pi))
    elif kind == "lorentzian":
        peak = 1.0 / (np.pi * narrowest)
    else:
        raise ValueError(f"unsupported broadening kind: {kind}")
    total = float(np.max(ebs.weights.sum(axis=1))) if ebs.n_kpoints else 0.0
    return total * float(peak)


def spectral_map(
    ebs: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    energy_grid: ArrayLike | None = None,
    n_energies: int = 400,
    emin: float | None = None,
    emax: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(energy_grid, intensity)`` for a heat map of ``A(k, E)``.

    ``intensity`` has shape ``(n_kpoints, n_energies)`` and is the same array
    :meth:`EffectiveBandStructure.spectral_function` returns; this wrapper only
    supplies a sensible default grid.
    """

    grid = (
        spectral_energy_grid(
            ebs,
            n_energies=n_energies,
            broadening=broadening,
            emin=emin,
            emax=emax,
        )
        if energy_grid is None
        else np.asarray(energy_grid, dtype=float)
    )
    intensity = ebs.spectral_function(grid, broadening=broadening, kind=kind)
    return grid, intensity


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

#: Anchor colours of the fallback colour map, dark to bright.
_RAMP = (
    (0.0, (0, 0, 4)),
    (0.25, (66, 10, 104)),
    (0.5, (147, 38, 103)),
    (0.75, (221, 81, 58)),
    (1.0, (252, 255, 164)),
)


def _colormap(values: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Map ``values`` in ``[0, 1]`` to RGB with a piecewise-linear ramp."""

    clipped = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    stops = np.array([stop for stop, _ in _RAMP], dtype=float)
    colours = np.array([colour for _, colour in _RAMP], dtype=float)
    rgb = np.empty((*clipped.shape, 3), dtype=float)
    for channel in range(3):
        rgb[..., channel] = np.interp(clipped, stops, colours[:, channel])
    return np.rint(rgb).astype(np.uint8)


def _png_bytes(rgb: NDArray[np.uint8]) -> bytes:
    """Encode an ``(height, width, 3)`` uint8 array as a PNG, stdlib only."""

    height, width, channels = rgb.shape
    if channels != 3:
        raise ValueError("expected an RGB image")
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(height))

    def chunk(tag: bytes, payload: bytes) -> bytes:
        body = tag + payload
        return (
            struct.pack(">I", len(payload))
            + body
            + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


def _raster(intensity: NDArray[np.float64], vmax: float) -> NDArray[np.uint8]:
    """Colour the intensity map with energy increasing *upwards*."""

    scaled = intensity / vmax if vmax > 0.0 else np.zeros_like(intensity)
    # intensity is (n_kpoints, n_energies); an image is (rows, columns) with
    # row 0 at the top, so transpose and flip the energy axis.
    return _colormap(scaled.T[::-1, :])


def plot_spectral_function(
    path: str | Path,
    ebs: EffectiveBandStructure,
    *,
    broadening: float | ArrayLike,
    kind: BroadeningKind = "gaussian",
    energy_grid: ArrayLike | None = None,
    n_energies: int = 400,
    emin: float | None = None,
    emax: float | None = None,
    ticks: tuple[list[float], list[str]] | None = None,
    vmax: float | None = None,
    cmap: str = "inferno",
) -> Path:
    """Write a heat map of the unfolded spectral function.

    ``vmax`` defaults to the largest value actually on the grid, which is what
    makes a single figure readable; pass :func:`spectral_intensity_bound` to
    share one absolute scale across several figures.

    Falls back to :func:`plot_spectral_function_svg` when matplotlib is absent.
    """

    grid, intensity = spectral_map(
        ebs,
        broadening=broadening,
        kind=kind,
        energy_grid=energy_grid,
        n_energies=n_energies,
        emin=emin,
        emax=emax,
    )
    out_path = Path(path)
    top = float(intensity.max()) if vmax is None else float(vmax)
    if top <= 0.0:
        top = 1.0
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
        distances = ebs.distances
        extent = (
            float(distances.min()),
            float(distances.max()),
            float(grid.min()),
            float(grid.max()),
        )
        if extent[0] == extent[1]:
            extent = (extent[0] - 0.5, extent[1] + 0.5, extent[2], extent[3])
        image = ax.imshow(
            intensity.T,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap=cmap,
            vmin=0.0,
            vmax=top,
            interpolation="bilinear",
        )
        if ticks is not None:
            tick_x, tick_labels = ticks
            for xpos in tick_x:
                ax.axvline(xpos, color="0.85", lw=0.7)
            if tick_x:
                ax.set_xticks(tick_x, tick_labels)
        ax.axhline(0.0, color="0.85", lw=0.8, ls="--")
        ax.set_xlabel("Primitive-cell k-path")
        ax.set_ylabel("Energy - reference (eV)")
        fig.colorbar(image, ax=ax, pad=0.02).set_label("A(k, E)")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        return out_path
    except Exception:
        return plot_spectral_function_svg(path, ebs, grid, intensity, ticks=ticks, vmax=vmax)


def plot_spectral_function_svg(
    path: str | Path,
    ebs: EffectiveBandStructure,
    energy_grid: ArrayLike,
    intensity: ArrayLike,
    *,
    ticks: tuple[list[float], list[str]] | None = None,
    vmax: float | None = None,
) -> Path:
    """Write the heat map as an SVG with an embedded PNG raster.

    Uses nothing outside the standard library and NumPy, so a run without
    matplotlib still produces a real figure rather than no figure.
    """

    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg":
        svg_path = svg_path.with_suffix(".svg")
    grid = np.asarray(energy_grid, dtype=float)
    values = np.asarray(intensity, dtype=float)
    if values.ndim != 2 or values.shape[1] != grid.size:
        raise ValueError("intensity must have shape (n_kpoints, n_energies)")
    top = float(values.max()) if vmax is None else float(vmax)
    if top <= 0.0:
        top = 1.0
    encoded = base64.b64encode(_png_bytes(_raster(values, top))).decode("ascii")

    width, height = 720.0, 480.0
    left, right, bottom, top_margin = 70.0, 20.0, 50.0, 20.0
    plot_w = width - left - right
    plot_h = height - bottom - top_margin
    distances = ebs.distances
    x_min, x_max = float(distances.min()), float(distances.max())
    if x_min == x_max:
        x_min, x_max = x_min - 0.5, x_max + 0.5

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="white"/>',
        f'<image x="{left:.1f}" y="{top_margin:.1f}" width="{plot_w:.1f}" '
        f'height="{plot_h:.1f}" preserveAspectRatio="none" '
        f'xlink:href="data:image/png;base64,{encoded}"/>',
        f'<rect x="{left:.1f}" y="{top_margin:.1f}" width="{plot_w:.1f}" '
        f'height="{plot_h:.1f}" fill="none" stroke="black" stroke-width="1"/>',
    ]
    if ticks is not None:
        for position, label in zip(*ticks, strict=False):
            fraction = (float(position) - x_min) / (x_max - x_min)
            x = left + fraction * plot_w
            parts.append(
                f'<line x1="{x:.1f}" y1="{top_margin:.1f}" x2="{x:.1f}" '
                f'y2="{top_margin + plot_h:.1f}" stroke="#dddddd" '
                f'stroke-width="0.8"/>'
            )
            parts.append(
                f'<text x="{x:.1f}" y="{top_margin + plot_h + 18:.1f}" '
                f'font-size="12" text-anchor="middle">{label}</text>'
            )
    for fraction, value in ((0.0, grid.min()), (0.5, grid.mean()), (1.0, grid.max())):
        y = top_margin + (1.0 - fraction) * plot_h
        parts.append(
            f'<text x="{left - 8:.1f}" y="{y + 4:.1f}" font-size="12" '
            f'text-anchor="end">{float(value):.2f}</text>'
        )
    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 8:.1f}" font-size="13" '
        f'text-anchor="middle">Primitive-cell k-path</text>'
    )
    parts.append(
        f'<text x="16" y="{height / 2:.1f}" font-size="13" text-anchor="middle" '
        f'transform="rotate(-90 16 {height / 2:.1f})">Energy - reference (eV)'
        f"</text>"
    )
    parts.append(f'<text x="{left:.1f}" y="14" font-size="11">A(k, E) max = {top:.4g}</text>')
    parts.append("</svg>")
    svg_path.write_text("\n".join(parts) + "\n")
    return svg_path
