"""Scatter plotting of an unfolded band structure, one marker per state.

Each ``(k-point, band)`` state is drawn at its energy, with the marker area and
colour set by the unfolding weight.  This is the *faithful* picture: it shows
exactly the numbers the unfolding produced, with no broadening interposed.
:mod:`unfoldlab.io.plot_spectral` draws the complementary ARPES-style picture,
the broadened spectral function on an energy grid.

Two renderers are provided, and :func:`plot_unfolded` picks between them:
matplotlib when it is importable, and otherwise :func:`plot_unfolded_svg`,
which writes an SVG using nothing outside the standard library.  The fallback
is not decoration -- ``matplotlib`` lives in the optional ``plot`` extra, so it
is the renderer an install without that extra actually uses.  The two are kept
deliberately close:

* a marker's *area* is affine in the weight in both, so a weight-0.5 state does
  not look twice as bright as it should in one of them;
* both clip to ``[emin, emax]``.  Matplotlib does this for free with
  ``set_ylim``; the SVG has to drop the out-of-window states itself, and not
  doing so used to scatter markers across the axis labels and the title;
* both survive a non-finite energy or weight.  A single ``nan`` used to set the
  SVG's whole y-axis to ``nan`` and destroy the figure, rather than costing the
  one state that carried it.

This module deliberately knows nothing about Quantum ESPRESSO.  It takes any
object with a ``distances`` array -- a :class:`unfoldlab.io.qe.QEKMap`, an
:class:`unfoldlab.core.spectral.EffectiveBandStructure`, or anything else --
which is why the VASP workflow can use it too.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "color_from_weight",
    "marker_area",
    "plot_unfolded",
    "plot_unfolded_svg",
]


class HasDistances(Protocol):
    """Anything carrying a k-path coordinate for every row of ``energies``."""

    @property
    def distances(self) -> NDArray[np.float64]: ...


def color_from_weight(weight: float) -> str:
    """A four-stop linear approximation of ``viridis``, as ``#rrggbb``.

    The matplotlib renderer uses the real colormap; this is what the SVG
    fallback has instead.  The two agree at the stops and drift by a few
    percent between them, which is a difference nobody can read off a figure.
    """

    stops = np.array(
        [[68, 1, 84], [49, 104, 142], [53, 183, 121], [253, 231, 37]],
        dtype=float,
    )
    value = float(np.clip(weight, 0.0, 1.0))
    scaled = value * (len(stops) - 1)
    idx = min(int(np.floor(scaled)), len(stops) - 2)
    frac = scaled - idx
    rgb = (1.0 - frac) * stops[idx] + frac * stops[idx + 1]
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in rgb)


def marker_area(weights: NDArray[np.float64], marker_scale: float) -> NDArray[np.float64]:
    """Marker *area* for each weight: matplotlib's ``s`` argument.

    Affine in the clipped weight, so the visual mass of a state is proportional
    to what it contributes to the sum rule.  The SVG renderer takes the square
    root of this to get a radius, which is the only way the two pictures can
    show the same thing.
    """

    clipped = np.clip(np.asarray(weights, dtype=float), 0.0, 1.0)
    return np.asarray(3.0 + float(marker_scale) * clipped, dtype=float)


def _validate(
    kmap: HasDistances,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    distances = np.asarray(kmap.distances, dtype=float)
    energy_array = np.asarray(energies, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    if energy_array.ndim != 2:
        raise ValueError("energies must have shape (n_kpoints, n_bands)")
    if weight_array.shape != energy_array.shape:
        raise ValueError("weights must have the same shape as energies")
    if distances.shape != (energy_array.shape[0],):
        raise ValueError("kmap.distances must have one entry per k-point")
    return distances, energy_array, weight_array


def _axis_range(values: NDArray[np.float64]) -> tuple[float, float]:
    """A non-degenerate range covering the finite entries of ``values``."""

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1.0, 1.0
    low = float(finite.min())
    high = float(finite.max())
    if np.isclose(low, high):
        return low - 1.0, high + 1.0
    return low, high


def plot_unfolded(
    path: str | Path,
    kmap: HasDistances,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    ticks: tuple[list[float], list[str]],
    *,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
) -> Path:
    """Draw the unfolded bands, with matplotlib if it is available.

    Without matplotlib this delegates to :func:`plot_unfolded_svg`, which
    forces the ``.svg`` extension -- so the returned path is not necessarily
    the one that was asked for, and callers should report what comes back
    rather than what they passed in.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return plot_unfolded_svg(
            path,
            kmap,
            energies,
            weights,
            ticks,
            fermi=fermi,
            emin=emin,
            emax=emax,
            marker_scale=marker_scale,
        )

    distances, energy_array, weight_array = _validate(kmap, energies, weights)
    out_path = Path(path)
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=180)
    x_values = np.repeat(distances, energy_array.shape[1])
    y_values = (energy_array - fermi).reshape(-1)
    clipped_weights = np.clip(weight_array.reshape(-1), 0.0, 1.0)
    scatter = ax.scatter(
        x_values,
        y_values,
        s=marker_area(clipped_weights, marker_scale),
        c=clipped_weights,
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.0,
        alpha=0.85,
    )
    tick_x, tick_labels = ticks
    for xpos in tick_x:
        ax.axvline(xpos, color="0.75", lw=0.7, zorder=0)
    if tick_x:
        ax.set_xticks(tick_x, tick_labels)
    ax.axhline(0.0, color="0.35", lw=0.8, ls="--")
    x_min, x_max = _axis_range(distances)
    ax.set_xlim(x_min, x_max)
    if emin is not None or emax is not None:
        ax.set_ylim(emin, emax)
    ax.set_ylabel("Energy - Ef (eV)" if fermi else "Energy (eV)")
    ax.set_xlabel("Primitive-cell k-path")
    fig.colorbar(scatter, ax=ax, pad=0.02).set_label("Spectral weight")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_unfolded_svg(
    path: str | Path,
    kmap: HasDistances,
    energies: NDArray[np.float64],
    weights: NDArray[np.float64],
    ticks: tuple[list[float], list[str]] = ([], []),
    *,
    fermi: float = 0.0,
    emin: float | None = None,
    emax: float | None = None,
    marker_scale: float = 28.0,
) -> Path:
    """The dependency-free renderer: an SVG built from the standard library.

    States outside ``[emin, emax]`` are omitted rather than drawn outside the
    axes box, and states with a non-finite energy or weight are omitted rather
    than allowed to poison the axis range.  The extension is forced to
    ``.svg``; the path actually written is returned.
    """

    distances, energy_array, weight_array = _validate(kmap, energies, weights)
    svg_path = Path(path)
    if svg_path.suffix.lower() != ".svg":
        svg_path = svg_path.with_suffix(".svg")
    width, height = 900, 600
    left, right, top, bottom = 82, 28, 30, 70
    plot_w = width - left - right
    plot_h = height - top - bottom

    x_min, x_max = _axis_range(distances)
    y_values = energy_array - fermi
    drawable = np.isfinite(y_values) & np.isfinite(weight_array)
    auto_min, auto_max = _axis_range(np.where(drawable, y_values, np.nan))
    y_min = auto_min if emin is None else float(emin)
    y_max = auto_max if emax is None else float(emax)
    if np.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_w

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_h

    tick_x, tick_labels = ticks
    rows = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" '
            'fill="white" stroke="#222" stroke-width="1"/>'
        ),
    ]
    for xpos, label in zip(tick_x, tick_labels, strict=True):
        px = sx(float(xpos))
        rows.append(
            f'<line x1="{px:.2f}" y1="{top}" x2="{px:.2f}" '
            f'y2="{top + plot_h}" stroke="#c8c8c8" stroke-width="1"/>'
        )
        rows.append(
            f'<text x="{px:.2f}" y="{height - 36}" '
            'font-family="Arial, sans-serif" font-size="15" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )
    if y_min <= 0.0 <= y_max:
        y0 = sy(0.0)
        rows.append(
            f'<line x1="{left}" y1="{y0:.2f}" x2="{left + plot_w}" '
            f'y2="{y0:.2f}" stroke="#666" stroke-width="1" '
            'stroke-dasharray="5,5"/>'
        )
    for tick in np.linspace(y_min, y_max, 6):
        py = sy(float(tick))
        rows.append(
            f'<line x1="{left - 5}" y1="{py:.2f}" x2="{left}" '
            f'y2="{py:.2f}" stroke="#222" stroke-width="1"/>'
        )
        rows.append(
            f'<text x="{left - 10}" y="{py + 5:.2f}" '
            'font-family="Arial, sans-serif" font-size="13" '
            f'text-anchor="end">{tick:.2f}</text>'
        )
    # ``sqrt`` of matplotlib's ``s``: the marker *area*, not its radius, is what
    # is affine in the weight, in both renderers.
    radii = np.sqrt(marker_area(np.where(drawable, weight_array, 0.0), marker_scale))
    in_window = drawable & (y_values >= y_min) & (y_values <= y_max)
    for ik, distance in enumerate(distances):
        px = sx(float(distance))
        for ib in range(energy_array.shape[1]):
            if not in_window[ik, ib]:
                continue
            weight = float(np.clip(weight_array[ik, ib], 0.0, 1.0))
            py = sy(float(y_values[ik, ib]))
            color = color_from_weight(weight)
            rows.append(
                f'<circle cx="{px:.2f}" cy="{py:.2f}" '
                f'r="{radii[ik, ib]:.2f}" '
                f'fill="{color}" fill-opacity="0.82"/>'
            )
    ylabel = "Energy - Ef (eV)" if fermi else "Energy (eV)"
    rows.extend(
        [
            (
                f'<text x="{left + plot_w / 2:.2f}" y="{height - 12}" '
                'font-family="Arial, sans-serif" font-size="16" '
                'text-anchor="middle">Primitive-cell k-path</text>'
            ),
            (
                f'<text x="22" y="{top + plot_h / 2:.2f}" '
                'font-family="Arial, sans-serif" font-size="16" '
                'text-anchor="middle" '
                f'transform="rotate(-90 22 {top + plot_h / 2:.2f})">'
                f"{html.escape(ylabel)}</text>"
            ),
            (
                '<text x="730" y="32" font-family="Arial, sans-serif" '
                'font-size="13">color/size = spectral weight</text>'
            ),
            "</svg>",
        ]
    )
    svg_path.write_text("\n".join(rows) + "\n")
    return svg_path
