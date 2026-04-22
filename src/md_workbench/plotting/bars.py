from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from ..config import PlotStyleConfig
from .theme import finalize_axes, publication_style, save_figure


def _bar_colors(values, style: PlotStyleConfig):
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size > 0 and np.nanmin(finite) < 0 < np.nanmax(finite):
        return [style.protein_color if value < 0 else style.accent_color for value in arr]
    return style.bar_color


def _crosses_zero(values) -> bool:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return bool(finite.size > 0 and np.nanmin(finite) < 0 < np.nanmax(finite))


def horizontal_bars(labels, means, sds, out_base, style: PlotStyleConfig, title: str, xlabel: str):
    height = max(4.4, 0.30 * len(labels) + 1.4)
    means_arr = np.asarray(means, dtype=float)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.8, height))
        y = np.arange(len(labels))
        ax.barh(
            y, means_arr, xerr=sds, alpha=0.92, height=0.72,
            color=_bar_colors(means_arr, style), edgecolor="white", linewidth=0.8, ecolor=style.spine_color, capsize=2.5,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        if _crosses_zero(means_arr):
            ax.axvline(0.0, color=style.spine_color, linewidth=0.9, linestyle="--", alpha=0.65, zorder=0)
        finalize_axes(ax, style, xlabel=xlabel, title=title)
        save_figure(fig, out_base, style)


def vertical_bars(x, heights, out_base, style: PlotStyleConfig, title: str, xlabel: str = "", ylabel: str = "", yerr=None):
    heights_arr = np.asarray(heights, dtype=float)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.6, 4.6))
        ax.bar(
            x, heights_arr, yerr=yerr, color=_bar_colors(heights_arr, style), edgecolor="white", linewidth=0.8,
            ecolor=style.spine_color if yerr is not None else None, capsize=2.5 if yerr is not None else 0,
        )
        if _crosses_zero(heights_arr):
            ax.axhline(0.0, color=style.spine_color, linewidth=0.9, linestyle="--", alpha=0.65, zorder=0)
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)


def ranked_lollipop(
    labels,
    means,
    sds,
    out_base,
    style: PlotStyleConfig,
    title: str,
    xlabel: str,
    color: str,
):
    labels_arr = list(labels)
    means_arr = np.asarray(means, dtype=float)
    sds_arr = np.asarray(sds, dtype=float)
    if means_arr.size == 0:
        return
    height = max(2.8, 0.34 * len(labels_arr) + 1.4)
    axis_limit = 1.0

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.5, height))
        y = np.arange(len(labels_arr))
        for idx in range(len(labels_arr)):
            if idx % 2 == 0:
                ax.axhspan(idx - 0.5, idx + 0.5, color="#F8FAFC", zorder=0)
        ax.hlines(y, 0.0, means_arr, color=color, linewidth=2.0, alpha=0.30, zorder=1)
        lower = np.clip(means_arr - np.maximum(sds_arr, 0.0), 0.0, None)
        upper = np.minimum(means_arr + np.maximum(sds_arr, 0.0), axis_limit)
        ax.errorbar(
            means_arr,
            y,
            xerr=[np.maximum(means_arr - lower, 0.0), np.maximum(upper - means_arr, 0.0)],
            fmt="none",
            ecolor=style.spine_color,
            elinewidth=1.1,
            capsize=2.8,
            alpha=0.85,
            zorder=2,
        )
        ax.scatter(
            means_arr,
            y,
            s=max(style.marker_size * 24.0, 48.0),
            color=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=3,
        )
        for xpos, ypos, value in zip(means_arr, y, means_arr):
            if not np.isfinite(value):
                continue
            ax.annotate(
                f"{value:.0%}",
                xy=(min(max(xpos, 0.0), axis_limit), ypos),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=max(style.tick_size - 0.1, 7.2),
                color=style.spine_color,
                annotation_clip=False,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels_arr)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.02)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        finalize_axes(ax, style, xlabel=xlabel, title=title)
        ax.grid(True, axis="x", linestyle="-", linewidth=0.7)
        ax.grid(False, axis="y")
        save_figure(fig, out_base, style)


def _distance_to_area(distances, min_area: float = 90.0, max_area: float = 460.0) -> np.ndarray:
    arr = np.asarray(distances, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.full(arr.shape, min_area, dtype=float)
    dmin = float(finite.min())
    dmax = float(finite.max())
    if np.isclose(dmin, dmax):
        return np.full(arr.shape, 0.5 * (min_area + max_area), dtype=float)
    proximity = (dmax - arr) / (dmax - dmin)
    proximity = np.clip(proximity, 0.0, 1.0)
    return min_area + proximity * (max_area - min_area)


def ranked_distance_lollipop(
    labels,
    means,
    sds,
    distances,
    out_base,
    style: PlotStyleConfig,
    title: str,
    xlabel: str,
    color: str,
):
    labels_arr = list(labels)
    means_arr = np.asarray(means, dtype=float)
    sds_arr = np.asarray(sds, dtype=float)
    distance_arr = np.asarray(distances, dtype=float)
    if means_arr.size == 0:
        return
    height = max(4.1, 0.35 * len(labels_arr) + 1.8)
    axis_limit = 1.0
    areas = _distance_to_area(distance_arr)

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(8.2, height))
        y = np.arange(len(labels_arr))
        for idx in range(len(labels_arr)):
            if idx % 2 == 0:
                ax.axhspan(idx - 0.5, idx + 0.5, color="#F8FAFC", zorder=0)
        ax.hlines(y, 0.0, means_arr, color=color, linewidth=2.15, alpha=0.24, zorder=1)
        lower = np.clip(means_arr - np.maximum(sds_arr, 0.0), 0.0, None)
        upper = np.minimum(means_arr + np.maximum(sds_arr, 0.0), axis_limit)
        ax.errorbar(
            means_arr,
            y,
            xerr=[np.maximum(means_arr - lower, 0.0), np.maximum(upper - means_arr, 0.0)],
            fmt="none",
            ecolor=style.spine_color,
            elinewidth=1.05,
            capsize=2.8,
            alpha=0.82,
            zorder=2,
        )
        ax.scatter(
            means_arr,
            y,
            s=areas,
            color=color,
            edgecolors="white",
            linewidths=1.0,
            alpha=0.92,
            zorder=3,
        )
        for xpos, ypos, value, dist in zip(means_arr, y, means_arr, distance_arr):
            if not np.isfinite(value):
                continue
            suffix = f" | {dist:.1f} A" if np.isfinite(dist) else ""
            ax.annotate(
                f"{value:.0%}{suffix}",
                xy=(min(max(xpos, 0.0), axis_limit), ypos),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=max(style.tick_size - 0.3, 7.0),
                color=style.spine_color,
                annotation_clip=False,
                zorder=4,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels_arr)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.05)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        finalize_axes(ax, style, xlabel=xlabel, title=title)
        ax.grid(True, axis="x", linestyle="-", linewidth=0.7)
        ax.grid(False, axis="y")
        save_figure(fig, out_base, style)
