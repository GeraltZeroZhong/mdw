from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
