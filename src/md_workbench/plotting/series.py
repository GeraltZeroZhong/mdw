from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ..config import PlotStyleConfig
from .theme import finalize_axes, publication_style, save_figure


def _legend_kwargs(n_items: int) -> dict:
    if n_items <= 4:
        return {"loc": "best", "ncol": 1}
    if n_items <= 8:
        return {"loc": "center left", "bbox_to_anchor": (1.01, 0.5), "borderaxespad": 0.0, "ncol": 1}
    return {"loc": "upper center", "bbox_to_anchor": (0.5, 1.02), "borderaxespad": 0.0, "ncol": 2}


def line_series(time_ns, ys, labels, ylabel, out_base, style: PlotStyleConfig, title=None, xlabel="Time (ns)", colors=None):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        palette = colors or style.categorical_palette
        for idx, (y, label) in enumerate(zip(ys, labels)):
            ax.plot(time_ns, y, linewidth=style.line_width, label=label, color=palette[idx % len(palette)])
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        if len(labels) > 1:
            ax.legend(frameon=False, **_legend_kwargs(len(labels)))
        save_figure(fig, out_base, style)


def mean_sd_series(time_ns, arr2d, ylabel, out_base, style: PlotStyleConfig, title=None, individual_labels=None, xlabel="Time (ns)"):
    arr2d = np.asarray(arr2d, dtype=float)
    mean = arr2d.mean(axis=0)
    sd = arr2d.std(axis=0, ddof=1) if arr2d.shape[0] > 1 else np.zeros_like(mean)

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        palette = style.categorical_palette
        for i in range(arr2d.shape[0]):
            label = individual_labels[i] if individual_labels is not None else None
            ax.plot(time_ns, arr2d[i], linewidth=style.thin_line_width, alpha=0.35, label=label, color=palette[i % len(palette)])
        ax.plot(time_ns, mean, linewidth=style.line_width + 0.4, label="Mean", color=style.mean_line_color)
        ax.fill_between(time_ns, mean - sd, mean + sd, alpha=0.24, linewidth=0.0, color=style.band_color)
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        if individual_labels is not None or arr2d.shape[0] > 1:
            n_items = len(individual_labels) if individual_labels is not None else arr2d.shape[0] + 1
            ax.legend(frameon=False, **_legend_kwargs(n_items))
        save_figure(fig, out_base, style)


def two_panel_series(time_ns, top_y, bottom_y, top_label, bottom_label, out_base, style: PlotStyleConfig, title=None, top_color=None, bottom_color=None):
    with publication_style(style):
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.1), sharex=False)
        axes[0].plot(time_ns, top_y, linewidth=style.line_width, color=top_color or style.protein_color)
        finalize_axes(axes[0], style, ylabel=top_label, title=title)
        axes[1].plot(time_ns, bottom_y, linewidth=style.line_width, color=bottom_color or style.ligand_color)
        finalize_axes(axes[1], style, xlabel="Time (ns)", ylabel=bottom_label)
        save_figure(fig, out_base, style)


def shaded_profile(x, y, ysd, ylabel, out_base, style: PlotStyleConfig, title=None, xlabel="Residue index", line_color=None):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.7))
        ax.plot(x, y, linewidth=style.line_width, color=line_color or style.protein_color)
        ax.fill_between(x, np.asarray(y) - np.asarray(ysd), np.asarray(y) + np.asarray(ysd), alpha=0.22, linewidth=0.0, color=style.band_color)
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)
