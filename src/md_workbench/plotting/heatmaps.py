from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from ..config import PlotStyleConfig
from .theme import finalize_axes, publication_style, save_figure


def _sparse_tick_spec(labels, max_labels: int):
    if len(labels) <= max_labels:
        idx = np.arange(len(labels))
        return idx, labels
    step = max(int(np.ceil(len(labels) / max_labels)), 1)
    idx = np.arange(0, len(labels), step)
    return idx, [labels[i] for i in idx]


def _heatmap_limits(arr, vmin=None, vmax=None, center: float | None = None):
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return vmin, vmax, None
    if center is None:
        return (
            float(np.nanmin(finite)) if vmin is None else vmin,
            float(np.nanmax(finite)) if vmax is None else vmax,
            None,
        )

    if vmin is None or vmax is None:
        max_abs = float(np.nanmax(np.abs(finite - center)))
        if max_abs <= 0:
            max_abs = 1.0
        if vmin is None:
            vmin = center - max_abs
        if vmax is None:
            vmax = center + max_abs
    if not float(vmin) < center:
        vmin = center - 1.0
    if not center < float(vmax):
        vmax = center + 1.0
    return vmin, vmax, TwoSlopeNorm(vmin=float(vmin), vcenter=float(center), vmax=float(vmax))


def _annotation_color(value: float, norm) -> str:
    if norm is None or not np.isfinite(value):
        return "#1F2937"
    mapped = float(norm(value))
    return "white" if mapped <= 0.22 or mapped >= 0.78 else "#1F2937"


def matrix_heatmap(
    matrix,
    row_labels,
    col_labels,
    out_base,
    style: PlotStyleConfig,
    title: str,
    xlabel: str = "",
    ylabel: str = "",
    cmap: str | None = None,
    annotate: bool = False,
    vmin=None,
    vmax=None,
    cbar_label: str = "",
    center: float | None = None,
    x_rotation: float = 30.0,
):
    arr = np.asarray(matrix, dtype=float)
    height = min(max(4.8, 0.095 * len(row_labels) + 2.1), 8.75)
    width = min(max(5.4, 0.8 * len(col_labels) + 2.0), 7.5)
    ytick_positions, ytick_labels = _sparse_tick_spec(list(row_labels), 55)
    xtick_positions, xtick_labels = _sparse_tick_spec(list(col_labels), 16)
    vmin_eff, vmax_eff, norm = _heatmap_limits(arr, vmin=vmin, vmax=vmax, center=center)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(width, height))
        im = ax.imshow(
            arr,
            aspect="auto",
            interpolation="nearest",
            cmap=cmap or ("RdBu_r" if center is not None else style.cmap_continuous),
            vmin=None if norm is not None else vmin_eff,
            vmax=None if norm is not None else vmax_eff,
            norm=norm,
        )
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(xtick_labels, rotation=x_rotation, ha="right")
        ax.set_yticks(ytick_positions)
        ax.set_yticklabels(ytick_labels)
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(False)
        if arr.size <= 180:
            ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
            ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
            ax.grid(which="minor", color="white", linestyle="-", linewidth=0.55, alpha=0.35)
            ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=max(style.tick_size - 0.6, 6.5))
        if cbar_label:
            cbar.set_label(cbar_label)
        if len(row_labels) > 60:
            ax.tick_params(axis="y", labelsize=max(style.tick_size - 1.0, 6.2))
        if annotate:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    value = arr[i, j]
                    if not np.isfinite(value):
                        continue
                    ax.text(
                        j,
                        i,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=max(style.tick_size - 1.5, 6.5),
                        color=_annotation_color(value, im.norm),
                    )
        save_figure(fig, out_base, style)


def stacked_fraction_area(time_ns, fractions_by_label: dict[str, np.ndarray], out_base, style: PlotStyleConfig, title: str, ylabel: str = "Fraction"):
    labels = list(fractions_by_label.keys())
    arrs = [np.asarray(fractions_by_label[label], dtype=float) for label in labels]
    colors = style.categorical_palette[:len(labels)]
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.stackplot(time_ns, arrs, labels=labels, colors=colors, alpha=0.9)
        finalize_axes(ax, style, xlabel="Time (ns)", ylabel=ylabel, title=title)
        ax.set_ylim(0.0, 1.0)
        ax.legend(frameon=False, ncol=len(labels))
        save_figure(fig, out_base, style)


def simple_boxplot(
    data_dict: dict[str, list[float]],
    out_base,
    style: PlotStyleConfig,
    title: str,
    ylabel: str,
    show_points: bool = False,
    hline_zero: bool = False,
):
    labels = list(data_dict.keys())
    data = [np.asarray(data_dict[k], dtype=float) for k in labels]
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(max(6.2, 0.75 * len(labels) + 2.4), 4.8))
        bp = ax.boxplot(data, patch_artist=True, widths=0.65, showfliers=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(style.categorical_palette[i % len(style.categorical_palette)])
            patch.set_alpha(0.85)
            patch.set_edgecolor("white")
        for key in ["whiskers", "caps", "medians"]:
            for item in bp[key]:
                item.set_color(style.spine_color)
                item.set_linewidth(1.0)
        if hline_zero:
            ax.axhline(0.0, color=style.spine_color, linewidth=0.9, linestyle="--", alpha=0.65, zorder=0)
        if show_points:
            rng = np.random.default_rng(20260419)
            for idx, values in enumerate(data, start=1):
                if values.size == 0:
                    continue
                x = idx + rng.uniform(-0.08, 0.08, size=values.size)
                ax.scatter(
                    x,
                    values,
                    s=max(style.marker_size * 4.0, 18.0),
                    color=style.mean_line_color,
                    alpha=0.75,
                    edgecolors="white",
                    linewidths=0.4,
                    zorder=3,
                )
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        finalize_axes(ax, style, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)
