from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..config import PlotStyleConfig
from .bars import horizontal_bars, vertical_bars
from .heatmaps import matrix_heatmap
from .series import line_series
from .theme import finalize_axes, publication_style, save_figure


def plot_mmgbsa_summary(rows, out_base, style: PlotStyleConfig):
    labels = [r["term"] for r in rows]
    means = [float(r["mean_kcal_mol"]) for r in rows]
    sds = [float(r.get("sd_kcal_mol", 0.0)) for r in rows]
    horizontal_bars(list(reversed(labels)), list(reversed(means)), list(reversed(sds)), out_base, style, title="MM/GBSA summary terms", xlabel="Energy (kcal/mol)")


def plot_mmgbsa_timeseries(time_or_frame, value_map, out_base, style: PlotStyleConfig, title: str, xlabel: str):
    ys = [np.asarray(v, dtype=float) for v in value_map.values()]
    labels = list(value_map.keys())
    line_series(np.asarray(time_or_frame, dtype=float), ys, labels, ylabel="Energy (kcal/mol)", out_base=out_base, style=style, title=title, xlabel=xlabel)


def plot_per_residue_decomp(labels, values, out_base, style: PlotStyleConfig):
    horizontal_bars(list(reversed(labels)), list(reversed(values)), [0.0] * len(labels), out_base, style, title="Per-residue MM/GBSA decomposition", xlabel="Contribution (kcal/mol)")


def plot_mmgbsa_replica_summary(replica_names, means, sds, out_base, style: PlotStyleConfig):
    vertical_bars(replica_names, means, out_base, style, title="MM/GBSA ΔG by replica", ylabel="ΔG (kcal/mol)", yerr=sds)


def plot_mmgbsa_delta_total_distribution(replica_to_values, out_base, style: PlotStyleConfig):
    with publication_style(style):
        fig = plt.figure(figsize=(7.2, 4.8))
        ax = fig.add_subplot(1, 1, 1)
        labels = list(replica_to_values.keys())
        data = [np.asarray(replica_to_values[k], dtype=float) for k in labels]
        bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(style.categorical_palette[i % len(style.categorical_palette)])
            patch.set_alpha(0.35)
            patch.set_edgecolor(style.spine_color)
        for part in ("whiskers", "caps", "medians"):
            for artist in bp[part]:
                artist.set_color(style.spine_color)
                artist.set_linewidth(style.axes_line_width)
        finalize_axes(ax, style, ylabel="ΔG (kcal/mol)", title="MM/GBSA ΔG distribution by replica")
        fig.tight_layout()
        save_figure(fig, Path(out_base), style)


def plot_mmgbsa_delta_total_heatmap(replica_names, residue_labels, matrix, out_base, style: PlotStyleConfig):
    matrix_heatmap(
        np.asarray(matrix, dtype=float),
        residue_labels,
        replica_names,
        Path(out_base),
        style,
        title="Per-residue MM/GBSA decomposition across replicas",
        xlabel="Replica",
        ylabel="Residue",
        center=0.0,
        cmap="RdBu_r",
        cbar_label="Contribution (kcal/mol)",
        annotate=False,
        x_rotation=35.0,
    )
