from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..config import PlotStyleConfig
from .theme import finalize_axes, publication_style, save_figure


def _project_xy(coords: np.ndarray):
    centered = coords - coords.mean(axis=0, keepdims=True)
    if centered.shape[0] < 3:
        return centered[:, :2]
    u, s, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:2].T


def _split_backbone_segments(xy: np.ndarray) -> list[np.ndarray]:
    if xy.shape[0] < 3:
        return [xy]
    steps = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    finite_steps = steps[np.isfinite(steps) & (steps > 1e-8)]
    if finite_steps.size == 0:
        return [xy]
    threshold = max(np.median(finite_steps) * 2.8, np.percentile(finite_steps, 90))
    breaks = np.where(steps > threshold)[0] + 1
    return [segment for segment in np.split(xy, breaks) if segment.size]


def snapshot_grid(snapshot_entries, out_base, style: PlotStyleConfig, title: str):
    if not snapshot_entries:
        return
    n = len(snapshot_entries)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))
    with publication_style(style):
        fig_width = min(7.2, 2.45 * cols)
        fig_height = min(8.2, 2.6 * rows + 0.4)
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis("off")
        for ax, entry in zip(axes, snapshot_entries):
            prot_xy = _project_xy(entry["protein_xyz"])
            lig_xy = _project_xy(entry["ligand_xyz"])
            for segment in _split_backbone_segments(prot_xy):
                if segment.shape[0] < 2:
                    continue
                ax.plot(segment[:, 0], segment[:, 1], color=style.spine_color, linewidth=2.2, alpha=0.10, zorder=1)
                ax.plot(segment[:, 0], segment[:, 1], color=style.protein_color, linewidth=1.5, alpha=0.82, zorder=2)
            ax.scatter(prot_xy[:, 0], prot_xy[:, 1], s=4, color=style.protein_color, linewidths=0.0, alpha=0.18, zorder=2)
            ax.scatter(lig_xy[:, 0], lig_xy[:, 1], s=30, color=style.ligand_color, edgecolors="white", linewidths=0.55, zorder=4)
            lig_centroid = lig_xy.mean(axis=0)
            ax.scatter(lig_centroid[0], lig_centroid[1], s=74, color=style.accent_color, edgecolors="white", linewidths=0.7, zorder=5)
            finalize_axes(ax, style, title=entry["title"])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")
            ax.margins(0.08)
            for side in ["left", "bottom"]:
                ax.spines[side].set_visible(False)
        fig.suptitle(title, y=0.995, fontsize=style.title_size + 0.3, weight="semibold")
        save_figure(fig, out_base, style)
