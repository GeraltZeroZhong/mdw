from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgb
from scipy.interpolate import CubicSpline

from ..config import PlotStyleConfig
from .theme import publication_style, save_figure


_PROTEIN_STYLE = {
    "helix": {"width": 2.60, "alpha": 0.90},
    "strand": {"width": 2.35, "alpha": 0.88},
    "coil": {"width": 2.10, "alpha": 0.84},
}


def _panel_grid(n_panels: int) -> tuple[int, int]:
    if n_panels <= 0:
        return 0, 0
    if n_panels <= 3:
        return 1, n_panels
    if n_panels == 4:
        return 2, 2
    return int(np.ceil(n_panels / 3.0)), 3


def _blend_rgb(base_color: str | tuple[float, float, float], target: tuple[float, float, float], fraction: float) -> tuple[float, float, float]:
    fraction = float(np.clip(fraction, 0.0, 1.0))
    base = np.asarray(to_rgb(base_color), dtype=float)
    tgt = np.asarray(target, dtype=float)
    return tuple((1.0 - fraction) * base + fraction * tgt)


def _style_class(code: str | None) -> str:
    if code == "H":
        return "helix"
    if code == "E":
        return "strand"
    return "coil"


def _shared_projection_basis(snapshot_entries) -> np.ndarray:
    centered_blocks = []
    for entry in snapshot_entries:
        protein_xyz = np.asarray(entry["protein_xyz"], dtype=float)
        if protein_xyz.size == 0:
            continue
        centered_blocks.append(protein_xyz - protein_xyz.mean(axis=0, keepdims=True))
    if not centered_blocks:
        return np.eye(3)
    pooled = np.vstack(centered_blocks)
    if pooled.shape[0] < 3:
        return np.eye(3)
    _, _, vh = np.linalg.svd(pooled, full_matrices=False)
    basis = vh[:3].T
    reference = centered_blocks[0]
    for dim in range(2):
        column = reference @ basis[:, dim]
        if np.abs(np.nanmin(column)) > np.abs(np.nanmax(column)):
            basis[:, dim] *= -1.0
    if np.dot(np.cross(basis[:, 0], basis[:, 1]), basis[:, 2]) < 0:
        basis[:, 2] *= -1.0
    return basis


def _project_entry(entry, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    protein_xyz = np.asarray(entry["protein_xyz"], dtype=float)
    ligand_xyz = np.asarray(entry["ligand_xyz"], dtype=float)
    protein_center = protein_xyz.mean(axis=0, keepdims=True)
    protein_proj = (protein_xyz - protein_center) @ basis
    ligand_proj = (ligand_xyz - protein_center) @ basis
    if protein_proj.size:
        protein_xy = protein_proj[:, :2]
        xy_shift = 0.5 * (np.nanmin(protein_xy, axis=0) + np.nanmax(protein_xy, axis=0))
        protein_proj = protein_proj.copy()
        ligand_proj = ligand_proj.copy()
        protein_proj[:, :2] -= xy_shift
        ligand_proj[:, :2] -= xy_shift
    return protein_proj, ligand_proj


def _shared_limits(projected_entries) -> tuple[float, float, float, float]:
    half_span = 0.0
    for protein_proj, ligand_proj in projected_entries:
        if protein_proj.size == 0:
            continue
        pooled = protein_proj[:, :2]
        half_span = max(half_span, float(np.nanmax(np.abs(pooled))))
    half_span = max(half_span, 1e-3) * 1.08
    return -half_span, half_span, -half_span, half_span


def _depth_limits(projected_entries) -> tuple[float, float]:
    z_values = []
    for protein_proj, ligand_proj in projected_entries:
        if protein_proj.size:
            z_values.append(protein_proj[:, 2])
        if ligand_proj.size:
            z_values.append(ligand_proj[:, 2])
    if not z_values:
        return -1.0, 1.0
    pooled = np.concatenate(z_values)
    z_min = float(np.nanmin(pooled))
    z_max = float(np.nanmax(pooled))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or abs(z_max - z_min) < 1e-8:
        return z_min - 0.5, z_max + 0.5
    return z_min, z_max


def _smooth_polyline(points: np.ndarray, subdivisions: int = 8) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.shape[0] < 4:
        return points
    diffs = np.linalg.norm(np.diff(points, axis=0), axis=1)
    if not np.isfinite(diffs).any():
        return points
    t = np.concatenate([[0.0], np.cumsum(np.maximum(diffs, 1e-6))])
    if t[-1] <= 0:
        return points
    n_samples = max(points.shape[0], (points.shape[0] - 1) * int(subdivisions) + 1)
    t_new = np.linspace(0.0, float(t[-1]), n_samples)
    smooth = []
    for dim in range(points.shape[1]):
        spline = CubicSpline(t, points[:, dim], bc_type="natural")
        smooth.append(spline(t_new))
    return np.column_stack(smooth)


def _secondary_structure_runs(indices: list[int], ss_codes: list[str] | None) -> list[tuple[list[int], str]]:
    if len(indices) < 2:
        return []
    if not ss_codes:
        return [(indices, "coil")]
    runs: list[tuple[list[int], str]] = []
    current = [int(indices[0])]
    current_class = _style_class(ss_codes[int(indices[0])])
    prev_idx = int(indices[0])
    for raw_idx in indices[1:]:
        idx = int(raw_idx)
        style_class = _style_class(ss_codes[idx])
        if style_class != current_class:
            if len(current) >= 2:
                runs.append((current, current_class))
            current = [prev_idx, idx]
            current_class = style_class
        else:
            current.append(idx)
        prev_idx = idx
    if len(current) >= 2:
        runs.append((current, current_class))
    return runs or [(indices, "coil")]


def _binary_runs(indices: list[int], mask: list[bool] | None) -> list[list[int]]:
    if len(indices) < 2 or not mask:
        return []
    runs: list[list[int]] = []
    current: list[int] = []
    previous = None
    for idx in indices:
        is_on = bool(mask[int(idx)]) if int(idx) < len(mask) else False
        if is_on:
            if current:
                current.append(int(idx))
            else:
                current = [int(previous), int(idx)] if previous is not None else [int(idx)]
        elif len(current) >= 2:
            runs.append(current)
            current = []
        previous = int(idx)
    if len(current) >= 2:
        runs.append(current)
    return runs


def _make_line_collection(segments, colors, widths, zorder: int) -> LineCollection:
    collection = LineCollection(segments, colors=colors, linewidths=widths, zorder=zorder)
    collection.set_capstyle("round")
    collection.set_joinstyle("round")
    return collection


def _draw_protein_worm(ax, protein_proj: np.ndarray, entry, style: PlotStyleConfig, z_min: float, z_max: float):
    protein_segments = entry.get("protein_segments") or [list(range(protein_proj.shape[0]))]
    ss_codes = entry.get("protein_secondary_structure")
    contact_mask = entry.get("protein_contact_mask") or []
    context_color = _blend_rgb(style.protein_color, (1.0, 1.0, 1.0), 0.60)
    highlight_color = _blend_rgb(style.protein_color, tuple(to_rgb(style.spine_color)), 0.48)
    z_span = max(z_max - z_min, 1e-6)

    for topology_segment in protein_segments:
        segment_xyz = protein_proj[np.asarray(topology_segment, dtype=int)]
        smooth_xyz = _smooth_polyline(segment_xyz, subdivisions=10)
        if smooth_xyz.shape[0] < 2:
            continue
        ax.plot(
            smooth_xyz[:, 0],
            smooth_xyz[:, 1],
            color=context_color,
            linewidth=4.60,
            alpha=0.18,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=0,
        )

    segment_records = []
    for topology_segment in protein_segments:
        for run_indices, style_class in _secondary_structure_runs(topology_segment, ss_codes):
            run_xyz = protein_proj[np.asarray(run_indices, dtype=int)]
            smooth_xyz = _smooth_polyline(run_xyz, subdivisions=11 if style_class != "coil" else 8)
            if smooth_xyz.shape[0] < 2:
                continue
            style_spec = _PROTEIN_STYLE[style_class]
            for start, end in zip(smooth_xyz[:-1], smooth_xyz[1:]):
                z_mean = 0.5 * float(start[2] + end[2])
                z_norm = float(np.clip((z_mean - z_min) / z_span, 0.0, 1.0))
                rgba = (
                    *_blend_rgb(style.protein_color, (1.0, 1.0, 1.0), 0.48 * (1.0 - z_norm)),
                    style_spec["alpha"] * (0.72 + 0.26 * z_norm),
                )
                width = style_spec["width"] * (0.93 + 0.10 * z_norm)
                segment_records.append((z_mean, np.stack([start[:2], end[:2]], axis=0), rgba, width))
    if segment_records:
        segment_records.sort(key=lambda item: item[0])
        segments = [item[1] for item in segment_records]
        main_colors = [item[2] for item in segment_records]
        main_widths = [item[3] for item in segment_records]
        outline_colors = [(1.0, 1.0, 1.0, min(0.96, color[3] + 0.04)) for color in main_colors]
        outline_widths = [width + 0.95 for width in main_widths]
        ax.add_collection(_make_line_collection(segments, outline_colors, outline_widths, zorder=1))
        ax.add_collection(_make_line_collection(segments, main_colors, main_widths, zorder=2))

    for topology_segment in protein_segments:
        for contact_run in _binary_runs(topology_segment, contact_mask):
            run_xyz = protein_proj[np.asarray(contact_run, dtype=int)]
            smooth_xyz = _smooth_polyline(run_xyz, subdivisions=12)
            if smooth_xyz.shape[0] < 2:
                continue
            ax.plot(
                smooth_xyz[:, 0],
                smooth_xyz[:, 1],
                color="white",
                linewidth=4.15,
                alpha=0.92,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=2.8,
            )
            ax.plot(
                smooth_xyz[:, 0],
                smooth_xyz[:, 1],
                color=highlight_color,
                linewidth=3.15,
                alpha=0.94,
                solid_capstyle="round",
                solid_joinstyle="round",
                zorder=3,
            )


def _draw_ligand(
    ax,
    ligand_proj: np.ndarray,
    entry,
    style: PlotStyleConfig,
    z_min: float,
    z_max: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    if ligand_proj.size == 0:
        return False
    z_span = max(z_max - z_min, 1e-6)
    ligand_core = _blend_rgb(style.accent_color, tuple(to_rgb(style.ligand_color)), 0.28)
    ligand_dark = _blend_rgb(ligand_core, tuple(to_rgb(style.spine_color)), 0.18)
    ligand_glow = _blend_rgb(ligand_core, (1.0, 1.0, 1.0), 0.45)

    ligand_centroid = ligand_proj.mean(axis=0)
    centroid_norm = float(np.clip((float(ligand_centroid[2]) - z_min) / z_span, 0.0, 1.0))
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    margin_x = 0.08 * x_span
    margin_y = 0.08 * y_span
    centroid_outside = not (
        x_min + margin_x <= ligand_centroid[0] <= x_max - margin_x
        and y_min + margin_y <= ligand_centroid[1] <= y_max - margin_y
    )
    if centroid_outside:
        clipped = np.array([
            np.clip(ligand_centroid[0], x_min + margin_x, x_max - margin_x),
            np.clip(ligand_centroid[1], y_min + margin_y, y_max - margin_y),
        ])
        direction = np.asarray(ligand_centroid[:2]) - clipped
        norm = float(np.linalg.norm(direction))
        if norm > 1e-6:
            unit = direction / norm
            start = clipped - 0.10 * min(x_span, y_span) * unit
            ax.annotate(
                "",
                xy=clipped,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=ligand_dark,
                    linewidth=1.35,
                    alpha=0.88,
                    shrinkA=0.0,
                    shrinkB=1.0,
                ),
                zorder=6,
            )
        ax.scatter(
            [clipped[0]],
            [clipped[1]],
            s=320.0,
            c=[ligand_glow],
            alpha=0.16,
            linewidths=0.0,
            zorder=4,
        )
        ax.scatter(
            [clipped[0]],
            [clipped[1]],
            s=92.0,
            c=[ligand_core],
            edgecolors="white",
            linewidths=1.0,
            zorder=7,
        )
        return True

    ax.scatter(
        [ligand_centroid[0]],
        [ligand_centroid[1]],
        s=360.0 * (0.94 + 0.10 * centroid_norm),
        c=[ligand_glow],
        alpha=0.18,
        linewidths=0.0,
        zorder=4,
    )

    bond_records = []
    for bond_a, bond_b in entry.get("ligand_bonds", []):
        start = ligand_proj[int(bond_a)]
        end = ligand_proj[int(bond_b)]
        z_mean = 0.5 * float(start[2] + end[2])
        z_norm = float(np.clip((z_mean - z_min) / z_span, 0.0, 1.0))
        rgba = (*_blend_rgb(ligand_dark, (1.0, 1.0, 1.0), 0.20 * (1.0 - z_norm)), 0.96)
        width = 2.10 * (0.92 + 0.12 * z_norm)
        bond_records.append((z_mean, np.stack([start[:2], end[:2]], axis=0), rgba, width))
    if bond_records:
        bond_records.sort(key=lambda item: item[0])
        bond_segments = [item[1] for item in bond_records]
        bond_colors = [item[2] for item in bond_records]
        bond_widths = [item[3] for item in bond_records]
        ax.add_collection(_make_line_collection(bond_segments, [(1.0, 1.0, 1.0, 0.96)] * len(bond_segments), [w + 0.95 for w in bond_widths], zorder=5))
        ax.add_collection(_make_line_collection(bond_segments, bond_colors, bond_widths, zorder=6))

    atom_records = []
    for atom_xyz in ligand_proj:
        z_norm = float(np.clip((float(atom_xyz[2]) - z_min) / z_span, 0.0, 1.0))
        atom_color = _blend_rgb(ligand_core, (1.0, 1.0, 1.0), 0.16 * (1.0 - z_norm))
        atom_records.append((float(atom_xyz[2]), atom_xyz[:2], atom_color, 38.0 * (0.92 + 0.20 * z_norm)))
    atom_records.sort(key=lambda item: item[0])
    ax.scatter(
        [item[1][0] for item in atom_records],
        [item[1][1] for item in atom_records],
        s=[item[3] for item in atom_records],
        c=[item[2] for item in atom_records],
        edgecolors="white",
        linewidths=0.90,
        zorder=7,
    )
    return False


def _panel_header(ax, entry, style: PlotStyleConfig, accent_color: str, panel_index: int, ligand_offscale: bool):
    panel_letter = chr(ord("a") + panel_index)
    header_font = min(max(style.legend_size + 0.25, 7.1), 8.0)
    sub_font = min(max(style.legend_size - 0.55, 6.0), 6.7)
    cluster_id = entry.get("cluster_id")
    population_fraction = entry.get("population_fraction")
    n_frames = entry.get("n_frames")
    title = str(entry.get("title", f"Snapshot {panel_index + 1}"))
    title_main, _, title_sub = title.partition("\n")
    ax.text(
        0.025,
        0.965,
        panel_letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=header_font,
        fontweight="bold",
        color=style.spine_color,
        bbox=dict(boxstyle="round,pad=0.26", facecolor=accent_color, edgecolor="none", alpha=0.17),
    )
    ax.text(
        0.125,
        0.965,
        f"State {cluster_id}" if cluster_id is not None else title_main,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=header_font,
        fontweight="semibold",
        color=style.spine_color,
    )
    if population_fraction is not None:
        ax.text(
            0.975,
            0.965,
            f"{100.0 * float(population_fraction):.1f}%",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=header_font,
            fontweight="semibold",
            color=style.spine_color,
        )
    ax.text(
        0.125,
        0.900,
        (
            (f"n = {int(n_frames):,}" if n_frames is not None else title_sub or "representative snapshot")
            + ("  |  ligand off-scale" if ligand_offscale else "")
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=sub_font,
        color=style.spine_color,
        alpha=0.78,
    )


def snapshot_grid(snapshot_entries, out_base, style: PlotStyleConfig, title: str):
    if not snapshot_entries:
        return
    n = len(snapshot_entries)
    rows, cols = _panel_grid(n)
    basis = _shared_projection_basis(snapshot_entries)
    projected_entries = [_project_entry(entry, basis) for entry in snapshot_entries]
    x_min, x_max, y_min, y_max = _shared_limits(projected_entries)
    z_min, z_max = _depth_limits(projected_entries)
    panel_width = 3.55 if cols <= 2 else 3.20
    panel_height = 3.55
    top_band = 0.36 if title else 0.10
    fig_width = cols * panel_width + 0.18
    fig_height = rows * panel_height + top_band
    with publication_style(style):
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
        axes = np.atleast_1d(axes).ravel()
        for ax in axes[n:]:
            ax.axis("off")
        fig.subplots_adjust(
            left=0.028,
            right=0.992,
            top=0.955 if title else 0.985,
            bottom=0.028,
            wspace=0.05 if cols == 1 else 0.06,
            hspace=0.07 if rows == 1 else 0.08,
        )
        for panel_index, (ax, entry, (protein_proj, ligand_proj)) in enumerate(zip(axes, snapshot_entries, projected_entries)):
            color_index = int(entry.get("cluster_id", panel_index)) % len(style.categorical_palette)
            accent_color = style.categorical_palette[color_index]
            _draw_protein_worm(ax, protein_proj, entry, style, z_min, z_max)
            ligand_offscale = _draw_ligand(ax, ligand_proj, entry, style, z_min, z_max, x_min, x_max, y_min, y_max)
            _panel_header(ax, entry, style, accent_color, panel_index, ligand_offscale)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            for spine in ax.spines.values():
                spine.set_visible(False)

        if title:
            fig.text(
                0.03,
                0.984,
                title,
                ha="left",
                va="top",
                fontsize=max(style.legend_size + 0.55, 8.0),
                fontweight="semibold",
                color=style.spine_color,
            )
            first_entry = snapshot_entries[0]
            if "n_clusters_shown" in first_entry and "n_clusters_total" in first_entry and "cumulative_population_fraction" in first_entry:
                fig.text(
                    0.992,
                    0.984,
                    (
                        f"Top {int(first_entry['n_clusters_shown'])} of {int(first_entry['n_clusters_total'])} states"
                        f"  |  cumulative occupancy {100.0 * float(first_entry['cumulative_population_fraction']):.1f}%"
                    ),
                    ha="right",
                    va="top",
                    fontsize=max(style.legend_size - 0.75, 6.2),
                    color=style.spine_color,
                    alpha=0.75,
                )
        save_figure(fig, out_base, style)
