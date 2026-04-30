from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import binary_dilation, gaussian_filter, minimum_filter
from scipy.stats import gaussian_kde

from ..config import PlotStyleConfig
from ..core import read_dict_csv, save_csv, write_json
from .bars import vertical_bars
from .heatmaps import matrix_heatmap
from .snapshots import snapshot_grid
from .theme import finalize_axes, publication_style, resolve_colormap, save_figure, subtle_fill_color


@dataclass(frozen=True)
class FESSurface:
    x: np.ndarray
    y: np.ndarray
    probability: np.ndarray
    energy: np.ndarray
    estimator: str
    n_samples: int
    n_unique_samples: int
    density_floor_fraction: float
    sparse_sampling_warning: str | None


def _prepare_xy(x, y):
    xy = np.column_stack([np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()])
    finite = np.all(np.isfinite(xy), axis=1)
    xy = xy[finite]
    if xy.shape[0] < 2:
        raise ValueError("FEL 至少需要两个有效投影点。")
    return xy[:, 0], xy[:, 1]


def _categorical_cmap(n_categories: int, style: PlotStyleConfig):
    if n_categories <= 0:
        raise ValueError("Categorical colormap requires at least one category.")
    repeats = int(np.ceil(n_categories / len(style.categorical_palette)))
    colors = (style.categorical_palette * repeats)[:n_categories]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, n_categories + 0.5, 1.0), cmap.N)
    return cmap, norm


def _build_relative_free_energy(probability, temperature_K, kB_kcal_mol_K):
    if not np.isfinite(probability).any():
        raise ValueError("无法从当前投影点估计有效的二维概率分布。")
    reference_prob = float(np.nanmax(probability))
    if reference_prob <= 0:
        raise ValueError("二维概率分布的最大值为零，无法计算相对自由能。")
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = -kB_kcal_mol_K * temperature_K * np.log(probability / reference_prob)
    return np.where(np.isfinite(energy), np.maximum(energy, 0.0), np.nan)


def _grid_from_ranges(x, y, grid_size):
    span_x = max(float(np.ptp(x)), 1e-6)
    span_y = max(float(np.ptp(y)), 1e-6)
    pad_x = 0.08 * span_x
    pad_y = 0.08 * span_y
    x_grid = np.linspace(float(np.min(x)) - pad_x, float(np.max(x)) + pad_x, int(grid_size))
    y_grid = np.linspace(float(np.min(y)) - pad_y, float(np.max(y)) + pad_y, int(grid_size))
    return np.meshgrid(x_grid, y_grid)


def _estimate_fes_surface_kde(x, y, temperature_K, kB_kcal_mol_K):
    n_samples = len(x)
    grid_size = int(min(220, max(80, round(np.sqrt(n_samples) * 8.0))))
    xx, yy = _grid_from_ranges(x, y, grid_size)
    values = np.vstack([x, y])
    density = gaussian_kde(values)(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    density_floor_fraction = 1e-3 if n_samples < 100 else 1e-4
    density = np.where(density >= float(np.nanmax(density)) * density_floor_fraction, density, np.nan)
    return FESSurface(
        x=xx,
        y=yy,
        probability=density,
        energy=_build_relative_free_energy(density, temperature_K, kB_kcal_mol_K),
        estimator="gaussian_kde",
        n_samples=n_samples,
        n_unique_samples=int(np.unique(np.column_stack([x, y]), axis=0).shape[0]),
        density_floor_fraction=density_floor_fraction,
        sparse_sampling_warning=(
            "Projected sampling is sparse; this FEL is a qualitative density estimate and low-energy basins should be interpreted cautiously."
            if n_samples < 25
            else None
        ),
    )


def _estimate_fes_surface_histogram(x, y, n_bins, temperature_K, kB_kcal_mol_K):
    n_samples = len(x)
    bins_eff = int(min(n_bins, max(4, round(np.sqrt(n_samples) * 1.5))))
    hist, xedges, yedges = np.histogram2d(x, y, bins=bins_eff, density=True)
    hist = hist.T
    support = hist > 0
    estimator = "histogram"
    if np.count_nonzero(support) >= 12:
        sigma = 0.85 if bins_eff >= 20 else 0.65
        hist = gaussian_filter(hist, sigma=sigma, mode="nearest")
        support = binary_dilation(support, iterations=1)
        hist = np.where(support, hist, np.nan)
        estimator = f"smoothed_histogram_sigma_{sigma:.2f}"
    else:
        hist = np.where(support, hist, np.nan)
    density_floor_fraction = 1e-4
    if np.isfinite(hist).any():
        hist = np.where(hist >= float(np.nanmax(hist)) * density_floor_fraction, hist, np.nan)
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(xc, yc)
    return FESSurface(
        x=xx,
        y=yy,
        probability=hist,
        energy=_build_relative_free_energy(hist, temperature_K, kB_kcal_mol_K),
        estimator=estimator,
        n_samples=n_samples,
        n_unique_samples=int(np.unique(np.column_stack([x, y]), axis=0).shape[0]),
        density_floor_fraction=density_floor_fraction,
        sparse_sampling_warning=(
            "Projected sampling is sparse; this FEL is based on coarse histogram occupancy and should be interpreted qualitatively."
            if n_samples < 25
            else None
        ),
    )


def free_energy_from_2d(x, y, n_bins, temperature_K, kB_kcal_mol_K):
    x, y = _prepare_xy(x, y)
    n_unique = int(np.unique(np.column_stack([x, y]), axis=0).shape[0])
    if n_unique >= 5 and len(x) <= 5000:
        try:
            return _estimate_fes_surface_kde(x, y, temperature_K, kB_kcal_mol_K)
        except Exception:
            pass
    return _estimate_fes_surface_histogram(x, y, n_bins, temperature_K, kB_kcal_mol_K)


def _energy_display_cap(energy):
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return 1.0
    if finite.size >= 25:
        cap = float(np.quantile(finite, 0.98))
    else:
        cap = float(np.nanmax(finite))
    if (not np.isfinite(cap)) or cap <= 0:
        cap = float(np.nanmax(finite))
    return cap if cap > 0 else 1.0


def _identify_low_energy_basins(energy, max_basins=4):
    finite = energy[np.isfinite(energy)]
    if finite.size == 0:
        return []
    arr = np.where(np.isfinite(energy), energy, np.inf)
    window = max(5, int(round(min(arr.shape) / 18)))
    if window % 2 == 0:
        window += 1
    low_energy_threshold = min(3.0, float(np.quantile(finite, 0.25)))
    low_energy_threshold = max(0.35, low_energy_threshold)
    minima_mask = (arr == minimum_filter(arr, size=window, mode="nearest")) & np.isfinite(arr) & (arr <= low_energy_threshold)
    candidates = sorted(np.argwhere(minima_mask), key=lambda ij: arr[tuple(ij)])
    min_sep = max(4, int(round(min(arr.shape) / 12)))
    selected: list[tuple[int, int]] = []
    for iy, ix in candidates:
        if any((iy - sy) ** 2 + (ix - sx) ** 2 < min_sep ** 2 for sy, sx in selected):
            continue
        selected.append((int(iy), int(ix)))
        if len(selected) >= max_basins:
            break
    if not selected:
        iy, ix = np.unravel_index(int(np.nanargmin(energy)), energy.shape)
        selected = [(int(iy), int(ix))]
    return selected


def _plot_fes_3d(surface, out_base, title, xlabel, ylabel, display_cap, basin_indices, basin_labels, style: PlotStyleConfig):
    z_floor = -0.12 * display_cap if display_cap > 0 else -0.2
    z_surface = np.ma.masked_invalid(np.where(np.isfinite(surface.energy), np.minimum(surface.energy, display_cap), np.nan))
    surface_title = f"{title} (3D view)"
    contour_levels = np.linspace(0.0, display_cap, 8)
    with publication_style(style):
        fig = plt.figure(figsize=(7.8, 5.8))
        ax = fig.add_subplot(111, projection="3d")
        cmap = resolve_colormap(style.cmap_continuous, style)
        mesh = ax.plot_surface(
            surface.x,
            surface.y,
            z_surface,
            cmap=cmap,
            linewidth=0.0,
            antialiased=True,
            shade=True,
            alpha=0.98,
            rcount=min(150, z_surface.shape[0]),
            ccount=min(150, z_surface.shape[1]),
        )
        ax.contour(
            surface.x,
            surface.y,
            z_surface,
            zdir="z",
            offset=z_floor,
            levels=contour_levels,
            cmap=cmap,
            linewidths=max(0.7, style.thin_line_width),
            alpha=0.92,
        )
        for idx, (iy, ix) in enumerate(basin_indices):
            x0 = float(surface.x[iy, ix])
            y0 = float(surface.y[iy, ix])
            z0 = float(surface.energy[iy, ix])
            is_global_min = idx == 0
            ax.scatter(
                [x0],
                [y0],
                [z0],
                s=88 if is_global_min else 36,
                marker="*" if is_global_min else "o",
                c="white",
                edgecolors=style.mean_line_color,
                linewidths=0.8,
                depthshade=False,
                zorder=6,
            )
            ax.plot(
                [x0, x0],
                [y0, y0],
                [z_floor, z0],
                linestyle="--",
                linewidth=max(0.7, style.thin_line_width),
                color=style.mean_line_color,
                alpha=0.55,
                zorder=5,
            )
            ax.text(
                x0,
                y0,
                z0 + 0.06 * max(display_cap, 1.0),
                basin_labels[(iy, ix)],
                color=style.mean_line_color,
                fontsize=style.legend_size,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.2),
                zorder=7,
            )
        ax.set_xlabel(xlabel, labelpad=8)
        ax.set_ylabel(ylabel, labelpad=8)
        ax.set_zlabel("Relative free energy, ΔG (kcal/mol)", labelpad=7)
        ax.set_title(surface_title, pad=12, weight="semibold")
        ax.set_zlim(z_floor, display_cap)
        ax.view_init(elev=34, azim=-58)
        ax.grid(style.show_grid)
        if hasattr(ax, "set_box_aspect"):
            ax.set_box_aspect((1.0, 1.0, 0.55))
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor(style.grid_color)
        if surface.sparse_sampling_warning:
            ax.text2D(
                0.02,
                0.03,
                "Qualitative only: sparse sampling",
                transform=ax.transAxes,
                fontsize=max(style.legend_size - 0.8, 6.2),
                color=style.spine_color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.25),
            )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.042, pad=0.12, shrink=0.78)
        cbar.set_label("ΔG (kcal/mol)", labelpad=8)
        save_figure(fig, out_base, style)


def plot_line_profile(x, y, out_base, xlabel, ylabel, title, style: PlotStyleConfig, color=None, yscale: str | None = None):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    if yscale == "log":
        keep &= y_arr > 0.0
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size == 0:
        return
    marker_stride = max(1, int(np.ceil(x_arr.size / 60)))
    markevery = marker_stride if x_arr.size > 80 else None
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(
            x_arr,
            y_arr,
            linewidth=style.line_width,
            marker="o",
            markersize=style.marker_size,
            markevery=markevery,
            color=color or style.accent_color,
        )
        if yscale:
            ax.set_yscale(yscale)
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)


def plot_fes(x, y, out_base, title, xlabel, ylabel, n_bins, temperature_K, kB_kcal_mol_K, style: PlotStyleConfig):
    surface = free_energy_from_2d(x, y, n_bins, temperature_K, kB_kcal_mol_K)
    x_points, y_points = _prepare_xy(x, y)
    display_cap = _energy_display_cap(surface.energy)
    energy_plot = np.where(np.isfinite(surface.energy), np.minimum(surface.energy, display_cap), np.nan)
    levels = np.linspace(0.0, display_cap, 13)
    basin_indices = _identify_low_energy_basins(surface.energy, max_basins=4)
    basin_labels = {}
    for idx, (iy, ix) in enumerate(basin_indices):
        basin_labels[(iy, ix)] = "GM" if idx == 0 else f"B{idx}"
    x_span = max(float(np.ptp(x_points)), 1.0)
    y_span = max(float(np.ptp(y_points)), 1.0)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        mesh = ax.contourf(surface.x, surface.y, energy_plot, levels=levels, cmap=resolve_colormap(style.cmap_continuous, style), extend="max")
        contour_levels = levels[1:-1:2]
        if len(contour_levels):
            ax.contour(surface.x, surface.y, energy_plot, levels=contour_levels, colors="white", linewidths=max(0.75, style.thin_line_width), alpha=0.78)
        ax.scatter(
            x_points,
            y_points,
            s=max(10.0, style.marker_size * 2.4),
            color=style.mean_line_color,
            alpha=0.18,
            edgecolors="none",
            zorder=3,
        )
        for idx, (iy, ix) in enumerate(basin_indices):
            x0 = float(surface.x[iy, ix])
            y0 = float(surface.y[iy, ix])
            is_global_min = idx == 0
            ax.scatter(
                [x0],
                [y0],
                s=115 if is_global_min else 42,
                marker="*" if is_global_min else "o",
                facecolors="white",
                edgecolors=style.mean_line_color,
                linewidths=0.85,
                zorder=4,
            )
            ax.text(
                x0 + 0.03 * x_span,
                y0 + 0.03 * y_span,
                basin_labels[(iy, ix)],
                fontsize=style.legend_size,
                color=style.mean_line_color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.2),
                zorder=5,
            )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        if surface.sparse_sampling_warning:
            ax.text(
                0.02,
                0.02,
                "Qualitative only: sparse sampling",
                transform=ax.transAxes,
                fontsize=max(style.legend_size - 0.8, 6.2),
                color=style.spine_color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.76, pad=0.25),
                zorder=6,
            )
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative free energy, ΔG (kcal/mol)")
        save_figure(fig, out_base, style)
    out_base_path = Path(out_base)
    _plot_fes_3d(
        surface,
        out_base_path.with_name(f"{out_base_path.name}_3d"),
        title,
        xlabel,
        ylabel,
        display_cap,
        basin_indices,
        basin_labels,
        style,
    )
    rows = []
    basin_metadata = []
    for iy, ix in basin_indices:
        basin_metadata.append(
            {
                "label": basin_labels[(iy, ix)],
                "x": float(surface.x[iy, ix]),
                "y": float(surface.y[iy, ix]),
                "relative_free_energy_kcal_mol": float(surface.energy[iy, ix]),
            }
        )
    for j in range(surface.y.shape[0]):
        for i in range(surface.x.shape[1]):
            basin_label = basin_labels.get((j, i), "")
            prob = surface.probability[j, i]
            energy = surface.energy[j, i]
            rows.append(
                [
                    float(surface.x[j, i]),
                    float(surface.y[j, i]),
                    float(prob) if np.isfinite(prob) else np.nan,
                    float(energy) if np.isfinite(energy) else np.nan,
                    bool(basin_label),
                    basin_label,
                ]
            )
    save_csv(
        Path(out_base).with_suffix(".csv"),
        [xlabel, ylabel, "probability_density", "relative_free_energy_kcal_mol", "is_local_minimum", "basin_label"],
        rows,
    )
    write_json(
        Path(out_base).with_suffix(".json"),
        {
            "title": title,
            "x_label": xlabel,
            "y_label": ylabel,
            "definition": "Relative free energy was estimated as -kB*T*ln(P/Pmax) on the projected 2D coordinate surface.",
            "temperature_K": float(temperature_K),
            "boltzmann_constant_kcal_mol_K": float(kB_kcal_mol_K),
            "estimator": surface.estimator,
            "n_input_samples": int(surface.n_samples),
            "n_unique_samples": int(surface.n_unique_samples),
            "grid_shape": [int(surface.energy.shape[0]), int(surface.energy.shape[1])],
            "density_floor_fraction": float(surface.density_floor_fraction),
            "display_energy_cap_kcal_mol": float(display_cap),
            "basin_markers": basin_metadata,
            "warning": surface.sparse_sampling_warning,
        },
    )


def _truthy_csv_value(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def plot_fes_from_csv(csv_path, out_base, title, xlabel, ylabel, style: PlotStyleConfig) -> bool:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return False
    rows = read_dict_csv(csv_path)
    if not rows:
        return False
    try:
        x_values = sorted({float(row[xlabel]) for row in rows})
        y_values = sorted({float(row[ylabel]) for row in rows})
    except (KeyError, TypeError, ValueError):
        return False
    if not x_values or not y_values:
        return False
    x_index = {value: idx for idx, value in enumerate(x_values)}
    y_index = {value: idx for idx, value in enumerate(y_values)}
    probability = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    energy = np.full_like(probability, np.nan)
    basin_labels = {}
    for row in rows:
        try:
            ix = x_index[float(row[xlabel])]
            iy = y_index[float(row[ylabel])]
            probability[iy, ix] = float(row["probability_density"])
            energy[iy, ix] = float(row["relative_free_energy_kcal_mol"])
        except (KeyError, TypeError, ValueError):
            continue
        label = str(row.get("basin_label", "")).strip()
        if label and _truthy_csv_value(row.get("is_local_minimum", "")):
            basin_labels[(iy, ix)] = label
    if not np.isfinite(energy).any():
        return False
    x_grid, y_grid = np.meshgrid(np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float))
    plot_fes_grid(
        x_grid,
        y_grid,
        probability,
        energy,
        out_base,
        title,
        xlabel,
        ylabel,
        style,
        basin_labels=basin_labels or None,
    )
    return True


def plot_fes_grid(
    x_grid,
    y_grid,
    probability,
    energy,
    out_base,
    title,
    xlabel,
    ylabel,
    style: PlotStyleConfig,
    basin_labels: dict[tuple[int, int], str] | None = None,
):
    surface = FESSurface(
        x=np.asarray(x_grid, dtype=float),
        y=np.asarray(y_grid, dtype=float),
        probability=np.asarray(probability, dtype=float),
        energy=np.asarray(energy, dtype=float),
        estimator="existing_grid",
        n_samples=0,
        n_unique_samples=0,
        density_floor_fraction=0.0,
        sparse_sampling_warning=None,
    )
    display_cap = _energy_display_cap(surface.energy)
    energy_plot = np.where(np.isfinite(surface.energy), np.minimum(surface.energy, display_cap), np.nan)
    levels = np.linspace(0.0, display_cap, 13)
    if basin_labels:
        basin_indices = sorted(basin_labels, key=lambda ij: float(surface.energy[ij]) if np.isfinite(surface.energy[ij]) else np.inf)[:4]
        labels = dict(basin_labels)
    else:
        basin_indices = _identify_low_energy_basins(surface.energy, max_basins=4)
        labels = {(iy, ix): ("GM" if idx == 0 else f"B{idx}") for idx, (iy, ix) in enumerate(basin_indices)}
    x_finite = surface.x[np.isfinite(surface.x)]
    y_finite = surface.y[np.isfinite(surface.y)]
    x_span = max(float(np.ptp(x_finite)), 1.0) if x_finite.size else 1.0
    y_span = max(float(np.ptp(y_finite)), 1.0) if y_finite.size else 1.0
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 5.4))
        mesh = ax.contourf(surface.x, surface.y, energy_plot, levels=levels, cmap=resolve_colormap(style.cmap_continuous, style), extend="max")
        contour_levels = levels[1:-1:2]
        if len(contour_levels):
            ax.contour(surface.x, surface.y, energy_plot, levels=contour_levels, colors="white", linewidths=max(0.75, style.thin_line_width), alpha=0.78)
        for idx, (iy, ix) in enumerate(basin_indices):
            x0 = float(surface.x[iy, ix])
            y0 = float(surface.y[iy, ix])
            is_global_min = idx == 0
            ax.scatter(
                [x0],
                [y0],
                s=115 if is_global_min else 42,
                marker="*" if is_global_min else "o",
                facecolors="white",
                edgecolors=style.mean_line_color,
                linewidths=0.85,
                zorder=4,
            )
            ax.text(
                x0 + 0.03 * x_span,
                y0 + 0.03 * y_span,
                labels.get((iy, ix), "GM" if idx == 0 else f"B{idx}"),
                fontsize=style.legend_size,
                color=style.mean_line_color,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.2),
                zorder=5,
            )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Relative free energy, ΔG (kcal/mol)")
        save_figure(fig, out_base, style)
    out_base_path = Path(out_base)
    _plot_fes_3d(
        surface,
        out_base_path.with_name(f"{out_base_path.name}_3d"),
        title,
        xlabel,
        ylabel,
        display_cap,
        basin_indices,
        labels,
        style,
    )


def scatter_by_replica(all_xy, replica_dirs, feature_list, out_base, xlabel, ylabel, title, style: PlotStyleConfig):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        start = 0
        for idx, (rd, feat) in enumerate(zip(replica_dirs, feature_list)):
            n = feat.shape[0]
            seg = all_xy[start:start+n]
            ax.scatter(seg[:, 0], seg[:, 1], s=10, alpha=0.32, label=rd.name, color=style.categorical_palette[idx % len(style.categorical_palette)], edgecolors="none")
            start += n
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(frameon=False, markerscale=1.6)
        save_figure(fig, out_base, style)


def scatter_clusters(embed, labels, centers, out_base, style: PlotStyleConfig):
    n_clusters = int(np.max(labels)) + 1 if len(labels) else 1
    cmap, norm = _categorical_cmap(n_clusters, style)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 5.2))
        sc = ax.scatter(embed[:, 0], embed[:, 1], c=labels, cmap=cmap, norm=norm, s=18, alpha=0.55, edgecolors="none")
        ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=90, linewidths=0.8, color=style.mean_line_color, edgecolors="white")
        finalize_axes(ax, style, xlabel="tIC1", ylabel="tIC2", title="KMeans clusters in tICA space")
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_ticks(np.arange(n_clusters))
        cbar.set_label("Cluster index")
        save_figure(fig, out_base, style)


def plot_cluster_population(clusters, fractions, out_base, style: PlotStyleConfig):
    vertical_bars(clusters, fractions, out_base, style, title="Overall cluster populations", xlabel="Cluster", ylabel="Population fraction")


def _coerce_state_labels(state_labels, n_states: int) -> list[str]:
    if state_labels is None:
        return [f"State {idx}" for idx in range(n_states)]
    labels = [str(item).strip() for item in state_labels]
    if len(labels) != n_states or any(not label for label in labels):
        return [f"State {idx}" for idx in range(n_states)]
    return labels


def _state_colors(n_states: int, style: PlotStyleConfig) -> list[str]:
    repeats = int(np.ceil(max(n_states, 1) / len(style.categorical_palette)))
    return (style.categorical_palette * repeats)[:n_states]


def _as_real_array(values) -> np.ndarray:
    return np.asarray(np.real_if_close(values), dtype=float)


def _summary_box(ax, lines, style: PlotStyleConfig, *, loc: str = "upper right") -> None:
    if not lines:
        return
    x = 0.98 if "right" in loc else 0.02
    ha = "right" if "right" in loc else "left"
    y = 0.98 if "upper" in loc else 0.02
    va = "top" if "upper" in loc else "bottom"
    ax.text(
        x,
        y,
        "\n".join(str(line) for line in lines if str(line).strip()),
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=max(style.tick_size - 0.7, 6.8),
        color=style.spine_color,
        bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=style.grid_color, linewidth=0.8, alpha=0.94),
        zorder=10,
    )


def _format_frame_value(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if value >= 1000:
        return f"{value:,.0f}"
    if value >= 100:
        return f"{value:.0f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _top_offdiagonal_edges(transition_matrix, flux_matrix=None, limit: int = 8):
    T = _as_real_array(transition_matrix)
    flux = None if flux_matrix is None else _as_real_array(flux_matrix)
    rows = []
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            if i == j or not np.isfinite(T[i, j]) or T[i, j] <= 0.0:
                continue
            metric = float(flux[i, j]) if flux is not None and np.isfinite(flux[i, j]) else float(T[i, j])
            rows.append((i, j, float(T[i, j]), metric))
    rows.sort(key=lambda item: (item[3], item[2], -item[0], -item[1]), reverse=True)
    return rows[:limit]


def plot_stationary_distribution(states, probs, out_base, style: PlotStyleConfig, state_labels=None, summary_lines=None):
    probs_arr = _as_real_array(probs)
    if probs_arr.size == 0:
        return
    labels = _coerce_state_labels(state_labels, probs_arr.size)
    order = np.argsort(probs_arr)[::-1]
    ordered_probs = probs_arr[order]
    ordered_labels = [labels[idx] for idx in order]
    colors = [_state_colors(len(labels), style)[idx] for idx in order]
    y = np.arange(len(ordered_probs))
    height = max(3.2, 0.58 * len(ordered_probs) + 1.8)
    uniform_prob = 1.0 / max(len(ordered_probs), 1)

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.6, height))
        row_fill = subtle_fill_color(style)
        for idx in range(len(ordered_probs)):
            if idx % 2 == 0:
                ax.axhspan(idx - 0.5, idx + 0.5, color=row_fill, zorder=0)
        ax.barh(y, ordered_probs, height=0.68, color=colors, alpha=0.18, edgecolor="none", zorder=1)
        ax.hlines(y, 0.0, ordered_probs, color=colors, linewidth=2.2, alpha=0.65, zorder=2)
        ax.scatter(ordered_probs, y, s=max(style.marker_size * 26.0, 52.0), color=colors, edgecolors="white", linewidths=1.0, zorder=3)
        ax.axvline(uniform_prob, linestyle="--", linewidth=1.0, color=style.mean_line_color, alpha=0.55, zorder=1)
        for ypos, prob in zip(y, ordered_probs):
            ax.annotate(
                f"{prob:.1%}",
                xy=(prob, ypos),
                xytext=(8, 0),
                textcoords="offset points",
                va="center",
                ha="left",
                fontsize=max(style.tick_size - 0.1, 7.1),
                color=style.spine_color,
                zorder=4,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(ordered_labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, min(1.02, max(0.40, float(np.nanmax(ordered_probs)) * 1.18)))
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        finalize_axes(ax, style, xlabel="Stationary probability", title="MSM stationary-state occupancy")
        ax.grid(True, axis="x", linestyle="-", linewidth=0.7)
        ax.grid(False, axis="y")
        save_figure(fig, out_base, style)


def _format_probability(value: float) -> str:
    value = float(value)
    if not np.isfinite(value):
        return "nan"
    if value == 0.0:
        return "0"
    if abs(value) < 0.01:
        return f"{value:.1e}"
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def plot_lag_scan(lag_rows, out_base, style: PlotStyleConfig, selected_lag: int | None = None, diagnostic_rows=None):
    arr = np.asarray(lag_rows, dtype=float)
    if arr.size == 0:
        return
    out_base = Path(out_base)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.9))

        for proc in sorted(set(arr[:, 1].astype(int))):
            sub = arr[arr[:, 1] == proc]
            sub = sub[np.argsort(sub[:, 0])]
            color = style.categorical_palette[(proc - 1) % len(style.categorical_palette)]
            ax.plot(
                sub[:, 0],
                sub[:, 2],
                marker="o",
                markersize=max(style.marker_size - 0.2, 3.8),
                linewidth=style.line_width,
                label=f"Process {proc}",
                color=color,
            )
        x0, x1 = float(arr[:, 0].min()), float(arr[:, 0].max())
        xref = np.linspace(x0, x1, 200)
        ax.plot(xref, xref, linestyle="--", linewidth=1.0, color=style.mean_line_color, alpha=0.72, label="t = lag")
        if selected_lag is not None:
            ax.axvline(float(selected_lag), linestyle=":", linewidth=1.0, color=style.spine_color, alpha=0.7)
        ax.set_yscale("log")
        finalize_axes(
            ax,
            style,
            xlabel="MSM lag time (frames)",
            ylabel="Implied timescale (frames)",
            title="Implied timescales on a fixed active-state set",
        )
        ax.legend(frameon=False, ncol=1, loc="best")
        save_figure(fig, out_base, style)

    if not diagnostic_rows:
        return

    diag = np.asarray(diagnostic_rows, dtype=float)
    if diag.size == 0:
        return
    diag = diag[np.argsort(diag[:, 0])]
    lags = diag[:, 0]
    usable_segments = diag[:, 2]
    usable_frame_fraction = np.divide(diag[:, 3], np.maximum(diag[:, 4], 1.0))
    x_positions = np.arange(len(lags))
    lag_labels = [f"{int(lag)}" if float(lag).is_integer() else f"{lag:g}" for lag in lags]
    selected_mask = np.isclose(lags, float(selected_lag)) if selected_lag is not None else np.zeros_like(lags, dtype=bool)
    segment_colors = [style.accent_color if selected else style.protein_color for selected in selected_mask]
    frame_colors = [style.accent_color if selected else style.distance_color for selected in selected_mask]

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(x_positions, usable_segments, width=0.72, color=segment_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        finalize_axes(ax, style, xlabel="MSM lag time (frames)", ylabel="Usable segments", title="Lag-scan usable active segments")
        ax.set_xticks(x_positions)
        ax.set_xticklabels(lag_labels)
        save_figure(fig, out_base.with_name("lag_scan_usable_segments"), style)

    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.bar(x_positions, usable_frame_fraction, width=0.72, color=frame_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(lag_labels)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        finalize_axes(ax, style, xlabel="MSM lag time (frames)", ylabel="Usable frame fraction", title="Lag-scan frame support")
        save_figure(fig, out_base.with_name("lag_scan_frame_support"), style)


def plot_state_network(
    transition_matrix,
    stationary_probs,
    out_base,
    style: PlotStyleConfig,
    threshold: float = 1e-6,
    state_labels=None,
    mfpt_matrix=None,
    summary_lines=None,
):
    T = np.asarray(transition_matrix, dtype=float)
    pi = np.asarray(stationary_probs, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("MSM transition matrix must be square for state-network plotting.")
    n = T.shape[0]
    if pi.shape[0] != n:
        raise ValueError("Stationary distribution length must match the MSM transition matrix.")
    T = np.where(np.isfinite(T), np.clip(T, 0.0, None), 0.0)
    pi = np.where(np.isfinite(pi), np.clip(pi, 0.0, None), 0.0)
    pi_sum = float(pi.sum())
    pi = pi / pi_sum if pi_sum > 0 else np.full(n, 1.0 / max(n, 1))
    if n == 1:
        xy = np.array([[0.0, 0.0]])
    elif n == 2:
        xy = np.array([[-0.78, 0.0], [0.78, 0.0]])
    else:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False) + np.pi / 2
        xy = np.column_stack([np.cos(theta), np.sin(theta)])

    flux = pi[:, None] * T
    offdiag = ~np.eye(n, dtype=bool)
    positive_offdiag = offdiag & (T > 0.0) & (flux > 0.0)
    max_offdiag_flux = float(np.max(flux[positive_offdiag])) if np.any(positive_offdiag) else 0.0
    requested_min_flux = max(float(threshold), 0.0)
    used_min_flux = requested_min_flux
    threshold_relaxed = False
    if max_offdiag_flux > 0 and not np.any(positive_offdiag & (flux >= requested_min_flux)):
        used_min_flux = max(max_offdiag_flux * 0.05, 1e-12)
        threshold_relaxed = True

    edges = [
        (i, j, float(T[i, j]), float(flux[i, j]))
        for i in range(n)
        for j in range(n)
        if i != j and T[i, j] > 0.0 and flux[i, j] >= used_min_flux
    ]
    max_edges = max(12, min(48, 3 * max(n, 1)))
    if len(edges) > max_edges:
        edges = sorted(edges, key=lambda item: item[3], reverse=True)[:max_edges]
    edges = sorted(edges, key=lambda item: (item[0], item[1]))
    max_edge_flux = max((edge[3] for edge in edges), default=0.0)
    edge_pairs = {(i, j) for i, j, _, _ in edges}

    labels = _coerce_state_labels(state_labels, n)
    colors = _state_colors(n, style)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 5.9))
        for i, j, transition_prob, edge_flux in edges:
            x0, y0 = xy[i]
            x1, y1 = xy[j]
            relative_flux = (edge_flux / max_edge_flux) ** 0.55 if max_edge_flux > 0 else 1.0
            has_reverse = (j, i) in edge_pairs
            if n == 2:
                curve = 0.24 if i < j else -0.24
            else:
                curve = 0.18 if has_reverse else 0.10
                if i > j and has_reverse:
                    curve *= -1.0
            patch = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={curve}",
                mutation_scale=11.0 + 5.0 * relative_flux,
                linewidth=0.75 + 3.2 * relative_flux,
                color=colors[i % len(colors)],
                alpha=0.28 + 0.48 * relative_flux,
                shrinkA=22,
                shrinkB=22,
                zorder=2,
            )
            ax.add_patch(patch)
            if n <= 12:
                dx, dy = x1 - x0, y1 - y0
                dist = max(float(np.hypot(dx, dy)), 1e-9)
                nx, ny = -dy / dist, dx / dist
                label_x = (x0 + x1) * 0.5 + nx * curve * 0.42
                label_y = (y0 + y1) * 0.5 + ny * curve * 0.42
                mfpt_label = ""
                if mfpt_matrix is not None:
                    try:
                        mfpt_value = float(np.real_if_close(mfpt_matrix[i, j]))
                        if np.isfinite(mfpt_value):
                            mfpt_label = f"\nMFPT={_format_frame_value(mfpt_value)}"
                    except Exception:
                        pass
                ax.text(
                    label_x,
                    label_y,
                    f"P={_format_probability(transition_prob)}{mfpt_label}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 1.3, 7.0),
                    color=style.spine_color,
                    zorder=4,
                )

        sizes = 620.0 + 2300.0 * np.sqrt(pi / pi.max()) if np.max(pi) > 0 else np.full(n, 800.0)
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, color=colors, edgecolors="white", linewidths=1.3, zorder=5)
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, f"S{i}", ha="center", va="center", color="white", fontsize=style.label_size, weight="bold", zorder=6)
            if n <= 16:
                if n == 2:
                    label_x, label_y = x, y - 0.35
                else:
                    radial = np.array([x, y], dtype=float)
                    radial_norm = max(float(np.linalg.norm(radial)), 1e-9)
                    label_x, label_y = (radial + radial / radial_norm * 0.26)
                ax.text(
                    label_x,
                    label_y,
                    f"{labels[i]}\nπ={_format_probability(pi[i])}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 0.8, 7.2),
                    color=style.spine_color,
                    zorder=6,
                )
        ax.set_aspect("equal")
        if n == 2:
            ax.set_xlim(-1.30, 1.30)
            ax.set_ylim(-0.96, 0.90)
        else:
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        ax.set_title("MSM state-exchange network", pad=8, weight="semibold")
        save_figure(fig, out_base, style)
    return {
        "edge_metric": "equilibrium_transition_flux_pi_i_P_ij",
        "edge_labels": "transition_probability_P_ij",
        "requested_min_flux": requested_min_flux,
        "used_min_flux": used_min_flux,
        "threshold_relaxed": threshold_relaxed,
        "edge_count": len(edges),
        "max_offdiagonal_flux": max_offdiag_flux,
    }


def plot_state_population_heatmap(replica_names, cluster_labels, matrix, out_base, style: PlotStyleConfig):
    matrix_heatmap(
        matrix,
        replica_names,
        [f"State {c}" for c in cluster_labels],
        out_base,
        style,
        title="State population by replica",
        xlabel="State",
        ylabel="Replica",
        vmin=0.0,
        vmax=max(1.0, float(np.nanmax(matrix))),
        cbar_label="Population fraction",
    )


def _plot_transition_probability_matrix(T, labels, out_base, style: PlotStyleConfig):
    n_states = T.shape[0]
    size = max(4.8, min(7.0, 0.62 * n_states + 3.0))
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(size + 0.45, size))
        mesh = ax.imshow(T, cmap=resolve_colormap(style.cmap_continuous, style), vmin=0.0, vmax=1.0)
        if n_states <= 14:
            for iy in range(n_states):
                for ix in range(n_states):
                    value = float(T[iy, ix])
                    text_color = "white" if value >= 0.55 else style.spine_color
                    ax.text(
                        ix,
                        iy,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=max(style.tick_size - 0.4, 6.8),
                        color=text_color,
                    )
        ax.set_xticks(np.arange(n_states))
        ax.set_xticklabels(labels, rotation=0.0 if n_states <= 4 else 25.0, ha="right" if n_states > 4 else "center")
        ax.set_yticks(np.arange(n_states))
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(-0.5, n_states, 1.0), minor=True)
        ax.set_yticks(np.arange(-0.5, n_states, 1.0), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.75, alpha=0.55)
        ax.tick_params(which="minor", bottom=False, left=False)
        finalize_axes(ax, style, xlabel="To state", ylabel="From state", title="MSM transition-probability matrix")
        cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.set_title("Probability", fontsize=style.legend_size, pad=6)
        save_figure(fig, out_base, style)


def _plot_dominant_exchange(edge_rows, labels, out_base, style: PlotStyleConfig, flux_matrix=None):
    out_base = Path(out_base)
    with publication_style(style):
        if edge_rows:
            height = max(3.8, 0.42 * len(edge_rows) + 1.8)
            fig, ax = plt.subplots(figsize=(7.4, height))
            bar_labels = [f"{labels[i]} -> {labels[j]}" for i, j, _, _ in edge_rows]
            metric_values = np.asarray([row[3] for row in edge_rows], dtype=float)
            y = np.arange(len(edge_rows))
            edge_colors = [_state_colors(len(labels), style)[row[0]] for row in edge_rows]
            ax.barh(y, metric_values, color=edge_colors, alpha=0.88, edgecolor="white", linewidth=0.8)
            x_max = float(np.nanmax(metric_values)) if metric_values.size else 1.0
            pad = max(x_max * 0.04, 1e-6)
            for ypos, (_, _, transition_prob, metric_value) in zip(y, edge_rows):
                ax.text(
                    metric_value + pad,
                    ypos,
                    f"P={transition_prob:.3f}",
                    va="center",
                    ha="left",
                    fontsize=max(style.tick_size - 0.4, 6.9),
                    color=style.spine_color,
                )
            ax.set_yticks(y)
            ax.set_yticklabels(bar_labels)
            ax.invert_yaxis()
            ax.set_xlim(0.0, x_max * 1.28 + pad)
            xlabel = "Equilibrium flux" if flux_matrix is not None else "Transition probability"
            finalize_axes(ax, style, xlabel=xlabel, title="Dominant directed state exchange")
            ax.grid(True, axis="x", linestyle="-", linewidth=0.7)
            ax.grid(False, axis="y")
        else:
            fig, ax = plt.subplots(figsize=(6.4, 3.6))
            ax.axis("off")
            ax.text(
                0.02,
                0.50,
                "No non-self transitions above the numerical cutoff.",
                transform=ax.transAxes,
                ha="left",
                va="center",
                color=style.spine_color,
                fontsize=style.label_size,
            )
        save_figure(fig, out_base, style)


def _plot_transition_residence_departure(T, labels, out_base, style: PlotStyleConfig):
    stay_probs = np.clip(np.diag(T), 0.0, 1.0)
    leave_probs = np.clip(1.0 - stay_probs, 0.0, 1.0)
    y = np.arange(T.shape[0])
    height = max(3.8, 0.46 * T.shape[0] + 1.8)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.0, height))
        ax.barh(y, stay_probs, color=style.band_color, alpha=0.95, edgecolor="white", linewidth=0.8, label="Stay")
        ax.barh(y, leave_probs, left=stay_probs, color=style.accent_color, alpha=0.88, edgecolor="white", linewidth=0.8, label="Leave")
        for ypos, stay_prob, leave_prob in zip(y, stay_probs, leave_probs):
            if stay_prob >= 0.12:
                ax.text(
                    stay_prob * 0.50,
                    ypos,
                    f"stay {stay_prob:.1%}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 0.6, 6.7),
                    color=style.spine_color,
                )
            if leave_prob >= 0.12:
                ax.text(
                    stay_prob + leave_prob * 0.50,
                    ypos,
                    f"leave {leave_prob:.1%}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 0.6, 6.7),
                    color=style.spine_color,
                )
            else:
                ax.annotate(
                    f"leave {leave_prob:.1%}",
                    xy=(1.0, ypos),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=max(style.tick_size - 0.6, 6.7),
                    color=style.spine_color,
                    clip_on=False,
                )
        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlim(0.0, 1.0)
        ax.xaxis.set_major_formatter(PercentFormatter(1.0))
        finalize_axes(ax, style, xlabel="Probability mass", title="Residence versus departure")
        save_figure(fig, out_base, style)


def plot_transition_matrix_heatmap(transition_matrix, out_base, style: PlotStyleConfig, state_labels=None, flux_matrix=None):
    T = np.asarray(transition_matrix, dtype=float)
    if T.ndim != 2 or T.shape[0] != T.shape[1]:
        raise ValueError("MSM transition matrix must be square for heatmap plotting.")
    labels = _coerce_state_labels(state_labels, T.shape[0])
    flux = None if flux_matrix is None else _as_real_array(flux_matrix)
    edge_rows = _top_offdiagonal_edges(T, flux_matrix=flux, limit=min(8, max(T.shape[0] * 2, 4)))
    out_base = Path(out_base)
    _plot_transition_probability_matrix(T, labels, out_base, style)
    _plot_dominant_exchange(edge_rows, labels, out_base.with_name("transition_dominant_exchange"), style, flux_matrix=flux)
    _plot_transition_residence_departure(T, labels, out_base.with_name("transition_residence_departure"), style)


def plot_chapman_kolmogorov_test(ck_test, out_base, style: PlotStyleConfig, state_labels=None, summary_lines=None):
    predictions = _as_real_array(ck_test.predictions)
    estimates = _as_real_array(ck_test.estimates)
    lagtimes = np.asarray(ck_test.lagtimes, dtype=float)
    if predictions.ndim != 3 or estimates.ndim != 3:
        raise ValueError("CK test predictions/estimates must be 3D arrays.")
    n_components = predictions.shape[1]
    labels = _coerce_state_labels(state_labels, n_components)

    with publication_style(style):
        fig, axes = plt.subplots(
            n_components,
            n_components,
            figsize=(max(6.0, 2.8 * n_components), max(5.2, 2.45 * n_components)),
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        for i in range(n_components):
            for j in range(n_components):
                ax = axes[i, j]
                color = style.categorical_palette[(i + j) % len(style.categorical_palette)]
                ax.plot(lagtimes, predictions[:, i, j], linewidth=style.line_width, color=color, label="Model prediction")
                ax.scatter(lagtimes, estimates[:, i, j], s=max(style.marker_size * 18.0, 30.0), color=style.mean_line_color, edgecolors="white", linewidths=0.8, zorder=3, label="Independent estimate")
                ax.set_ylim(-0.02, 1.02)
                finalize_axes(ax, style)
                if i == 0:
                    ax.set_title(f"to {labels[j]}", pad=6)
                if j == 0:
                    ax.set_ylabel(f"from {labels[i]}\nProbability")
                if i == n_components - 1:
                    ax.set_xlabel("Lag time (frames)")
                if i == 0 and j == 0:
                    ax.legend(frameon=False, loc="lower left")
        fig.suptitle("Chapman-Kolmogorov validation", y=0.995, weight="semibold")
        if summary_lines:
            fig.text(
                0.99,
                0.01,
                " | ".join(str(line) for line in summary_lines if str(line).strip()),
                ha="right",
                va="bottom",
                fontsize=max(style.tick_size - 0.8, 6.6),
                color=style.spine_color,
            )
        save_figure(fig, out_base, style)


def plot_snapshot_grid(snapshot_entries, out_base, style: PlotStyleConfig, title: str):
    snapshot_grid(snapshot_entries, out_base, style, title)
