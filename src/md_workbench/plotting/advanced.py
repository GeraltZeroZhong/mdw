from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import binary_dilation, gaussian_filter, minimum_filter
from scipy.stats import gaussian_kde

from ..config import PlotStyleConfig
from ..core import save_csv, write_json
from .bars import vertical_bars
from .heatmaps import matrix_heatmap
from .snapshots import snapshot_grid
from .theme import finalize_axes, publication_style, save_figure


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
        mesh = ax.plot_surface(
            surface.x,
            surface.y,
            z_surface,
            cmap=style.cmap_continuous,
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
            cmap=style.cmap_continuous,
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


def plot_line_profile(x, y, out_base, xlabel, ylabel, title, style: PlotStyleConfig, color=None):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        ax.plot(x, y, linewidth=style.line_width, marker="o", markersize=style.marker_size, color=color or style.accent_color)
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
        mesh = ax.contourf(surface.x, surface.y, energy_plot, levels=levels, cmap=style.cmap_continuous, extend="max")
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


def plot_stationary_distribution(states, probs, out_base, style: PlotStyleConfig):
    vertical_bars(states, probs, out_base, style, title="MSM stationary distribution", xlabel="MSM state", ylabel="Stationary probability")


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


def plot_lag_scan(lag_rows, out_base, style: PlotStyleConfig):
    arr = np.asarray(lag_rows, dtype=float)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.5, 4.7))
        for proc in sorted(set(arr[:, 1].astype(int))):
            sub = arr[arr[:, 1] == proc]
            color = style.categorical_palette[(proc - 1) % len(style.categorical_palette)]
            ax.plot(sub[:, 0], sub[:, 2], marker="o", markersize=style.marker_size, linewidth=style.line_width, label=f"Process {proc}", color=color)
        x0, x1 = float(arr[:, 0].min()), float(arr[:, 0].max())
        ax.plot([x0, x1], [x0, x1], linestyle="--", linewidth=1.0, color=style.mean_line_color, alpha=0.7)
        finalize_axes(ax, style, xlabel="MSM lag time (frames)", ylabel="Implied timescale (frames)", title="Implied timescales vs lag")
        ax.legend(frameon=False, ncol=1)
        save_figure(fig, out_base, style)


def plot_state_network(transition_matrix, stationary_probs, out_base, style: PlotStyleConfig, threshold: float = 1e-6):
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

    cmap, norm = _categorical_cmap(n, style)
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(6.3, 6.0))
        for i, j, transition_prob, edge_flux in edges:
            x0, y0 = xy[i]
            x1, y1 = xy[j]
            relative_flux = (edge_flux / max_edge_flux) ** 0.55 if max_edge_flux > 0 else 1.0
            has_reverse = (j, i) in edge_pairs
            if n == 2:
                curve = 0.24
            else:
                curve = 0.18 if has_reverse else 0.10
            patch = FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={curve}",
                mutation_scale=11.0 + 5.0 * relative_flux,
                linewidth=0.75 + 3.2 * relative_flux,
                color=style.spine_color,
                alpha=0.34 + 0.50 * relative_flux,
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
                ax.text(
                    label_x,
                    label_y,
                    f"P={_format_probability(transition_prob)}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 1.3, 7.0),
                    color=style.spine_color,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.78),
                    zorder=4,
                )

        sizes = 620.0 + 2300.0 * np.sqrt(pi / pi.max()) if np.max(pi) > 0 else np.full(n, 800.0)
        ax.scatter(xy[:, 0], xy[:, 1], s=sizes, c=np.arange(n), cmap=cmap, norm=norm, edgecolors="white", linewidths=1.2, zorder=5)
        for i, (x, y) in enumerate(xy):
            ax.text(x, y, str(i), ha="center", va="center", color="white", fontsize=style.label_size, weight="bold", zorder=6)
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
                    f"pi={_format_probability(pi[i])}",
                    ha="center",
                    va="center",
                    fontsize=max(style.tick_size - 0.8, 7.2),
                    color=style.spine_color,
                    zorder=6,
                )
        ax.set_aspect("equal")
        if n == 2:
            ax.set_xlim(-1.25, 1.25)
            ax.set_ylim(-0.82, 0.86)
        else:
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.25, 1.25)
        ax.axis("off")
        ax.set_title("MSM state network", pad=8, weight="semibold")
        legend_lines = [
            "Node area: stationary probability",
            "Edge width: pi_i P_ij flux",
            "Edge label: transition probability P_ij",
            f"Min flux: {_format_probability(used_min_flux)}" + (" (adaptive)" if threshold_relaxed else ""),
        ]
        if not edges:
            legend_lines.append("No non-self transitions above cutoff")
        ax.text(
            0.02,
            0.98,
            "\n".join(legend_lines),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=max(style.tick_size - 1.4, 6.8),
            color=style.spine_color,
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", edgecolor=style.grid_color, linewidth=0.6, alpha=0.92),
            zorder=7,
        )
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


def plot_transition_matrix_heatmap(transition_matrix, out_base, style: PlotStyleConfig):
    T = np.asarray(transition_matrix, dtype=float)
    labels = [f"State {i}" for i in range(T.shape[0])]
    matrix_heatmap(
        T,
        labels,
        labels,
        out_base,
        style,
        title="MSM transition matrix",
        xlabel="To state",
        ylabel="From state",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Transition probability",
        annotate=T.size <= 144,
        annotation_format="{:.3f}",
        x_rotation=0.0 if T.shape[0] <= 6 else 30.0,
    )


def plot_snapshot_grid(snapshot_entries, out_base, style: PlotStyleConfig, title: str):
    snapshot_grid(snapshot_entries, out_base, style, title)
