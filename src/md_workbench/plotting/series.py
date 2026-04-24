from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import t as student_t

from ..config import PlotStyleConfig
from .theme import finalize_axes, publication_style, save_figure


def _legend_kwargs(n_items: int) -> dict:
    if n_items <= 4:
        return {"loc": "best", "ncol": 1}
    if n_items <= 8:
        return {"loc": "center left", "bbox_to_anchor": (1.01, 0.5), "borderaxespad": 0.0, "ncol": 1}
    return {"loc": "upper center", "bbox_to_anchor": (0.5, 1.02), "borderaxespad": 0.0, "ncol": 2}


def _blend_color(color: str, target: str, fraction: float) -> tuple[float, float, float]:
    base = np.asarray(to_rgb(color), dtype=float)
    end = np.asarray(to_rgb(target), dtype=float)
    return tuple((1.0 - fraction) * base + fraction * end)


def _line_shadow(linewidth: float) -> list[pe.AbstractPathEffect]:
    return [
        pe.Stroke(linewidth=linewidth + 1.1, foreground=(1.0, 1.0, 1.0, 0.95)),
        pe.Normal(),
    ]


def _rolling_window_size(n_points: int, window_fraction: float) -> int:
    if n_points <= 2 or window_fraction <= 0:
        return 1
    window = max(5, int(round(n_points * float(window_fraction))))
    if window % 2 == 0:
        window += 1
    if window >= n_points:
        window = n_points if n_points % 2 == 1 else n_points - 1
    return max(window, 1)


def smooth_series(values, window_fraction: float = 0.10) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr.copy()
    window = _rolling_window_size(arr.size, window_fraction)
    if window <= 1:
        return arr.copy()
    finite = np.isfinite(arr)
    numer = np.convolve(np.where(finite, arr, 0.0), np.ones(window, dtype=float), mode="same")
    denom = np.convolve(finite.astype(float), np.ones(window, dtype=float), mode="same")
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = denom > 0
    out[valid] = numer[valid] / denom[valid]
    return out


def _last_finite_point(xs, ys) -> tuple[float, float]:
    x_arr = np.asarray(xs, dtype=float)
    y_arr = np.asarray(ys, dtype=float)
    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.any(finite):
        return float(x_arr[-1]), float("nan")
    idx = int(np.flatnonzero(finite)[-1])
    return float(x_arr[idx]), float(y_arr[idx])


def _spread_positions(values, lower: float, upper: float, min_gap: float) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return arr
    available = max(upper - lower, 1e-12)
    min_gap = min(float(min_gap), available / max(arr.size - 1, 1))
    order = np.argsort(arr)
    sorted_vals = arr[order].copy()
    sorted_vals[0] = max(sorted_vals[0], lower)
    for idx in range(1, sorted_vals.size):
        sorted_vals[idx] = max(sorted_vals[idx], sorted_vals[idx - 1] + min_gap)
    overflow = sorted_vals[-1] - upper
    if overflow > 0:
        sorted_vals -= overflow
    for idx in range(sorted_vals.size - 2, -1, -1):
        sorted_vals[idx] = min(sorted_vals[idx], sorted_vals[idx + 1] - min_gap)
    underflow = lower - sorted_vals[0]
    if underflow > 0:
        sorted_vals += underflow
    out = np.empty_like(sorted_vals)
    out[order] = sorted_vals
    return out


def _mean_and_confidence_interval(arr2d, confidence: float = 0.95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(arr2d, dtype=float)
    if arr.ndim != 2:
        raise ValueError("Expected a 2D replicate stack")
    valid = np.isfinite(arr)
    counts = valid.sum(axis=0)
    mean = np.full(arr.shape[1], np.nan, dtype=float)
    has_data = counts > 0
    if np.any(has_data):
        mean[has_data] = np.where(valid, arr, 0.0).sum(axis=0)[has_data] / counts[has_data]

    sd = np.zeros(arr.shape[1], dtype=float)
    enough = counts > 1
    if np.any(enough):
        centered = np.where(valid, arr - mean, 0.0)
        variance = np.where(
            enough,
            np.sum(centered ** 2, axis=0) / np.maximum(counts - 1, 1),
            0.0,
        )
        sd[enough] = np.sqrt(variance[enough])

    half_width = np.zeros(arr.shape[1], dtype=float)
    if np.any(enough):
        crit = np.ones(arr.shape[1], dtype=float)
        for n_reps in np.unique(counts[enough]):
            crit[counts == n_reps] = float(student_t.ppf(0.5 + confidence / 2.0, int(n_reps) - 1))
        half_width[enough] = crit[enough] * sd[enough] / np.sqrt(counts[enough])

    lower = mean - half_width
    upper = mean + half_width
    if np.nanmin(arr) >= -1e-12:
        lower = np.maximum(lower, 0.0)
    return mean, lower, upper


def _replicate_handles(
    replicate_colors: list[str],
    replicate_labels: list[str],
    mean_color: str,
    band_color: str,
    confidence: float,
    mean_label: str = "Mean",
) -> list:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=1.5,
            label=label,
        )
        for color, label in zip(replicate_colors, replicate_labels)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=mean_color,
            linewidth=2.4,
            label=mean_label,
        )
    )
    handles.append(
        Patch(
            facecolor=band_color,
            edgecolor="none",
            alpha=0.28,
            label=f"{int(round(confidence * 100.0))}% CI",
        )
    )
    return handles


def draw_publication_replicate_summary(
    ax,
    time_ns,
    arr2d,
    style: PlotStyleConfig,
    *,
    replicate_labels: list[str] | None = None,
    rolling_window_fraction: float = 0.10,
    confidence: float = 0.95,
    replicate_colors: list[str] | None = None,
    mean_color: str | None = None,
    band_color: str | None = None,
    show_endpoint: bool = True,
):
    arr2d = np.asarray(arr2d, dtype=float)
    time_arr = np.asarray(time_ns, dtype=float)
    n_replicas = arr2d.shape[0]
    replicate_labels = replicate_labels or [f"Replica {idx + 1}" for idx in range(n_replicas)]
    palette = replicate_colors or style.categorical_palette
    line_colors = [palette[idx % len(palette)] for idx in range(n_replicas)]
    mean_line_color = mean_color or style.mean_line_color
    interval_color = band_color or _blend_color(style.band_color, "#FFFFFF", 0.35)
    window = _rolling_window_size(time_arr.size, rolling_window_fraction)

    if window > 1:
        for color, row in zip(line_colors, arr2d):
            ax.plot(
                time_arr,
                row,
                linewidth=max(style.thin_line_width * 0.65, 0.55),
                alpha=0.18,
                color=color,
                zorder=1,
            )
    display_rows = np.vstack([smooth_series(row, rolling_window_fraction) for row in arr2d])
    mean_s, lower_s, upper_s = _mean_and_confidence_interval(display_rows, confidence=confidence)

    ax.fill_between(
        time_arr,
        lower_s,
        upper_s,
        alpha=0.18,
        linewidth=0.0,
        color=interval_color,
        zorder=1.5,
    )
    for color, label, row_s in zip(line_colors, replicate_labels, display_rows):
        line = ax.plot(
            time_arr,
            row_s,
            linewidth=max(style.thin_line_width + 0.2, 1.05),
            alpha=0.95,
            color=color,
            label=label,
            zorder=2,
        )[0]
        line.set_path_effects(_line_shadow(line.get_linewidth()))

    mean_line = ax.plot(
        time_arr,
        mean_s,
        linewidth=style.line_width + 0.65,
        color=mean_line_color,
        zorder=3,
    )[0]
    mean_line.set_path_effects(_line_shadow(mean_line.get_linewidth()))
    if show_endpoint and time_arr.size > 0:
        x_end, y_end = _last_finite_point(time_arr, mean_s)
        if np.isfinite(y_end):
            ax.scatter(
                [x_end],
                [y_end],
                s=max(style.marker_size * 6.0, 18.0),
                color=mean_line_color,
                edgecolors="white",
                linewidths=0.8,
                zorder=4,
            )
    ax.margins(x=0.01)
    return {
        "mean": mean_s,
        "lower": lower_s,
        "upper": upper_s,
        "replicate_colors": line_colors,
        "replicate_labels": list(replicate_labels),
        "mean_color": mean_line_color,
        "band_color": interval_color,
        "legend_handles": _replicate_handles(
            line_colors,
            list(replicate_labels),
            mean_line_color,
            interval_color,
            confidence,
            mean_label="Mean (rolling)" if window > 1 else "Mean",
        ),
    }


def draw_summary_band(
    ax,
    time_ns,
    mean,
    sd,
    style: PlotStyleConfig,
    color: str,
    *,
    rolling_window_fraction: float = 0.10,
    band_color: str | None = None,
    show_endpoint: bool = True,
):
    time_arr = np.asarray(time_ns, dtype=float)
    mean_arr = np.asarray(mean, dtype=float)
    sd_arr = np.asarray(sd, dtype=float)
    mean_s = smooth_series(mean_arr, rolling_window_fraction)
    lower_s = smooth_series(mean_arr - sd_arr, rolling_window_fraction)
    upper_s = smooth_series(mean_arr + sd_arr, rolling_window_fraction)
    if np.nanmin(mean_arr) >= -1e-12 and np.nanmin(sd_arr) >= -1e-12:
        lower_s = np.maximum(lower_s, 0.0)
    ax.fill_between(
        time_arr,
        lower_s,
        upper_s,
        alpha=0.32,
        linewidth=0.0,
        color=band_color or _blend_color(color, "#FFFFFF", 0.68),
        zorder=2,
    )
    ax.plot(time_arr, mean_s, linewidth=style.line_width + 0.35, color=color, zorder=3)
    if show_endpoint and time_arr.size > 0:
        x_end, y_end = _last_finite_point(time_arr, mean_s)
        if np.isfinite(y_end):
            ax.scatter(
                [x_end],
                [y_end],
                s=max(style.marker_size * 6.5, 18.0),
                color=color,
                edgecolors="white",
                linewidths=0.7,
                zorder=4,
            )
    return mean_s, lower_s, upper_s


def draw_replicate_summary(
    ax,
    time_ns,
    arr2d,
    style: PlotStyleConfig,
    *,
    color: str,
    rolling_window_fraction: float = 0.10,
    raw_color: str | None = None,
    raw_alpha: float = 0.24,
):
    arr2d = np.asarray(arr2d, dtype=float)
    time_arr = np.asarray(time_ns, dtype=float)
    neutral = raw_color or _blend_color(style.spine_color, "#FFFFFF", 0.78)
    for row in arr2d:
        ax.plot(
            time_arr,
            row,
            linewidth=max(style.thin_line_width * 0.9, 0.65),
            alpha=raw_alpha,
            color=neutral,
            zorder=1,
        )
    mean = arr2d.mean(axis=0)
    sd = arr2d.std(axis=0, ddof=1) if arr2d.shape[0] > 1 else np.zeros_like(mean)
    return draw_summary_band(
        ax,
        time_arr,
        mean,
        sd,
        style,
        color,
        rolling_window_fraction=rolling_window_fraction,
    )


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


def summary_band_series(
    time_ns,
    mean,
    sd,
    ylabel,
    out_base,
    style: PlotStyleConfig,
    *,
    title=None,
    xlabel="Time (ns)",
    color: str | None = None,
    rolling_window_fraction: float = 0.10,
    figsize: tuple[float, float] = (7.2, 4.6),
):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=figsize)
        draw_summary_band(
            ax,
            time_ns,
            mean,
            sd,
            style,
            color or style.mean_line_color,
            rolling_window_fraction=rolling_window_fraction,
        )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)


def publication_replicate_series(
    time_ns,
    arr2d,
    ylabel,
    out_base,
    style: PlotStyleConfig,
    *,
    title=None,
    xlabel="Time (ns)",
    replicate_labels: list[str] | None = None,
    rolling_window_fraction: float = 0.10,
    figsize: tuple[float, float] = (7.2, 4.6),
    confidence: float = 0.95,
    legend_ncol: int = 5,
):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=figsize)
        payload = draw_publication_replicate_summary(
            ax,
            time_ns,
            arr2d,
            style,
            replicate_labels=replicate_labels,
            rolling_window_fraction=rolling_window_fraction,
            confidence=confidence,
        )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(
            handles=payload["legend_handles"],
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=legend_ncol,
            borderaxespad=0.0,
        )
        save_figure(fig, out_base, style)


def replica_trend_series(
    time_ns,
    arr2d,
    ylabel,
    out_base,
    style: PlotStyleConfig,
    *,
    title=None,
    xlabel="Time (ns)",
    color: str | None = None,
    rolling_window_fraction: float = 0.10,
    figsize: tuple[float, float] = (7.2, 4.6),
):
    with publication_style(style):
        fig, ax = plt.subplots(figsize=figsize)
        draw_replicate_summary(
            ax,
            time_ns,
            arr2d,
            style,
            color=color or style.mean_line_color,
            rolling_window_fraction=rolling_window_fraction,
        )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
        save_figure(fig, out_base, style)


def direct_label_line_series(
    time_ns,
    ys,
    labels,
    ylabel,
    out_base,
    style: PlotStyleConfig,
    *,
    title=None,
    xlabel="Time (ns)",
    colors=None,
    rolling_window_fraction: float = 0.10,
    figsize: tuple[float, float] = (7.2, 4.6),
):
    time_arr = np.asarray(time_ns, dtype=float)
    window = _rolling_window_size(time_arr.size, rolling_window_fraction)
    smoothed = [smooth_series(y, rolling_window_fraction) for y in ys]
    palette = colors or style.categorical_palette
    with publication_style(style):
        fig, ax = plt.subplots(figsize=figsize)
        end_values = []
        x_end_values = []
        for idx, (raw_y, y, label) in enumerate(zip(ys, smoothed, labels)):
            color = palette[idx % len(palette)]
            if window > 1:
                ax.plot(
                    time_arr,
                    raw_y,
                    linewidth=max(style.thin_line_width * 0.65, 0.55),
                    alpha=0.16,
                    color=_blend_color(color, "#FFFFFF", 0.35),
                    zorder=1,
                )
            ax.plot(time_arr, y, linewidth=style.line_width, color=color, zorder=2)
            x_end, y_end = _last_finite_point(time_arr, y)
            x_end_values.append(x_end)
            end_values.append(y_end)
            if np.isfinite(y_end):
                ax.scatter(
                    [x_end],
                    [y_end],
                    s=max(style.marker_size * 5.5, 16.0),
                    color=color,
                    edgecolors="white",
                    linewidths=0.6,
                    zorder=3,
                )
        finite_ends = np.asarray([value for value in end_values if np.isfinite(value)], dtype=float)
        if finite_ends.size:
            lower = float(np.nanmin(finite_ends))
            upper = float(np.nanmax(finite_ends))
            label_gap = max((upper - lower) * 0.06, 0.02)
            label_positions = _spread_positions(end_values, lower, upper, label_gap)
        else:
            label_positions = np.asarray(end_values, dtype=float)
        x_span = float(time_arr[-1] - time_arr[0]) if time_arr.size > 1 else 1.0
        x_pad = max(x_span * 0.08, 0.8)
        if time_arr.size:
            ax.set_xlim(float(time_arr[0]), float(time_arr[-1]) + x_pad)
        for idx, (label, y_end, y_label, x_end) in enumerate(zip(labels, end_values, label_positions, x_end_values)):
            if not np.isfinite(y_end):
                continue
            color = palette[idx % len(palette)]
            leader_x = x_end + x_pad * 0.18
            text_x = x_end + x_pad * 0.30
            ax.plot([x_end, leader_x], [y_end, y_label], linewidth=0.8, color=_blend_color(color, "#FFFFFF", 0.38), zorder=1)
            ax.text(
                text_x,
                y_label,
                label,
                color=color,
                fontsize=max(style.legend_size + 0.15, 7.0),
                va="center",
                ha="left",
            )
        finalize_axes(ax, style, xlabel=xlabel, ylabel=ylabel, title=title)
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
