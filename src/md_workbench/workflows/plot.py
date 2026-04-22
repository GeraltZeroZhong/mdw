from __future__ import annotations

from collections import defaultdict
from copy import deepcopy

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from ..config import WorkflowConfig
from ..core import ensure_project_layout, normalize_workflow_paths, organize_outputs, read_dict_csv
from ..core.progress import ProgressCallback, emit_progress
from ..plotting.residue_labels import compact_residue_label
from ..plotting.series import (
    direct_label_line_series,
    draw_replicate_summary,
    draw_summary_band,
    replica_trend_series,
    summary_band_series,
)
from ..plotting.theme import finalize_axes, publication_style, save_figure
from ..postprocess.mmgbsa import run_mmgbsa_postprocess, summarize_mmgbsa_postprocess_result


def _plot_progress_total_units(
    cfg: WorkflowConfig,
    *,
    include_mmgbsa_postprocess: bool,
    include_organize_outputs: bool,
) -> int:
    total = 0
    if cfg.do_basic_analysis:
        total += 1
    if cfg.do_waterbridge_analysis:
        total += 1
    if include_mmgbsa_postprocess and cfg.do_mmgbsa_postprocess:
        total += 1
    if include_organize_outputs:
        total += 1
    return max(total, 1)


def _numeric_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _replica_stack(rows: list[dict[str, str]], value_suffix: str) -> np.ndarray:
    keys = [key for key in rows[0].keys() if key.startswith("replica_") and key.endswith(value_suffix)]
    if not keys:
        raise KeyError(f"Missing replica columns with suffix '{value_suffix}'")
    return np.vstack([_numeric_column(rows, key) for key in keys])


def _replot_replicate_metric(
    csv_path: Path,
    out_base: Path,
    *,
    value_suffix: str,
    ylabel: str,
    title: str,
    color: str,
    rolling_window_fraction: float,
    cfg: WorkflowConfig,
) -> bool:
    if not csv_path.exists():
        return False
    rows = read_dict_csv(csv_path)
    if not rows:
        return False
    time_ns = _numeric_column(rows, "time_ns")
    stack = _replica_stack(rows, value_suffix)
    replica_trend_series(
        time_ns,
        stack,
        ylabel,
        out_base,
        cfg.plot_style,
        title=title,
        color=color,
        rolling_window_fraction=rolling_window_fraction,
    )
    return True


def _replot_combined_rmsd(basic_root: Path, cfg: WorkflowConfig) -> bool:
    csv_path = basic_root / "rmsd_combined.csv"
    if not csv_path.exists():
        return False
    rows = read_dict_csv(csv_path)
    if not rows:
        return False

    time_ns = _numeric_column(rows, "time_ns")
    protein_stack = _replica_stack(rows, "protein_backbone_rmsd_A")
    ligand_stack = _replica_stack(rows, "ligand_heavy_rmsd_A")
    with publication_style(cfg.plot_style):
        fig, axes = plt.subplots(2, 1, figsize=(7.6, 5.8), sharex=True)
        draw_replicate_summary(
            axes[0],
            time_ns,
            protein_stack,
            cfg.plot_style,
            color=cfg.plot_style.protein_color,
            rolling_window_fraction=cfg.basic.rolling_window_fraction,
        )
        draw_replicate_summary(
            axes[1],
            time_ns,
            ligand_stack,
            cfg.plot_style,
            color=cfg.plot_style.ligand_color,
            rolling_window_fraction=cfg.basic.rolling_window_fraction,
        )
        finalize_axes(axes[0], cfg.plot_style, ylabel="Protein backbone RMSD (Å)", title="Protein backbone")
        finalize_axes(axes[1], cfg.plot_style, xlabel="Time (ns)", ylabel="Ligand heavy-atom RMSD (Å)", title="Ligand heavy-atom")
        fig.suptitle("RMSD across replicas", y=1.02, weight="semibold", fontsize=cfg.plot_style.title_size + 1.0)
        fig.text(
            0.015,
            0.99,
            "thin gray = replica trajectories, thick color = rolling mean, band = replica spread",
            fontsize=max(cfg.plot_style.legend_size - 0.2, 6.7),
            color=cfg.plot_style.spine_color,
            alpha=0.78,
            va="top",
        )
        save_figure(fig, basic_root / "rmsd_combined", cfg.plot_style)

    protein_mean = _numeric_column(rows, "protein_backbone_rmsd_mean_A")
    protein_sd = _numeric_column(rows, "protein_backbone_rmsd_sd_A")
    ligand_mean = _numeric_column(rows, "ligand_heavy_rmsd_mean_A")
    ligand_sd = _numeric_column(rows, "ligand_heavy_rmsd_sd_A")
    summary_band_series(
        time_ns,
        protein_mean,
        protein_sd,
        "Protein backbone RMSD (Å)",
        basic_root / "rmsd_replot_protein",
        cfg.plot_style,
        title="Protein backbone RMSD replot",
        color=cfg.plot_style.protein_color,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    summary_band_series(
        time_ns,
        ligand_mean,
        ligand_sd,
        "Ligand heavy-atom RMSD (Å)",
        basic_root / "rmsd_replot_ligand",
        cfg.plot_style,
        title="Ligand heavy-atom RMSD replot",
        color=cfg.plot_style.ligand_color,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    return True


def _replot_key_contact_traces(basic_root: Path, cfg: WorkflowConfig) -> bool:
    csv_path = basic_root / "key_contact_distance_traces.csv"
    if not csv_path.exists():
        return False
    rows = read_dict_csv(csv_path)
    if not rows:
        return False

    occupancy_map: dict[str, float] = {}
    occupancy_csv = basic_root / "contact_occupancy_distance_summary.csv"
    if occupancy_csv.exists():
        for row in read_dict_csv(occupancy_csv):
            occupancy_map[str(row["protein_residue"])] = float(row["contact_occupancy_mean"])

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protein_residue"])].append(row)
    if not grouped:
        return False

    labels = sorted(
        grouped.keys(),
        key=lambda label: (-occupancy_map.get(label, 0.0), compact_residue_label(label)),
    )
    panel_payloads = []
    for idx, label in enumerate(labels):
        residue_rows = grouped[label]
        panel_payloads.append(
            {
                "display_label": compact_residue_label(label),
                "time_ns": _numeric_column(residue_rows, "time_ns"),
                "mean": _numeric_column(residue_rows, "mean_distance_A"),
                "sd": _numeric_column(residue_rows, "sd_distance_A"),
                "color": cfg.plot_style.categorical_palette[idx % len(cfg.plot_style.categorical_palette)],
                "occupancy": occupancy_map.get(label, np.nan),
            }
        )

    with publication_style(cfg.plot_style):
        n_panels = len(panel_payloads)
        fig, axes = plt.subplots(n_panels, 1, figsize=(7.6, 1.42 * n_panels + 1.4), sharex=True, sharey=True)
        axes_arr = np.atleast_1d(axes)
        upper = np.concatenate([payload["mean"] + payload["sd"] for payload in panel_payloads])
        finite_upper = upper[np.isfinite(upper)]
        axis_cap = None
        clipped = False
        if finite_upper.size:
            focus_limit = float(np.nanpercentile(finite_upper, 97.5))
            cutoff_focus = cfg.basic.contact_cutoff_nm * 10.0 * 2.4
            axis_cap = max(6.0, cutoff_focus, focus_limit)
            axis_cap = float(np.ceil(axis_cap * 2.0) / 2.0)
            clipped = float(np.nanmax(finite_upper)) > axis_cap + 1e-9
        for idx, (ax, payload) in enumerate(zip(axes_arr, panel_payloads)):
            ax.axhline(
                cfg.basic.contact_cutoff_nm * 10.0,
                color=cfg.plot_style.spine_color,
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                alpha=0.48,
                zorder=0,
            )
            draw_summary_band(
                ax,
                payload["time_ns"],
                payload["mean"],
                payload["sd"],
                cfg.plot_style,
                color=payload["color"],
                rolling_window_fraction=cfg.basic.rolling_window_fraction,
            )
            finalize_axes(ax, cfg.plot_style, xlabel="Time (ns)" if idx == n_panels - 1 else None)
            if idx != n_panels - 1:
                ax.tick_params(labelbottom=False)
            ax.set_ylabel("")
            panel_label = payload["display_label"]
            if np.isfinite(payload["occupancy"]):
                panel_label = f"{panel_label}   {payload['occupancy']:.0%}"
            ax.text(
                0.01,
                0.86,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=payload["color"],
                fontsize=max(cfg.plot_style.legend_size + 0.3, 7.0),
                weight="bold",
            )
        if axis_cap is not None:
            axes_arr[0].set_ylim(0.0, axis_cap)
        fig.suptitle("Key contact distance trajectories", y=1.02, weight="semibold", fontsize=cfg.plot_style.title_size + 1.0)
        fig.supylabel("Minimum heavy-atom distance (Å)")
        helper = "each hotspot gets its own panel; colored line = rolling mean; band = across-replica spread"
        if clipped and axis_cap is not None:
            helper += f"; spikes clipped above {axis_cap:.1f} Å"
        fig.text(
            0.015,
            0.99,
            helper,
            fontsize=max(cfg.plot_style.legend_size - 0.25, 6.6),
            color=cfg.plot_style.spine_color,
            alpha=0.78,
            va="top",
        )
        save_figure(fig, basic_root / "key_contact_distance_traces", cfg.plot_style)
    return True


def _run_basic_replot(cfg: WorkflowConfig) -> str | None:
    basic_root = Path(cfg.basic.analysis_root) / "combined"
    csv_path = basic_root / "rmsd_combined.csv"
    if not csv_path.exists():
        return None
    rolling_window_fraction = cfg.basic.rolling_window_fraction
    _replot_combined_rmsd(basic_root, cfg)
    _replot_replicate_metric(
        basic_root / "min_distance_combined.csv",
        basic_root / "min_distance_combined",
        value_suffix="min_distance_A",
        ylabel="Minimum ligand-protein distance (Å)",
        title="Ligand-protein minimum heavy-atom distance",
        color=cfg.plot_style.distance_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "contact_count_combined.csv",
        basic_root / "contact_count_combined",
        value_suffix="contact_count",
        ylabel="Count",
        title="Contact count",
        color=cfg.plot_style.protein_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "hbond_count_combined.csv",
        basic_root / "hbond_count_combined",
        value_suffix="hbond_count",
        ylabel="Count",
        title="H-bond count",
        color=cfg.plot_style.distance_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "salt_bridge_count_combined.csv",
        basic_root / "salt_bridge_count_combined",
        value_suffix="salt_bridge_count",
        ylabel="Count",
        title="Salt-bridge count",
        color=cfg.plot_style.accent_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_key_contact_traces(basic_root, cfg)
    _replot_replicate_metric(
        basic_root / "radius_of_gyration_combined.csv",
        basic_root / "radius_of_gyration_combined",
        value_suffix="rg_A",
        ylabel="Radius of gyration (Å)",
        title="Radius of gyration (Å)",
        color=cfg.plot_style.protein_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "buried_surface_combined.csv",
        basic_root / "buried_surface_combined",
        value_suffix="buried_surface_A2",
        ylabel="Area (Å²)",
        title="Buried surface area (Å²)",
        color=cfg.plot_style.accent_color,
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    sasa_csv = basic_root / "sasa_components_combined.csv"
    if sasa_csv.exists():
        rows = read_dict_csv(sasa_csv)
        if rows:
            time_ns = _numeric_column(rows, "time_ns")
            direct_label_line_series(
                time_ns,
                [
                    _numeric_column(rows, "complex_sasa_mean_A2"),
                    _numeric_column(rows, "protein_sasa_mean_A2"),
                    _numeric_column(rows, "ligand_sasa_mean_A2"),
                ],
                ["Complex SASA", "Protein SASA", "Ligand SASA"],
                "Area (Å²)",
                basic_root / "sasa_components_combined",
                cfg.plot_style,
                title="SASA components across replicas",
                colors=[cfg.plot_style.protein_color, cfg.plot_style.accent_color, cfg.plot_style.ligand_color],
                rolling_window_fraction=rolling_window_fraction,
            )
    return str(basic_root.resolve())


def _run_waterbridge_replot(cfg: WorkflowConfig) -> str | None:
    water_root = Path(cfg.waterbridge.analysis_root) / "combined"
    csv_path = water_root / "waterbridge_count_combined.csv"
    if not csv_path.exists():
        return None
    rows = read_dict_csv(csv_path)
    time_ns = np.asarray([float(r["time_ns"]) for r in rows], dtype=float)
    mean_key = "mean_n_bridging_waters" if "mean_n_bridging_waters" in rows[0] else "mean_count"
    sd_key = "sd_n_bridging_waters" if "sd_n_bridging_waters" in rows[0] else "sd_count"
    mean_count = np.asarray([float(r[mean_key]) for r in rows], dtype=float)
    sd_count = np.asarray([float(r[sd_key]) for r in rows], dtype=float)
    count_stack = _replica_stack(rows, "count")
    replica_trend_series(
        time_ns,
        count_stack,
        "Number of bridging waters",
        water_root / "waterbridge_count_combined",
        cfg.plot_style,
        title="Strict water-bridge count",
        color=cfg.plot_style.distance_color,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    summary_band_series(
        time_ns,
        mean_count,
        sd_count,
        "Number of bridging waters",
        water_root / "waterbridge_count_replot",
        cfg.plot_style,
        title="Strict water-bridge count replot",
        color=cfg.plot_style.distance_color,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    return str(water_root.resolve())


def run_plot_postprocess(
    cfg: WorkflowConfig,
    progress_callback: ProgressCallback | None = None,
    *,
    include_mmgbsa_postprocess: bool = True,
    include_organize_outputs: bool = True,
):
    total_units = _plot_progress_total_units(
        cfg,
        include_mmgbsa_postprocess=include_mmgbsa_postprocess,
        include_organize_outputs=include_organize_outputs,
    )
    completed_units = 0
    outputs = {}
    if cfg.do_basic_analysis:
        if cfg.plot_selection.enabled("plot_workflow_basic_replot"):
            emit_progress(progress_callback, completed_units, total_units, "basic_replot", "Replotting combined basic-analysis figures")
            basic_replot_dir = _run_basic_replot(cfg)
            if basic_replot_dir is not None:
                outputs["basic_replot_dir"] = basic_replot_dir
                detail = "Completed combined basic-analysis replot"
            else:
                detail = "Skipping combined basic-analysis replot because rmsd_combined.csv was not found"
        else:
            detail = "Skipping combined basic-analysis replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "basic_replot", detail)
    if cfg.do_waterbridge_analysis:
        if cfg.plot_selection.enabled("plot_workflow_waterbridge_replot"):
            emit_progress(progress_callback, completed_units, total_units, "waterbridge_replot", "Replotting combined water-bridge figures")
            waterbridge_replot_dir = _run_waterbridge_replot(cfg)
            if waterbridge_replot_dir is not None:
                outputs["waterbridge_replot_dir"] = waterbridge_replot_dir
                detail = "Completed combined water-bridge replot"
            else:
                detail = "Skipping combined water-bridge replot because waterbridge_count_combined.csv was not found"
        else:
            detail = "Skipping combined water-bridge replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "waterbridge_replot", detail)
    if include_mmgbsa_postprocess and cfg.do_mmgbsa_postprocess:
        emit_progress(progress_callback, completed_units, total_units, "mmgbsa_postprocess", "Running MM/GBSA postprocess")
        outputs["mmgbsa_postprocess"] = run_mmgbsa_postprocess(cfg.mmgbsa, cfg.plot_style, cfg.plot_selection)
        completed_units += 1
        emit_progress(
            progress_callback,
            completed_units,
            total_units,
            "mmgbsa_postprocess",
            summarize_mmgbsa_postprocess_result(outputs["mmgbsa_postprocess"]),
        )

    if include_organize_outputs:
        emit_progress(progress_callback, completed_units, total_units, "organize_outputs", "Organizing workflow outputs")
        outputs["organized_outputs"] = organize_outputs(
            cfg.output_bundle,
            cfg.run,
            cfg.basic,
            cfg.waterbridge,
            cfg.advanced,
            cfg.mmgbsa,
            cfg.do_basic_analysis,
            cfg.do_waterbridge_analysis,
            cfg.do_advanced_analysis,
            cfg.do_mmgbsa_postprocess,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "organize_outputs", "Workflow outputs organized")
    return outputs


def run_plot_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = normalize_workflow_paths(deepcopy(cfg))
    ensure_project_layout(cfg.workspace_root)
    return run_plot_postprocess(cfg, progress_callback=progress_callback)
