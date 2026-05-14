from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import json
import re
from types import SimpleNamespace

from pathlib import Path
import numpy as np

from ..config import WorkflowConfig
from ..config.plot_style_defaults import apply_plot_style_palette
from ..core import (
    ensure_project_layout,
    find_ligand_residues,
    ligand_heavy_atom_indices_from_residues,
    normalize_workflow_paths,
    organize_outputs,
    read_dict_csv,
)
from ..core.progress import ProgressCallback, emit_progress
from ..plotting.basic_combined import _key_contact_axis_cap, _key_contact_out_base, plot_key_contact_distance_figure
from ..plotting.bars import ranked_distance_lollipop, ranked_lollipop
from ..plotting.advanced import (
    plot_chapman_kolmogorov_test,
    plot_cluster_population,
    plot_fes_from_csv,
    plot_lag_scan,
    plot_line_profile,
    plot_snapshot_grid,
    plot_state_network,
    plot_state_population_heatmap,
    plot_stationary_distribution,
    plot_transition_matrix_heatmap,
    scatter_by_replica,
    scatter_clusters,
)
from ..plotting.heatmaps import interaction_fingerprint_heatmap, matrix_heatmap, simple_boxplot, stacked_fraction_area
from ..plotting.residue_labels import compact_replica_name, compact_residue_label
from ..plotting.series import (
    direct_label_line_series,
    publication_replicate_series,
    shaded_profile,
)
from ..plotting.theme import remove_figure_outputs
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
    if cfg.do_advanced_analysis:
        total += 1
    if include_mmgbsa_postprocess and cfg.do_mmgbsa_postprocess:
        total += 1
    if include_organize_outputs:
        total += 1
    return max(total, 1)


def _csv_has_rows(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return bool(read_dict_csv(path))
    except Exception:
        return False


def reusable_basic_csv_available(cfg: WorkflowConfig) -> bool:
    basic_root = Path(cfg.basic.analysis_root) / "combined"
    return _csv_has_rows(basic_root / "rmsd_combined.csv")


def reusable_waterbridge_csv_available(cfg: WorkflowConfig) -> bool:
    water_root = Path(cfg.waterbridge.analysis_root) / "combined"
    return _csv_has_rows(water_root / "waterbridge_count_combined.csv")


def reusable_advanced_csv_available(cfg: WorkflowConfig) -> bool:
    advanced_root = Path(cfg.advanced.analysis_root)
    assign_root = advanced_root / "per_replica_assignments"
    checks = []
    if cfg.plot_selection.enabled("advanced_pca"):
        checks.append(_csv_has_rows(advanced_root / "pca" / "explained_variance_ratio.csv"))
        checks.append(_csv_has_rows(advanced_root / "pca" / "free_energy_landscape_pc1_pc2.csv"))
        checks.append(any(_csv_has_rows(path) for path in assign_root.glob("replica_*_pc.csv")))
    if cfg.plot_selection.enabled("advanced_tica"):
        checks.append(_csv_has_rows(advanced_root / "tica" / "singular_values.csv"))
        checks.append(_csv_has_rows(advanced_root / "tica" / "free_energy_landscape_tic1_tic2.csv"))
        checks.append(any(_csv_has_rows(path) for path in assign_root.glob("replica_*_tic.csv")))
    if cfg.plot_selection.enabled("advanced_clustering"):
        checks.append(_csv_has_rows(advanced_root / "clustering" / "cluster_population_overall.csv"))
        checks.append(_csv_has_rows(advanced_root / "clustering" / "cluster_population_per_replica.csv"))
        checks.append(_csv_has_rows(advanced_root / "clustering" / "cluster_centers.csv"))
    if cfg.plot_selection.enabled("advanced_snapshots"):
        checks.append(_csv_has_rows(advanced_root / "snapshots" / "representative_frames.csv"))
    if cfg.plot_selection.enabled("advanced_msm"):
        checks.append(_csv_has_rows(advanced_root / "msm" / "stationary_distribution.csv"))
        checks.append(_csv_has_rows(advanced_root / "msm" / "transition_matrix.csv"))
        checks.append(_csv_has_rows(advanced_root / "msm" / "implied_timescales_lag_scan.csv"))
    return any(checks)


def reusable_analysis_csv_available(cfg: WorkflowConfig, section: str) -> bool:
    section = str(section).strip().lower()
    if section == "basic":
        return reusable_basic_csv_available(cfg)
    if section == "waterbridge":
        return reusable_waterbridge_csv_available(cfg)
    if section == "advanced":
        return reusable_advanced_csv_available(cfg)
    raise ValueError(f"Unknown reusable analysis section: {section}")


def run_basic_replot_from_csv(cfg: WorkflowConfig) -> str | None:
    if not reusable_basic_csv_available(cfg):
        return None
    return _run_basic_replot(cfg)


def run_waterbridge_replot_from_csv(cfg: WorkflowConfig) -> str | None:
    if not reusable_waterbridge_csv_available(cfg):
        return None
    return _run_waterbridge_replot(cfg)


def run_advanced_replot_from_csv(cfg: WorkflowConfig) -> str | None:
    if not reusable_advanced_csv_available(cfg):
        return None
    return _run_advanced_replot(cfg)


def _numeric_column(rows: list[dict[str, str]], key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in rows], dtype=float)


def _replica_stack(rows: list[dict[str, str]], value_suffix: str) -> np.ndarray:
    keys = [key for key in rows[0].keys() if key.startswith("replica_") and key.endswith(value_suffix)]
    if not keys:
        raise KeyError(f"Missing replica columns with suffix '{value_suffix}'")
    keys.sort(
        key=lambda key: (
            int(re.match(r"replica_(\d+)_", key).group(1)) if re.match(r"replica_(\d+)_", key) else 10**9,
            key,
        )
    )
    return np.vstack([_numeric_column(rows, key) for key in keys])


def _replica_labels(rows: list[dict[str, str]], value_suffix: str) -> list[str]:
    keys = [key for key in rows[0].keys() if key.startswith("replica_") and key.endswith(value_suffix)]
    keys.sort(
        key=lambda key: (
            int(re.match(r"replica_(\d+)_", key).group(1)) if re.match(r"replica_(\d+)_", key) else 10**9,
            key,
        )
    )
    suffix_token = f"_{value_suffix}"
    return [compact_replica_name(key[: -len(suffix_token)]) for key in keys]


def _metric_max_abs_from_csv(csv_path: Path, value_suffix: str) -> float | None:
    if not csv_path.exists():
        return None
    rows = read_dict_csv(csv_path)
    if not rows:
        return None
    try:
        stack = _replica_stack(rows, value_suffix)
    except Exception:
        return None
    finite = np.asarray(stack, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.nanmax(np.abs(finite)))


def _replot_replicate_metric(
    csv_path: Path,
    out_base: Path,
    *,
    value_suffix: str,
    ylabel: str,
    title: str,
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
    publication_replicate_series(
        time_ns,
        stack,
        ylabel,
        out_base,
        cfg.plot_style,
        title=title,
        replicate_labels=_replica_labels(rows, value_suffix),
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
    protein_labels = _replica_labels(rows, "protein_backbone_rmsd_A")
    ligand_labels = _replica_labels(rows, "ligand_heavy_rmsd_A")
    remove_figure_outputs(basic_root / "rmsd_combined")

    publication_replicate_series(
        time_ns,
        protein_stack,
        "Protein backbone RMSD (Å)",
        basic_root / "rmsd_replot_protein",
        cfg.plot_style,
        title="Protein backbone RMSD across replicas",
        replicate_labels=protein_labels,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    publication_replicate_series(
        time_ns,
        ligand_stack,
        "Ligand heavy-atom RMSD (Å)",
        basic_root / "rmsd_replot_ligand",
        cfg.plot_style,
        title="Ligand heavy-atom RMSD across replicas",
        replicate_labels=ligand_labels,
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
                "label": label,
                "display_label": compact_residue_label(label),
                "time_ns": _numeric_column(residue_rows, "time_ns"),
                "mean": _numeric_column(residue_rows, "mean_distance_A"),
                "sd": _numeric_column(residue_rows, "sd_distance_A"),
                "color": cfg.plot_style.categorical_palette[idx % len(cfg.plot_style.categorical_palette)],
                "occupancy": occupancy_map.get(label, np.nan),
            }
        )

    remove_figure_outputs(basic_root / "key_contact_distance_traces")
    for payload in panel_payloads:
        axis_cap, clipped = _key_contact_axis_cap(payload["mean"], payload["sd"], cfg.basic.contact_cutoff_nm * 10.0)
        plot_key_contact_distance_figure(
            payload["time_ns"],
            payload["mean"],
            payload["sd"],
            _key_contact_out_base(basic_root, payload["label"]),
            cfg.plot_style,
            display_label=payload["display_label"],
            color=payload["color"],
            occupancy=payload["occupancy"],
            contact_cutoff_A=cfg.basic.contact_cutoff_nm * 10.0,
            rolling_window_fraction=cfg.basic.rolling_window_fraction,
            axis_cap=axis_cap,
            clipped=clipped,
        )
    return True


def _display_metric_name(metric: str) -> str:
    return {
        "protein_rmsd_A": "Protein RMSD",
        "ligand_rmsd_A": "Ligand RMSD",
        "min_distance_A": "Min distance",
        "radius_of_gyration_A": "Radius of gyration",
        "buried_surface_A2": "Buried surface",
        "hbond_count": "H-bond count",
    }.get(metric, metric.replace("_", " "))


def _replot_basic_rmsf(basic_root: Path, cfg: WorkflowConfig) -> None:
    csv_path = basic_root / "rmsf_ca_combined.csv"
    if not csv_path.exists():
        return
    rows = read_dict_csv(csv_path)
    if not rows:
        return
    shaded_profile(
        _numeric_column(rows, "resSeq"),
        _numeric_column(rows, "rmsf_mean_A"),
        _numeric_column(rows, "rmsf_sd_A"),
        "Cα RMSF (Å)",
        basic_root / "rmsf_ca_combined",
        cfg.plot_style,
        title="Protein Cα RMSF across replicas",
    )


def _replot_basic_occupancy(basic_root: Path, cfg: WorkflowConfig) -> None:
    top_n = int(cfg.basic.top_n_contacts_plot)
    distance_path = basic_root / "contact_occupancy_distance_summary.csv"
    if distance_path.exists():
        rows = read_dict_csv(distance_path)[:top_n]
        if rows:
            ranked_distance_lollipop(
                [compact_residue_label(row["protein_residue"]) for row in rows],
                [float(row["contact_occupancy_mean"]) for row in rows],
                [float(row["contact_occupancy_sd"]) for row in rows],
                [float(row["min_distance_mean_A"]) for row in rows],
                basic_root / "contact_occupancy_top20",
                cfg.plot_style,
                title="Persistent contact hotspots\nCircle area = mean minimum distance (closer = larger)",
                xlabel="Contact occupancy",
                color=cfg.plot_style.protein_color,
            )
    for filename, mean_key, sd_key, out_name, title, xlabel, color in [
        (
            "hbond_residue_occupancy_combined.csv",
            "hbond_occupancy_mean",
            "hbond_occupancy_sd",
            "hbond_residue_occupancy_top20",
            "Persistent H-bond hotspots",
            "H-bond occupancy",
            cfg.plot_style.distance_color,
        ),
        (
            "salt_bridge_residue_occupancy_combined.csv",
            "salt_bridge_occupancy_mean",
            "salt_bridge_occupancy_sd",
            "salt_bridge_residue_occupancy",
            "Persistent salt-bridge hotspots",
            "Salt-bridge occupancy",
            cfg.plot_style.accent_color,
        ),
    ]:
        path = basic_root / filename
        if not path.exists():
            continue
        rows = read_dict_csv(path)[:top_n]
        if not rows:
            continue
        ranked_lollipop(
            [compact_residue_label(row["protein_residue"]) for row in rows],
            [float(row[mean_key]) for row in rows],
            [float(row[sd_key]) for row in rows],
            basic_root / out_name,
            cfg.plot_style,
            title=title,
            xlabel=xlabel,
            color=color,
        )


def _replot_basic_dssp(basic_root: Path, cfg: WorkflowConfig) -> None:
    occupancy_path = basic_root / "dssp_residue_occupancy_combined.csv"
    if occupancy_path.exists():
        rows = read_dict_csv(occupancy_path)
        if rows:
            matrix_heatmap(
                np.asarray(
                    [
                        [
                            float(row["helix_fraction_mean"]),
                            float(row["sheet_fraction_mean"]),
                            float(row["coil_fraction_mean"]),
                        ]
                        for row in rows
                    ],
                    dtype=float,
                ),
                [compact_residue_label(row["protein_residue"]) for row in rows],
                ["Helix", "Sheet", "Coil"],
                basic_root / "dssp_residue_occupancy_combined",
                cfg.plot_style,
                title="Combined residue secondary-structure occupancy",
                xlabel="Secondary-structure state",
                ylabel="Residue",
                vmin=0.0,
                vmax=1.0,
                cbar_label="Occupancy fraction",
            )
    replica_paths = sorted((basic_root.parent).glob("replica_*/dssp_fractions_timeseries.csv"), key=lambda path: _replica_sort_key(path.parent.name))
    if not replica_paths:
        return
    replica_rows = [read_dict_csv(path) for path in replica_paths]
    replica_rows = [rows for rows in replica_rows if rows]
    if not replica_rows:
        return
    min_n = min(len(rows) for rows in replica_rows)
    time_ns = _numeric_column(replica_rows[0][:min_n], "time_ns")
    mean_fracs = {}
    for label, key in [("Helix", "helix_fraction"), ("Sheet", "sheet_fraction"), ("Coil", "coil_fraction")]:
        stack = np.vstack([_numeric_column(rows[:min_n], key) for rows in replica_rows])
        mean_fracs[label] = stack.mean(axis=0)
    stacked_fraction_area(time_ns, mean_fracs, basic_root / "dssp_fractions_combined", cfg.plot_style, title="Combined secondary-structure fractions")


def _replot_basic_interaction_heatmaps(basic_root: Path, cfg: WorkflowConfig) -> None:
    contact_path = basic_root / "contact_occupancy_combined.csv"
    if not contact_path.exists():
        return
    contact_rows = read_dict_csv(contact_path)
    if not contact_rows:
        return
    top_n = int(cfg.basic.top_n_contacts_plot)
    top_residues = [row["protein_residue"] for row in contact_rows[:top_n]]
    replica_dirs = sorted((basic_root.parent).glob("replica_*"), key=lambda path: _replica_sort_key(path.name))
    per_rep_matrix = []
    for residue in top_residues:
        values = []
        for replica_dir in replica_dirs:
            rows = read_dict_csv(replica_dir / "contact_occupancy.csv") if (replica_dir / "contact_occupancy.csv").exists() else []
            match = next((float(row["contact_occupancy"]) for row in rows if row["protein_residue"] == residue), 0.0)
            values.append(match)
        per_rep_matrix.append(values)
    if per_rep_matrix and replica_dirs:
        matrix_heatmap(
            np.asarray(per_rep_matrix, dtype=float),
            [compact_residue_label(label) for label in top_residues],
            [compact_replica_name(path.name) for path in replica_dirs],
            basic_root / "contact_replicate_heatmap",
            cfg.plot_style,
            title="Contact hotspot reproducibility",
            xlabel="Replica",
            ylabel="Residue",
            vmin=0.0,
            vmax=1.0,
            cmap=cfg.plot_style.cmap_continuous,
            annotate=True,
            annotation_format="{:.0%}",
            annotation_min_abs=0.30,
            cbar_label="Contact occupancy",
            x_rotation=0.0,
        )

    metric_specs = [
        ("Contact", "contact_occupancy_combined.csv", "contact_occupancy_mean", cfg.plot_style.protein_color),
        ("H-bond", "hbond_residue_occupancy_combined.csv", "hbond_occupancy_mean", cfg.plot_style.distance_color),
        ("Salt bridge", "salt_bridge_residue_occupancy_combined.csv", "salt_bridge_occupancy_mean", cfg.plot_style.accent_color),
    ]
    residue_summary: dict[str, dict[str, float]] = {}
    for metric_name, filename, mean_key, _color in metric_specs:
        path = basic_root / filename
        if not path.exists():
            continue
        for row in read_dict_csv(path)[:top_n]:
            residue_summary.setdefault(str(row["protein_residue"]), {})[metric_name] = float(row[mean_key])
    residues = sorted(
        residue_summary,
        key=lambda label: (
            -sum(value > 0.08 for value in residue_summary[label].values()),
            -sum(residue_summary[label].values()),
            compact_residue_label(label),
        ),
    )[: max(top_n, 12)]
    if residues:
        interaction_fingerprint_heatmap(
            np.asarray([[residue_summary[residue].get(metric_name, 0.0) for metric_name, *_rest in metric_specs] for residue in residues], dtype=float),
            [compact_residue_label(label) for label in residues],
            [metric_name for metric_name, *_rest in metric_specs],
            basic_root / "interaction_fingerprint_heatmap",
            cfg.plot_style,
            title="Interaction fingerprint by hotspot residue",
            xlabel="Interaction class",
            ylabel="Residue",
            interaction_colors=[color for *_prefix, color in metric_specs],
            annotation_min=0.22,
        )


def _replot_basic_convergence(basic_root: Path, cfg: WorkflowConfig) -> None:
    zscore_path = basic_root / "convergence_block_zscores_combined.csv"
    if zscore_path.exists():
        rows = read_dict_csv(zscore_path)
        metrics = []
        blocks = []
        for row in rows:
            metric = row["metric"]
            block = int(row["block"])
            if metric not in metrics:
                metrics.append(metric)
            if block not in blocks:
                blocks.append(block)
        if metrics and blocks:
            lookup = {(row["metric"], int(row["block"])): float(row["zscore"]) for row in rows}
            matrix_heatmap(
                np.asarray([[lookup.get((metric, block), np.nan) for block in blocks] for metric in metrics], dtype=float),
                [_display_metric_name(metric) for metric in metrics],
                [f"Block {block}" for block in blocks],
                basic_root / "convergence_block_heatmap",
                cfg.plot_style,
                title="Convergence block deviations across replicas",
                xlabel="Trajectory block",
                ylabel="Metric",
                annotate=True,
                center=0.0,
                cbar_label="Deviation from metric mean (SD)",
            )
    consistency_path = basic_root / "replicate_consistency_zscores.csv"
    if not consistency_path.exists():
        return
    rows = read_dict_csv(consistency_path)
    if not rows:
        return
    replicas = []
    metrics = []
    for row in rows:
        if row["replica"] not in replicas:
            replicas.append(row["replica"])
        if row["metric"] not in metrics:
            metrics.append(row["metric"])
    box_data = {metric: [float(row["zscore"]) for row in rows if row["metric"] == metric] for metric in metrics}
    simple_boxplot(
        box_data,
        basic_root / "replicate_consistency_boxplot",
        cfg.plot_style,
        title="Replica consistency summary (metric-standardized)",
        ylabel="Deviation from across-replica mean (SD)",
        show_points=True,
        hline_zero=True,
    )
    lookup = {(row["replica"], row["metric"]): float(row["zscore"]) for row in rows}
    matrix_heatmap(
        np.asarray([[lookup.get((replica, metric), np.nan) for metric in metrics] for replica in replicas], dtype=float),
        replicas,
        metrics,
        basic_root / "replicate_consistency_zscore_heatmap",
        cfg.plot_style,
        title="Replica consistency by metric",
        xlabel="Metric",
        ylabel="Replica",
        annotate=True,
        center=0.0,
        cbar_label="Deviation from across-replica mean (SD)",
    )


def _run_basic_replot(cfg: WorkflowConfig) -> str | None:
    basic_root = Path(cfg.basic.analysis_root) / "combined"
    if not reusable_basic_csv_available(cfg):
        return None
    rolling_window_fraction = cfg.basic.rolling_window_fraction
    _replot_combined_rmsd(basic_root, cfg)
    _replot_basic_rmsf(basic_root, cfg)
    _replot_basic_occupancy(basic_root, cfg)
    _replot_basic_dssp(basic_root, cfg)
    _replot_basic_interaction_heatmaps(basic_root, cfg)
    _replot_basic_convergence(basic_root, cfg)
    _replot_replicate_metric(
        basic_root / "min_distance_combined.csv",
        basic_root / "min_distance_combined",
        value_suffix="min_distance_A",
        ylabel="Minimum ligand-protein distance (Å)",
        title="Ligand-protein minimum heavy-atom distance",
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "contact_count_combined.csv",
        basic_root / "contact_count_combined",
        value_suffix="contact_count",
        ylabel="Count",
        title="Contact count",
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "hbond_count_combined.csv",
        basic_root / "hbond_count_combined",
        value_suffix="hbond_count",
        ylabel="Count",
        title="H-bond count",
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "salt_bridge_count_combined.csv",
        basic_root / "salt_bridge_count_combined",
        value_suffix="salt_bridge_count",
        ylabel="Count",
        title="Salt-bridge count",
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
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    buried_csv = basic_root / "buried_surface_combined.csv"
    if (_metric_max_abs_from_csv(buried_csv, "buried_surface_A2") or 0.0) >= 1.0:
        _replot_replicate_metric(
            buried_csv,
            basic_root / "buried_surface_combined",
            value_suffix="buried_surface_A2",
            ylabel="Area (Å²)",
            title="Buried surface area (Å²)",
            rolling_window_fraction=rolling_window_fraction,
            cfg=cfg,
        )
    else:
        remove_figure_outputs(basic_root / "buried_surface_combined")
    _replot_replicate_metric(
        basic_root / "ligand_com_distance_combined.csv",
        basic_root / "ligand_com_distance_combined",
        value_suffix="com_distance_A",
        ylabel="Distance (Å)",
        title="Ligand-pocket COM distance (Å)",
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    _replot_replicate_metric(
        basic_root / "ligand_orientation_angle_combined.csv",
        basic_root / "ligand_orientation_angle_combined",
        value_suffix="orientation_angle_deg",
        ylabel="Angle (deg)",
        title="Ligand orientation angle (deg)",
        rolling_window_fraction=rolling_window_fraction,
        cfg=cfg,
    )
    sasa_csv = basic_root / "sasa_components_combined.csv"
    if sasa_csv.exists():
        rows = read_dict_csv(sasa_csv)
        if rows:
            time_ns = _numeric_column(rows, "time_ns")
            remove_figure_outputs(basic_root / "sasa_components_combined")
            direct_label_line_series(
                time_ns,
                [
                    _numeric_column(rows, "complex_sasa_mean_A2"),
                    _numeric_column(rows, "protein_sasa_mean_A2"),
                ],
                ["Complex SASA", "Protein SASA"],
                "Area (Å²)",
                basic_root / "sasa_complex_protein_combined",
                cfg.plot_style,
                title="Protein-complex SASA across replicas",
                colors=[cfg.plot_style.protein_color, cfg.plot_style.accent_color],
                rolling_window_fraction=rolling_window_fraction,
            )
            direct_label_line_series(
                time_ns,
                [_numeric_column(rows, "ligand_sasa_mean_A2")],
                ["Ligand SASA"],
                "Area (Å²)",
                basic_root / "ligand_sasa_combined",
                cfg.plot_style,
                title="Ligand SASA across replicas",
                colors=[cfg.plot_style.ligand_color],
                rolling_window_fraction=rolling_window_fraction,
            )
    return str(basic_root.resolve())


def _run_waterbridge_replot(cfg: WorkflowConfig) -> str | None:
    water_root = Path(cfg.waterbridge.analysis_root) / "combined"
    csv_path = water_root / "waterbridge_count_combined.csv"
    if not reusable_waterbridge_csv_available(cfg):
        return None
    rows = read_dict_csv(csv_path)
    if not rows:
        return None
    time_ns = np.asarray([float(r["time_ns"]) for r in rows], dtype=float)
    count_stack = _replica_stack(rows, "count")
    count_labels = _replica_labels(rows, "count")
    publication_replicate_series(
        time_ns,
        count_stack,
        "Number of bridging waters",
        water_root / "waterbridge_count_combined",
        cfg.plot_style,
        title="Strict water-bridge count",
        replicate_labels=count_labels,
        rolling_window_fraction=cfg.basic.rolling_window_fraction,
    )
    remove_figure_outputs(water_root / "waterbridge_count_replot")
    occupancy_path = water_root / "waterbridge_residue_occupancy_combined.csv"
    if occupancy_path.exists():
        occupancy_rows = read_dict_csv(occupancy_path)[:20]
        if occupancy_rows:
            ranked_lollipop(
                [compact_residue_label(row["protein_residue"]) for row in occupancy_rows],
                [float(row["waterbridge_occupancy_mean"]) for row in occupancy_rows],
                [float(row["waterbridge_occupancy_sd"]) for row in occupancy_rows],
                water_root / "waterbridge_residue_occupancy_top20",
                cfg.plot_style,
                title="Persistent water-bridge hotspots",
                xlabel="Water-bridge occupancy",
                color=cfg.plot_style.potential_energy_color,
            )
    return str(water_root.resolve())


def _replica_sort_key(name: str) -> tuple[int, str]:
    match = re.match(r"replica_(\d+)", str(name))
    return (int(match.group(1)) if match else 10**9, str(name))


def _projection_stack(assign_root: Path, suffix: str, x_key: str, y_key: str):
    paths = sorted(assign_root.glob(f"replica_*_{suffix}.csv"), key=lambda path: _replica_sort_key(path.stem))
    names = []
    arrays = []
    for path in paths:
        rows = read_dict_csv(path)
        if not rows:
            continue
        name = path.stem[: -len(f"_{suffix}")]
        arrays.append(np.column_stack([_numeric_column(rows, x_key), _numeric_column(rows, y_key)]))
        names.append(name)
    if not arrays:
        return [], [], np.empty((0, 2), dtype=float)
    return names, arrays, np.vstack(arrays)


def _read_square_matrix(path: Path):
    if not path.exists():
        return [], np.empty((0, 0), dtype=float)
    rows = read_dict_csv(path)
    if not rows:
        return [], np.empty((0, 0), dtype=float)
    labels = [key for key in rows[0].keys() if key != "from\\to"]
    matrix = np.asarray([[float(row[label]) for label in labels] for row in rows], dtype=float)
    return labels, matrix


def _msm_state_labels(msm_root: Path) -> list[str]:
    mapping_path = msm_root / "active_state_mapping.csv"
    if not mapping_path.exists():
        return []
    rows = sorted(read_dict_csv(mapping_path), key=lambda row: int(row["msm_state"]))
    return [f"S{int(row['msm_state'])} [C{int(row['source_cluster'])}]" for row in rows]


def _load_json_if_exists(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def _ck_test_from_csv(path: Path):
    if not path.exists():
        return None
    rows = read_dict_csv(path)
    if not rows:
        return None
    lagtimes = sorted({int(float(row["lag_frames"])) for row in rows})
    state_count = max(
        max(int(float(row["from_state"])), int(float(row["to_state"])))
        for row in rows
    ) + 1
    predictions = np.full((len(lagtimes), state_count, state_count), np.nan, dtype=float)
    estimates = np.full_like(predictions, np.nan)
    lag_index = {lag: idx for idx, lag in enumerate(lagtimes)}
    for row in rows:
        lag_idx = lag_index[int(float(row["lag_frames"]))]
        from_idx = int(float(row["from_state"]))
        to_idx = int(float(row["to_state"]))
        predictions[lag_idx, from_idx, to_idx] = float(row["predicted_probability"])
        estimates[lag_idx, from_idx, to_idx] = float(row["estimated_probability"])
    return SimpleNamespace(predictions=predictions, estimates=estimates, lagtimes=np.asarray(lagtimes, dtype=float))


def _snapshot_entries_from_existing_pdbs(snapshot_root: Path, cfg: WorkflowConfig, n_show: int) -> list[dict]:
    frame_csv = snapshot_root / "representative_frames.csv"
    if not frame_csv.exists():
        return []
    rows = sorted(read_dict_csv(frame_csv), key=lambda row: int(row["rank_by_population"]))
    rows = rows[: max(1, min(int(n_show), len(rows)))]
    if not rows:
        return []

    import mdtraj as md
    from ..analysis.advanced.workflow import (
        _continuous_ca_segments,
        _infer_ligand_bonds,
        _protein_contact_mask,
        _protein_secondary_structure,
    )

    entries = []
    cumulative_fraction = sum(float(row["fraction"]) for row in rows)
    for row in rows:
        pdb_path = Path(str(row["pdb_path"]))
        if not pdb_path.exists():
            pdb_path = snapshot_root / pdb_path.name
        if not pdb_path.exists():
            continue
        frame = md.load_pdb(str(pdb_path))
        protein_ca = frame.topology.select("protein and name CA")
        ligand_residues = find_ligand_residues(frame.topology)
        ligand_heavy = ligand_heavy_atom_indices_from_residues(ligand_residues)
        cluster_id = int(float(row["cluster"]))
        entries.append(
            {
                "title": f"State {cluster_id}\n{row['replica']} frame {int(float(row['frame']))}",
                "cluster_id": cluster_id,
                "population_rank": int(float(row["rank_by_population"])),
                "n_frames": int(float(row["n_frames"])),
                "population_fraction": float(row["fraction"]),
                "replica_name": row["replica"],
                "frame_idx": int(float(row["frame"])),
                "distance_to_center": float(row["distance_to_center"]),
                "n_clusters_total": len(read_dict_csv(frame_csv)),
                "n_clusters_shown": len(rows),
                "cumulative_population_fraction": cumulative_fraction,
                "protein_segments": _continuous_ca_segments(frame, protein_ca),
                "protein_secondary_structure": _protein_secondary_structure(frame, protein_ca),
                "protein_contact_mask": _protein_contact_mask(frame, protein_ca, ligand_heavy, cfg.advanced.pocket_ca_cutoff_nm),
                "ligand_bonds": _infer_ligand_bonds(frame, ligand_heavy),
                "protein_xyz": frame.xyz[0, protein_ca, :],
                "ligand_xyz": frame.xyz[0, ligand_heavy, :],
            }
        )
    return entries


def _run_advanced_replot(cfg: WorkflowConfig) -> str | None:
    advanced_root = Path(cfg.advanced.analysis_root)
    if not advanced_root.exists():
        return None
    assign_root = advanced_root / "per_replica_assignments"
    did_plot = False

    if cfg.plot_selection.enabled("advanced_pca"):
        evr_path = advanced_root / "pca" / "explained_variance_ratio.csv"
        if evr_path.exists():
            rows = read_dict_csv(evr_path)
            if rows:
                plot_line_profile(
                    _numeric_column(rows, "component"),
                    _numeric_column(rows, "explained_variance_ratio"),
                    advanced_root / "pca" / "explained_variance_ratio",
                    "PC index",
                    "Explained variance ratio",
                    "PCA explained variance",
                    cfg.plot_style,
                    color=cfg.plot_style.accent_color,
                )
                did_plot = True
        names, arrays, xy = _projection_stack(assign_root, "pc", "PC1", "PC2")
        if xy.size:
            if plot_fes_from_csv(
                advanced_root / "pca" / "free_energy_landscape_pc1_pc2.csv",
                advanced_root / "pca" / "free_energy_landscape_pc1_pc2",
                "PCA free-energy landscape",
                "PC1",
                "PC2",
                cfg.plot_style,
            ):
                did_plot = True
            scatter_by_replica(
                xy,
                [SimpleNamespace(name=name) for name in names],
                arrays,
                advanced_root / "pca" / "pc1_pc2_scatter",
                "PC1",
                "PC2",
                "PCA projection by replica",
                cfg.plot_style,
            )
            did_plot = True

    if cfg.plot_selection.enabled("advanced_tica"):
        singular_path = advanced_root / "tica" / "singular_values.csv"
        if singular_path.exists():
            rows = read_dict_csv(singular_path)
            if rows:
                rows = rows[:40]
                plot_line_profile(
                    _numeric_column(rows, "component"),
                    _numeric_column(rows, "singular_value"),
                    advanced_root / "tica" / "singular_values",
                    "tIC index",
                    "Singular value",
                    "Leading tICA singular values",
                    cfg.plot_style,
                    color=cfg.plot_style.accent_color,
                )
                did_plot = True
        names, arrays, xy = _projection_stack(assign_root, "tic", "TIC1", "TIC2")
        if xy.size:
            if plot_fes_from_csv(
                advanced_root / "tica" / "free_energy_landscape_tic1_tic2.csv",
                advanced_root / "tica" / "free_energy_landscape_tic1_tic2",
                "tICA free-energy landscape",
                "tIC1",
                "tIC2",
                cfg.plot_style,
            ):
                did_plot = True
            scatter_by_replica(
                xy,
                [SimpleNamespace(name=name) for name in names],
                arrays,
                advanced_root / "tica" / "tic1_tic2_scatter",
                "tIC1",
                "tIC2",
                "tICA projection by replica",
                cfg.plot_style,
            )
            did_plot = True

    if cfg.plot_selection.enabled("advanced_clustering"):
        population_path = advanced_root / "clustering" / "cluster_population_overall.csv"
        if population_path.exists():
            rows = read_dict_csv(population_path)
            if rows:
                plot_cluster_population(
                    _numeric_column(rows, "cluster").astype(int),
                    _numeric_column(rows, "fraction"),
                    advanced_root / "clustering" / "cluster_population_overall",
                    cfg.plot_style,
                )
                did_plot = True
        tic_names, _tic_arrays, tic_xy = _projection_stack(assign_root, "tic", "TIC1", "TIC2")
        label_arrays = []
        for name in tic_names:
            assignment_path = assign_root / f"{name}_cluster_assignment.csv"
            if assignment_path.exists():
                label_arrays.append(_numeric_column(read_dict_csv(assignment_path), "cluster").astype(int))
        center_path = advanced_root / "clustering" / "cluster_centers.csv"
        if tic_xy.size and label_arrays and center_path.exists():
            center_rows = sorted(read_dict_csv(center_path), key=lambda row: int(row["cluster"]))
            centers = np.asarray([[float(row["coord1"]), float(row["coord2"])] for row in center_rows], dtype=float)
            scatter_clusters(
                tic_xy,
                np.concatenate(label_arrays),
                centers,
                advanced_root / "clustering" / "clusters_tic1_tic2",
                cfg.plot_style,
            )
            did_plot = True
        pop_matrix_path = advanced_root / "clustering" / "cluster_population_per_replica.csv"
        if pop_matrix_path.exists():
            rows = read_dict_csv(pop_matrix_path)
            if rows:
                replica_names = sorted({row["replica"] for row in rows}, key=_replica_sort_key)
                cluster_labels = sorted({int(row["cluster"]) for row in rows})
                matrix = np.zeros((len(replica_names), len(cluster_labels)), dtype=float)
                replica_index = {name: idx for idx, name in enumerate(replica_names)}
                cluster_index = {cluster: idx for idx, cluster in enumerate(cluster_labels)}
                for row in rows:
                    matrix[replica_index[row["replica"]], cluster_index[int(row["cluster"])]] = float(row["fraction"])
                plot_state_population_heatmap(
                    replica_names,
                    cluster_labels,
                    matrix,
                    advanced_root / "clustering" / "state_population_by_replica",
                    cfg.plot_style,
                )
                did_plot = True

    if cfg.plot_selection.enabled("advanced_snapshots"):
        snapshot_root = advanced_root / "snapshots"
        for count, out_base in [
            (cfg.advanced.representative_snapshot_clusters, snapshot_root / "representative_state_snapshots"),
            (6, snapshot_root / "top_6_states" / "representative_state_snapshots"),
            (8, snapshot_root / "top_8_states" / "representative_state_snapshots"),
        ]:
            entries = _snapshot_entries_from_existing_pdbs(snapshot_root, cfg, count)
            if entries:
                plot_snapshot_grid(entries, out_base, cfg.plot_style, title="Representative structures of dominant states")
                did_plot = True

    if cfg.plot_selection.enabled("advanced_msm"):
        msm_root = advanced_root / "msm"
        notes = _load_json_if_exists(msm_root / "msm_notes.json")
        state_labels = _msm_state_labels(msm_root)
        stationary_path = msm_root / "stationary_distribution.csv"
        if stationary_path.exists():
            rows = read_dict_csv(stationary_path)
            if rows:
                active_clusters = ", ".join(str(int(row["source_cluster"])) for row in rows)
                summary_lines = [
                    f"Active clusters: {active_clusters}",
                    f"Coverage: {float(notes.get('selected_component_frame_fraction', 0.0)):.1%}",
                    f"MSM lag: {int(notes.get('used_safe_msm_lag_frames', cfg.advanced.msm_lag_frames))} frames",
                    f"Effective states: {float(notes.get('effective_state_count', 0.0)):.2f}",
                ]
                plot_stationary_distribution(
                    _numeric_column(rows, "msm_state").astype(int),
                    _numeric_column(rows, "stationary_probability"),
                    msm_root / "stationary_distribution",
                    cfg.plot_style,
                    state_labels=state_labels or None,
                    summary_lines=summary_lines,
                )
                did_plot = True
        _matrix_labels, transition_matrix = _read_square_matrix(msm_root / "transition_matrix.csv")
        _flux_labels, flux_matrix = _read_square_matrix(msm_root / "equilibrium_transition_flux.csv")
        _mfpt_labels, mfpt_matrix = _read_square_matrix(msm_root / "mean_first_passage_times.csv")
        if transition_matrix.size:
            plot_transition_matrix_heatmap(
                transition_matrix,
                msm_root / "transition_matrix_heatmap",
                cfg.plot_style,
                state_labels=state_labels or None,
                flux_matrix=flux_matrix if flux_matrix.size else None,
            )
            if stationary_path.exists():
                stationary_rows = read_dict_csv(stationary_path)
                pi = _numeric_column(stationary_rows, "stationary_probability") if stationary_rows else np.array([])
                if pi.size == transition_matrix.shape[0]:
                    plot_state_network(
                        transition_matrix,
                        pi,
                        msm_root / "state_network",
                        cfg.plot_style,
                        cfg.advanced.state_network_threshold,
                        state_labels=state_labels or None,
                        mfpt_matrix=mfpt_matrix if mfpt_matrix.size else None,
                        summary_lines=[
                            f"Coverage {float(notes.get('selected_component_frame_fraction', 0.0)):.1%}",
                            f"max detailed-balance residual = {float(notes.get('max_detailed_balance_residual', 0.0)):.2e}",
                        ],
                    )
            did_plot = True
        its_path = msm_root / "implied_timescales_single_lag.csv"
        if its_path.exists():
            rows = read_dict_csv(its_path)
            if rows:
                plot_line_profile(
                    _numeric_column(rows, "index"),
                    _numeric_column(rows, "timescale_frames"),
                    msm_root / "implied_timescales_single_lag",
                    "Process index",
                    "Timescale (frames)",
                    f"MSM implied timescales at lag = {int(notes.get('used_safe_msm_lag_frames', cfg.advanced.msm_lag_frames))} frames",
                    cfg.plot_style,
                    color=cfg.plot_style.accent_color,
                    yscale="log",
                )
                did_plot = True
        lag_scan_path = msm_root / "implied_timescales_lag_scan.csv"
        if lag_scan_path.exists():
            lag_rows = [
                [float(row["lag_frames"]), float(row["process_index"]), float(row["timescale_frames"])]
                for row in read_dict_csv(lag_scan_path)
            ]
            diag_path = msm_root / "lag_scan_diagnostics.csv"
            diag_rows = (
                [
                    [
                        float(row["lag_frames"]),
                        float(row["used_lag_frames"]),
                        float(row["usable_segments"]),
                        float(row["usable_frames"]),
                        float(row["active_frame_count"]),
                    ]
                    for row in read_dict_csv(diag_path)
                ]
                if diag_path.exists()
                else None
            )
            if lag_rows:
                plot_lag_scan(
                    lag_rows,
                    msm_root / "implied_timescales_lag_scan",
                    cfg.plot_style,
                    selected_lag=int(notes.get("used_safe_msm_lag_frames", cfg.advanced.msm_lag_frames)),
                    diagnostic_rows=diag_rows,
                )
                did_plot = True
        ck_test = _ck_test_from_csv(msm_root / "chapman_kolmogorov_test.csv")
        if ck_test is not None:
            ck_notes = notes.get("chapman_kolmogorov_validation", {})
            plot_chapman_kolmogorov_test(
                ck_test,
                msm_root / "chapman_kolmogorov_test",
                cfg.plot_style,
                state_labels=[f"Meta {idx}" for idx in range(ck_test.predictions.shape[1])],
                summary_lines=[
                    f"base lag {ck_notes.get('base_lag_frames', notes.get('used_safe_msm_lag_frames', cfg.advanced.msm_lag_frames))}",
                    f"RMSE {float(ck_notes.get('rmse', np.nan)):.3e}",
                    f"max abs diff {float(ck_notes.get('max_absolute_difference', np.nan)):.3f}",
                ],
            )
            did_plot = True

    return str(advanced_root.resolve()) if did_plot else None


def run_plot_postprocess(
    cfg: WorkflowConfig,
    progress_callback: ProgressCallback | None = None,
    *,
    include_mmgbsa_postprocess: bool = True,
    include_organize_outputs: bool = True,
):
    apply_plot_style_palette(cfg.plot_style)
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
            basic_replot_dir = run_basic_replot_from_csv(cfg)
            if basic_replot_dir is not None:
                outputs["basic_replot_dir"] = basic_replot_dir
                outputs.setdefault("replotted_from_csv_sections", []).append("basic")
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
            waterbridge_replot_dir = run_waterbridge_replot_from_csv(cfg)
            if waterbridge_replot_dir is not None:
                outputs["waterbridge_replot_dir"] = waterbridge_replot_dir
                outputs.setdefault("replotted_from_csv_sections", []).append("waterbridge")
                detail = "Completed combined water-bridge replot"
            else:
                detail = "Skipping combined water-bridge replot because waterbridge_count_combined.csv was not found"
        else:
            detail = "Skipping combined water-bridge replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "waterbridge_replot", detail)
    if cfg.do_advanced_analysis:
        if cfg.plot_selection.enabled("plot_workflow_advanced_replot"):
            emit_progress(progress_callback, completed_units, total_units, "advanced_replot", "Replotting advanced-analysis figures")
            advanced_replot_dir = run_advanced_replot_from_csv(cfg)
            if advanced_replot_dir is not None:
                outputs["advanced_replot_dir"] = advanced_replot_dir
                outputs.setdefault("replotted_from_csv_sections", []).append("advanced")
                detail = "Completed advanced-analysis replot"
            else:
                detail = "Skipping advanced-analysis replot because reusable advanced CSV outputs were not found"
        else:
            detail = "Skipping advanced-analysis replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "advanced_replot", detail)
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
            workflow_cfg=cfg,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "organize_outputs", "Workflow outputs organized")
    return outputs


def run_plot_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = normalize_workflow_paths(deepcopy(cfg))
    apply_plot_style_palette(cfg.plot_style)
    ensure_project_layout(cfg.workspace_root)
    return run_plot_postprocess(cfg, progress_callback=progress_callback)
