from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from ..config import PlotSelectionConfig, PlotStyleConfig
from ..core import aggregate_metric_rows, save_csv, write_dict_csv
from .bars import horizontal_bars
from .heatmaps import matrix_heatmap, simple_boxplot, stacked_fraction_area
from .series import line_series, mean_sd_series, shaded_profile
from .theme import finalize_axes, publication_style, save_figure
import matplotlib.pyplot as plt


def _zscore_within(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.full(arr.shape, np.nan, dtype=float)
    if np.count_nonzero(finite) == 0:
        return out
    center = float(np.nanmean(arr[finite]))
    scale = float(np.nanstd(arr[finite], ddof=1)) if np.count_nonzero(finite) > 1 else 0.0
    if scale <= 1e-12:
        out[finite] = 0.0
    else:
        out[finite] = (arr[finite] - center) / scale
    return out


def _display_metric_name(metric: str) -> str:
    return {
        "protein_rmsd_A": "Protein RMSD",
        "ligand_rmsd_A": "Ligand RMSD",
        "min_distance_A": "Min distance",
        "radius_of_gyration_A": "Radius of gyration",
        "buried_surface_A2": "Buried surface",
        "hbond_count": "H-bond count",
    }.get(metric, metric.replace("_", " "))


def write_overall_summary(replica_results, combined_dir: Path):
    summary_rows = [r["summary"] for r in replica_results]
    write_dict_csv(combined_dir / "summary_per_replica.csv", summary_rows, list(summary_rows[0].keys()))
    overall_rows = []
    numeric_keys = [k for k, v in summary_rows[0].items() if isinstance(v, (int, float))]
    for key in numeric_keys:
        vals = np.asarray([r[key] for r in summary_rows], dtype=float)
        overall_rows.append({"metric": key, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1) if len(vals) > 1 else 0.0), "min": float(vals.min()), "max": float(vals.max()), "n_replicas": int(len(vals))})
    write_dict_csv(combined_dir / "summary_overall.csv", overall_rows, ["metric", "mean", "sd", "min", "max", "n_replicas"])


def _common_time_and_stack(replica_results, key: str):
    min_n = min(len(r["time_ns"]) for r in replica_results)
    time_ns = replica_results[0]["time_ns"][:min_n]
    stack = np.vstack([np.asarray(r[key][:min_n], dtype=float) for r in replica_results])
    names = [r["replica_name"] for r in replica_results]
    return time_ns, stack, names


def _write_stack_csv(path: Path, time_ns, stack, names, mean_label: str, sd_label: str):
    save_csv(path, ["time_ns"] + names + [mean_label, sd_label], np.column_stack([time_ns, stack.T, stack.mean(axis=0), stack.std(axis=0, ddof=1) if stack.shape[0] > 1 else np.zeros(stack.shape[1])]).tolist())


def plot_combined_rmsd(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    min_n_frames = min(len(r["time_ns"]) for r in replica_results)
    common_time_ns = replica_results[0]["time_ns"][:min_n_frames]
    names = [r["replica_name"] for r in replica_results]
    protein = np.vstack([r["protein_rmsd_A"][:min_n_frames] for r in replica_results])
    ligand = np.vstack([r["ligand_rmsd_A"][:min_n_frames] for r in replica_results])
    save_csv(combined_dir / "rmsd_combined.csv", ["time_ns"] + [f"{name}_protein_backbone_rmsd_A" for name in names] + [f"{name}_ligand_heavy_rmsd_A" for name in names] + ["protein_backbone_rmsd_mean_A", "protein_backbone_rmsd_sd_A", "ligand_heavy_rmsd_mean_A", "ligand_heavy_rmsd_sd_A"], np.column_stack([common_time_ns, protein.T, ligand.T, protein.mean(axis=0), protein.std(axis=0, ddof=1) if protein.shape[0] > 1 else np.zeros(min_n_frames), ligand.mean(axis=0), ligand.std(axis=0, ddof=1) if ligand.shape[0] > 1 else np.zeros(min_n_frames)]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_combined_rmsd"):
        with publication_style(style):
            fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.2), sharex=True)
            for i, name in enumerate(names):
                color = style.categorical_palette[i % len(style.categorical_palette)]
                axes[0].plot(common_time_ns, protein[i], linewidth=style.thin_line_width, alpha=0.35, label=name, color=color)
                axes[1].plot(common_time_ns, ligand[i], linewidth=style.thin_line_width, alpha=0.35, label=name, color=color)
            p_mean, p_sd = protein.mean(axis=0), protein.std(axis=0, ddof=1) if protein.shape[0] > 1 else np.zeros_like(protein.mean(axis=0))
            l_mean, l_sd = ligand.mean(axis=0), ligand.std(axis=0, ddof=1) if ligand.shape[0] > 1 else np.zeros_like(ligand.mean(axis=0))
            axes[0].plot(common_time_ns, p_mean, linewidth=style.line_width + 0.4, label="Mean", color=style.protein_color)
            axes[0].fill_between(common_time_ns, p_mean - p_sd, p_mean + p_sd, alpha=0.22, linewidth=0.0, color=style.band_color)
            axes[1].plot(common_time_ns, l_mean, linewidth=style.line_width + 0.4, label="Mean", color=style.ligand_color)
            axes[1].fill_between(common_time_ns, l_mean - l_sd, l_mean + l_sd, alpha=0.22, linewidth=0.0, color=style.band_color)
            finalize_axes(axes[0], style, ylabel="Protein backbone RMSD (Å)", title="RMSD across replicas")
            finalize_axes(axes[1], style, xlabel="Time (ns)", ylabel="Ligand heavy-atom RMSD (Å)")
            legend_handles, legend_labels = axes[0].get_legend_handles_labels()
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.01),
                ncol=min(max(len(legend_labels), 1), 3),
                frameon=False,
            )
            save_figure(fig, combined_dir / "rmsd_combined", style)


def plot_combined_min_distance(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    time_ns, stack, names = _common_time_and_stack(replica_results, "global_min_distance_A")
    _write_stack_csv(combined_dir / "min_distance_combined.csv", time_ns, stack, [f"{n}_min_distance_A" for n in names], "mean_min_distance_A", "sd_min_distance_A")
    if plot_selection is None or plot_selection.enabled("basic_combined_min_distance"):
        mean_sd_series(time_ns, stack, "Minimum ligand-protein distance (Å)", combined_dir / "min_distance_combined", style, title="Ligand-protein minimum heavy-atom distance", individual_labels=names)


def plot_combined_rmsf(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    rmsf_values = defaultdict(list)
    residue_seq_map = {}
    for result in replica_results:
        for row in result["rmsf_rows"]:
            rmsf_values[row["protein_residue"]].append(row["rmsf_A"])
            residue_seq_map[row["protein_residue"]] = row["resSeq"]
    rows = []
    for residue, vals in rmsf_values.items():
        vals = np.asarray(vals, dtype=float)
        rows.append({"protein_residue": residue, "resSeq": residue_seq_map[residue], "rmsf_mean_A": float(vals.mean()), "rmsf_sd_A": float(vals.std(ddof=1) if len(vals) > 1 else 0.0), "n_replicas": int(len(vals))})
    rows.sort(key=lambda x: x["resSeq"])
    write_dict_csv(combined_dir / "rmsf_ca_combined.csv", rows, ["protein_residue", "resSeq", "rmsf_mean_A", "rmsf_sd_A", "n_replicas"])
    x = np.asarray([r["resSeq"] for r in rows], dtype=float)
    y = np.asarray([r["rmsf_mean_A"] for r in rows], dtype=float)
    ysd = np.asarray([r["rmsf_sd_A"] for r in rows], dtype=float)
    if plot_selection is None or plot_selection.enabled("basic_combined_rmsf"):
        shaded_profile(x, y, ysd, "Cα RMSF (Å)", combined_dir / "rmsf_ca_combined", style, title="Protein Cα RMSF across replicas")


def plot_combined_occupancy_bars(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    top_n_contacts_plot: int,
    plot_selection: PlotSelectionConfig | None = None,
):
    specs = [
        ("contact_rows", "contact_occupancy", "contact_occupancy", "contact_occupancy_combined.csv", "contact_occupancy_top20", "Top residue contact occupancy", "Contact occupancy"),
        ("hbond_residues", "hbond_occupancy", "hbond_occupancy", "hbond_residue_occupancy_combined.csv", "hbond_residue_occupancy_top20", "Top residue H-bond occupancy", "H-bond occupancy"),
        ("salt_residue_rows", "salt_bridge_occupancy", "salt_bridge_occupancy", "salt_bridge_residue_occupancy_combined.csv", "salt_bridge_residue_occupancy", "Salt bridge occupancy", "Salt bridge occupancy"),
    ]
    combined_rows = {}
    for key_name, value_name, out_field_name, csv_name, base_name, title, xlabel in specs:
        rows = aggregate_metric_rows(replica_results, key_name, value_name, out_field_name)
        combined_rows[out_field_name] = rows
        write_dict_csv(combined_dir / csv_name, rows, ["protein_residue", f"{out_field_name}_mean", f"{out_field_name}_sd", "n_replicas"])
        if rows and (plot_selection is None or plot_selection.enabled("basic_combined_occupancy_bars")):
            top_rows = rows[:top_n_contacts_plot]
            horizontal_bars([r["protein_residue"] for r in reversed(top_rows)], [r[f"{out_field_name}_mean"] for r in reversed(top_rows)], [r[f"{out_field_name}_sd"] for r in reversed(top_rows)], combined_dir / base_name, style, title=title, xlabel=xlabel)
    return combined_rows


def plot_key_contact_traces(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    top_n_key_distance_residues: int,
    plot_selection: PlotSelectionConfig | None = None,
):
    contact_combined = aggregate_metric_rows(replica_results, "contact_rows", "contact_occupancy", "contact_occupancy")
    key_labels = [r["protein_residue"] for r in contact_combined[:top_n_key_distance_residues]]
    if not key_labels:
        return
    min_n_frames = min(len(r["time_ns"]) for r in replica_results)
    common_time_ns = replica_results[0]["time_ns"][:min_n_frames]
    rows = []
    plotting_enabled = plot_selection is None or plot_selection.enabled("basic_combined_key_contact_traces")
    if plotting_enabled:
        with publication_style(style):
            fig, ax = plt.subplots(figsize=(7.4, 4.9))
            for idx, label in enumerate(key_labels):
                curves = []
                for result in replica_results:
                    curve = result["contact_curves_A"].get(label)
                    if curve is not None:
                        curves.append(curve[:min_n_frames])
                if not curves:
                    continue
                arr = np.vstack(curves)
                mean = arr.mean(axis=0)
                sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
                color = style.categorical_palette[idx % len(style.categorical_palette)]
                ax.plot(common_time_ns, mean, linewidth=style.line_width, label=label, color=color)
                ax.fill_between(common_time_ns, mean - sd, mean + sd, alpha=0.16, linewidth=0.0, color=color)
                rows.extend([[label, t, m, s] for t, m, s in zip(common_time_ns, mean, sd)])
            finalize_axes(ax, style, xlabel="Time (ns)", ylabel="Distance (Å)", title="Key residue-ligand distance traces")
            ax.legend(frameon=False, ncol=1)
            save_figure(fig, combined_dir / "key_contact_distance_traces", style)
    else:
        for label in key_labels:
            curves = []
            for result in replica_results:
                curve = result["contact_curves_A"].get(label)
                if curve is not None:
                    curves.append(curve[:min_n_frames])
            if not curves:
                continue
            arr = np.vstack(curves)
            mean = arr.mean(axis=0)
            sd = arr.std(axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
            rows.extend([[label, t, m, s] for t, m, s in zip(common_time_ns, mean, sd)])
    save_csv(combined_dir / "key_contact_distance_traces.csv", ["protein_residue", "time_ns", "mean_distance_A", "sd_distance_A"], rows)


def plot_combined_counts_and_shapes(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    series_specs = [
        ("contact_count", "contact_count_combined", "Contact count", "Count"),
        ("hbond_count", "hbond_count_combined", "H-bond count", "Count"),
        ("salt_bridge_count", "salt_bridge_count_combined", "Salt-bridge count", "Count"),
        ("rg_A", "radius_of_gyration_combined", "Radius of gyration (Å)", "Radius of gyration (Å)"),
        ("buried_surface_A2", "buried_surface_combined", "Buried surface area (Å²)", "Area (Å²)"),
        ("com_distance_A", "ligand_com_distance_combined", "Ligand-pocket COM distance (Å)", "Distance (Å)"),
        ("orientation_angle_deg", "ligand_orientation_angle_combined", "Ligand orientation angle (deg)", "Angle (deg)"),
    ]
    for key, base, title, ylabel in series_specs:
        time_ns, stack, names = _common_time_and_stack(replica_results, key)
        _write_stack_csv(combined_dir / f"{base}.csv", time_ns, stack, [f"{n}_{key}" for n in names], f"mean_{key}", f"sd_{key}")
        if plot_selection is None or plot_selection.enabled("basic_combined_counts_and_shapes"):
            mean_sd_series(time_ns, stack, ylabel, combined_dir / base, style, title=title, individual_labels=names)

    time_ns, complex_stack, names = _common_time_and_stack(replica_results, "complex_sasa_A2")
    _, protein_stack, _ = _common_time_and_stack(replica_results, "protein_sasa_A2")
    _, ligand_stack, _ = _common_time_and_stack(replica_results, "ligand_sasa_A2")
    save_csv(combined_dir / "sasa_components_combined.csv", ["time_ns", "complex_sasa_mean_A2", "protein_sasa_mean_A2", "ligand_sasa_mean_A2"], np.column_stack([time_ns, complex_stack.mean(axis=0), protein_stack.mean(axis=0), ligand_stack.mean(axis=0)]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_combined_counts_and_shapes"):
        line_series(time_ns, [complex_stack.mean(axis=0), protein_stack.mean(axis=0), ligand_stack.mean(axis=0)], ["Complex SASA", "Protein SASA", "Ligand SASA"], "Area (Å²)", combined_dir / "sasa_components_combined", style, title="SASA components across replicas")


def plot_combined_dssp(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    residue_map = defaultdict(list)
    frac_map = {"Helix": [], "Sheet": [], "Coil": []}
    for result in replica_results:
        for row in result["dssp_residue_rows"]:
            residue_map[row["protein_residue"]].append([row["helix_fraction"], row["sheet_fraction"], row["coil_fraction"]])
        for key in frac_map:
            frac_map[key].append(np.asarray(result["dssp_fractions"][key], dtype=float))
    rows = []
    residue_labels = []
    matrix = []
    for residue in sorted(residue_map.keys()):
        arr = np.asarray(residue_map[residue], dtype=float)
        mean = arr.mean(axis=0)
        residue_labels.append(residue)
        matrix.append(mean)
        rows.append({"protein_residue": residue, "helix_fraction_mean": float(mean[0]), "sheet_fraction_mean": float(mean[1]), "coil_fraction_mean": float(mean[2])})
    write_dict_csv(combined_dir / "dssp_residue_occupancy_combined.csv", rows, ["protein_residue", "helix_fraction_mean", "sheet_fraction_mean", "coil_fraction_mean"])
    plotting_enabled = plot_selection is None or plot_selection.enabled("basic_combined_dssp")
    if matrix and plotting_enabled:
        matrix_heatmap(
            np.asarray(matrix, dtype=float),
            residue_labels,
            ["Helix", "Sheet", "Coil"],
            combined_dir / "dssp_residue_occupancy_combined",
            style,
            title="Combined residue secondary-structure occupancy",
            xlabel="Secondary-structure state",
            ylabel="Residue",
            vmin=0.0,
            vmax=1.0,
            cbar_label="Occupancy fraction",
        )
    time_ns = replica_results[0]["time_ns"][: min(len(r["time_ns"]) for r in replica_results)]
    mean_fracs = {key: np.vstack([arr[:len(time_ns)] for arr in frac_map[key]]).mean(axis=0) for key in frac_map}
    if plotting_enabled:
        stacked_fraction_area(time_ns, mean_fracs, combined_dir / "dssp_fractions_combined", style, title="Combined secondary-structure fractions")


def plot_interaction_heatmaps(
    replica_results,
    occupancy_rows,
    combined_dir: Path,
    style: PlotStyleConfig,
    top_n_contacts_plot: int,
    plot_selection: PlotSelectionConfig | None = None,
):
    contact_rows = occupancy_rows.get("contact_occupancy", [])
    if not contact_rows:
        return
    if plot_selection is not None and not plot_selection.enabled("basic_combined_interaction_heatmaps"):
        return
    top_res = [r["protein_residue"] for r in contact_rows[:top_n_contacts_plot]]
    per_rep_matrix = []
    for residue in top_res:
        row = []
        for result in replica_results:
            val = 0.0
            for entry in result["contact_rows"]:
                if entry["protein_residue"] == residue:
                    val = float(entry["contact_occupancy"])
                    break
            row.append(val)
        per_rep_matrix.append(row)
    matrix_heatmap(
        np.asarray(per_rep_matrix, dtype=float),
        top_res,
        [r["replica_name"] for r in replica_results],
        combined_dir / "contact_replicate_heatmap",
        style,
        title="Residue contact occupancy by replica",
        xlabel="Replica",
        ylabel="Residue",
        vmin=0.0,
        vmax=1.0,
        cbar_label="Contact occupancy",
        x_rotation=35.0,
    )

    metric_lookup = {
        "Contact": occupancy_rows.get("contact_occupancy", []),
        "H-bond": occupancy_rows.get("hbond_occupancy", []),
        "Salt bridge": occupancy_rows.get("salt_bridge_occupancy", []),
    }
    all_res = []
    for rows in metric_lookup.values():
        all_res.extend([r["protein_residue"] for r in rows[:top_n_contacts_plot]])
    all_res = list(dict.fromkeys(all_res))[: max(top_n_contacts_plot, 12)]
    matrix = []
    for residue in all_res:
        vals = []
        for metric_name, rows in metric_lookup.items():
            mean_key = {
                "Contact": "contact_occupancy_mean",
                "H-bond": "hbond_occupancy_mean",
                "Salt bridge": "salt_bridge_occupancy_mean",
            }[metric_name]
            match = next((float(r[mean_key]) for r in rows if r["protein_residue"] == residue), 0.0)
            vals.append(match)
        matrix.append(vals)
    matrix_heatmap(
        np.asarray(matrix, dtype=float),
        all_res,
        list(metric_lookup.keys()),
        combined_dir / "interaction_fingerprint_heatmap",
        style,
        title="Residue interaction fingerprint heatmap",
        xlabel="Interaction class",
        ylabel="Residue",
        vmin=0.0,
        vmax=1.0,
        annotate=False,
        cbar_label="Interaction occupancy",
    )


def plot_convergence(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    block_names = sorted({int(row["block"]) for result in replica_results for row in result["convergence_blocks"]})
    metrics = ["protein_rmsd_A", "ligand_rmsd_A", "min_distance_A", "radius_of_gyration_A", "buried_surface_A2", "hbond_count"]
    rows = []
    matrix = []
    row_labels = []
    zscore_rows = []
    zscore_matrix = []
    for metric in metrics:
        metric_block_values = []
        for block in block_names:
            vals = []
            for result in replica_results:
                for row in result["convergence_blocks"]:
                    if int(row["block"]) == block and row["metric"] == metric:
                        vals.append(float(row["mean"]))
            metric_block_values.append(np.mean(vals) if vals else np.nan)
            rows.append({"metric": metric, "block": block, "mean": float(np.mean(vals) if vals else np.nan), "sd": float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)})
        row_labels.append(_display_metric_name(metric))
        matrix.append(metric_block_values)
        zscores = _zscore_within(metric_block_values)
        zscore_matrix.append(zscores)
        for block, zvalue in zip(block_names, zscores):
            zscore_rows.append({"metric": metric, "block": block, "zscore": float(zvalue) if np.isfinite(zvalue) else np.nan})
    write_dict_csv(combined_dir / "convergence_block_means_combined.csv", rows, ["metric", "block", "mean", "sd"])
    write_dict_csv(combined_dir / "convergence_block_zscores_combined.csv", zscore_rows, ["metric", "block", "zscore"])
    plotting_enabled = plot_selection is None or plot_selection.enabled("basic_combined_convergence")
    if plotting_enabled:
        matrix_heatmap(
            np.asarray(zscore_matrix, dtype=float),
            row_labels,
            [f"Block {b}" for b in block_names],
            combined_dir / "convergence_block_heatmap",
            style,
            title="Convergence block deviations across replicas",
            xlabel="Trajectory block",
            ylabel="Metric",
            annotate=True,
            center=0.0,
            cbar_label="Deviation from metric mean (SD)",
        )

    summary_rows = [r["summary"] for r in replica_results]
    metric_values = {
        "Protein RMSD": [float(r["protein_backbone_rmsd_mean_A"]) for r in summary_rows],
        "Ligand RMSD": [float(r["ligand_heavy_rmsd_mean_A"]) for r in summary_rows],
        "Buried area": [float(r["buried_surface_mean_A2"]) for r in summary_rows],
        "H-bond count": [float(r["hbond_count_mean"]) for r in summary_rows],
    }
    replica_names = [r["replica"] for r in summary_rows]
    box_data = {metric: _zscore_within(values).tolist() for metric, values in metric_values.items()}
    zscore_table_rows = []
    zscore_matrix = []
    metric_labels = list(metric_values.keys())
    for ridx, replica_name in enumerate(replica_names):
        row = []
        for metric_name in metric_labels:
            zvalue = float(box_data[metric_name][ridx])
            row.append(zvalue)
            zscore_table_rows.append({"replica": replica_name, "metric": metric_name, "zscore": zvalue})
        zscore_matrix.append(row)
    write_dict_csv(combined_dir / "replicate_consistency_zscores.csv", zscore_table_rows, ["replica", "metric", "zscore"])
    if plotting_enabled:
        simple_boxplot(
            box_data,
            combined_dir / "replicate_consistency_boxplot",
            style,
            title="Replica consistency summary (metric-standardized)",
            ylabel="Deviation from across-replica mean (SD)",
            show_points=True,
            hline_zero=True,
        )
        matrix_heatmap(
            np.asarray(zscore_matrix, dtype=float),
            replica_names,
            metric_labels,
            combined_dir / "replicate_consistency_zscore_heatmap",
            style,
            title="Replica consistency by metric",
            xlabel="Metric",
            ylabel="Replica",
            annotate=True,
            center=0.0,
            cbar_label="Deviation from across-replica mean (SD)",
        )


def plot_combined_basic_results(
    replica_results,
    analysis_root: str,
    style: PlotStyleConfig,
    top_n_contacts_plot: int = 20,
    top_n_key_distance_residues: int = 5,
    plot_selection: PlotSelectionConfig | None = None,
):
    combined_dir = Path(analysis_root) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_overall_summary(replica_results, combined_dir)
    plot_combined_rmsd(replica_results, combined_dir, style, plot_selection)
    plot_combined_min_distance(replica_results, combined_dir, style, plot_selection)
    plot_combined_rmsf(replica_results, combined_dir, style, plot_selection)
    occupancy_rows = plot_combined_occupancy_bars(replica_results, combined_dir, style, top_n_contacts_plot, plot_selection)
    plot_key_contact_traces(replica_results, combined_dir, style, top_n_key_distance_residues, plot_selection)
    plot_combined_counts_and_shapes(replica_results, combined_dir, style, plot_selection)
    plot_combined_dssp(replica_results, combined_dir, style, plot_selection)
    plot_interaction_heatmaps(replica_results, occupancy_rows, combined_dir, style, top_n_contacts_plot, plot_selection)
    plot_convergence(replica_results, combined_dir, style, plot_selection)
