from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np

from ..config import PlotSelectionConfig, PlotStyleConfig
from ..core import aggregate_metric_rows, save_csv, write_dict_csv
from .bars import ranked_distance_lollipop, ranked_lollipop
from .heatmaps import interaction_fingerprint_heatmap, matrix_heatmap, simple_boxplot, stacked_fraction_area
from .residue_labels import compact_replica_name, compact_residue_label
from .series import (
    direct_label_line_series,
    draw_replicate_summary,
    publication_replicate_series,
    shaded_profile,
)
from .theme import finalize_axes, publication_style, remove_figure_outputs, save_figure
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


def _interaction_color(style: PlotStyleConfig, metric_key: str) -> str:
    return {
        "contact_occupancy": style.protein_color,
        "hbond_occupancy": style.distance_color,
        "salt_bridge_occupancy": style.accent_color,
    }.get(metric_key, style.bar_color)


def _interaction_rank_title(metric_key: str) -> str:
    return {
        "contact_occupancy": "Persistent contact hotspots",
        "hbond_occupancy": "Persistent H-bond hotspots",
        "salt_bridge_occupancy": "Persistent salt-bridge hotspots",
    }.get(metric_key, "Interaction hotspots")


def _interaction_rank_xlabel(metric_key: str) -> str:
    return {
        "contact_occupancy": "Contact occupancy",
        "hbond_occupancy": "H-bond occupancy",
        "salt_bridge_occupancy": "Salt-bridge occupancy",
    }.get(metric_key, "Occupancy")


def _series_summary_color(metric_key: str, style: PlotStyleConfig) -> str:
    return {
        "protein_rmsd_A": style.protein_color,
        "ligand_rmsd_A": style.ligand_color,
        "global_min_distance_A": style.distance_color,
        "contact_count": style.protein_color,
        "hbond_count": style.distance_color,
        "salt_bridge_count": style.accent_color,
        "rg_A": style.protein_color,
        "buried_surface_A2": style.accent_color,
        "com_distance_A": style.distance_color,
        "orientation_angle_deg": style.ligand_color,
    }.get(metric_key, style.mean_line_color)


def _safe_figure_token(label: str) -> str:
    token = re.sub(r"[^A-Za-z0-9]+", "_", compact_residue_label(label)).strip("_").lower()
    return token or "residue"


def _key_contact_out_base(combined_dir: Path, label: str) -> Path:
    return combined_dir / "key_contact_distance_traces" / f"key_contact_distance_{_safe_figure_token(label)}"


def _key_contact_axis_cap(mean, sd, contact_cutoff_A: float | None = None) -> tuple[float | None, bool]:
    upper = np.asarray(mean, dtype=float) + np.asarray(sd, dtype=float)
    finite_upper = upper[np.isfinite(upper)]
    if not finite_upper.size:
        return None, False
    focus_limit = float(np.nanpercentile(finite_upper, 98.5))
    raw_max = float(np.nanmax(finite_upper))
    if raw_max <= focus_limit * 1.35:
        focus_limit = raw_max
    cutoff_focus = float(contact_cutoff_A) * 1.25 if contact_cutoff_A is not None else 0.0
    axis_cap = max(5.5, cutoff_focus, focus_limit * 1.08)
    axis_cap = float(np.ceil(axis_cap * 2.0) / 2.0)
    clipped = raw_max > axis_cap + 1e-9
    return axis_cap, clipped


def plot_key_contact_distance_figure(
    time_ns,
    mean,
    sd,
    out_base: Path,
    style: PlotStyleConfig,
    *,
    display_label: str,
    color: str,
    occupancy: float = np.nan,
    contact_cutoff_A: float | None = None,
    rolling_window_fraction: float = 0.10,
    arr2d=None,
    axis_cap: float | None = None,
    clipped: bool = False,
) -> None:
    with publication_style(style):
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        if contact_cutoff_A is not None:
            ax.axhline(
                float(contact_cutoff_A),
                color=style.spine_color,
                linewidth=0.9,
                linestyle=(0, (3, 3)),
                alpha=0.48,
                zorder=0,
            )
        if arr2d is not None:
            draw_replicate_summary(
                ax,
                time_ns,
                arr2d,
                style,
                color=color,
                rolling_window_fraction=rolling_window_fraction,
            )
        else:
            from .series import draw_summary_band

            draw_summary_band(
                ax,
                time_ns,
                mean,
                sd,
                style,
                color=color,
                rolling_window_fraction=rolling_window_fraction,
            )
        if axis_cap is not None:
            ax.set_ylim(0.0, axis_cap)
        finalize_axes(
            ax,
            style,
            xlabel="Time (ns)",
            ylabel="Minimum heavy-atom distance (Å)",
            title=f"Key contact distance: {display_label}",
        )
        if np.isfinite(occupancy):
            ax.text(
                0.012,
                0.965,
                f"contact occupancy {occupancy:.0%}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color=color,
                fontsize=max(style.legend_size + 0.1, 7.0),
                weight="semibold",
            )
        if clipped and axis_cap is not None:
            ax.text(
                0.988,
                0.035,
                f"spikes clipped above {axis_cap:.1f} Å",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                color=style.spine_color,
                fontsize=max(style.legend_size - 0.5, 6.5),
                alpha=0.75,
            )
        save_figure(fig, out_base, style)


def _build_contact_distance_summary(replica_results):
    by_residue: dict[str, dict[str, list[float]]] = {}
    for result in replica_results:
        for row in result["contact_rows"]:
            residue = str(row["protein_residue"])
            payload = by_residue.setdefault(
                residue,
                {"contact_occupancy": [], "min_distance_mean_A": [], "min_distance_min_A": []},
            )
            payload["contact_occupancy"].append(float(row["contact_occupancy"]))
            payload["min_distance_mean_A"].append(float(row["min_distance_mean_A"]))
            payload["min_distance_min_A"].append(float(row["min_distance_min_A"]))

    rows = []
    for residue, payload in by_residue.items():
        occupancy = np.asarray(payload["contact_occupancy"], dtype=float)
        distance_mean = np.asarray(payload["min_distance_mean_A"], dtype=float)
        distance_min = np.asarray(payload["min_distance_min_A"], dtype=float)
        rows.append(
            {
                "protein_residue": residue,
                "contact_occupancy_mean": float(occupancy.mean()),
                "contact_occupancy_sd": float(occupancy.std(ddof=1) if occupancy.size > 1 else 0.0),
                "min_distance_mean_A": float(distance_mean.mean()),
                "min_distance_sd_A": float(distance_mean.std(ddof=1) if distance_mean.size > 1 else 0.0),
                "min_distance_min_A": float(distance_min.mean()),
                "n_replicas": int(occupancy.size),
            }
        )
    rows.sort(
        key=lambda row: (
            -float(row["contact_occupancy_mean"]),
            float(row["min_distance_mean_A"]),
            compact_residue_label(str(row["protein_residue"])),
        )
    )
    return rows


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
    rolling_window_fraction: float = 0.10,
    plot_selection: PlotSelectionConfig | None = None,
):
    min_n_frames = min(len(r["time_ns"]) for r in replica_results)
    common_time_ns = replica_results[0]["time_ns"][:min_n_frames]
    names = [r["replica_name"] for r in replica_results]
    protein = np.vstack([r["protein_rmsd_A"][:min_n_frames] for r in replica_results])
    ligand = np.vstack([r["ligand_rmsd_A"][:min_n_frames] for r in replica_results])
    save_csv(combined_dir / "rmsd_combined.csv", ["time_ns"] + [f"{name}_protein_backbone_rmsd_A" for name in names] + [f"{name}_ligand_heavy_rmsd_A" for name in names] + ["protein_backbone_rmsd_mean_A", "protein_backbone_rmsd_sd_A", "ligand_heavy_rmsd_mean_A", "ligand_heavy_rmsd_sd_A"], np.column_stack([common_time_ns, protein.T, ligand.T, protein.mean(axis=0), protein.std(axis=0, ddof=1) if protein.shape[0] > 1 else np.zeros(min_n_frames), ligand.mean(axis=0), ligand.std(axis=0, ddof=1) if ligand.shape[0] > 1 else np.zeros(min_n_frames)]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_combined_rmsd"):
        remove_figure_outputs(combined_dir / "rmsd_combined")
        replicate_labels = [compact_replica_name(name) for name in names]
        publication_replicate_series(
            common_time_ns,
            protein,
            "Protein backbone RMSD (Å)",
            combined_dir / "rmsd_replot_protein",
            style,
            title="Protein backbone RMSD across replicas",
            replicate_labels=replicate_labels,
            rolling_window_fraction=rolling_window_fraction,
        )
        publication_replicate_series(
            common_time_ns,
            ligand,
            "Ligand heavy-atom RMSD (Å)",
            combined_dir / "rmsd_replot_ligand",
            style,
            title="Ligand heavy-atom RMSD across replicas",
            replicate_labels=replicate_labels,
            rolling_window_fraction=rolling_window_fraction,
        )


def plot_combined_min_distance(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    rolling_window_fraction: float = 0.10,
    plot_selection: PlotSelectionConfig | None = None,
):
    time_ns, stack, names = _common_time_and_stack(replica_results, "global_min_distance_A")
    _write_stack_csv(combined_dir / "min_distance_combined.csv", time_ns, stack, [f"{n}_min_distance_A" for n in names], "mean_min_distance_A", "sd_min_distance_A")
    if plot_selection is None or plot_selection.enabled("basic_combined_min_distance"):
        publication_replicate_series(
            time_ns,
            stack,
            "Minimum ligand-protein distance (Å)",
            combined_dir / "min_distance_combined",
            style,
            title="Ligand-protein minimum heavy-atom distance",
            replicate_labels=[compact_replica_name(name) for name in names],
            rolling_window_fraction=rolling_window_fraction,
        )


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
        write_dict_csv(
            combined_dir / csv_name,
            rows,
            ["protein_residue", f"{out_field_name}_mean", f"{out_field_name}_sd", "n_replicas", "n_present_replicas"],
        )
        if rows and (plot_selection is None or plot_selection.enabled("basic_combined_occupancy_bars")):
            if out_field_name == "contact_occupancy":
                distance_rows = [row for row in _build_contact_distance_summary(replica_results) if float(row["contact_occupancy_mean"]) > 0.0]
                write_dict_csv(
                    combined_dir / "contact_occupancy_distance_summary.csv",
                    distance_rows,
                    [
                        "protein_residue",
                        "contact_occupancy_mean",
                        "contact_occupancy_sd",
                        "min_distance_mean_A",
                        "min_distance_sd_A",
                        "min_distance_min_A",
                        "n_replicas",
                    ],
                )
                top_rows = distance_rows[:top_n_contacts_plot]
                ranked_distance_lollipop(
                    [compact_residue_label(str(r["protein_residue"])) for r in top_rows],
                    [float(r["contact_occupancy_mean"]) for r in top_rows],
                    [float(r["contact_occupancy_sd"]) for r in top_rows],
                    [float(r["min_distance_mean_A"]) for r in top_rows],
                    combined_dir / base_name,
                    style,
                    title="Persistent contact hotspots\nCircle area = mean minimum distance (closer = larger)",
                    xlabel="Contact occupancy",
                    color=_interaction_color(style, out_field_name),
                )
            else:
                top_rows = rows[:top_n_contacts_plot]
                ordered_rows = list(top_rows)
                ranked_lollipop(
                    [compact_residue_label(r["protein_residue"]) for r in ordered_rows],
                    [r[f"{out_field_name}_mean"] for r in ordered_rows],
                    [r[f"{out_field_name}_sd"] for r in ordered_rows],
                    combined_dir / base_name,
                    style,
                    title=_interaction_rank_title(out_field_name),
                    xlabel=_interaction_rank_xlabel(out_field_name),
                    color=_interaction_color(style, out_field_name),
                )
    return combined_rows


def plot_key_contact_traces(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    top_n_key_distance_residues: int,
    contact_cutoff_A: float | None = None,
    rolling_window_fraction: float = 0.10,
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
    panel_payloads = []
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
        occupancy_value = next((float(row["contact_occupancy_mean"]) for row in contact_combined if row["protein_residue"] == label), np.nan)
        panel_payloads.append(
            {
                "label": label,
                "display_label": compact_residue_label(label),
                "arr": arr,
                "mean": mean,
                "sd": sd,
                "color": color,
                "occupancy": occupancy_value,
            }
        )
        rows.extend([[label, t, m, s] for t, m, s in zip(common_time_ns, mean, sd)])
    if plotting_enabled and panel_payloads:
        remove_figure_outputs(combined_dir / "key_contact_distance_traces")
        for payload in panel_payloads:
            axis_cap, clipped = _key_contact_axis_cap(payload["mean"], payload["sd"], contact_cutoff_A)
            plot_key_contact_distance_figure(
                common_time_ns,
                payload["mean"],
                payload["sd"],
                _key_contact_out_base(combined_dir, payload["label"]),
                style,
                display_label=payload["display_label"],
                color=payload["color"],
                occupancy=payload["occupancy"],
                contact_cutoff_A=contact_cutoff_A,
                rolling_window_fraction=rolling_window_fraction,
                arr2d=payload["arr"],
                axis_cap=axis_cap,
                clipped=clipped,
            )
    save_csv(combined_dir / "key_contact_distance_traces.csv", ["protein_residue", "time_ns", "mean_distance_A", "sd_distance_A"], rows)


def plot_combined_counts_and_shapes(
    replica_results,
    combined_dir: Path,
    style: PlotStyleConfig,
    rolling_window_fraction: float = 0.10,
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
        if base == "buried_surface_combined":
            finite = np.asarray(stack, dtype=float)
            finite = finite[np.isfinite(finite)]
            if finite.size == 0 or float(np.nanmax(np.abs(finite))) < 1.0:
                remove_figure_outputs(combined_dir / base)
                continue
        if plot_selection is None or plot_selection.enabled("basic_combined_counts_and_shapes"):
            publication_replicate_series(
                time_ns,
                stack,
                ylabel,
                combined_dir / base,
                style,
                title=title,
                replicate_labels=[compact_replica_name(name) for name in names],
                rolling_window_fraction=rolling_window_fraction,
            )

    time_ns, complex_stack, names = _common_time_and_stack(replica_results, "complex_sasa_A2")
    _, protein_stack, _ = _common_time_and_stack(replica_results, "protein_sasa_A2")
    _, ligand_stack, _ = _common_time_and_stack(replica_results, "ligand_sasa_A2")
    save_csv(combined_dir / "sasa_components_combined.csv", ["time_ns", "complex_sasa_mean_A2", "protein_sasa_mean_A2", "ligand_sasa_mean_A2"], np.column_stack([time_ns, complex_stack.mean(axis=0), protein_stack.mean(axis=0), ligand_stack.mean(axis=0)]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_combined_counts_and_shapes"):
        remove_figure_outputs(combined_dir / "sasa_components_combined")
        direct_label_line_series(
            time_ns,
            [complex_stack.mean(axis=0), protein_stack.mean(axis=0)],
            ["Complex SASA", "Protein SASA"],
            "Area (Å²)",
            combined_dir / "sasa_complex_protein_combined",
            style,
            title="Protein-complex SASA across replicas",
            colors=[style.protein_color, style.accent_color],
            rolling_window_fraction=rolling_window_fraction,
        )
        direct_label_line_series(
            time_ns,
            [ligand_stack.mean(axis=0)],
            ["Ligand SASA"],
            "Area (Å²)",
            combined_dir / "ligand_sasa_combined",
            style,
            title="Ligand SASA across replicas",
            colors=[style.ligand_color],
            rolling_window_fraction=rolling_window_fraction,
        )


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
        [compact_residue_label(label) for label in top_res],
        [compact_replica_name(r["replica_name"]) for r in replica_results],
        combined_dir / "contact_replicate_heatmap",
        style,
        title="Contact hotspot reproducibility",
        xlabel="Replica",
        ylabel="Residue",
        vmin=0.0,
        vmax=1.0,
        cmap=style.cmap_continuous,
        annotate=True,
        annotation_format="{:.0%}",
        annotation_min_abs=0.30,
        cbar_label="Contact occupancy",
        x_rotation=0.0,
    )

    metric_lookup = {
        "Contact": occupancy_rows.get("contact_occupancy", []),
        "H-bond": occupancy_rows.get("hbond_occupancy", []),
        "Salt bridge": occupancy_rows.get("salt_bridge_occupancy", []),
    }
    residue_summary: dict[str, dict[str, float]] = {}
    mean_key_lookup = {
        "Contact": "contact_occupancy_mean",
        "H-bond": "hbond_occupancy_mean",
        "Salt bridge": "salt_bridge_occupancy_mean",
    }
    for metric_name, rows in metric_lookup.items():
        mean_key = mean_key_lookup[metric_name]
        for row in rows[:top_n_contacts_plot]:
            residue_summary.setdefault(str(row["protein_residue"]), {})[metric_name] = float(row[mean_key])
    all_res = sorted(
        residue_summary,
        key=lambda label: (
            -sum(value > 0.08 for value in residue_summary[label].values()),
            -sum(residue_summary[label].values()),
            -max(residue_summary[label].values() or [0.0]),
            compact_residue_label(label),
        ),
    )[: max(top_n_contacts_plot, 12)]
    matrix = []
    for residue in all_res:
        vals = []
        for metric_name, rows in metric_lookup.items():
            mean_key = mean_key_lookup[metric_name]
            match = next((float(r[mean_key]) for r in rows if r["protein_residue"] == residue), 0.0)
            vals.append(match)
        matrix.append(vals)
    interaction_fingerprint_heatmap(
        np.asarray(matrix, dtype=float),
        [compact_residue_label(label) for label in all_res],
        list(metric_lookup.keys()),
        combined_dir / "interaction_fingerprint_heatmap",
        style,
        title="Interaction fingerprint by hotspot residue",
        xlabel="Interaction class",
        ylabel="Residue",
        interaction_colors=[
            _interaction_color(style, "contact_occupancy"),
            _interaction_color(style, "hbond_occupancy"),
            _interaction_color(style, "salt_bridge_occupancy"),
        ],
        annotation_min=0.22,
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
    contact_cutoff_A: float = 4.5,
    rolling_window_fraction: float = 0.10,
    plot_selection: PlotSelectionConfig | None = None,
):
    combined_dir = Path(analysis_root) / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    write_overall_summary(replica_results, combined_dir)
    plot_combined_rmsd(replica_results, combined_dir, style, rolling_window_fraction, plot_selection)
    plot_combined_min_distance(replica_results, combined_dir, style, rolling_window_fraction, plot_selection)
    plot_combined_rmsf(replica_results, combined_dir, style, plot_selection)
    occupancy_rows = plot_combined_occupancy_bars(replica_results, combined_dir, style, top_n_contacts_plot, plot_selection)
    plot_key_contact_traces(replica_results, combined_dir, style, top_n_key_distance_residues, contact_cutoff_A=contact_cutoff_A, rolling_window_fraction=rolling_window_fraction, plot_selection=plot_selection)
    plot_combined_counts_and_shapes(replica_results, combined_dir, style, rolling_window_fraction, plot_selection)
    plot_combined_dssp(replica_results, combined_dir, style, plot_selection)
    plot_interaction_heatmaps(replica_results, occupancy_rows, combined_dir, style, top_n_contacts_plot, plot_selection)
    plot_convergence(replica_results, combined_dir, style, plot_selection)
