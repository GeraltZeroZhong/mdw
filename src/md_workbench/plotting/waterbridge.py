from __future__ import annotations

from pathlib import Path
import numpy as np

from ..config import PlotSelectionConfig, PlotStyleConfig
from ..core import save_csv, write_dict_csv
from .bars import horizontal_bars
from .series import line_series, mean_sd_series


def plot_replica_waterbridge_counts(time_ns, waterbridge_count, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(
        time_ns,
        [waterbridge_count],
        ["Number of bridging waters"],
        "Number of bridging waters",
        out_dir / "waterbridge_counts_timeseries",
        style,
        title=f"{replica_name}: strict water-bridge count",
        colors=[style.distance_color],
    )


def plot_combined_waterbridge(
    replica_results,
    analysis_root: str,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    combined = Path(analysis_root) / "combined"
    combined.mkdir(parents=True, exist_ok=True)

    summary_rows = [r["summary"] for r in replica_results]
    write_dict_csv(
        combined / "summary_per_replica.csv",
        summary_rows,
        [
            "replica",
            "ligand_residue",
            "n_frames",
            "mean_n_bridging_waters",
            "max_n_bridging_waters",
            "n_bridge_residues",
            "n_bridge_waters",
            "n_protein_water_hbond_triplets",
            "n_ligand_water_hbond_triplets",
        ],
    )

    values = {}
    for result in replica_results:
        local = {row["protein_residue"]: row["waterbridge_occupancy"] for row in result["residue_rows"]}
        for key, value in local.items():
            values.setdefault(key, []).append(value)

    rows = []
    for key, arr in values.items():
        arr = np.asarray(arr, dtype=float)
        rows.append(
            {
                "protein_residue": key,
                "waterbridge_occupancy_mean": float(arr.mean()),
                "waterbridge_occupancy_sd": float(arr.std(ddof=1) if len(arr) > 1 else 0.0),
                "n_replicas": int(len(arr)),
            }
        )
    rows.sort(key=lambda x: x["waterbridge_occupancy_mean"], reverse=True)
    write_dict_csv(combined / "waterbridge_residue_occupancy_combined.csv", rows, ["protein_residue", "waterbridge_occupancy_mean", "waterbridge_occupancy_sd", "n_replicas"])
    if rows and (plot_selection is None or plot_selection.enabled("waterbridge_combined_occupancy")):
        top_rows = rows[:20]
        horizontal_bars(
            [r["protein_residue"] for r in reversed(top_rows)],
            [r["waterbridge_occupancy_mean"] for r in reversed(top_rows)],
            [r["waterbridge_occupancy_sd"] for r in reversed(top_rows)],
            combined / "waterbridge_residue_occupancy_top20",
            style,
            title="Top residue water-bridge occupancy",
            xlabel="Water-bridge occupancy",
        )

    min_n_frames = min(len(r["time_ns"]) for r in replica_results)
    common_time_ns = replica_results[0]["time_ns"][:min_n_frames]
    names = [r["replica_name"] for r in replica_results]
    count_stack = np.vstack([r["count"][:min_n_frames] for r in replica_results])
    save_csv(
        combined / "waterbridge_count_combined.csv",
        ["time_ns"] + [f"{name}_count" for name in names] + ["mean_count", "sd_count"],
        np.column_stack(
            [
                common_time_ns,
                count_stack.T,
                count_stack.mean(axis=0),
                count_stack.std(axis=0, ddof=1) if count_stack.shape[0] > 1 else np.zeros(min_n_frames),
            ]
        ).tolist(),
    )
    if plot_selection is None or plot_selection.enabled("waterbridge_combined_counts"):
        mean_sd_series(
            common_time_ns,
            count_stack,
            "Number of bridging waters",
            combined / "waterbridge_count_combined",
            style,
            title="Strict water-bridge count",
            individual_labels=names,
        )
