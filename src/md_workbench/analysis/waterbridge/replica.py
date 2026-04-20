from __future__ import annotations

from pathlib import Path
from typing import Callable
import numpy as np

from ...config import PlotSelectionConfig, PlotStyleConfig, WaterBridgeConfig
from ...core import check_input_file, get_time_ns_from_nframes, require_nonempty_file, write_dict_csv, save_csv
from ...plotting.waterbridge import plot_replica_waterbridge_counts
from .geometry import analyze_waterbridge_trajectory


ReplicaStepCallback = Callable[[int, int, str], None]


def _emit_replica_step(callback: ReplicaStepCallback | None, current: int, total: int, detail: str) -> None:
    if callback is None:
        return
    safe_total = max(int(total), 1)
    safe_current = min(max(int(current), 0), safe_total)
    callback(safe_current, safe_total, detail)


def process_waterbridge_replica(
    replica_dir,
    out_dir,
    cfg: WaterBridgeConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
    progress_callback: ReplicaStepCallback | None = None,
):
    step_total = 4
    _emit_replica_step(progress_callback, 0, step_total, "Loading trajectory for water-bridge analysis")
    top = check_input_file(Path(replica_dir) / cfg.top_name)
    dcd = require_nonempty_file(
        Path(replica_dir) / cfg.traj_name,
        label=f"轨迹文件 {cfg.traj_name}",
        empty_hint="这通常表示 MD 生产阶段没有写出任何轨迹帧，请确认 production_steps >= dcd_interval。",
    )
    try:
        _emit_replica_step(progress_callback, 1, step_total, "Computing protein-water-ligand bridge geometry")
        analysis = analyze_waterbridge_trajectory(
            dcd,
            top,
            cfg.hbond_distance_cutoff_nm,
            cfg.hbond_angle_cutoff_deg,
        )
    except OSError as exc:
        raise ValueError(f"{Path(replica_dir).name}: 无法读取轨迹文件 {dcd}") from exc
    time_ns = get_time_ns_from_nframes(analysis["n_frames"], cfg.timestep_ps, cfg.dcd_interval_steps)
    ligand_residue = analysis["ligand_residue"]
    protein_triplets = analysis["protein_triplets"]
    ligand_triplets = analysis["ligand_triplets"]
    residue_rows = analysis["residue_rows"]
    water_rows = analysis["water_rows"]
    waterbridge_count = analysis["waterbridge_count"]

    _emit_replica_step(progress_callback, 2, step_total, "Writing water-bridge occupancy tables")
    write_dict_csv(
        Path(out_dir) / "protein_water_hbond_triplets.csv",
        protein_triplets,
        ["type", "donor_atom", "hydrogen_atom", "acceptor_atom", "protein_residue", "water_residue", "occupancy", "mean_HA_distance_A", "min_HA_distance_A", "mean_DA_distance_A", "min_DA_distance_A"],
    )
    write_dict_csv(
        Path(out_dir) / "ligand_water_hbond_triplets.csv",
        ligand_triplets,
        ["type", "donor_atom", "hydrogen_atom", "acceptor_atom", "water_residue", "occupancy", "mean_HA_distance_A", "min_HA_distance_A", "mean_DA_distance_A", "min_DA_distance_A"],
    )
    write_dict_csv(Path(out_dir) / "waterbridge_residue_occupancy.csv", residue_rows, ["protein_residue", "waterbridge_occupancy"])
    write_dict_csv(Path(out_dir) / "waterbridge_water_residue_occupancy.csv", water_rows, ["water_residue", "waterbridge_occupancy"])
    save_csv(Path(out_dir) / "waterbridge_counts_timeseries.csv", ["time_ns", "n_bridging_waters"], np.column_stack([time_ns, waterbridge_count]).tolist())
    _emit_replica_step(progress_callback, 3, step_total, "Rendering water-bridge figures")
    if plot_selection is None or plot_selection.enabled("waterbridge_replica_counts"):
        plot_replica_waterbridge_counts(time_ns, waterbridge_count, out_dir, Path(replica_dir).name, style)
    _emit_replica_step(progress_callback, step_total, step_total, "Replica water-bridge analysis completed")

    return {
        "replica_name": Path(replica_dir).name,
        "time_ns": time_ns,
        "count": waterbridge_count,
        "residue_rows": residue_rows,
        "summary": {
            "replica": Path(replica_dir).name,
            "ligand_residue": f"chain{ligand_residue.chain.index}_{ligand_residue.name}{ligand_residue.resSeq}",
            "n_frames": int(analysis["n_frames"]),
            "mean_n_bridging_waters": float(np.mean(waterbridge_count)),
            "max_n_bridging_waters": int(np.max(waterbridge_count)),
            "n_bridge_residues": int(len(residue_rows)),
            "n_bridge_waters": int(len(water_rows)),
            "n_protein_water_hbond_triplets": int(len(protein_triplets)),
            "n_ligand_water_hbond_triplets": int(len(ligand_triplets)),
        },
    }
