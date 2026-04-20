from __future__ import annotations

from pathlib import Path
from typing import Callable

import mdtraj as md
import numpy as np

from ...config import BasicAnalysisConfig, PlotSelectionConfig, PlotStyleConfig
from ...core import (
    check_input_file,
    find_ligand_residue,
    get_time_ns_from_nframes,
    require_nonempty_file,
    write_dict_csv,
    save_csv,
)
from ...plotting.basic_replica import (
    plot_replica_counts,
    plot_replica_dssp,
    plot_replica_min_distance,
    plot_replica_pose_metrics,
    plot_replica_rg_sasa,
    plot_replica_rmsd,
    plot_replica_rmsf,
    plot_replica_snapshots,
    plot_replica_thermo,
)
from .contacts import compute_contact_occupancy, group_protein_heavy_atoms_by_residue
from .extended import (
    compute_counts_from_boolean_dict,
    compute_dssp_metrics,
    compute_ligand_pose_metrics,
    compute_ligand_torsions,
    compute_rg,
    compute_sasa_metrics,
    detect_ligand_rotatable_dihedrals,
)
from .hbonds import compute_hbond_occupancy
from .logs import parse_md_log
from .rms import compute_rmsd_and_rmsf
from .rms import build_average_reference
from .salt_bridges import compute_salt_bridges


ReplicaStepCallback = Callable[[int, int, str], None]


def _emit_replica_step(callback: ReplicaStepCallback | None, current: int, total: int, detail: str) -> None:
    if callback is None:
        return
    safe_total = max(int(total), 1)
    safe_current = min(max(int(current), 0), safe_total)
    callback(safe_current, safe_total, detail)


def _export_representative_snapshots(traj, protein_ca, ligand_heavy, out_dir: Path, replica_name: str, n_frames: int):
    snapshot_dir = out_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if traj.n_frames == 0:
        return []
    frame_ids = np.linspace(0, traj.n_frames - 1, num=max(1, min(n_frames, traj.n_frames)), dtype=int)
    entries = []
    for idx, frame in enumerate(frame_ids):
        title = f"frame {int(frame)}"
        pdb_path = snapshot_dir / f"{replica_name}_frame_{int(frame):05d}.pdb"
        traj[frame].save_pdb(str(pdb_path))
        entries.append(
            {
                "title": title,
                "pdb_path": str(pdb_path),
                "protein_xyz": traj.xyz[frame, protein_ca, :],
                "ligand_xyz": traj.xyz[frame, ligand_heavy, :],
            }
        )
    save_csv(snapshot_dir / "snapshot_manifest.csv", ["title", "pdb_path"], [[e["title"], e["pdb_path"]] for e in entries])
    return entries


def _write_timeseries_csv(out_dir: Path, name: str, time_ns, value_map: dict[str, np.ndarray]):
    header = ["time_ns"] + list(value_map.keys())
    rows = np.column_stack([time_ns] + [np.asarray(v, dtype=float) for v in value_map.values()]).tolist()
    save_csv(out_dir / f"{name}.csv", header, rows)


def _write_block_means(out_dir: Path, time_ns, metric_map: dict[str, np.ndarray], n_blocks: int):
    n_frames = len(time_ns)
    n_blocks = max(2, min(n_blocks, n_frames))
    edges = np.linspace(0, n_frames, n_blocks + 1, dtype=int)
    rows = []
    for block_idx in range(n_blocks):
        start, end = edges[block_idx], edges[block_idx + 1]
        if end <= start:
            continue
        t0 = float(time_ns[start])
        t1 = float(time_ns[end - 1])
        for metric_name, arr in metric_map.items():
            vals = np.asarray(arr, dtype=float)[start:end]
            rows.append({"block": block_idx + 1, "time_start_ns": t0, "time_end_ns": t1, "metric": metric_name, "mean": float(vals.mean()), "sd": float(vals.std(ddof=1) if len(vals) > 1 else 0.0)})
    write_dict_csv(out_dir / "convergence_block_means.csv", rows, ["block", "time_start_ns", "time_end_ns", "metric", "mean", "sd"])
    return rows


def process_replica(
    replica_dir,
    ligand_sdf_path,
    out_dir,
    cfg: BasicAnalysisConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
    progress_callback: ReplicaStepCallback | None = None,
):
    step_total = 10
    _emit_replica_step(progress_callback, 0, step_total, "Loading trajectory, topology, and MD log")
    top_path = check_input_file(Path(replica_dir) / "system_solvated.pdb")
    traj_path = require_nonempty_file(
        Path(replica_dir) / "trajectory.dcd",
        label="轨迹文件 trajectory.dcd",
        empty_hint=(
            "这通常表示 MD 生产阶段没有写出任何帧。"
            "如果这是本次工作流刚生成的结果，请确认 production_steps >= dcd_interval。"
        ),
    )
    log_path = require_nonempty_file(
        Path(replica_dir) / "md.log",
        label="日志文件 md.log",
        empty_hint=(
            "这通常表示 MD 生产阶段没有写出任何日志记录。"
            "如果这是本次工作流刚生成的结果，请确认 production_steps >= log_interval。"
        ),
    )

    try:
        traj = md.load(str(traj_path), top=str(top_path))
    except OSError as exc:
        raise ValueError(f"{Path(replica_dir).name}: 无法读取轨迹文件 {traj_path}") from exc
    ligand_residue = find_ligand_residue(traj.topology)
    ligand_all = [atom.index for atom in ligand_residue.atoms]
    ligand_heavy = [atom.index for atom in ligand_residue.atoms if atom.element is not None and atom.element.symbol != "H"]
    protein_bb = traj.topology.select("protein and backbone")
    protein_ca = traj.topology.select("protein and name CA")
    protein_all = traj.topology.select("protein")
    if len(protein_bb) == 0 or len(ligand_heavy) == 0:
        raise ValueError(f"{replica_dir}: missing protein backbone or ligand heavy atoms")

    _emit_replica_step(progress_callback, 1, step_total, "Aligning trajectory and preparing analysis reference")
    reference = build_average_reference(traj, protein_bb)
    traj.superpose(reference, 0, atom_indices=protein_bb, ref_atom_indices=protein_bb)
    reference.save_pdb(str(Path(out_dir) / "analysis_reference.pdb"))
    time_ns = get_time_ns_from_nframes(traj.n_frames, cfg.timestep_ps, cfg.dcd_interval_steps)

    _emit_replica_step(progress_callback, 2, step_total, "Computing RMSD and RMSF")
    protein_rmsd_A, ligand_rmsd_A, rmsf_rows = compute_rmsd_and_rmsf(traj, protein_bb, ligand_heavy, protein_ca, reference)
    save_csv(out_dir / "rmsd_timeseries.csv", ["time_ns", "protein_backbone_rmsd_A", "ligand_heavy_rmsd_A"], np.column_stack([time_ns, protein_rmsd_A, ligand_rmsd_A]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_replica_rmsd"):
        plot_replica_rmsd(time_ns, protein_rmsd_A, ligand_rmsd_A, out_dir, Path(replica_dir).name, style)

    _emit_replica_step(progress_callback, 3, step_total, "Computing contact occupancy and minimum distances")
    protein_heavy_atoms_by_residue = group_protein_heavy_atoms_by_residue(traj)
    contact_rows, residue_min_distance_curves, contact_bool, global_min_distance_A = compute_contact_occupancy(traj, ligand_heavy, protein_heavy_atoms_by_residue, cfg.contact_cutoff_nm)
    write_dict_csv(out_dir / "contact_occupancy.csv", contact_rows, ["protein_residue", "contact_occupancy", "min_distance_mean_A", "min_distance_min_A"])
    save_csv(out_dir / "min_distance_timeseries.csv", ["time_ns", "ligand_protein_min_heavy_distance_A"], np.column_stack([time_ns, global_min_distance_A]).tolist())
    if plot_selection is None or plot_selection.enabled("basic_replica_min_distance"):
        plot_replica_min_distance(time_ns, global_min_distance_A, out_dir, Path(replica_dir).name, style)
    contact_count = compute_counts_from_boolean_dict(contact_bool, traj.n_frames)

    _emit_replica_step(progress_callback, 4, step_total, "Computing hydrogen-bond occupancy")
    hbond_triplets, hbond_residues, hbond_residue_present, hbond_triplet_present = compute_hbond_occupancy(traj, set(ligand_all), cfg.hbond_distance_nm, cfg.hbond_angle_deg)
    write_dict_csv(out_dir / "hbond_triplets.csv", hbond_triplets, ["direction", "donor_atom", "hydrogen_atom", "acceptor_atom", "protein_residue", "occupancy", "mean_HA_distance_A", "min_HA_distance_A", "mean_DA_distance_A", "min_DA_distance_A"])
    write_dict_csv(out_dir / "hbond_residue_occupancy.csv", hbond_residues, ["protein_residue", "hbond_occupancy"])
    hbond_count = compute_counts_from_boolean_dict(hbond_triplet_present, traj.n_frames)

    _emit_replica_step(progress_callback, 5, step_total, "Computing salt bridges")
    salt_pair_rows, salt_residue_rows, salt_residue_present, salt_pair_present = compute_salt_bridges(traj, ligand_residue, ligand_sdf_path, cfg.salt_bridge_cutoff_nm)
    write_dict_csv(out_dir / "salt_bridge_atom_pairs.csv", salt_pair_rows, ["type", "protein_residue", "protein_atom", "ligand_atom", "occupancy", "mean_distance_A", "min_distance_A"])
    write_dict_csv(out_dir / "salt_bridge_residue_occupancy.csv", salt_residue_rows, ["protein_residue", "salt_bridge_occupancy"])
    salt_bridge_count = compute_counts_from_boolean_dict(salt_pair_present, traj.n_frames)

    _emit_replica_step(progress_callback, 6, step_total, "Computing shape, SASA, and DSSP metrics")
    write_dict_csv(out_dir / "rmsf_ca.csv", rmsf_rows, ["protein_residue", "resSeq", "resname", "rmsf_A"])
    if plot_selection is None or plot_selection.enabled("basic_replica_rmsf"):
        plot_replica_rmsf(rmsf_rows, out_dir, Path(replica_dir).name, style)

    rg_A = compute_rg(traj, protein_all)
    complex_sasa_A2, protein_sasa_A2, ligand_sasa_A2, buried_surface_A2 = compute_sasa_metrics(traj, protein_all, ligand_all, cfg.sasa_probe_radius_nm)
    dssp_raw, dssp_residue_labels, dssp_fractions, dssp_occupancy = compute_dssp_metrics(traj)
    save_csv(out_dir / "dssp_fractions_timeseries.csv", ["time_ns", "helix_fraction", "sheet_fraction", "coil_fraction"], np.column_stack([time_ns, dssp_fractions["Helix"], dssp_fractions["Sheet"], dssp_fractions["Coil"]]).tolist())
    write_dict_csv(out_dir / "dssp_residue_occupancy.csv", [{"protein_residue": lab, "helix_fraction": float(row[0]), "sheet_fraction": float(row[1]), "coil_fraction": float(row[2])} for lab, row in zip(dssp_residue_labels, dssp_occupancy)], ["protein_residue", "helix_fraction", "sheet_fraction", "coil_fraction"])

    com_distance_A, orientation_angle_deg = compute_ligand_pose_metrics(traj, ligand_heavy, protein_heavy_atoms_by_residue, reference, cfg.pose_cutoff_nm)
    torsion_quads = detect_ligand_rotatable_dihedrals(ligand_sdf_path, ligand_all)
    torsion_map = compute_ligand_torsions(traj, torsion_quads)
    torsion_csv_header = ["time_ns"] + list(torsion_map.keys())
    if torsion_map:
        save_csv(out_dir / "ligand_torsions.csv", torsion_csv_header, np.column_stack([time_ns] + [torsion_map[k] for k in torsion_map]).tolist())
    save_csv(out_dir / "ligand_pose_metrics.csv", ["time_ns", "ligand_pocket_com_distance_A", "ligand_orientation_angle_deg"], np.column_stack([time_ns, com_distance_A, orientation_angle_deg]).tolist())

    _emit_replica_step(progress_callback, 7, step_total, "Writing timeseries tables and replica figures")
    _write_timeseries_csv(out_dir, "interaction_counts_timeseries", time_ns, {
        "contact_count": contact_count,
        "hbond_count": hbond_count,
        "salt_bridge_count": salt_bridge_count,
    })
    _write_timeseries_csv(out_dir, "shape_timeseries", time_ns, {
        "radius_of_gyration_A": rg_A,
        "complex_sasa_A2": complex_sasa_A2,
        "protein_sasa_A2": protein_sasa_A2,
        "ligand_sasa_A2": ligand_sasa_A2,
        "buried_surface_A2": buried_surface_A2,
    })

    if plot_selection is None or plot_selection.enabled("basic_replica_counts"):
        plot_replica_counts(time_ns, {"Contacts": contact_count, "H-bonds": hbond_count, "Salt bridges": salt_bridge_count}, out_dir, Path(replica_dir).name, style)
    if plot_selection is None or plot_selection.enabled("basic_replica_rg_sasa"):
        plot_replica_rg_sasa(time_ns, rg_A, complex_sasa_A2, buried_surface_A2, out_dir, Path(replica_dir).name, style)
    if plot_selection is None or plot_selection.enabled("basic_replica_pose_metrics"):
        plot_replica_pose_metrics(time_ns, com_distance_A, orientation_angle_deg, torsion_map, out_dir, Path(replica_dir).name, style)
    if plot_selection is None or plot_selection.enabled("basic_replica_dssp"):
        plot_replica_dssp(time_ns, dssp_fractions, dssp_residue_labels, dssp_occupancy, out_dir, Path(replica_dir).name, style)

    if plot_selection is None or plot_selection.enabled("basic_replica_snapshots"):
        snapshot_entries = _export_representative_snapshots(traj, protein_ca, ligand_heavy, Path(out_dir), Path(replica_dir).name, cfg.snapshot_n_frames)
        plot_replica_snapshots(snapshot_entries, out_dir, Path(replica_dir).name, style)

    _emit_replica_step(progress_callback, 8, step_total, "Computing convergence blocks and thermodynamic summaries")
    block_rows = _write_block_means(out_dir, time_ns, {
        "protein_rmsd_A": protein_rmsd_A,
        "ligand_rmsd_A": ligand_rmsd_A,
        "min_distance_A": global_min_distance_A,
        "radius_of_gyration_A": rg_A,
        "buried_surface_A2": buried_surface_A2,
        "hbond_count": hbond_count,
    }, cfg.convergence_n_blocks)

    log_rows = parse_md_log(log_path)
    write_dict_csv(out_dir / "md_log_parsed.csv", log_rows, ["step", "time_ps", "potential_energy_kjmol", "kinetic_energy_kjmol", "total_energy_kjmol", "temperature_K", "volume_nm3", "density_gmL", "speed_nsd"])
    log_time_ns = np.asarray([r["time_ps"] / 1000.0 for r in log_rows], dtype=float)
    temperature = np.asarray([r["temperature_K"] for r in log_rows], dtype=float)
    density = np.asarray([r["density_gmL"] for r in log_rows], dtype=float)
    potential = np.asarray([r["potential_energy_kjmol"] for r in log_rows], dtype=float)
    total = np.asarray([r["total_energy_kjmol"] for r in log_rows], dtype=float)
    if plot_selection is None or plot_selection.enabled("basic_replica_thermo"):
        plot_replica_thermo(log_time_ns, temperature, density, potential, total, out_dir, Path(replica_dir).name, style)

    _emit_replica_step(progress_callback, 9, step_total, "Assembling replica summary")
    summary = {
        "replica": Path(replica_dir).name,
        "n_frames": int(traj.n_frames),
        "simulation_time_ns": float(time_ns[-1] if len(time_ns) > 0 else 0.0),
        "ligand_residue": f"chain{ligand_residue.chain.index}_{ligand_residue.name}{ligand_residue.resSeq}",
        "protein_backbone_rmsd_mean_A": float(np.mean(protein_rmsd_A)),
        "protein_backbone_rmsd_max_A": float(np.max(protein_rmsd_A)),
        "ligand_heavy_rmsd_mean_A": float(np.mean(ligand_rmsd_A)),
        "ligand_heavy_rmsd_max_A": float(np.max(ligand_rmsd_A)),
        "ligand_protein_min_distance_mean_A": float(np.mean(global_min_distance_A)),
        "ligand_protein_min_distance_min_A": float(np.min(global_min_distance_A)),
        "radius_of_gyration_mean_A": float(np.mean(rg_A)),
        "buried_surface_mean_A2": float(np.mean(buried_surface_A2)),
        "complex_sasa_mean_A2": float(np.mean(complex_sasa_A2)),
        "contact_count_mean": float(np.mean(contact_count)),
        "hbond_count_mean": float(np.mean(hbond_count)),
        "salt_bridge_count_mean": float(np.mean(salt_bridge_count)),
        "ligand_pocket_com_distance_mean_A": float(np.mean(com_distance_A)),
        "ligand_orientation_angle_mean_deg": float(np.mean(orientation_angle_deg)),
        "temperature_mean_K": float(np.mean(temperature)),
        "temperature_sd_K": float(np.std(temperature, ddof=1) if len(temperature) > 1 else 0.0),
        "density_mean_gmL": float(np.mean(density)),
        "density_sd_gmL": float(np.std(density, ddof=1) if len(density) > 1 else 0.0),
        "potential_energy_mean_kjmol": float(np.mean(potential)),
        "potential_energy_sd_kjmol": float(np.std(potential, ddof=1) if len(potential) > 1 else 0.0),
        "total_energy_mean_kjmol": float(np.mean(total)),
        "total_energy_sd_kjmol": float(np.std(total, ddof=1) if len(total) > 1 else 0.0),
    }
    _emit_replica_step(progress_callback, step_total, step_total, "Replica analysis completed")

    return {
        "replica_name": Path(replica_dir).name,
        "time_ns": time_ns,
        "protein_rmsd_A": protein_rmsd_A,
        "ligand_rmsd_A": ligand_rmsd_A,
        "global_min_distance_A": global_min_distance_A,
        "rmsf_rows": rmsf_rows,
        "contact_rows": contact_rows,
        "contact_curves_A": {key: value * 10.0 for key, value in residue_min_distance_curves.items()},
        "contact_bool": contact_bool,
        "contact_count": contact_count,
        "hbond_triplets": hbond_triplets,
        "hbond_residues": hbond_residues,
        "hbond_count": hbond_count,
        "salt_pair_rows": salt_pair_rows,
        "salt_residue_rows": salt_residue_rows,
        "salt_bridge_count": salt_bridge_count,
        "rg_A": rg_A,
        "complex_sasa_A2": complex_sasa_A2,
        "protein_sasa_A2": protein_sasa_A2,
        "ligand_sasa_A2": ligand_sasa_A2,
        "buried_surface_A2": buried_surface_A2,
        "dssp_residue_rows": [{"protein_residue": lab, "helix_fraction": float(row[0]), "sheet_fraction": float(row[1]), "coil_fraction": float(row[2])} for lab, row in zip(dssp_residue_labels, dssp_occupancy)],
        "dssp_fractions": dssp_fractions,
        "com_distance_A": com_distance_A,
        "orientation_angle_deg": orientation_angle_deg,
        "convergence_blocks": block_rows,
        "log_rows": log_rows,
        "summary": summary,
    }
