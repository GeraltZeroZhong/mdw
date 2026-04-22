from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from ...config import AdvancedAnalysisConfig, PlotSelectionConfig, PlotStyleConfig
from ...core import check_input_file, ensure_dir, require_nonempty_file, resolve_replica_dirs, save_csv, write_json
from ...core.progress import ProgressCallback, emit_progress
from ...plotting.advanced import (
    plot_cluster_population,
    plot_fes,
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
from ..basic.rms import build_average_reference
from .features import featurize_traj, pick_feature_atoms
from .msm import MaximumLikelihoodMSM, safe_msm_fit
from .projection import build_dtrajs_from_labels, run_clustering, run_pca, run_tica, save_projection_per_replica


_ION_OR_WATER_NAMES = {"HOH", "WAT", "SOL", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "CS", "RB", "LI", "F", "BR", "I"}
_COVALENT_RADII_NM = {
    "B": 0.084,
    "C": 0.076,
    "N": 0.071,
    "O": 0.066,
    "F": 0.057,
    "P": 0.107,
    "S": 0.105,
    "CL": 0.102,
    "BR": 0.120,
    "I": 0.139,
}


def _continuous_ca_segments(traj_frame: md.Trajectory, ca_indices: np.ndarray) -> list[list[int]]:
    if len(ca_indices) < 2:
        return []
    segments: list[list[int]] = []
    current: list[int] = []
    prev_chain = None
    prev_residue = None
    for rel_idx, atom_idx in enumerate(np.asarray(ca_indices, dtype=int)):
        atom = traj_frame.topology.atom(int(atom_idx))
        chain_id = int(atom.residue.chain.index)
        residue_id = int(atom.residue.index)
        if current and (chain_id != prev_chain or residue_id != prev_residue + 1):
            if len(current) >= 2:
                segments.append(current)
            current = []
        current.append(int(rel_idx))
        prev_chain = chain_id
        prev_residue = residue_id
    if len(current) >= 2:
        segments.append(current)
    return segments


def _protein_secondary_structure(traj_frame: md.Trajectory, ca_indices: np.ndarray) -> list[str]:
    try:
        dssp = md.compute_dssp(traj_frame, simplified=True)[0]
    except Exception:
        return ["C"] * len(ca_indices)
    codes: list[str] = []
    for atom_idx in np.asarray(ca_indices, dtype=int):
        residue = traj_frame.topology.atom(int(atom_idx)).residue
        code = str(dssp[residue.index]) if residue.index < len(dssp) else "C"
        codes.append("C" if code == "NA" else code)
    return codes


def _protein_contact_mask(traj_frame: md.Trajectory, ca_indices: np.ndarray, ligand_heavy: list[int], cutoff_nm: float) -> list[bool]:
    if len(ca_indices) == 0:
        return []
    if not ligand_heavy:
        return [False] * len(ca_indices)
    protein_xyz = traj_frame.xyz[0, np.asarray(ca_indices, dtype=int), :]
    ligand_xyz = traj_frame.xyz[0, np.asarray(ligand_heavy, dtype=int), :]
    distances = np.linalg.norm(protein_xyz[:, None, :] - ligand_xyz[None, :, :], axis=2)
    mask = np.nanmin(distances, axis=1) <= max(float(cutoff_nm), 0.35)
    if mask.any():
        padded = mask.copy()
        padded[1:] |= mask[:-1]
        padded[:-1] |= mask[1:]
        mask = padded
    return [bool(value) for value in mask]


def _infer_ligand_bonds(traj_frame: md.Trajectory, ligand_heavy: list[int]) -> list[list[int]]:
    rel_by_atom = {int(atom_idx): rel_idx for rel_idx, atom_idx in enumerate(ligand_heavy)}
    topo_pairs = []
    for atom_a, atom_b in traj_frame.topology.bonds:
        ia = int(atom_a.index)
        ib = int(atom_b.index)
        if ia in rel_by_atom and ib in rel_by_atom:
            topo_pairs.append(sorted((int(rel_by_atom[ia]), int(rel_by_atom[ib]))))
    if topo_pairs:
        return [list(pair) for pair in sorted({tuple(pair) for pair in topo_pairs})]

    coords = traj_frame.xyz[0, ligand_heavy, :]
    atoms = [traj_frame.topology.atom(int(atom_idx)) for atom_idx in ligand_heavy]
    inferred = []
    for i in range(len(ligand_heavy)):
        symbol_i = (atoms[i].element.symbol if atoms[i].element is not None else "C").upper()
        radius_i = _COVALENT_RADII_NM.get(symbol_i, 0.077)
        for j in range(i + 1, len(ligand_heavy)):
            symbol_j = (atoms[j].element.symbol if atoms[j].element is not None else "C").upper()
            radius_j = _COVALENT_RADII_NM.get(symbol_j, 0.077)
            cutoff = 1.23 * (radius_i + radius_j)
            distance = float(np.linalg.norm(coords[i] - coords[j]))
            if 0.055 <= distance <= cutoff:
                inferred.append([int(i), int(j)])
    return inferred


def _export_cluster_snapshots(replica_dirs, trajectories, labels, embed, frame_owner, n_clusters_use: int, centers, out_dir: Path, cfg: AdvancedAnalysisConfig):
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_entries = []
    rep_rows = []
    cluster_counts = np.bincount(np.asarray(labels, dtype=int), minlength=n_clusters_use)
    cluster_fractions = cluster_counts / max(1, len(labels))
    population_order = sorted(
        [cluster_id for cluster_id in range(n_clusters_use) if int(cluster_counts[cluster_id]) > 0],
        key=lambda cluster_id: (-int(cluster_counts[cluster_id]), int(cluster_id)),
    )
    n_show = max(1, min(cfg.representative_snapshot_clusters, len(population_order)))
    top_clusters = population_order[:n_show]
    rank_by_population = {cluster_id: rank + 1 for rank, cluster_id in enumerate(population_order)}
    cumulative_fraction = float(np.sum(cluster_fractions[top_clusters])) if top_clusters else 0.0

    for cluster_id in population_order:
        idx = np.where(labels == cluster_id)[0]
        if len(idx) == 0:
            continue
        local = embed[idx]
        center = centers[cluster_id]
        d2 = ((local - center) ** 2).sum(axis=1)
        best_global = idx[np.argmin(d2)]
        replica_name, frame_idx = frame_owner[best_global]
        replica_idx = [rd.name for rd in replica_dirs].index(replica_name)
        traj = trajectories[replica_idx]
        frame = traj[int(frame_idx)]
        pdb_path = out_dir / f"cluster_{cluster_id:02d}_{replica_name}_frame_{int(frame_idx):05d}.pdb"
        frame.save_pdb(str(pdb_path))
        distance_to_center = float(np.sqrt(np.min(d2)))
        rep_rows.append([
            int(cluster_id),
            int(rank_by_population[cluster_id]),
            int(cluster_counts[cluster_id]),
            float(cluster_fractions[cluster_id]),
            replica_name,
            int(frame_idx),
            distance_to_center,
            bool(cluster_id in top_clusters),
            str(pdb_path),
        ])
        protein_ca = frame.topology.select("protein and name CA")
        ligand_res = next(res for res in frame.topology.residues if (not res.is_protein) and res.name.upper() not in _ION_OR_WATER_NAMES)
        ligand_heavy = [a.index for a in ligand_res.atoms if a.element is not None and a.element.symbol != "H"]
        protein_segments = _continuous_ca_segments(frame, protein_ca)
        protein_ss = _protein_secondary_structure(frame, protein_ca)
        protein_contact_mask = _protein_contact_mask(frame, protein_ca, ligand_heavy, cfg.pocket_ca_cutoff_nm)
        ligand_bonds = _infer_ligand_bonds(frame, ligand_heavy)
        snapshot_entries.append({
            "title": f"State {cluster_id}\n{replica_name} frame {frame_idx}",
            "cluster_id": int(cluster_id),
            "population_rank": int(rank_by_population[cluster_id]),
            "n_frames": int(cluster_counts[cluster_id]),
            "population_fraction": float(cluster_fractions[cluster_id]),
            "replica_name": replica_name,
            "frame_idx": int(frame_idx),
            "distance_to_center": distance_to_center,
            "n_clusters_total": int(len(population_order)),
            "n_clusters_shown": int(n_show),
            "cumulative_population_fraction": cumulative_fraction,
            "protein_segments": protein_segments,
            "protein_secondary_structure": protein_ss,
            "protein_contact_mask": protein_contact_mask,
            "ligand_bonds": ligand_bonds,
            "protein_xyz": frame.xyz[0, protein_ca, :],
            "ligand_xyz": frame.xyz[0, ligand_heavy, :],
        })
    save_csv(
        out_dir / "representative_frames.csv",
        [
            "cluster",
            "rank_by_population",
            "n_frames",
            "fraction",
            "replica",
            "frame",
            "distance_to_center",
            "selected_for_figure",
            "pdb_path",
        ],
        rep_rows,
    )
    return snapshot_entries


def _select_snapshot_entries(snapshot_entries, n_show: int):
    if not snapshot_entries:
        return []
    n_use = max(1, min(int(n_show), len(snapshot_entries)))
    selected = [dict(entry) for entry in snapshot_entries[:n_use]]
    cumulative_fraction = float(sum(float(entry["population_fraction"]) for entry in selected))
    for entry in selected:
        entry["n_clusters_shown"] = int(n_use)
        entry["cumulative_population_fraction"] = cumulative_fraction
    return selected


def run_advanced_analysis(
    cfg: AdvancedAnalysisConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
    progress_callback: ProgressCallback | None = None,
):
    step_total = 8
    emit_progress(progress_callback, 0, step_total, "advanced_analysis", "Resolving replica trajectories")
    replica_dirs = resolve_replica_dirs(cfg.replica_root, cfg.replica_glob)
    if not replica_dirs:
        raise FileNotFoundError(f"No replica directories found: {cfg.replica_glob}")

    for sub in ["pca", "tica", "clustering", "msm", "per_replica_assignments", "snapshots"]:
        ensure_dir(Path(cfg.analysis_root) / sub)

    trajectories = []
    for replica_dir in replica_dirs:
        top = check_input_file(replica_dir / cfg.top_name)
        dcd = require_nonempty_file(
            replica_dir / cfg.traj_name,
            label=f"轨迹文件 {cfg.traj_name}",
            empty_hint="这通常表示 MD 生产阶段没有写出任何轨迹帧，请确认 production_steps >= dcd_interval。",
        )
        try:
            traj = md.load(str(dcd), top=str(top))
        except OSError as exc:
            raise ValueError(f"{replica_dir.name}: 无法读取轨迹文件 {dcd}") from exc
        trajectories.append(traj)

    emit_progress(progress_callback, 1, step_total, "advanced_analysis", "Aligning trajectories to the reference structure")
    align_atoms = trajectories[0].topology.select(cfg.align_selection)
    if len(align_atoms) == 0:
        raise ValueError(f"{replica_dirs[0].name}: no alignment atoms found for {cfg.align_selection}")
    reference = build_average_reference(trajectories[0], align_atoms)
    reference.save_pdb(str(Path(cfg.analysis_root) / "alignment_reference.pdb"))
    for replica_dir, traj in zip(replica_dirs, trajectories):
        traj_align_atoms = traj.topology.select(cfg.align_selection)
        if len(traj_align_atoms) != len(align_atoms):
            raise ValueError(
                f"{replica_dir.name}: alignment selection '{cfg.align_selection}' does not match the reference topology."
            )
        traj.superpose(reference, 0, atom_indices=traj_align_atoms, ref_atom_indices=align_atoms)

    emit_progress(progress_callback, 2, step_total, "advanced_analysis", "Selecting features and featurizing trajectories")
    feature_pairs, metadata = pick_feature_atoms(trajectories, cfg)
    write_json(Path(cfg.analysis_root) / "feature_metadata.json", metadata)

    feature_list = []
    frame_owner = []
    for idx, traj in enumerate(trajectories):
        X = featurize_traj(traj, feature_pairs)
        feature_list.append(X)
        frame_owner.extend([(replica_dirs[idx].name, frame_idx) for frame_idx in range(traj.n_frames)])
    X_all = np.vstack(feature_list)
    if X_all.shape[0] < 3:
        raise ValueError("Total frame count is too small for advanced analysis.")

    plot_pca = plot_selection is None or plot_selection.enabled("advanced_pca")
    plot_tica = plot_selection is None or plot_selection.enabled("advanced_tica")
    plot_clustering = plot_selection is None or plot_selection.enabled("advanced_clustering")
    plot_snapshots = plot_selection is None or plot_selection.enabled("advanced_snapshots")
    plot_msm = plot_selection is None or plot_selection.enabled("advanced_msm")

    emit_progress(progress_callback, 3, step_total, "advanced_analysis", "Running PCA and exporting projections")
    scaler, _X_scaled, pca_model, X_pca = run_pca(X_all, cfg)
    pca_evr = np.asarray(pca_model.explained_variance_ratio_, dtype=float)
    save_csv(Path(cfg.analysis_root) / "pca" / "explained_variance_ratio.csv", ["component", "explained_variance_ratio", "cumulative_explained_variance_ratio"], [[i + 1, float(v), float(np.cumsum(pca_evr)[i])] for i, v in enumerate(pca_evr)])
    if plot_pca:
        plot_line_profile(np.arange(1, len(pca_evr) + 1), pca_evr, Path(cfg.analysis_root) / "pca" / "explained_variance_ratio", "PC index", "Explained variance ratio", "PCA explained variance", style, color=style.accent_color)
        plot_fes(X_pca[:, 0], X_pca[:, 1], Path(cfg.analysis_root) / "pca" / "free_energy_landscape_pc1_pc2", "PCA free-energy landscape", "PC1", "PC2", cfg.n_bins, cfg.temperature_K, cfg.kB_kcal_mol_K, style)
        scatter_by_replica(X_pca[:, :2], replica_dirs, feature_list, Path(cfg.analysis_root) / "pca" / "pc1_pc2_scatter", "PC1", "PC2", "PCA projection by replica", style)
    save_projection_per_replica(replica_dirs, feature_list, X_pca, "pc", Path(cfg.analysis_root) / "per_replica_assignments", cfg.max_saved_components)

    emit_progress(progress_callback, 4, step_total, "advanced_analysis", "Running tICA and exporting slow modes")
    scaled_feature_list = [scaler.transform(X) for X in feature_list]
    safe_tica_lag, tica_model, X_tica = run_tica(scaled_feature_list, cfg)
    save_projection_per_replica(replica_dirs, feature_list, X_tica, "tic", Path(cfg.analysis_root) / "per_replica_assignments", cfg.max_saved_components)
    try:
        singular_vals = np.asarray(getattr(tica_model, "singular_values", None))
        if singular_vals.ndim == 1 and singular_vals.size > 0:
            save_csv(Path(cfg.analysis_root) / "tica" / "singular_values.csv", ["component", "singular_value"], [[i + 1, float(v)] for i, v in enumerate(singular_vals)])
            if plot_tica:
                plot_line_profile(np.arange(1, len(singular_vals) + 1), singular_vals, Path(cfg.analysis_root) / "tica" / "singular_values", "tIC index", "Singular value", "tICA singular values", style, color=style.accent_color)
    except Exception:
        pass
    if plot_tica:
        plot_fes(X_tica[:, 0], X_tica[:, 1], Path(cfg.analysis_root) / "tica" / "free_energy_landscape_tic1_tic2", "tICA free-energy landscape", "tIC1", "tIC2", cfg.n_bins, cfg.temperature_K, cfg.kB_kcal_mol_K, style)
        scatter_by_replica(X_tica[:, :2], replica_dirs, feature_list, Path(cfg.analysis_root) / "tica" / "tic1_tic2_scatter", "tIC1", "tIC2", "tICA projection by replica", style)

    emit_progress(progress_callback, 5, step_total, "advanced_analysis", "Running clustering and assigning frames to states")
    embed, n_clusters_use, kmeans, labels = run_clustering(X_tica, cfg)
    save_csv(Path(cfg.analysis_root) / "clustering" / "cluster_centers.csv", ["cluster"] + [f"coord{i+1}" for i in range(embed.shape[1])], [[i, *kmeans.cluster_centers_[i].tolist()] for i in range(n_clusters_use)])
    dtrajs = build_dtrajs_from_labels(replica_dirs, feature_list, labels)
    cluster_pop_rows = []
    pop_matrix = np.zeros((len(replica_dirs), n_clusters_use), dtype=float)
    for ridx, (replica_dir, traj_labels) in enumerate(zip(replica_dirs, dtrajs)):
        save_csv(Path(cfg.analysis_root) / "per_replica_assignments" / f"{replica_dir.name}_cluster_assignment.csv", ["frame", "cluster"], [[i, int(c)] for i, c in enumerate(traj_labels)])
        uniq, cnt = np.unique(traj_labels, return_counts=True)
        for u, c in zip(uniq, cnt):
            frac = float(c / len(traj_labels))
            cluster_pop_rows.append([replica_dir.name, int(u), int(c), frac])
            pop_matrix[ridx, int(u)] = frac
    save_csv(Path(cfg.analysis_root) / "clustering" / "cluster_population_per_replica.csv", ["replica", "cluster", "n_frames", "fraction"], cluster_pop_rows)
    uniq_all, cnt_all = np.unique(labels, return_counts=True)
    frac_all = cnt_all / len(labels)
    save_csv(Path(cfg.analysis_root) / "clustering" / "cluster_population_overall.csv", ["cluster", "n_frames", "fraction"], [[int(u), int(c), float(f)] for u, c, f in zip(uniq_all, cnt_all, frac_all)])
    if plot_clustering:
        plot_cluster_population(uniq_all, frac_all, Path(cfg.analysis_root) / "clustering" / "cluster_population_overall", style)
        scatter_clusters(embed, labels, kmeans.cluster_centers_, Path(cfg.analysis_root) / "clustering" / "clusters_tic1_tic2", style)
        plot_state_population_heatmap([rd.name for rd in replica_dirs], list(range(n_clusters_use)), pop_matrix, Path(cfg.analysis_root) / "clustering" / "state_population_by_replica", style)

    if plot_snapshots:
        emit_progress(progress_callback, 6, step_total, "advanced_analysis", "Exporting representative structural snapshots")
        snapshot_entries_all = _export_cluster_snapshots(replica_dirs, trajectories, labels, embed, frame_owner, n_clusters_use, kmeans.cluster_centers_, Path(cfg.analysis_root) / "snapshots", cfg)
        snapshot_entries = _select_snapshot_entries(snapshot_entries_all, cfg.representative_snapshot_clusters)
        plot_snapshot_grid(snapshot_entries, Path(cfg.analysis_root) / "snapshots" / "representative_state_snapshots", style, title="Representative structures of dominant states")
        for extra_count in (6, 8):
            if extra_count == int(cfg.representative_snapshot_clusters) or len(snapshot_entries_all) < extra_count:
                continue
            variant_entries = _select_snapshot_entries(snapshot_entries_all, extra_count)
            plot_snapshot_grid(
                variant_entries,
                Path(cfg.analysis_root) / "snapshots" / f"top_{extra_count}_states" / "representative_state_snapshots",
                style,
                title="Representative structures of dominant states",
            )

    emit_progress(progress_callback, 7, step_total, "advanced_analysis", "Building exploratory MSM outputs")
    msm_notes = {"warning": "MSM output from short trajectories is exploratory.", "used_safe_tica_lag_frames": safe_tica_lag}
    try:
        msm, safe_msm_lag, traj_lengths = safe_msm_fit(dtrajs, cfg.msm_lag_frames)
        msm_notes["input_discrete_trajectory_lengths"] = traj_lengths
        msm_notes["requested_msm_lag_frames"] = cfg.msm_lag_frames
        msm_notes["used_safe_msm_lag_frames"] = safe_msm_lag
        T = np.asarray(msm.transition_matrix)
        save_csv(Path(cfg.analysis_root) / "msm" / "transition_matrix.csv", ["from\\to"] + [f"state_{i}" for i in range(T.shape[1])], [[f"state_{i}", *T[i].tolist()] for i in range(T.shape[0])])
        pi = np.asarray(msm.stationary_distribution)
        save_csv(Path(cfg.analysis_root) / "msm" / "stationary_distribution.csv", ["state", "stationary_probability"], [[i, float(v)] for i, v in enumerate(pi)])
        flux = pi[:, None] * T
        save_csv(Path(cfg.analysis_root) / "msm" / "equilibrium_transition_flux.csv", ["from\\to"] + [f"state_{i}" for i in range(flux.shape[1])], [[f"state_{i}", *flux[i].tolist()] for i in range(flux.shape[0])])
        msm_notes["stationary_distribution_sum"] = float(np.sum(pi))
        msm_notes["transition_matrix_row_sums"] = [float(v) for v in np.sum(T, axis=1)]
        msm_notes["state_network_edge_definition"] = "Edges are filtered and scaled by equilibrium transition flux pi_i * P_ij; edge labels show transition probabilities P_ij."
        if plot_msm:
            plot_stationary_distribution(np.arange(len(pi)), pi, Path(cfg.analysis_root) / "msm" / "stationary_distribution", style)
            plot_transition_matrix_heatmap(T, Path(cfg.analysis_root) / "msm" / "transition_matrix_heatmap", style)
            msm_notes["state_network_plot"] = plot_state_network(T, pi, Path(cfg.analysis_root) / "msm" / "state_network", style, cfg.state_network_threshold)
        try:
            its = np.asarray(msm.timescales())
            if its.ndim == 1 and its.size > 0:
                save_csv(Path(cfg.analysis_root) / "msm" / "implied_timescales_single_lag.csv", ["index", "timescale_frames"], [[i + 1, float(v)] for i, v in enumerate(its)])
                if plot_msm:
                    plot_line_profile(np.arange(1, len(its) + 1), its, Path(cfg.analysis_root) / "msm" / "implied_timescales_single_lag", "Process index", "Timescale (frames)", f"MSM implied timescales at lag = {safe_msm_lag} frames", style, color=style.accent_color)
        except Exception as exc:
            msm_notes["timescales_error"] = str(exc)

        lag_scan_rows = []
        for lag in [5, 10, 20, 40, 80]:
            try:
                if MaximumLikelihoodMSM is None:
                    break
                safe_lag = min(lag, max(1, min(traj_lengths) - 1))
                model = MaximumLikelihoodMSM(reversible=True).fit(dtrajs, lagtime=safe_lag).fetch_model()
                vals = np.asarray(model.timescales())[:5]
                for i, v in enumerate(vals):
                    lag_scan_rows.append([lag, i + 1, float(v)])
            except Exception:
                continue
        if lag_scan_rows:
            save_csv(Path(cfg.analysis_root) / "msm" / "implied_timescales_lag_scan.csv", ["lag_frames", "process_index", "timescale_frames"], lag_scan_rows)
            if plot_msm:
                plot_lag_scan(lag_scan_rows, Path(cfg.analysis_root) / "msm" / "implied_timescales_lag_scan", style)
    except Exception as exc:
        msm_notes["msm_error"] = str(exc)
    write_json(Path(cfg.analysis_root) / "msm" / "msm_notes.json", msm_notes)
    emit_progress(progress_callback, step_total, step_total, "advanced_analysis", "Advanced analysis completed")
    return str(Path(cfg.analysis_root).resolve())
