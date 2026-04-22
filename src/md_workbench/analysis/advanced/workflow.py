from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np

from ...config import AdvancedAnalysisConfig, PlotSelectionConfig, PlotStyleConfig
from ...core import check_input_file, ensure_dir, require_nonempty_file, resolve_replica_dirs, save_csv, write_json
from ...core.progress import ProgressCallback, emit_progress
from ...plotting.advanced import (
    plot_chapman_kolmogorov_test,
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
from .msm import fit_msm_on_active_symbols, fit_population_connected_msm
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


def _msm_state_labels(active_symbols: np.ndarray) -> list[str]:
    symbols = np.asarray(active_symbols, dtype=int).ravel()
    return [f"S{idx} [C{int(symbol)}]" for idx, symbol in enumerate(symbols)]


def _dense_real_array(matrix) -> np.ndarray:
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    return np.asarray(np.real_if_close(matrix), dtype=float)


def _save_labeled_square_csv(path: Path, labels: list[str], matrix) -> None:
    arr = _dense_real_array(matrix)
    save_csv(path, ["from\\to"] + labels, [[labels[i], *arr[i].tolist()] for i in range(arr.shape[0])])


def _mfpt_matrix(msm) -> np.ndarray:
    n_states = int(msm.n_states)
    matrix = np.full((n_states, n_states), np.nan, dtype=float)
    for i in range(n_states):
        matrix[i, i] = 0.0
        for j in range(n_states):
            if i == j:
                continue
            try:
                matrix[i, j] = float(np.real_if_close(msm.mfpt(i, j)))
            except Exception:
                matrix[i, j] = np.nan
    return matrix


def _lag_candidates(base_lag: int, max_segment_length: int) -> list[int]:
    candidates = {5, 10, 20, 40, 80, int(base_lag), int(base_lag) * 2, int(base_lag) * 4}
    return sorted(int(lag) for lag in candidates if 1 <= int(lag) < max_segment_length)


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
        msm_fit = fit_population_connected_msm(dtrajs, cfg.msm_lag_frames)
        msm = msm_fit.model
        safe_msm_lag = int(msm_fit.safe_lag)
        traj_lengths = msm_fit.traj_lengths
        active_symbols = np.asarray(msm_fit.active_symbols, dtype=int)
        state_labels = _msm_state_labels(active_symbols)
        state_csv_labels = [f"msm_{idx}_cluster_{int(symbol)}" for idx, symbol in enumerate(active_symbols)]
        histogram = np.asarray(getattr(msm_fit.counts_full, "state_histogram", []), dtype=int)
        selected_component = next((item for item in msm_fit.connected_sets if item.symbols == tuple(int(v) for v in active_symbols.tolist())), None)
        connected_component_rows = [
            [
                int(item.rank),
                " ".join(str(symbol) for symbol in item.symbols),
                int(item.n_states),
                int(item.frame_count),
                float(item.frame_fraction),
                bool(item.symbols == tuple(int(v) for v in active_symbols.tolist())),
            ]
            for item in msm_fit.connected_sets
        ]
        save_csv(
            Path(cfg.analysis_root) / "msm" / "connected_component_summary.csv",
            ["rank", "source_clusters", "n_states", "frame_count", "frame_fraction", "selected_for_msm"],
            connected_component_rows,
        )
        active_state_rows = []
        for msm_state, symbol in enumerate(active_symbols):
            frame_count = int(histogram[int(symbol)]) if histogram.size else 0
            active_state_rows.append(
                [
                    int(msm_state),
                    int(symbol),
                    frame_count,
                    float(frame_count / max(msm_fit.total_frame_count, 1)),
                    float(frame_count / max(msm_fit.active_frame_count, 1)),
                ]
            )
        save_csv(
            Path(cfg.analysis_root) / "msm" / "active_state_mapping.csv",
            ["msm_state", "source_cluster", "frame_count", "frame_fraction_total", "frame_fraction_within_selected_component"],
            active_state_rows,
        )
        _save_labeled_square_csv(Path(cfg.analysis_root) / "msm" / "transition_count_matrix.csv", state_csv_labels, msm_fit.counts_active.count_matrix)

        msm_notes["input_discrete_trajectory_lengths"] = traj_lengths
        msm_notes["requested_msm_lag_frames"] = cfg.msm_lag_frames
        msm_notes["used_safe_msm_lag_frames"] = safe_msm_lag
        msm_notes["component_selection_method"] = "most_populated_strongly_connected_component_at_requested_lag"
        msm_notes["selected_connected_component_rank"] = int(selected_component.rank) if selected_component is not None else None
        msm_notes["selected_connected_component_symbols"] = [int(value) for value in active_symbols.tolist()]
        msm_notes["connected_component_count"] = len(msm_fit.connected_sets)
        msm_notes["connected_components"] = [
            {
                "rank": int(item.rank),
                "symbols": [int(value) for value in item.symbols],
                "n_states": int(item.n_states),
                "frame_count": int(item.frame_count),
                "frame_fraction": float(item.frame_fraction),
                "selected_for_msm": bool(item.symbols == tuple(int(v) for v in active_symbols.tolist())),
            }
            for item in msm_fit.connected_sets
        ]
        coverage_fraction = float(msm_fit.active_frame_count / max(msm_fit.total_frame_count, 1))
        msm_notes["selected_component_frame_count"] = int(msm_fit.active_frame_count)
        msm_notes["selected_component_frame_fraction"] = coverage_fraction
        warnings = []
        if len(msm_fit.connected_sets) > 1:
            warnings.append("Discrete state space is disconnected at the chosen MSM lag; kinetics are reported only for the selected strongly connected component.")
        if coverage_fraction < 0.5:
            warnings.append("Selected MSM component covers less than half of all clustered frames, so kinetics should be interpreted as exploratory.")
        if warnings:
            msm_notes["warnings"] = warnings

        T = _dense_real_array(msm.transition_matrix)
        _save_labeled_square_csv(Path(cfg.analysis_root) / "msm" / "transition_matrix.csv", state_csv_labels, T)
        pi = np.asarray(np.real_if_close(msm.stationary_distribution), dtype=float)
        save_csv(
            Path(cfg.analysis_root) / "msm" / "stationary_distribution.csv",
            ["msm_state", "source_cluster", "stationary_probability", "frame_count", "frame_fraction_total"],
            [
                [
                    int(i),
                    int(active_symbols[i]),
                    float(pi[i]),
                    int(histogram[int(active_symbols[i])]) if histogram.size else 0,
                    float((int(histogram[int(active_symbols[i])]) if histogram.size else 0) / max(msm_fit.total_frame_count, 1)),
                ]
                for i in range(len(pi))
            ],
        )
        flux = pi[:, None] * T
        _save_labeled_square_csv(Path(cfg.analysis_root) / "msm" / "equilibrium_transition_flux.csv", state_csv_labels, flux)
        mfpt = _mfpt_matrix(msm)
        _save_labeled_square_csv(Path(cfg.analysis_root) / "msm" / "mean_first_passage_times.csv", state_csv_labels, mfpt)
        stationary_entropy_bits = float(-np.sum(pi[pi > 0] * np.log2(pi[pi > 0]))) if np.any(pi > 0) else 0.0
        effective_states = float(1.0 / np.sum(pi ** 2)) if np.any(pi > 0) else 0.0
        detailed_balance_residual = flux - flux.T
        max_db_residual = float(np.nanmax(np.abs(detailed_balance_residual))) if detailed_balance_residual.size else 0.0
        msm_notes["stationary_distribution_sum"] = float(np.sum(pi))
        msm_notes["transition_matrix_row_sums"] = [float(v) for v in np.sum(T, axis=1)]
        msm_notes["stationary_entropy_bits"] = stationary_entropy_bits
        msm_notes["effective_state_count"] = effective_states
        msm_notes["max_detailed_balance_residual"] = max_db_residual
        msm_notes["state_network_edge_definition"] = "Edges are filtered and scaled by equilibrium transition flux pi_i * P_ij; edge labels show transition probabilities P_ij."
        stationary_summary_lines = [
            f"Active clusters: {', '.join(str(int(value)) for value in active_symbols)}",
            f"Coverage: {coverage_fraction:.1%}",
            f"MSM lag: {safe_msm_lag} frames",
            f"Effective states: {effective_states:.2f}",
        ]
        if plot_msm:
            plot_stationary_distribution(
                np.arange(len(pi)),
                pi,
                Path(cfg.analysis_root) / "msm" / "stationary_distribution",
                style,
                state_labels=state_labels,
                summary_lines=stationary_summary_lines,
            )
            plot_transition_matrix_heatmap(
                T,
                Path(cfg.analysis_root) / "msm" / "transition_matrix_heatmap",
                style,
                state_labels=state_labels,
                flux_matrix=flux,
            )
            msm_notes["state_network_plot"] = plot_state_network(
                T,
                pi,
                Path(cfg.analysis_root) / "msm" / "state_network",
                style,
                cfg.state_network_threshold,
                state_labels=state_labels,
                mfpt_matrix=mfpt,
                summary_lines=[
                    f"Coverage {coverage_fraction:.1%}",
                    f"max |π_iP_ij-π_jP_ji| = {max_db_residual:.2e}",
                ],
            )
        try:
            its = np.asarray(np.real_if_close(msm.timescales()), dtype=float)
            if its.ndim == 1 and its.size > 0:
                save_csv(Path(cfg.analysis_root) / "msm" / "implied_timescales_single_lag.csv", ["index", "timescale_frames"], [[i + 1, float(v)] for i, v in enumerate(its)])
                if plot_msm:
                    plot_line_profile(np.arange(1, len(its) + 1), its, Path(cfg.analysis_root) / "msm" / "implied_timescales_single_lag", "Process index", "Timescale (frames)", f"MSM implied timescales at lag = {safe_msm_lag} frames", style, color=style.accent_color)
        except Exception as exc:
            msm_notes["timescales_error"] = str(exc)

        lag_scan_rows = []
        lag_scan_diag_rows = []
        lag_models = {}
        max_segment_length = max((len(segment) for segment in msm_fit.active_dtrajs), default=0)
        for lag in _lag_candidates(safe_msm_lag, max_segment_length):
            try:
                model_lag, active_segments, _counts_full, _counts_active = fit_msm_on_active_symbols(dtrajs, active_symbols, lag)
                lag_models[int(lag)] = model_lag
                usable_segments = [segment for segment in active_segments if len(segment) > int(lag)]
                usable_frames = int(sum(len(segment) for segment in usable_segments))
                lag_scan_diag_rows.append([int(lag), int(lag), int(len(usable_segments)), usable_frames, int(msm_fit.active_frame_count)])
                vals = np.asarray(np.real_if_close(model_lag.timescales()), dtype=float)[:5]
                for i, v in enumerate(vals):
                    lag_scan_rows.append([lag, i + 1, float(v)])
            except Exception as exc:
                lag_scan_diag_rows.append([int(lag), int(lag), 0, 0, int(msm_fit.active_frame_count)])
                msm_notes.setdefault("lag_scan_errors", {})[str(int(lag))] = str(exc)
                continue
        if lag_scan_rows:
            save_csv(Path(cfg.analysis_root) / "msm" / "implied_timescales_lag_scan.csv", ["lag_frames", "process_index", "timescale_frames"], lag_scan_rows)
            save_csv(
                Path(cfg.analysis_root) / "msm" / "lag_scan_diagnostics.csv",
                ["lag_frames", "used_lag_frames", "usable_segments", "usable_frames", "active_frame_count"],
                lag_scan_diag_rows,
            )
            if plot_msm:
                plot_lag_scan(
                    lag_scan_rows,
                    Path(cfg.analysis_root) / "msm" / "implied_timescales_lag_scan",
                    style,
                    selected_lag=safe_msm_lag,
                    diagnostic_rows=lag_scan_diag_rows,
                )

        eligible_ck_lags = sorted(lag for lag in lag_models if lag >= safe_msm_lag and lag % max(safe_msm_lag, 1) == 0)
        if len(eligible_ck_lags) >= 2 and int(msm.n_states) >= 2:
            try:
                ck_models = [lag_models[lag] for lag in eligible_ck_lags]
                ck_test = lag_models[safe_msm_lag].ck_test(ck_models, n_metastable_sets=min(int(msm.n_states), 4))
                predictions = _dense_real_array(ck_test.predictions)
                estimates = _dense_real_array(ck_test.estimates)
                ck_rows = []
                finite_mask = np.isfinite(predictions) & np.isfinite(estimates)
                ck_rmse = float(np.sqrt(np.mean((predictions[finite_mask] - estimates[finite_mask]) ** 2))) if np.any(finite_mask) else np.nan
                ck_max_abs_diff = float(np.nanmax(np.abs(predictions - estimates))) if predictions.size else np.nan
                for lag_idx, lag_value in enumerate(np.asarray(ck_test.lagtimes, dtype=int)):
                    for from_idx in range(predictions.shape[1]):
                        for to_idx in range(predictions.shape[2]):
                            pred_value = float(predictions[lag_idx, from_idx, to_idx])
                            est_value = float(estimates[lag_idx, from_idx, to_idx])
                            abs_error = abs(pred_value - est_value) if np.isfinite(pred_value) and np.isfinite(est_value) else np.nan
                            ck_rows.append([int(lag_value), int(from_idx), int(to_idx), pred_value, est_value, abs_error])
                save_csv(
                    Path(cfg.analysis_root) / "msm" / "chapman_kolmogorov_test.csv",
                    ["lag_frames", "from_state", "to_state", "predicted_probability", "estimated_probability", "absolute_error"],
                    ck_rows,
                )
                msm_notes["chapman_kolmogorov_validation"] = {
                    "base_lag_frames": int(safe_msm_lag),
                    "comparison_lags_frames": [int(value) for value in eligible_ck_lags],
                    "rmse": ck_rmse,
                    "max_absolute_difference": ck_max_abs_diff,
                }
                if plot_msm:
                    ck_labels = [f"Meta {idx}" for idx in range(predictions.shape[1])]
                    plot_chapman_kolmogorov_test(
                        ck_test,
                        Path(cfg.analysis_root) / "msm" / "chapman_kolmogorov_test",
                        style,
                        state_labels=ck_labels,
                        summary_lines=[
                            f"base lag {safe_msm_lag}",
                            f"RMSE {ck_rmse:.3e}",
                            f"max |Δ| {ck_max_abs_diff:.3f}",
                        ],
                    )
            except Exception as exc:
                msm_notes["chapman_kolmogorov_error"] = str(exc)
    except Exception as exc:
        msm_notes["msm_error"] = str(exc)
    write_json(Path(cfg.analysis_root) / "msm" / "msm_notes.json", msm_notes)
    emit_progress(progress_callback, step_total, step_total, "advanced_analysis", "Advanced analysis completed")
    return str(Path(cfg.analysis_root).resolve())
