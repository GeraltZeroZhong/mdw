from __future__ import annotations

import numpy as np
import mdtraj as md

from ...config import AdvancedAnalysisConfig
from ...core import atom_label, find_ligand_residues, ligand_heavy_atom_indices_from_residues, ligand_residue_summary


def featurize_traj(traj, feature_pairs):
    return md.compute_distances(traj, feature_pairs, periodic=False)


def _min_ca_ligand_distance_by_atom(traj, protein_ca, ligand_heavy):
    pairs = [(int(i), int(j)) for i in protein_ca for j in ligand_heavy]
    distances = md.compute_distances(traj, pairs, periodic=False)
    min_by_ca = {int(i): np.inf for i in protein_ca}
    for pair_idx, (ca_idx, _lig_idx) in enumerate(pairs):
        min_by_ca[int(ca_idx)] = min(min_by_ca[int(ca_idx)], float(np.min(distances[:, pair_idx])))
    return min_by_ca


def pick_feature_atoms(trajectories, cfg: AdvancedAnalysisConfig):
    if not trajectories:
        raise ValueError("No trajectories were provided for feature selection.")
    first_traj = trajectories[0]
    ligand_residues = find_ligand_residues(first_traj.topology)
    ligand_heavy = np.asarray(ligand_heavy_atom_indices_from_residues(ligand_residues), dtype=int)
    protein_ca = first_traj.topology.select("protein and name CA")
    if len(protein_ca) == 0:
        raise ValueError("未找到 protein Cα 原子。")
    within = set()
    for traj in trajectories:
        min_by_ca = _min_ca_ligand_distance_by_atom(traj, protein_ca, ligand_heavy)
        within |= {atom_idx for atom_idx, dist in min_by_ca.items() if dist <= cfg.pocket_ca_cutoff_nm}
    pocket_ca = np.asarray(sorted(within), dtype=int)
    if len(pocket_ca) == 0:
        raise ValueError("未选到 pocket Cα 原子，请增大 pocket_ca_cutoff_nm。")
    feature_pairs = [(int(ca_idx), int(lig_idx)) for ca_idx in pocket_ca for lig_idx in ligand_heavy]
    metadata = {
        "feature_type": "protein_ca_ligand_heavy_distances",
        "ligand_residue": ligand_residue_summary(ligand_residues),
        "n_ligand_heavy_atoms": int(len(ligand_heavy)),
        "n_pocket_ca_atoms": int(len(pocket_ca)),
        "pocket_ca_atom_indices": pocket_ca.tolist(),
        "ligand_heavy_atom_indices": ligand_heavy.tolist(),
        "n_feature_pairs": int(len(feature_pairs)),
        "feature_pairs": [[int(i), int(j)] for i, j in feature_pairs],
        "feature_pair_labels": [
            f"{atom_label(first_traj.topology.atom(int(i)))}__{atom_label(first_traj.topology.atom(int(j)))}"
            for i, j in feature_pairs
        ],
        "pocket_ca_cutoff_nm": cfg.pocket_ca_cutoff_nm,
    }
    return feature_pairs, metadata
