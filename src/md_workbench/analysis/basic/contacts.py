from __future__ import annotations

import mdtraj as md
import numpy as np


def group_protein_heavy_atoms_by_residue(traj):
    groups = {}
    for residue in traj.topology.residues:
        if not residue.is_protein:
            continue
        atom_indices = []
        for atom in residue.atoms:
            if atom.element is None or atom.element.symbol == "H":
                continue
            atom_indices.append(atom.index)
        if atom_indices:
            groups[f"chain{residue.chain.index}_{residue.name}{residue.resSeq}"] = atom_indices
    return groups


def compute_contact_occupancy(traj, ligand_heavy, protein_heavy_atoms_by_residue, contact_cutoff_nm: float):
    residue_rows = []
    residue_min_distance_curves = {}
    per_residue_contact_boolean = {}

    for residue_label, atom_indices in protein_heavy_atoms_by_residue.items():
        pairs = [(i, j) for i in atom_indices for j in ligand_heavy]
        if not pairs:
            continue
        distances = md.compute_distances(traj, pairs, periodic=True)
        min_distance = distances.min(axis=1)
        occupancy = float(np.mean(min_distance < contact_cutoff_nm))
        residue_min_distance_curves[residue_label] = min_distance
        per_residue_contact_boolean[residue_label] = (min_distance < contact_cutoff_nm)
        residue_rows.append(
            {
                "protein_residue": residue_label,
                "contact_occupancy": occupancy,
                "min_distance_mean_A": float(min_distance.mean() * 10.0),
                "min_distance_min_A": float(min_distance.min() * 10.0),
            }
        )

    residue_rows.sort(key=lambda x: x["contact_occupancy"], reverse=True)
    if not residue_min_distance_curves:
        raise ValueError("没有计算出任何蛋白-配体接触距离。")
    all_curves = np.vstack([residue_min_distance_curves[k] for k in residue_min_distance_curves])
    global_min_distance_A = all_curves.min(axis=0) * 10.0
    return residue_rows, residue_min_distance_curves, per_residue_contact_boolean, global_min_distance_A
