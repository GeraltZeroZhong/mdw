from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import numpy as np
import mdtraj as md
from rdkit import Chem

from ...core import atom_label, residue_label


def load_ligand_formal_charge_atoms(sdf_path, ligand_mdtraj_atom_count):
    if not str(sdf_path).strip() or Path(sdf_path).suffix.lower() not in {".sdf", ".sd"}:
        return [], []
    mol = Chem.MolFromMolFile(str(sdf_path), removeHs=False)
    if mol is None:
        print(f"警告：RDKit 无法读取配体 SDF，将跳过形式电荷盐桥分析: {sdf_path}")
        return [], []
    if mol.GetNumAtoms() != ligand_mdtraj_atom_count:
        print("警告：配体 SDF 原子数与轨迹不一致，将跳过形式电荷盐桥分析。")
        return [], []
    positive, negative = [], []
    for atom in mol.GetAtoms():
        q = atom.GetFormalCharge()
        if q > 0:
            positive.append(atom.GetIdx())
        elif q < 0:
            negative.append(atom.GetIdx())
    return positive, negative


def compute_salt_bridges(traj, ligand_atom_indices, ligand_sdf_path, salt_bridge_cutoff_nm: float):
    ligand_atom_indices = [int(idx) for idx in ligand_atom_indices]
    ligand_positive_local, ligand_negative_local = load_ligand_formal_charge_atoms(ligand_sdf_path, len(ligand_atom_indices))
    if not ligand_positive_local and not ligand_negative_local:
        return [], [], {}, {}

    ligand_positive_global = [ligand_atom_indices[i] for i in ligand_positive_local]
    ligand_negative_global = [ligand_atom_indices[i] for i in ligand_negative_local]

    acidic_atoms = defaultdict(list)
    basic_atoms = defaultdict(list)
    for atom in traj.topology.atoms:
        residue = atom.residue
        if not residue.is_protein:
            continue
        label = residue_label(residue)
        if residue.name == "ASP" and atom.name in {"OD1", "OD2"}:
            acidic_atoms[label].append(atom.index)
        elif residue.name == "GLU" and atom.name in {"OE1", "OE2"}:
            acidic_atoms[label].append(atom.index)
        elif residue.name == "LYS" and atom.name == "NZ":
            basic_atoms[label].append(atom.index)
        elif residue.name == "ARG" and atom.name in {"NE", "NH1", "NH2"}:
            basic_atoms[label].append(atom.index)

    pair_rows = []
    residue_present = {}
    pair_present_rows = {}

    if ligand_positive_global:
        for residue_label_, atom_indices in acidic_atoms.items():
            pairs = [(i, j) for i in atom_indices for j in ligand_positive_global]
            if not pairs:
                continue
            distances = md.compute_distances(traj, pairs, periodic=True)
            min_distance = distances.min(axis=1)
            residue_present[("acidic", residue_label_)] = (min_distance < salt_bridge_cutoff_nm)
            pair_occ = (distances < salt_bridge_cutoff_nm).mean(axis=0)
            for k, (i, j) in enumerate(pairs):
                present = distances[:, k] < salt_bridge_cutoff_nm
                pair_present_rows[f"acidic|{residue_label_}|{i}|{j}"] = present
                pair_rows.append(
                    {
                        "type": "protein_acidic_to_ligand_positive",
                        "protein_residue": residue_label_,
                        "protein_atom": atom_label(traj.topology.atom(i)),
                        "ligand_atom": atom_label(traj.topology.atom(j)),
                        "occupancy": float(pair_occ[k]),
                        "mean_distance_A": float(distances[:, k].mean() * 10.0),
                        "min_distance_A": float(distances[:, k].min() * 10.0),
                    }
                )

    if ligand_negative_global:
        for residue_label_, atom_indices in basic_atoms.items():
            pairs = [(i, j) for i in atom_indices for j in ligand_negative_global]
            if not pairs:
                continue
            distances = md.compute_distances(traj, pairs, periodic=True)
            min_distance = distances.min(axis=1)
            residue_present[("basic", residue_label_)] = (min_distance < salt_bridge_cutoff_nm)
            pair_occ = (distances < salt_bridge_cutoff_nm).mean(axis=0)
            for k, (i, j) in enumerate(pairs):
                present = distances[:, k] < salt_bridge_cutoff_nm
                pair_present_rows[f"basic|{residue_label_}|{i}|{j}"] = present
                pair_rows.append(
                    {
                        "type": "protein_basic_to_ligand_negative",
                        "protein_residue": residue_label_,
                        "protein_atom": atom_label(traj.topology.atom(i)),
                        "ligand_atom": atom_label(traj.topology.atom(j)),
                        "occupancy": float(pair_occ[k]),
                        "mean_distance_A": float(distances[:, k].mean() * 10.0),
                        "min_distance_A": float(distances[:, k].min() * 10.0),
                    }
                )

    pair_rows.sort(key=lambda x: x["occupancy"], reverse=True)
    residue_rows = [{"protein_residue": label, "salt_bridge_occupancy": float(present.mean())} for (_, label), present in residue_present.items()]
    residue_rows.sort(key=lambda x: x["salt_bridge_occupancy"], reverse=True)
    collapsed_residue_present = {label: present for (_, label), present in residue_present.items()}
    return pair_rows, residue_rows, collapsed_residue_present, pair_present_rows
