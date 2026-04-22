from __future__ import annotations

from pathlib import Path
import numpy as np
import mdtraj as md
from rdkit import Chem

from ...core import residue_label


def compute_counts_from_boolean_dict(boolean_dict: dict[str, np.ndarray], n_frames: int) -> np.ndarray:
    if not boolean_dict:
        return np.zeros(n_frames, dtype=int)
    stack = np.vstack([np.asarray(v, dtype=bool) for v in boolean_dict.values()])
    return stack.sum(axis=0).astype(int)


def compute_rg(traj, protein_atom_indices) -> np.ndarray:
    return md.compute_rg(traj.atom_slice(protein_atom_indices)) * 10.0


def compute_sasa_metrics(traj, protein_atom_indices, ligand_atom_indices, probe_radius_nm: float = 0.14):
    protein_atom_indices = [int(idx) for idx in protein_atom_indices]
    ligand_atom_indices = [int(idx) for idx in ligand_atom_indices]
    protein_atom_set = set(protein_atom_indices)
    solute_atom_indices = protein_atom_indices + [idx for idx in ligand_atom_indices if idx not in protein_atom_set]
    if not solute_atom_indices:
        raise ValueError("No protein or ligand atoms available for SASA analysis.")
    solute_traj = traj.atom_slice(solute_atom_indices)
    atom_sasa = md.shrake_rupley(solute_traj, mode="atom", probe_radius=probe_radius_nm)
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(solute_atom_indices)}
    protein_local = [index_map[idx] for idx in protein_atom_indices if idx in index_map]
    ligand_local = [index_map[idx] for idx in ligand_atom_indices if idx in index_map]
    complex_sasa = atom_sasa.sum(axis=1) * 100.0
    protein_sasa = atom_sasa[:, protein_local].sum(axis=1) * 100.0
    ligand_sasa = atom_sasa[:, ligand_local].sum(axis=1) * 100.0
    buried = np.maximum(protein_sasa + ligand_sasa - complex_sasa, 0.0)
    return complex_sasa, protein_sasa, ligand_sasa, buried


def compute_dssp_metrics(traj):
    protein_atom_indices = traj.topology.select("protein")
    if len(protein_atom_indices) == 0:
        raise ValueError("Trajectory does not contain any protein residues for DSSP analysis.")
    protein_traj = traj.atom_slice(protein_atom_indices)
    dssp = md.compute_dssp(protein_traj, simplified=True)
    if dssp.ndim != 2:
        raise ValueError("DSSP output is not two-dimensional.")
    residues = list(protein_traj.topology.residues)
    states = {"Helix": "H", "Sheet": "E", "Coil": "C"}
    fractions = {name: (dssp == code).mean(axis=1) for name, code in states.items()}
    residue_labels = [residue_label(res) for res in residues]
    occupancy = np.zeros((len(residue_labels), 3), dtype=float)
    for idx, code in enumerate(["H", "E", "C"]):
        occupancy[:, idx] = (dssp == code).mean(axis=0)
    return dssp, residue_labels, fractions, occupancy


def _principal_axis(coords: np.ndarray) -> np.ndarray:
    centered = coords - coords.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    cov = centered.T @ centered
    vals, vecs = np.linalg.eigh(cov)
    axis = vecs[:, np.argmax(vals)]
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else np.array([1.0, 0.0, 0.0], dtype=float)


def _compute_com(xyz: np.ndarray) -> np.ndarray:
    return xyz.mean(axis=1)


def compute_ligand_pose_metrics(traj, ligand_heavy, protein_heavy_atoms_by_residue, reference, pose_cutoff_nm: float = 0.6):
    ligand_xyz = traj.xyz[:, ligand_heavy, :]
    ligand_com = _compute_com(ligand_xyz)
    ref_axis = _principal_axis(reference.xyz[0, ligand_heavy, :])

    pocket_atom_indices = []
    for atom_indices in protein_heavy_atoms_by_residue.values():
        prot = traj.xyz[:, atom_indices, :]
        d = np.linalg.norm(prot[:, :, None, :] - ligand_xyz[:, None, :, :], axis=3)
        if np.min(d) <= pose_cutoff_nm:
            pocket_atom_indices.extend(atom_indices)
    if not pocket_atom_indices:
        pocket_atom_indices = [idx for inds in protein_heavy_atoms_by_residue.values() for idx in inds]
    pocket_atom_indices = sorted(set(pocket_atom_indices))
    pocket_xyz = traj.xyz[:, pocket_atom_indices, :]
    pocket_com = _compute_com(pocket_xyz)
    com_distance_A = np.linalg.norm(ligand_com - pocket_com, axis=1) * 10.0

    orientation_angles = []
    for frame in range(traj.n_frames):
        axis = _principal_axis(ligand_xyz[frame])
        cosang = abs(float(np.clip(np.dot(axis, ref_axis), -1.0, 1.0)))
        orientation_angles.append(float(np.degrees(np.arccos(cosang))))
    return np.asarray(com_distance_A, dtype=float), np.asarray(orientation_angles, dtype=float)


def detect_ligand_rotatable_dihedrals(ligand_sdf_path: str, ligand_atom_indices: list[int]):
    path = Path(ligand_sdf_path)
    if not path.exists():
        return []
    mol = Chem.MolFromMolFile(str(path), removeHs=False)
    if mol is None or mol.GetNumAtoms() != len(ligand_atom_indices):
        return []
    torsions = []
    for bond in mol.GetBonds():
        if bond.GetBondTypeAsDouble() != 1.0 or bond.IsInRing():
            continue
        a2 = bond.GetBeginAtomIdx()
        a3 = bond.GetEndAtomIdx()
        atom2 = mol.GetAtomWithIdx(a2)
        atom3 = mol.GetAtomWithIdx(a3)
        nbr2 = [n.GetIdx() for n in atom2.GetNeighbors() if n.GetIdx() != a3 and n.GetAtomicNum() > 1]
        nbr3 = [n.GetIdx() for n in atom3.GetNeighbors() if n.GetIdx() != a2 and n.GetAtomicNum() > 1]
        if not nbr2 or not nbr3:
            continue
        a1, a4 = nbr2[0], nbr3[0]
        torsions.append((ligand_atom_indices[a1], ligand_atom_indices[a2], ligand_atom_indices[a3], ligand_atom_indices[a4]))
    return torsions[:6]


def compute_ligand_torsions(traj, torsion_atom_quads):
    if not torsion_atom_quads:
        return {}
    angles = md.compute_dihedrals(traj, torsion_atom_quads)
    out = {}
    for idx, quad in enumerate(torsion_atom_quads):
        out[f"torsion_{idx+1}_{quad[0]}_{quad[1]}_{quad[2]}_{quad[3]}"] = np.degrees(angles[:, idx])
    return out
