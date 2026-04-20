from __future__ import annotations

import mdtraj as md
import numpy as np

from ...core import residue_label


def build_average_reference(traj, align_atom_indices, n_iter: int = 3):
    if traj.n_frames == 0:
        raise ValueError("Cannot build an average reference from an empty trajectory.")
    working = traj[:]
    working.superpose(working, 0, atom_indices=align_atom_indices)
    reference = working[0]
    for _ in range(max(1, int(n_iter))):
        mean_xyz = working.xyz.mean(axis=0)
        reference = md.Trajectory(mean_xyz[None, :, :].copy(), working.topology)
        working.superpose(reference, 0, atom_indices=align_atom_indices, ref_atom_indices=align_atom_indices)
    return reference


def compute_rmsd_and_rmsf(traj, protein_bb, ligand_heavy, protein_ca, reference):
    protein_rmsd_A = md.rmsd(
        traj,
        reference,
        0,
        atom_indices=protein_bb,
        ref_atom_indices=protein_bb,
        superpose=False,
    ) * 10.0
    ligand_rmsd_A = md.rmsd(
        traj,
        reference,
        0,
        atom_indices=ligand_heavy,
        ref_atom_indices=ligand_heavy,
        superpose=False,
    ) * 10.0
    rmsf_nm = md.rmsf(
        traj,
        reference,
        0,
        atom_indices=protein_ca,
        ref_atom_indices=protein_ca,
    )
    rmsf_rows = []
    for atom_idx, rmsf_value in zip(protein_ca, rmsf_nm):
        atom = traj.topology.atom(atom_idx)
        rmsf_rows.append(
            {
                "protein_residue": residue_label(atom.residue),
                "resSeq": atom.residue.resSeq,
                "resname": atom.residue.name,
                "rmsf_A": float(rmsf_value * 10.0),
            }
        )
    return protein_rmsd_A, ligand_rmsd_A, rmsf_rows
