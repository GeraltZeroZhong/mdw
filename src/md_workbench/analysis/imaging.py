from __future__ import annotations

import mdtraj as md


def _copy_or_make_whole(traj: md.Trajectory) -> md.Trajectory:
    try:
        return traj.make_molecules_whole(inplace=False)
    except Exception:
        return traj[:]


def _ligand_atom_set(ligand_residue_or_residues) -> set:
    residues = ligand_residue_or_residues if isinstance(ligand_residue_or_residues, (list, tuple, set)) else [ligand_residue_or_residues]
    return {atom for residue in residues for atom in residue.atoms}


def image_ligand_near_protein(traj: md.Trajectory, ligand_residue) -> md.Trajectory:
    """Return a trajectory with protein anchors centered and the ligand imaged nearby."""
    if traj.n_frames == 0:
        return traj[:]
    if traj.unitcell_lengths is None:
        return _copy_or_make_whole(traj)

    ligand_atoms = _ligand_atom_set(ligand_residue)
    try:
        molecules = traj.topology.find_molecules()
    except Exception as exc:
        raise ValueError("Cannot infer molecules for protein-ligand PBC imaging.") from exc

    protein_molecules = [mol for mol in molecules if any(atom.residue.is_protein for atom in mol)]
    ligand_molecules = [mol for mol in molecules if any(atom in ligand_atoms for atom in mol)]
    if not protein_molecules or not ligand_molecules:
        raise ValueError("Cannot identify protein and ligand molecules for PBC imaging.")

    try:
        return traj.image_molecules(
            anchor_molecules=protein_molecules,
            other_molecules=ligand_molecules,
            make_whole=True,
            inplace=False,
        )
    except Exception as exc:
        raise ValueError("Failed to image ligand near protein before analysis.") from exc
