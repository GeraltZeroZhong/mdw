from __future__ import annotations

import math
import numpy as np
import mdtraj as md

from ...core import atom_label, residue_label


_HBOND_DONOR_ELEMENTS = {"N", "O"}
_HBOND_ACCEPTOR_ELEMENTS = {"N", "O"}
_HBOND_BATCH_TARGET_BYTES = 128 * 1024 * 1024
_HBOND_MAX_BATCH_SIZE = 2048


def _element_symbol(atom) -> str | None:
    element = getattr(atom, "element", None)
    if element is None:
        return None
    symbol = getattr(element, "symbol", None)
    return str(symbol).upper() if symbol else None


def _can_participate(atom) -> bool:
    return not getattr(atom.residue, "is_water", False)


def _donor_hydrogen_pairs(topology) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    saw_bond = False
    for atom0, atom1 in topology.bonds:
        saw_bond = True
        if not (_can_participate(atom0) and _can_participate(atom1)):
            continue
        symbol0 = _element_symbol(atom0)
        symbol1 = _element_symbol(atom1)
        if symbol0 in _HBOND_DONOR_ELEMENTS and symbol1 == "H":
            pairs.append((int(atom0.index), int(atom1.index)))
        elif symbol1 in _HBOND_DONOR_ELEMENTS and symbol0 == "H":
            pairs.append((int(atom1.index), int(atom0.index)))
    if not saw_bond:
        raise ValueError(
            "No bonds found in topology. Try using traj._topology.create_standard_bonds() before H-bond analysis."
        )
    return pairs


def _acceptor_atoms(topology) -> list[int]:
    return [
        int(atom.index)
        for atom in topology.atoms
        if _can_participate(atom) and _element_symbol(atom) in _HBOND_ACCEPTOR_ELEMENTS
    ]


def _split_hbond_participants(topology, ligand_atom_set: set[int]):
    protein_donors: list[tuple[int, int]] = []
    ligand_donors: list[tuple[int, int]] = []
    for donor_idx, hydrogen_idx in _donor_hydrogen_pairs(topology):
        donor_atom = topology.atom(donor_idx)
        if donor_idx in ligand_atom_set:
            ligand_donors.append((donor_idx, hydrogen_idx))
        elif donor_atom.residue.is_protein:
            protein_donors.append((donor_idx, hydrogen_idx))

    protein_acceptors: list[int] = []
    ligand_acceptors: list[int] = []
    for acceptor_idx in _acceptor_atoms(topology):
        acceptor_atom = topology.atom(acceptor_idx)
        if acceptor_idx in ligand_atom_set:
            ligand_acceptors.append(acceptor_idx)
        elif acceptor_atom.residue.is_protein:
            protein_acceptors.append(acceptor_idx)

    return protein_donors, ligand_donors, protein_acceptors, ligand_acceptors


def _hbond_batch_size(n_frames: int) -> int:
    # Each candidate needs H-A distances, D-H-A angles, and temporary masks.
    bytes_per_candidate = max(int(n_frames), 1) * 4 * 4
    return max(1, min(_HBOND_MAX_BATCH_SIZE, _HBOND_BATCH_TARGET_BYTES // bytes_per_candidate))


def _iter_candidate_batches(
    protein_donors: list[tuple[int, int]],
    ligand_donors: list[tuple[int, int]],
    protein_acceptors: list[int],
    ligand_acceptors: list[int],
    batch_size: int,
):
    batch: list[tuple[int, int, int, str]] = []
    groups = [
        ("protein_donor_to_ligand_acceptor", protein_donors, ligand_acceptors),
        ("ligand_donor_to_protein_acceptor", ligand_donors, protein_acceptors),
    ]
    for direction, donors, acceptors in groups:
        for donor_idx, hydrogen_idx in donors:
            for acceptor_idx in acceptors:
                if donor_idx == acceptor_idx:
                    continue
                batch.append((donor_idx, hydrogen_idx, acceptor_idx, direction))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def compute_hbond_occupancy(traj, ligand_atom_set, hbond_distance_nm: float, hbond_angle_deg: float):
    if traj.topology is None:
        raise ValueError("Hydrogen-bond analysis requires trajectory topology information.")
    ligand_atom_set = {int(idx) for idx in ligand_atom_set}
    protein_donors, ligand_donors, protein_acceptors, ligand_acceptors = _split_hbond_participants(
        traj.topology,
        ligand_atom_set,
    )
    triplet_rows = []
    residue_present = {}
    triplet_present_rows = {}
    angle_cut = math.radians(hbond_angle_deg)
    batch_size = _hbond_batch_size(traj.n_frames)

    for batch in _iter_candidate_batches(
        protein_donors,
        ligand_donors,
        protein_acceptors,
        ligand_acceptors,
        batch_size,
    ):
        triplets = np.asarray([(d, h, a) for d, h, a, _direction in batch], dtype=np.int32)
        ha = md.compute_distances(traj, triplets[:, [1, 2]], periodic=True)
        dha = md.compute_angles(traj, triplets, periodic=True)
        present_matrix = (ha < hbond_distance_nm) & (dha > angle_cut)
        active_columns = np.flatnonzero(np.any(present_matrix, axis=0))
        if active_columns.size == 0:
            continue
        da = md.compute_distances(traj, triplets[active_columns][:, [0, 2]], periodic=True)

        for da_col, col in enumerate(active_columns):
            donor_idx, hydrogen_idx, acceptor_idx, direction = batch[int(col)]
            donor_atom = traj.topology.atom(donor_idx)
            hydrogen_atom = traj.topology.atom(hydrogen_idx)
            acceptor_atom = traj.topology.atom(acceptor_idx)
            donor_res = donor_atom.residue
            acceptor_res = acceptor_atom.residue

            if direction == "protein_donor_to_ligand_acceptor":
                protein_res_label = residue_label(donor_res)
            else:
                protein_res_label = residue_label(acceptor_res)

            present = present_matrix[:, int(col)].copy()
            occupancy = float(np.mean(present))
            existing_present = residue_present.get(protein_res_label)
            if existing_present is None:
                residue_present[protein_res_label] = present.copy()
            else:
                existing_present |= present

            triplet_key = f"{protein_res_label}|{donor_idx}|{hydrogen_idx}|{acceptor_idx}"
            triplet_present_rows[triplet_key] = present
            ha_col = ha[:, int(col)]
            da_values = da[:, int(da_col)]
            triplet_rows.append(
                {
                    "direction": direction,
                    "donor_atom": atom_label(donor_atom),
                    "hydrogen_atom": atom_label(hydrogen_atom),
                    "acceptor_atom": atom_label(acceptor_atom),
                    "protein_residue": protein_res_label,
                    "occupancy": occupancy,
                    "mean_HA_distance_A": float(ha_col.mean() * 10.0),
                    "min_HA_distance_A": float(ha_col.min() * 10.0),
                    "mean_DA_distance_A": float(da_values.mean() * 10.0),
                    "min_DA_distance_A": float(da_values.min() * 10.0),
                }
            )

    triplet_rows.sort(key=lambda x: x["occupancy"], reverse=True)
    residue_rows = [{"protein_residue": key, "hbond_occupancy": float(np.mean(present))} for key, present in residue_present.items()]
    residue_rows.sort(key=lambda x: x["hbond_occupancy"], reverse=True)
    return triplet_rows, residue_rows, residue_present, triplet_present_rows
