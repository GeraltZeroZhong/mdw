from __future__ import annotations

import math
import mdtraj as md
import numpy as np

from ...core import atom_label, residue_label


def compute_hbond_occupancy(traj, ligand_atom_set, hbond_distance_nm: float, hbond_angle_deg: float):
    raw_triplets = md.baker_hubbard(
        traj,
        periodic=True,
        exclude_water=True,
        freq=0.0,
        distance_cutoff=hbond_distance_nm,
        angle_cutoff=hbond_angle_deg,
    )
    triplet_rows = []
    residue_present = {}
    triplet_present_rows = {}
    angle_cut = math.radians(hbond_angle_deg)

    for donor_idx, hydrogen_idx, acceptor_idx in raw_triplets:
        donor_atom = traj.topology.atom(donor_idx)
        hydrogen_atom = traj.topology.atom(hydrogen_idx)
        acceptor_atom = traj.topology.atom(acceptor_idx)

        donor_res = donor_atom.residue
        acceptor_res = acceptor_atom.residue
        donor_is_lig = donor_idx in ligand_atom_set
        acceptor_is_lig = acceptor_idx in ligand_atom_set
        if donor_is_lig == acceptor_is_lig:
            continue
        if not ((donor_res.is_protein and acceptor_is_lig) or (acceptor_res.is_protein and donor_is_lig)):
            continue

        ha = md.compute_distances(traj, [[hydrogen_idx, acceptor_idx]], periodic=True)[:, 0]
        da = md.compute_distances(traj, [[donor_idx, acceptor_idx]], periodic=True)[:, 0]
        dha = md.compute_angles(traj, [[donor_idx, hydrogen_idx, acceptor_idx]], periodic=True)[:, 0]
        present = (ha < hbond_distance_nm) & (dha > angle_cut)
        occupancy = float(np.mean(present))

        if donor_res.is_protein and acceptor_is_lig:
            protein_res_label = residue_label(donor_res)
            direction = "protein_donor_to_ligand_acceptor"
        else:
            protein_res_label = residue_label(acceptor_res)
            direction = "ligand_donor_to_protein_acceptor"

        residue_present[protein_res_label] = residue_present.get(protein_res_label, np.zeros_like(present, dtype=bool)) | present
        triplet_key = f"{protein_res_label}|{donor_idx}|{hydrogen_idx}|{acceptor_idx}"
        triplet_present_rows[triplet_key] = present
        triplet_rows.append(
            {
                "direction": direction,
                "donor_atom": atom_label(donor_atom),
                "hydrogen_atom": atom_label(hydrogen_atom),
                "acceptor_atom": atom_label(acceptor_atom),
                "protein_residue": protein_res_label,
                "occupancy": occupancy,
                "mean_HA_distance_A": float(ha.mean() * 10.0),
                "min_HA_distance_A": float(ha.min() * 10.0),
                "mean_DA_distance_A": float(da.mean() * 10.0),
                "min_DA_distance_A": float(da.min() * 10.0),
            }
        )

    triplet_rows.sort(key=lambda x: x["occupancy"], reverse=True)
    residue_rows = [{"protein_residue": key, "hbond_occupancy": float(np.mean(present))} for key, present in residue_present.items()]
    residue_rows.sort(key=lambda x: x["hbond_occupancy"], reverse=True)
    return triplet_rows, residue_rows, residue_present, triplet_present_rows
