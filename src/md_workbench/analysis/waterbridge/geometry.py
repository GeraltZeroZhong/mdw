from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mdtraj as md
import numpy as np
from mdtraj.formats import DCDTrajectoryFile

from ...core import atom_label, find_ligand_residue, residue_label

_FRAME_CHUNK_SIZE = 25
_TRIPLET_BATCH_SIZE = 256
_HBOND_PRESELECT_MARGIN_NM = 0.12
_MIN_PRESELECT_CUTOFF_NM = 0.40


@dataclass(frozen=True)
class _TripletMetadata:
    triplet: tuple[int, int, int]
    type: str
    water_residue: str
    protein_residue: str | None = None


@dataclass
class _TopologyIndex:
    ligand_residue: object
    ligand_interaction_atom_indices: np.ndarray
    protein_interaction_atom_indices: np.ndarray
    water_acceptor_indices: np.ndarray
    ligand_acceptor_set: set[int]
    protein_acceptor_set: set[int]
    ligand_donor_pairs_by_atom: dict[int, list[tuple[int, int]]]
    protein_donor_pairs_by_atom: dict[int, list[tuple[int, int]]]
    water_donor_pairs_by_residue: dict[int, list[tuple[int, int]]]
    water_oxygen_to_residue: dict[int, int]
    water_oxygen_by_residue: dict[int, int]


def _collect_donor_pairs(mapping: dict[int, list[tuple[int, int]]], atom_indices: set[int]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for atom_idx in atom_indices:
        pairs.extend(mapping.get(atom_idx, ()))
    return pairs


def _iter_donor_pairs(topology) -> list[tuple[int, int]]:
    donor_pairs: list[tuple[int, int]] = []
    for atom0, atom1 in topology.bonds:
        symbols = {atom0.element.symbol, atom1.element.symbol}
        if symbols not in ({"N", "H"}, {"O", "H"}):
            continue
        donor_atom, hydrogen_atom = (atom0, atom1) if atom0.element.symbol in {"N", "O"} else (atom1, atom0)
        donor_pairs.append((donor_atom.index, hydrogen_atom.index))
    return donor_pairs


def _build_topology_index(topology) -> _TopologyIndex:
    ligand_residue = find_ligand_residue(topology)
    ligand_acceptor_set: set[int] = set()
    protein_acceptor_set: set[int] = set()
    water_acceptor_indices: list[int] = []
    water_oxygen_to_residue: dict[int, int] = {}
    water_oxygen_by_residue: dict[int, int] = {}

    for atom in topology.atoms:
        if atom.element.symbol not in {"O", "N"}:
            continue
        if atom.residue.is_water:
            water_acceptor_indices.append(atom.index)
            water_oxygen_to_residue[atom.index] = atom.residue.index
            water_oxygen_by_residue[atom.residue.index] = atom.index
        elif atom.residue == ligand_residue:
            ligand_acceptor_set.add(atom.index)
        elif atom.residue.is_protein:
            protein_acceptor_set.add(atom.index)

    ligand_donor_pairs_by_atom: dict[int, list[tuple[int, int]]] = defaultdict(list)
    protein_donor_pairs_by_atom: dict[int, list[tuple[int, int]]] = defaultdict(list)
    water_donor_pairs_by_residue: dict[int, list[tuple[int, int]]] = defaultdict(list)
    ligand_donor_atoms: set[int] = set()
    protein_donor_atoms: set[int] = set()

    for donor_idx, hydrogen_idx in _iter_donor_pairs(topology):
        donor_atom = topology.atom(donor_idx)
        if donor_atom.residue.is_water:
            water_donor_pairs_by_residue[donor_atom.residue.index].append((donor_idx, hydrogen_idx))
        elif donor_atom.residue == ligand_residue:
            ligand_donor_pairs_by_atom[donor_idx].append((donor_idx, hydrogen_idx))
            ligand_donor_atoms.add(donor_idx)
        elif donor_atom.residue.is_protein:
            protein_donor_pairs_by_atom[donor_idx].append((donor_idx, hydrogen_idx))
            protein_donor_atoms.add(donor_idx)

    ligand_interaction_atom_indices = np.asarray(sorted(ligand_acceptor_set | ligand_donor_atoms), dtype=int)
    protein_interaction_atom_indices = np.asarray(sorted(protein_acceptor_set | protein_donor_atoms), dtype=int)
    return _TopologyIndex(
        ligand_residue=ligand_residue,
        ligand_interaction_atom_indices=ligand_interaction_atom_indices,
        protein_interaction_atom_indices=protein_interaction_atom_indices,
        water_acceptor_indices=np.asarray(sorted(water_acceptor_indices), dtype=int),
        ligand_acceptor_set=ligand_acceptor_set,
        protein_acceptor_set=protein_acceptor_set,
        ligand_donor_pairs_by_atom=dict(ligand_donor_pairs_by_atom),
        protein_donor_pairs_by_atom=dict(protein_donor_pairs_by_atom),
        water_donor_pairs_by_residue=dict(water_donor_pairs_by_residue),
        water_oxygen_to_residue=water_oxygen_to_residue,
        water_oxygen_by_residue=water_oxygen_by_residue,
    )


def _build_triplet_array(donor_pairs: list[tuple[int, int]], acceptor_indices: list[int]) -> np.ndarray:
    if not donor_pairs or not acceptor_indices:
        return np.zeros((0, 3), dtype=int)
    rows = [(donor_idx, hydrogen_idx, acceptor_idx) for donor_idx, hydrogen_idx in donor_pairs for acceptor_idx in acceptor_indices if donor_idx != acceptor_idx]
    if not rows:
        return np.zeros((0, 3), dtype=int)
    return np.asarray(rows, dtype=int)


def _make_single_frame_buffer(chunk: md.Trajectory) -> md.Trajectory:
    kwargs = {}
    if chunk.time is not None:
        kwargs["time"] = chunk.time[:1].copy()
    if chunk.unitcell_lengths is not None:
        kwargs["unitcell_lengths"] = chunk.unitcell_lengths[:1].copy()
    if chunk.unitcell_angles is not None:
        kwargs["unitcell_angles"] = chunk.unitcell_angles[:1].copy()
    return md.Trajectory(chunk.xyz[:1].copy(), chunk.topology, **kwargs)


def _update_single_frame_buffer(frame: md.Trajectory, chunk: md.Trajectory, local_frame_index: int) -> None:
    frame.xyz[0, :, :] = chunk.xyz[local_frame_index]
    if frame.time is not None and chunk.time is not None:
        frame.time[0] = chunk.time[local_frame_index]
    if frame.unitcell_lengths is not None and chunk.unitcell_lengths is not None:
        frame.unitcell_lengths[0, :] = chunk.unitcell_lengths[local_frame_index]
    if frame.unitcell_angles is not None and chunk.unitcell_angles is not None:
        frame.unitcell_angles[0, :] = chunk.unitcell_angles[local_frame_index]


def _present_triplets(frame, triplets: np.ndarray, hbond_distance_cutoff_nm: float, angle_cutoff_rad: float) -> np.ndarray:
    if triplets.size == 0:
        return np.zeros((0, 3), dtype=int)
    ha = md.compute_distances(frame, triplets[:, [1, 2]], periodic=True)[0]
    dha = md.compute_angles(frame, triplets, periodic=True)[0]
    return triplets[(ha < hbond_distance_cutoff_nm) & (dha > angle_cutoff_rad)]


def _metadata_from_triplet(topology, triplet: tuple[int, int, int], triplet_type: str) -> _TripletMetadata:
    donor_idx, _, acceptor_idx = triplet
    donor_atom = topology.atom(donor_idx)
    acceptor_atom = topology.atom(acceptor_idx)
    if triplet_type == "protein_donor_to_water_acceptor":
        return _TripletMetadata(triplet=triplet, type=triplet_type, protein_residue=residue_label(donor_atom.residue), water_residue=residue_label(acceptor_atom.residue))
    if triplet_type == "water_donor_to_protein_acceptor":
        return _TripletMetadata(triplet=triplet, type=triplet_type, protein_residue=residue_label(acceptor_atom.residue), water_residue=residue_label(donor_atom.residue))
    if triplet_type == "ligand_donor_to_water_acceptor":
        return _TripletMetadata(triplet=triplet, type=triplet_type, water_residue=residue_label(acceptor_atom.residue))
    if triplet_type == "water_donor_to_ligand_acceptor":
        return _TripletMetadata(triplet=triplet, type=triplet_type, water_residue=residue_label(donor_atom.residue))
    raise ValueError(f"未知的水桥 triplet 类型: {triplet_type}")


def _count_dcd_frames(dcd_path: Path) -> int:
    with DCDTrajectoryFile(str(dcd_path), "r") as handle:
        return len(handle)


def _discover_bridge_relevant_triplets(
    dcd_path: Path,
    top_path: Path,
    topology_index: _TopologyIndex,
    hbond_distance_cutoff_nm: float,
    hbond_angle_cutoff_deg: float,
) -> tuple[dict[tuple[int, int, int], _TripletMetadata], dict[tuple[int, int, int], _TripletMetadata], int]:
    n_frames = _count_dcd_frames(dcd_path)
    if n_frames <= 0:
        raise ValueError(f"轨迹文件不包含任何帧: {dcd_path}")

    preselect_cutoff_nm = max(hbond_distance_cutoff_nm + _HBOND_PRESELECT_MARGIN_NM, _MIN_PRESELECT_CUTOFF_NM)
    angle_cutoff_rad = math.radians(hbond_angle_cutoff_deg)
    observed_protein_triplets: dict[tuple[int, int, int], _TripletMetadata] = {}
    observed_ligand_triplets: dict[tuple[int, int, int], _TripletMetadata] = {}

    for chunk in md.iterload(str(dcd_path), top=str(top_path), chunk=_FRAME_CHUNK_SIZE):
        frame = _make_single_frame_buffer(chunk)
        ligand_near_waters_per_frame = md.compute_neighbors(
            chunk,
            preselect_cutoff_nm,
            topology_index.ligand_interaction_atom_indices,
            haystack_indices=topology_index.water_acceptor_indices,
            periodic=True,
        )

        for local_frame_index, near_ligand_water_acceptors in enumerate(ligand_near_waters_per_frame):
            if near_ligand_water_acceptors.size == 0:
                continue
            _update_single_frame_buffer(frame, chunk, local_frame_index)
            candidate_water_acceptors = [int(atom_idx) for atom_idx in near_ligand_water_acceptors]
            nearby_ligand_atoms = md.compute_neighbors(
                frame,
                preselect_cutoff_nm,
                np.asarray(candidate_water_acceptors, dtype=int),
                haystack_indices=topology_index.ligand_interaction_atom_indices,
                periodic=True,
            )[0]
            nearby_protein_atoms = md.compute_neighbors(
                frame,
                preselect_cutoff_nm,
                np.asarray(candidate_water_acceptors, dtype=int),
                haystack_indices=topology_index.protein_interaction_atom_indices,
                periodic=True,
            )[0]
            if nearby_ligand_atoms.size == 0 or nearby_protein_atoms.size == 0:
                continue

            candidate_water_residue_indices = {
                topology_index.water_oxygen_to_residue[int(atom_idx)] for atom_idx in candidate_water_acceptors
            }
            candidate_water_donor_pairs: list[tuple[int, int]] = []
            for residue_index in candidate_water_residue_indices:
                candidate_water_donor_pairs.extend(topology_index.water_donor_pairs_by_residue.get(residue_index, ()))

            nearby_ligand_atom_set = {int(atom_idx) for atom_idx in nearby_ligand_atoms}
            nearby_protein_atom_set = {int(atom_idx) for atom_idx in nearby_protein_atoms}
            ligand_donor_pairs = _collect_donor_pairs(topology_index.ligand_donor_pairs_by_atom, nearby_ligand_atom_set)
            protein_donor_pairs = _collect_donor_pairs(topology_index.protein_donor_pairs_by_atom, nearby_protein_atom_set)
            ligand_acceptors = [atom_idx for atom_idx in nearby_ligand_atom_set if atom_idx in topology_index.ligand_acceptor_set]
            protein_acceptors = [atom_idx for atom_idx in nearby_protein_atom_set if atom_idx in topology_index.protein_acceptor_set]

            protein_triplet_groups = (
                ("protein_donor_to_water_acceptor", _build_triplet_array(protein_donor_pairs, candidate_water_acceptors)),
                ("water_donor_to_protein_acceptor", _build_triplet_array(candidate_water_donor_pairs, protein_acceptors)),
            )
            ligand_triplet_groups = (
                ("ligand_donor_to_water_acceptor", _build_triplet_array(ligand_donor_pairs, candidate_water_acceptors)),
                ("water_donor_to_ligand_acceptor", _build_triplet_array(candidate_water_donor_pairs, ligand_acceptors)),
            )

            for triplet_type, candidate_triplets in protein_triplet_groups:
                for triplet_row in _present_triplets(frame, candidate_triplets, hbond_distance_cutoff_nm, angle_cutoff_rad):
                    triplet = tuple(int(value) for value in triplet_row)
                    observed_protein_triplets.setdefault(triplet, _metadata_from_triplet(frame.topology, triplet, triplet_type))

            for triplet_type, candidate_triplets in ligand_triplet_groups:
                for triplet_row in _present_triplets(frame, candidate_triplets, hbond_distance_cutoff_nm, angle_cutoff_rad):
                    triplet = tuple(int(value) for value in triplet_row)
                    observed_ligand_triplets.setdefault(triplet, _metadata_from_triplet(frame.topology, triplet, triplet_type))

    return observed_protein_triplets, observed_ligand_triplets, n_frames


def _init_triplet_stats(n_triplets: int) -> dict[str, np.ndarray]:
    return {
        "present_count": np.zeros(n_triplets, dtype=int),
        "ha_sum": np.zeros(n_triplets, dtype=float),
        "ha_min": np.full(n_triplets, np.inf, dtype=float),
        "da_sum": np.zeros(n_triplets, dtype=float),
        "da_min": np.full(n_triplets, np.inf, dtype=float),
    }


def _update_triplet_stats(
    chunk,
    metadata: list[_TripletMetadata],
    stats: dict[str, np.ndarray],
    hbond_distance_cutoff_nm: float,
    angle_cutoff_rad: float,
    per_water_presence: defaultdict[str, np.ndarray] | None = None,
    per_water_residue_presence: defaultdict[str, dict[str, np.ndarray]] | None = None,
) -> None:
    if not metadata:
        return

    triplets = np.asarray([meta.triplet for meta in metadata], dtype=int)
    for start in range(0, len(metadata), _TRIPLET_BATCH_SIZE):
        stop = min(start + _TRIPLET_BATCH_SIZE, len(metadata))
        triplet_batch = triplets[start:stop]
        ha = md.compute_distances(chunk, triplet_batch[:, [1, 2]], periodic=True)
        da = md.compute_distances(chunk, triplet_batch[:, [0, 2]], periodic=True)
        dha = md.compute_angles(chunk, triplet_batch, periodic=True)
        present = (ha < hbond_distance_cutoff_nm) & (dha > angle_cutoff_rad)

        stats["present_count"][start:stop] += present.sum(axis=0)
        stats["ha_sum"][start:stop] += ha.sum(axis=0)
        stats["da_sum"][start:stop] += da.sum(axis=0)
        stats["ha_min"][start:stop] = np.minimum(stats["ha_min"][start:stop], ha.min(axis=0))
        stats["da_min"][start:stop] = np.minimum(stats["da_min"][start:stop], da.min(axis=0))

        if per_water_presence is None:
            continue

        batch_metadata = metadata[start:stop]
        for local_index, meta in enumerate(batch_metadata):
            present_series = present[:, local_index]
            if not np.any(present_series):
                continue
            per_water_presence[meta.water_residue] |= present_series
            if per_water_residue_presence is not None and meta.protein_residue is not None:
                per_water_residue_presence[meta.water_residue][meta.protein_residue] |= present_series


def _build_triplet_rows(topology, metadata: list[_TripletMetadata], stats: dict[str, np.ndarray], n_frames: int) -> list[dict]:
    rows: list[dict] = []
    for index, meta in enumerate(metadata):
        donor_atom = topology.atom(meta.triplet[0])
        hydrogen_atom = topology.atom(meta.triplet[1])
        acceptor_atom = topology.atom(meta.triplet[2])
        row = {
            "type": meta.type,
            "donor_atom": atom_label(donor_atom),
            "hydrogen_atom": atom_label(hydrogen_atom),
            "acceptor_atom": atom_label(acceptor_atom),
            "water_residue": meta.water_residue,
            "occupancy": float(stats["present_count"][index] / n_frames),
            "mean_HA_distance_A": float(stats["ha_sum"][index] * 10.0 / n_frames),
            "min_HA_distance_A": float(stats["ha_min"][index] * 10.0),
            "mean_DA_distance_A": float(stats["da_sum"][index] * 10.0 / n_frames),
            "min_DA_distance_A": float(stats["da_min"][index] * 10.0),
        }
        if meta.protein_residue is not None:
            row["protein_residue"] = meta.protein_residue
        rows.append(row)
    rows.sort(key=lambda item: item["occupancy"], reverse=True)
    return rows


def _summarize_bridge_statistics(
    dcd_path: Path,
    top_path: Path,
    topology,
    protein_triplets: dict[tuple[int, int, int], _TripletMetadata],
    ligand_triplets: dict[tuple[int, int, int], _TripletMetadata],
    n_frames: int,
    hbond_distance_cutoff_nm: float,
    hbond_angle_cutoff_deg: float,
) -> tuple[list[dict], list[dict], list[dict], list[dict], np.ndarray]:
    protein_metadata = list(protein_triplets.values())
    ligand_metadata = list(ligand_triplets.values())
    protein_stats = _init_triplet_stats(len(protein_metadata))
    ligand_stats = _init_triplet_stats(len(ligand_metadata))
    waterbridge_count = np.zeros(n_frames, dtype=int)
    water_bridge_counts: dict[str, int] = defaultdict(int)
    residue_bridge_counts: dict[str, int] = defaultdict(int)
    angle_cutoff_rad = math.radians(hbond_angle_cutoff_deg)
    frame_offset = 0

    for chunk in md.iterload(str(dcd_path), top=str(top_path), chunk=_FRAME_CHUNK_SIZE):
        n_chunk_frames = chunk.n_frames
        chunk_waterbridge_count = np.zeros(n_chunk_frames, dtype=int)
        ligand_presence_by_water: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_chunk_frames, dtype=bool))
        protein_presence_by_water: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_chunk_frames, dtype=bool))
        protein_presence_by_water_and_residue: defaultdict[str, dict[str, np.ndarray]] = defaultdict(
            lambda: defaultdict(lambda: np.zeros(n_chunk_frames, dtype=bool))
        )
        residue_bridge_presence: defaultdict[str, np.ndarray] = defaultdict(lambda: np.zeros(n_chunk_frames, dtype=bool))

        _update_triplet_stats(
            chunk,
            ligand_metadata,
            ligand_stats,
            hbond_distance_cutoff_nm,
            angle_cutoff_rad,
            per_water_presence=ligand_presence_by_water,
        )
        _update_triplet_stats(
            chunk,
            protein_metadata,
            protein_stats,
            hbond_distance_cutoff_nm,
            angle_cutoff_rad,
            per_water_presence=protein_presence_by_water,
            per_water_residue_presence=protein_presence_by_water_and_residue,
        )

        for water_residue, ligand_present in ligand_presence_by_water.items():
            protein_present = protein_presence_by_water.get(water_residue)
            if protein_present is None:
                continue
            bridge_present = ligand_present & protein_present
            if not np.any(bridge_present):
                continue
            chunk_waterbridge_count += bridge_present.astype(int)
            water_bridge_counts[water_residue] += int(bridge_present.sum())
            for protein_residue, residue_present in protein_presence_by_water_and_residue[water_residue].items():
                residue_bridge = ligand_present & residue_present
                if np.any(residue_bridge):
                    residue_bridge_presence[protein_residue] |= residue_bridge

        for protein_residue, residue_present in residue_bridge_presence.items():
            residue_bridge_counts[protein_residue] += int(residue_present.sum())

        waterbridge_count[frame_offset:frame_offset + n_chunk_frames] = chunk_waterbridge_count
        frame_offset += n_chunk_frames

    protein_rows = _build_triplet_rows(topology, protein_metadata, protein_stats, n_frames)
    ligand_rows = _build_triplet_rows(topology, ligand_metadata, ligand_stats, n_frames)
    residue_rows = [
        {"protein_residue": residue_label_key, "waterbridge_occupancy": float(frame_count / n_frames)}
        for residue_label_key, frame_count in residue_bridge_counts.items()
    ]
    residue_rows.sort(key=lambda item: item["waterbridge_occupancy"], reverse=True)
    water_rows = [
        {"water_residue": water_label, "waterbridge_occupancy": float(frame_count / n_frames)}
        for water_label, frame_count in water_bridge_counts.items()
    ]
    water_rows.sort(key=lambda item: item["waterbridge_occupancy"], reverse=True)
    return protein_rows, ligand_rows, residue_rows, water_rows, waterbridge_count


def analyze_waterbridge_trajectory(
    dcd_path: str | Path,
    top_path: str | Path,
    hbond_distance_cutoff_nm: float,
    hbond_angle_cutoff_deg: float,
) -> dict:
    dcd_path = Path(dcd_path)
    top_path = Path(top_path)
    sample = md.load_frame(str(dcd_path), 0, top=str(top_path))
    topology_index = _build_topology_index(sample.topology)
    protein_triplets, ligand_triplets, n_frames = _discover_bridge_relevant_triplets(
        dcd_path,
        top_path,
        topology_index,
        hbond_distance_cutoff_nm,
        hbond_angle_cutoff_deg,
    )
    protein_rows, ligand_rows, residue_rows, water_rows, waterbridge_count = _summarize_bridge_statistics(
        dcd_path,
        top_path,
        sample.topology,
        protein_triplets,
        ligand_triplets,
        n_frames,
        hbond_distance_cutoff_nm,
        hbond_angle_cutoff_deg,
    )
    return {
        "ligand_residue": topology_index.ligand_residue,
        "n_frames": n_frames,
        "protein_triplets": protein_rows,
        "ligand_triplets": ligand_rows,
        "residue_rows": residue_rows,
        "water_rows": water_rows,
        "waterbridge_count": waterbridge_count,
    }
