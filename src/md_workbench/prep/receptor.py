from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from ..core import check_input_file

STANDARD_RESIDUES = {
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "HID", "HIE", "HIP", "ASH", "GLH",
    "LYN", "CYX", "ACE", "NME",
}

EXCLUDED_HETATM_RESNAMES = {
    "HOH", "WAT", "SOL", "DOD",
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU", "CO", "NI",
    "CD", "HG", "CS", "RB", "LI", "F", "BR", "I",
    "SO4", "PO4", "PEG", "GOL", "EDO", "DMS", "ACT", "ACY", "FMT", "MES", "TRS",
}

STANDARD_POLYMER_RESIDUES = {
    # proteins
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET",
    "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE", "HID", "HIE", "HIP", "ASH", "GLH",
    "LYN", "CYX", "ACE", "NME",
    # nucleic acids commonly encountered in template sets
    "A", "C", "G", "U", "DA", "DC", "DG", "DT", "DU",
}


def assess_receptor_for_preprocess(input_pdb: str) -> dict:
    path = check_input_file(input_pdb)
    reasons: list[str] = []
    has_h = False
    has_nonwater_heterogen = False
    has_nonstandard = False
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            resname = line[17:20].strip().upper()
            atom_name = line[12:16].strip().upper()
            element = line[76:78].strip().upper()
            if atom_name.startswith("H") or element == "H":
                has_h = True
            if line.startswith("HETATM") and resname not in {"HOH", "WAT", "SOL"}:
                has_nonwater_heterogen = True
            if line.startswith("ATOM") and resname and resname not in STANDARD_RESIDUES:
                has_nonstandard = True
    if not has_h:
        reasons.append("Hydrogens are not present; MD-ready protonation is still needed.")
    if has_nonwater_heterogen:
        reasons.append("Non-water heterogens were detected and will be ignored for receptor preparation by default.")
    if has_nonstandard:
        reasons.append("Nonstandard residues were detected and may need replacement.")
    filename = Path(input_pdb).name.lower()
    if "alphafold" in filename or filename.startswith("af-"):
        reasons.append("The filename suggests an AlphaFold-style model; preprocessing is usually recommended.")
    return {"recommended": bool(reasons), "reasons": reasons}


def _parse_coord_triplet_from_pdb_line(line: str):
    try:
        return (
            float(line[30:38]),
            float(line[38:46]),
            float(line[46:54]),
        )
    except ValueError:
        parts = line.split()
        if len(parts) < 9:
            raise
        return float(parts[6]), float(parts[7]), float(parts[8])


def _iter_pdb_atoms(input_pdb: str):
    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            try:
                x, y, z = _parse_coord_triplet_from_pdb_line(line)
            except ValueError:
                continue
            resname = line[17:20].strip().upper()
            chain = line[21].strip() or "_"
            resseq = line[22:26].strip()
            icode = line[26].strip()
            atom_name = line[12:16].strip().upper()
            element = line[76:78].strip().upper() or atom_name[:1]
            yield {
                "record": line[:6].strip().upper(),
                "resname": resname,
                "chain": chain,
                "resseq": resseq,
                "icode": icode,
                "atom_name": atom_name,
                "element": element,
                "coord": np.array([x, y, z], dtype=float),
            }


def _candidate_bound_ligand_groups(input_pdb: str) -> dict[tuple[str, str, str, str], list[np.ndarray]]:
    groups: dict[tuple[str, str, str, str], list[np.ndarray]] = defaultdict(list)
    for atom in _iter_pdb_atoms(input_pdb):
        if atom["record"] != "HETATM":
            continue
        if atom["resname"] in EXCLUDED_HETATM_RESNAMES:
            continue
        if atom["element"] == "H":
            continue
        key = (atom["resname"], atom["chain"], atom["resseq"], atom["icode"])
        groups[key].append(atom["coord"])
    return groups


def infer_search_space_from_pdb(
    input_pdb: str,
    padding_angstrom: float = 8.0,
    min_size_angstrom: float = 16.0,
    fallback_size_angstrom: Iterable[float] = (20.0, 20.0, 20.0),
    allow_protein_centroid_fallback: bool = False,
) -> dict:
    input_pdb = str(check_input_file(input_pdb))
    ligand_groups = _candidate_bound_ligand_groups(input_pdb)
    if ligand_groups:
        best_key, coords = max(ligand_groups.items(), key=lambda item: len(item[1]))
        arr = np.vstack(coords)
        center = arr.mean(axis=0)
        span = arr.max(axis=0) - arr.min(axis=0)
        size = np.maximum(span + 2.0 * float(padding_angstrom), float(min_size_angstrom))
        return {
            "source": "bound_heterogen",
            "residue": f"{best_key[0]}:{best_key[1]}:{best_key[2]}{best_key[3]}",
            "center": center.tolist(),
            "size": size.tolist(),
        }

    if not allow_protein_centroid_fallback:
        raise ValueError(
            "No bound heterogen suitable for automatic docking-box inference was found in the receptor structure. "
            "Provide an explicit docking box, or enable allow_protein_centroid_box_fallback if you intentionally "
            "want the less rigorous protein-centroid fallback."
        )

    protein_coords = [atom["coord"] for atom in _iter_pdb_atoms(input_pdb) if atom["record"] == "ATOM" and atom["element"] != "H"]
    if protein_coords:
        arr = np.vstack(protein_coords)
        center = arr.mean(axis=0)
        return {
            "source": "protein_centroid_fallback",
            "residue": "",
            "center": center.tolist(),
            "size": list(fallback_size_angstrom),
        }

    raise ValueError("No valid receptor atoms were found for docking search-space inference. Please verify that the input file is a real PDB file with ATOM/HETATM records and Cartesian coordinates.")


def _parse_occupancy_from_pdb_line(line: str) -> float:
    try:
        return float(line[54:60])
    except Exception:
        return 0.0


def sanitize_receptor_for_docking(input_pdb: str, output_pdb: str) -> str:
    """Write a Meeko-friendly receptor PDB.

    The project intentionally ignores embedded ligands and other heterogens for
    receptor preparation. In practice, mmCIF/PDB-to-PDB conversion and alternate
    locations can leave records that make RDKit sanitization inside Meeko fail.
    For docking we therefore keep only polymer ATOM records from standard
    residues, drop terminal OXT atoms that Meeko/RDKit can mis-connect, collapse
    alternate locations to a single representative per atom, and write a clean
    PDB file for ``mk_prepare_receptor.py``.
    """
    input_pdb = str(check_input_file(input_pdb))
    out = Path(output_pdb)
    out.parent.mkdir(parents=True, exist_ok=True)

    grouped: dict[tuple[str, str, str, str, str], list[tuple[str, float]]] = defaultdict(list)
    order: list[tuple[str, str, str, str, str]] = []
    with open(input_pdb, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            resname = line[17:20].strip().upper()
            if resname and resname not in STANDARD_POLYMER_RESIDUES:
                continue
            chain = line[21]
            resseq = line[22:26]
            icode = line[26]
            atom_name = line[12:16]
            if atom_name.strip().upper() == "OXT":
                continue
            key = (chain, resseq, icode, resname, atom_name)
            if key not in grouped:
                order.append(key)
            grouped[key].append((line, _parse_occupancy_from_pdb_line(line)))

    if not grouped:
        raise ValueError(
            "No polymer ATOM records remained after receptor sanitization. "
            "Please provide a protein-like receptor structure in PDB/mmCIF format."
        )

    def choose_record(records: list[tuple[str, float]]) -> str:
        def sort_key(item: tuple[str, float]):
            line, occ = item
            altloc = line[16].strip()
            pref_blank = 0 if altloc == "" else 1
            pref_a = 0 if altloc == "A" else 1
            return (pref_blank, pref_a, -occ, altloc)

        return sorted(records, key=sort_key)[0][0]

    serial = 1
    last_residue: tuple[str, str, str, str] | None = None
    lines_out: list[str] = []
    for key in order:
        line = choose_record(grouped[key])
        chain, resseq, icode, resname, _atom_name = key
        residue_id = (chain, resseq, icode, resname)
        if last_residue is not None and chain != last_residue[0]:
            prev_chain, prev_resseq, prev_icode, prev_resname = last_residue
            ter = f"TER   {serial:>5d}      {prev_resname:>3s} {(prev_chain if prev_chain.strip() else ' ')}{prev_resseq}{prev_icode}\n"
            lines_out.append(ter)
            serial += 1

        chars = list(line.rstrip("\n"))
        if len(chars) < 80:
            chars.extend([" "] * (80 - len(chars)))
        chars[0:6] = list("ATOM  ")
        chars[6:11] = list(f"{serial:>5d}")
        chars[16] = " "
        lines_out.append("".join(chars).rstrip() + "\n")
        serial += 1
        last_residue = residue_id

    if last_residue is not None:
        prev_chain, prev_resseq, prev_icode, prev_resname = last_residue
        ter = f"TER   {serial:>5d}      {prev_resname:>3s} {(prev_chain if prev_chain.strip() else ' ')}{prev_resseq}{prev_icode}\n"
        lines_out.append(ter)
    lines_out.append("END\n")

    out.write_text("".join(lines_out), encoding="utf-8")
    return str(out)


def fix_receptor(
    input_pdb: str,
    output_pdb: str,
    ph: float = 7.4,
    replace_nonstandard_residues: bool = True,
    remove_heterogens_keep_water: bool = False,
    missing_residue_policy: str = "internal",
) -> str:
    check_input_file(input_pdb)
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    fixer = PDBFixer(filename=input_pdb)
    fixer.findMissingResidues()
    policy = str(missing_residue_policy).strip().lower()
    if policy == "none":
        fixer.missingResidues = {}
    elif policy == "internal":
        chains = list(fixer.topology.chains())
        for key in list(fixer.missingResidues.keys()):
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]
    elif policy != "all":
        raise ValueError("missing_residue_policy must be one of: internal, all, none.")
    if replace_nonstandard_residues:
        fixer.findNonstandardResidues()
        if fixer.nonstandardResidues:
            fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(remove_heterogens_keep_water)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    Path(output_pdb).parent.mkdir(parents=True, exist_ok=True)
    with open(output_pdb, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(fixer.topology, fixer.positions, handle, keepIds=True)
    return output_pdb


def meeko_delete_residues_spec(input_pdb: str) -> str:
    """Return a Meeko --delete_residues spec that removes all HETATM residues.

    This mirrors the current receptor-preparation policy of ignoring embedded small
    molecules, ions, cofactors, and waters unless the user manually keeps them in a
    custom workflow. The returned format follows Meeko's documented syntax such as
    ``A:350,B:15,16``.
    """
    by_chain: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for atom in _iter_pdb_atoms(input_pdb):
        if atom["record"] != "HETATM":
            continue
        chain = atom["chain"]
        resid = f"{atom['resseq']}{atom['icode']}".strip()
        key = (chain, atom["resseq"], atom["icode"])
        if key in seen:
            continue
        seen.add(key)
        by_chain.setdefault(chain, []).append(resid)
    parts: list[str] = []
    for chain, residues in by_chain.items():
        chain_prefix = "" if chain == "_" else chain
        parts.append(f"{chain_prefix}:{','.join(residues)}")
    return ",".join(parts)
