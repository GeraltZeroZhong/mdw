from __future__ import annotations

EXCLUDED_NONLIGAND_RESNAMES = {
    "HOH", "WAT", "SOL",
    "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE", "CU",
    "CS", "RB", "LI", "F", "BR", "I",
}


def residue_label(residue) -> str:
    return f"chain{residue.chain.index}_{residue.name}{residue.resSeq}"


def atom_label(atom) -> str:
    return f"{residue_label(atom.residue)}:{atom.name}"


def is_water_residue(residue) -> bool:
    return residue.name.upper() in {"HOH", "WAT", "SOL"}


def find_ligand_residue(topology, excluded=None):
    excluded = excluded or EXCLUDED_NONLIGAND_RESNAMES
    ligand_residues = []
    for residue in topology.residues:
        if residue.is_protein:
            continue
        if residue.name.upper() in excluded:
            continue
        ligand_residues.append(residue)
    if not ligand_residues:
        raise ValueError("没有找到可识别的配体残基。")
    if len(ligand_residues) > 1:
        print("警告：检测到多个候选配体残基，将使用第一个：", residue_label(ligand_residues[0]))
    return ligand_residues[0]
