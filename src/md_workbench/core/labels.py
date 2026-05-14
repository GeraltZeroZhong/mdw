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
    ligand_residues = find_ligand_residues(topology, excluded=excluded)
    if not ligand_residues:
        raise ValueError("没有找到可识别的配体残基。")
    if len(ligand_residues) > 1:
        print("警告：检测到多个候选配体残基，将使用第一个：", residue_label(ligand_residues[0]))
    return ligand_residues[0]


def find_ligand_residues(topology, excluded=None):
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
    return ligand_residues


def ligand_atom_indices_from_residues(ligand_residues) -> list[int]:
    return [int(atom.index) for residue in ligand_residues for atom in residue.atoms]


def ligand_heavy_atom_indices_from_residues(ligand_residues) -> list[int]:
    return [
        int(atom.index)
        for residue in ligand_residues
        for atom in residue.atoms
        if atom.element is not None and atom.element.symbol != "H"
    ]


def ligand_residue_summary(ligand_residues) -> str:
    residues = list(ligand_residues)
    if not residues:
        return ""
    if len(residues) == 1:
        return residue_label(residues[0])
    return f"{residue_label(residues[0])}..{residue_label(residues[-1])} ({len(residues)} residues)"
