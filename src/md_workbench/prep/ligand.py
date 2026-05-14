from __future__ import annotations

from pathlib import Path
import shutil

from rdkit import Chem
from rdkit.Chem import AllChem

from ..core import check_input_file


def _read_sdf_molecules(sdf_path: str, error_message: str = "No valid molecules were read from the SDF file.") -> list[Chem.Mol]:
    check_input_file(sdf_path)
    molecules = [m for m in Chem.SDMolSupplier(str(sdf_path), removeHs=False) if m is not None]
    if not molecules:
        raise ValueError(error_message)
    return molecules


def validate_sdf_has_molecules(sdf_path: str, error_message: str = "No valid molecules were read from the SDF file.") -> str:
    _read_sdf_molecules(sdf_path, error_message=error_message)
    return str(sdf_path)


def validate_pdb_has_atoms(pdb_path: str, error_message: str = "No ATOM/HETATM records were read from the ligand PDB file.") -> str:
    path = check_input_file(pdb_path)
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                return str(path)
    raise ValueError(error_message)


def _read_single_input_ligand_molecule(sdf_path: str, error_message: str) -> Chem.Mol:
    molecules = _read_sdf_molecules(sdf_path, error_message=error_message)
    if len(molecules) != 1:
        raise ValueError(
            f"Expected exactly one ligand molecule in {sdf_path}, but found {len(molecules)}. "
            "Please split the SDF so that the ligand input contains a single molecule."
        )
    return Chem.Mol(molecules[0])


def extract_pose1(docking_sdf: str, out_sdf: str, out_pdb: str | None = None) -> str:
    molecules = _read_sdf_molecules(docking_sdf, error_message="No valid molecules were read from docking_results.sdf.")
    mol = molecules[0]
    Path(out_sdf).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(out_sdf)
    writer.write(mol)
    writer.close()
    if out_pdb:
        Path(out_pdb).parent.mkdir(parents=True, exist_ok=True)
        Chem.MolToPDBFile(mol, out_pdb)
    return out_sdf


def smiles_to_sdf(smiles: str, out_sdf: str, random_seed: int = 42) -> str:
    if not smiles.strip():
        raise ValueError("Ligand SMILES is empty.")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Failed to parse the ligand SMILES string.")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = random_seed
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("3D conformer generation failed.")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol)
    Path(out_sdf).parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(out_sdf)
    writer.write(mol)
    writer.close()
    return out_sdf


def prepare_ligand_from_sdf(input_sdf: str, out_sdf: str) -> str:
    src = check_input_file(input_sdf)
    mol = _read_single_input_ligand_molecule(
        str(src),
        error_message="No valid molecules were read from the ligand SDF input.",
    )

    # Meeko expects 3D coordinates and all hydrogens as real atoms.
    # Add missing hydrogens even when the input already has a conformer so that
    # the downstream PDBQT preparation follows the documented contract.
    if not any(atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()):
        mol = Chem.AddHs(mol, addCoords=True)

    if mol.GetNumConformers() == 0:
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        status = AllChem.EmbedMolecule(mol, params)
        if status != 0:
            raise RuntimeError("The ligand SDF does not contain 3D coordinates and conformer generation failed.")
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            AllChem.UFFOptimizeMolecule(mol)

    dst = Path(out_sdf)
    dst.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(dst))
    writer.write(mol)
    writer.close()
    return str(dst)


def prepare_ligand_from_pdb(input_pdb: str, out_pdb: str) -> str:
    src = Path(validate_pdb_has_atoms(input_pdb))
    dst = Path(out_pdb)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return str(dst)


def write_docking_box_config(center_xyz: tuple[float, float, float], size_xyz: tuple[float, float, float], out_path: str) -> str:
    lines = [
        f"center_x = {center_xyz[0]:.3f}",
        f"center_y = {center_xyz[1]:.3f}",
        f"center_z = {center_xyz[2]:.3f}",
        f"size_x = {size_xyz[0]:.3f}",
        f"size_y = {size_xyz[1]:.3f}",
        f"size_z = {size_xyz[2]:.3f}",
    ]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_path
