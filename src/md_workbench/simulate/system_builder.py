from __future__ import annotations

from pathlib import Path

from openmm.app import ForceField, Modeller, PDBFile
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import AmberToolsToolkitWrapper, RDKitToolkitWrapper, ToolkitRegistry
from openff.units.openmm import to_openmm
from openmmforcefields.generators import GAFFTemplateGenerator

from ..core import check_input_file, current_env_bin_dir, ensure_current_env_bin_on_path


def load_single_molecule_from_sdf(path_str: str) -> Molecule:
    mol = Molecule.from_file(path_str)
    if isinstance(mol, list):
        if len(mol) == 0:
            raise ValueError(f"SDF 中没有读到分子: {path_str}")
        if len(mol) > 1:
            raise ValueError(
                f"{path_str} 中读到 {len(mol)} 个分子。MD 建模阶段要求输入 SDF 恰好包含 1 个配体分子。"
            )
        mol = mol[0]
    return mol


def _assign_ligand_am1bcc_charges(ligand_mol: Molecule) -> None:
    ensure_current_env_bin_on_path()
    if not AmberToolsToolkitWrapper.is_available():
        env_bin = current_env_bin_dir()
        antechamber = env_bin / "antechamber"
        raise RuntimeError(
            "AM1-BCC charge assignment requires AmberToolsToolkitWrapper, but it is not available. "
            f"Expected antechamber under the active Python environment, for example: {antechamber}"
        )

    registry = ToolkitRegistry(toolkit_precedence=[AmberToolsToolkitWrapper, RDKitToolkitWrapper])
    ligand_mol.assign_partial_charges(
        partial_charge_method="am1bcc",
        toolkit_registry=registry,
        normalize_partial_charges=True,
    )


def build_modeller_and_forcefield(protein_pdb: str, ligand_sdf: str):
    protein_path = check_input_file(protein_pdb)
    ligand_path = check_input_file(ligand_sdf)

    protein = PDBFile(str(protein_path))
    ligand_mol = load_single_molecule_from_sdf(str(ligand_path))
    if len(ligand_mol.conformers) == 0:
        raise ValueError("配体 SDF 缺少 3D conformer。")
    _assign_ligand_am1bcc_charges(ligand_mol)

    ligand_top = ligand_mol.to_topology().to_openmm()
    ligand_pos = to_openmm(ligand_mol.conformers[0])
    try:
        for residue in ligand_top.residues():
            residue.name = "LIG"
    except Exception:
        pass

    forcefield = ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )
    gaff = GAFFTemplateGenerator(molecules=[ligand_mol], forcefield="gaff-2.11")
    gaff.add_molecules(ligand_mol)
    forcefield.registerTemplateGenerator(gaff.generator)

    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(ligand_top, ligand_pos)
    return modeller, forcefield
