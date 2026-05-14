from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openmm.app import ForceField, Modeller, PDBFile
from openff.toolkit.topology import Molecule
from openff.toolkit.utils import AmberToolsToolkitWrapper, RDKitToolkitWrapper, ToolkitRegistry
from openff.units.openmm import to_openmm
from openmmforcefields.generators import GAFFTemplateGenerator

from ..core import check_input_file, current_env_bin_dir, ensure_current_env_bin_on_path


@dataclass(frozen=True)
class LigandBuildInfo:
    source_path: str
    input_format: str
    output_residue_name: str = "LIG"
    ligand_chain_indices: tuple[int, ...] = ()


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


def _ligand_input_format(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".pdb", ".ent"}:
        return "pdb"
    if suffix in {".sdf", ".sd"}:
        return "sdf"
    raise ValueError(f"Unsupported ligand input format '{suffix}'. Use SDF for small molecules or PDB for peptide/polymer ligands.")


def _base_forcefield() -> ForceField:
    return ForceField(
        "amber/protein.ff14SB.xml",
        "amber/tip3p_standard.xml",
        "amber/tip3p_HFE_multivalent.xml",
    )


def _topology_has_hydrogen(topology) -> bool:
    for atom in topology.atoms():
        element = getattr(atom, "element", None)
        symbol = getattr(element, "symbol", "")
        if str(symbol).upper() == "H" or str(atom.name).upper().startswith("H"):
            return True
    return False


def tag_ligand_residues_for_output(topology, ligand_info: LigandBuildInfo) -> None:
    """Rename PDB ligand residues after force-field systems have been created.

    Peptide PDB ligands must keep their standard residue names while OpenMM
    matches AMBER templates. After systems are built, renaming the appended
    ligand chain to LIG makes downstream MDTraj selections treat it as ligand
    instead of receptor protein.
    """
    if ligand_info.input_format != "pdb":
        return
    ligand_chains = set(int(idx) for idx in ligand_info.ligand_chain_indices)
    if not ligand_chains:
        return
    for chain in topology.chains():
        if int(chain.index) not in ligand_chains:
            continue
        for residue in chain.residues():
            residue.name = ligand_info.output_residue_name


def build_modeller_and_forcefield(protein_pdb: str, ligand_sdf: str):
    protein_path = check_input_file(protein_pdb)
    ligand_path = check_input_file(ligand_sdf)

    protein = PDBFile(str(protein_path))
    ligand_format = _ligand_input_format(ligand_path)
    forcefield = _base_forcefield()

    if ligand_format == "pdb":
        ligand = PDBFile(str(ligand_path))
        modeller = Modeller(protein.topology, protein.positions)
        first_ligand_chain = len(list(modeller.topology.chains()))
        ligand_chain_count = len(list(ligand.topology.chains()))
        modeller.add(ligand.topology, ligand.positions)
        if not _topology_has_hydrogen(ligand.topology):
            try:
                modeller.addHydrogens(forcefield, pH=7.4)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to add hydrogens for the PDB ligand complex. "
                    "For peptide or cyclic-peptide ligands, provide a chemically complete PDB with standard AMBER-compatible "
                    "residue and atom names, correct CONECT records for noncanonical/cyclic bonds, and no terminal atoms "
                    "that conflict with the intended cyclization."
                ) from exc
        ligand_info = LigandBuildInfo(
            source_path=str(ligand_path),
            input_format="pdb",
            ligand_chain_indices=tuple(range(first_ligand_chain, first_ligand_chain + ligand_chain_count)),
        )
        return modeller, forcefield, ligand_info

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

    gaff = GAFFTemplateGenerator(molecules=[ligand_mol], forcefield="gaff-2.11")
    gaff.add_molecules(ligand_mol)
    forcefield.registerTemplateGenerator(gaff.generator)

    modeller = Modeller(protein.topology, protein.positions)
    modeller.add(ligand_top, ligand_pos)
    ligand_info = LigandBuildInfo(source_path=str(ligand_path), input_format="sdf")
    return modeller, forcefield, ligand_info
