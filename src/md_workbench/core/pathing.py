from __future__ import annotations

from pathlib import Path
from copy import deepcopy

from ..config.models import WorkflowConfig


def resolve_under_root(root: str | Path, value: str) -> str:
    if value is None:
        return value
    raw = str(value).strip()
    if raw == "":
        return raw
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path.resolve())
    return str((Path(root) / path).resolve())




def relativize_under_root(root: str | Path, value: str) -> str:
    if value is None:
        return value
    raw = str(value).strip()
    if raw == "":
        return raw
    path = Path(raw).expanduser()
    root_path = Path(root).expanduser().resolve()
    try:
        resolved = path.resolve() if path.is_absolute() else (root_path / path).resolve()
    except Exception:
        return raw
    try:
        rel = resolved.relative_to(root_path)
        return "." if str(rel) == "." else str(rel)
    except Exception:
        return str(resolved)


def make_config_portable(cfg: WorkflowConfig) -> WorkflowConfig:
    cfg = deepcopy(cfg)
    workspace_root = Path(cfg.workspace_root).expanduser().resolve()
    cfg.workspace_root = "."

    prep_fields = ["receptor_input", "receptor_output"]
    docking_fields = [
        "ligand_sdf_input", "ligand_output_sdf", "receptor_pdbqt", "receptor_json", "ligand_pdbqt",
        "docking_pdbqt", "docking_sdf", "docking_log", "extracted_pose_sdf", "extracted_pose_pdb",
        "external_docking_sdf", "docking_box_config",
    ]
    run_fields = ["protein_pdb", "ligand_sdf", "output_root"]
    basic_fields = ["replica_root", "ligand_sdf", "analysis_root"]
    water_fields = ["replica_root", "analysis_root"]
    advanced_fields = ["replica_root", "analysis_root"]
    mmgbsa_fields = [
        "analysis_root", "source_root", "mmpbsa_input_file", "complex_solvated_prmtop", "complex_prmtop",
        "receptor_prmtop", "ligand_prmtop", "trajectory_nc", "final_dat", "final_csv", "per_frame_csv",
        "per_residue_dat", "per_residue_csv",
    ]
    bundle_fields = ["root"]

    for name in prep_fields:
        setattr(cfg.prep, name, relativize_under_root(workspace_root, getattr(cfg.prep, name)))
    for name in docking_fields:
        setattr(cfg.docking, name, relativize_under_root(workspace_root, getattr(cfg.docking, name)))
    for name in run_fields:
        setattr(cfg.run, name, relativize_under_root(workspace_root, getattr(cfg.run, name)))
    for name in basic_fields:
        setattr(cfg.basic, name, relativize_under_root(workspace_root, getattr(cfg.basic, name)))
    for name in water_fields:
        setattr(cfg.waterbridge, name, relativize_under_root(workspace_root, getattr(cfg.waterbridge, name)))
    for name in advanced_fields:
        setattr(cfg.advanced, name, relativize_under_root(workspace_root, getattr(cfg.advanced, name)))
    for name in mmgbsa_fields:
        setattr(cfg.mmgbsa, name, relativize_under_root(workspace_root, getattr(cfg.mmgbsa, name)))
    for name in bundle_fields:
        setattr(cfg.output_bundle, name, relativize_under_root(workspace_root, getattr(cfg.output_bundle, name)))
    return cfg

def infer_run_input_paths(cfg: WorkflowConfig) -> tuple[str, str]:
    protein = str(cfg.run.protein_pdb).strip()
    ligand = str(cfg.run.ligand_sdf).strip()
    if not protein:
        protein = str(cfg.prep.receptor_output).strip() if cfg.do_prep else str(cfg.prep.receptor_input).strip()
    if not ligand:
        candidates: list[str] = []
        docking_mode = str(cfg.docking.docking_mode).strip().lower()
        if cfg.do_prep and docking_mode in {"auto", "external"}:
            candidates.append(str(cfg.docking.extracted_pose_sdf).strip())
        if cfg.do_prep:
            candidates.append(str(cfg.docking.ligand_output_sdf).strip())
        else:
            if str(cfg.docking.ligand_input_mode).strip().lower() == "sdf":
                candidates.append(str(cfg.docking.ligand_sdf_input).strip())
            else:
                candidates.append(str(cfg.docking.ligand_output_sdf).strip())
        for cand in candidates:
            if cand:
                ligand = cand
                break
    return protein, ligand


def normalize_workflow_paths(cfg: WorkflowConfig) -> WorkflowConfig:
    workspace_root = Path(cfg.workspace_root).expanduser().resolve()
    cfg.workspace_root = str(workspace_root)

    prep_fields = ["receptor_input", "receptor_output"]
    docking_fields = [
        "ligand_sdf_input", "ligand_output_sdf", "receptor_pdbqt", "receptor_json", "ligand_pdbqt",
        "docking_pdbqt", "docking_sdf", "docking_log", "extracted_pose_sdf", "extracted_pose_pdb",
        "external_docking_sdf", "docking_box_config",
    ]
    run_fields = ["protein_pdb", "ligand_sdf", "output_root"]
    basic_fields = ["replica_root", "ligand_sdf", "analysis_root"]
    water_fields = ["replica_root", "analysis_root"]
    advanced_fields = ["replica_root", "analysis_root"]
    mmgbsa_fields = [
        "analysis_root", "source_root", "mmpbsa_input_file", "complex_solvated_prmtop", "complex_prmtop",
        "receptor_prmtop", "ligand_prmtop", "trajectory_nc", "final_dat", "final_csv", "per_frame_csv",
        "per_residue_dat", "per_residue_csv",
    ]
    bundle_fields = ["root"]

    for name in prep_fields:
        setattr(cfg.prep, name, resolve_under_root(workspace_root, getattr(cfg.prep, name)))
    for name in docking_fields:
        setattr(cfg.docking, name, resolve_under_root(workspace_root, getattr(cfg.docking, name)))
    for name in run_fields:
        setattr(cfg.run, name, resolve_under_root(workspace_root, getattr(cfg.run, name)))
    for name in basic_fields:
        setattr(cfg.basic, name, resolve_under_root(workspace_root, getattr(cfg.basic, name)))
    for name in water_fields:
        setattr(cfg.waterbridge, name, resolve_under_root(workspace_root, getattr(cfg.waterbridge, name)))
    for name in advanced_fields:
        setattr(cfg.advanced, name, resolve_under_root(workspace_root, getattr(cfg.advanced, name)))
    for name in mmgbsa_fields:
        setattr(cfg.mmgbsa, name, resolve_under_root(workspace_root, getattr(cfg.mmgbsa, name)))
    for name in bundle_fields:
        setattr(cfg.output_bundle, name, resolve_under_root(workspace_root, getattr(cfg.output_bundle, name)))
    return cfg
