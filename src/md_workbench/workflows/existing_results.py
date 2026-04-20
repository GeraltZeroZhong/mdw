from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from ..config import WorkflowConfig
from ..core import normalize_workflow_paths, preflight_validate_existing_results
from .md import run_full_md_workflow


def _pick_ligand_sdf(cfg: WorkflowConfig) -> str:
    candidates = [
        cfg.run.ligand_sdf,
        cfg.docking.extracted_pose_sdf,
        cfg.docking.ligand_output_sdf,
        cfg.docking.ligand_sdf_input,
        cfg.basic.ligand_sdf,
    ]
    normalized_candidates = [str(value).strip() for value in candidates if str(value).strip()]
    for candidate in normalized_candidates:
        if Path(candidate).exists():
            return str(Path(candidate).resolve())
    return normalized_candidates[0] if normalized_candidates else str(Path(cfg.basic.ligand_sdf).resolve())


def prepare_existing_results_workflow_config(cfg: WorkflowConfig) -> WorkflowConfig:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    cfg.do_prep = False
    cfg.do_run_md = False

    replica_root = str(Path(cfg.run.output_root).resolve())
    cfg.basic.replica_root = replica_root
    cfg.basic.ligand_sdf = _pick_ligand_sdf(cfg)
    cfg.basic.timestep_ps = cfg.run.timestep_ps
    cfg.basic.dcd_interval_steps = cfg.run.dcd_interval

    cfg.waterbridge.replica_root = replica_root
    cfg.waterbridge.timestep_ps = cfg.run.timestep_ps
    cfg.waterbridge.dcd_interval_steps = cfg.run.dcd_interval

    cfg.advanced.replica_root = replica_root
    cfg.mmgbsa.source_root = replica_root
    return cfg


def run_existing_results_workflow(cfg: WorkflowConfig, progress_callback=None):
    cfg = prepare_existing_results_workflow_config(cfg)
    validation = preflight_validate_existing_results(cfg)
    if validation.errors:
        bullets = "\n".join(f"- {item}" for item in validation.errors)
        raise ValueError(f"Existing-results workflow preflight validation failed:\n{bullets}")
    return run_full_md_workflow(cfg, progress_callback=progress_callback)
