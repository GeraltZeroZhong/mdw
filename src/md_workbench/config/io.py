from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from .models import (
    AdvancedAnalysisConfig,
    BasicAnalysisConfig,
    DockingConfig,
    MMGBSAConfig,
    OutputBundleConfig,
    PlotSelectionConfig,
    PlotStyleConfig,
    PrepConfig,
    RunConfig,
    WaterBridgeConfig,
    WorkflowConfig,
)

T = TypeVar("T")


def _coerce_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
    return cls(**valid)


def _workspace_root_candidate_score(root: Path, config_path: Path) -> int:
    score = 0
    if config_path.name == "project_config.json":
        if config_path == (root / "project_config.json"):
            score += 8
        if config_path == (root / "inputs" / "project_config.json"):
            score += 10
    for rel_path, weight in [
        ("inputs", 2),
        ("work", 2),
        ("results", 2),
        ("logs", 2),
        ("work/analysis", 1),
        ("logs/runs", 1),
    ]:
        if (root / rel_path).exists():
            score += weight
    return score


def _resolve_workspace_root(config_path: Path, raw_workspace_root: str) -> str:
    config_path = config_path.expanduser().resolve()
    raw = str(raw_workspace_root).strip()
    if raw not in {"", "."}:
        workspace_root = Path(raw).expanduser()
        if workspace_root.is_absolute():
            return str(workspace_root.resolve())
        return str((config_path.parent / workspace_root).resolve())

    if config_path.name != "project_config.json":
        return str(config_path.parent)

    candidates = [config_path.parent, *config_path.parent.parents]
    best_root = config_path.parent
    best_score = -1
    for candidate in candidates:
        score = _workspace_root_candidate_score(candidate, config_path)
        if score > best_score:
            best_root = candidate
            best_score = score
    return str(best_root)


def workflow_from_dict(data: dict[str, Any]) -> WorkflowConfig:
    cfg = WorkflowConfig()
    if "prep" in data:
        cfg.prep = _coerce_dataclass(PrepConfig, data["prep"])

    docking_data = dict(data.get("docking", {})) if isinstance(data.get("docking"), dict) else {}
    legacy = data.get("prep", {}) if isinstance(data.get("prep"), dict) else {}

    if "smiles" in legacy and "ligand_smiles" not in docking_data:
        docking_data["ligand_smiles"] = legacy.get("smiles", "")
    if "smiles_output_sdf" in legacy and "ligand_output_sdf" not in docking_data:
        docking_data["ligand_output_sdf"] = legacy.get("smiles_output_sdf", "prepared_ligand.sdf")
    if "do_smiles_to_sdf" in legacy and legacy.get("do_smiles_to_sdf") and "ligand_input_mode" not in docking_data:
        docking_data["ligand_input_mode"] = "smiles"
    if "smiles" in docking_data and "ligand_smiles" not in docking_data:
        docking_data["ligand_smiles"] = docking_data.get("smiles", "")
    if "smiles_output_sdf" in docking_data and "ligand_output_sdf" not in docking_data:
        docking_data["ligand_output_sdf"] = docking_data.get("smiles_output_sdf", "prepared_ligand.sdf")
    if "do_smiles_to_sdf" in docking_data and docking_data.get("do_smiles_to_sdf") and "ligand_input_mode" not in docking_data:
        docking_data["ligand_input_mode"] = "smiles"
    if "do_extract_pose1" in docking_data and docking_data.get("do_extract_pose1") and "docking_mode" not in docking_data:
        docking_data["docking_mode"] = "external"
    if "external_docking_sdf" not in docking_data and "docking_sdf" in docking_data and docking_data.get("docking_mode") == "external":
        docking_data["external_docking_sdf"] = docking_data.get("docking_sdf", "")

    cfg.docking = _coerce_dataclass(DockingConfig, docking_data)

    if "run" in data:
        cfg.run = _coerce_dataclass(RunConfig, data["run"])
    if "basic" in data:
        cfg.basic = _coerce_dataclass(BasicAnalysisConfig, data["basic"])
    if "waterbridge" in data:
        cfg.waterbridge = _coerce_dataclass(WaterBridgeConfig, data["waterbridge"])
    if "advanced" in data:
        cfg.advanced = _coerce_dataclass(AdvancedAnalysisConfig, data["advanced"])
    if "mmgbsa" in data:
        cfg.mmgbsa = _coerce_dataclass(MMGBSAConfig, data["mmgbsa"])
    if "plot_selection" in data:
        cfg.plot_selection = _coerce_dataclass(PlotSelectionConfig, data["plot_selection"])
    if "plot_style" in data:
        cfg.plot_style = _coerce_dataclass(PlotStyleConfig, data["plot_style"])
    if "output_bundle" in data:
        cfg.output_bundle = _coerce_dataclass(OutputBundleConfig, data["output_bundle"])
    if "workspace_root" in data:
        cfg.workspace_root = data["workspace_root"]
    for key in [
        "do_prep",
        "do_run_md",
        "do_basic_analysis",
        "do_waterbridge_analysis",
        "do_advanced_analysis",
        "do_mmgbsa_postprocess",
    ]:
        if key in data:
            setattr(cfg, key, data[key])
    return cfg


def load_workflow_config(path: str | Path) -> WorkflowConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        cfg = workflow_from_dict(json.load(handle))
    cfg.workspace_root = _resolve_workspace_root(path, cfg.workspace_root)
    return cfg


def save_workflow_config(cfg: WorkflowConfig, path: str | Path) -> None:
    from ..core.pathing import make_config_portable
    portable = make_config_portable(cfg)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(portable.to_dict(), handle, ensure_ascii=False, indent=2)
