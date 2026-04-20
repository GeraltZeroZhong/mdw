from __future__ import annotations

from pathlib import Path

from ..config.models import WorkflowConfig

PROJECT_CONFIG_FILENAME = "project_config.json"
DEFAULT_PROJECT_SUBDIRS = [
    "inputs",
    "work/prep",
    "work/docking",
    "work/md",
    "work/analysis/basic",
    "work/analysis/waterbridge",
    "work/analysis/advanced",
    "work/analysis/mmgbsa",
    "results",
    "logs/runs",
]


def project_config_path(project_root: str | Path) -> Path:
    return Path(project_root).expanduser().resolve() / PROJECT_CONFIG_FILENAME


def ensure_project_layout(project_root: str | Path) -> Path:
    project_root = Path(project_root).expanduser().resolve()
    project_root.mkdir(parents=True, exist_ok=True)
    for rel in DEFAULT_PROJECT_SUBDIRS:
        (project_root / rel).mkdir(parents=True, exist_ok=True)
    return project_root


def initialize_project_config(project_root: str | Path, cfg: WorkflowConfig | None = None) -> WorkflowConfig:
    cfg = cfg or WorkflowConfig()
    cfg.workspace_root = str(ensure_project_layout(project_root))
    return cfg
