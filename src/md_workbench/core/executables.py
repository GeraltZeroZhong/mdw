from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def current_env_bin_dir() -> Path:
    return Path(sys.executable).resolve().parent


def ensure_current_env_bin_on_path() -> None:
    env_bin = str(current_env_bin_dir())
    current = os.environ.get("PATH", "")
    entries = [item for item in current.split(os.pathsep) if item]
    if env_bin not in entries:
        os.environ["PATH"] = os.pathsep.join([env_bin, *entries]) if entries else env_bin


def _candidate_binary_names(name: str) -> list[str]:
    names = [name]
    pathext = [ext for ext in os.environ.get("PATHEXT", "").split(os.pathsep) if ext]
    if pathext and not Path(name).suffix:
        for ext in pathext:
            names.append(f"{name}{ext}")
    return names


def find_binary(name: str) -> str | None:
    ensure_current_env_bin_on_path()

    path = shutil.which(name)
    if path:
        return path

    env_bin = current_env_bin_dir()
    for candidate_name in _candidate_binary_names(name):
        candidate = env_bin / candidate_name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def require_binary(name: str) -> str:
    path = find_binary(name)
    if path:
        return path
    env_bin = current_env_bin_dir()
    raise RuntimeError(
        f"Required executable was not found: {name}. "
        f"Checked PATH and the active Python environment bin directory: {env_bin}"
    )
