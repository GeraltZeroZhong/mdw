from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


REPLICA_COMPLETE_REQUIRED_FILES = (
    "system_solvated.pdb",
    "trajectory.dcd",
    "md.log",
    "final_frame.pdb",
)

REPLICA_GENERATED_FILES = (
    "complex_from_components.pdb",
    "system_solvated.pdb",
    "minimized_solvated.pdb",
    "equilibrated_solvated.pdb",
    "final_frame.pdb",
    "trajectory.dcd",
    "md.log",
    "platform_info.json",
    "replica_status.json",
    "complex.prmtop",
    "complex.inpcrd",
    "complex_solvated.prmtop",
    "complex_solvated.inpcrd",
    "receptor.prmtop",
    "receptor.inpcrd",
    "ligand.prmtop",
    "ligand.inpcrd",
    "complex.nc",
)


def replica_dir(output_root: str | Path, replica_id: int) -> Path:
    return Path(output_root) / f"replica_{int(replica_id)}"


def replica_id_from_name(name: str) -> int | None:
    match = re.fullmatch(r"replica_(\d+)", str(name).strip())
    if match is None:
        return None
    return int(match.group(1))


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def missing_required_replica_files(replica_path: str | Path) -> list[str]:
    root = Path(replica_path)
    return [name for name in REPLICA_COMPLETE_REQUIRED_FILES if not _nonempty_file(root / name)]


def _read_replica_status(replica_path: Path) -> dict[str, Any] | None:
    status_path = replica_path / "replica_status.json"
    if not status_path.exists():
        return None
    try:
        with open(status_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def is_replica_complete(replica_path: str | Path) -> bool:
    root = Path(replica_path)
    if not root.is_dir():
        return False
    if missing_required_replica_files(root):
        return False
    status = _read_replica_status(root)
    if status is not None and status.get("status") != "completed":
        return False
    return True


def clear_replica_generated_outputs(replica_path: str | Path) -> None:
    root = Path(replica_path)
    for name in REPLICA_GENERATED_FILES:
        path = root / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def write_replica_status(replica_path: str | Path, replica_id: int, payload: dict[str, Any] | None = None) -> Path:
    root = Path(replica_path)
    root.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "completed",
        "replica_id": int(replica_id),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "required_files": list(REPLICA_COMPLETE_REQUIRED_FILES),
    }
    if payload:
        status.update(payload)
    status_path = root / "replica_status.json"
    with open(status_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, ensure_ascii=False, indent=2)
    return status_path


def summarize_replica_status(output_root: str | Path, target_n_replicas: int) -> dict[str, Any]:
    target = max(int(target_n_replicas), 1)
    root = Path(output_root)
    replicas = []
    completed_ids: list[int] = []
    next_id: int | None = None
    for replica_id in range(1, target + 1):
        path = replica_dir(root, replica_id)
        missing = missing_required_replica_files(path)
        completed = is_replica_complete(path)
        if completed:
            completed_ids.append(replica_id)
        elif next_id is None:
            next_id = replica_id
        replicas.append(
            {
                "replica_id": replica_id,
                "name": path.name,
                "path": str(path.resolve()),
                "exists": path.is_dir(),
                "completed": completed,
                "missing_required_files": missing,
            }
        )
    return {
        "output_root": str(root.resolve()),
        "target_n_replicas": target,
        "completed_replica_ids": completed_ids,
        "completed_replicas": len(completed_ids),
        "remaining_replicas": target - len(completed_ids),
        "next_replica_id": next_id,
        "all_replicas_completed": next_id is None,
        "replicas": replicas,
    }


def next_pending_replica_id(output_root: str | Path, target_n_replicas: int) -> int | None:
    return summarize_replica_status(output_root, target_n_replicas)["next_replica_id"]
