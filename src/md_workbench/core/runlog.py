from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import threading
import traceback
from typing import Any

from .progress import ProgressEvent



def _now() -> datetime:
    return datetime.now(timezone.utc)



def _timestamp_for_name(dt: datetime) -> str:
    return dt.strftime("%Y%m%d_%H%M%S_%f")



def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            return str(obj)
    return str(obj)


@dataclass
class RunLogSession:
    workspace_root: Path
    run_type: str
    run_id: str
    started_at: str
    log_path: Path
    meta_path: Path
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _meta: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._meta.update(
            {
                "run_id": self.run_id,
                "run_type": self.run_type,
                "workspace_root": str(self.workspace_root),
                "started_at": self.started_at,
                "status": "running",
                "log_path": str(self.log_path),
            }
        )
        self._write_meta()
        self.log(f"Run started: {self.run_type}")
        self.log(f"Workspace root: {self.workspace_root}")

    def _append_line(self, line: str) -> None:
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def _write_meta(self) -> None:
        with self._lock:
            with open(self.meta_path, "w", encoding="utf-8") as handle:
                json.dump(self._meta, handle, ensure_ascii=False, indent=2, default=_json_default)

    def log(self, message: str, level: str = "INFO") -> None:
        timestamp = _now().isoformat()
        self._append_line(f"[{timestamp}] [{level}] {message}")

    def log_json(self, title: str, obj: Any, level: str = "INFO") -> None:
        self.log(title, level=level)
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default)
        for line in text.splitlines() or [""]:
            self._append_line(line)

    def log_progress(self, event: ProgressEvent) -> None:
        message = f"Progress {event.current}/{event.total} [{event.stage}]: {event.detail}"
        if event.subtotal > 0:
            message += f" | Subprogress {event.subcurrent}/{event.subtotal}: {event.subdetail or event.detail}"
            if event.subeta_seconds is not None:
                message += f" | ETA {event.subeta_seconds}s"
        self.log(message)

    def log_exception(self, exc: BaseException | None = None) -> None:
        if exc is not None:
            self.log(f"Exception: {exc}", level="ERROR")
        tb = traceback.format_exc()
        self.log("Traceback:", level="ERROR")
        for line in tb.rstrip().splitlines():
            self._append_line(line)

    def finalize(self, status: str, payload: Any | None = None, error: str | None = None) -> None:
        ended_at = _now().isoformat()
        self._meta["status"] = status
        self._meta["ended_at"] = ended_at
        if payload is not None:
            self._meta["payload"] = payload
        if error is not None:
            self._meta["error"] = error
        self._write_meta()
        self.log(f"Run finished with status: {status}", level="INFO" if status == "completed" else "ERROR")
        if error:
            self.log(error, level="ERROR")



def start_run_log(workspace_root: str | Path, run_type: str) -> RunLogSession:
    root = Path(str(workspace_root).strip() or ".").expanduser().resolve()
    logs_root = root / "logs" / "runs"
    logs_root.mkdir(parents=True, exist_ok=True)
    started = _now()
    run_id = f"{_timestamp_for_name(started)}_{run_type}"
    log_path = logs_root / f"{run_id}.log"
    meta_path = logs_root / f"{run_id}.json"
    return RunLogSession(
        workspace_root=root,
        run_type=run_type,
        run_id=run_id,
        started_at=started.isoformat(),
        log_path=log_path,
        meta_path=meta_path,
    )
