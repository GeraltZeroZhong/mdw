from __future__ import annotations

from pathlib import Path
import csv
import json


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_input_file(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"找不到输入文件: {path.resolve()}")
    return path


def require_nonempty_file(path_str: str | Path, *, label: str | None = None, empty_hint: str | None = None) -> Path:
    path = Path(path_str)
    name = label or "文件"
    if not path.exists():
        raise FileNotFoundError(f"{name}不存在: {path.resolve()}")
    if not path.is_file():
        raise FileNotFoundError(f"{name}不是普通文件: {path.resolve()}")
    if path.stat().st_size <= 0:
        message = f"{name}为空文件: {path.resolve()}"
        if empty_hint:
            message = f"{message}。{empty_hint}"
        raise ValueError(message)
    return path


def save_csv(path: str | Path, header, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_dict_csv(path: str | Path, rows, fieldnames) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_dict_csv(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)



def resolve_replica_dirs(replica_root: str | Path, replica_glob: str):
    root = Path(replica_root)
    return sorted([path for path in root.glob(replica_glob) if path.is_dir()])
