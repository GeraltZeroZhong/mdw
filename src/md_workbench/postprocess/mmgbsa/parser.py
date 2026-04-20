from __future__ import annotations

from pathlib import Path
import csv
import re
from typing import Iterable


def parse_mmpbsa_results(input_path: str | Path, output_path: str | Path | None = None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name("mmpbsa_summary_parsed.csv")
    output_path = Path(output_path)

    patterns = {
        "DELTA TOTAL": r"DELTA TOTAL\s+([-\d\.]+)\s+([-\d\.]+)",
        "DELTA G gas": r"DELTA G gas\s+([-\d\.]+)\s+([-\d\.]+)",
        "DELTA G solv": r"DELTA G solv\s+([-\d\.]+)\s+([-\d\.]+)",
    }

    rows = []
    text = input_path.read_text(encoding="utf-8", errors="ignore")
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            rows.append({"term": key, "mean_kcal_mol": float(m.group(1)), "sd_kcal_mol": float(m.group(2))})

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["term", "mean_kcal_mol", "sd_kcal_mol"])
        w.writeheader()
        w.writerows(rows)
    return output_path


def read_csv_rows(path: str | Path):
    with open(path, "r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def infer_numeric_columns(rows: list[dict[str, str]]):
    if not rows:
        return []
    cols = []
    keys = list(rows[0].keys())
    for key in keys:
        vals = [r.get(key, "") for r in rows[: min(len(rows), 20)]]
        if vals and sum(_is_float(v) for v in vals if v != "") >= max(1, len([v for v in vals if v != ""]) // 2):
            cols.append(key)
    return cols
