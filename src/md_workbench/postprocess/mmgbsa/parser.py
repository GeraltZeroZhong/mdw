from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import csv
import math
import re
import statistics
from typing import Iterable


_FLOAT_PATTERN = r"[+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
_SUMMARY_ROW_RE = re.compile(
    rf"^(?P<term>.*?)\s+(?P<mean>{_FLOAT_PATTERN})\s+(?P<sd>{_FLOAT_PATTERN})\s+(?P<sem>{_FLOAT_PATTERN})$",
    re.IGNORECASE,
)
_SUMMARY_SECTIONS = {
    "Complex": "Complex",
    "Receptor": "Receptor",
    "Ligand": "Ligand",
    "Differences (Complex - Receptor - Ligand)": "Differences",
}
_DECOMP_COMPONENTS = [
    ("internal", "Internal"),
    ("vdw", "van der Waals"),
    ("electrostatic", "Electrostatic"),
    ("polar_solvation", "Polar Solvation"),
    ("nonpolar_solvation", "Non-Polar Solv."),
    ("total", "TOTAL"),
]


def _parse_float(value: str) -> float:
    return float(value.strip())


def _next_nonempty_index(lines: list[str], start: int) -> int | None:
    for index in range(start, len(lines)):
        if lines[index].strip():
            return index
    return None


def _read_csv_block(lines: list[str], header_index: int) -> list[dict[str, str]]:
    header = next(csv.reader([lines[header_index]]))
    rows: list[dict[str, str]] = []
    for index in range(header_index + 1, len(lines)):
        line = lines[index]
        if not line.strip():
            break
        values = next(csv.reader([line]))
        if len(values) < len(header):
            values.extend([""] * (len(header) - len(values)))
        if len(values) > len(header):
            values = values[: len(header)]
        rows.append(dict(zip(header, values)))
    return rows


def _clean_residue_label(value: str) -> str:
    return " ".join(str(value).split())


def _coerce_energy_value(row: dict[str, str], target_column: str, component_columns: Iterable[str]) -> float | None:
    raw = str(row.get(target_column, "")).strip()
    if raw:
        try:
            value = _parse_float(raw)
        except Exception:
            value = None
        else:
            if math.isfinite(value):
                return value
    component_values: list[float] = []
    for key in component_columns:
        raw_component = str(row.get(key, "")).strip()
        if not raw_component:
            return None
        try:
            component_value = _parse_float(raw_component)
        except Exception:
            return None
        if not math.isfinite(component_value):
            return None
        component_values.append(component_value)
    return float(sum(component_values)) if component_values else None


def parse_mmpbsa_summary_rows(input_path: str | Path, section: str | None = "Differences") -> list[dict[str, object]]:
    lines = Path(input_path).read_text(encoding="utf-8", errors="ignore").splitlines()
    current_section: str | None = None
    rows: list[dict[str, object]] = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("|") or set(stripped) == {"-"}:
            continue
        if stripped.endswith(":"):
            label = stripped[:-1]
            current_section = _SUMMARY_SECTIONS.get(label)
            continue
        if current_section is None:
            continue
        match = _SUMMARY_ROW_RE.match(stripped)
        if not match:
            continue
        rows.append(
            {
                "section": current_section,
                "term": match.group("term").strip(),
                "mean_kcal_mol": _parse_float(match.group("mean")),
                "sd_kcal_mol": _parse_float(match.group("sd")),
                "sem_kcal_mol": _parse_float(match.group("sem")),
            }
        )
    if section is None:
        return rows
    return [row for row in rows if row.get("section") == section]


def parse_mmpbsa_results(input_path: str | Path, output_path: str | Path | None = None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_name("mmpbsa_summary_parsed.csv")
    output_path = Path(output_path)

    rows = parse_mmpbsa_summary_rows(input_path, section="Differences")
    if not rows:
        rows = parse_mmpbsa_summary_rows(input_path, section=None)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["section", "term", "mean_kcal_mol", "sd_kcal_mol", "sem_kcal_mol"])
        w.writeheader()
        w.writerows(rows)
    return output_path


def parse_mmpbsa_csv_section_rows(path: str | Path, section_name: str) -> list[dict[str, str]]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    for index, line in enumerate(lines):
        if line.strip() != section_name:
            continue
        header_index = _next_nonempty_index(lines, index + 1)
        if header_index is None:
            return []
        return _read_csv_block(lines, header_index)
    return []


def parse_mmpbsa_final_results_delta_rows(path: str | Path) -> list[dict[str, str]]:
    return parse_mmpbsa_csv_section_rows(path, "DELTA Energy Terms")


def parse_mmpbsa_decomp_delta_frame_rows(path: str | Path) -> list[dict[str, str]]:
    return parse_mmpbsa_csv_section_rows(path, "DELTA,Total Energy Decomposition:")


def summarize_mmpbsa_decomp_frame_rows(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    component_columns = ["Internal", "van der Waals", "Electrostatic", "Polar Solvation", "Non-Polar Solv."]
    for row in rows:
        label = _clean_residue_label(row.get("Residue", ""))
        location = _clean_residue_label(row.get("Location", ""))
        if not label:
            continue
        value = _coerce_energy_value(row, "TOTAL", component_columns)
        if value is None or not math.isfinite(value):
            continue
        grouped[(label, location)].append(value)
    summary_rows: list[dict[str, object]] = []
    for (label, location), values in grouped.items():
        summary_rows.append(
            {
                "label": label,
                "location": location,
                "mean_kcal_mol": statistics.fmean(values),
                "sd_kcal_mol": statistics.stdev(values) if len(values) > 1 else 0.0,
                "n_frames": len(values),
            }
        )
    return summary_rows


def parse_mmpbsa_decomp_summary_rows(path: str | Path, section_name: str = "DELTAS") -> list[dict[str, object]]:
    lines = Path(path).read_text(encoding="utf-8", errors="ignore").splitlines()
    in_target_section = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped == f"{section_name}:":
            in_target_section = True
            continue
        if not in_target_section or stripped != "Total Energy Decomposition:":
            continue
        header_index = _next_nonempty_index(lines, index + 1)
        second_header_index = _next_nonempty_index(lines, (header_index or index) + 1)
        data_index = _next_nonempty_index(lines, (second_header_index or index) + 1)
        if header_index is None or second_header_index is None or data_index is None:
            return []
        rows: list[dict[str, object]] = []
        for current in range(data_index, len(lines)):
            raw_line = lines[current]
            if not raw_line.strip():
                break
            values = next(csv.reader([raw_line]))
            if len(values) < 20:
                continue
            row: dict[str, object] = {
                "section": section_name,
                "residue": _clean_residue_label(values[0]),
                "location": _clean_residue_label(values[1]),
            }
            cursor = 2
            for prefix, _ in _DECOMP_COMPONENTS:
                row[f"{prefix}_avg"] = _parse_float(values[cursor])
                row[f"{prefix}_sd"] = _parse_float(values[cursor + 1])
                row[f"{prefix}_sem"] = _parse_float(values[cursor + 2])
                cursor += 3
            rows.append(row)
        return rows
    return []


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
