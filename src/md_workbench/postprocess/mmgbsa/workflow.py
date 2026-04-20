from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from ...config import MMGBSAConfig, PlotSelectionConfig, PlotStyleConfig
from ...core import ensure_dir, find_binary, write_dict_csv, write_json
from ...plotting.mmgbsa import (
    plot_mmgbsa_delta_total_distribution,
    plot_mmgbsa_delta_total_heatmap,
    plot_mmgbsa_replica_summary,
    plot_mmgbsa_summary,
    plot_mmgbsa_timeseries,
    plot_per_residue_decomp,
)
from .parser import infer_numeric_columns, parse_mmpbsa_results, read_csv_rows


def _pick_time_or_frame(rows, numeric_cols):
    for preferred in ["time_ns", "time", "frame", "Frame", "Time(ns)"]:
        if preferred in numeric_cols:
            return preferred
    return numeric_cols[0] if numeric_cols else None


def _pick_energy_column(rows, numeric_cols):
    preferred_names = [
        "DELTA TOTAL", "DeltaTOTAL", "TOTAL", "total", "delta_total", "TOTAL_ENERGY", "Energy", "DG",
    ]
    for name in preferred_names:
        if name in numeric_cols:
            return name
    numeric_wo_time = [c for c in numeric_cols if c not in {"time_ns", "time", "frame", "Frame", "Time(ns)"}]
    return numeric_wo_time[0] if numeric_wo_time else None


def _binary(name: str) -> str | None:
    return find_binary(name)


def _write_default_input(path: Path, cfg: MMGBSAConfig) -> Path:
    text = (
        "&general\n"
        f"  startframe={int(cfg.startframe)}, interval={int(cfg.interval)}, verbose=1, keep_files=0,\n"
        "/\n"
        "&gb\n"
        f"  igb={int(cfg.igb)}, saltcon={float(cfg.saltcon):.3f},\n"
        "/\n"
        "&decomp\n"
        f"  idecomp={int(cfg.idecomp)}, dec_verbose=0,\n"
        "/\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


def _run_command(args: list[str], cwd: str | None = None) -> tuple[int, str, str]:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _auto_run_replica(replica_dir: Path, out_dir: Path, cfg: MMGBSAConfig) -> dict:
    required = {
        "complex_solvated_prmtop": replica_dir / cfg.complex_solvated_prmtop,
        "complex_prmtop": replica_dir / cfg.complex_prmtop,
        "receptor_prmtop": replica_dir / cfg.receptor_prmtop,
        "ligand_prmtop": replica_dir / cfg.ligand_prmtop,
        "trajectory_nc": replica_dir / cfg.trajectory_nc,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return {"status": "skipped_missing_inputs", "missing": missing}

    binary = _binary("MMPBSA.py") or _binary("MMPBSA.py.MPI")
    if not binary:
        return {"status": "skipped_missing_binary", "missing": ["MMPBSA.py"]}

    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / Path(cfg.mmpbsa_input_file).name
    _write_default_input(input_file, cfg)

    final_dat = out_dir / Path(cfg.final_dat).name
    final_csv = out_dir / Path(cfg.final_csv).name
    decomp_dat = out_dir / Path(cfg.per_residue_dat).name
    decomp_csv = out_dir / Path(cfg.per_residue_csv).name

    args = [
        binary,
        "-O",
        "-i", str(input_file),
        "-sp", str(required["complex_solvated_prmtop"]),
        "-cp", str(required["complex_prmtop"]),
        "-rp", str(required["receptor_prmtop"]),
        "-lp", str(required["ligand_prmtop"]),
        "-y", str(required["trajectory_nc"]),
        "-o", str(final_dat),
        "-eo", str(final_csv),
        "-do", str(decomp_dat),
        "-deo", str(decomp_csv),
    ]
    code, stdout, stderr = _run_command(args, cwd=str(out_dir))
    result = {
        "status": "ok" if code == 0 else "failed",
        "command": args,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": code,
        "outputs": {
            "final_dat": str(final_dat),
            "final_csv": str(final_csv),
            "per_residue_dat": str(decomp_dat),
            "per_residue_csv": str(decomp_csv),
        },
    }
    return result


def _parse_summary_table(replica_out_dir: Path, cfg: MMGBSAConfig) -> list[dict] | None:
    final_dat = replica_out_dir / Path(cfg.final_dat).name
    summary_csv = replica_out_dir / "mmpbsa_summary_parsed.csv"
    if final_dat.exists():
        parse_mmpbsa_results(final_dat, summary_csv)
    if summary_csv.exists():
        rows = read_csv_rows(summary_csv)
        return rows or None
    return None


def _parse_per_frame(replica_out_dir: Path, cfg: MMGBSAConfig):
    path = replica_out_dir / Path(cfg.final_csv).name
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    numeric_cols = infer_numeric_columns(rows)
    if not numeric_cols:
        return None
    time_col = _pick_time_or_frame(rows, numeric_cols)
    energy_col = _pick_energy_column(rows, numeric_cols)
    if not time_col or not energy_col:
        return None
    try:
        x = [float(r[time_col]) for r in rows]
        y = [float(r[energy_col]) for r in rows]
    except Exception:
        return None
    return {"x": x, "y": y, "xlabel": time_col, "energy_col": energy_col}


def _parse_per_residue(replica_out_dir: Path, cfg: MMGBSAConfig):
    path = replica_out_dir / Path(cfg.per_residue_csv).name
    if not path.exists():
        return None
    rows = read_csv_rows(path)
    if not rows:
        return None
    numeric_cols = infer_numeric_columns(rows)
    if not numeric_cols:
        return None
    keys = list(rows[0].keys())
    text_col = keys[0]
    target = _pick_energy_column(rows, numeric_cols)
    if not target:
        return None
    parsed = []
    for row in rows:
        label = row.get(text_col, "")
        try:
            value = float(row.get(target, ""))
        except Exception:
            continue
        parsed.append({"label": label, "value": value})
    return parsed or None


def _combined_outputs(
    results: list[dict],
    analysis_root: Path,
    cfg: MMGBSAConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
) -> dict:
    combined_dir = ensure_dir(analysis_root / "combined")
    plot_summary = plot_selection is None or plot_selection.enabled("mmgbsa_summary")
    plot_per_frame = plot_selection is None or plot_selection.enabled("mmgbsa_per_frame")
    plot_per_residue = plot_selection is None or plot_selection.enabled("mmgbsa_per_residue")
    summary_rows = []
    dist_map = {}
    time_map = {}
    time_x = None
    xlabel = "frame"
    per_residue_map: dict[str, dict[str, float]] = {}

    for item in results:
        name = item["replica"]
        if item.get("summary_rows"):
            delta = next((r for r in item["summary_rows"] if r.get("term") == "DELTA TOTAL"), None)
            if delta:
                summary_rows.append({
                    "replica": name,
                    "mean_kcal_mol": float(delta["mean_kcal_mol"]),
                    "sd_kcal_mol": float(delta.get("sd_kcal_mol", 0.0)),
                })
        if item.get("per_frame"):
            pf = item["per_frame"]
            time_map[name] = pf["y"]
            dist_map[name] = pf["y"]
            if time_x is None:
                time_x = pf["x"]
                xlabel = pf["xlabel"]
        if item.get("per_residue"):
            per_residue_map[name] = {row["label"]: row["value"] for row in item["per_residue"]}

    outputs: dict[str, str] = {}
    if summary_rows:
        write_dict_csv(combined_dir / "mmgbsa_delta_total_summary.csv", summary_rows, ["replica", "mean_kcal_mol", "sd_kcal_mol"])
        if plot_summary:
            plot_mmgbsa_replica_summary(
                [r["replica"] for r in summary_rows],
                [r["mean_kcal_mol"] for r in summary_rows],
                [r["sd_kcal_mol"] for r in summary_rows],
                combined_dir / "mmgbsa_delta_total_summary",
                style,
            )
        outputs["summary_csv"] = str((combined_dir / "mmgbsa_delta_total_summary.csv").resolve())
    if time_x is not None and time_map:
        if plot_per_frame:
            plot_mmgbsa_timeseries(time_x, time_map, combined_dir / "mmgbsa_delta_total_per_frame", style, title="MM/GBSA ΔG per frame by replica", xlabel=xlabel)
            plot_mmgbsa_delta_total_distribution(dist_map, combined_dir / "mmgbsa_delta_total_distribution", style)
    if per_residue_map:
        residue_scores: dict[str, list[float]] = {}
        for _, mapping in per_residue_map.items():
            for label, value in mapping.items():
                residue_scores.setdefault(label, []).append(value)
        ranked = sorted(residue_scores, key=lambda k: abs(np.mean(residue_scores[k])), reverse=True)[: cfg.top_n_residues_plot]
        matrix = []
        replica_names = list(per_residue_map.keys())
        for label in ranked:
            matrix.append([per_residue_map.get(rep, {}).get(label, 0.0) for rep in replica_names])
        if plot_per_residue:
            plot_mmgbsa_delta_total_heatmap(replica_names, ranked, matrix, combined_dir / "mmgbsa_per_residue_heatmap", style)
            mean_values = [float(np.mean(residue_scores[label])) for label in ranked]
            plot_per_residue_decomp(ranked, mean_values, combined_dir / "mmgbsa_per_residue_top", style)
    return outputs


def _single_root_existing_files(
    cfg: MMGBSAConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    analysis_root = ensure_dir(cfg.analysis_root)
    plot_summary = plot_selection is None or plot_selection.enabled("mmgbsa_summary")
    plot_per_frame = plot_selection is None or plot_selection.enabled("mmgbsa_per_frame")
    plot_per_residue = plot_selection is None or plot_selection.enabled("mmgbsa_per_residue")
    outputs = {}
    summary_csv = None
    if str(cfg.final_dat).strip() and Path(cfg.final_dat).exists():
        summary_csv = parse_mmpbsa_results(cfg.final_dat, analysis_root / "mmpbsa_summary_parsed.csv")
    elif str(cfg.final_csv).strip() and Path(cfg.final_csv).exists():
        rows = read_csv_rows(cfg.final_csv)
        if rows and {"term", "mean_kcal_mol"}.issubset(rows[0].keys()):
            summary_csv = Path(cfg.final_csv)
    if summary_csv and Path(summary_csv).exists():
        rows = read_csv_rows(summary_csv)
        if rows:
            write_dict_csv(analysis_root / "mmpbsa_summary.csv", rows, list(rows[0].keys()))
            if plot_summary:
                plot_mmgbsa_summary(rows, analysis_root / "mmgbsa_summary", style)
            outputs["summary_csv"] = str(Path(summary_csv).resolve())

    if str(cfg.per_frame_csv).strip() and Path(cfg.per_frame_csv).exists():
        rows = read_csv_rows(cfg.per_frame_csv)
        if rows:
            numeric_cols = infer_numeric_columns(rows)
            time_col = _pick_time_or_frame(rows, numeric_cols)
            if time_col:
                x = [float(r[time_col]) for r in rows]
                value_cols = [c for c in numeric_cols if c != time_col][:6]
                value_map = {col: [float(r[col]) for r in rows] for col in value_cols}
                if value_map:
                    if plot_per_frame:
                        plot_mmgbsa_timeseries(x, value_map, analysis_root / "mmgbsa_per_frame", style, title="MM/GBSA per-frame energies", xlabel=time_col)
                    outputs["per_frame_csv"] = str(Path(cfg.per_frame_csv).resolve())

    if str(cfg.per_residue_csv).strip() and Path(cfg.per_residue_csv).exists():
        rows = read_csv_rows(cfg.per_residue_csv)
        if rows:
            keys = list(rows[0].keys())
            text_col = keys[0]
            numeric_cols = infer_numeric_columns(rows)
            target_col = _pick_energy_column(rows, numeric_cols)
            if target_col:
                trimmed = sorted(rows, key=lambda r: abs(float(r[target_col])), reverse=True)[: cfg.top_n_residues_plot]
                labels = [r[text_col] for r in trimmed]
                values = [float(r[target_col]) for r in trimmed]
                if plot_per_residue:
                    plot_per_residue_decomp(labels, values, analysis_root / "mmgbsa_per_residue", style)
                outputs["per_residue_csv"] = str(Path(cfg.per_residue_csv).resolve())
    outputs["analysis_root"] = str(Path(analysis_root).resolve())
    return outputs


def run_mmgbsa_postprocess(
    cfg: MMGBSAConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    analysis_root = ensure_dir(cfg.analysis_root)
    results = []
    if cfg.auto_run:
        source_root = Path(cfg.source_root)
        replica_dirs = sorted([p for p in source_root.glob("replica_*") if p.is_dir()])
        if replica_dirs:
            for replica_dir in replica_dirs:
                replica_out = ensure_dir(analysis_root / replica_dir.name)
                run_result = _auto_run_replica(replica_dir, replica_out, cfg)
                item = {"replica": replica_dir.name, "run": run_result}
                if run_result.get("status") in {"ok", "failed"}:
                    item["summary_rows"] = _parse_summary_table(replica_out, cfg)
                    item["per_frame"] = _parse_per_frame(replica_out, cfg)
                    item["per_residue"] = _parse_per_residue(replica_out, cfg)
                results.append(item)
            write_json(analysis_root / "replica_mmgbsa_status.json", results)
            combined = _combined_outputs(results, analysis_root, cfg, style, plot_selection)
            return {
                "analysis_root": str(Path(analysis_root).resolve()),
                "replica_results": results,
                "combined": combined,
                "status": "ok" if any(r.get("summary_rows") for r in results) else "skipped_or_failed",
            }

    return _single_root_existing_files(cfg, style, plot_selection)
