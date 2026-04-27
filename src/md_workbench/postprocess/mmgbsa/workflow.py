from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import numpy as np

from ...config import MMGBSAConfig, PlotSelectionConfig, PlotStyleConfig
from ...config.plot_style_defaults import apply_plot_style_palette
from ...core import ensure_dir, find_binary, write_dict_csv, write_json
from ...plotting.mmgbsa import (
    plot_mmgbsa_delta_total_distribution,
    plot_mmgbsa_delta_total_heatmap,
    plot_mmgbsa_replica_summary,
    plot_mmgbsa_summary,
    plot_mmgbsa_timeseries,
    plot_per_residue_decomp,
)
from .parser import (
    infer_numeric_columns,
    parse_mmpbsa_decomp_delta_frame_rows,
    parse_mmpbsa_decomp_summary_rows,
    parse_mmpbsa_final_results_delta_rows,
    parse_mmpbsa_results,
    read_csv_rows,
    summarize_mmpbsa_decomp_frame_rows,
)


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


def _mpi_enabled(cfg: MMGBSAConfig) -> bool:
    return bool(getattr(cfg, "use_mpi", False)) and max(int(getattr(cfg, "mpi_ranks", 1)), 1) > 1


def _resolve_mmgbsa_command(cfg: MMGBSAConfig) -> tuple[list[str] | None, list[str]]:
    if _mpi_enabled(cfg):
        launcher = _binary("mpirun") or _binary("mpiexec")
        binary = _binary("MMPBSA.py.MPI")
        missing = []
        if launcher is None:
            missing.append("mpirun")
        if binary is None:
            missing.append("MMPBSA.py.MPI")
        if importlib.util.find_spec("mpi4py") is None:
            missing.append("mpi4py")
        if missing:
            return None, missing
        ranks = max(int(getattr(cfg, "mpi_ranks", 1)), 1)
        return [launcher, "-np", str(ranks), binary], []

    binary = _binary("MMPBSA.py") or _binary("MMPBSA.py.MPI")
    if not binary:
        return None, ["MMPBSA.py"]
    return [binary], []


def _write_default_input(path: Path, cfg: MMGBSAConfig) -> Path:
    text = (
        "&general\n"
        f"  startframe={int(cfg.startframe)}, interval={int(cfg.interval)}, verbose=1, keep_files=0,\n"
        "/\n"
        "&gb\n"
        f"  igb={int(cfg.igb)}, saltcon={float(cfg.saltcon):.3f},\n"
        "/\n"
    )
    if int(cfg.idecomp) > 0:
        text += (
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
        "complex_solvated_prmtop": replica_dir / Path(cfg.complex_solvated_prmtop).name,
        "complex_prmtop": replica_dir / Path(cfg.complex_prmtop).name,
        "receptor_prmtop": replica_dir / Path(cfg.receptor_prmtop).name,
        "ligand_prmtop": replica_dir / Path(cfg.ligand_prmtop).name,
        "trajectory_nc": replica_dir / Path(cfg.trajectory_nc).name,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        return {"status": "skipped_missing_inputs", "missing": missing}

    command_prefix, missing = _resolve_mmgbsa_command(cfg)
    if command_prefix is None:
        return {"status": "skipped_missing_binary", "missing": missing}

    out_dir.mkdir(parents=True, exist_ok=True)
    input_file = out_dir / Path(cfg.mmpbsa_input_file).name
    _write_default_input(input_file, cfg)

    final_dat = out_dir / Path(cfg.final_dat).name
    final_csv = out_dir / Path(cfg.final_csv).name
    decomp_enabled = int(cfg.idecomp) > 0
    decomp_dat = out_dir / Path(cfg.per_residue_dat).name
    decomp_csv = out_dir / Path(cfg.per_residue_csv).name

    args = [
        *command_prefix,
        "-O",
        "-i", str(input_file),
        "-sp", str(required["complex_solvated_prmtop"]),
        "-cp", str(required["complex_prmtop"]),
        "-rp", str(required["receptor_prmtop"]),
        "-lp", str(required["ligand_prmtop"]),
        "-y", str(required["trajectory_nc"]),
        "-o", str(final_dat),
        "-eo", str(final_csv),
    ]
    if decomp_enabled:
        args.extend([
            "-do", str(decomp_dat),
            "-deo", str(decomp_csv),
        ])
    code, stdout, stderr = _run_command(args, cwd=str(out_dir))
    output_paths = {
        "final_dat": str(final_dat),
        "final_csv": str(final_csv),
    }
    if decomp_enabled:
        output_paths["per_residue_dat"] = str(decomp_dat)
        output_paths["per_residue_csv"] = str(decomp_csv)
    result = {
        "status": "ok" if code == 0 else "failed",
        "used_mpi": _mpi_enabled(cfg),
        "command": args,
        "stdout": stdout,
        "stderr": stderr,
        "returncode": code,
        "outputs": output_paths,
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
    rows = parse_mmpbsa_final_results_delta_rows(path)
    if not rows:
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
    x = []
    y = []
    for row in rows:
        try:
            x_value = float(row[time_col])
            y_value = float(row[energy_col])
        except Exception:
            continue
        if not (np.isfinite(x_value) and np.isfinite(y_value)):
            continue
        x.append(x_value)
        y.append(y_value)
    if not x or not y:
        return None
    return {"x": x, "y": y, "xlabel": time_col, "energy_col": energy_col}


def _parse_per_residue(replica_out_dir: Path, cfg: MMGBSAConfig):
    dat_path = replica_out_dir / Path(cfg.per_residue_dat).name
    if dat_path.exists():
        rows = parse_mmpbsa_decomp_summary_rows(dat_path)
        parsed = []
        for row in rows:
            try:
                value = float(row.get("total_avg", float("nan")))
            except Exception:
                continue
            if not np.isfinite(value):
                continue
            parsed.append({"label": str(row.get("residue", "")).strip(), "value": value})
        if parsed:
            return parsed

    csv_path = replica_out_dir / Path(cfg.per_residue_csv).name
    if not csv_path.exists():
        return None
    summary_rows = summarize_mmpbsa_decomp_frame_rows(parse_mmpbsa_decomp_delta_frame_rows(csv_path))
    if not summary_rows:
        rows = read_csv_rows(csv_path)
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
            if not np.isfinite(value):
                continue
            parsed.append({"label": label, "value": value})
        return parsed or None
    return [{"label": str(row["label"]), "value": float(row["mean_kcal_mol"])} for row in summary_rows]


def _parse_replica_outputs(replica_out_dir: Path, cfg: MMGBSAConfig) -> dict:
    parsed = {}
    summary_rows = _parse_summary_table(replica_out_dir, cfg)
    if summary_rows:
        parsed["summary_rows"] = summary_rows
    per_frame = _parse_per_frame(replica_out_dir, cfg)
    if per_frame:
        parsed["per_frame"] = per_frame
    per_residue = _parse_per_residue(replica_out_dir, cfg)
    if per_residue:
        parsed["per_residue"] = per_residue
    return parsed


def _load_existing_replica_results(analysis_root: Path, cfg: MMGBSAConfig) -> list[dict]:
    reused = []
    for replica_out in sorted(path for path in analysis_root.glob("replica_*") if path.is_dir()):
        parsed = _parse_replica_outputs(replica_out, cfg)
        if parsed:
            reused.append(
                {
                    "replica": replica_out.name,
                    "run": {"status": "reused_existing_outputs"},
                    **parsed,
                }
            )
    return reused


def _clear_generated_replica_outputs(replica_out_dir: Path, cfg: MMGBSAConfig) -> None:
    generated_names = {
        Path(cfg.final_dat).name,
        Path(cfg.final_csv).name,
        Path(cfg.per_residue_dat).name,
        Path(cfg.per_residue_csv).name,
        "mmpbsa_summary_parsed.csv",
    }
    for name in generated_names:
        path = replica_out_dir / name
        if path.exists() and path.is_file():
            path.unlink()


def summarize_mmgbsa_postprocess_result(result: dict | None) -> str:
    if not result:
        return "MM/GBSA postprocess completed without a result payload"
    detail = str(result.get("detail", "")).strip()
    if detail:
        return detail
    status = str(result.get("status", "")).strip()
    if status == "ok":
        return "Completed MM/GBSA postprocess"
    if status == "skipped_missing_inputs":
        return "Skipped MM/GBSA because Amber inputs were not found"
    if status == "skipped_missing_binary":
        return "Skipped MM/GBSA because MMPBSA.py was not available"
    if status == "failed":
        return "MM/GBSA postprocess failed"
    if status == "skipped_or_failed":
        return "MM/GBSA postprocess produced no usable outputs"
    return "Completed MM/GBSA postprocess"


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
                mean_value = float(delta["mean_kcal_mol"])
                sd_value = float(delta.get("sd_kcal_mol", 0.0))
                if np.isfinite(mean_value):
                    summary_rows.append({
                        "replica": name,
                        "mean_kcal_mol": mean_value,
                        "sd_kcal_mol": sd_value if np.isfinite(sd_value) else 0.0,
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
    summary_rows = None
    if str(cfg.final_dat).strip() and Path(cfg.final_dat).exists():
        summary_csv = parse_mmpbsa_results(cfg.final_dat, analysis_root / "mmpbsa_summary_parsed.csv")
    if summary_csv and Path(summary_csv).exists():
        summary_rows = read_csv_rows(summary_csv)
        if summary_rows:
            write_dict_csv(analysis_root / "mmpbsa_summary.csv", summary_rows, list(summary_rows[0].keys()))
            if plot_summary:
                plot_mmgbsa_summary(summary_rows, analysis_root / "mmgbsa_summary", style)
            outputs["summary_csv"] = str((analysis_root / "mmpbsa_summary.csv").resolve())

    per_frame_path = None
    if str(cfg.final_csv).strip() and Path(cfg.final_csv).exists():
        per_frame_path = Path(cfg.final_csv)
    elif str(cfg.per_frame_csv).strip() and Path(cfg.per_frame_csv).exists():
        per_frame_path = Path(cfg.per_frame_csv)
    if per_frame_path is not None and per_frame_path.exists():
        rows = parse_mmpbsa_final_results_delta_rows(per_frame_path)
        if not rows and str(cfg.per_frame_csv).strip() and Path(cfg.per_frame_csv).exists():
            rows = read_csv_rows(cfg.per_frame_csv)
        if rows:
            numeric_cols = infer_numeric_columns(rows)
            time_col = _pick_time_or_frame(rows, numeric_cols)
            if time_col:
                x = []
                value_cols = [c for c in numeric_cols if c != time_col][:6]
                value_map = {col: [] for col in value_cols}
                for row in rows:
                    try:
                        time_value = float(row[time_col])
                    except Exception:
                        continue
                    candidate_values: dict[str, float] = {}
                    valid = np.isfinite(time_value)
                    for col in value_cols:
                        try:
                            candidate_values[col] = float(row[col])
                        except Exception:
                            valid = False
                            break
                        if not np.isfinite(candidate_values[col]):
                            valid = False
                            break
                    if not valid:
                        continue
                    x.append(time_value)
                    for col, value in candidate_values.items():
                        value_map[col].append(value)
                value_map = {col: values for col, values in value_map.items() if values}
                if x and value_map:
                    parsed_per_frame_csv = analysis_root / "mmpbsa_per_frame_delta.csv"
                    write_dict_csv(parsed_per_frame_csv, rows, list(rows[0].keys()))
                    if plot_per_frame:
                        plot_mmgbsa_timeseries(x, value_map, analysis_root / "mmgbsa_per_frame", style, title="MM/GBSA per-frame energies", xlabel=time_col)
                    outputs["per_frame_csv"] = str(parsed_per_frame_csv.resolve())

    residue_rows = []
    if str(cfg.per_residue_dat).strip() and Path(cfg.per_residue_dat).exists():
        residue_rows = [
            {
                "label": str(row["residue"]),
                "location": str(row["location"]),
                "mean_kcal_mol": float(row["total_avg"]),
                "sd_kcal_mol": float(row["total_sd"]),
                "sem_kcal_mol": float(row["total_sem"]),
            }
            for row in parse_mmpbsa_decomp_summary_rows(cfg.per_residue_dat)
            if np.isfinite(float(row["total_avg"]))
        ]
    elif str(cfg.per_residue_csv).strip() and Path(cfg.per_residue_csv).exists():
        residue_rows = summarize_mmpbsa_decomp_frame_rows(parse_mmpbsa_decomp_delta_frame_rows(cfg.per_residue_csv))
    if residue_rows:
        parsed_per_residue_csv = analysis_root / "mmpbsa_per_residue_summary.csv"
        write_dict_csv(parsed_per_residue_csv, residue_rows, list(residue_rows[0].keys()))
        trimmed = sorted(residue_rows, key=lambda r: abs(float(r["mean_kcal_mol"])), reverse=True)[: cfg.top_n_residues_plot]
        labels = [str(r["label"]) for r in trimmed]
        values = [float(r["mean_kcal_mol"]) for r in trimmed]
        if plot_per_residue:
            plot_per_residue_decomp(labels, values, analysis_root / "mmgbsa_per_residue", style)
        outputs["per_residue_csv"] = str(parsed_per_residue_csv.resolve())
    outputs["analysis_root"] = str(Path(analysis_root).resolve())
    return outputs


def summarize_mmgbsa_postprocess_result(result: dict | None) -> str:
    if not isinstance(result, dict):
        return "Completed MM/GBSA postprocess"

    status = str(result.get("status", "")).strip()
    if status == "failed_non_blocking":
        return "MM/GBSA postprocess failed but was kept non-blocking"

    replica_results = result.get("replica_results")
    if isinstance(replica_results, list):
        total = len(replica_results)
        summarized = sum(1 for item in replica_results if item.get("summary_rows"))
        failed = sum(1 for item in replica_results if isinstance(item.get("run"), dict) and item["run"].get("status") == "failed")
        skipped = max(total - summarized - failed, 0)
        if summarized:
            detail = f"Completed MM/GBSA postprocess for {summarized}/{total} replicas"
            if failed:
                detail += f"; {failed} failed"
            elif skipped:
                detail += f"; {skipped} skipped"
            return detail
        if total:
            if failed == total:
                return "MM/GBSA postprocess failed for all replicas"
            return "MM/GBSA postprocess finished without parsed replica summaries"

    produced = []
    if result.get("summary_csv"):
        produced.append("summary")
    if result.get("per_frame_csv"):
        produced.append("per-frame")
    if result.get("per_residue_csv"):
        produced.append("per-residue")
    if produced:
        return f"Completed MM/GBSA postprocess from existing files ({', '.join(produced)})"

    if status == "skipped_or_failed":
        return "MM/GBSA postprocess finished without parsed outputs"
    return "Completed MM/GBSA postprocess"


def run_mmgbsa_postprocess(
    cfg: MMGBSAConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    apply_plot_style_palette(style)
    analysis_root = ensure_dir(cfg.analysis_root)
    results = []
    reused = _load_existing_replica_results(analysis_root, cfg) if cfg.reuse_existing_outputs else []
    if reused:
        write_json(analysis_root / "replica_mmgbsa_status.json", reused)
        combined = _combined_outputs(reused, analysis_root, cfg, style, plot_selection)
        return {
            "analysis_root": str(Path(analysis_root).resolve()),
            "replica_results": reused,
            "combined": combined,
            "status_counts": {"reused_existing_outputs": len(reused)},
            "status": "ok",
            "detail": f"Reused existing MM/GBSA outputs for {len(reused)} replicas",
        }
    if cfg.auto_run:
        source_root = Path(cfg.source_root)
        replica_dirs = sorted([p for p in source_root.glob("replica_*") if p.is_dir()])
        if replica_dirs:
            for replica_dir in replica_dirs:
                replica_out = ensure_dir(analysis_root / replica_dir.name)
                if not cfg.reuse_existing_outputs:
                    _clear_generated_replica_outputs(replica_out, cfg)
                run_result = _auto_run_replica(replica_dir, replica_out, cfg)
                item = {"replica": replica_dir.name, "run": run_result}
                item.update(_parse_replica_outputs(replica_out, cfg))
                results.append(item)
            write_json(analysis_root / "replica_mmgbsa_status.json", results)
            combined = _combined_outputs(results, analysis_root, cfg, style, plot_selection)
            replica_count = len(results)
            parsed_count = sum(1 for r in results if r.get("summary_rows"))
            status_counts: dict[str, int] = {}
            for item in results:
                run_status = str(item.get("run", {}).get("status", "unknown"))
                status_counts[run_status] = status_counts.get(run_status, 0) + 1
            if parsed_count > 0:
                status = "ok"
                detail = f"Completed MM/GBSA for {parsed_count}/{replica_count} replicas"
            else:
                reused = _load_existing_replica_results(analysis_root, cfg) if cfg.reuse_existing_outputs else []
                if reused:
                    write_json(analysis_root / "replica_mmgbsa_status.json", reused)
                    combined = _combined_outputs(reused, analysis_root, cfg, style, plot_selection)
                    return {
                        "analysis_root": str(Path(analysis_root).resolve()),
                        "replica_results": reused,
                        "combined": combined,
                        "status_counts": {"reused_existing_outputs": len(reused)},
                        "status": "ok",
                        "detail": f"Reused existing MM/GBSA outputs for {len(reused)} replicas after auto-run produced no parsed results",
                    }
                if status_counts.get("skipped_missing_inputs", 0) == replica_count:
                    status = "skipped_missing_inputs"
                    detail = f"Skipped MM/GBSA: Amber inputs were missing in all {replica_count} replicas"
                elif status_counts.get("skipped_missing_binary", 0) == replica_count:
                    status = "skipped_missing_binary"
                    detail = "Skipped MM/GBSA: MMPBSA.py was not available"
                elif status_counts.get("failed", 0) > 0:
                    status = "failed"
                    detail = f"MM/GBSA failed in {status_counts['failed']}/{replica_count} replicas and produced no summary tables"
                else:
                    status = "skipped_or_failed"
                    detail = "MM/GBSA produced no usable outputs"
            return {
                "analysis_root": str(Path(analysis_root).resolve()),
                "replica_results": results,
                "combined": combined,
                "status_counts": status_counts,
                "status": status,
                "detail": detail,
            }
        reused = _load_existing_replica_results(analysis_root, cfg) if cfg.reuse_existing_outputs else []
        if reused:
            write_json(analysis_root / "replica_mmgbsa_status.json", reused)
            combined = _combined_outputs(reused, analysis_root, cfg, style, plot_selection)
            return {
                "analysis_root": str(Path(analysis_root).resolve()),
                "replica_results": reused,
                "combined": combined,
                "status_counts": {"reused_existing_outputs": len(reused)},
                "status": "ok",
                "detail": f"Reused existing MM/GBSA outputs for {len(reused)} replicas",
            }

    reused = _load_existing_replica_results(analysis_root, cfg) if cfg.reuse_existing_outputs else []
    if reused:
        write_json(analysis_root / "replica_mmgbsa_status.json", reused)
        combined = _combined_outputs(reused, analysis_root, cfg, style, plot_selection)
        return {
            "analysis_root": str(Path(analysis_root).resolve()),
            "replica_results": reused,
            "combined": combined,
            "status_counts": {"reused_existing_outputs": len(reused)},
            "status": "ok",
            "detail": f"Reused existing MM/GBSA outputs for {len(reused)} replicas",
        }

    outputs = _single_root_existing_files(cfg, style, plot_selection)
    usable_keys = {k for k in outputs.keys() if k != "analysis_root"}
    outputs["status"] = "ok" if usable_keys else "skipped_or_failed"
    outputs["detail"] = "Completed MM/GBSA postprocess" if usable_keys else "MM/GBSA postprocess produced no usable outputs"
    return outputs
