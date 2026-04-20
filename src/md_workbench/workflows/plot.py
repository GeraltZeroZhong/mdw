from __future__ import annotations

from copy import deepcopy

from pathlib import Path
import numpy as np

from ..config import WorkflowConfig
from ..core import ensure_project_layout, normalize_workflow_paths, organize_outputs, read_dict_csv
from ..core.progress import ProgressCallback, emit_progress
from ..plotting.series import mean_sd_series
from ..postprocess.mmgbsa import run_mmgbsa_postprocess


def _plot_progress_total_units(
    cfg: WorkflowConfig,
    *,
    include_mmgbsa_postprocess: bool,
    include_organize_outputs: bool,
) -> int:
    total = 0
    if cfg.do_basic_analysis:
        total += 1
    if cfg.do_waterbridge_analysis:
        total += 1
    if include_mmgbsa_postprocess and cfg.do_mmgbsa_postprocess:
        total += 1
    if include_organize_outputs:
        total += 1
    return max(total, 1)


def _run_basic_replot(cfg: WorkflowConfig) -> str | None:
    basic_root = Path(cfg.basic.analysis_root) / "combined"
    csv_path = basic_root / "rmsd_combined.csv"
    if not csv_path.exists():
        return None
    rows = read_dict_csv(csv_path)
    time_ns = np.asarray([float(r["time_ns"]) for r in rows], dtype=float)
    protein_mean = np.asarray([float(r["protein_backbone_rmsd_mean_A"]) for r in rows], dtype=float)
    protein_sd = np.asarray([float(r["protein_backbone_rmsd_sd_A"]) for r in rows], dtype=float)
    ligand_mean = np.asarray([float(r["ligand_heavy_rmsd_mean_A"]) for r in rows], dtype=float)
    ligand_sd = np.asarray([float(r["ligand_heavy_rmsd_sd_A"]) for r in rows], dtype=float)
    mean_sd_series(
        time_ns,
        np.vstack([protein_mean - protein_sd, protein_mean, protein_mean + protein_sd]),
        "Protein backbone RMSD (Å)",
        basic_root / "rmsd_replot_protein",
        cfg.plot_style,
        title="Protein backbone RMSD replot",
        individual_labels=["Mean-SD", "Mean", "Mean+SD"],
    )
    mean_sd_series(
        time_ns,
        np.vstack([ligand_mean - ligand_sd, ligand_mean, ligand_mean + ligand_sd]),
        "Ligand heavy-atom RMSD (Å)",
        basic_root / "rmsd_replot_ligand",
        cfg.plot_style,
        title="Ligand heavy-atom RMSD replot",
        individual_labels=["Mean-SD", "Mean", "Mean+SD"],
    )
    return str(basic_root.resolve())


def _run_waterbridge_replot(cfg: WorkflowConfig) -> str | None:
    water_root = Path(cfg.waterbridge.analysis_root) / "combined"
    csv_path = water_root / "waterbridge_count_combined.csv"
    if not csv_path.exists():
        return None
    rows = read_dict_csv(csv_path)
    time_ns = np.asarray([float(r["time_ns"]) for r in rows], dtype=float)
    mean_key = "mean_n_bridging_waters" if "mean_n_bridging_waters" in rows[0] else "mean_count"
    sd_key = "sd_n_bridging_waters" if "sd_n_bridging_waters" in rows[0] else "sd_count"
    mean_count = np.asarray([float(r[mean_key]) for r in rows], dtype=float)
    sd_count = np.asarray([float(r[sd_key]) for r in rows], dtype=float)
    mean_sd_series(
        time_ns,
        np.vstack([mean_count - sd_count, mean_count, mean_count + sd_count]),
        "Number of bridging waters",
        water_root / "waterbridge_count_replot",
        cfg.plot_style,
        title="Strict water-bridge count replot",
        individual_labels=["Mean-SD", "Mean", "Mean+SD"],
    )
    return str(water_root.resolve())


def run_plot_postprocess(
    cfg: WorkflowConfig,
    progress_callback: ProgressCallback | None = None,
    *,
    include_mmgbsa_postprocess: bool = True,
    include_organize_outputs: bool = True,
):
    total_units = _plot_progress_total_units(
        cfg,
        include_mmgbsa_postprocess=include_mmgbsa_postprocess,
        include_organize_outputs=include_organize_outputs,
    )
    completed_units = 0
    outputs = {}
    if cfg.do_basic_analysis:
        if cfg.plot_selection.enabled("plot_workflow_basic_replot"):
            emit_progress(progress_callback, completed_units, total_units, "basic_replot", "Replotting combined basic-analysis figures")
            basic_replot_dir = _run_basic_replot(cfg)
            if basic_replot_dir is not None:
                outputs["basic_replot_dir"] = basic_replot_dir
                detail = "Completed combined basic-analysis replot"
            else:
                detail = "Skipping combined basic-analysis replot because rmsd_combined.csv was not found"
        else:
            detail = "Skipping combined basic-analysis replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "basic_replot", detail)
    if cfg.do_waterbridge_analysis:
        if cfg.plot_selection.enabled("plot_workflow_waterbridge_replot"):
            emit_progress(progress_callback, completed_units, total_units, "waterbridge_replot", "Replotting combined water-bridge figures")
            waterbridge_replot_dir = _run_waterbridge_replot(cfg)
            if waterbridge_replot_dir is not None:
                outputs["waterbridge_replot_dir"] = waterbridge_replot_dir
                detail = "Completed combined water-bridge replot"
            else:
                detail = "Skipping combined water-bridge replot because waterbridge_count_combined.csv was not found"
        else:
            detail = "Skipping combined water-bridge replot because it is disabled in plot selection"
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "waterbridge_replot", detail)
    if include_mmgbsa_postprocess and cfg.do_mmgbsa_postprocess:
        emit_progress(progress_callback, completed_units, total_units, "mmgbsa_postprocess", "Running MM/GBSA postprocess")
        outputs["mmgbsa_postprocess"] = run_mmgbsa_postprocess(cfg.mmgbsa, cfg.plot_style, cfg.plot_selection)
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "mmgbsa_postprocess", "Completed MM/GBSA postprocess")

    if include_organize_outputs:
        emit_progress(progress_callback, completed_units, total_units, "organize_outputs", "Organizing workflow outputs")
        outputs["organized_outputs"] = organize_outputs(
            cfg.output_bundle,
            cfg.run,
            cfg.basic,
            cfg.waterbridge,
            cfg.advanced,
            cfg.mmgbsa,
            cfg.do_basic_analysis,
            cfg.do_waterbridge_analysis,
            cfg.do_advanced_analysis,
            cfg.do_mmgbsa_postprocess,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "organize_outputs", "Workflow outputs organized")
    return outputs


def run_plot_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = normalize_workflow_paths(deepcopy(cfg))
    ensure_project_layout(cfg.workspace_root)
    return run_plot_postprocess(cfg, progress_callback=progress_callback)
