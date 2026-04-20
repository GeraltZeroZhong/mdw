from __future__ import annotations

from copy import deepcopy

from pathlib import Path

from ..config import WorkflowConfig
from ..core import ensure_project_layout, infer_run_input_paths, normalize_workflow_paths, organize_outputs, preflight_validate
from ..core.progress import ProgressCallback, ProgressEvent, emit_progress
from .plot import run_plot_postprocess


def _resolve_md_inputs(cfg: WorkflowConfig) -> None:
    protein_pdb, ligand_sdf = infer_run_input_paths(cfg)
    cfg.run.protein_pdb = protein_pdb
    cfg.run.ligand_sdf = ligand_sdf


def _bind_analysis_inputs_to_md_outputs(cfg: WorkflowConfig) -> None:
    if not cfg.do_run_md:
        return
    replica_root = str(Path(cfg.run.output_root).resolve())
    ligand_sdf = str(Path(cfg.run.ligand_sdf).resolve())
    cfg.basic.replica_root = replica_root
    cfg.basic.ligand_sdf = ligand_sdf
    cfg.basic.timestep_ps = cfg.run.timestep_ps
    cfg.basic.dcd_interval_steps = cfg.run.dcd_interval
    cfg.waterbridge.replica_root = replica_root
    cfg.waterbridge.timestep_ps = cfg.run.timestep_ps
    cfg.waterbridge.dcd_interval_steps = cfg.run.dcd_interval
    cfg.advanced.replica_root = replica_root
    cfg.mmgbsa.source_root = replica_root


def _summarize_outputs(cfg: WorkflowConfig, outputs: dict) -> None:
    outputs["replica_root"] = str(Path(cfg.run.output_root).resolve())
    outputs["n_replicas"] = cfg.run.n_replicas
    if cfg.do_basic_analysis:
        outputs["basic_combined_dir"] = str((Path(cfg.basic.analysis_root) / "combined").resolve())
    if cfg.do_waterbridge_analysis:
        outputs["waterbridge_combined_dir"] = str((Path(cfg.waterbridge.analysis_root) / "combined").resolve())
    if cfg.do_advanced_analysis:
        outputs["advanced_dir"] = str(Path(cfg.advanced.analysis_root).resolve())
    if cfg.do_mmgbsa_postprocess:
        outputs["mmgbsa_dir"] = str(Path(cfg.mmgbsa.analysis_root).resolve())


def _progress_total_units(cfg: WorkflowConfig) -> int:
    total = 0
    if cfg.do_prep:
        total += 1
    if cfg.do_run_md:
        total += max(int(cfg.run.n_replicas), 1)
    if cfg.do_basic_analysis:
        total += 1
    if cfg.do_waterbridge_analysis:
        total += 1
    if cfg.do_advanced_analysis:
        total += 1
    if cfg.do_mmgbsa_postprocess:
        total += 1
    total += 1
    return max(total, 1)


def run_full_md_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = normalize_workflow_paths(deepcopy(cfg))
    total_units = _progress_total_units(cfg)
    completed_units = 0
    emit_progress(progress_callback, completed_units, total_units, "initialize", "Preparing workflow inputs")
    ensure_project_layout(cfg.workspace_root)
    validation = preflight_validate(cfg)
    if validation.errors:
        bullets = "\n".join(f"- {item}" for item in validation.errors)
        raise ValueError(f"Workflow preflight validation failed:\n{bullets}")
    _resolve_md_inputs(cfg)
    _bind_analysis_inputs_to_md_outputs(cfg)
    outputs = {}
    if cfg.do_prep:
        emit_progress(progress_callback, completed_units, total_units, "prep", "Preparing receptor and ligand inputs")
        from ..prep import run_prep_workflow
        outputs["prep"] = run_prep_workflow(cfg.prep, cfg.docking)
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "prep", "Prepared receptor and ligand inputs")
    if cfg.do_run_md:
        md_units = max(int(cfg.run.n_replicas), 1)
        emit_progress(progress_callback, completed_units, total_units, "md", f"Running {cfg.run.n_replicas} MD replicas")
        from ..simulate import run_md_workflow
        md_offset = completed_units

        def _md_progress(event: ProgressEvent) -> None:
            total_replica_units = max(int(event.total), md_units)
            bounded_completed = min(max(int(event.current), 0), total_replica_units)
            emit_progress(
                progress_callback,
                md_offset + bounded_completed,
                total_units,
                "md",
                event.detail,
                subcurrent=event.subcurrent,
                subtotal=event.subtotal,
                subdetail=event.subdetail or event.detail,
                subeta_seconds=event.subeta_seconds,
            )

        outputs["md"] = run_md_workflow(cfg.run, progress_callback=_md_progress)
        completed_units += md_units
        emit_progress(progress_callback, completed_units, total_units, "md", f"Completed {cfg.run.n_replicas} MD replicas")
    if cfg.do_basic_analysis:
        emit_progress(progress_callback, completed_units, total_units, "basic_analysis", "Running basic analysis")
        from ..analysis import run_basic_analysis

        def _basic_progress(event: ProgressEvent) -> None:
            emit_progress(
                progress_callback,
                completed_units,
                total_units,
                "basic_analysis",
                event.detail,
                subcurrent=event.subcurrent if event.subtotal > 0 else event.current,
                subtotal=event.subtotal if event.subtotal > 0 else event.total,
                subdetail=event.subdetail or event.detail,
            )

        outputs["basic_analysis"] = run_basic_analysis(
            cfg.basic,
            cfg.plot_style,
            cfg.plot_selection,
            progress_callback=_basic_progress,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "basic_analysis", "Completed basic analysis")
    if cfg.do_waterbridge_analysis:
        emit_progress(progress_callback, completed_units, total_units, "waterbridge_analysis", "Running water-bridge analysis")
        from ..analysis import run_waterbridge_analysis

        def _waterbridge_progress(event: ProgressEvent) -> None:
            emit_progress(
                progress_callback,
                completed_units,
                total_units,
                "waterbridge_analysis",
                event.detail,
                subcurrent=event.subcurrent if event.subtotal > 0 else event.current,
                subtotal=event.subtotal if event.subtotal > 0 else event.total,
                subdetail=event.subdetail or event.detail,
            )

        outputs["waterbridge_analysis"] = run_waterbridge_analysis(
            cfg.waterbridge,
            cfg.plot_style,
            cfg.plot_selection,
            progress_callback=_waterbridge_progress,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "waterbridge_analysis", "Completed water-bridge analysis")
    if cfg.do_basic_analysis or cfg.do_waterbridge_analysis:
        plot_units = int(cfg.do_basic_analysis) + int(cfg.do_waterbridge_analysis)
        plot_offset = completed_units

        def _plot_progress(event: ProgressEvent) -> None:
            total_plot_units = max(int(event.total), plot_units)
            bounded_completed = min(max(int(event.current), 0), total_plot_units)
            emit_progress(
                progress_callback,
                plot_offset + bounded_completed,
                total_units,
                event.stage,
                event.detail,
            )

        plot_outputs = run_plot_postprocess(
            cfg,
            progress_callback=_plot_progress,
            include_mmgbsa_postprocess=False,
            include_organize_outputs=False,
        )
        outputs.update(plot_outputs)
        completed_units += plot_units
    if cfg.do_advanced_analysis:
        emit_progress(progress_callback, completed_units, total_units, "advanced_analysis", "Running advanced analysis")
        from ..analysis import run_advanced_analysis

        def _advanced_progress(event: ProgressEvent) -> None:
            emit_progress(
                progress_callback,
                completed_units,
                total_units,
                "advanced_analysis",
                event.detail,
                subcurrent=event.current,
                subtotal=event.total,
                subdetail=event.detail,
            )

        outputs["advanced_analysis"] = run_advanced_analysis(
            cfg.advanced,
            cfg.plot_style,
            cfg.plot_selection,
            progress_callback=_advanced_progress,
        )
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "advanced_analysis", "Completed advanced analysis")
    if cfg.do_mmgbsa_postprocess:
        emit_progress(progress_callback, completed_units, total_units, "mmgbsa_postprocess", "Running MM/GBSA postprocess")
        from ..postprocess.mmgbsa import run_mmgbsa_postprocess
        try:
            outputs["mmgbsa_postprocess"] = run_mmgbsa_postprocess(cfg.mmgbsa, cfg.plot_style, cfg.plot_selection)
            mmgbsa_detail = "Completed MM/GBSA postprocess"
        except Exception as exc:
            if cfg.mmgbsa.non_blocking:
                outputs["mmgbsa_postprocess"] = {"status": "failed_non_blocking", "error": str(exc)}
                mmgbsa_detail = "MM/GBSA postprocess failed but was kept non-blocking"
            else:
                raise
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "mmgbsa_postprocess", mmgbsa_detail)
    _summarize_outputs(cfg, outputs)
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
