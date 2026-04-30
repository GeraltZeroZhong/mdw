from __future__ import annotations

from copy import deepcopy

from pathlib import Path

from ..config import WorkflowConfig
from ..config.plot_style_defaults import apply_plot_style_palette
from ..core import (
    ensure_project_layout,
    infer_run_input_paths,
    normalize_workflow_paths,
    organize_outputs,
    preflight_validate,
    summarize_replica_status,
)
from ..core.progress import ProgressCallback, ProgressEvent, emit_progress
from .plot import (
    reusable_advanced_csv_available,
    reusable_basic_csv_available,
    reusable_waterbridge_csv_available,
    run_advanced_replot_from_csv,
    run_basic_replot_from_csv,
    run_plot_postprocess,
    run_waterbridge_replot_from_csv,
)


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


def _bind_existing_analysis_inputs(cfg: WorkflowConfig) -> None:
    ligand_candidates = [
        cfg.basic.ligand_sdf,
        cfg.docking.extracted_pose_sdf,
        cfg.run.ligand_sdf,
        cfg.docking.ligand_output_sdf,
        cfg.docking.ligand_sdf_input,
    ]
    for candidate in ligand_candidates:
        candidate_path = Path(str(candidate).strip()) if str(candidate).strip() else None
        if candidate_path is not None and candidate_path.exists():
            cfg.basic.ligand_sdf = str(candidate_path.resolve())
            break


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


def prepare_next_replica_workflow_config(cfg: WorkflowConfig) -> WorkflowConfig:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    cfg.do_run_md = True
    if cfg.do_prep:
        protein_pdb, ligand_sdf = infer_run_input_paths(cfg)
        prepared_inputs_exist = (
            str(protein_pdb).strip()
            and str(ligand_sdf).strip()
            and Path(protein_pdb).is_file()
            and Path(ligand_sdf).is_file()
        )
        if prepared_inputs_exist:
            cfg.do_prep = False
            cfg.run.protein_pdb = protein_pdb
            cfg.run.ligand_sdf = ligand_sdf
    return cfg


def prepare_docking_only_workflow_config(cfg: WorkflowConfig) -> WorkflowConfig:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    cfg.do_prep = True
    cfg.do_run_md = False
    cfg.do_basic_analysis = False
    cfg.do_waterbridge_analysis = False
    cfg.do_advanced_analysis = False
    cfg.do_mmgbsa_postprocess = False
    cfg.output_bundle.enabled = False
    return cfg


def _next_replica_total_units(cfg: WorkflowConfig) -> int:
    return max(1 + int(bool(cfg.do_prep)), 1)


def run_next_replica_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = prepare_next_replica_workflow_config(cfg)
    total_units = _next_replica_total_units(cfg)
    completed_units = 0
    emit_progress(progress_callback, completed_units, total_units, "initialize", "Preparing next-replica workflow inputs")
    ensure_project_layout(cfg.workspace_root)
    validation = preflight_validate(cfg)
    if validation.errors:
        bullets = "\n".join(f"- {item}" for item in validation.errors)
        raise ValueError(f"Next-replica workflow preflight validation failed:\n{bullets}")

    before_status = summarize_replica_status(cfg.run.output_root, cfg.run.n_replicas)
    next_replica_id = before_status["next_replica_id"]
    outputs = {
        "status": "pending" if next_replica_id is not None else "all_replicas_completed",
        "replica_status_before": before_status,
        "target_n_replicas": before_status["target_n_replicas"],
        "next_replica_id": next_replica_id,
    }
    if next_replica_id is None:
        emit_progress(
            progress_callback,
            total_units,
            total_units,
            "md",
            f"All {before_status['target_n_replicas']} target MD replicas are already complete",
        )
        outputs["replica_status_after"] = before_status
        return outputs

    if cfg.do_prep:
        emit_progress(progress_callback, completed_units, total_units, "prep", "Preparing receptor and ligand inputs")
        from ..prep import run_prep_workflow

        outputs["prep"] = run_prep_workflow(cfg.prep, cfg.docking)
        completed_units += 1
        emit_progress(progress_callback, completed_units, total_units, "prep", "Prepared receptor and ligand inputs")

    _resolve_md_inputs(cfg)
    _bind_analysis_inputs_to_md_outputs(cfg)
    _bind_existing_analysis_inputs(cfg)

    emit_progress(
        progress_callback,
        completed_units,
        total_units,
        "md",
        f"Running next MD replica {next_replica_id}/{cfg.run.n_replicas}",
    )
    from ..simulate import run_single_replica_workflow

    md_offset = completed_units

    def _md_progress(event: ProgressEvent) -> None:
        emit_progress(
            progress_callback,
            md_offset + min(max(int(event.current), 0), 1),
            total_units,
            "md",
            event.detail,
            subcurrent=event.subcurrent,
            subtotal=event.subtotal,
            subdetail=event.subdetail or event.detail,
            subeta_seconds=event.subeta_seconds,
        )

    outputs["md"] = run_single_replica_workflow(cfg.run, int(next_replica_id), progress_callback=_md_progress)
    completed_units += 1
    after_status = summarize_replica_status(cfg.run.output_root, cfg.run.n_replicas)
    outputs["replica_status_after"] = after_status
    outputs["completed_replicas"] = after_status["completed_replicas"]
    outputs["remaining_replicas"] = after_status["remaining_replicas"]
    outputs["all_replicas_completed"] = after_status["all_replicas_completed"]
    ran_replica_status = next(
        (item for item in after_status["replicas"] if item["replica_id"] == int(next_replica_id)),
        {},
    )
    outputs["ran_replica_completed"] = bool(ran_replica_status.get("completed"))
    outputs["status"] = "completed" if outputs["ran_replica_completed"] else "incomplete_outputs"
    detail = (
        f"Completed MD replica {next_replica_id}/{cfg.run.n_replicas}"
        if outputs["ran_replica_completed"]
        else f"Finished MD replica {next_replica_id}/{cfg.run.n_replicas}, but required outputs are incomplete"
    )
    emit_progress(
        progress_callback,
        completed_units,
        total_units,
        "md",
        detail,
    )
    return outputs


def run_full_md_workflow(cfg: WorkflowConfig, progress_callback: ProgressCallback | None = None):
    cfg = normalize_workflow_paths(deepcopy(cfg))
    apply_plot_style_palette(cfg.plot_style)
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
    _bind_existing_analysis_inputs(cfg)
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
        if cfg.plot_selection.enabled("plot_workflow_reuse_csv") and reusable_basic_csv_available(cfg):
            emit_progress(progress_callback, completed_units, total_units, "basic_analysis", "Reusing existing basic-analysis CSV outputs")
            outputs["basic_replot_dir"] = run_basic_replot_from_csv(cfg)
            outputs.setdefault("reused_csv_sections", []).append("basic")
            outputs.setdefault("replotted_from_csv_sections", []).append("basic")
        else:
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
        if cfg.plot_selection.enabled("plot_workflow_reuse_csv") and reusable_waterbridge_csv_available(cfg):
            emit_progress(progress_callback, completed_units, total_units, "waterbridge_analysis", "Reusing existing water-bridge CSV outputs")
            outputs["waterbridge_replot_dir"] = run_waterbridge_replot_from_csv(cfg)
            outputs.setdefault("reused_csv_sections", []).append("waterbridge")
            outputs.setdefault("replotted_from_csv_sections", []).append("waterbridge")
        else:
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
        basic_reused = "basic" in outputs.get("reused_csv_sections", [])
        waterbridge_reused = "waterbridge" in outputs.get("reused_csv_sections", [])
        plot_cfg = deepcopy(cfg)
        plot_cfg.do_basic_analysis = bool(cfg.do_basic_analysis and not basic_reused)
        plot_cfg.do_waterbridge_analysis = bool(cfg.do_waterbridge_analysis and not waterbridge_reused)
        plot_cfg.do_advanced_analysis = False
        plot_cfg.do_mmgbsa_postprocess = False
        if plot_cfg.do_basic_analysis or plot_cfg.do_waterbridge_analysis:
            plot_units = int(plot_cfg.do_basic_analysis) + int(plot_cfg.do_waterbridge_analysis)
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
                plot_cfg,
                progress_callback=_plot_progress,
                include_mmgbsa_postprocess=False,
                include_organize_outputs=False,
            )
            replot_sections = plot_outputs.pop("replotted_from_csv_sections", [])
            outputs.update(plot_outputs)
            if replot_sections:
                seen_replots = set(outputs.get("replotted_from_csv_sections", []))
                outputs.setdefault("replotted_from_csv_sections", []).extend(
                    section for section in replot_sections if section not in seen_replots
                )
            completed_units += plot_units
    if cfg.do_advanced_analysis:
        if cfg.plot_selection.enabled("plot_workflow_reuse_csv") and reusable_advanced_csv_available(cfg):
            emit_progress(progress_callback, completed_units, total_units, "advanced_analysis", "Reusing existing advanced-analysis CSV outputs")
            outputs["advanced_replot_dir"] = run_advanced_replot_from_csv(cfg)
            outputs.setdefault("reused_csv_sections", []).append("advanced")
            outputs.setdefault("replotted_from_csv_sections", []).append("advanced")
        else:
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
        from ..postprocess.mmgbsa import run_mmgbsa_postprocess, summarize_mmgbsa_postprocess_result
        try:
            outputs["mmgbsa_postprocess"] = run_mmgbsa_postprocess(cfg.mmgbsa, cfg.plot_style, cfg.plot_selection)
            mmgbsa_detail = summarize_mmgbsa_postprocess_result(outputs["mmgbsa_postprocess"])
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
