from __future__ import annotations

from pathlib import Path

from ...config import BasicAnalysisConfig, PlotSelectionConfig, PlotStyleConfig
from ...core import ensure_dir, resolve_replica_dirs
from ...core.progress import ProgressCallback, emit_progress
from .combine import combine_basic_results
from .replica import process_replica


def run_basic_analysis(
    cfg: BasicAnalysisConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
    progress_callback: ProgressCallback | None = None,
):
    replica_dirs = resolve_replica_dirs(cfg.replica_root, cfg.replica_glob)
    if not replica_dirs:
        raise FileNotFoundError(f"没有找到任何重复目录: {cfg.replica_glob}")

    ensure_dir(cfg.analysis_root)
    replica_results = []
    total_replicas = len(replica_dirs)
    for idx, replica_dir in enumerate(replica_dirs, start=1):
        out_dir = ensure_dir(Path(cfg.analysis_root) / replica_dir.name)
        emit_progress(
            progress_callback,
            idx - 1,
            total_replicas,
            "basic_analysis",
            f"Processing basic-analysis replica {idx}/{total_replicas}: {replica_dir.name}",
        )

        def _replica_progress(current: int, total: int, detail: str) -> None:
            emit_progress(
                progress_callback,
                idx - 1,
                total_replicas,
                "basic_analysis",
                f"Processing basic-analysis replica {idx}/{total_replicas}: {replica_dir.name}",
                subcurrent=current,
                subtotal=total,
                subdetail=detail,
            )

        replica_results.append(
            process_replica(
                replica_dir,
                cfg.ligand_sdf,
                out_dir,
                cfg,
                style,
                plot_selection,
                progress_callback=_replica_progress,
            )
        )
        emit_progress(
            progress_callback,
            idx,
            total_replicas,
            "basic_analysis",
            f"Completed basic-analysis replica {idx}/{total_replicas}: {replica_dir.name}",
            subcurrent=1,
            subtotal=1,
            subdetail="Replica completed",
        )
    return combine_basic_results(replica_results, cfg, style, plot_selection)
