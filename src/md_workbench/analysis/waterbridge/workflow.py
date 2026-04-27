from __future__ import annotations

from pathlib import Path

from ...config import PlotSelectionConfig, PlotStyleConfig, WaterBridgeConfig
from ...config.plot_style_defaults import apply_plot_style_palette
from ...core import ensure_dir, resolve_replica_dirs
from ...core.progress import ProgressCallback, emit_progress
from ...plotting.waterbridge import plot_combined_waterbridge
from .replica import process_waterbridge_replica


def run_waterbridge_analysis(
    cfg: WaterBridgeConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
    progress_callback: ProgressCallback | None = None,
):
    apply_plot_style_palette(style)
    replica_dirs = resolve_replica_dirs(cfg.replica_root, cfg.replica_glob)
    if not replica_dirs:
        raise FileNotFoundError(f"没有找到重复目录: {cfg.replica_glob}")

    ensure_dir(cfg.analysis_root)
    replica_results = []
    total_replicas = len(replica_dirs)
    for idx, replica_dir in enumerate(replica_dirs, start=1):
        out_dir = ensure_dir(Path(cfg.analysis_root) / replica_dir.name)
        emit_progress(
            progress_callback,
            idx - 1,
            total_replicas,
            "waterbridge_analysis",
            f"Processing water-bridge replica {idx}/{total_replicas}: {replica_dir.name}",
        )

        def _replica_progress(current: int, total: int, detail: str) -> None:
            emit_progress(
                progress_callback,
                idx - 1,
                total_replicas,
                "waterbridge_analysis",
                f"Processing water-bridge replica {idx}/{total_replicas}: {replica_dir.name}",
                subcurrent=current,
                subtotal=total,
                subdetail=detail,
            )

        replica_results.append(
            process_waterbridge_replica(
                replica_dir,
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
            "waterbridge_analysis",
            f"Completed water-bridge replica {idx}/{total_replicas}: {replica_dir.name}",
            subcurrent=1,
            subtotal=1,
            subdetail="Replica completed",
        )
    plot_combined_waterbridge(replica_results, cfg.analysis_root, style, plot_selection=plot_selection)
    return str(Path(cfg.analysis_root).resolve())
