from __future__ import annotations

from ...config import BasicAnalysisConfig, PlotSelectionConfig, PlotStyleConfig
from ...plotting.basic_combined import plot_combined_basic_results


def combine_basic_results(
    replica_results,
    cfg: BasicAnalysisConfig,
    style: PlotStyleConfig,
    plot_selection: PlotSelectionConfig | None = None,
):
    plot_combined_basic_results(
        replica_results,
        analysis_root=cfg.analysis_root,
        style=style,
        top_n_contacts_plot=cfg.top_n_contacts_plot,
        top_n_key_distance_residues=cfg.top_n_key_distance_residues,
        plot_selection=plot_selection,
    )
    return str(cfg.analysis_root)
