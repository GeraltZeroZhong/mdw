from __future__ import annotations

from pathlib import Path
from collections import Counter
import json
import shutil
import textwrap

from ..config import AdvancedAnalysisConfig, BasicAnalysisConfig, MMGBSAConfig, OutputBundleConfig, RunConfig, WaterBridgeConfig
from .figure_clusters import BUNDLE_CLUSTER_ORDER, classify_bundle_figure
from .files import ensure_dir

FIGURE_EXTS = {".png", ".svg", ".pdf"}
DATA_EXTS = {".csv", ".json", ".txt", ".log", ".dat", ".pdb"}
THEMATIC_FIGURE_GROUPS = {
    "interaction_networks": {
        "contact_count_combined",
        "contact_occupancy_top20",
        "contact_replicate_heatmap",
        "hbond_count_combined",
        "hbond_residue_occupancy_top20",
        "interaction_fingerprint_heatmap",
        "salt_bridge_count_combined",
        "salt_bridge_residue_occupancy",
        "waterbridge_count_combined",
        "waterbridge_residue_occupancy_top20",
    },
    "stability_compaction": {
        "buried_surface_combined",
        "convergence_block_heatmap",
        "ligand_sasa_combined",
        "min_distance_combined",
        "radius_of_gyration_combined",
        "replicate_consistency_boxplot",
        "replicate_consistency_zscore_heatmap",
        "rmsd_replot_ligand",
        "rmsd_replot_protein",
        "sasa_complex_protein_combined",
    },
    "structure_pose": {
        "dssp_fractions_combined",
        "dssp_residue_occupancy_combined",
        "ligand_com_distance_combined",
        "ligand_orientation_angle_combined",
        "rmsf_ca_combined",
    },
    "conformational_landscape": {
        "cluster_population_overall",
        "clusters_tic1_tic2",
        "explained_variance_ratio",
        "free_energy_landscape_pc1_pc2",
        "free_energy_landscape_pc1_pc2_3d",
        "free_energy_landscape_tic1_tic2",
        "free_energy_landscape_tic1_tic2_3d",
        "pc1_pc2_scatter",
        "representative_state_snapshots",
        "singular_values",
        "state_population_by_replica",
        "tic1_tic2_scatter",
    },
    "kinetics": {
        "chapman_kolmogorov_test",
        "implied_timescales_lag_scan",
        "implied_timescales_single_lag",
        "lag_scan_frame_support",
        "lag_scan_usable_segments",
        "state_network",
        "stationary_distribution",
        "transition_dominant_exchange",
        "transition_matrix_heatmap",
        "transition_residence_departure",
    },
    "binding_energy": {
        "mmgbsa_summary",
        "mmgbsa_per_frame",
        "mmgbsa_per_residue",
        "mmgbsa_delta_total_summary",
        "mmgbsa_delta_total_per_frame",
        "mmgbsa_delta_total_distribution",
        "mmgbsa_per_residue_heatmap",
        "mmgbsa_per_residue_top",
    },
}
THEMATIC_FIGURE_LOOKUP = {
    stem: group
    for group, stems in THEMATIC_FIGURE_GROUPS.items()
    for stem in stems
}
THEMATIC_FIGURE_PREFIX_LOOKUP = {
    "key_contact_distance_": "interaction_networks",
}
DEPRECATED_BUNDLE_FIGURE_STEMS = {
    "key_contact_distance_traces",
    "rmsd_combined",
    "sasa_components_combined",
    "waterbridge_count_replot",
}
FIGURE_CLUSTER_ORDER = list(THEMATIC_FIGURE_GROUPS.keys()) + ["uncategorized"]
NONFUNCTIONAL_RELATIVE_PREFIXES = {"combined", "pca", "tica", "clustering", "msm", "snapshots"}


def _same_resolved_path(source: Path, dest: Path) -> bool:
    return source.resolve() == dest.resolve()


def _copy_with_structure(file_path: Path, source_root: Path, target_root: Path) -> str:
    rel = file_path.relative_to(source_root)
    dest = target_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _same_resolved_path(file_path, dest):
        shutil.copy2(file_path, dest)
    return str(dest.resolve())


def _copy_to_path(file_path: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not _same_resolved_path(file_path, dest):
        shutil.copy2(file_path, dest)
    return str(dest.resolve())


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _refresh_bundle_layout(bundle_root: Path, bundle_cfg: OutputBundleConfig) -> None:
    for relative in [bundle_cfg.figures_dir_name, bundle_cfg.data_dir_name]:
        target = bundle_root / relative
        if target.exists():
            shutil.rmtree(target)
    manifest_path = bundle_root / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _cluster_relative_path(file_path: Path, source_root: Path) -> Path:
    parts = list(file_path.relative_to(source_root).parts)
    while parts and parts[0] in NONFUNCTIONAL_RELATIVE_PREFIXES:
        parts.pop(0)
    return Path(*parts) if parts else Path(file_path.name)


def _classify_figure_destination(file_path: Path, source_root: Path, label: str) -> tuple[str, Path]:
    group = THEMATIC_FIGURE_LOOKUP.get(file_path.stem)
    if group is None:
        group = next(
            (prefix_group for prefix, prefix_group in THEMATIC_FIGURE_PREFIX_LOOKUP.items() if file_path.stem.startswith(prefix)),
            "uncategorized",
        )
    relative = _cluster_relative_path(file_path, source_root)
    if group == "uncategorized":
        return group, Path(label) / relative
    return group, relative


def _collect_clustered_figures(src_root: Path, target_root: Path, label: str):
    copied = []
    group_counts: Counter[str] = Counter()
    if src_root.exists():
        for path in sorted(_iter_files(src_root)):
            if path.suffix.lower() in FIGURE_EXTS:
                if path.stem in DEPRECATED_BUNDLE_FIGURE_STEMS:
                    continue
                group, relative = _classify_figure_destination(path, src_root, label)
                dest = target_root / group / relative
                copied.append(_copy_to_path(path, dest))
                group_counts[group] += 1
    return copied, group_counts


def _collect_analysis_data(src_root: Path, target_root: Path):
    copied = []
    if src_root.exists():
        for path in sorted(_iter_files(src_root)):
            if path.suffix.lower() in DATA_EXTS:
                copied.append(_copy_with_structure(path, src_root, target_root))
    return copied


def _write_organized_figure_notes(figure_files: list[str]):
    if not figure_files:
        return []

    # Regenerate notes in the bundled figures tree so the referenced data paths
    # point to bundled `process_data` outputs instead of the original work tree.
    from ..plotting import write_figure_note

    base_to_formats: dict[Path, set[str]] = {}
    for file_path in figure_files:
        path = Path(file_path)
        if path.suffix.lower() not in FIGURE_EXTS:
            continue
        base_to_formats.setdefault(path.with_suffix(""), set()).add(path.suffix.lstrip(".").lower())

    note_files = []
    for base_path, formats in sorted(base_to_formats.items()):
        note_path = write_figure_note(base_path, formats=sorted(formats))
        note_files.append(str(note_path.resolve()))
    return note_files


def _preview_label(file_path: Path, group_root: Path) -> str:
    label = file_path.relative_to(group_root).with_suffix("").as_posix()
    if len(label) <= 54:
        return label
    return "\n".join(textwrap.wrap(label, width=42, max_lines=2, placeholder="..."))


def _write_figure_preview_sheets(figures_root: Path) -> list[str]:
    try:
        import matplotlib.image as mpimg
        import matplotlib.pyplot as plt
    except Exception:
        return []

    preview_root = ensure_dir(figures_root / "_previews")
    preview_files: list[str] = []
    for group in FIGURE_CLUSTER_ORDER:
        group_root = figures_root / group
        if not group_root.exists():
            continue
        image_paths = sorted(path for path in group_root.rglob("*.png") if "_previews" not in path.parts)
        if not image_paths:
            continue

        cols = min(4, max(1, len(image_paths)))
        rows = (len(image_paths) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 2.6), squeeze=False)
        fig.patch.set_facecolor("white")

        for ax in axes.ravel():
            ax.set_axis_off()

        for ax, image_path in zip(axes.ravel(), image_paths):
            try:
                ax.imshow(mpimg.imread(image_path))
            except Exception:
                continue
            ax.set_title(_preview_label(image_path, group_root), fontsize=7.2, color="black", pad=4)

        fig.suptitle(f"{group} preview", fontsize=12, color="black", weight="semibold")
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
        out_path = preview_root / f"{group}.png"
        fig.savefig(out_path, dpi=180, facecolor="white", bbox_inches="tight")
        plt.close(fig)
        preview_files.append(str(out_path.resolve()))
    return preview_files


def _collect_simulation_logs(run_cfg: RunConfig, target_root: Path):
    copied = []
    root = Path(run_cfg.output_root)
    if not root.exists():
        return copied
    for replica_dir in sorted(p for p in root.glob("replica_*") if p.is_dir()):
        path = replica_dir / "md.log"
        if path.exists():
            copied.append(_copy_with_structure(path, root, target_root / "simulation_logs"))
    return copied


def _clear_dir_contents(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def organize_outputs(bundle_cfg: OutputBundleConfig, run_cfg: RunConfig, basic_cfg: BasicAnalysisConfig, water_cfg: WaterBridgeConfig, advanced_cfg: AdvancedAnalysisConfig, mmgbsa_cfg: MMGBSAConfig, do_basic: bool, do_water: bool, do_advanced: bool, do_mmgbsa: bool) -> dict[str, str | int]:
    if not bundle_cfg.enabled:
        return {}

    bundle_root = ensure_dir(bundle_cfg.root)
    _refresh_bundle_layout(bundle_root, bundle_cfg)
    figures_root = ensure_dir(bundle_root / bundle_cfg.figures_dir_name)
    data_root = ensure_dir(bundle_root / bundle_cfg.data_dir_name)

    cleared_roots = set()
    bundle_root_resolved = bundle_root.resolve()
    for root in (figures_root, data_root):
        resolved = root.resolve()
        if resolved == bundle_root_resolved or resolved in cleared_roots:
            continue
        _clear_dir_contents(root)
        cleared_roots.add(resolved)

    figure_files = []
    data_files = []
    figure_cluster_counts: Counter[str] = Counter()

    if do_basic:
        copied, counts = _collect_clustered_figures(Path(basic_cfg.analysis_root) / "combined", figures_root, "basic")
        figure_files.extend(copied)
        figure_cluster_counts.update(counts)
        data_files.extend(_collect_analysis_data(Path(basic_cfg.analysis_root), data_root / "basic"))
    if do_water:
        copied, counts = _collect_clustered_figures(Path(water_cfg.analysis_root) / "combined", figures_root, "waterbridge")
        figure_files.extend(copied)
        figure_cluster_counts.update(counts)
        data_files.extend(_collect_analysis_data(Path(water_cfg.analysis_root), data_root / "waterbridge"))
    if do_advanced:
        copied, counts = _collect_clustered_figures(Path(advanced_cfg.analysis_root), figures_root, "advanced")
        figure_files.extend(copied)
        figure_cluster_counts.update(counts)
        data_files.extend(_collect_analysis_data(Path(advanced_cfg.analysis_root), data_root / "advanced"))
    if do_mmgbsa:
        copied, counts = _collect_clustered_figures(Path(mmgbsa_cfg.analysis_root), figures_root, "mmgbsa")
        figure_files.extend(copied)
        figure_cluster_counts.update(counts)
        data_files.extend(_collect_analysis_data(Path(mmgbsa_cfg.analysis_root), data_root / "mmgbsa"))
    if bundle_cfg.include_simulation_logs:
        data_files.extend(_collect_simulation_logs(run_cfg, data_root))
    note_files = _write_organized_figure_notes(figure_files)
    preview_files = _write_figure_preview_sheets(figures_root)

    ordered_cluster_counts = {
        group: int(figure_cluster_counts[group])
        for group in FIGURE_CLUSTER_ORDER
        if figure_cluster_counts.get(group, 0) > 0
    }

    manifest = {
        "bundle_root": str(bundle_root.resolve()),
        "figures_root": str(figures_root.resolve()),
        "data_root": str(data_root.resolve()),
        "n_figure_files": len(figure_files),
        "n_figure_note_files": len(note_files),
        "n_preview_files": len(preview_files),
        "n_data_files": len(data_files),
        "figure_clusters": ordered_cluster_counts,
        "preview_files": preview_files,
        "notes": [
            "This project stores generated process files under the project work directory and curated deliverables under the project results directory.",
            "figures_root organizes bundled figures by information clusters so panels that belong in one manuscript composite figure stay together.",
            "figures_root/_previews contains PNG-only contact sheets for quick visual review and is not treated as a manuscript figure set.",
            "Each bundled figure note is regenerated inside figures_root so its reproducibility section references bundled process_data files.",
            "data_root collects CSV/JSON/TXT/LOG/DAT/PDB analysis outputs and simulation logs, but does not duplicate heavy trajectory DCD files.",
            "The figure and process-data bundle directories are refreshed on each organize step to avoid stale outputs from earlier layouts.",
        ],
    }
    with open(bundle_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
