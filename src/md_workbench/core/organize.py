from __future__ import annotations

from pathlib import Path
import json
import shutil

from ..config import AdvancedAnalysisConfig, BasicAnalysisConfig, MMGBSAConfig, OutputBundleConfig, RunConfig, WaterBridgeConfig
from .figure_clusters import BUNDLE_CLUSTER_ORDER, classify_bundle_figure
from .files import ensure_dir

FIGURE_EXTS = {".png", ".svg", ".pdf"}
DATA_EXTS = {".csv", ".json", ".txt", ".log", ".dat", ".pdb"}


def _copy_with_structure(file_path: Path, source_root: Path, target_root: Path) -> str:
    rel = file_path.relative_to(source_root)
    dest = target_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return str(dest.resolve())


def _copy_to_path(file_path: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return str(dest.resolve())


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _collect_clustered_figures(src_root: Path, target_root: Path, source_kind: str):
    copied = []
    if src_root.exists():
        for path in sorted(_iter_files(src_root)):
            if path.suffix.lower() in FIGURE_EXTS:
                cluster, rel_dest, source_hint = classify_bundle_figure(path, src_root, source_kind)
                dest = target_root / cluster / rel_dest
                if dest.exists():
                    dest = target_root / cluster / source_hint / rel_dest
                copied.append(_copy_to_path(path, dest))
    return copied


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

    if do_basic:
        figure_files.extend(_collect_clustered_figures(Path(basic_cfg.analysis_root) / "combined", figures_root, "basic"))
        data_files.extend(_collect_analysis_data(Path(basic_cfg.analysis_root), data_root / "basic"))
    if do_water:
        figure_files.extend(_collect_clustered_figures(Path(water_cfg.analysis_root) / "combined", figures_root, "waterbridge"))
        data_files.extend(_collect_analysis_data(Path(water_cfg.analysis_root), data_root / "waterbridge"))
    if do_advanced:
        figure_files.extend(_collect_clustered_figures(Path(advanced_cfg.analysis_root), figures_root, "advanced"))
        data_files.extend(_collect_analysis_data(Path(advanced_cfg.analysis_root), data_root / "advanced"))
    if do_mmgbsa:
        figure_files.extend(_collect_clustered_figures(Path(mmgbsa_cfg.analysis_root), figures_root, "mmgbsa"))
        data_files.extend(_collect_analysis_data(Path(mmgbsa_cfg.analysis_root), data_root / "mmgbsa"))
    if bundle_cfg.include_simulation_logs:
        data_files.extend(_collect_simulation_logs(run_cfg, data_root))
    note_files = _write_organized_figure_notes(figure_files)

    cluster_counts = {cluster: 0 for cluster in BUNDLE_CLUSTER_ORDER}
    for file_path in figure_files:
        rel = Path(file_path).resolve().relative_to(figures_root.resolve())
        cluster = rel.parts[0] if rel.parts else ""
        cluster_counts.setdefault(cluster, 0)
        cluster_counts[cluster] += 1
    cluster_counts = {name: count for name, count in cluster_counts.items() if count > 0}

    manifest = {
        "bundle_root": str(bundle_root.resolve()),
        "figures_root": str(figures_root.resolve()),
        "data_root": str(data_root.resolve()),
        "n_figure_files": len(figure_files),
        "n_figure_note_files": len(note_files),
        "n_data_files": len(data_files),
        "figure_clusters": cluster_counts,
        "notes": [
            "This project stores generated process files under the project work directory and curated deliverables under the project results directory.",
            "figures_root organizes bundled figures by information clusters so panels that belong in one manuscript composite figure stay together.",
            "Each bundled figure note is regenerated inside figures_root so its reproducibility section references bundled process_data files.",
            "data_root collects CSV/JSON/TXT/LOG/DAT/PDB analysis outputs and simulation logs, but does not duplicate heavy trajectory DCD files.",
            "The figure and process-data bundle directories are refreshed on each organize step to avoid stale outputs from earlier layouts.",
        ],
    }
    with open(bundle_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
