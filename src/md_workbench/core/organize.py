from __future__ import annotations

from pathlib import Path
import json
import shutil

from ..config import AdvancedAnalysisConfig, BasicAnalysisConfig, MMGBSAConfig, OutputBundleConfig, RunConfig, WaterBridgeConfig
from .files import ensure_dir

FIGURE_EXTS = {".png", ".svg", ".pdf"}
DATA_EXTS = {".csv", ".json", ".txt", ".log", ".dat", ".pdb"}


def _copy_with_structure(file_path: Path, source_root: Path, target_root: Path) -> str:
    rel = file_path.relative_to(source_root)
    dest = target_root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dest)
    return str(dest.resolve())


def _iter_files(root: Path):
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def _collect_combined_figures(src_root: Path, target_root: Path, label: str):
    copied = []
    if src_root.exists():
        for path in sorted(_iter_files(src_root)):
            if path.suffix.lower() in FIGURE_EXTS:
                copied.append(_copy_with_structure(path, src_root.parent if src_root.name == "combined" else src_root, target_root / label))
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


def organize_outputs(bundle_cfg: OutputBundleConfig, run_cfg: RunConfig, basic_cfg: BasicAnalysisConfig, water_cfg: WaterBridgeConfig, advanced_cfg: AdvancedAnalysisConfig, mmgbsa_cfg: MMGBSAConfig, do_basic: bool, do_water: bool, do_advanced: bool, do_mmgbsa: bool) -> dict[str, str | int]:
    if not bundle_cfg.enabled:
        return {}

    bundle_root = ensure_dir(bundle_cfg.root)
    figures_root = ensure_dir(bundle_root / bundle_cfg.figures_dir_name)
    data_root = ensure_dir(bundle_root / bundle_cfg.data_dir_name)

    figure_files = []
    data_files = []

    if do_basic:
        figure_files.extend(_collect_combined_figures(Path(basic_cfg.analysis_root) / "combined", figures_root, "basic"))
        data_files.extend(_collect_analysis_data(Path(basic_cfg.analysis_root), data_root / "basic"))
    if do_water:
        figure_files.extend(_collect_combined_figures(Path(water_cfg.analysis_root) / "combined", figures_root, "waterbridge"))
        data_files.extend(_collect_analysis_data(Path(water_cfg.analysis_root), data_root / "waterbridge"))
    if do_advanced:
        figure_files.extend(_collect_combined_figures(Path(advanced_cfg.analysis_root), figures_root, "advanced"))
        data_files.extend(_collect_analysis_data(Path(advanced_cfg.analysis_root), data_root / "advanced"))
    if do_mmgbsa:
        figure_files.extend(_collect_combined_figures(Path(mmgbsa_cfg.analysis_root), figures_root, "mmgbsa"))
        data_files.extend(_collect_analysis_data(Path(mmgbsa_cfg.analysis_root), data_root / "mmgbsa"))
    if bundle_cfg.include_simulation_logs:
        data_files.extend(_collect_simulation_logs(run_cfg, data_root))
    note_files = _write_organized_figure_notes(figure_files)

    manifest = {
        "bundle_root": str(bundle_root.resolve()),
        "figures_root": str(figures_root.resolve()),
        "data_root": str(data_root.resolve()),
        "n_figure_files": len(figure_files),
        "n_figure_note_files": len(note_files),
        "n_data_files": len(data_files),
        "notes": [
            "This project stores generated process files under the project work directory and curated deliverables under the project results directory.",
            "figures_root collects combined figures derived from three-replica analyses and optional MM/GBSA post-processing.",
            "Each bundled figure note is regenerated inside figures_root so its reproducibility section references bundled process_data files.",
            "data_root collects CSV/JSON/TXT/LOG/DAT/PDB analysis outputs and simulation logs, but does not duplicate heavy trajectory DCD files.",
        ],
    }
    with open(bundle_root / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    return manifest
