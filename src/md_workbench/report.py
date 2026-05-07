from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
from copy import deepcopy
import csv
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import zipfile

from .config import WorkflowConfig
from .core.pathing import normalize_workflow_paths


EMU_PER_INCH = 914400
DOCX_FIXED_TIMESTAMP = (1980, 1, 1, 0, 0, 0)

FIGURE_GROUP_ORDER = [
    "_previews",
    "interaction_networks",
    "stability_compaction",
    "structure_pose",
    "conformational_landscape",
    "kinetics",
    "binding_energy",
    "uncategorized",
]

FIGURE_GROUP_TITLES = {
    "_previews": "图表总览",
    "interaction_networks": "相互作用网络",
    "stability_compaction": "稳定性与紧致性",
    "structure_pose": "结构与配体姿态",
    "conformational_landscape": "构象景观",
    "kinetics": "动力学与 MSM",
    "binding_energy": "结合自由能",
    "uncategorized": "未分类图",
}

PREVIEW_NAME_MAP = {
    "interaction_networks": "相互作用网络预览图",
    "stability_compaction": "稳定性与紧致性预览图",
    "structure_pose": "结构与配体姿态预览图",
    "conformational_landscape": "构象景观预览图",
    "kinetics": "动力学与 MSM 预览图",
    "binding_energy": "结合自由能预览图",
    "uncategorized": "未分类图预览图",
}

METHOD_REFERENCES = [
    "Eberhardt, J., Santos-Martins, D., Tillack, A. F., & Forli, S. (2021). AutoDock Vina 1.2.0: New docking methods, expanded force field, and Python bindings. Journal of Chemical Information and Modeling, 61(8), 3891–3898. https://doi.org/10.1021/acs.jcim.1c00203",
    "Eastman, P., Galvelis, R., Peláez, R. P., Abreu, C. R. A., Farr, S. E., Gallicchio, E., Gorenko, A., Henry, M. M., Hu, F., Huang, J., Krämer, A., Michel, J., Mitchell, J. A., Pande, V. S., Rodrigues, J. P. G. L. M., Rodriguez-Guerra, J., Simmonett, A. C., Singh, S., Swails, J., Turner, P., Wang, Y., Zhang, I., Chodera, J. D., De Fabritiis, G., & Markland, T. E. (2024). OpenMM 8: Molecular dynamics simulation with machine learning potentials. The Journal of Physical Chemistry B, 128(1), 109–116. https://doi.org/10.1021/acs.jpcb.3c06662",
    "Krivák, R., & Hoksza, D. (2018). P2Rank: Machine learning based tool for rapid and accurate prediction of ligand binding sites from protein structure. Journal of Cheminformatics, 10(1), Article 39. https://doi.org/10.1186/s13321-018-0285-8",
    "Zhong, S., & Jiang, Y. (2026). ProtCross: Bridging the PDB-AlphaFold Gap for Binding Site Prediction with Protein Point Clouds. Journal of Chemical Information and Modeling, 66(7), 3688–3701. https://doi.org/10.1021/acs.jcim.5c03224",
    "Santos-Martins, D., He, Y., Eberhardt, J., Sharma, P., Bruciaferri, N., Holcomb, M., Llanos, M. A., Hansel-Harris, A., Barkdull, A. P., Tillack, A. F., Bianco, G., Paulsen, M.-L., Mato, J., Taneja, I., & Forli, S. (2025). Meeko: Molecule parametrization and software interoperability for docking and beyond. Journal of Chemical Information and Modeling, 65(24), 13045–13050. https://doi.org/10.1021/acs.jcim.5c02271",
    "Trott, O., & Olson, A. J. (2010). AutoDock Vina: Improving the speed and accuracy of docking with a new scoring function, efficient optimization, and multithreading. Journal of Computational Chemistry, 31(2), 455–461. https://doi.org/10.1002/jcc.21334",
]

COMMON_NAME_MAP = {
    "protein_backbone_rmsd_mean_A": "蛋白主链 RMSD",
    "ligand_heavy_rmsd_mean_A": "配体重原子 RMSD",
    "mean_min_distance_A": "配体-蛋白最小距离",
    "mean_rg_A": "蛋白回转半径",
    "mean_contact_count": "接触数",
    "mean_hbond_count": "氢键数",
    "mean_salt_bridge_count": "盐桥数",
    "mean_count": "水桥数量",
    "mean_com_distance_A": "配体-口袋质心距离",
    "mean_orientation_angle_deg": "配体取向角",
    "complex_sasa_mean_A2": "复合物 SASA",
    "protein_sasa_mean_A2": "蛋白 SASA",
    "ligand_sasa_mean_A2": "配体 SASA",
    "rmsf_mean_A": "Cα RMSF",
    "contact_occupancy_mean": "接触占据比例",
    "hbond_occupancy_mean": "氢键占据比例",
    "salt_bridge_occupancy_mean": "盐桥占据比例",
    "waterbridge_occupancy_mean": "水桥占据比例",
    "explained_variance_ratio": "解释方差比例",
    "cumulative_explained_variance_ratio": "累计解释方差比例",
    "singular_value": "奇异值",
    "stationary_probability": "平稳概率",
    "fraction": "占比",
    "frame_fraction_total": "总帧占比",
    "timescale_frames": "时间尺度",
    "absolute_error": "绝对误差",
}


@dataclass
class FigureReportItem:
    image_path: Path
    group: str
    relative_stem: str
    display_name: str
    explanation: str
    caption_example: str
    data_paths: list[Path]

    @property
    def heading_label(self) -> str:
        return f"{self.display_name}（{self.relative_stem}）"


@dataclass
class DockingReportPayload:
    summary_text: str
    figure_path: Path | None = None
    figure_status: str = ""
    pml_path: Path | None = None


def build_workflow_report_for_config(cfg: WorkflowConfig) -> dict[str, str | int]:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    return build_workflow_report(
        Path(cfg.output_bundle.root),
        figures_dir_name=cfg.output_bundle.figures_dir_name,
        data_dir_name=cfg.output_bundle.data_dir_name,
        report_docx_name=getattr(cfg.output_bundle, "report_docx_name", "workflow_report.docx"),
        workflow_cfg=cfg,
    )


def build_workflow_report(
    bundle_root: str | Path,
    *,
    figures_dir_name: str = "figures_combined",
    data_dir_name: str = "process_data",
    report_docx_name: str = "workflow_report.docx",
    workflow_cfg: WorkflowConfig | None = None,
) -> dict[str, str | int]:
    bundle_root = Path(bundle_root)
    figures_root = bundle_root / figures_dir_name
    data_root = bundle_root / data_dir_name
    if not figures_root.exists():
        raise FileNotFoundError(f"找不到图片目录: {figures_root.resolve()}")

    items = _collect_report_items(figures_root, data_root)
    if not items:
        raise ValueError(f"未在 {figures_root.resolve()} 中找到可写入报告的 PNG 图片")

    docking_payload = _build_docking_payload(bundle_root, workflow_cfg)
    out_path = Path(report_docx_name)
    if not out_path.is_absolute():
        out_path = bundle_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_docx_report(out_path, figures_root, data_root, items, workflow_cfg, docking_payload)
    return {
        "report_docx": str(out_path.resolve()),
        "n_report_figures": len(items),
        "docking_figure": str(docking_payload.figure_path.resolve()) if docking_payload.figure_path else "",
    }


def _collect_report_items(figures_root: Path, data_root: Path) -> list[FigureReportItem]:
    image_paths = [path for path in figures_root.rglob("*.png") if path.is_file()]

    def sort_key(path: Path) -> tuple[int, str]:
        rel = path.relative_to(figures_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        try:
            group_idx = FIGURE_GROUP_ORDER.index(group)
        except ValueError:
            group_idx = len(FIGURE_GROUP_ORDER)
        return group_idx, rel.as_posix()

    items: list[FigureReportItem] = []
    for image_path in sorted(image_paths, key=sort_key):
        rel = image_path.relative_to(figures_root)
        group = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        relative_stem = rel.with_suffix("").as_posix()
        if group != "uncategorized" and relative_stem.startswith(f"{group}/"):
            relative_stem = relative_stem[len(group) + 1 :]

        note = _parse_figure_note(image_path.with_suffix(".txt"))
        display_name = note.get("name") or _humanize_stem(image_path.stem)
        if group == "_previews" and "name" not in note:
            display_name = PREVIEW_NAME_MAP.get(image_path.stem, f"{_humanize_stem(image_path.stem)}预览图")
        data_paths = _resolve_data_paths(note.get("data_paths", []), figures_root, data_root)
        if not data_paths:
            data_paths = _guess_data_paths(image_path.stem, data_root)
        explanation = note.get("explanation") or f"该图展示“{display_name}”对应的分析结果。"
        caption_example = note.get("caption_example") or ""
        if group == "_previews" and "explanation" not in note:
            explanation = "该图为对应主题下主要分析图的总览拼图，用于概括该主题包含的图件类型和整体结果范围。"
            caption_example = f"{display_name}。该图汇总展示该主题下的主要分析图件，不作为单独定量指标。"

        items.append(
            FigureReportItem(
                image_path=image_path,
                group=group,
                relative_stem=relative_stem,
                display_name=display_name,
                explanation=explanation,
                caption_example=caption_example,
                data_paths=data_paths,
            )
        )
    return items


def _parse_figure_note(note_path: Path) -> dict[str, str | list[str]]:
    if not note_path.exists():
        return {}
    text = note_path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    result: dict[str, str | list[str]] = {}
    data_paths: list[str] = []
    explanation: list[str] = []
    caption: list[str] = []
    section: str | None = None
    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("图片名称:"):
            result["name"] = line.split(":", 1)[1].strip()
            section = None
            continue
        if line == "图片解释说明:":
            section = "explanation"
            continue
        if line == "图注示例:":
            section = "caption"
            continue
        if line == "用于复现该图片的所用数据:":
            section = "data"
            continue
        if section == "explanation":
            if line:
                explanation.append(line)
            continue
        if section == "caption":
            if line:
                caption.append(line)
            continue
        if section == "data" and line.startswith("- "):
            data_paths.append(line[2:].strip())
    if explanation:
        result["explanation"] = " ".join(explanation)
    if caption:
        result["caption_example"] = " ".join(caption)
    if data_paths:
        result["data_paths"] = data_paths
    return result


def _resolve_data_paths(raw_paths: list[str] | str, figures_root: Path, data_root: Path) -> list[Path]:
    if isinstance(raw_paths, str):
        raw_iter = [raw_paths]
    else:
        raw_iter = raw_paths
    bundle_root = figures_root.parent
    seen: set[Path] = set()
    resolved_paths: list[Path] = []

    def add_candidate(candidate: Path) -> None:
        if not candidate.exists() or not candidate.is_file():
            return
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            resolved_paths.append(resolved)

    for raw in raw_iter:
        clean = str(raw).strip()
        if not clean or clean.startswith("未自动匹配"):
            continue
        path = Path(clean).expanduser()
        add_candidate(path)
        if "process_data" in path.parts:
            idx = path.parts.index("process_data")
            suffix = Path(*path.parts[idx + 1 :])
            add_candidate(data_root / suffix)
        if not path.is_absolute():
            add_candidate(bundle_root / path)
            add_candidate(data_root / path)
    return resolved_paths


def _guess_data_paths(stem: str, data_root: Path) -> list[Path]:
    if not data_root.exists():
        return []
    candidates: list[Path] = []
    for suffix in (".csv", ".json", ".txt"):
        candidates.extend(sorted(data_root.rglob(f"{stem}{suffix}")))
    if stem.startswith("key_contact_distance_"):
        candidates.extend(sorted(data_root.rglob("key_contact_distance_traces.csv")))
        candidates.extend(sorted(data_root.rglob("contact_occupancy_distance_summary.csv")))
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique


def _build_docking_payload(bundle_root: Path, workflow_cfg: WorkflowConfig | None) -> DockingReportPayload:
    if workflow_cfg is None:
        existing = bundle_root / "report_assets" / "docking_pose_cartoon.png"
        if existing.exists():
            return DockingReportPayload(
                summary_text="结果包中包含最佳打分结合构象的对接展示图；由于未提供项目配置文件，本节不重新解析对接日志。",
                figure_path=existing,
                figure_status="下图展示最佳打分结合构象及结合口袋邻近残基。",
                pml_path=bundle_root / "report_assets" / "docking_pose_cartoon.pml",
            )
        return DockingReportPayload(summary_text="未提供项目配置文件，因此本节无法解析对接日志或展示对接构象图。")

    log_path = Path(workflow_cfg.docking.docking_log)
    parsed = _parse_vina_log(log_path)
    summary_text = _docking_summary_text(parsed, log_path)
    figure_path, pml_path, figure_status = _generate_docking_pose_figure(bundle_root, workflow_cfg, parsed)
    return DockingReportPayload(
        summary_text=summary_text,
        figure_path=figure_path,
        figure_status=figure_status,
        pml_path=pml_path,
    )


def _parse_vina_log(log_path: Path) -> dict[str, object]:
    parsed: dict[str, object] = {
        "modes": [],
        "center": None,
        "size": None,
        "seed": None,
        "exhaustiveness": None,
    }
    if not log_path.exists():
        return parsed
    text = log_path.read_text(encoding="utf-8", errors="replace")
    mode_by_rank: dict[int, tuple[int, float, float | None, float | None]] = {}
    mode_pattern = re.compile(r"^\s*(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")
    for line in text.splitlines():
        stripped = line.strip()
        mode_match = mode_pattern.match(line)
        if mode_match:
            rank = int(mode_match.group(1))
            mode_by_rank.setdefault(
                rank,
                (
                    rank,
                    float(mode_match.group(2)),
                    float(mode_match.group(3)),
                    float(mode_match.group(4)),
                ),
            )
            continue
        center_match = re.search(r"Grid center:\s*X\s*([-\d.]+)\s*Y\s*([-\d.]+)\s*Z\s*([-\d.]+)", stripped)
        if center_match and parsed["center"] is None:
            parsed["center"] = tuple(float(center_match.group(i)) for i in range(1, 4))
            continue
        size_match = re.search(r"Grid size\s*:\s*X\s*([-\d.]+)\s*Y\s*([-\d.]+)\s*Z\s*([-\d.]+)", stripped)
        if size_match and parsed["size"] is None:
            parsed["size"] = tuple(float(size_match.group(i)) for i in range(1, 4))
            continue
        seed_match = re.search(r"random seed:\s*(\d+)", stripped)
        if seed_match and parsed["seed"] is None:
            parsed["seed"] = int(seed_match.group(1))
            continue
        exhaustiveness_match = re.search(r"Exhaustiveness:\s*(\d+)", stripped)
        if exhaustiveness_match and parsed["exhaustiveness"] is None:
            parsed["exhaustiveness"] = int(exhaustiveness_match.group(1))
    parsed["modes"] = [mode_by_rank[key] for key in sorted(mode_by_rank)]
    return parsed


def _docking_summary_text(parsed: dict[str, object], log_path: Path) -> str:
    modes = parsed.get("modes") or []
    if not modes:
        if log_path.exists():
            return f"已找到对接日志 {log_path.name}，但未自动解析到 Vina 打分表。"
        return "未找到可解析的 Vina 对接日志，因此无法自动提取对接打分。"
    mode_rows = list(modes)
    best = min(mode_rows, key=lambda item: item[1])
    affinities = [item[1] for item in mode_rows]
    parts = [
        f"AutoDock Vina 共输出 {len(mode_rows)} 个结合模式；最佳模式为 mode {best[0]}，亲和力为 {_fmt_value(best[1], 'energy_kcal_mol')}。",
        f"全部模式评分范围为 {_fmt_value(min(affinities), 'energy_kcal_mol')} 至 {_fmt_value(max(affinities), 'energy_kcal_mol')}。",
    ]
    center = parsed.get("center")
    size = parsed.get("size")
    if isinstance(center, tuple) and isinstance(size, tuple):
        parts.append(
            "对接搜索盒中心为 "
            f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) Å，"
            f"盒子尺寸为 {size[0]:.1f} x {size[1]:.1f} x {size[2]:.1f} Å。"
        )
    seed = parsed.get("seed")
    exhaustiveness = parsed.get("exhaustiveness")
    if seed is not None or exhaustiveness is not None:
        details = []
        if exhaustiveness is not None:
            details.append(f"穷举度 {exhaustiveness}")
        if seed is not None:
            details.append(f"随机种子 {seed}")
        parts.append("参数记录: " + "，".join(details) + "。")
    return "".join(parts)


def _generate_docking_pose_figure(
    bundle_root: Path,
    workflow_cfg: WorkflowConfig,
    parsed: dict[str, object],
) -> tuple[Path | None, Path | None, str]:
    assets_root = bundle_root / "report_assets"
    assets_root.mkdir(parents=True, exist_ok=True)
    receptor = _first_existing_path([workflow_cfg.prep.receptor_output, workflow_cfg.prep.receptor_input])
    ligand = _first_existing_path(
        [
            workflow_cfg.docking.extracted_pose_pdb,
            workflow_cfg.docking.extracted_pose_sdf,
            workflow_cfg.docking.docking_sdf,
            workflow_cfg.docking.ligand_output_sdf,
            workflow_cfg.docking.ligand_sdf_input,
        ]
    )
    if receptor is None or ligand is None:
        return None, None, "未找到可用于绘制对接构象图的受体 PDB 或配体构象文件。"

    pymol_cmd = _find_pymol_command()
    if not pymol_cmd:
        return None, None, "当前环境未找到 PyMOL 命令，因此本报告未包含对接构象图。"

    receptor_asset = assets_root / "docking_receptor.pdb"
    ligand_asset = assets_root / f"docking_ligand{ligand.suffix.lower()}"
    shutil.copy2(receptor, receptor_asset)
    shutil.copy2(ligand, ligand_asset)

    out_png = assets_root / "docking_pose_cartoon.png"
    pml_path = assets_root / "docking_pose_cartoon.pml"
    log_path = assets_root / "docking_pose_cartoon_pymol.log"
    best_affinity = None
    modes = parsed.get("modes") or []
    if modes:
        best_affinity = min(list(modes), key=lambda item: item[1])[1]
    pml_path.write_text(_pymol_script(receptor_asset, ligand_asset, out_png, best_affinity), encoding="utf-8")
    try:
        completed = subprocess.run(
            [*pymol_cmd, "-cq", str(pml_path)],
            cwd=assets_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        log_path.write_text(str(exc) + "\n", encoding="utf-8")
        return None, pml_path, f"对接构象图渲染失败: {exc}"

    log_path.write_text(
        "COMMAND: " + " ".join([*pymol_cmd, "-cq", str(pml_path)]) + "\n\n"
        f"RETURN CODE: {completed.returncode}\n\nSTDOUT:\n{completed.stdout}\n\nSTDERR:\n{completed.stderr}\n",
        encoding="utf-8",
    )
    if completed.returncode != 0 or not out_png.exists() or out_png.stat().st_size <= 0:
        return None, pml_path, f"对接构象图未成功生成；渲染日志见 {_relative_or_absolute(log_path, bundle_root)}。"
    return out_png, pml_path, "下图展示最佳打分结合构象及结合口袋邻近残基。"


def _first_existing_path(values: list[str]) -> Path | None:
    for value in values:
        if not str(value).strip():
            continue
        path = Path(value)
        if path.exists():
            return path.resolve()
    return None


def _find_pymol_command() -> list[str] | None:
    home = Path.home()
    for candidate in [
        home / "miniconda3" / "envs" / "pymolfig" / "bin" / "pymol",
        home / "mambaforge" / "envs" / "pymolfig" / "bin" / "pymol",
    ]:
        if candidate.exists():
            return [str(candidate)]
    env_cmd = os.environ.get("PYMOL_CMD", "").strip()
    if env_cmd:
        return shlex.split(env_cmd)
    direct = shutil.which("pymol")
    if direct:
        return [direct]
    for candidate in [
        home / "miniconda3" / "envs" / "mdw" / "bin" / "pymol",
        home / "mambaforge" / "envs" / "mdw" / "bin" / "pymol",
    ]:
        if candidate.exists():
            return [str(candidate)]
    return None


def _pymol_script(receptor_asset: Path, ligand_asset: Path, out_png: Path, best_affinity: float | None) -> str:
    label = ""
    if best_affinity is not None and math.isfinite(float(best_affinity)):
        label = f"Best Vina affinity: {float(best_affinity):.2f} kcal/mol"
    label_commands = ""
    if label:
        label_commands = f"""
pseudoatom docking_label, pos=[0, 0, 0], label={_pymol_quote(label)}
hide everything, docking_label
"""
    return f"""
reinitialize
set retain_order, 1
set bg_rgb, [1, 1, 1]
set ray_opaque_background, on
set orthoscopic, on
set antialias, 2
set ray_trace_mode, 1
set ambient, 0.45
set direct, 0.55
set specular, 0.25
set shininess, 18
set ray_shadows, off
load {receptor_asset.name}, receptor
load {ligand_asset.name}, ligand
remove solvent
remove hydro
hide everything
show cartoon, receptor
color gray85, receptor
set cartoon_fancy_helices, on
set cartoon_smooth_loops, on
select pocket, byres (receptor within 4.0 of ligand)
show sticks, pocket
color marine, pocket
show sticks, ligand
color tv_magenta, ligand
util.cnc ligand
set stick_radius, 0.18, ligand
set stick_radius, 0.10, pocket
distance polar_contacts, ligand, pocket, 3.5, mode=2
hide labels, polar_contacts
set dash_color, yellow, polar_contacts
set dash_width, 2.0, polar_contacts
set dash_gap, 0.35, polar_contacts
set valence, on
set cartoon_transparency, 0.08, receptor
orient ligand or pocket
zoom ligand or pocket, 8
{label_commands}
ray 2400, 1800
png {out_png.name}, dpi=300
quit
"""


def _pymol_quote(value: str) -> str:
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _write_docx_report(
    out_path: Path,
    figures_root: Path,
    data_root: Path,
    items: list[FigureReportItem],
    workflow_cfg: WorkflowConfig | None,
    docking_payload: DockingReportPayload,
) -> None:
    image_rels: dict[Path, str] = {}
    media_files: list[tuple[str, Path]] = []
    report_images = [item.image_path for item in items]
    if docking_payload.figure_path is not None and docking_payload.figure_path.exists():
        report_images.insert(0, docking_payload.figure_path)
    seen_images: set[Path] = set()
    unique_images: list[Path] = []
    for image_path in report_images:
        resolved = image_path.resolve()
        if resolved not in seen_images:
            seen_images.add(resolved)
            unique_images.append(image_path)
    for idx, image_path in enumerate(unique_images, start=1):
        rel_id = f"rIdImage{idx}"
        image_rels[image_path.resolve()] = rel_id
        media_files.append((f"word/media/image{idx}{image_path.suffix.lower()}", image_path))

    document_xml = _document_xml(figures_root, data_root, items, image_rels, workflow_cfg, docking_payload)
    rels_xml = _document_relationships_xml(media_files)
    content_types_xml = _content_types_xml(media_files)

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        _writestr(archive, "[Content_Types].xml", content_types_xml)
        _writestr(archive, "_rels/.rels", _package_relationships_xml())
        _writestr(archive, "word/document.xml", document_xml)
        _writestr(archive, "word/styles.xml", _styles_xml())
        _writestr(archive, "word/_rels/document.xml.rels", rels_xml)
        for archive_name, source_path in media_files:
            _write_file(archive, archive_name, source_path)


def _document_xml(
    figures_root: Path,
    data_root: Path,
    items: list[FigureReportItem],
    image_rels: dict[Path, str],
    workflow_cfg: WorkflowConfig | None,
    docking_payload: DockingReportPayload,
) -> str:
    body: list[str] = []
    body.append(_paragraph("分子动力学模拟与结合模式分析报告", style="Title"))
    body.extend(_method_section_xml(workflow_cfg))
    body.extend(_docking_section_xml(docking_payload, image_rels))

    grouped = _group_items(items)
    section_no = 2
    image_no = 1 if docking_payload.figure_path is not None and docking_payload.figure_path.exists() else 0
    for group, group_items in grouped:
        section_no += 1
        title = FIGURE_GROUP_TITLES.get(group, _humanize_stem(group))
        body.append(_paragraph(f"{section_no}. {title}（共 {len(group_items)} 图）", style="Heading1"))
        included = "、".join(item.heading_label for item in group_items)
        body.append(_paragraph(f"本节图件包括: {included}。"))
        for idx, item in enumerate(group_items, start=1):
            image_no += 1
            body.append(_paragraph(f"{section_no}.{idx} {item.heading_label}", style="Heading2"))
            body.append(_paragraph(f"图件说明: {item.explanation}"))
            body.append(_paragraph(f"主要结果: {_summarize_figure_results(item)}"))
            source_paths = _source_path_summary(item.data_paths, figures_root.parent)
            if source_paths:
                body.append(_paragraph(f"复现数据: {source_paths}"))
            rel_id = image_rels[item.image_path.resolve()]
            width, height = _image_extent_emu(item.image_path)
            body.append(_image_paragraph(rel_id, image_no, item.display_name, width, height))
            if item.caption_example:
                body.append(_paragraph(f"图注: {item.caption_example}", style="Caption"))

    body.append(_section_properties())
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
        'xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        "<w:body>"
        + "".join(body)
        + "</w:body></w:document>"
    )


def _method_section_xml(workflow_cfg: WorkflowConfig | None) -> list[str]:
    body = [_paragraph("1. 方法", style="Heading1")]
    if workflow_cfg is None:
        body.append(_paragraph("未提供项目配置文件；因此本节仅保留方法章节占位，无法自动写入本项目的实际模拟步数、温度、压力、搜索盒和保存间隔等参数。"))
        return body
    body.extend(_paragraph(text) for text in _method_paragraphs(workflow_cfg))
    body.append(_paragraph("参考文献:"))
    body.extend(_paragraph(ref) for ref in METHOD_REFERENCES)
    return body


def _method_paragraphs(cfg: WorkflowConfig) -> list[str]:
    prep = cfg.prep
    docking = cfg.docking
    run = cfg.run
    heterogen_text = "移除了杂原子及水分子"
    if prep.remove_heterogens_keep_water:
        heterogen_text = "移除了非水杂原子并保留水分子"
    nonstandard_text = "自动替换了非标准残基" if prep.replace_nonstandard_residues else "未自动替换非标准残基"
    production_ns = float(run.production_steps) * float(run.timestep_ps) / 1000.0
    equil_ns = float(run.equil_steps) * float(run.timestep_ps) / 1000.0
    dcd_ps = float(run.dcd_interval) * float(run.timestep_ps)
    return [
        (
            f"在系统准备阶段，受体结构在 pH {prep.ph:g} 条件下进行预处理，{nonstandard_text}并{heterogen_text}；"
            "针对受体蛋白的结合位点预测，本研究对照采用 P2Rank (Krivák & Hoksza, 2018) "
            "和 ProtCross (Zhong & Jiang, 2026) 算法以辅助界定靶区。随后，配体的参数化以及受体-配体结构向 "
            "PDBQT 格式的转换均使用 Meeko (Santos-Martins et al., 2025) 工具集完成。"
        ),
        (
            "分子对接过程采用 AutoDock Vina (Eberhardt et al., 2021; Trott & Olson, 2010) 引擎执行，"
            f"基于预测位点设定的对接搜索空间边缘填充范围为 {docking.search_padding_angstrom:g} Å，并保证最小盒子尺寸不低于 "
            f"{docking.search_min_size_angstrom:g} Å；对接的全局搜索穷举度设为 {docking.vina_exhaustiveness}，"
            f"最大输出结合模式数为 {docking.vina_num_modes}，能量极差为 {docking.vina_energy_range:g} kcal/mol，"
            f"随机种子为 {docking.vina_seed}，以确保结果可重复。提取打分最优的结合构象作为初始坐标后，"
            "后续分子动力学模拟使用 OpenMM (Eastman et al., 2024) 引擎开展。"
        ),
        (
            "在构建 MD 模拟体系时，受体蛋白采用 AMBER ff14SB 力场进行拓扑描述，配体小分子采用 GAFF v2.11 力场进行参数化，"
            "并通过 OpenFF 工具包及 AM1-BCC 方法为其分配部分电荷。复合物系统随后被置于缓冲边缘为 "
            f"{run.solvent_padding_nm:g} nm 的 TIP3P 水分子溶剂盒子中，并添加对应离子（离子参数采用 amber/tip3p_HFE_multivalent）"
            f"使体系等效离子强度达到 {run.ionic_strength_molar:g} M。"
        ),
        (
            f"MD 模拟阶段共设置 {run.n_replicas} 个独立平行副本，全程启用"
            f"{'混合精度加速' if run.use_mixed_precision else '标准精度设置'}，以 {run.timestep_ps:g} ps（{run.timestep_ps * 1000:g} fs）"
            f"时间步长进行牛顿运动方程积分。模拟环境的温度与压力分别维持在 {run.temperature_kelvin:g} K 和 {run.pressure_bar:g} bar，"
            f"控温器碰撞摩擦系数设定为 {run.friction_per_ps:g} ps⁻¹。所有副本均率先经历 {_fmt_int(run.equil_steps)} 步"
            f"（{equil_ns:g} ns）的系统平衡期，随后开展 {_fmt_int(run.production_steps)} 步（{production_ns:g} ns）的生产期动力学模拟，"
            f"并每隔 {_fmt_int(run.dcd_interval)} 步（{dcd_ps:g} ps）保存一次三维坐标轨迹。"
        ),
    ]


def _docking_section_xml(docking_payload: DockingReportPayload, image_rels: dict[Path, str]) -> list[str]:
    body = [_paragraph("2. 对接结果", style="Heading1")]
    body.append(_paragraph(docking_payload.summary_text))
    if docking_payload.figure_status:
        body.append(_paragraph(docking_payload.figure_status))
    if docking_payload.pml_path is not None and docking_payload.pml_path.exists():
        body.append(_paragraph(f"图像复现脚本: {_relative_or_absolute(docking_payload.pml_path, docking_payload.pml_path.parents[1])}"))
    if docking_payload.figure_path is not None and docking_payload.figure_path.exists():
        rel_id = image_rels[docking_payload.figure_path.resolve()]
        width, height = _image_extent_emu(docking_payload.figure_path)
        body.append(_image_paragraph(rel_id, 1, "PyMOL 对接结果卡通展示", width, height))
        body.append(_paragraph("图注: 受体蛋白以卡通形式显示，配体以棒状模型显示，结合口袋邻近残基以棒状形式突出显示，黄色虚线表示几何标准识别的极性接触。", style="Caption"))
    return body


def _group_items(items: list[FigureReportItem]) -> list[tuple[str, list[FigureReportItem]]]:
    groups: dict[str, list[FigureReportItem]] = {}
    for item in items:
        groups.setdefault(item.group, []).append(item)

    def sort_key(group: str) -> tuple[int, str]:
        try:
            return FIGURE_GROUP_ORDER.index(group), group
        except ValueError:
            return len(FIGURE_GROUP_ORDER), group

    return [(group, groups[group]) for group in sorted(groups, key=sort_key)]


def _summarize_figure_results(item: FigureReportItem) -> str:
    stem = item.image_path.stem
    csv_paths = [path for path in item.data_paths if path.suffix.lower() == ".csv"]
    json_paths = [path for path in item.data_paths if path.suffix.lower() == ".json"]

    specific = _specific_summary(stem, csv_paths, json_paths)
    if specific:
        return specific

    for path in csv_paths:
        rows = _read_csv(path)
        if not rows:
            continue
        summary = _generic_csv_summary(path, rows)
        if summary:
            return summary
    for path in json_paths:
        summary = _json_summary(path)
        if summary:
            return summary
    return "该图不对应单一结构化数值指标；请结合图中标尺、图例和复现数据解读。"


def _specific_summary(stem: str, csv_paths: list[Path], json_paths: list[Path]) -> str:
    if stem in {"rmsd_replot_protein", "rmsd_replot_ligand"}:
        path = _find_named_path(csv_paths, "rmsd_combined.csv")
        rows = _read_csv(path) if path else []
        column = "protein_backbone_rmsd_mean_A" if stem == "rmsd_replot_protein" else "ligand_heavy_rmsd_mean_A"
        return _time_series_summary(rows, [column]) if rows else ""

    time_series_map = {
        "min_distance_combined": ("min_distance_combined.csv", ["mean_min_distance_A"]),
        "radius_of_gyration_combined": ("radius_of_gyration_combined.csv", ["mean_rg_A"]),
        "contact_count_combined": ("contact_count_combined.csv", ["mean_contact_count"]),
        "hbond_count_combined": ("hbond_count_combined.csv", ["mean_hbond_count"]),
        "salt_bridge_count_combined": ("salt_bridge_count_combined.csv", ["mean_salt_bridge_count"]),
        "waterbridge_count_combined": ("waterbridge_count_combined.csv", ["mean_count"]),
        "ligand_com_distance_combined": ("ligand_com_distance_combined.csv", ["mean_com_distance_A"]),
        "ligand_orientation_angle_combined": ("ligand_orientation_angle_combined.csv", ["mean_orientation_angle_deg"]),
        "sasa_complex_protein_combined": ("sasa_components_combined.csv", ["complex_sasa_mean_A2", "protein_sasa_mean_A2"]),
        "ligand_sasa_combined": ("sasa_components_combined.csv", ["ligand_sasa_mean_A2"]),
    }
    if stem in time_series_map:
        name, columns = time_series_map[stem]
        path = _find_named_path(csv_paths, name)
        rows = _read_csv(path) if path else []
        return _time_series_summary(rows, columns) if rows else ""

    occupancy_map = {
        "contact_occupancy_top20": ("contact_occupancy_distance_summary.csv", "contact_occupancy_mean", "contact_occupancy_sd"),
        "hbond_residue_occupancy_top20": ("hbond_residue_occupancy_combined.csv", "hbond_occupancy_mean", "hbond_occupancy_sd"),
        "salt_bridge_residue_occupancy": ("salt_bridge_residue_occupancy_combined.csv", "salt_bridge_occupancy_mean", "salt_bridge_occupancy_sd"),
        "waterbridge_residue_occupancy_top20": ("waterbridge_residue_occupancy_combined.csv", "waterbridge_occupancy_mean", "waterbridge_occupancy_sd"),
    }
    if stem in occupancy_map:
        name, mean_col, sd_col = occupancy_map[stem]
        path = _find_named_path(csv_paths, name)
        rows = _read_csv(path) if path else []
        return _occupancy_summary(rows, mean_col, sd_col) if rows else ""

    if stem.startswith("key_contact_distance_"):
        return _key_contact_summary(stem, csv_paths)

    if stem == "rmsf_ca_combined":
        path = _find_named_path(csv_paths, "rmsf_ca_combined.csv")
        rows = _read_csv(path) if path else []
        return _top_rows_summary(rows, "protein_residue", "rmsf_mean_A", "rmsf_sd_A", top_n=5) if rows else ""

    if stem == "dssp_residue_occupancy_combined":
        path = _find_named_path(csv_paths, "dssp_residue_occupancy_combined.csv")
        rows = _read_csv(path) if path else []
        return _dssp_occupancy_summary(rows) if rows else ""

    if stem == "dssp_fractions_combined":
        return _dssp_fraction_summary(csv_paths)

    if stem == "explained_variance_ratio":
        path = _find_named_path(csv_paths, "explained_variance_ratio.csv")
        rows = _read_csv(path) if path else []
        return _explained_variance_summary(rows) if rows else ""

    if stem == "singular_values":
        path = _find_named_path(csv_paths, "singular_values.csv")
        rows = _read_csv(path) if path else []
        return _top_rows_summary(rows, "component", "singular_value", None, top_n=5, item_prefix="成分") if rows else ""

    if stem in {"free_energy_landscape_pc1_pc2", "free_energy_landscape_pc1_pc2_3d"}:
        path = _find_named_path(json_paths, "free_energy_landscape_pc1_pc2.json")
        return _free_energy_json_summary(path) if path else ""

    if stem in {"free_energy_landscape_tic1_tic2", "free_energy_landscape_tic1_tic2_3d"}:
        path = _find_named_path(json_paths, "free_energy_landscape_tic1_tic2.json")
        return _free_energy_json_summary(path) if path else ""

    if stem in {"pc1_pc2_scatter", "tic1_tic2_scatter"}:
        axis_cols = ("PC1", "PC2") if stem == "pc1_pc2_scatter" else ("tIC1", "tIC2")
        return _projection_assignment_summary(csv_paths, axis_cols)

    if stem == "cluster_population_overall":
        path = _find_named_path(csv_paths, "cluster_population_overall.csv")
        rows = _read_csv(path) if path else []
        return _cluster_population_summary(rows) if rows else ""

    if stem == "state_population_by_replica":
        path = _find_named_path(csv_paths, "cluster_population_per_replica.csv")
        rows = _read_csv(path) if path else []
        return _state_population_by_replica_summary(rows) if rows else ""

    if stem == "representative_state_snapshots":
        path = _find_named_path(csv_paths, "representative_frames.csv")
        rows = _read_csv(path) if path else []
        return _representative_frames_summary(rows) if rows else ""

    if stem == "stationary_distribution":
        path = _find_named_path(csv_paths, "stationary_distribution.csv")
        rows = _read_csv(path) if path else []
        return _top_rows_summary(rows, "msm_state", "stationary_probability", None, top_n=5, item_prefix="状态") if rows else ""

    if stem in {"transition_matrix_heatmap", "transition_residence_departure"}:
        path = _find_named_path(csv_paths, "transition_matrix.csv")
        rows = _read_csv(path) if path else []
        return _transition_matrix_summary(rows) if rows else ""

    if stem in {"transition_dominant_exchange", "state_network"}:
        flux_path = _find_named_path(csv_paths, "equilibrium_transition_flux.csv")
        rows = _read_csv(flux_path) if flux_path else []
        return _matrix_offdiag_summary(rows, label="平衡转移通量") if rows else ""

    if stem == "implied_timescales_single_lag":
        path = _find_named_path(csv_paths, "implied_timescales_single_lag.csv")
        rows = _read_csv(path) if path else []
        return _top_rows_summary(rows, "index", "timescale_frames", None, top_n=4, item_prefix="过程") if rows else ""

    if stem == "implied_timescales_lag_scan":
        path = _find_named_path(csv_paths, "implied_timescales_lag_scan.csv")
        rows = _read_csv(path) if path else []
        return _lag_scan_timescale_summary(rows) if rows else ""

    if stem in {"lag_scan_usable_segments", "lag_scan_frame_support"}:
        path = _find_named_path(csv_paths, "lag_scan_diagnostics.csv")
        rows = _read_csv(path) if path else []
        return _lag_scan_diagnostics_summary(rows) if rows else ""

    if stem == "chapman_kolmogorov_test":
        path = _find_named_path(csv_paths, "chapman_kolmogorov_test.csv")
        rows = _read_csv(path) if path else []
        return _ck_summary(rows) if rows else ""

    return ""


def _read_csv(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def _read_json(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _find_named_path(paths: list[Path], name: str) -> Path | None:
    for path in paths:
        if path.name == name:
            return path
    return None


def _generic_csv_summary(path: Path, rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    if "time_ns" in rows[0]:
        numeric_cols = [col for col in rows[0] if col != "time_ns" and _all_numeric(rows, col)]
        preferred = [col for col in numeric_cols if col.startswith("mean_") or col.endswith("_mean_A2")]
        return _time_series_summary(rows, preferred[:3] or numeric_cols[:2])
    if "protein_residue" in rows[0]:
        for col in rows[0]:
            if col.endswith("_occupancy_mean"):
                sd_col = col.replace("_mean", "_sd")
                return _occupancy_summary(rows, col, sd_col if sd_col in rows[0] else None)
        if "rmsf_mean_A" in rows[0]:
            return _top_rows_summary(rows, "protein_residue", "rmsf_mean_A", "rmsf_sd_A", top_n=5)
    if "cluster" in rows[0] and "fraction" in rows[0]:
        return _cluster_population_summary(rows)
    if "stationary_probability" in rows[0]:
        return _top_rows_summary(rows, "msm_state", "stationary_probability", None, top_n=5, item_prefix="状态")

    numeric_cols = [col for col in rows[0] if _all_numeric(rows, col)]
    if not numeric_cols:
        return f"{path.name} 包含 {len(rows)} 行记录。"
    pieces = []
    for col in numeric_cols[:3]:
        values = _numeric_values(rows, col)
        if values:
            pieces.append(f"{_display_col(col)}均值 {_fmt_value(sum(values) / len(values), col)}，范围 {_fmt_value(min(values), col)}-{_fmt_value(max(values), col)}")
    return "；".join(pieces) + "。" if pieces else ""


def _time_series_summary(rows: list[dict[str, str]], columns: list[str]) -> str:
    if not rows:
        return ""
    time_values = _numeric_values(rows, "time_ns")
    time_text = ""
    if time_values:
        start_time = _fmt_value(min(time_values), "time_ns")
        end_time = _fmt_value(max(time_values), "time_ns")
        time_text = f"时间范围 {start_time}-{end_time}；"
    pieces = []
    for column in columns:
        if column not in rows[0]:
            continue
        values = _numeric_values(rows, column)
        if not values:
            continue
        mean = sum(values) / len(values)
        sd = _sample_sd(values)
        final = values[-1]
        pieces.append(
            f"{_display_col(column)}均值 {_fmt_value(mean, column)}"
            f"（时间序列 SD {_fmt_value(sd, column)}，范围 {_fmt_value(min(values), column)}-{_fmt_value(max(values), column)}，末帧 {_fmt_value(final, column)}）"
        )
    return time_text + "；".join(pieces) + "。" if pieces else ""


def _occupancy_summary(rows: list[dict[str, str]], mean_col: str, sd_col: str | None) -> str:
    sorted_rows = sorted(rows, key=lambda row: _to_float(row.get(mean_col)), reverse=True)
    pieces = []
    for row in sorted_rows[:5]:
        label = row.get("protein_residue") or row.get("residue") or row.get("label") or ""
        mean = _to_float(row.get(mean_col))
        if not math.isfinite(mean):
            continue
        sd_text = ""
        if sd_col and sd_col in row:
            sd = _to_float(row.get(sd_col))
            if math.isfinite(sd):
                sd_text = f" ± {_fmt_value(sd, mean_col)}"
        distance_text = ""
        if "min_distance_mean_A" in row:
            distance = _to_float(row.get("min_distance_mean_A"))
            if math.isfinite(distance):
                distance_text = f"，平均最小距离 {_fmt_value(distance, 'min_distance_mean_A')}"
        pieces.append(f"{_compact_label(label)} {_fmt_value(mean, mean_col)}{sd_text}{distance_text}")
    return "最高占据位点: " + "；".join(pieces) + "。" if pieces else ""


def _key_contact_summary(stem: str, csv_paths: list[Path]) -> str:
    trace_path = _find_named_path(csv_paths, "key_contact_distance_traces.csv")
    rows = _read_csv(trace_path) if trace_path else []
    residue = _residue_from_key_contact_stem(stem)
    if residue and rows:
        rows = [row for row in rows if _residue_matches(row.get("protein_residue", ""), residue)]
    summary = _time_series_summary(rows, ["mean_distance_A"]) if rows else ""
    distance_path = _find_named_path(csv_paths, "contact_occupancy_distance_summary.csv")
    distance_rows = _read_csv(distance_path) if distance_path else []
    if residue and distance_rows:
        matched = [row for row in distance_rows if _residue_matches(row.get("protein_residue", ""), residue)]
        if matched:
            row = matched[0]
            occupancy = _to_float(row.get("contact_occupancy_mean"))
            min_distance = _to_float(row.get("min_distance_min_A"))
            extra = []
            if math.isfinite(occupancy):
                extra.append(f"接触占据 {_fmt_value(occupancy, 'contact_occupancy_mean')}")
            if math.isfinite(min_distance):
                extra.append(f"最短观测距离 {_fmt_value(min_distance, 'min_distance_min_A')}")
            if extra:
                summary = (summary + " " if summary else "") + "；".join(extra) + "。"
    return summary


def _dssp_occupancy_summary(rows: list[dict[str, str]]) -> str:
    pieces = []
    for col, label in [
        ("helix_fraction_mean", "螺旋"),
        ("sheet_fraction_mean", "折叠"),
        ("coil_fraction_mean", "无规卷曲"),
    ]:
        valid = [(row.get("protein_residue", ""), _to_float(row.get(col))) for row in rows]
        valid = [(res, val) for res, val in valid if math.isfinite(val)]
        if valid:
            res, val = max(valid, key=lambda item: item[1])
            pieces.append(f"{label}最高为 {_compact_label(res)} {_fmt_value(val, col)}")
    return "；".join(pieces) + "。" if pieces else ""


def _dssp_fraction_summary(csv_paths: list[Path]) -> str:
    values = {"helix_fraction": [], "sheet_fraction": [], "coil_fraction": []}
    for path in csv_paths:
        if path.name != "dssp_fractions_timeseries.csv":
            continue
        for row in _read_csv(path):
            for col in values:
                value = _to_float(row.get(col))
                if math.isfinite(value):
                    values[col].append(value)
    pieces = []
    labels = {
        "helix_fraction": "螺旋",
        "sheet_fraction": "折叠",
        "coil_fraction": "无规卷曲",
    }
    for col, vals in values.items():
        if vals:
            pieces.append(f"{labels[col]}平均 {_fmt_value(sum(vals) / len(vals), col)}")
    return "跨重复二级结构组分: " + "，".join(pieces) + "。" if pieces else ""


def _explained_variance_summary(rows: list[dict[str, str]]) -> str:
    pieces = []
    for row in rows[:5]:
        comp = row.get("component", "")
        val = _to_float(row.get("explained_variance_ratio"))
        if math.isfinite(val):
            pieces.append(f"PC{comp} {_fmt_value(val, 'explained_variance_ratio')}")
    cumulative = _to_float(rows[-1].get("cumulative_explained_variance_ratio")) if rows else math.nan
    tail = f"；前 {len(rows)} 个主成分累计 {_fmt_value(cumulative, 'cumulative_explained_variance_ratio')}" if math.isfinite(cumulative) else ""
    return "解释方差: " + "，".join(pieces) + tail + "。"


def _free_energy_json_summary(path: Path) -> str:
    data = _read_json(path)
    if not isinstance(data, dict):
        return ""
    pieces = []
    n_samples = data.get("n_input_samples")
    cap = data.get("display_energy_cap_kcal_mol")
    if isinstance(n_samples, (int, float)):
        pieces.append(f"输入样本数 {int(n_samples)}")
    if isinstance(cap, (int, float)):
        pieces.append(f"显示自由能上限 {_fmt_value(float(cap), 'energy_kcal_mol')}")
    basins = data.get("basin_markers")
    if isinstance(basins, list) and basins:
        basin_text = []
        for basin in basins[:5]:
            if not isinstance(basin, dict):
                continue
            label = str(basin.get("label", "")).strip()
            energy = basin.get("relative_free_energy_kcal_mol")
            if isinstance(energy, (int, float)):
                basin_text.append(f"{label} {_fmt_value(float(energy), 'energy_kcal_mol')}")
        if basin_text:
            pieces.append("低能盆地 " + "、".join(basin_text))
    return "；".join(pieces) + "。" if pieces else ""


def _projection_assignment_summary(csv_paths: list[Path], axis_cols: tuple[str, str]) -> str:
    total = 0
    ranges = {axis_cols[0]: [], axis_cols[1]: []}
    replicas = 0
    for path in csv_paths:
        if not path.name.startswith("replica_") or not path.name.endswith(("_pc.csv", "_tic.csv")):
            continue
        rows = _read_csv(path)
        if not rows or axis_cols[0] not in rows[0] or axis_cols[1] not in rows[0]:
            continue
        replicas += 1
        total += len(rows)
        for col in axis_cols:
            ranges[col].extend(_numeric_values(rows, col))
    pieces = [f"包含 {replicas} 个重复、{total} 帧投影"]
    for col in axis_cols:
        vals = ranges[col]
        if vals:
            pieces.append(f"{col} 范围 {_fmt_value(min(vals), col)}-{_fmt_value(max(vals), col)}")
    return "；".join(pieces) + "。"


def _cluster_population_summary(rows: list[dict[str, str]]) -> str:
    sorted_rows = sorted(rows, key=lambda row: _to_float(row.get("fraction")), reverse=True)
    pieces = []
    for row in sorted_rows[:5]:
        cluster = row.get("cluster", "")
        fraction = _to_float(row.get("fraction"))
        n_frames = _to_float(row.get("n_frames"))
        frame_text = f"，{int(n_frames)} 帧" if math.isfinite(n_frames) else ""
        if math.isfinite(fraction):
            pieces.append(f"状态 {cluster} {_fmt_value(fraction, 'fraction')}{frame_text}")
    return "主要构象状态占比: " + "；".join(pieces) + "。" if pieces else ""


def _state_population_by_replica_summary(rows: list[dict[str, str]]) -> str:
    by_replica: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_replica.setdefault(row.get("replica", ""), []).append(row)
    pieces = []
    for replica in sorted(by_replica)[:6]:
        best = max(by_replica[replica], key=lambda row: _to_float(row.get("fraction")))
        pieces.append(f"{replica} 以状态 {best.get('cluster', '')} 为主（{_fmt_value(_to_float(best.get('fraction')), 'fraction')}）")
    return "各重复主导状态: " + "；".join(pieces) + "。" if pieces else ""


def _representative_frames_summary(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    selected = [row for row in rows if str(row.get("selected_for_figure", "")).strip().lower() == "true"]
    source = selected or rows
    pieces = []
    for row in source[:8]:
        pieces.append(
            f"状态 {row.get('cluster', '')}/rank {row.get('rank_by_population', '')}"
            f"（{row.get('replica', '')}, frame {row.get('frame', '')}, 占比 {_fmt_value(_to_float(row.get('fraction')), 'fraction')}）"
        )
    return f"代表性帧共 {len(source)} 个: " + "；".join(pieces) + "。"


def _transition_matrix_summary(rows: list[dict[str, str]]) -> str:
    diagonal = []
    offdiag = []
    for row in rows:
        from_state = row.get("from\\to") or row.get("from/to") or row.get("")
        for to_state, raw in row.items():
            if to_state in {"from\\to", "from/to", ""}:
                continue
            value = _to_float(raw)
            if not math.isfinite(value):
                continue
            if to_state == from_state:
                diagonal.append((from_state, value))
            else:
                offdiag.append((from_state, to_state, value))
    diag_text = ""
    if diagonal:
        strongest_diag = sorted(diagonal, key=lambda item: item[1], reverse=True)[:3]
        diag_text = "自保持最高 " + "、".join(f"{state} {_fmt_value(val, 'probability')}" for state, val in strongest_diag)
    off_text = ""
    if offdiag:
        strongest_off = sorted(offdiag, key=lambda item: item[2], reverse=True)[:3]
        off_text = "主要非自转移 " + "、".join(f"{src}->{dst} {_fmt_value(val, 'probability')}" for src, dst, val in strongest_off)
    return "；".join(part for part in [diag_text, off_text] if part) + "。"


def _matrix_offdiag_summary(rows: list[dict[str, str]], *, label: str) -> str:
    values = []
    for row in rows:
        from_state = row.get("from\\to") or row.get("from/to") or row.get("")
        for to_state, raw in row.items():
            if to_state in {"from\\to", "from/to", ""} or to_state == from_state:
                continue
            value = _to_float(raw)
            if math.isfinite(value) and value > 0:
                values.append((from_state, to_state, value))
    values.sort(key=lambda item: item[2], reverse=True)
    pieces = [f"{src}->{dst} {_fmt_value(val, 'probability')}" for src, dst, val in values[:5]]
    return f"最高{label}: " + "；".join(pieces) + "。" if pieces else ""


def _lag_scan_timescale_summary(rows: list[dict[str, str]]) -> str:
    by_process: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        process = row.get("process_index", "")
        lag = _to_float(row.get("lag_frames"))
        timescale = _to_float(row.get("timescale_frames"))
        if math.isfinite(lag) and math.isfinite(timescale):
            by_process.setdefault(process, []).append((lag, timescale))
    pieces = []
    for process in sorted(by_process, key=lambda x: int(x) if str(x).isdigit() else 999)[:4]:
        vals = sorted(by_process[process])
        pieces.append(
            f"过程 {process} 在 lag {int(vals[0][0])}-{int(vals[-1][0])} 帧范围内为 "
            f"{_fmt_value(min(v for _, v in vals), 'timescale_frames')}-{_fmt_value(max(v for _, v in vals), 'timescale_frames')}"
        )
    return "；".join(pieces) + "。" if pieces else ""


def _lag_scan_diagnostics_summary(rows: list[dict[str, str]]) -> str:
    lags = _numeric_values(rows, "lag_frames")
    segments = _numeric_values(rows, "usable_segments")
    frames = _numeric_values(rows, "usable_frames")
    active = _numeric_values(rows, "active_frame_count")
    pieces = []
    if lags:
        pieces.append(f"lag 范围 {int(min(lags))}-{int(max(lags))} 帧")
    if segments:
        pieces.append(f"可用片段数范围 {int(min(segments))}-{int(max(segments))}")
    if frames and active and max(active) > 0:
        ratios = [frame / act for frame, act in zip(frames, active) if act > 0]
        if ratios:
            pieces.append(f"可用帧比例 {_fmt_value(min(ratios), 'fraction')}-{_fmt_value(max(ratios), 'fraction')}")
    return "；".join(pieces) + "。" if pieces else ""


def _ck_summary(rows: list[dict[str, str]]) -> str:
    errors = _numeric_values(rows, "absolute_error")
    if not errors:
        return ""
    return f"CK 验证绝对误差平均 {_fmt_number(sum(errors) / len(errors))}，最大 {_fmt_number(max(errors))}。"


def _top_rows_summary(
    rows: list[dict[str, str]],
    label_col: str,
    value_col: str,
    sd_col: str | None,
    *,
    top_n: int,
    item_prefix: str = "",
) -> str:
    sorted_rows = sorted(rows, key=lambda row: _to_float(row.get(value_col)), reverse=True)
    pieces = []
    for row in sorted_rows[:top_n]:
        value = _to_float(row.get(value_col))
        if not math.isfinite(value):
            continue
        label = str(row.get(label_col, "")).strip()
        name = f"{item_prefix} {label}".strip() if item_prefix else _compact_label(label)
        sd_text = ""
        if sd_col and sd_col in row:
            sd = _to_float(row.get(sd_col))
            if math.isfinite(sd):
                sd_text = f" ± {_fmt_value(sd, value_col)}"
        pieces.append(f"{name} {_fmt_value(value, value_col)}{sd_text}")
    return f"{_display_col(value_col)}最高项: " + "；".join(pieces) + "。" if pieces else ""


def _json_summary(path: Path) -> str:
    data = _read_json(path)
    if isinstance(data, dict):
        pieces = []
        for key, value in data.items():
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                pieces.append(f"{_display_col(key)} {_fmt_value(float(value), key)}")
            if len(pieces) >= 4:
                break
        return "；".join(pieces) + "。" if pieces else ""
    if isinstance(data, list):
        return f"{path.name} 包含 {len(data)} 条 JSON 记录。"
    return ""


def _source_path_summary(paths: list[Path], bundle_root: Path) -> str:
    filtered = [path for path in paths if path.suffix.lower() in {".csv", ".json"}]
    if not filtered:
        return ""
    labels = [_relative_or_absolute(path, bundle_root) for path in filtered[:5]]
    suffix = f" 等 {len(filtered)} 个文件" if len(filtered) > 5 else ""
    return "、".join(labels) + suffix


def _relative_or_absolute(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return str(path.resolve())


def _all_numeric(rows: list[dict[str, str]], col: str) -> bool:
    return bool(_numeric_values(rows, col))


def _numeric_values(rows: list[dict[str, str]], col: str) -> list[float]:
    values = []
    for row in rows:
        value = _to_float(row.get(col))
        if math.isfinite(value):
            values.append(value)
    return values


def _to_float(value) -> float:
    try:
        if value is None or value == "":
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def _sample_sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def _fmt_value(value: float, column: str) -> str:
    if not math.isfinite(value):
        return "NA"
    lower = column.lower()
    if any(token in lower for token in ("fraction", "probability", "occupancy", "ratio")) and abs(value) <= 1.5:
        return f"{value * 100:.1f}%"
    if "time_ns" in lower:
        return f"{value:.2f} ns"
    if "energy" in lower or "kcal" in lower:
        return f"{value:.2f} kcal/mol"
    if "a2" in lower or "surface" in lower or "sasa" in lower:
        return f"{value:.1f} Å²"
    if lower.endswith("_a") or "_a_" in lower or "distance" in lower or "rmsd" in lower or "rmsf" in lower or "_rg_" in lower or lower.endswith("_rg_a"):
        return f"{value:.2f} Å"
    if "deg" in lower or "angle" in lower:
        return f"{value:.1f}°"
    if "timescale_frames" in lower or lower.endswith("_frames"):
        return f"{value:.1f} 帧"
    return _fmt_number(value)


def _fmt_number(value: float) -> str:
    if not math.isfinite(value):
        return "NA"
    abs_value = abs(value)
    if abs_value >= 1000:
        return f"{value:.1f}"
    if abs_value >= 10:
        return f"{value:.2f}"
    if abs_value >= 0.01:
        return f"{value:.3f}"
    if abs_value == 0:
        return "0"
    return f"{value:.3g}"


def _fmt_int(value: int | float) -> str:
    return f"{int(value):,}"


def _display_col(column: str) -> str:
    if column in COMMON_NAME_MAP:
        return COMMON_NAME_MAP[column]
    text = column.replace("_mean", "").replace("_", " ")
    return text


def _compact_label(label: str) -> str:
    text = str(label).strip()
    if text.startswith("chain"):
        text = re.sub(r"^chain\d+_", "", text)
    return text or "NA"


def _residue_from_key_contact_stem(stem: str) -> str:
    match = re.search(r"_([a-z]{3}\d+)$", stem.lower())
    return match.group(1).upper() if match else ""


def _residue_matches(label: str, token: str) -> bool:
    clean = re.sub(r"[^A-Za-z0-9]", "", label).upper()
    return clean.endswith(token.upper())


def _humanize_stem(stem: str) -> str:
    return stem.replace("_", " ").strip() or stem


def _paragraph(text: str, *, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f"<w:p>{ppr}<w:r><w:t>{escape(text)}</w:t></w:r></w:p>"


def _image_paragraph(rel_id: str, image_no: int, title: str, cx: int, cy: int) -> str:
    safe_title = escape(title)
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:effectExtent l="0" t="0" r="0" b="0"/>
        <wp:docPr id="{image_no}" name="{safe_title}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{image_no}" name="{safe_title}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
"""


def _section_properties() -> str:
    return (
        "<w:sectPr>"
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134" w:header="708" w:footer="708" w:gutter="0"/>'
        "</w:sectPr>"
    )


def _image_extent_emu(path: Path) -> tuple[int, int]:
    width_px, height_px = _image_size(path)
    if width_px <= 0 or height_px <= 0:
        return int(6.2 * EMU_PER_INCH), int(4.0 * EMU_PER_INCH)
    natural_width = width_px / 150.0
    natural_height = height_px / 150.0
    max_width = 6.2
    max_height = 4.8
    scale = min(max_width / natural_width, max_height / natural_height, 1.0)
    display_width = max(natural_width * scale, 1.0)
    display_height = max(natural_height * scale, 0.8)
    return int(display_width * EMU_PER_INCH), int(display_height * EMU_PER_INCH)


def _image_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
        return struct.unpack(">II", data[16:24])
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    return 0, 0


def _jpeg_size(data: bytes) -> tuple[int, int]:
    idx = 2
    while idx + 9 < len(data):
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > len(data):
            break
        length = int.from_bytes(data[idx : idx + 2], "big")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            height = int.from_bytes(data[idx + 3 : idx + 5], "big")
            width = int.from_bytes(data[idx + 5 : idx + 7], "big")
            return width, height
        idx += length
    return 0, 0


def _content_types_xml(media_files: list[tuple[str, Path]]) -> str:
    defaults = {
        "rels": "application/vnd.openxmlformats-package.relationships+xml",
        "xml": "application/xml",
    }
    for _, path in media_files:
        ext = path.suffix.lower().lstrip(".")
        if ext == "jpg":
            ext = "jpeg"
        defaults[ext] = f"image/{ext}"
    default_xml = "".join(
        f'<Default Extension="{escape(ext)}" ContentType="{escape(content_type)}"/>'
        for ext, content_type in sorted(defaults.items())
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        + default_xml
        + '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        + '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        + "</Types>"
    )


def _package_relationships_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )


def _document_relationships_xml(media_files: list[tuple[str, Path]]) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for idx, (archive_name, _) in enumerate(media_files, start=1):
        target = archive_name.replace("word/", "")
        rels.append(
            f'<Relationship Id="rIdImage{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{escape(target)}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + "</Relationships>"
    )


def _styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault><w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimSun"/><w:sz w:val="22"/></w:rPr></w:rPrDefault>
    <w:pPrDefault><w:pPr><w:spacing w:after="120" w:line="276" w:lineRule="auto"/></w:pPr></w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:pPr><w:spacing w:after="300"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:b/><w:sz w:val="36"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="360" w:after="180"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:b/><w:sz w:val="30"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:pPr><w:keepNext/><w:spacing w:before="240" w:after="120"/></w:pPr>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimHei"/><w:b/><w:sz w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/><w:basedOn w:val="Normal"/>
    <w:rPr><w:rFonts w:ascii="Arial" w:hAnsi="Arial" w:eastAsia="SimSun"/><w:i/><w:sz w:val="20"/></w:rPr>
  </w:style>
</w:styles>"""


def _writestr(archive: zipfile.ZipFile, name: str, data: str) -> None:
    info = zipfile.ZipInfo(name, DOCX_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, data.encode("utf-8"))


def _write_file(archive: zipfile.ZipFile, name: str, source_path: Path) -> None:
    info = zipfile.ZipInfo(name, DOCX_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, source_path.read_bytes())
