from __future__ import annotations

from copy import deepcopy

import importlib.util
import json
import sys
from pathlib import Path

from .config import WorkflowConfig
from .core import infer_run_input_paths, normalize_workflow_paths, preflight_validate, resolve_replica_dirs
from .core.executables import find_binary


MODULE_CHECKS = {
    "base": ["numpy", "scipy", "matplotlib", "sklearn", "mdtraj", "rdkit"],
    "prep": ["pdbfixer", "meeko"],
    "md": ["openmm", "openff.toolkit", "openff.units.openmm", "openmmforcefields.generators", "parmed"],
    "advanced": ["deeptime"],
    "gui": ["tkinter"],
}

BINARY_CHECKS = {
    "docking": ["mk_prepare_receptor.py", "mk_prepare_ligand.py", "mk_export.py", "vina"],
    "mmgbsa": ["MMPBSA.py", "cpptraj"],
}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def _check_modules(cfg: WorkflowConfig) -> dict:
    wanted = set(MODULE_CHECKS["base"]) | set(MODULE_CHECKS["gui"])
    if cfg.do_prep:
        wanted |= set(MODULE_CHECKS["prep"])
    if cfg.do_run_md:
        wanted |= set(MODULE_CHECKS["md"])
    if cfg.do_advanced_analysis:
        wanted |= set(MODULE_CHECKS["advanced"])
    return {name: _has_module(name) for name in sorted(wanted)}


def _check_binaries(cfg: WorkflowConfig) -> dict:
    wanted: set[str] = set()
    if cfg.do_prep and str(cfg.docking.docking_mode).strip().lower() == "auto":
        wanted |= set(BINARY_CHECKS["docking"])
    if cfg.do_mmgbsa_postprocess and cfg.mmgbsa.auto_run:
        wanted |= set(BINARY_CHECKS["mmgbsa"])
    return {name: find_binary(name) is not None for name in sorted(wanted)}


def _replica_file_status(replica_dirs: list[Path], required_files: list[str]) -> list[dict]:
    rows = []
    for replica_dir in replica_dirs:
        missing = [name for name in required_files if not (replica_dir / name).exists()]
        empty = [
            name
            for name in required_files
            if (replica_dir / name).exists()
            and (replica_dir / name).is_file()
            and (replica_dir / name).stat().st_size <= 0
        ]
        rows.append({
            "replica": replica_dir.name,
            "path": str(replica_dir.resolve()),
            "ok": len(missing) == 0 and len(empty) == 0,
            "missing_files": missing,
            "empty_files": empty,
        })
    return rows


def run_self_check(cfg: WorkflowConfig) -> dict:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    report: dict = {
        "python": sys.version.split()[0],
        "workspace_root": cfg.workspace_root,
        "module_availability": _check_modules(cfg),
        "binary_availability": _check_binaries(cfg),
        "checks": [],
        "warnings": [],
        "errors": [],
    }

    validation = preflight_validate(cfg)
    if validation.warnings:
        report["warnings"].extend(validation.warnings)
    if validation.errors:
        report["errors"].extend(validation.errors)
        for item in validation.errors:
            report["checks"].append({"name": "preflight", "ok": False, "detail": item})

    def ok(name: str, detail):
        report["checks"].append({"name": name, "ok": True, "detail": detail})

    def fail(name: str, detail):
        report["checks"].append({"name": name, "ok": False, "detail": detail})
        report["errors"].append(f"{name}: {detail}")

    root = Path(cfg.workspace_root)
    (ok if root.exists() else fail)("workspace_root_exists", str(root.resolve()))

    if cfg.do_prep:
        p = Path(cfg.prep.receptor_input)
        (ok if p.exists() else fail)("prep_receptor_input", str(p))
        mode = str(cfg.docking.ligand_input_mode).strip().lower()
        if mode == "smiles":
            if str(cfg.docking.ligand_smiles).strip():
                ok("ligand_smiles", "SMILES provided")
            else:
                fail("ligand_smiles", "ligand_input_mode=smiles but no SMILES string was provided")
        elif mode == "sdf":
            p = Path(cfg.docking.ligand_sdf_input)
            (ok if p.exists() else fail)("ligand_sdf_input", str(p))
        elif mode == "pdb":
            p = Path(cfg.docking.ligand_pdb_input)
            (ok if p.exists() else fail)("ligand_pdb_input", str(p))
        else:
            fail("ligand_input_mode", f"Unsupported ligand_input_mode: {cfg.docking.ligand_input_mode}")

        docking_mode = str(cfg.docking.docking_mode).strip().lower()
        if docking_mode not in {"auto", "external", "skip"}:
            fail("docking_mode", f"Unsupported docking_mode: {cfg.docking.docking_mode}")
        elif docking_mode == "external":
            p = Path(cfg.docking.external_docking_sdf)
            (ok if p.exists() else fail)("external_docking_sdf", str(p))
        if mode == "pdb" and docking_mode in {"auto", "external"}:
            fail("pdb_ligand_docking_mode", "ligand_input_mode=pdb currently requires docking_mode=skip")
        elif docking_mode == "auto" and mode != "pdb":
            for binary_name, present in report["binary_availability"].items():
                (ok if present else fail)(f"binary_{binary_name}", binary_name)
            if str(cfg.docking.search_space_mode).strip().lower() == "manual":
                centers = [cfg.docking.search_center_x, cfg.docking.search_center_y, cfg.docking.search_center_z]
                sizes = [
                    cfg.docking.search_size_x if cfg.docking.search_size_x is not None else cfg.docking.default_box_size_x,
                    cfg.docking.search_size_y if cfg.docking.search_size_y is not None else cfg.docking.default_box_size_y,
                    cfg.docking.search_size_z if cfg.docking.search_size_z is not None else cfg.docking.default_box_size_z,
                ]
                if any(v is None for v in centers):
                    fail("manual_search_center", "search_space_mode=manual but center_x/y/z is incomplete")
                elif min(float(v) for v in sizes) <= 0:
                    fail("manual_search_size", "Manual docking box sizes must be positive")
                else:
                    ok("manual_search_size", [float(v) for v in sizes])

    if cfg.do_run_md:
        protein_pdb, ligand_sdf = infer_run_input_paths(cfg)
        for label, path in [("run_protein_pdb", protein_pdb), ("run_ligand_structure", ligand_sdf)]:
            p = Path(path)
            if p.exists():
                ok(label, str(p))
            elif cfg.do_prep and str(p) in {
                str(Path(cfg.prep.receptor_output)),
                str(Path(cfg.docking.extracted_pose_sdf)),
                str(Path(cfg.docking.ligand_output_sdf)),
                str(Path(cfg.docking.ligand_output_pdb)),
            }:
                ok(label, f"Will be generated during prep: {p}")
            else:
                fail(label, str(p))
        ok("run_output_root", str(Path(cfg.run.output_root)))

    replica_roots = []
    if cfg.do_basic_analysis:
        replica_roots.append(("basic", Path(cfg.basic.replica_root)))
    if cfg.do_waterbridge_analysis:
        replica_roots.append(("waterbridge", Path(cfg.waterbridge.replica_root)))
    if cfg.do_advanced_analysis:
        replica_roots.append(("advanced", Path(cfg.advanced.replica_root)))
    unique_roots = sorted({str(p.resolve()) for _, p in replica_roots if p.exists()})
    if len(unique_roots) > 1:
        report["warnings"].append("Different analysis modules are using different replica_root values. Confirm that this is intentional.")

    if cfg.do_mmgbsa_postprocess:
        if cfg.mmgbsa.auto_run:
            report["warnings"].append("MM/GBSA is configured to auto-run after MD. Result files are not expected to exist yet before the workflow starts.")
        else:
            for label, path in [("mmgbsa_final_dat", cfg.mmgbsa.final_dat), ("mmgbsa_final_csv", cfg.mmgbsa.final_csv), ("mmgbsa_per_frame_csv", cfg.mmgbsa.per_frame_csv), ("mmgbsa_per_residue_csv", cfg.mmgbsa.per_residue_csv)]:
                if str(path).strip():
                    p = Path(path)
                    (ok if p.exists() else fail)(label, str(p))

    if not cfg.do_run_md:
        if cfg.do_basic_analysis:
            dirs = resolve_replica_dirs(cfg.basic.replica_root, cfg.basic.replica_glob)
            if dirs:
                ok("basic_replica_dirs", f"Found {len(dirs)} replica directories")
                report["basic_replica_status"] = _replica_file_status(dirs, ["system_solvated.pdb", "trajectory.dcd", "md.log"])
            else:
                fail("basic_replica_dirs", f"No {cfg.basic.replica_glob} directories found under {cfg.basic.replica_root}")
            ligand = Path(cfg.basic.ligand_sdf)
            (ok if ligand.exists() else fail)("basic_ligand_structure", str(ligand))
        if cfg.do_waterbridge_analysis:
            dirs = resolve_replica_dirs(cfg.waterbridge.replica_root, cfg.waterbridge.replica_glob)
            if dirs:
                ok("waterbridge_replica_dirs", f"Found {len(dirs)} replica directories")
                report["waterbridge_replica_status"] = _replica_file_status(dirs, [cfg.waterbridge.top_name, cfg.waterbridge.traj_name])
            else:
                fail("waterbridge_replica_dirs", f"No {cfg.waterbridge.replica_glob} directories found under {cfg.waterbridge.replica_root}")
        if cfg.do_advanced_analysis:
            dirs = resolve_replica_dirs(cfg.advanced.replica_root, cfg.advanced.replica_glob)
            if dirs:
                ok("advanced_replica_dirs", f"Found {len(dirs)} replica directories")
                report["advanced_replica_status"] = _replica_file_status(dirs, [cfg.advanced.top_name, cfg.advanced.traj_name])
            else:
                fail("advanced_replica_dirs", f"No {cfg.advanced.replica_glob} directories found under {cfg.advanced.replica_root}")

    report["summary"] = {
        "n_errors": len(report["errors"]),
        "n_warnings": len(report["warnings"]),
        "ready": len(report["errors"]) == 0,
    }
    return report


def save_self_check_report(report: dict, output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path.resolve())
