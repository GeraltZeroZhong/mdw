from __future__ import annotations

import subprocess
from pathlib import Path

from ..config import DockingConfig
from ..core import check_input_file, require_binary
from ..prep.ligand import extract_pose1
from ..prep.receptor import meeko_delete_residues_spec, sanitize_receptor_for_docking


def _run_command(args: list[str], cwd: str | None = None, log_path: str | None = None) -> None:
    proc = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as _fh:
            _fh.write(f"COMMAND:\n{' '.join(args)}\n\nSTDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}\n\n")
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({' '.join(args)}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def _has_unrecognized_argument_error(exc: RuntimeError, flag: str) -> bool:
    return f"unrecognized arguments: {flag}" in str(exc)


def _prepare_receptor_with_meeko(input_pdb: str, cfg: DockingConfig) -> tuple[str, str]:
    binary = require_binary("mk_prepare_receptor.py")
    Path(cfg.receptor_pdbqt).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg.receptor_json).parent.mkdir(parents=True, exist_ok=True)
    sanitized_pdb = str(Path(cfg.receptor_pdbqt).with_name("receptor_for_meeko.pdb"))
    sanitize_receptor_for_docking(input_pdb, sanitized_pdb)

    base_args = [
        binary,
        "--read_pdb", sanitized_pdb,
        "--write_pdbqt", cfg.receptor_pdbqt,
        "--write_json", cfg.receptor_json,
    ]
    delete_spec = meeko_delete_residues_spec(sanitized_pdb)
    if delete_spec:
        base_args.extend(["--delete_residues", delete_spec])

    # Meeko renamed this flag across releases. Try the newer spelling first,
    # then fall back to the legacy one only when the CLI rejects it.
    for bad_res_flag in ("--allow_bad_res", "--delete_bad_res"):
        try:
            _run_command([*base_args, bad_res_flag], log_path=cfg.docking_log)
            return cfg.receptor_pdbqt, cfg.receptor_json
        except RuntimeError as exc:
            if not _has_unrecognized_argument_error(exc, bad_res_flag):
                raise

    raise RuntimeError(
        "mk_prepare_receptor.py rejected both supported bad-residue flags "
        "('--allow_bad_res' and '--delete_bad_res')."
    )


def _prepare_ligand_with_meeko(input_sdf: str, output_pdbqt: str, log_path: str) -> str:
    binary = require_binary("mk_prepare_ligand.py")
    Path(output_pdbqt).parent.mkdir(parents=True, exist_ok=True)
    args = [
        binary,
        "-i", input_sdf,
        "-o", output_pdbqt,
    ]
    _run_command(args, log_path=log_path)
    return output_pdbqt


def _run_vina(receptor_pdbqt: str, ligand_pdbqt: str, cfg: DockingConfig) -> tuple[str, str]:
    binary = require_binary("vina")
    Path(cfg.docking_pdbqt).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        binary,
        "--receptor", receptor_pdbqt,
        "--ligand", ligand_pdbqt,
        "--center_x", f"{cfg.search_center_x:.3f}",
        "--center_y", f"{cfg.search_center_y:.3f}",
        "--center_z", f"{cfg.search_center_z:.3f}",
        "--size_x", f"{cfg.search_size_x:.3f}",
        "--size_y", f"{cfg.search_size_y:.3f}",
        "--size_z", f"{cfg.search_size_z:.3f}",
        "--exhaustiveness", str(cfg.vina_exhaustiveness),
        "--num_modes", str(cfg.vina_num_modes),
        "--energy_range", f"{cfg.vina_energy_range:.3f}",
        "--seed", str(cfg.vina_seed),
        "--out", cfg.docking_pdbqt,
    ]
    _run_command(cmd, log_path=cfg.docking_log)
    return cfg.docking_pdbqt, cfg.docking_log


def _export_docking_results_with_meeko(docking_pdbqt: str, output_sdf: str, log_path: str) -> str:
    binary = require_binary("mk_export.py")
    Path(output_sdf).parent.mkdir(parents=True, exist_ok=True)
    args = [
        binary,
        docking_pdbqt,
        "-s", output_sdf,
    ]
    _run_command(args, log_path=log_path)
    return output_sdf


def run_auto_docking(receptor_pdb: str, ligand_sdf: str, cfg: DockingConfig) -> dict[str, str]:
    receptor_pdb = str(check_input_file(receptor_pdb))
    ligand_sdf = str(check_input_file(ligand_sdf))

    receptor_pdbqt, receptor_json = _prepare_receptor_with_meeko(receptor_pdb, cfg)
    ligand_pdbqt = _prepare_ligand_with_meeko(ligand_sdf, cfg.ligand_pdbqt, cfg.docking_log)
    docking_pdbqt, docking_log = _run_vina(receptor_pdbqt, ligand_pdbqt, cfg)
    if not Path(docking_pdbqt).exists() or Path(docking_pdbqt).stat().st_size == 0:
        raise RuntimeError(f"Docking completed but produced no pose file: {docking_pdbqt}")

    docking_sdf = _export_docking_results_with_meeko(docking_pdbqt, cfg.docking_sdf, cfg.docking_log)
    best_pose_sdf = extract_pose1(docking_sdf, cfg.extracted_pose_sdf, cfg.extracted_pose_pdb)
    return {
        "receptor_pdbqt": receptor_pdbqt,
        "receptor_json": receptor_json,
        "ligand_pdbqt": ligand_pdbqt,
        "docking_pdbqt": docking_pdbqt,
        "docking_log": docking_log,
        "docking_sdf": docking_sdf,
        "best_pose_sdf": best_pose_sdf,
        "best_pose_pdb": cfg.extracted_pose_pdb,
    }
