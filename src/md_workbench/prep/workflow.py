from __future__ import annotations

import shutil
from pathlib import Path

from ..config import DockingConfig, PrepConfig
from ..prep.ligand import extract_pose1, prepare_ligand_from_sdf, smiles_to_sdf, write_docking_box_config
from ..prep.receptor import assess_receptor_for_preprocess, infer_search_space_from_pdb, fix_receptor


def _should_preprocess(cfg: PrepConfig) -> tuple[bool, list[str]]:
    mode = str(cfg.preprocess_mode).strip().lower()
    if mode == "always":
        return True, ["preprocess_mode=always"]
    if mode == "never":
        return False, ["preprocess_mode=never"]
    assessment = assess_receptor_for_preprocess(cfg.receptor_input)
    return bool(assessment["recommended"]), list(assessment["reasons"])


def _prepare_ligand(cfg: DockingConfig) -> str:
    mode = str(cfg.ligand_input_mode).strip().lower()
    if mode == "smiles":
        return smiles_to_sdf(cfg.ligand_smiles, cfg.ligand_output_sdf)
    if mode == "sdf":
        return prepare_ligand_from_sdf(cfg.ligand_sdf_input, cfg.ligand_output_sdf)
    raise ValueError("ligand_input_mode must be either 'smiles' or 'sdf'.")


def _complete_user_box(cfg: DockingConfig) -> bool:
    values = [cfg.search_center_x, cfg.search_center_y, cfg.search_center_z, cfg.search_size_x, cfg.search_size_y, cfg.search_size_z]
    return all(v is not None for v in values)


def _user_box_center_complete(cfg: DockingConfig) -> bool:
    values = [cfg.search_center_x, cfg.search_center_y, cfg.search_center_z]
    return all(v is not None for v in values)


def _resolved_box_sizes(cfg: DockingConfig) -> tuple[float, float, float]:
    fallback = _fallback_size(cfg)
    return (
        float(cfg.search_size_x) if cfg.search_size_x is not None else fallback[0],
        float(cfg.search_size_y) if cfg.search_size_y is not None else fallback[1],
        float(cfg.search_size_z) if cfg.search_size_z is not None else fallback[2],
    )


def _manual_box(cfg: DockingConfig, source: str = "user_provided") -> dict:
    size_x, size_y, size_z = _resolved_box_sizes(cfg)
    return {
        "source": source,
        "residue": "",
        "center": [float(cfg.search_center_x), float(cfg.search_center_y), float(cfg.search_center_z)],
        "size": [size_x, size_y, size_z],
    }


def _fallback_size(cfg: DockingConfig) -> tuple[float, float, float]:
    return (
        float(cfg.default_box_size_x),
        float(cfg.default_box_size_y),
        float(cfg.default_box_size_z),
    )


def _resolve_search_space(receptor_input: str, docking_cfg: DockingConfig) -> dict:
    search_mode = str(docking_cfg.search_space_mode).strip().lower()
    if search_mode not in {"auto", "manual"}:
        raise ValueError("search_space_mode must be one of: auto, manual.")
    if _complete_user_box(docking_cfg):
        return _manual_box(docking_cfg)
    if _user_box_center_complete(docking_cfg):
        size_x, size_y, size_z = _resolved_box_sizes(docking_cfg)
        docking_cfg.search_size_x = size_x
        docking_cfg.search_size_y = size_y
        docking_cfg.search_size_z = size_z
        return _manual_box(docking_cfg, source="user_center_default_size")
    if search_mode == "manual":
        raise ValueError(
            "search_space_mode=manual requires docking box center_x/y/z. "
            "Box size_x/y/z may be omitted and will fall back to default_box_size_x/y/z."
        )

    inferred = infer_search_space_from_pdb(
        receptor_input,
        padding_angstrom=docking_cfg.search_padding_angstrom,
        min_size_angstrom=docking_cfg.search_min_size_angstrom,
        fallback_size_angstrom=_fallback_size(docking_cfg),
        allow_protein_centroid_fallback=docking_cfg.allow_protein_centroid_box_fallback,
    )
    docking_cfg.search_center_x, docking_cfg.search_center_y, docking_cfg.search_center_z = inferred["center"]
    docking_cfg.search_size_x, docking_cfg.search_size_y, docking_cfg.search_size_z = inferred["size"]
    return inferred


def run_prep_workflow(cfg: PrepConfig, docking_cfg: DockingConfig | None = None) -> dict[str, str | list[str] | bool | dict]:
    outputs: dict[str, str | list[str] | bool | dict] = {}
    docking_cfg = docking_cfg or DockingConfig()

    run_fix, reasons = _should_preprocess(cfg)
    outputs["preprocess_recommended_reasons"] = reasons
    outputs["preprocess_applied"] = run_fix

    if run_fix:
        receptor_path = fix_receptor(
            input_pdb=cfg.receptor_input,
            output_pdb=cfg.receptor_output,
            ph=cfg.ph,
            replace_nonstandard_residues=cfg.replace_nonstandard_residues,
            remove_heterogens_keep_water=cfg.remove_heterogens_keep_water,
            missing_residue_policy=cfg.missing_residue_policy,
        )
    else:
        src = Path(cfg.receptor_input)
        dst = Path(cfg.receptor_output)
        if src.resolve() != dst.resolve():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        receptor_path = str(dst)
    outputs["receptor"] = receptor_path

    ligand_path = _prepare_ligand(docking_cfg)
    outputs["ligand"] = ligand_path

    docking_mode = str(docking_cfg.docking_mode).strip().lower()
    if docking_mode == "auto":
        search_space = _resolve_search_space(cfg.receptor_input, docking_cfg)
        outputs["search_space"] = search_space
        if docking_cfg.do_write_docking_box:
            outputs["docking_box_config"] = write_docking_box_config(
                (float(docking_cfg.search_center_x), float(docking_cfg.search_center_y), float(docking_cfg.search_center_z)),
                (float(docking_cfg.search_size_x), float(docking_cfg.search_size_y), float(docking_cfg.search_size_z)),
                docking_cfg.docking_box_config,
            )
        from ..docking import run_auto_docking
        outputs["docking"] = run_auto_docking(receptor_path, ligand_path, docking_cfg)
    elif docking_mode == "external":
        outputs["pose1"] = extract_pose1(
            docking_cfg.external_docking_sdf,
            docking_cfg.extracted_pose_sdf,
            docking_cfg.extracted_pose_pdb,
        )
    elif docking_mode == "skip":
        outputs["docking"] = {"skipped": True}
    else:
        raise ValueError("docking_mode must be one of: auto, external, skip.")
    return outputs
