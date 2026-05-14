from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from copy import deepcopy

from ..config.models import WorkflowConfig
from .files import resolve_replica_dirs
from .pathing import infer_run_input_paths, normalize_workflow_paths
from .executables import find_binary


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def _nonempty(value: object) -> bool:
    return str(value).strip() != ""


def _box_is_complete(cfg: WorkflowConfig) -> bool:
    values = [
        cfg.docking.search_center_x,
        cfg.docking.search_center_y,
        cfg.docking.search_center_z,
        cfg.docking.search_size_x,
        cfg.docking.search_size_y,
        cfg.docking.search_size_z,
    ]
    return all(v is not None for v in values)


def _box_center_is_complete(cfg: WorkflowConfig) -> bool:
    values = [
        cfg.docking.search_center_x,
        cfg.docking.search_center_y,
        cfg.docking.search_center_z,
    ]
    return all(v is not None for v in values)


def _effective_box_sizes(cfg: WorkflowConfig) -> list[float]:
    return [
        float(cfg.docking.search_size_x) if cfg.docking.search_size_x is not None else float(cfg.docking.default_box_size_x),
        float(cfg.docking.search_size_y) if cfg.docking.search_size_y is not None else float(cfg.docking.default_box_size_y),
        float(cfg.docking.search_size_z) if cfg.docking.search_size_z is not None else float(cfg.docking.default_box_size_z),
    ]


def _check_binary(name: str) -> bool:
    return find_binary(name) is not None


def preflight_validate(cfg: WorkflowConfig) -> ValidationResult:
    # Resolve all project-relative paths against workspace_root before checking existence.
    # Work on a deep copy so validation never mutates the live GUI config object.
    cfg = normalize_workflow_paths(deepcopy(cfg))

    errors: list[str] = []
    warnings: list[str] = []

    def add_error(message: str) -> None:
        if message not in errors:
            errors.append(message)

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    root = Path(cfg.workspace_root)
    if not root.exists():
        add_error(f"Workspace root does not exist: {root}")

    if cfg.do_prep:
        missing_residue_policy = str(cfg.prep.missing_residue_policy).strip().lower()
        if missing_residue_policy not in {"internal", "all", "none"}:
            add_error(
                "prep.missing_residue_policy must be one of: internal, all, none."
            )

        receptor = Path(cfg.prep.receptor_input)
        if not receptor.exists():
            add_error(f"Required receptor PDB was not found: {receptor}")

        ligand_mode = str(cfg.docking.ligand_input_mode).strip().lower()
        if ligand_mode == "smiles":
            if not _nonempty(cfg.docking.ligand_smiles):
                add_error("Ligand input mode is 'smiles' but no SMILES string was provided.")
        elif ligand_mode == "sdf":
            ligand_sdf = Path(cfg.docking.ligand_sdf_input)
            if not ligand_sdf.exists():
                add_error(f"Ligand input mode is 'sdf' but the SDF file was not found: {ligand_sdf}")
        elif ligand_mode == "pdb":
            ligand_pdb = Path(cfg.docking.ligand_pdb_input)
            if not ligand_pdb.exists():
                add_error(f"Ligand input mode is 'pdb' but the PDB file was not found: {ligand_pdb}")
        else:
            add_error(f"Unsupported ligand input mode: {cfg.docking.ligand_input_mode}")

        docking_mode = str(cfg.docking.docking_mode).strip().lower()
        if docking_mode not in {"auto", "external", "skip"}:
            add_error(f"Unsupported docking mode: {cfg.docking.docking_mode}")
        elif docking_mode == "external":
            external = Path(cfg.docking.external_docking_sdf)
            if not external.exists():
                add_error(f"Docking mode is 'external' but no external docking SDF was found: {external}")
        elif docking_mode == "skip":
            add_warning("Docking mode is 'skip'. MD will use the prepared ligand directly instead of a docked pose.")

        if ligand_mode == "pdb" and docking_mode in {"auto", "external"}:
            add_error("ligand_input_mode='pdb' currently supports docking_mode='skip' only. Peptide/PDB docking is not implemented.")

        if docking_mode == "auto" and ligand_mode != "pdb":
            search_mode = str(cfg.docking.search_space_mode).strip().lower()
            if search_mode not in {"auto", "manual"}:
                add_error(f"Unsupported docking search-space mode: {cfg.docking.search_space_mode}")
            else:
                if _box_is_complete(cfg):
                    sizes = [cfg.docking.search_size_x, cfg.docking.search_size_y, cfg.docking.search_size_z]
                    if min(float(s) for s in sizes if s is not None) <= 0:
                        add_error("Docking box sizes must all be positive numbers when a user box is provided.")
                elif _box_center_is_complete(cfg):
                    if min(_effective_box_sizes(cfg)) <= 0:
                        add_error(
                            "Docking box sizes must all be positive numbers. "
                            "Missing size axes fall back to default_box_size_x/y/z."
                        )
                elif search_mode == "manual":
                    add_error(
                        "search_space_mode='manual' requires docking box center_x/y/z. "
                        "Box size_x/y/z may be omitted and will fall back to default_box_size_x/y/z."
                    )
                elif receptor.exists():
                    try:
                        from ..prep.receptor import infer_search_space_from_pdb
                        inferred = infer_search_space_from_pdb(
                            str(receptor),
                            padding_angstrom=cfg.docking.search_padding_angstrom,
                            min_size_angstrom=cfg.docking.search_min_size_angstrom,
                            fallback_size_angstrom=(
                                cfg.docking.default_box_size_x,
                                cfg.docking.default_box_size_y,
                                cfg.docking.default_box_size_z,
                            ),
                            allow_protein_centroid_fallback=cfg.docking.allow_protein_centroid_box_fallback,
                        )
                    except Exception as exc:
                        add_error(f"Docking search-space inference failed: {exc}")
                    else:
                        if inferred.get("source") == "protein_centroid_fallback":
                            add_warning(
                                "Docking box inference is using the protein-centroid fallback. "
                                "This is convenient for exploration but weaker than a literature-grade, site-specific box."
                            )

        if docking_mode == "auto" and ligand_mode != "pdb":
            required_binaries = ["mk_prepare_receptor.py", "mk_prepare_ligand.py", "mk_export.py", "vina"]
            missing = [name for name in required_binaries if not _check_binary(name)]
            for name in missing:
                add_error(
                    f"Required docking binary is not available in PATH: {name}. Activate an environment with Meeko and AutoDock Vina installed."
                )

    if cfg.do_run_md:
        if cfg.run.n_replicas <= 0:
            add_error("MD n_replicas must be a positive integer.")
        if cfg.run.production_steps <= 0:
            add_error("MD production_steps must be a positive integer.")
        if cfg.run.equil_steps < 0:
            add_error("MD equil_steps cannot be negative.")
        if cfg.run.timestep_ps <= 0:
            add_error("MD timestep_ps must be a positive number.")
        if cfg.run.dcd_interval <= 0:
            add_error("MD dcd_interval must be a positive integer.")
        if cfg.run.log_interval <= 0:
            add_error("MD log_interval must be a positive integer.")
        if cfg.run.stdout_interval <= 0:
            add_error("MD stdout_interval must be a positive integer.")

        need_trajectory_frames = any([
            cfg.do_basic_analysis,
            cfg.do_waterbridge_analysis,
            cfg.do_advanced_analysis,
            cfg.do_mmgbsa_postprocess and cfg.mmgbsa.auto_run,
        ])
        if need_trajectory_frames and cfg.run.production_steps < cfg.run.dcd_interval:
            add_error(
                "MD production_steps is smaller than dcd_interval. "
                "No trajectory frames will be written, so downstream analyses cannot read trajectory.dcd."
            )
        if cfg.do_basic_analysis and cfg.run.production_steps < cfg.run.log_interval:
            add_error(
                "MD production_steps is smaller than log_interval. "
                "No md.log records will be written, so basic analysis cannot parse thermodynamic data."
            )

        try:
            protein_pdb, ligand_sdf = infer_run_input_paths(cfg)
        except Exception as exc:
            add_error(f"Could not infer MD input files: {exc}")
        else:
            protein_path = Path(protein_pdb)
            ligand_path = Path(ligand_sdf)
            if cfg.do_prep:
                generated = {
                    str(Path(cfg.prep.receptor_output)),
                    str(Path(cfg.docking.extracted_pose_sdf)),
                    str(Path(cfg.docking.ligand_output_sdf)),
                    str(Path(cfg.docking.ligand_output_pdb)),
                }
                if str(protein_path) not in generated and not protein_path.exists():
                    add_error(f"MD protein input file was not found: {protein_path}")
                if str(ligand_path) not in generated and not ligand_path.exists():
                    add_error(f"MD ligand input file was not found: {ligand_path}")
            else:
                if not protein_path.exists():
                    add_error(f"MD protein input file was not found: {protein_path}")
                if not ligand_path.exists():
                    add_error(f"MD ligand input file was not found: {ligand_path}")

        out_root = Path(cfg.run.output_root)
        parent = out_root if out_root.suffix == "" else out_root.parent
        if not parent.exists():
            try:
                parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                add_error(f"Could not create MD output directory parent '{parent}': {exc}")

    if cfg.do_mmgbsa_postprocess:
        if cfg.mmgbsa.auto_run:
            for name in ["MMPBSA.py", "cpptraj"]:
                if not _check_binary(name):
                    add_warning(
                        f"MM/GBSA binary not available in PATH: {name}. MM/GBSA will be skipped unless AmberTools is activated."
                    )
        if not cfg.mmgbsa.auto_run and not any(
            _nonempty(x)
            for x in [cfg.mmgbsa.final_dat, cfg.mmgbsa.final_csv, cfg.mmgbsa.per_frame_csv, cfg.mmgbsa.per_residue_csv]
        ):
            add_warning("MM/GBSA postprocess is enabled but no MM/GBSA input files were configured.")
        if cfg.mmgbsa.interval <= 0:
            add_error("MM/GBSA interval must be a positive integer.")
        if cfg.mmgbsa.startframe <= 0:
            add_error("MM/GBSA startframe must be a positive integer.")

    return ValidationResult(errors=errors, warnings=warnings)


def preflight_validate_existing_results(cfg: WorkflowConfig) -> ValidationResult:
    cfg = normalize_workflow_paths(deepcopy(cfg))
    base = preflight_validate(cfg)
    errors = list(base.errors)
    warnings = list(base.warnings)

    def add_error(message: str) -> None:
        if message not in errors:
            errors.append(message)

    def add_warning(message: str) -> None:
        if message not in warnings:
            warnings.append(message)

    def _require_replica_files(replica_root: str, replica_glob: str, required_files: list[str], label: str) -> None:
        replica_dirs = resolve_replica_dirs(replica_root, replica_glob)
        if not replica_dirs:
            add_error(f"{label}: no replica directories matching '{replica_glob}' were found under {replica_root}.")
            return
        for replica_dir in replica_dirs:
            for name in required_files:
                path = replica_dir / name
                if not path.exists():
                    add_error(f"{label}: required file is missing in {replica_dir.name}: {path}")
                elif path.is_file() and path.stat().st_size <= 0:
                    add_error(f"{label}: required file is empty in {replica_dir.name}: {path}")

    if cfg.do_basic_analysis:
        ligand = Path(cfg.basic.ligand_sdf)
        if not ligand.exists():
            add_error(f"Basic analysis ligand structure file was not found: {ligand}")
        _require_replica_files(
            cfg.basic.replica_root,
            cfg.basic.replica_glob,
            ["system_solvated.pdb", "trajectory.dcd", "md.log"],
            "Basic analysis",
        )

    if cfg.do_waterbridge_analysis:
        _require_replica_files(
            cfg.waterbridge.replica_root,
            cfg.waterbridge.replica_glob,
            [cfg.waterbridge.top_name, cfg.waterbridge.traj_name],
            "Water-bridge analysis",
        )

    if cfg.do_advanced_analysis:
        _require_replica_files(
            cfg.advanced.replica_root,
            cfg.advanced.replica_glob,
            [cfg.advanced.top_name, cfg.advanced.traj_name],
            "Advanced analysis",
        )

    if cfg.do_mmgbsa_postprocess and cfg.mmgbsa.auto_run:
        replica_dirs = resolve_replica_dirs(cfg.mmgbsa.source_root, "replica_*")
        required = [
            Path(cfg.mmgbsa.complex_solvated_prmtop).name,
            Path(cfg.mmgbsa.complex_prmtop).name,
            Path(cfg.mmgbsa.receptor_prmtop).name,
            Path(cfg.mmgbsa.ligand_prmtop).name,
            Path(cfg.mmgbsa.trajectory_nc).name,
        ]
        if replica_dirs:
            missing_by_replica = []
            for replica_dir in replica_dirs:
                missing = []
                for name in required:
                    path = replica_dir / name
                    if not path.exists() or (path.is_file() and path.stat().st_size <= 0):
                        missing.append(name)
                if missing:
                    missing_by_replica.append(f"{replica_dir.name}: {', '.join(missing)}")
            if missing_by_replica:
                preview = "; ".join(missing_by_replica[:3])
                if len(missing_by_replica) > 3:
                    preview = f"{preview}; ..."
                add_warning(
                    "MM/GBSA auto-run is enabled for existing results, but Amber-style inputs are missing, "
                    f"so MM/GBSA will be skipped. {preview}"
                )

    return ValidationResult(errors=errors, warnings=warnings)
