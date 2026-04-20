from __future__ import annotations

import json
import math
from pathlib import Path
import re
import subprocess
from sys import stdout
import sys
import time
from typing import Callable

from openmm import LangevinMiddleIntegrator, MonteCarloBarostat, Platform
from openmm.app import DCDReporter, HBonds, PME, NoCutoff, PDBFile, Simulation, StateDataReporter
from openmm.unit import bar, kelvin, molar, nanometer, nanometers, picosecond, picoseconds

from ..config import RunConfig
from ..core import ensure_dir
from ..core.progress import ProgressCallback, emit_progress
from .system_builder import build_modeller_and_forcefield


ReplicaStepCallback = Callable[[int, int, str, int | None], None]
ReplicaProgressCallback = ProgressCallback


def get_cuda_platform(use_mixed_precision: bool = True) -> tuple[Platform, dict[str, str]]:
    names = [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]
    if "CUDA" not in names:
        available = ", ".join(names) if names else "none"
        raise RuntimeError(f"OpenMM CUDA platform is not available. Available platforms: {available}")
    properties = {"Precision": "mixed" if use_mixed_precision else "single"}
    return Platform.getPlatformByName("CUDA"), properties


def _make_integrator(cfg: RunConfig, seed: int):
    integrator = LangevinMiddleIntegrator(
        cfg.temperature_kelvin * kelvin,
        cfg.friction_per_ps / picosecond,
        cfg.timestep_ps * picoseconds,
    )
    try:
        integrator.setRandomNumberSeed(seed)
    except Exception:
        pass
    return integrator


def _platform_error_hint(message: str) -> str | None:
    if "CUDA_ERROR_UNSUPPORTED_PTX_VERSION" in message:
        hint = (
            "The installed NVIDIA driver cannot load the CUDA PTX generated for this OpenMM build. "
            "This usually means the system driver supports an older CUDA/PTX level than the CUDA user-space "
            "libraries in the current conda environment."
        )
        mismatch = _describe_detected_cuda_mismatch()
        if mismatch:
            return f"{hint} {mismatch}"
        return hint
    return None


def _version_tuple(value: str) -> tuple[int, ...]:
    parts = []
    for item in re.split(r"[._-]", value):
        if item.isdigit():
            parts.append(int(item))
        else:
            break
    return tuple(parts)


def _read_nvidia_smi_info() -> dict[str, str]:
    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except Exception:
        return {}
    if proc.returncode != 0:
        return {}
    text = proc.stdout
    info: dict[str, str] = {}
    header_match = re.search(r"Driver Version:\s*([0-9.]+)\s+CUDA Version:\s*([0-9.]+)", text)
    if header_match:
        info["driver_version"] = header_match.group(1)
        info["system_cuda_version"] = header_match.group(2)
    gpu_match = re.search(r"\|\s+\d+\s+([A-Za-z0-9 ._-]+?)\s{2,}", text)
    if gpu_match:
        info["gpu_name"] = gpu_match.group(1).strip()
    return info


def _read_conda_package_version(package_name: str) -> str | None:
    meta_dir = Path(sys.prefix) / "conda-meta"
    if not meta_dir.exists():
        return None
    for meta_file in sorted(meta_dir.glob(f"{package_name}-*.json"), reverse=True):
        try:
            payload = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        version = payload.get("version")
        if isinstance(version, str) and version:
            return version
    return None


def _describe_detected_cuda_mismatch() -> str | None:
    nvidia = _read_nvidia_smi_info()
    system_cuda = nvidia.get("system_cuda_version")
    env_cuda = _read_conda_package_version("cuda-version")
    env_nvrtc = _read_conda_package_version("cuda-nvrtc")
    openmm_version = _read_conda_package_version("openmm")

    details: list[str] = []
    if nvidia.get("gpu_name"):
        details.append(f"GPU: {nvidia['gpu_name']}.")
    if nvidia.get("driver_version"):
        details.append(f"Driver: {nvidia['driver_version']}.")
    if system_cuda:
        details.append(f"Driver CUDA support: {system_cuda}.")
    if env_cuda:
        details.append(f"Conda cuda-version: {env_cuda}.")
    if env_nvrtc:
        details.append(f"Conda cuda-nvrtc: {env_nvrtc}.")
    if openmm_version:
        details.append(f"OpenMM: {openmm_version}.")

    if system_cuda and env_cuda and _version_tuple(env_cuda) > _version_tuple(system_cuda):
        details.append("The conda environment targets a newer CUDA user-space stack than the installed NVIDIA driver supports.")

    if not details:
        return None
    return "Detected versions: " + " ".join(details)


def _create_cuda_simulation_or_raise(topology, system, cfg: RunConfig, seed: int, platform, properties):
    integrator = _make_integrator(cfg, seed)
    try:
        simulation = Simulation(topology, system, integrator, platform, properties)
    except Exception as exc:
        message = [f"Failed to initialize OpenMM CUDA context: {exc}"]
        hint = _platform_error_hint(str(exc))
        if hint:
            message.append(f"Hint: {hint}")
        raise RuntimeError("\n".join(message)) from exc
    return simulation, {
        "selected_platform": "CUDA",
        "platform_properties": dict(properties),
        "available_platforms": [Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())],
        "used_fallback": False,
    }


def save_pdb(topology, positions, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as handle:
        PDBFile.writeFile(topology, positions, handle)


def _attach_production_reporters(simulation: Simulation, out_dir: Path, cfg: RunConfig) -> None:
    simulation.reporters.clear()
    simulation.reporters.append(
        StateDataReporter(
            stdout,
            cfg.stdout_interval,
            step=True,
            temperature=True,
            potentialEnergy=True,
            totalEnergy=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
        )
    )
    simulation.reporters.append(DCDReporter(str(out_dir / "trajectory.dcd"), cfg.dcd_interval))
    simulation.reporters.append(
        StateDataReporter(
            str(out_dir / "md.log"),
            cfg.log_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            separator=",",
        )
    )


def _try_export_amber_artifacts(out_dir: Path, dry_modeller, dry_system, solvated_modeller, solvated_system) -> dict[str, str]:
    try:
        import parmed as pmd
        import mdtraj as md
    except Exception:
        return {}

    outputs: dict[str, str] = {}
    try:
        dry_structure = pmd.openmm.load_topology(dry_modeller.topology, dry_system, xyz=dry_modeller.positions)
        solv_structure = pmd.openmm.load_topology(solvated_modeller.topology, solvated_system, xyz=solvated_modeller.positions)

        dry_structure.save(str(out_dir / "complex.prmtop"), overwrite=True)
        dry_structure.save(str(out_dir / "complex.inpcrd"), overwrite=True)
        solv_structure.save(str(out_dir / "complex_solvated.prmtop"), overwrite=True)
        solv_structure.save(str(out_dir / "complex_solvated.inpcrd"), overwrite=True)

        try:
            receptor_structure = dry_structure['!:LIG']
            ligand_structure = dry_structure[':LIG']
            receptor_structure.save(str(out_dir / "receptor.prmtop"), overwrite=True)
            receptor_structure.save(str(out_dir / "receptor.inpcrd"), overwrite=True)
            ligand_structure.save(str(out_dir / "ligand.prmtop"), overwrite=True)
            ligand_structure.save(str(out_dir / "ligand.inpcrd"), overwrite=True)
        except Exception:
            pass

        dcd_path = out_dir / "trajectory.dcd"
        top_path = out_dir / "system_solvated.pdb"
        if dcd_path.exists() and top_path.exists():
            traj = md.load(str(dcd_path), top=str(top_path))
            traj.save_netcdf(str(out_dir / "complex.nc"))

        for name in [
            "complex.prmtop", "complex_solvated.prmtop", "receptor.prmtop", "ligand.prmtop", "complex.nc"
        ]:
            path = out_dir / name
            if path.exists():
                outputs[name] = str(path.resolve())
    except Exception:
        return outputs
    return outputs


def _write_platform_info(out_dir: Path, replica_id: int, platform_info: dict) -> None:
    payload = {"replica_id": replica_id, **platform_info}
    with open(out_dir / "platform_info.json", "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _progress_interval(total_steps: int) -> int:
    return max(1, int(total_steps) // 200)


def _run_steps_with_progress(
    simulation: Simulation,
    n_steps: int,
    *,
    start_step: int,
    total_steps: int,
    detail_prefix: str,
    progress_callback: ReplicaStepCallback | None,
    eta_enabled: bool = False,
) -> None:
    if n_steps <= 0:
        return
    interval = _progress_interval(n_steps)
    completed = 0
    started_at = time.monotonic() if eta_enabled else 0.0
    while completed < n_steps:
        chunk = min(interval, n_steps - completed)
        simulation.step(chunk)
        completed += chunk
        if progress_callback is not None:
            current_step = min(start_step + completed, total_steps)
            eta_seconds = None
            if eta_enabled and completed < n_steps:
                elapsed = time.monotonic() - started_at
                if elapsed > 0:
                    rate = completed / elapsed
                    if rate > 0:
                        eta_seconds = max(0, int(math.ceil((n_steps - completed) / rate)))
            progress_callback(current_step, total_steps, f"{detail_prefix}: {completed}/{n_steps} steps", eta_seconds)


def run_single_replica(
    replica_id: int,
    cfg: RunConfig,
    modeller_template,
    forcefield,
    platform,
    properties,
    progress_callback: ReplicaStepCallback | None = None,
):
    seed = cfg.base_seed + replica_id
    out_dir = ensure_dir(Path(cfg.output_root) / f"replica_{replica_id}")
    total_steps = max(int(cfg.equil_steps), 0) + max(int(cfg.production_steps), 0)
    if progress_callback is not None:
        progress_callback(0, max(total_steps, 1), "Preparing solvated system and force field", None)

    dry_modeller = modeller_template.__class__(modeller_template.topology, modeller_template.positions)
    save_pdb(dry_modeller.topology, dry_modeller.positions, out_dir / "complex_from_components.pdb")

    dry_system = forcefield.createSystem(
        dry_modeller.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds,
    )

    modeller = modeller_template.__class__(modeller_template.topology, modeller_template.positions)
    modeller.addSolvent(
        forcefield,
        model="tip3p",
        padding=cfg.solvent_padding_nm * nanometers,
        ionicStrength=cfg.ionic_strength_molar * molar,
    )
    save_pdb(modeller.topology, modeller.positions, out_dir / "system_solvated.pdb")

    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1.0 * nanometer,
        constraints=HBonds,
    )

    barostat = MonteCarloBarostat(cfg.pressure_bar * bar, cfg.temperature_kelvin * kelvin, 25)
    try:
        barostat.setRandomNumberSeed(seed + 1000)
    except Exception:
        pass
    system.addForce(barostat)

    simulation, platform_info = _create_cuda_simulation_or_raise(
        modeller.topology,
        system,
        cfg,
        seed,
        platform,
        properties,
    )
    _write_platform_info(out_dir, replica_id, platform_info)

    simulation.context.setPositions(modeller.positions)
    if progress_callback is not None:
        progress_callback(0, max(total_steps, 1), "Minimizing energy", None)
    simulation.minimizeEnergy(maxIterations=5000)

    state = simulation.context.getState(getPositions=True)
    save_pdb(simulation.topology, state.getPositions(), out_dir / "minimized_solvated.pdb")

    try:
        simulation.context.setVelocitiesToTemperature(cfg.temperature_kelvin * kelvin, seed)
    except TypeError:
        simulation.context.setVelocitiesToTemperature(cfg.temperature_kelvin * kelvin)

    if cfg.equil_steps > 0:
        if progress_callback is not None:
            progress_callback(0, max(total_steps, 1), f"Equilibrating replica {replica_id}", None)
        _run_steps_with_progress(
            simulation,
            cfg.equil_steps,
            start_step=0,
            total_steps=max(total_steps, 1),
            detail_prefix="Equilibration",
            progress_callback=progress_callback,
            eta_enabled=False,
        )
    equilibrated_state = simulation.context.getState(getPositions=True)
    save_pdb(simulation.topology, equilibrated_state.getPositions(), out_dir / "equilibrated_solvated.pdb")

    simulation.context.setTime(0.0)
    simulation.context.setStepCount(0)
    simulation.currentStep = 0
    _attach_production_reporters(simulation, out_dir, cfg)
    if progress_callback is not None:
        progress_callback(max(int(cfg.equil_steps), 0), max(total_steps, 1), f"Producing replica {replica_id}", None)
    _run_steps_with_progress(
        simulation,
        cfg.production_steps,
        start_step=max(int(cfg.equil_steps), 0),
        total_steps=max(total_steps, 1),
        detail_prefix="Production",
        progress_callback=progress_callback,
        eta_enabled=True,
    )

    final_state = simulation.context.getState(getPositions=True)
    save_pdb(simulation.topology, final_state.getPositions(), out_dir / "final_frame.pdb")
    if progress_callback is not None:
        progress_callback(max(total_steps, 1), max(total_steps, 1), "Exporting final structures and Amber artifacts", None)
    amber_artifacts = _try_export_amber_artifacts(out_dir, dry_modeller, dry_system, modeller, system)
    return {
        "replica_dir": str(out_dir.resolve()),
        "amber_artifacts": amber_artifacts if amber_artifacts else {},
        "platform": platform_info,
    }


def _emit_replica_progress(
    callback: ReplicaProgressCallback | None,
    current: int,
    total: int,
    detail: str,
    *,
    subcurrent: int = 0,
    subtotal: int = 0,
    subdetail: str = "",
    subeta_seconds: int | None = None,
) -> None:
    emit_progress(
        callback,
        current,
        total,
        "md",
        detail,
        subcurrent=subcurrent,
        subtotal=subtotal,
        subdetail=subdetail,
        subeta_seconds=subeta_seconds,
    )


def run_md_workflow(cfg: RunConfig, progress_callback: ReplicaProgressCallback | None = None) -> list[dict[str, object]]:
    total_replicas = max(int(cfg.n_replicas), 1)
    _emit_replica_progress(progress_callback, 0, total_replicas, "Building the solvated MD system")
    modeller, forcefield = build_modeller_and_forcefield(cfg.protein_pdb, cfg.ligand_sdf)
    _emit_replica_progress(progress_callback, 0, total_replicas, "Initializing the OpenMM CUDA platform")
    platform, properties = get_cuda_platform(cfg.use_mixed_precision)
    outputs = []
    for replica_id in range(1, cfg.n_replicas + 1):
        replica_total_steps = max(int(cfg.equil_steps), 0) + max(int(cfg.production_steps), 0)
        _emit_replica_progress(
            progress_callback,
            replica_id - 1,
            total_replicas,
            f"Running MD replica {replica_id}/{cfg.n_replicas}",
            subcurrent=0,
            subtotal=max(replica_total_steps, 1),
            subdetail="Preparing replica inputs",
        )

        def _step_progress(current_step: int, total_steps: int, step_detail: str, eta_seconds: int | None) -> None:
            _emit_replica_progress(
                progress_callback,
                replica_id - 1,
                total_replicas,
                f"Running MD replica {replica_id}/{cfg.n_replicas}",
                subcurrent=current_step,
                subtotal=total_steps,
                subdetail=step_detail,
                subeta_seconds=eta_seconds,
            )

        replica_output = run_single_replica(
            replica_id,
            cfg,
            modeller,
            forcefield,
            platform,
            properties,
            progress_callback=_step_progress,
        )
        outputs.append(replica_output)
        _emit_replica_progress(
            progress_callback,
            replica_id,
            total_replicas,
            f"Completed MD replica {replica_id}/{cfg.n_replicas}",
            subcurrent=max(replica_total_steps, 1),
            subtotal=max(replica_total_steps, 1),
            subdetail="Replica completed",
        )
    return outputs
