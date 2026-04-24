from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .plot_style_defaults import (
    DEFAULT_PLOT_RENDERING,
    SCIENTIFIC_TEAL_PINK_COLORS,
    SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP,
    default_categorical_palette,
    default_plot_formats,
)


@dataclass
class PrepConfig:
    receptor_input: str = "inputs/receptor.pdb"
    receptor_output: str = "work/prep/prepared_receptor.pdb"
    ph: float = 7.4
    replace_nonstandard_residues: bool = True
    remove_heterogens_keep_water: bool = False
    missing_residue_policy: str = "internal"
    preprocess_mode: str = "auto"


@dataclass
class DockingConfig:
    docking_mode: str = "auto"
    ligand_input_mode: str = "smiles"
    ligand_smiles: str = ""
    ligand_sdf_input: str = "inputs/ligand_input.sdf"
    ligand_output_sdf: str = "work/prep/prepared_ligand.sdf"

    search_space_mode: str = "auto"
    search_center_x: float | None = None
    search_center_y: float | None = None
    search_center_z: float | None = None
    search_size_x: float | None = None
    search_size_y: float | None = None
    search_size_z: float | None = None
    search_padding_angstrom: float = 8.0
    search_min_size_angstrom: float = 16.0
    default_box_size_x: float = 20.0
    default_box_size_y: float = 20.0
    default_box_size_z: float = 20.0
    allow_protein_centroid_box_fallback: bool = False

    receptor_pdbqt: str = "work/docking/prepared_receptor.pdbqt"
    receptor_json: str = "work/docking/prepared_receptor.json"
    ligand_pdbqt: str = "work/docking/prepared_ligand.pdbqt"
    docking_pdbqt: str = "work/docking/docking_results.pdbqt"
    docking_sdf: str = "work/docking/docking_results.sdf"
    docking_log: str = "work/docking/docking.log"
    extracted_pose_sdf: str = "work/docking/best_ligand.sdf"
    extracted_pose_pdb: str = "work/docking/best_ligand.pdb"
    external_docking_sdf: str = ""
    docking_box_config: str = "work/docking/docking_box.txt"

    vina_exhaustiveness: int = 8
    vina_num_modes: int = 9
    vina_energy_range: float = 3.0
    vina_seed: int = 20260408

    do_write_docking_box: bool = True


@dataclass
class RunConfig:
    protein_pdb: str = ""
    ligand_sdf: str = ""
    output_root: str = "work/md"
    n_replicas: int = 3
    production_steps: int = 50_000_000
    equil_steps: int = 50_000
    temperature_kelvin: float = 300.0
    friction_per_ps: float = 1.0
    timestep_ps: float = 0.002
    pressure_bar: float = 1.0
    solvent_padding_nm: float = 1.0
    ionic_strength_molar: float = 0.15
    dcd_interval: int = 5_000
    log_interval: int = 5_000
    stdout_interval: int = 10_000
    base_seed: int = 20260408
    use_mixed_precision: bool = True


@dataclass
class BasicAnalysisConfig:
    replica_root: str = "work/md"
    replica_glob: str = "replica_*"
    ligand_sdf: str = "best_ligand.sdf"
    analysis_root: str = "work/analysis/basic"
    timestep_ps: float = 0.002
    dcd_interval_steps: int = 5000
    contact_cutoff_nm: float = 0.45
    salt_bridge_cutoff_nm: float = 0.40
    hbond_distance_nm: float = 0.35
    hbond_angle_deg: float = 120.0
    top_n_contacts_plot: int = 20
    top_n_key_distance_residues: int = 5
    sasa_probe_radius_nm: float = 0.14
    convergence_n_blocks: int = 5
    rolling_window_fraction: float = 0.10
    snapshot_n_frames: int = 3
    pose_cutoff_nm: float = 0.60


@dataclass
class WaterBridgeConfig:
    replica_root: str = "work/md"
    replica_glob: str = "replica_*"
    analysis_root: str = "work/analysis/waterbridge"
    top_name: str = "system_solvated.pdb"
    traj_name: str = "trajectory.dcd"
    timestep_ps: float = 0.002
    dcd_interval_steps: int = 5000
    hbond_distance_cutoff_nm: float = 0.25
    hbond_angle_cutoff_deg: float = 120.0


@dataclass
class AdvancedAnalysisConfig:
    replica_root: str = "work/md"
    replica_glob: str = "replica_*"
    analysis_root: str = "work/analysis/advanced"
    top_name: str = "system_solvated.pdb"
    traj_name: str = "trajectory.dcd"
    pocket_ca_cutoff_nm: float = 0.80
    align_selection: str = "protein and backbone"
    n_pca: int = 5
    tica_lag_frames: int = 10
    n_tics: int = 5
    n_clusters: int = 8
    msm_lag_frames: int = 20
    random_state: int = 20260408
    n_bins: int = 80
    temperature_K: float = 300.0
    kB_kcal_mol_K: float = 0.0019872041
    max_saved_components: int = 5
    state_network_threshold: float = 1e-6
    representative_snapshot_clusters: int = 4


@dataclass
class MMGBSAConfig:
    analysis_root: str = "work/analysis/mmgbsa"
    source_root: str = "work/md"
    auto_run: bool = True
    reuse_existing_outputs: bool = True
    non_blocking: bool = True
    use_mpi: bool = False
    mpi_ranks: int = 4
    mmpbsa_input_file: str = "mmpbsa.in"
    complex_solvated_prmtop: str = "complex_solvated.prmtop"
    complex_prmtop: str = "complex.prmtop"
    receptor_prmtop: str = "receptor.prmtop"
    ligand_prmtop: str = "ligand.prmtop"
    trajectory_nc: str = "complex.nc"
    final_dat: str = "mmpbsa_FINAL_RESULTS.dat"
    final_csv: str = "mmpbsa_FINAL_RESULTS.csv"
    per_frame_csv: str = "mmpbsa_FINAL_RESULTS.csv"
    per_residue_dat: str = "mmpbsa_DECOMP.dat"
    per_residue_csv: str = "mmpbsa_DECOMP.csv"
    top_n_residues_plot: int = 20
    startframe: int = 1
    interval: int = 10
    igb: int = 5
    saltcon: float = 0.150
    idecomp: int = 1


@dataclass
class PlotSelectionConfig:
    basic_replica_rmsd: bool = True
    basic_replica_min_distance: bool = True
    basic_replica_rmsf: bool = True
    basic_replica_counts: bool = True
    basic_replica_rg_sasa: bool = True
    basic_replica_pose_metrics: bool = True
    basic_replica_dssp: bool = True
    basic_replica_snapshots: bool = True
    basic_replica_thermo: bool = True

    basic_combined_rmsd: bool = True
    basic_combined_min_distance: bool = True
    basic_combined_rmsf: bool = True
    basic_combined_occupancy_bars: bool = True
    basic_combined_key_contact_traces: bool = True
    basic_combined_counts_and_shapes: bool = True
    basic_combined_dssp: bool = True
    basic_combined_interaction_heatmaps: bool = True
    basic_combined_convergence: bool = True

    plot_workflow_basic_replot: bool = True

    waterbridge_replica_counts: bool = True
    waterbridge_combined_occupancy: bool = True
    waterbridge_combined_counts: bool = True
    plot_workflow_waterbridge_replot: bool = True

    advanced_pca: bool = True
    advanced_tica: bool = True
    advanced_clustering: bool = True
    advanced_snapshots: bool = True
    advanced_msm: bool = True

    mmgbsa_summary: bool = True
    mmgbsa_per_frame: bool = True
    mmgbsa_per_residue: bool = True

    plot_workflow_reuse_csv: bool = True
    plot_workflow_advanced_replot: bool = True

    def enabled(self, name: str) -> bool:
        return bool(getattr(self, name, True))


@dataclass
class PlotStyleConfig:
    formats: list[str] = field(default_factory=default_plot_formats)
    dpi: int = DEFAULT_PLOT_RENDERING["dpi"]
    font_family: str = DEFAULT_PLOT_RENDERING["font_family"]
    title_size: float = DEFAULT_PLOT_RENDERING["title_size"]
    label_size: float = DEFAULT_PLOT_RENDERING["label_size"]
    tick_size: float = DEFAULT_PLOT_RENDERING["tick_size"]
    legend_size: float = DEFAULT_PLOT_RENDERING["legend_size"]
    line_width: float = DEFAULT_PLOT_RENDERING["line_width"]
    thin_line_width: float = DEFAULT_PLOT_RENDERING["thin_line_width"]
    marker_size: float = DEFAULT_PLOT_RENDERING["marker_size"]
    axes_line_width: float = DEFAULT_PLOT_RENDERING["axes_line_width"]
    grid_alpha: float = DEFAULT_PLOT_RENDERING["grid_alpha"]
    show_grid: bool = DEFAULT_PLOT_RENDERING["show_grid"]
    transparent_background: bool = DEFAULT_PLOT_RENDERING["transparent_background"]
    use_minor_ticks: bool = DEFAULT_PLOT_RENDERING["use_minor_ticks"]
    spine_color: str = SCIENTIFIC_TEAL_PINK_COLORS["spine_color"]
    grid_color: str = SCIENTIFIC_TEAL_PINK_COLORS["grid_color"]
    mean_line_color: str = SCIENTIFIC_TEAL_PINK_COLORS["mean_line_color"]
    band_color: str = SCIENTIFIC_TEAL_PINK_COLORS["band_color"]
    protein_color: str = SCIENTIFIC_TEAL_PINK_COLORS["protein_color"]
    ligand_color: str = SCIENTIFIC_TEAL_PINK_COLORS["ligand_color"]
    distance_color: str = SCIENTIFIC_TEAL_PINK_COLORS["distance_color"]
    temperature_color: str = SCIENTIFIC_TEAL_PINK_COLORS["temperature_color"]
    density_color: str = SCIENTIFIC_TEAL_PINK_COLORS["density_color"]
    potential_energy_color: str = SCIENTIFIC_TEAL_PINK_COLORS["potential_energy_color"]
    total_energy_color: str = SCIENTIFIC_TEAL_PINK_COLORS["total_energy_color"]
    bar_color: str = SCIENTIFIC_TEAL_PINK_COLORS["bar_color"]
    accent_color: str = SCIENTIFIC_TEAL_PINK_COLORS["accent_color"]
    cmap_continuous: str = SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP
    categorical_palette: list[str] = field(default_factory=default_categorical_palette)


@dataclass
class OutputBundleConfig:
    enabled: bool = True
    root: str = "results"
    figures_dir_name: str = "figures_combined"
    data_dir_name: str = "process_data"
    include_simulation_logs: bool = True


@dataclass
class WorkflowConfig:
    workspace_root: str = "."
    prep: PrepConfig = field(default_factory=PrepConfig)
    docking: DockingConfig = field(default_factory=DockingConfig)
    run: RunConfig = field(default_factory=RunConfig)
    basic: BasicAnalysisConfig = field(default_factory=BasicAnalysisConfig)
    waterbridge: WaterBridgeConfig = field(default_factory=WaterBridgeConfig)
    advanced: AdvancedAnalysisConfig = field(default_factory=AdvancedAnalysisConfig)
    mmgbsa: MMGBSAConfig = field(default_factory=MMGBSAConfig)
    plot_selection: PlotSelectionConfig = field(default_factory=PlotSelectionConfig)
    plot_style: PlotStyleConfig = field(default_factory=PlotStyleConfig)
    output_bundle: OutputBundleConfig = field(default_factory=OutputBundleConfig)

    do_prep: bool = True
    do_run_md: bool = True
    do_basic_analysis: bool = True
    do_waterbridge_analysis: bool = True
    do_advanced_analysis: bool = True
    do_mmgbsa_postprocess: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
