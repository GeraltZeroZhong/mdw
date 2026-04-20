from __future__ import annotations

from pathlib import Path
import numpy as np

from ..config import PlotStyleConfig
from .heatmaps import stacked_fraction_area, matrix_heatmap
from .series import line_series, two_panel_series, shaded_profile
from .snapshots import snapshot_grid


def plot_replica_rmsd(time_ns, protein_rmsd_A, ligand_rmsd_A, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, [protein_rmsd_A, ligand_rmsd_A], ["Protein backbone RMSD", "Ligand heavy-atom RMSD"], "RMSD (Å)", out_dir / "rmsd_timeseries", style, title=f"{replica_name}: RMSD", colors=[style.protein_color, style.ligand_color])


def plot_replica_min_distance(time_ns, min_distance_A, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, [min_distance_A], ["Ligand-protein minimum heavy-atom distance"], "Minimum distance (Å)", out_dir / "min_distance_timeseries", style, title=f"{replica_name}: Minimum ligand-protein distance", colors=[style.distance_color])


def plot_replica_rmsf(rmsf_rows, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    x = [row["resSeq"] for row in rmsf_rows]
    y = [row["rmsf_A"] for row in rmsf_rows]
    shaded_profile(x, y, [0.0] * len(y), "Cα RMSF (Å)", out_dir / "rmsf_ca", style, title=f"{replica_name}: Protein Cα RMSF")


def plot_replica_thermo(log_time_ns, temperature, density, potential, total, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    two_panel_series(log_time_ns, temperature, density, "Temperature (K)", "Density (g/mL)", out_dir / "temperature_density", style, title=f"{replica_name}: Thermodynamic stability", top_color=style.temperature_color, bottom_color=style.density_color)
    two_panel_series(log_time_ns, potential, total, "Potential energy (kJ/mol)", "Total energy (kJ/mol)", out_dir / "energy", style, title=f"{replica_name}: Energies", top_color=style.potential_energy_color, bottom_color=style.total_energy_color)


def plot_replica_counts(time_ns, count_map: dict[str, np.ndarray], out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, list(count_map.values()), list(count_map.keys()), "Count", out_dir / "interaction_counts_timeseries", style, title=f"{replica_name}: interaction counts")


def plot_replica_rg_sasa(time_ns, rg_A, complex_sasa_A2, buried_A2, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, [rg_A], ["Protein Rg"], "Radius of gyration (Å)", out_dir / "radius_of_gyration", style, title=f"{replica_name}: protein compactness", colors=[style.protein_color])
    line_series(time_ns, [complex_sasa_A2, buried_A2], ["Complex SASA", "Buried surface area"], "Area (Å²)", out_dir / "sasa_buried_surface", style, title=f"{replica_name}: solvent exposure and burial", colors=[style.protein_color, style.accent_color])


def plot_replica_pose_metrics(time_ns, com_distance_A, orientation_angle_deg, torsion_map, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, [com_distance_A, orientation_angle_deg], ["Ligand-pocket COM distance", "Ligand orientation angle"], "Metric value", out_dir / "ligand_pose_metrics", style, title=f"{replica_name}: ligand pose metrics", colors=[style.distance_color, style.accent_color])
    if torsion_map:
        names = list(torsion_map.keys())[:4]
        line_series(time_ns, [torsion_map[n] for n in names], names, "Dihedral angle (deg)", out_dir / "ligand_torsions", style, title=f"{replica_name}: ligand torsions")


def plot_replica_dssp(time_ns, fractions, residue_labels, occupancy, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    stacked_fraction_area(time_ns, fractions, out_dir / "dssp_fractions", style, title=f"{replica_name}: secondary-structure fractions")
    matrix_heatmap(
        occupancy,
        residue_labels,
        ["Helix", "Sheet", "Coil"],
        out_dir / "dssp_residue_occupancy",
        style,
        title=f"{replica_name}: residue secondary-structure occupancy",
        xlabel="Secondary-structure state",
        ylabel="Residue",
        annotate=False,
        vmin=0.0,
        vmax=1.0,
        cbar_label="Occupancy fraction",
    )


def plot_replica_snapshots(snapshot_entries, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    snapshot_grid(snapshot_entries, out_dir / "representative_snapshots", style, title=f"{replica_name}: representative structure snapshots")
