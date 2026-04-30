from __future__ import annotations

from pathlib import Path
import numpy as np

from ..config import PlotStyleConfig
from .heatmaps import stacked_fraction_area, matrix_heatmap
from .series import line_series, shaded_profile
from .snapshots import snapshot_grid
from .theme import remove_figure_outputs


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
    remove_figure_outputs(out_dir / "temperature_density")
    remove_figure_outputs(out_dir / "energy")
    line_series(log_time_ns, [temperature], ["Temperature"], "Temperature (K)", out_dir / "temperature", style, title=f"{replica_name}: temperature stability", colors=[style.temperature_color])
    line_series(log_time_ns, [density], ["Density"], "Density (g/mL)", out_dir / "density", style, title=f"{replica_name}: density stability", colors=[style.density_color])
    line_series(log_time_ns, [potential], ["Potential energy"], "Potential energy (kJ/mol)", out_dir / "potential_energy", style, title=f"{replica_name}: potential energy", colors=[style.potential_energy_color])
    line_series(log_time_ns, [total], ["Total energy"], "Total energy (kJ/mol)", out_dir / "total_energy", style, title=f"{replica_name}: total energy", colors=[style.total_energy_color])


def plot_replica_counts(time_ns, count_map: dict[str, np.ndarray], out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, list(count_map.values()), list(count_map.keys()), "Count", out_dir / "interaction_counts_timeseries", style, title=f"{replica_name}: interaction counts")


def plot_replica_rg_sasa(time_ns, rg_A, complex_sasa_A2, buried_A2, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    line_series(time_ns, [rg_A], ["Protein Rg"], "Radius of gyration (Å)", out_dir / "radius_of_gyration", style, title=f"{replica_name}: protein compactness", colors=[style.protein_color])
    buried_arr = np.asarray(buried_A2, dtype=float)
    finite_buried = buried_arr[np.isfinite(buried_arr)]
    sasa_series = [complex_sasa_A2]
    sasa_labels = ["Complex SASA"]
    sasa_colors = [style.protein_color]
    if finite_buried.size and float(np.nanmax(np.abs(finite_buried))) >= 1.0:
        sasa_series.append(buried_A2)
        sasa_labels.append("Buried surface area")
        sasa_colors.append(style.accent_color)
    line_series(time_ns, sasa_series, sasa_labels, "Area (Å²)", out_dir / "sasa_buried_surface", style, title=f"{replica_name}: solvent exposure", colors=sasa_colors)


def plot_replica_pose_metrics(time_ns, com_distance_A, orientation_angle_deg, torsion_map, out_dir: str | Path, replica_name: str, style: PlotStyleConfig):
    out_dir = Path(out_dir)
    remove_figure_outputs(out_dir / "ligand_pose_metrics")
    line_series(time_ns, [com_distance_A], ["Ligand-pocket COM distance"], "Distance (Å)", out_dir / "ligand_com_distance", style, title=f"{replica_name}: ligand-pocket COM distance", colors=[style.distance_color])
    line_series(time_ns, [orientation_angle_deg], ["Ligand orientation angle"], "Angle (deg)", out_dir / "ligand_orientation_angle", style, title=f"{replica_name}: ligand orientation angle", colors=[style.accent_color])
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
