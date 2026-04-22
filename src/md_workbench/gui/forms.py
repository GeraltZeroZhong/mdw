from __future__ import annotations

import math
import os
import subprocess
import sys
from dataclasses import fields
from pathlib import Path
import tkinter as tk
from tkinter import colorchooser, filedialog, ttk
from typing import Callable, get_type_hints

from .i18n import field_label, safe_text, section_hint, section_label, tr


DIR_FIELDS = {
    "workspace_root",
    "output_root",
    "replica_root",
    "analysis_root",
    "source_root",
    "root",
}

REQUIRED_FIELDS = {
    "receptor_input",
    "ligand_input_mode",
}
GENERATED_FIELDS = {
    "receptor_output",
    "ligand_output_sdf",
    "receptor_pdbqt",
    "receptor_json",
    "ligand_pdbqt",
    "docking_pdbqt",
    "docking_sdf",
    "docking_log",
    "extracted_pose_sdf",
    "extracted_pose_pdb",
    "docking_box_config",
}
OPTIONAL_OVERRIDE_FIELDS = {
    "protein_pdb",
    "ligand_sdf",
}
CONDITIONAL_REQUIRED_FIELDS = {
    "ligand_smiles",
    "ligand_sdf_input",
    "external_docking_sdf",
}

INPUT_FILE_FIELDS = {
    "receptor_input",
    "ligand_sdf_input",
    "external_docking_sdf",
    "protein_pdb",
    "ligand_sdf",
    "top_name",
    "traj_name",
    "final_dat",
    "final_csv",
    "per_frame_csv",
    "per_residue_csv",
    "mmpbsa_input_file",
    "complex_solvated_prmtop",
    "complex_prmtop",
    "receptor_prmtop",
    "ligand_prmtop",
    "trajectory_nc",
    "per_residue_dat",
}

OUTPUT_FILE_FIELDS = {
    "receptor_output",
    "ligand_output_sdf",
    "receptor_pdbqt",
    "receptor_json",
    "ligand_pdbqt",
    "docking_pdbqt",
    "docking_sdf",
    "docking_log",
    "extracted_pose_sdf",
    "extracted_pose_pdb",
    "docking_box_config",
}

ENUM_FIELDS = {
    "missing_residue_policy": ["internal", "all", "none"],
    "preprocess_mode": ["auto", "always", "never"],
    "ligand_input_mode": ["smiles", "sdf"],
    "docking_mode": ["auto", "external", "skip"],
    "search_space_mode": ["auto", "manual"],
}

PATH_SUFFIXES = (".pdb", ".sdf", ".dcd", ".csv", ".json", ".txt", ".svg", ".png", ".dat", ".nc", ".prmtop")

BASIC_FIELD_GROUPS = {
    "prep": {
        "receptor_input",
    },
    "docking": {
        "ligand_input_mode", "ligand_smiles", "ligand_sdf_input",
        "search_center_x", "search_center_y", "search_center_z",
    },
    "run": {
        "production_steps",
    },
    "basic": set(),
    "waterbridge": set(),
    "advanced": set(),
    "plot_selection": set(),
    "plot_style": {"formats"},
    "output_bundle": set(),
    "mmgbsa": set(),
}

BASIC_REQUIRED_FIELDS = {
    "search_center_x",
    "search_center_y",
    "search_center_z",
    "production_steps",
    "formats",
}

ADVANCED_FIELD_GROUPS = {
    "prep": [
        {
            "title": "Input",
            "hint": "Start from the raw receptor structure and choose how missing residues should be treated.",
            "fields": ["receptor_input", "preprocess_mode", "missing_residue_policy", "ph"],
        },
        {
            "title": "Cleanup",
            "hint": "These options control standardization before docking and MD begin.",
            "fields": ["replace_nonstandard_residues", "remove_heterogens_keep_water"],
        },
        {
            "title": "Generated Output",
            "hint": "Prepared receptor files are written automatically and usually do not need manual edits.",
            "fields": ["receptor_output"],
        },
    ],
    "docking": [
        {
            "title": "Ligand Input",
            "hint": "Choose the ligand source and whether docking is run automatically, skipped, or imported from an external SDF.",
            "fields": [
                "docking_mode",
                "ligand_input_mode",
                "ligand_smiles",
                "ligand_sdf_input",
                "external_docking_sdf",
                "ligand_output_sdf",
            ],
        },
        {
            "title": "Search Space",
            "hint": "Set the docking box center explicitly, or switch to auto mode to infer it from the receptor structure.",
            "fields": [
                "search_space_mode",
                "search_center_x",
                "search_center_y",
                "search_center_z",
                "search_size_x",
                "search_size_y",
                "search_size_z",
                "search_padding_angstrom",
                "search_min_size_angstrom",
                "default_box_size_x",
                "default_box_size_y",
                "default_box_size_z",
                "allow_protein_centroid_box_fallback",
            ],
        },
        {
            "title": "Docking Engine",
            "hint": "These parameters control the Vina search depth, pose count, randomness, and exported docking artifacts.",
            "fields": [
                "vina_exhaustiveness",
                "vina_num_modes",
                "vina_energy_range",
                "vina_seed",
                "do_write_docking_box",
                "docking_box_config",
                "receptor_pdbqt",
                "receptor_json",
                "ligand_pdbqt",
                "docking_pdbqt",
                "docking_sdf",
                "docking_log",
                "extracted_pose_sdf",
                "extracted_pose_pdb",
            ],
        },
    ],
    "run": [
        {
            "title": "Input Overrides",
            "hint": "Leave these blank to reuse the prepared receptor and docked ligand generated upstream.",
            "fields": ["protein_pdb", "ligand_sdf", "output_root"],
        },
        {
            "title": "Sampling Plan",
            "hint": "These settings determine how many replicas are run and how much production sampling is collected.",
            "fields": ["n_replicas", "production_steps", "equil_steps", "timestep_ps"],
        },
        {
            "title": "Simulation Environment",
            "hint": "Temperature, solvent, pressure, and integrator settings live here. Most users can keep the defaults.",
            "fields": [
                "temperature_kelvin",
                "friction_per_ps",
                "pressure_bar",
                "solvent_padding_nm",
                "ionic_strength_molar",
                "use_mixed_precision",
            ],
        },
        {
            "title": "Output Cadence",
            "hint": "These intervals control how often trajectories and logs are written.",
            "fields": ["dcd_interval", "log_interval", "stdout_interval", "base_seed"],
        },
    ],
    "basic": [
        {
            "title": "Input Paths",
            "hint": "Point the analysis to replica folders and the ligand definition used for contact-based metrics.",
            "fields": ["replica_root", "replica_glob", "ligand_sdf", "analysis_root", "timestep_ps", "dcd_interval_steps"],
        },
        {
            "title": "Interaction Rules",
            "hint": "These cutoffs define contacts, salt bridges, hydrogen bonds, and ligand-pocket pose metrics.",
            "fields": [
                "contact_cutoff_nm",
                "salt_bridge_cutoff_nm",
                "hbond_distance_nm",
                "hbond_angle_deg",
                "pose_cutoff_nm",
                "sasa_probe_radius_nm",
            ],
        },
        {
            "title": "Reporting",
            "hint": "Control how many residues, blocks, and representative frames are emphasized in the output figures.",
            "fields": [
                "top_n_contacts_plot",
                "top_n_key_distance_residues",
                "convergence_n_blocks",
                "rolling_window_fraction",
                "snapshot_n_frames",
            ],
        },
    ],
    "waterbridge": [
        {
            "title": "Input Paths",
            "hint": "Water-bridge analysis reads one topology and one trajectory file from each replica directory.",
            "fields": ["replica_root", "replica_glob", "analysis_root", "top_name", "traj_name", "timestep_ps", "dcd_interval_steps"],
        },
        {
            "title": "Geometry Cutoffs",
            "hint": "These values control the strictness of the water-mediated hydrogen-bond definition.",
            "fields": ["hbond_distance_cutoff_nm", "hbond_angle_cutoff_deg"],
        },
    ],
    "advanced": [
        {
            "title": "Input Paths",
            "hint": "These files define the trajectories and output folders used for PCA, tICA, clustering, and MSM analyses.",
            "fields": ["replica_root", "replica_glob", "analysis_root", "top_name", "traj_name", "align_selection"],
        },
        {
            "title": "Feature Extraction",
            "hint": "Start by defining which pocket motions are tracked and how many components are saved downstream.",
            "fields": ["pocket_ca_cutoff_nm", "max_saved_components", "representative_snapshot_clusters"],
        },
        {
            "title": "Dimensionality Reduction",
            "hint": "Tune PCA and tICA dimensionality before clustering and free-energy projection.",
            "fields": ["n_pca", "tica_lag_frames", "n_tics", "n_bins"],
        },
        {
            "title": "State Models",
            "hint": "These parameters shape clustering, MSM construction, and the state-network flux view.",
            "fields": ["n_clusters", "msm_lag_frames", "state_network_threshold", "random_state"],
        },
        {
            "title": "Thermodynamics",
            "hint": "Use these constants when converting projected densities into free-energy surfaces.",
            "fields": ["temperature_K", "kB_kcal_mol_K"],
        },
    ],
    "plot_selection": [
        {
            "title": "Basic Replica Figures",
            "hint": "Single-replica plots help inspect each trajectory independently.",
            "fields": [
                "basic_replica_rmsd",
                "basic_replica_min_distance",
                "basic_replica_rmsf",
                "basic_replica_counts",
                "basic_replica_rg_sasa",
                "basic_replica_pose_metrics",
                "basic_replica_dssp",
                "basic_replica_snapshots",
                "basic_replica_thermo",
            ],
        },
        {
            "title": "Basic Combined Figures",
            "hint": "Combined plots summarize replica agreement and the major interaction trends across the ensemble.",
            "fields": [
                "basic_combined_rmsd",
                "basic_combined_min_distance",
                "basic_combined_rmsf",
                "basic_combined_occupancy_bars",
                "basic_combined_key_contact_traces",
                "basic_combined_counts_and_shapes",
                "basic_combined_dssp",
                "basic_combined_interaction_heatmaps",
                "basic_combined_convergence",
                "plot_workflow_basic_replot",
            ],
        },
        {
            "title": "Water-Bridge Figures",
            "hint": "Enable these when water-mediated contacts matter for the binding story.",
            "fields": [
                "waterbridge_replica_counts",
                "waterbridge_combined_occupancy",
                "waterbridge_combined_counts",
                "plot_workflow_waterbridge_replot",
            ],
        },
        {
            "title": "Advanced State-Space Figures",
            "hint": "These plots cover dimensionality reduction, clustering, representative states, and MSM summaries.",
            "fields": [
                "advanced_pca",
                "advanced_tica",
                "advanced_clustering",
                "advanced_snapshots",
                "advanced_msm",
            ],
        },
        {
            "title": "MM/GBSA Figures",
            "hint": "Energy summaries can be enabled independently from the core structural analyses.",
            "fields": ["mmgbsa_summary", "mmgbsa_per_frame", "mmgbsa_per_residue"],
        },
    ],
    "plot_style": [
        {
            "title": "Export",
            "hint": "Choose the figure formats and rendering resolution for all generated plots.",
            "fields": ["formats", "dpi", "transparent_background", "font_family"],
        },
        {
            "title": "Typography",
            "hint": "Keep title, label, tick, and legend sizes visually balanced across all figure families.",
            "fields": ["title_size", "label_size", "tick_size", "legend_size"],
        },
        {
            "title": "Lines and Markers",
            "hint": "These values control line weights, marker sizes, and axis emphasis.",
            "fields": ["line_width", "thin_line_width", "marker_size", "axes_line_width"],
        },
        {
            "title": "Grid and Theme",
            "hint": "Grid visibility, alpha, and minor ticks help tune readability without changing the data itself.",
            "fields": ["show_grid", "grid_alpha", "use_minor_ticks", "spine_color", "grid_color", "mean_line_color", "band_color"],
        },
        {
            "title": "Color Mapping",
            "hint": "Set the default colors used by structural, thermodynamic, and categorical plot elements.",
            "fields": [
                "protein_color",
                "ligand_color",
                "distance_color",
                "temperature_color",
                "density_color",
                "potential_energy_color",
                "total_energy_color",
                "bar_color",
                "accent_color",
                "cmap_continuous",
                "categorical_palette",
            ],
        },
    ],
    "output_bundle": [
        {
            "title": "Bundle Layout",
            "hint": "Bundle figures and process data into stable folders for sharing and downstream review.",
            "fields": ["enabled", "root", "figures_dir_name", "data_dir_name", "include_simulation_logs"],
        },
    ],
    "mmgbsa": [
        {
            "title": "Execution",
            "hint": "Choose whether MM/GBSA runs automatically and whether failures should remain non-blocking.",
            "fields": ["analysis_root", "source_root", "auto_run", "non_blocking", "use_mpi", "mpi_ranks", "startframe", "interval", "igb", "saltcon", "idecomp"],
        },
        {
            "title": "Input Files",
            "hint": "These filenames are read from the replica outputs when Amber-style MM/GBSA inputs are available.",
            "fields": [
                "mmpbsa_input_file",
                "complex_solvated_prmtop",
                "complex_prmtop",
                "receptor_prmtop",
                "ligand_prmtop",
                "trajectory_nc",
            ],
        },
        {
            "title": "Parsed Outputs",
            "hint": "Parsed tables and plotting settings are stored separately so you can rerun the visualization later.",
            "fields": ["final_dat", "final_csv", "per_frame_csv", "per_residue_dat", "per_residue_csv", "top_n_residues_plot"],
        },
    ],
}

RUN_DURATION_PRESETS_NS = {
    "smoke": 0.02,
    "verify_20ns": 20.0,
    "standard_50ns": 50.0,
    "publish_200ns": 200.0,
}



def _visible_fields(section_key: str, all_fields, mode: str, obj=None):
    if mode == "advanced":
        return list(all_fields)
    keep = BASIC_FIELD_GROUPS.get(section_key)
    if keep is None:
        return list(all_fields)
    visible = [field for field in all_fields if field.name in keep]
    if section_key == "docking" and obj is not None:
        ligand_mode = str(getattr(obj, "ligand_input_mode", "smiles")).strip().lower()
        if ligand_mode == "smiles":
            visible = [field for field in visible if field.name != "ligand_sdf_input"]
        elif ligand_mode == "sdf":
            visible = [field for field in visible if field.name != "ligand_smiles"]
    return visible


def _grouped_visible_fields(section_key: str, visible_fields, mode: str):
    if mode != "advanced":
        return [(None, None, list(visible_fields))]
    definitions = ADVANCED_FIELD_GROUPS.get(section_key)
    if not definitions:
        return [(None, None, list(visible_fields))]

    by_name = {field.name: field for field in visible_fields}
    groups = []
    used: set[str] = set()
    for definition in definitions:
        group_fields = [by_name[name] for name in definition["fields"] if name in by_name]
        if not group_fields:
            continue
        groups.append((definition["title"], definition["hint"], group_fields))
        used.update(field.name for field in group_fields)

    remaining = [field for field in visible_fields if field.name not in used]
    if remaining:
        groups.append((None, None, remaining))
    return groups


def _field_status(section_key: str, field_name: str, obj, mode: str = "advanced") -> str:
    if mode == "basic" and field_name in BASIC_REQUIRED_FIELDS:
        return "required"
    if field_name in REQUIRED_FIELDS:
        return "required"
    if field_name == "ligand_smiles":
        return "conditional_required" if getattr(obj, "ligand_input_mode", "smiles") == "smiles" else "optional"
    if field_name == "ligand_sdf_input":
        return "conditional_required" if getattr(obj, "ligand_input_mode", "smiles") == "sdf" else "optional"
    if field_name == "external_docking_sdf":
        return "conditional_required" if getattr(obj, "docking_mode", "auto") == "external" else "optional"
    if field_name in {"search_center_x", "search_center_y", "search_center_z", "search_size_x", "search_size_y", "search_size_z"}:
        return "optional"
    if field_name in GENERATED_FIELDS:
        return "generated"
    if field_name in OPTIONAL_OVERRIDE_FIELDS:
        return "optional_override"
    if field_name in CONDITIONAL_REQUIRED_FIELDS:
        return "conditional_required"
    return "optional"


class ScrollableFrame(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0, borderwidth=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()

    def _on_canvas_configure(self, event):
        self.canvas.itemconfigure(self.window, width=event.width)

    def _bind_mousewheel(self):
        def _on_mousewheel(event):
            delta = 0
            if hasattr(event, "delta") and event.delta:
                delta = -1 * int(event.delta / 120)
            elif getattr(event, "num", None) == 4:
                delta = -1
            elif getattr(event, "num", None) == 5:
                delta = 1
            if delta:
                self.canvas.yview_scroll(delta, "units")

        self.canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.canvas.bind_all("<Button-4>", _on_mousewheel)
        self.canvas.bind_all("<Button-5>", _on_mousewheel)


def _parse_by_type(raw, typ, original_value):
    from typing import get_args, get_origin

    if original_value is None and str(raw).strip() == "":
        return None
    origin = get_origin(typ)
    args = get_args(typ)
    if origin is not None and type(None) in args:
        concrete = next((arg for arg in args if arg is not type(None)), str)
        if str(raw).strip() == "":
            return None
        return _parse_by_type(raw, concrete, None if original_value is None else original_value)
    if typ is bool:
        return bool(raw)
    if typ is int:
        return int(raw)
    if typ is float:
        return float(raw)
    if isinstance(original_value, list):
        return [item.strip() for item in str(raw).split(",") if item.strip()]
    return raw


def _looks_like_color(name: str, value) -> bool:
    return isinstance(value, str) and name.endswith("_color")


def _path_kind(field_name: str, value) -> str | None:
    if field_name in DIR_FIELDS:
        return "dir"
    if field_name in INPUT_FILE_FIELDS:
        return "input_file"
    if field_name in OUTPUT_FILE_FIELDS:
        return "output_file"
    if isinstance(value, str):
        if value.endswith(PATH_SUFFIXES):
            return "input_file"
        if field_name.endswith("_file"):
            return "input_file"
        if field_name.endswith("_dir"):
            return "dir"
    return None


def _open_location(path_str: str, workspace_root: str):
    raw = (path_str or "").strip()
    if not raw:
        target = Path(workspace_root or ".")
    else:
        p = Path(raw)
        if not p.is_absolute():
            p = Path(workspace_root or ".") / p
        target = p if p.is_dir() else p.parent
    target = target.expanduser().resolve()
    if sys.platform.startswith("win"):
        os.startfile(str(target))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])


class DataclassFrame(ttk.LabelFrame):
    def __init__(
        self,
        master,
        section_key: str,
        obj,
        lang: str = "en_US",
        workspace_root_getter=None,
        display_mode_getter=None,
        action_logger: Callable[[str], None] | None = None,
    ):
        self.section_key = section_key
        self.lang = lang
        self.workspace_root_getter = workspace_root_getter or (lambda: ".")
        self.display_mode_getter = display_mode_getter or (lambda: "advanced")
        self.action_logger = action_logger or (lambda _text: None)
        super().__init__(master, text=section_label(section_key, lang), padding=0)
        self.obj = obj
        self.type_hints = get_type_hints(type(obj))
        self.vars = {}
        self.widgets = {}
        self.editor_frames = {}
        self.action_frames = {}
        self._last_logged_values = {}
        self._run_duration_updating = False
        self._build()

    def refresh(self):
        self._build()

    def _log_action(self, text: str) -> None:
        try:
            self.action_logger(safe_text(text))
        except Exception:
            pass

    def _format_value_for_log(self, value) -> str:
        if isinstance(value, bool):
            return "Enabled" if value else "Disabled"
        text = "" if value is None else str(value).strip()
        if not text:
            return "(empty)"
        if len(text) > 180:
            return text[:177] + "..."
        return text

    def _remember_logged_value(self, field_name: str, value) -> None:
        self._last_logged_values[field_name] = self._format_value_for_log(value)

    def _log_field_value(self, field_name: str, value, action: str = "Updated") -> None:
        value_text = self._format_value_for_log(value)
        if self._last_logged_values.get(field_name) == value_text:
            return
        self._last_logged_values[field_name] = value_text
        self._log_action(
            f"{action} {section_label(self.section_key, self.lang)} -> {field_label(field_name, self.lang)}: {value_text}"
        )

    def _open_field_location(self, field_name: str, var: tk.Variable) -> None:
        _open_location(str(var.get()), self.workspace_root_getter())
        self._log_action(
            f"Opened location for {section_label(self.section_key, self.lang)} -> {field_label(field_name, self.lang)}: {self._format_value_for_log(var.get())}"
        )

    def _choose_dir(self, field_name: str, var: tk.StringVar):
        base = self.workspace_root_getter() or "."
        initial = var.get().strip() or base
        if not Path(initial).is_absolute():
            initial = str((Path(base) / initial).resolve())
        selected = filedialog.askdirectory(initialdir=initial if Path(initial).exists() else base)
        if selected:
            try:
                chosen = os.path.relpath(selected, base)
                chosen = chosen if len(chosen) < len(selected) else selected
            except Exception:
                chosen = selected
            var.set(chosen)
            self._log_field_value(field_name, chosen, action="Selected folder for")

    def _choose_file(self, field_name: str, var: tk.StringVar, save: bool = False):
        base = self.workspace_root_getter() or "."
        current = var.get().strip()
        initialdir = base
        initialfile = ""
        if current:
            p = Path(current)
            if not p.is_absolute():
                p = Path(base) / p
            initialdir = str(p.parent if p.parent.exists() else Path(base))
            initialfile = p.name
        dialog = filedialog.asksaveasfilename if save else filedialog.askopenfilename
        selected = dialog(initialdir=initialdir, initialfile=initialfile)
        if selected:
            try:
                chosen = os.path.relpath(selected, base)
                chosen = chosen if len(chosen) < len(selected) else selected
            except Exception:
                chosen = selected
            var.set(chosen)
            self._log_field_value(field_name, chosen, action="Selected output file for" if save else "Selected file for")

    def _pick_color(self, field_name: str, var: tk.StringVar):
        color = colorchooser.askcolor(color=var.get() or "#000000")
        if color and color[1]:
            var.set(color[1])
            self._log_field_value(field_name, color[1], action="Selected color for")

    def _make_editor(self, editor_frame, field_name: str, value):
        if isinstance(value, bool):
            var = tk.BooleanVar(value=value)
            widget = ttk.Checkbutton(
                editor_frame,
                variable=var,
                style="Switch.TCheckbutton",
                command=lambda fn=field_name, v=var: self._log_field_value(fn, v.get(), action="Toggled"),
            )
            widget.pack(anchor="w")
            return var, widget
        shown = ", ".join(value) if isinstance(value, list) else ("" if value is None else str(value))
        var = tk.StringVar(value=shown)
        if field_name in ENUM_FIELDS:
            widget = ttk.Combobox(editor_frame, textvariable=var, values=ENUM_FIELDS[field_name], state="readonly")
            widget.bind("<<ComboboxSelected>>", lambda _event, fn=field_name, v=var: self._log_field_value(fn, v.get(), action="Selected"))
        else:
            widget = ttk.Entry(editor_frame, textvariable=var, width=58)
            widget.bind("<FocusOut>", lambda _event, fn=field_name, v=var: self._log_field_value(fn, v.get(), action="Updated"))
            widget.bind("<Return>", lambda _event, fn=field_name, v=var: self._log_field_value(fn, v.get(), action="Updated"))
        widget.pack(fill="x", expand=True)
        return var, widget

    def _safe_int_from_var(self, var: tk.StringVar) -> int | None:
        text = str(var.get()).strip()
        if not text:
            return None
        try:
            return int(text)
        except Exception:
            return None

    def _safe_float_from_var(self, var: tk.StringVar) -> float | None:
        text = str(var.get()).strip()
        if not text:
            return None
        try:
            return float(text)
        except Exception:
            return None

    def _format_duration_ns(self, duration_ns: float) -> str:
        if duration_ns >= 1000.0:
            return f"{duration_ns / 1000.0:.3g} us"
        if duration_ns >= 1.0:
            return f"{duration_ns:.3g} ns"
        if duration_ns >= 0.001:
            return f"{duration_ns * 1000.0:.3g} ps"
        return f"{duration_ns * 1_000_000.0:.3g} fs"

    def _format_timestep(self, timestep_ps: float) -> str:
        timestep_fs = timestep_ps * 1000.0
        if timestep_fs >= 1.0:
            return f"{timestep_fs:.3g} fs/step"
        return f"{timestep_ps:.3g} ps/step"

    def _steps_to_ns(self, steps: int, timestep_ps: float) -> float:
        return float(steps) * float(timestep_ps) / 1000.0

    def _ns_to_steps(self, duration_ns: float, timestep_ps: float) -> int:
        if timestep_ps <= 0:
            raise ValueError("timestep_ps must be positive")
        return max(1, int(round(float(duration_ns) * 1000.0 / float(timestep_ps))))

    def _with_run_duration_update(self, callback):
        self._run_duration_updating = True
        try:
            callback()
        finally:
            self._run_duration_updating = False

    def _attach_run_duration_helpers(self) -> None:
        if self.section_key != "run":
            return
        if "production_steps" not in self.vars or "equil_steps" not in self.vars or "timestep_ps" not in self.vars:
            return

        prod_var = self.vars["production_steps"][0]
        equil_var = self.vars["equil_steps"][0]
        timestep_var = self.vars["timestep_ps"][0]
        prod_editor = self.editor_frames["production_steps"]
        equil_editor = self.editor_frames["equil_steps"]

        preset_options = [
            (tr("run_preset_smoke", self.lang), "smoke"),
            (tr("run_preset_20ns", self.lang), "verify_20ns"),
            (tr("run_preset_50ns", self.lang), "standard_50ns"),
            (tr("run_preset_200ns", self.lang), "publish_200ns"),
            (tr("run_preset_custom", self.lang), "custom"),
        ]
        label_to_key = {label: key for label, key in preset_options}
        key_to_label = {key: label for label, key in preset_options}

        controls = ttk.Frame(prod_editor)
        controls.pack(fill="x", expand=True, pady=(6, 0))
        ttk.Label(controls, text=tr("run_length_preset", self.lang), style="Muted.TLabel").pack(side="left")
        preset_var = tk.StringVar(value=key_to_label["custom"])
        preset_combo = ttk.Combobox(
            controls,
            textvariable=preset_var,
            values=[label for label, _ in preset_options],
            state="readonly",
            width=26,
        )
        preset_combo.pack(side="left", padx=(6, 12))
        ttk.Label(controls, text=tr("run_length_custom_ns", self.lang), style="Muted.TLabel").pack(side="left")
        custom_ns_var = tk.StringVar(value="")
        custom_ns_entry = ttk.Entry(controls, textvariable=custom_ns_var, width=12)
        custom_ns_entry.pack(side="left", padx=(6, 0))

        prod_hint_var = tk.StringVar(value="")
        ttk.Label(
            prod_editor,
            textvariable=prod_hint_var,
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        equil_hint_var = tk.StringVar(value="")
        ttk.Label(
            equil_editor,
            textvariable=equil_hint_var,
            style="Muted.TLabel",
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))

        def update_hints() -> None:
            prod_steps = self._safe_int_from_var(prod_var)
            equil_steps = self._safe_int_from_var(equil_var)
            timestep_ps = self._safe_float_from_var(timestep_var)
            if timestep_ps is None or timestep_ps <= 0:
                prod_hint_var.set(tr("run_duration_invalid", self.lang))
                equil_hint_var.set(tr("run_duration_invalid", self.lang))
                return
            if prod_steps is None or prod_steps < 0:
                prod_hint_var.set(tr("run_duration_invalid", self.lang))
            else:
                prod_hint_var.set(
                    tr("run_duration_hint", self.lang).format(
                        steps=f"{prod_steps:,}",
                        duration=self._format_duration_ns(self._steps_to_ns(prod_steps, timestep_ps)),
                        timestep=self._format_timestep(timestep_ps),
                    )
                )
            if equil_steps is None or equil_steps < 0:
                equil_hint_var.set(tr("run_duration_invalid", self.lang))
            else:
                equil_hint_var.set(
                    tr("equil_duration_hint", self.lang).format(
                        steps=f"{equil_steps:,}",
                        duration=self._format_duration_ns(self._steps_to_ns(equil_steps, timestep_ps)),
                    )
                )

        def sync_preset_from_steps() -> None:
            prod_steps = self._safe_int_from_var(prod_var)
            timestep_ps = self._safe_float_from_var(timestep_var)
            if prod_steps is None or prod_steps < 0 or timestep_ps is None or timestep_ps <= 0:
                update_hints()
                return
            duration_ns = self._steps_to_ns(prod_steps, timestep_ps)
            step_tolerance_ns = timestep_ps / 1000.0
            matched_key = "custom"
            for key, preset_duration_ns in RUN_DURATION_PRESETS_NS.items():
                if math.isclose(duration_ns, preset_duration_ns, rel_tol=0.0, abs_tol=step_tolerance_ns):
                    matched_key = key
                    break

            def update_controls():
                preset_var.set(key_to_label[matched_key])
                custom_ns_var.set(f"{duration_ns:.6g}")
                update_hints()

            self._with_run_duration_update(update_controls)

        def apply_duration(duration_ns: float, preset_key: str | None = None) -> None:
            timestep_ps = self._safe_float_from_var(timestep_var)
            if timestep_ps is None or timestep_ps <= 0 or duration_ns <= 0:
                update_hints()
                return
            prod_steps = self._ns_to_steps(duration_ns, timestep_ps)

            def update_controls():
                prod_var.set(str(prod_steps))
                custom_ns_var.set(f"{duration_ns:.6g}")
                if preset_key is not None:
                    preset_var.set(key_to_label[preset_key])
                update_hints()

            self._with_run_duration_update(update_controls)

        def on_preset_change(*_args) -> None:
            if self._run_duration_updating:
                return
            key = label_to_key.get(preset_var.get(), "custom")
            if key == "custom":
                update_hints()
                return
            apply_duration(RUN_DURATION_PRESETS_NS[key], preset_key=key)
            self._log_action(f"Selected {section_label(self.section_key, self.lang)} run length preset: {preset_var.get()}")

        def on_custom_ns_change(*_args) -> None:
            if self._run_duration_updating:
                return
            duration_ns = self._safe_float_from_var(custom_ns_var)
            if duration_ns is None or duration_ns <= 0:
                update_hints()
                return
            apply_duration(duration_ns, preset_key="custom")

        def on_timestep_change(*_args) -> None:
            if self._run_duration_updating:
                return
            key = label_to_key.get(preset_var.get(), "custom")
            if key in RUN_DURATION_PRESETS_NS:
                apply_duration(RUN_DURATION_PRESETS_NS[key], preset_key=key)
                return
            duration_ns = self._safe_float_from_var(custom_ns_var)
            if duration_ns is not None and duration_ns > 0:
                apply_duration(duration_ns, preset_key="custom")
                return
            update_hints()

        def on_steps_change(*_args) -> None:
            if self._run_duration_updating:
                return
            sync_preset_from_steps()

        preset_var.trace_add("write", on_preset_change)
        custom_ns_var.trace_add("write", on_custom_ns_change)
        timestep_var.trace_add("write", on_timestep_change)
        prod_var.trace_add("write", on_steps_change)
        equil_var.trace_add("write", lambda *_args: update_hints())

        def _log_custom_duration(_event=None) -> None:
            duration_ns = self._safe_float_from_var(custom_ns_var)
            if duration_ns is not None and duration_ns > 0:
                self._log_action(f"Updated {section_label(self.section_key, self.lang)} custom run length: {duration_ns:.6g} ns")

        custom_ns_entry.bind("<FocusOut>", _log_custom_duration)
        custom_ns_entry.bind("<Return>", _log_custom_duration)

        sync_preset_from_steps()

    def _build(self):
        for child in self.winfo_children():
            child.destroy()
        self.configure(text=section_label(self.section_key, self.lang))
        self.vars = {}
        self.widgets = {}
        self.editor_frames = {}
        self.action_frames = {}
        self._last_logged_values = {}

        scroll = ScrollableFrame(self)
        scroll.pack(fill="both", expand=True, padx=10, pady=10)
        container = scroll.inner

        hint = section_hint(self.section_key, self.lang)
        if hint:
            ttk.Label(container, text=safe_text(hint), style="Muted.TLabel", wraplength=1080, justify="left").grid(
                row=0, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 12)
            )

        all_fields = list(fields(self.obj))
        visible_fields = _visible_fields(self.section_key, all_fields, self.display_mode_getter(), self.obj)
        grouped_fields = _grouped_visible_fields(self.section_key, visible_fields, self.display_mode_getter())

        content_row = 1
        hidden_count = len(all_fields) - len(visible_fields)
        if self.display_mode_getter() == "basic" and hidden_count > 0:
            ttk.Label(
                container,
                text=tr("basic_mode_hint", self.lang).format(shown=len(visible_fields), total=len(all_fields), hidden=hidden_count),
                style="Muted.TLabel",
                wraplength=1080,
                justify="left",
            ).grid(row=content_row, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 10))
            content_row += 1

        group_count = len(grouped_fields)
        for group_idx, (group_title, group_hint, fields_in_group) in enumerate(grouped_fields):
            if group_title:
                if group_idx > 0:
                    ttk.Separator(container, orient="horizontal").grid(
                        row=content_row,
                        column=0,
                        columnspan=3,
                        sticky="ew",
                        padx=6,
                        pady=(4, 10),
                    )
                    content_row += 1
                ttk.Label(container, text=safe_text(group_title), style="GroupHeader.TLabel").grid(
                    row=content_row,
                    column=0,
                    columnspan=3,
                    sticky="w",
                    padx=6,
                    pady=(0, 2),
                )
                content_row += 1
                if group_hint:
                    ttk.Label(
                        container,
                        text=safe_text(group_hint),
                        style="GroupHint.TLabel",
                        wraplength=1080,
                        justify="left",
                    ).grid(row=content_row, column=0, columnspan=3, sticky="ew", padx=6, pady=(0, 8))
                    content_row += 1
            elif group_idx > 0 and group_count > 1:
                ttk.Separator(container, orient="horizontal").grid(
                    row=content_row,
                    column=0,
                    columnspan=3,
                    sticky="ew",
                    padx=6,
                    pady=(4, 10),
                )
                content_row += 1

            for field in fields_in_group:
                value = getattr(self.obj, field.name)
                label_frame = ttk.Frame(container)
                label_frame.grid(row=content_row, column=0, sticky="nw", padx=(6, 10), pady=6)
                ttk.Label(label_frame, text=safe_text(field_label(field.name, self.lang)), style="FieldLabel.TLabel").pack(anchor="w")
                status_key = _field_status(self.section_key, field.name, self.obj, self.display_mode_getter()) + "_badge"
                ttk.Label(label_frame, text=safe_text(tr(status_key, self.lang)), style="Muted.TLabel").pack(anchor="w", pady=(2, 0))

                editor_frame = ttk.Frame(container)
                editor_frame.grid(row=content_row, column=1, sticky="ew", padx=4, pady=6)
                actions_frame = ttk.Frame(container)
                actions_frame.grid(row=content_row, column=2, sticky="ne", padx=(8, 6), pady=6)

                var, widget = self._make_editor(editor_frame, field.name, value)
                self._remember_logged_value(field.name, var.get())

                if isinstance(value, list):
                    ttk.Label(editor_frame, text=tr("list_hint", self.lang), style="Muted.TLabel").pack(anchor="w", pady=(4, 0))

                path_kind = _path_kind(field.name, value)
                if path_kind == "dir":
                    ttk.Button(actions_frame, text=tr("choose_directory", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._choose_dir(fn, v)).pack(side="left", padx=2)
                    ttk.Button(actions_frame, text=tr("open_location", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._open_field_location(fn, v)).pack(side="left", padx=2)
                    ttk.Label(editor_frame, text=tr("path_hint_dir", self.lang), style="Muted.TLabel", wraplength=720, justify="left").pack(anchor="w", pady=(4, 0))
                elif path_kind == "input_file":
                    ttk.Button(actions_frame, text=tr("choose_file", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._choose_file(fn, v, save=False)).pack(side="left", padx=2)
                    ttk.Button(actions_frame, text=tr("open_location", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._open_field_location(fn, v)).pack(side="left", padx=2)
                    ttk.Label(editor_frame, text=tr("path_hint_file", self.lang), style="Muted.TLabel", wraplength=720, justify="left").pack(anchor="w", pady=(4, 0))
                elif path_kind == "output_file":
                    ttk.Button(actions_frame, text=tr("choose_output_file", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._choose_file(fn, v, save=True)).pack(side="left", padx=2)
                    ttk.Button(actions_frame, text=tr("open_location", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._open_field_location(fn, v)).pack(side="left", padx=2)
                    ttk.Label(editor_frame, text=tr("path_hint_file", self.lang), style="Muted.TLabel", wraplength=720, justify="left").pack(anchor="w", pady=(4, 0))
                elif _looks_like_color(field.name, value):
                    ttk.Button(actions_frame, text=tr("pick_color", self.lang), style="Small.TButton", command=lambda fn=field.name, v=var: self._pick_color(fn, v)).pack(side="left", padx=2)
                    swatch = tk.Label(actions_frame, width=2, background=str(value), relief="solid", borderwidth=1)
                    swatch.pack(side="left", padx=(6, 0), ipady=8)
                    var.trace_add("write", lambda *_a, v=var, s=swatch: s.configure(background=v.get() or "#FFFFFF"))

                declared_type = self.type_hints.get(field.name, type(value))
                self.vars[field.name] = (var, declared_type, value)
                self.widgets[field.name] = widget
                self.editor_frames[field.name] = editor_frame
                self.action_frames[field.name] = actions_frame
                content_row += 1

        container.columnconfigure(1, weight=1)
        self._attach_run_duration_helpers()

    def write_back(self):
        for name, (var, typ, original_value) in self.vars.items():
            try:
                value = _parse_by_type(var.get(), typ, original_value)
            except Exception as exc:
                label = field_label(name, self.lang)
                raise ValueError(f"Invalid value for '{label}': {exc}") from exc
            setattr(self.obj, name, value)
