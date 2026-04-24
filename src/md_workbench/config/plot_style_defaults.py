from __future__ import annotations

DEFAULT_PLOT_STYLE_PRESET = "scientific_teal_pink"

DEFAULT_PLOT_FORMATS = ("png", "svg")

DEFAULT_PLOT_RENDERING = {
    "dpi": 600,
    "font_family": "DejaVu Sans",
    "title_size": 10.6,
    "label_size": 9.2,
    "tick_size": 8.2,
    "legend_size": 7.8,
    "line_width": 1.8,
    "thin_line_width": 0.95,
    "marker_size": 4.6,
    "axes_line_width": 0.9,
    "grid_alpha": 0.1,
    "show_grid": True,
    "transparent_background": False,
    "use_minor_ticks": False,
}

SCIENTIFIC_TEAL_PINK_COLORS = {
    "spine_color": "#3F5658",
    "grid_color": "#DCEDEE",
    "mean_line_color": "#2F6F70",
    "band_color": "#CFEAF1",
    "protein_color": "#96CCCB",
    "ligand_color": "#F6CAE5",
    "distance_color": "#C4A5DE",
    "temperature_color": "#96CCCB",
    "density_color": "#B883D4",
    "potential_energy_color": "#B883D4",
    "total_energy_color": "#96CCCB",
    "bar_color": "#96CCCB",
    "accent_color": "#F6CAE5",
}

SCIENTIFIC_TEAL_PINK_CATEGORICAL_PALETTE = (
    "#96CCCB",
    "#F6CAE5",
    "#C4A5DE",
    "#B883D4",
    "#CFEAF1",
    "#6FAEAD",
    "#E8A8D1",
    "#A7DCE5",
    "#A985CF",
    "#7A8FBE",
)

SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP = "mdw_teal_purple"
SCIENTIFIC_TEAL_PINK_DIVERGING_CMAP = "mdw_teal_pink_diverging"


def default_plot_formats() -> list[str]:
    return list(DEFAULT_PLOT_FORMATS)


def default_categorical_palette() -> list[str]:
    return list(SCIENTIFIC_TEAL_PINK_CATEGORICAL_PALETTE)
