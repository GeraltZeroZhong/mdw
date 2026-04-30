from __future__ import annotations

DEFAULT_PLOT_STYLE_PRESET = "palette1"
CUSTOM_PLOT_STYLE_PRESET = "custom"
PLOT_STYLE_PRESET_CHOICES = ("palette1", "palette2", "zero", "nature", CUSTOM_PLOT_STYLE_PRESET)
AXES_TEXT_COLOR = "#000000"

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
    "spine_color": AXES_TEXT_COLOR,
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

DEEP_BLUE_MAGENTA_COLORS = {
    "spine_color": AXES_TEXT_COLOR,
    "grid_color": "#B9D2F3",
    "mean_line_color": "#496CCE",
    "band_color": "#E6F0FE",
    "protein_color": "#82AAE7",
    "ligand_color": "#F7A6BF",
    "distance_color": "#496CCE",
    "temperature_color": "#82AAE7",
    "density_color": "#C03F67",
    "potential_energy_color": "#C03F67",
    "total_energy_color": "#82AAE7",
    "bar_color": "#82AAE7",
    "accent_color": "#E46B90",
}

ZERO_BLUE_CYAN_PINK_COLORS = {
    "spine_color": AXES_TEXT_COLOR,
    "grid_color": "#DCEFFC",
    "mean_line_color": "#19324D",
    "band_color": "#DCEFFC",
    "protein_color": "#0F6C7A",
    "ligand_color": "#2FA7C9",
    "distance_color": "#3B6FF5",
    "temperature_color": "#0F6C7A",
    "density_color": "#A8C8C1",
    "potential_energy_color": "#19324D",
    "total_energy_color": "#2FA7C9",
    "bar_color": "#2FA7C9",
    "accent_color": "#E66AA3",
}

NATURE_RED_BLACK_COLORS = {
    "spine_color": AXES_TEXT_COLOR,
    "grid_color": "#D9D9D9",
    "mean_line_color": "#E64B35",
    "band_color": "#FCE5E0",
    "protein_color": "#000000",
    "ligand_color": "#E64B35",
    "distance_color": "#DC0000",
    "temperature_color": "#4D4D4D",
    "density_color": "#7F7F7F",
    "potential_energy_color": "#000000",
    "total_energy_color": "#DC0000",
    "bar_color": "#E64B35",
    "accent_color": "#DC0000",
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

DEEP_BLUE_MAGENTA_CATEGORICAL_PALETTE = (
    "#1A318B",
    "#496CCE",
    "#82AAE7",
    "#B9D2F3",
    "#E6F0FE",
    "#F9DBE5",
    "#F7A6BF",
    "#E46B90",
    "#C03F67",
    "#9A133D",
)

ZERO_BLUE_CYAN_PINK_CATEGORICAL_PALETTE = (
    "#19324D",
    "#0F6C7A",
    "#2FA7C9",
    "#3B6FF5",
    "#A8C8C1",
    "#DCEFFC",
    "#F6C8D7",
    "#FF9BB8",
    "#E66AA3",
)

NATURE_RED_BLACK_CATEGORICAL_PALETTE = (
    "#000000",
    "#E64B35",
    "#4D4D4D",
    "#DC0000",
    "#7F7F7F",
    "#B22222",
    "#BDBDBD",
    "#8B0000",
    "#F39B7F",
    "#2B2B2B",
)

SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP = "mdw_teal_purple"
SCIENTIFIC_TEAL_PINK_DIVERGING_CMAP = "mdw_teal_pink_diverging"

PLOT_STYLE_PRESETS = {
    "palette1": {
        "colors": SCIENTIFIC_TEAL_PINK_COLORS,
        "categorical_palette": SCIENTIFIC_TEAL_PINK_CATEGORICAL_PALETTE,
        "cmap_continuous": SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP,
    },
    "palette2": {
        "colors": DEEP_BLUE_MAGENTA_COLORS,
        "categorical_palette": DEEP_BLUE_MAGENTA_CATEGORICAL_PALETTE,
        "cmap_continuous": SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP,
    },
    "zero": {
        "colors": ZERO_BLUE_CYAN_PINK_COLORS,
        "categorical_palette": ZERO_BLUE_CYAN_PINK_CATEGORICAL_PALETTE,
        "cmap_continuous": SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP,
    },
    "nature": {
        "colors": NATURE_RED_BLACK_COLORS,
        "categorical_palette": NATURE_RED_BLACK_CATEGORICAL_PALETTE,
        "cmap_continuous": SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP,
    },
}

PLOT_STYLE_PRESET_ALIASES = {
    "scientific_teal_pink": "palette1",
    "deep_blue_magenta": "palette2",
    "blue_magenta": "palette2",
    "zero_palette": "zero",
    "zero_color": "zero",
    "zero_colours": "zero",
    "zero_colors": "zero",
    "nature_red_black": "nature",
    "npg_red_black": "nature",
    "nature_style": "nature",
}


def default_plot_formats() -> list[str]:
    return list(DEFAULT_PLOT_FORMATS)


def default_categorical_palette() -> list[str]:
    return list(SCIENTIFIC_TEAL_PINK_CATEGORICAL_PALETTE)


def canonical_plot_style_palette(name: str) -> str:
    key = str(name).strip().lower()
    return PLOT_STYLE_PRESET_ALIASES.get(key, key)


def plot_style_palette_settings(name: str) -> dict | None:
    return PLOT_STYLE_PRESETS.get(canonical_plot_style_palette(name))


def apply_plot_style_palette(style) -> None:
    palette_name = canonical_plot_style_palette(getattr(style, "color_palette", ""))
    preset = plot_style_palette_settings(palette_name)
    if preset is None:
        return
    style.color_palette = palette_name
    for field_name, value in preset["colors"].items():
        setattr(style, field_name, value)
    style.categorical_palette = list(preset["categorical_palette"])
    style.cmap_continuous = preset["cmap_continuous"]


def infer_plot_style_palette(style) -> str:
    for name, preset in PLOT_STYLE_PRESETS.items():
        colors_match = all(getattr(style, field_name, None) == value for field_name, value in preset["colors"].items())
        palette_match = list(getattr(style, "categorical_palette", [])) == list(preset["categorical_palette"])
        cmap_match = getattr(style, "cmap_continuous", None) == preset["cmap_continuous"]
        if colors_match and palette_match and cmap_match:
            return name
    return CUSTOM_PLOT_STYLE_PRESET
