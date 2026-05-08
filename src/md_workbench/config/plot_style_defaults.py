from __future__ import annotations

DEFAULT_PLOT_STYLE_PRESET = "nature"
CUSTOM_PLOT_STYLE_PRESET = "custom"
PLOT_STYLE_PRESET_CHOICES = ("palette1", "palette2", "zero", "nature", CUSTOM_PLOT_STYLE_PRESET)
AXES_TEXT_COLOR = "#000000"

DEFAULT_PLOT_FORMATS = ("png", "svg")

DEFAULT_PLOT_RENDERING = {
    "dpi": 600,
    "font_family": "Arial",
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
    "protein_color": "#4FA7A6",
    "ligand_color": "#D95CA6",
    "distance_color": "#8F63C6",
    "temperature_color": "#4FA7A6",
    "density_color": "#7E6BC4",
    "potential_energy_color": "#7E6BC4",
    "total_energy_color": "#4FA7A6",
    "bar_color": "#4FA7A6",
    "accent_color": "#D95CA6",
}

DEEP_BLUE_MAGENTA_COLORS = {
    "spine_color": AXES_TEXT_COLOR,
    "grid_color": "#B9D2F3",
    "mean_line_color": "#496CCE",
    "band_color": "#E6F0FE",
    "protein_color": "#3F7EDB",
    "ligand_color": "#D84B78",
    "distance_color": "#496CCE",
    "temperature_color": "#3F7EDB",
    "density_color": "#C03F67",
    "potential_energy_color": "#C03F67",
    "total_energy_color": "#3F7EDB",
    "bar_color": "#3F7EDB",
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
    "density_color": "#668F88",
    "potential_energy_color": "#19324D",
    "total_energy_color": "#2FA7C9",
    "bar_color": "#2FA7C9",
    "accent_color": "#E66AA3",
}

NATURE_NPG_COLORS = {
    "spine_color": AXES_TEXT_COLOR,
    "grid_color": "#D9DDE8",
    "mean_line_color": "#E64B35",
    "band_color": "#E6F4F1",
    "protein_color": "#4DBBD5",
    "ligand_color": "#E64B35",
    "distance_color": "#00A087",
    "temperature_color": "#4DBBD5",
    "density_color": "#8491B4",
    "potential_energy_color": "#3C5488",
    "total_energy_color": "#E64B35",
    "bar_color": "#E64B35",
    "accent_color": "#DC0000",
}

SCIENTIFIC_TEAL_PINK_CATEGORICAL_PALETTE = (
    "#2F6F70",
    "#D95CA6",
    "#8F63C6",
    "#4FA7A6",
    "#C05AAE",
    "#5F8FD3",
    "#7E6BC4",
    "#B24D6E",
    "#1B8A8F",
    "#6A4C93",
)

DEEP_BLUE_MAGENTA_CATEGORICAL_PALETTE = (
    "#1A318B",
    "#496CCE",
    "#3F7EDB",
    "#7A5CCF",
    "#D84B78",
    "#B8275B",
    "#2B5BA8",
    "#8A2E67",
    "#5D7FC8",
    "#A02352",
)

ZERO_BLUE_CYAN_PINK_CATEGORICAL_PALETTE = (
    "#19324D",
    "#0F6C7A",
    "#2FA7C9",
    "#3B6FF5",
    "#668F88",
    "#E66AA3",
    "#B64B8A",
    "#245D8F",
    "#7A70D6",
    "#1F8A9E",
)

NATURE_NPG_CATEGORICAL_PALETTE = (
    "#E64B35",
    "#4DBBD5",
    "#00A087",
    "#3C5488",
    "#F39B7F",
    "#8491B4",
    "#91D1C2",
    "#DC0000",
    "#7E6148",
    "#B09C85",
)

SCIENTIFIC_TEAL_PINK_CONTINUOUS_CMAP = "mdw_teal_purple"
SCIENTIFIC_TEAL_PINK_DIVERGING_CMAP = "mdw_teal_pink_diverging"
NATURE_NPG_CONTINUOUS_CMAP = "mdw_nature_npg"

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
        "colors": NATURE_NPG_COLORS,
        "categorical_palette": NATURE_NPG_CATEGORICAL_PALETTE,
        "cmap_continuous": NATURE_NPG_CONTINUOUS_CMAP,
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
    "nature_npg": "nature",
    "nature_classic": "nature",
    "npg": "nature",
    "npg_nrc": "nature",
    "nature_red_black": "nature",
    "npg_red_black": "nature",
    "nature_style": "nature",
}


def default_plot_formats() -> list[str]:
    return list(DEFAULT_PLOT_FORMATS)


def default_categorical_palette() -> list[str]:
    return list(NATURE_NPG_CATEGORICAL_PALETTE)


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
