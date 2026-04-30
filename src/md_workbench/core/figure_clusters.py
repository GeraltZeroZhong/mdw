from __future__ import annotations

from pathlib import Path


STABILITY_COMPACTION_CLUSTER = "stability_compaction"
INTERACTION_NETWORKS_CLUSTER = "interaction_networks"
STRUCTURE_POSE_CLUSTER = "structure_pose"
CONFORMATIONAL_LANDSCAPE_CLUSTER = "conformational_landscape"
KINETICS_CLUSTER = "kinetics"
ENERGETICS_CLUSTER = "energetics"
MISC_CLUSTER = "miscellaneous"

BUNDLE_CLUSTER_ORDER = [
    STABILITY_COMPACTION_CLUSTER,
    INTERACTION_NETWORKS_CLUSTER,
    STRUCTURE_POSE_CLUSTER,
    CONFORMATIONAL_LANDSCAPE_CLUSTER,
    KINETICS_CLUSTER,
    ENERGETICS_CLUSTER,
    MISC_CLUSTER,
]

_BASIC_STEM_TO_CLUSTER = {
    "rmsd_replot_protein": STABILITY_COMPACTION_CLUSTER,
    "rmsd_replot_ligand": STABILITY_COMPACTION_CLUSTER,
    "min_distance_combined": STABILITY_COMPACTION_CLUSTER,
    "radius_of_gyration_combined": STABILITY_COMPACTION_CLUSTER,
    "buried_surface_combined": STABILITY_COMPACTION_CLUSTER,
    "sasa_complex_protein_combined": STABILITY_COMPACTION_CLUSTER,
    "ligand_sasa_combined": STABILITY_COMPACTION_CLUSTER,
    "convergence_block_heatmap": STABILITY_COMPACTION_CLUSTER,
    "replicate_consistency_boxplot": STABILITY_COMPACTION_CLUSTER,
    "replicate_consistency_zscore_heatmap": STABILITY_COMPACTION_CLUSTER,
    "contact_count_combined": INTERACTION_NETWORKS_CLUSTER,
    "hbond_count_combined": INTERACTION_NETWORKS_CLUSTER,
    "salt_bridge_count_combined": INTERACTION_NETWORKS_CLUSTER,
    "contact_occupancy_top20": INTERACTION_NETWORKS_CLUSTER,
    "hbond_residue_occupancy_top20": INTERACTION_NETWORKS_CLUSTER,
    "salt_bridge_residue_occupancy": INTERACTION_NETWORKS_CLUSTER,
    "contact_replicate_heatmap": INTERACTION_NETWORKS_CLUSTER,
    "interaction_fingerprint_heatmap": INTERACTION_NETWORKS_CLUSTER,
    "rmsf_ca_combined": STRUCTURE_POSE_CLUSTER,
    "dssp_fractions_combined": STRUCTURE_POSE_CLUSTER,
    "dssp_residue_occupancy_combined": STRUCTURE_POSE_CLUSTER,
    "ligand_com_distance_combined": STRUCTURE_POSE_CLUSTER,
    "ligand_orientation_angle_combined": STRUCTURE_POSE_CLUSTER,
}

_BASIC_PREFIX_TO_CLUSTER = {
    "key_contact_distance_": INTERACTION_NETWORKS_CLUSTER,
}

_ADVANCED_DIR_TO_CLUSTER = {
    "pca": CONFORMATIONAL_LANDSCAPE_CLUSTER,
    "tica": CONFORMATIONAL_LANDSCAPE_CLUSTER,
    "clustering": CONFORMATIONAL_LANDSCAPE_CLUSTER,
    "snapshots": CONFORMATIONAL_LANDSCAPE_CLUSTER,
    "msm": KINETICS_CLUSTER,
}

_CLUSTER_DATA_REL_DIRS = {
    STABILITY_COMPACTION_CLUSTER: (
        Path("basic") / "combined",
    ),
    INTERACTION_NETWORKS_CLUSTER: (
        Path("basic") / "combined",
        Path("waterbridge") / "combined",
    ),
    STRUCTURE_POSE_CLUSTER: (
        Path("basic") / "combined",
    ),
    CONFORMATIONAL_LANDSCAPE_CLUSTER: (
        Path("advanced") / "pca",
        Path("advanced") / "tica",
        Path("advanced") / "clustering",
        Path("advanced") / "snapshots",
    ),
    KINETICS_CLUSTER: (
        Path("advanced") / "msm",
    ),
    ENERGETICS_CLUSTER: (
        Path("mmgbsa") / "combined",
        Path("mmgbsa"),
    ),
    MISC_CLUSTER: (
        Path("basic") / "combined",
        Path("waterbridge") / "combined",
        Path("advanced") / "pca",
        Path("advanced") / "tica",
        Path("advanced") / "clustering",
        Path("advanced") / "snapshots",
        Path("advanced") / "msm",
        Path("mmgbsa") / "combined",
        Path("mmgbsa"),
    ),
}


def classify_bundle_figure(source_path: Path, source_root: Path, source_kind: str) -> tuple[str, Path, str]:
    rel = source_path.relative_to(source_root)
    if source_kind == "basic":
        stem = source_path.stem
        cluster = _BASIC_STEM_TO_CLUSTER.get(stem)
        if cluster is None:
            cluster = next((value for prefix, value in _BASIC_PREFIX_TO_CLUSTER.items() if stem.startswith(prefix)), MISC_CLUSTER)
        nested = rel.parts[1:-1] if rel.parts[:1] == ("combined",) else rel.parts[:-1]
        target_rel = Path(*nested, rel.name) if nested else Path(rel.name)
        return cluster, target_rel, "basic"

    if source_kind == "waterbridge":
        nested = rel.parts[1:-1] if rel.parts[:1] == ("combined",) else rel.parts[:-1]
        target_rel = Path(*nested, rel.name) if nested else Path(rel.name)
        return INTERACTION_NETWORKS_CLUSTER, target_rel, "waterbridge"

    if source_kind == "advanced":
        family = rel.parts[0] if rel.parts else ""
        cluster = _ADVANCED_DIR_TO_CLUSTER.get(family, MISC_CLUSTER)
        nested = rel.parts[1:-1]
        target_rel = Path(*nested, rel.name) if nested else Path(rel.name)
        return cluster, target_rel, f"advanced_{family or 'root'}"

    if source_kind == "mmgbsa":
        family = rel.parts[0] if rel.parts else ""
        nested = rel.parts[1:-1] if family == "combined" else rel.parts[:-1]
        target_rel = Path(*nested, rel.name) if nested else Path(rel.name)
        return ENERGETICS_CLUSTER, target_rel, f"mmgbsa_{family or 'root'}"

    nested = rel.parts[:-1]
    target_rel = Path(*nested, rel.name) if nested else Path(rel.name)
    return MISC_CLUSTER, target_rel, source_kind or "misc"


def bundled_data_candidates_for_figure(
    base_path: Path,
    *,
    figures_dir_name: str = "figures_combined",
    data_dir_name: str = "process_data",
) -> list[Path]:
    parts = base_path.parts
    if figures_dir_name not in parts:
        return []

    idx = parts.index(figures_dir_name)
    bundle_root = Path(*parts[:idx])
    rel_parts = parts[idx + 1 : -1]
    if not rel_parts:
        return []

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        if path not in seen:
            seen.add(path)
            candidates.append(path)

    add(bundle_root / data_dir_name / Path(*rel_parts))

    cluster = rel_parts[0]
    nested = rel_parts[1:]
    nested_path = Path(*nested) if nested else None
    for rel_dir in _CLUSTER_DATA_REL_DIRS.get(cluster, ()):
        if nested_path is not None:
            add(bundle_root / data_dir_name / rel_dir / nested_path)
        add(bundle_root / data_dir_name / rel_dir)

    return candidates
