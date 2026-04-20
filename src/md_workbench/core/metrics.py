from __future__ import annotations

from collections import defaultdict
import numpy as np


def get_time_ns_from_nframes(n_frames: int, timestep_ps: float, dcd_interval_steps: int):
    dt_ps = timestep_ps * dcd_interval_steps
    return np.arange(n_frames) * dt_ps / 1000.0


def aggregate_metric_rows(replica_results, key_name: str, value_name: str, out_field_name: str):
    values = defaultdict(list)
    for result in replica_results:
        local = {row["protein_residue"]: row[value_name] for row in result[key_name]}
        for residue, value in local.items():
            values[residue].append(value)

    rows = []
    for residue, vals in values.items():
        vals = np.asarray(vals, dtype=float)
        rows.append(
            {
                "protein_residue": residue,
                f"{out_field_name}_mean": float(vals.mean()),
                f"{out_field_name}_sd": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                "n_replicas": int(len(vals)),
            }
        )
    rows.sort(key=lambda x: x[f"{out_field_name}_mean"], reverse=True)
    return rows
