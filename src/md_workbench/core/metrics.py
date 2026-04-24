from __future__ import annotations

import numpy as np


def get_time_ns_from_nframes(n_frames: int, timestep_ps: float, dcd_interval_steps: int):
    dt_ps = timestep_ps * dcd_interval_steps
    return np.arange(n_frames) * dt_ps / 1000.0


def aggregate_metric_rows(replica_results, key_name: str, value_name: str, out_field_name: str):
    per_replica_values = []
    labels = set()
    total_replicas = len(replica_results)
    for result in replica_results:
        local = {row["protein_residue"]: row[value_name] for row in result[key_name]}
        per_replica_values.append(local)
        labels.update(local)

    rows = []
    for residue in labels:
        vals = np.asarray([local.get(residue, 0.0) for local in per_replica_values], dtype=float)
        present_count = sum(1 for local in per_replica_values if residue in local)
        rows.append(
            {
                "protein_residue": residue,
                f"{out_field_name}_mean": float(vals.mean()),
                f"{out_field_name}_sd": float(vals.std(ddof=1) if len(vals) > 1 else 0.0),
                "n_replicas": int(total_replicas),
                "n_present_replicas": int(present_count),
            }
        )
    rows.sort(key=lambda x: (-float(x[f"{out_field_name}_mean"]), str(x["protein_residue"])))
    return rows
