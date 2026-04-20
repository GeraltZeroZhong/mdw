from __future__ import annotations

import re
import numpy as np

from ...core import require_nonempty_file


def parse_md_log(log_path):
    require_nonempty_file(
        log_path,
        label="日志文件 md.log",
        empty_hint="这通常表示 MD 生产阶段没有写出任何日志记录，请确认 production_steps >= log_interval。",
    )
    data = []
    with open(log_path, "r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or not re.match(r"^\d+,", text):
                continue
            parts = text.split(",")
            if len(parts) < 8:
                continue
            data.append(
                {
                    "step": int(float(parts[0])),
                    "time_ps": float(parts[1]),
                    "potential_energy_kjmol": float(parts[2]),
                    "kinetic_energy_kjmol": float(parts[3]),
                    "total_energy_kjmol": float(parts[4]),
                    "temperature_K": float(parts[5]),
                    "volume_nm3": float(parts[6]),
                    "density_gmL": float(parts[7]),
                    "speed_nsd": float(parts[8]) if len(parts) > 8 else np.nan,
                }
            )
    if not data:
        raise ValueError(f"未能从 md.log 解析到数据: {log_path}")
    return data
