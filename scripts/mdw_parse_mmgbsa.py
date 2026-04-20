#!/usr/bin/env python3
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from md_workbench.postprocess.mmgbsa import parse_mmpbsa_results

input_path = sys.argv[1] if len(sys.argv) > 1 else "mmpbsa_FINAL_RESULTS.dat"
out = parse_mmpbsa_results(input_path)
print(out)
