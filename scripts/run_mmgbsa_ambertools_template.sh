#!/usr/bin/env bash
set -euo pipefail

# AmberTools MMPBSA.py template
#
# Usage:
#   bash run_mmgbsa_ambertools_template.sh
#
# You must prepare these files yourself first:
#   complex_solvated.prmtop   # solvated complex topology (for -sp)
#   complex.prmtop            # stripped complex topology
#   receptor.prmtop           # stripped receptor topology
#   ligand.prmtop             # stripped ligand topology
#   complex.nc                # trajectory in Amber-readable format
#
# Notes:
# - This is a TEMPLATE only.
# - It is intended for AmberTools MMPBSA.py workflow.
# - Your current OpenMM run did not automatically produce Amber topology files,
#   so this script is the downstream template after you prepare those.
#
# Amber official tutorials page lists MM-PBSA as a dedicated free-energy tutorial. See:
# https://www.ambermd.org/tutorials

INPUT="mmpbsa.in"
SOLVATED_TOPO="complex_solvated.prmtop"
COMPLEX_TOPO="complex.prmtop"
RECEPTOR_TOPO="receptor.prmtop"
LIGAND_TOPO="ligand.prmtop"
TRAJ="complex.nc"

OUT_PREFIX="mmpbsa"

MMPBSA.py \
  -O \
  -i "${INPUT}" \
  -sp "${SOLVATED_TOPO}" \
  -cp "${COMPLEX_TOPO}" \
  -rp "${RECEPTOR_TOPO}" \
  -lp "${LIGAND_TOPO}" \
  -y "${TRAJ}" \
  -o "${OUT_PREFIX}_FINAL_RESULTS.dat" \
  -eo "${OUT_PREFIX}_FINAL_RESULTS.csv"

echo "MM/GBSA 完成: ${OUT_PREFIX}_FINAL_RESULTS.dat"
