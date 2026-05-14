# MD Workbench

Internal molecular dynamics workflow utilities for the Zhong lab. The PyPI package is `mdw-zhong`, and the Python import package is `md_workbench`.

Use `environment.yml` or `scripts/install_mdw_env.sh` to prepare the conda runtime, then run `mdw gui`, `mdw run --config default_config.json`, `mdw plot --config default_config.json`, or `mdw report --config default_config.json`.

For short-peptide or cyclic-peptide ligands supplied as PDB, use `docking.ligand_input_mode = "pdb"` with `docking.docking_mode = "skip"`.
