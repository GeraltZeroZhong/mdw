# MD Workbench

Internal molecular dynamics workflow utilities.

The PyPI package name is `mdw-zhong`; the Python import package is `md_workbench`.

## Install

The full runtime depends on conda-forge scientific packages and command-line tools such as OpenMM, RDKit, AmberTools, AutoDock Vina, Meeko, and PyMOL. For a working environment, install the conda environment first:

```bash
mamba env create -f environment.yml
mamba activate mdw
pip install mdw-zhong
```

Automated install or update from a checkout:

```bash
bash scripts/install_mdw_env.sh
```

Automated install on a machine without this repository checked out:

```bash
bash -c "$(curl -fsSL https://raw.githubusercontent.com/GeraltZeroZhong/mdw/main/scripts/install_mdw_env.sh)"
```

For local development from a checkout:

```bash
pip install -e .
```

## Commands

```bash
mdw init-config
mdw self-check --config default_config.json
mdw gui
mdw run --config default_config.json
mdw plot --config default_config.json
mdw report --config default_config.json
```

Compatibility command names are also installed, including `mdw-run`, `mdw-plot`, `mdw-report`, `mdw-mmgbsa`, and `mdw-self-check`.

## Packaging Note

`pip install mdw-zhong` installs the MD Workbench Python package and console commands. It intentionally does not try to install or create the full conda environment.
