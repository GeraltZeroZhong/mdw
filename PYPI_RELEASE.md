# PyPI Release Checklist

This project publishes to PyPI as `mdw-zhong`.

1. Verify the version in `pyproject.toml` and `src/md_workbench/__init__.py`.
2. Build and check the package:

```bash
python -m pip install -e ".[release]"
rm -rf dist build *.egg-info src/*.egg-info
python -m build
python -m twine check dist/*
```

3. Inspect the generated files:

```bash
tar -tzf dist/*.tar.gz | sed -n '1,120p'
python -m zipfile -l dist/*.whl | sed -n '1,120p'
```

4. Upload when ready:

```bash
python -m twine upload dist/*
```

The wheel installs the Python package and command-line entry points. The full scientific runtime should still be installed from `environment.yml`.

After publishing, test the PyPI install path in a conda environment:

```bash
mamba env create -f environment.yml
mamba run -n mdw python -m pip install --upgrade mdw-zhong
mamba run -n mdw mdw --help
```
