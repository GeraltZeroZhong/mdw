#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="mdw"
ENV_FILE="environment.yml"
PACKAGE_NAME="mdw-zhong"
INSTALL_PACKAGE="1"
DEFAULT_ENV_URL="https://raw.githubusercontent.com/GeraltZeroZhong/mdw/main/environment.yml"

usage() {
  cat <<'EOF'
Install or update the MD Workbench conda environment.

Usage:
  scripts/install_mdw_env.sh [options]

Options:
  --env-name NAME      Conda environment name. Default: mdw
  --env-file PATH_URL  environment.yml path or URL. Default: ./environment.yml,
                       falling back to the GitHub main branch raw URL.
  --package NAME       PyPI package to install inside the environment. Default: mdw-zhong
  --no-package         Only create/update the conda environment.
  -h, --help           Show this help.

Examples:
  scripts/install_mdw_env.sh
  scripts/install_mdw_env.sh --env-name mdw-test
  scripts/install_mdw_env.sh --env-file https://raw.githubusercontent.com/GeraltZeroZhong/mdw/main/environment.yml
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --package)
      PACKAGE_NAME="$2"
      shift 2
      ;;
    --no-package)
      INSTALL_PACKAGE="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if command -v mamba >/dev/null 2>&1; then
  ENV_TOOL="mamba"
elif command -v conda >/dev/null 2>&1; then
  ENV_TOOL="conda"
else
  echo "Neither mamba nor conda was found on PATH." >&2
  echo "Install Miniforge/Mambaforge or Miniconda first, then rerun this script." >&2
  exit 1
fi

TMP_DIR=""
cleanup() {
  if [[ -n "$TMP_DIR" && -d "$TMP_DIR" ]]; then
    rm -rf "$TMP_DIR"
  fi
}
trap cleanup EXIT

resolve_env_file() {
  local source="$1"
  if [[ -f "$source" ]]; then
    printf '%s\n' "$source"
    return
  fi

  if [[ "$source" != http://* && "$source" != https://* ]]; then
    source="$DEFAULT_ENV_URL"
  fi

  TMP_DIR="$(mktemp -d)"
  local target="$TMP_DIR/environment.yml"
  if command -v curl >/dev/null 2>&1; then
    curl -fsSL "$source" -o "$target"
  elif command -v wget >/dev/null 2>&1; then
    wget -q "$source" -O "$target"
  else
    echo "Need curl or wget to download environment.yml from: $source" >&2
    exit 1
  fi
  printf '%s\n' "$target"
}

env_exists() {
  "$ENV_TOOL" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"
}

WORK_ENV_FILE="$(resolve_env_file "$ENV_FILE")"

if env_exists; then
  echo "Updating conda environment: $ENV_NAME"
  "$ENV_TOOL" env update -n "$ENV_NAME" -f "$WORK_ENV_FILE"
else
  echo "Creating conda environment: $ENV_NAME"
  "$ENV_TOOL" env create -n "$ENV_NAME" -f "$WORK_ENV_FILE"
fi

if [[ "$INSTALL_PACKAGE" == "1" ]]; then
  echo "Installing PyPI package inside $ENV_NAME: $PACKAGE_NAME"
  "$ENV_TOOL" run -n "$ENV_NAME" python -m pip install --upgrade "$PACKAGE_NAME"
fi

echo
echo "Done. Activate with:"
echo "  conda activate $ENV_NAME"
echo
echo "Quick check:"
echo "  $ENV_TOOL run -n $ENV_NAME mdw --help"
