#!/usr/bin/env bash
# Keep managed Python and caches within the project on restricted cloud machines.
set -euo pipefail
ant_sim_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ant_sim_root"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ant_sim_root/.cache/uv}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-$ant_sim_root/.python}"
if ! command -v uv >/dev/null 2>&1; then
  echo 'Install uv using https://docs.astral.sh/uv/getting-started/installation/ and rerun.' >&2
  exit 1
fi
uv sync --frozen --extra ml --extra dev
