#!/usr/bin/env bash
# End-to-end infrastructure smoke check using the saved Level 1 view models.
set -euo pipefail
ant_sim_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ant_sim_root"
ant_sim_output="${1:-artifacts/smoke}"
if [[ -e "$ant_sim_output" ]]; then
  echo "Choose a new output directory; refusing to overwrite $ant_sim_output" >&2
  exit 1
fi
mkdir -p "$ant_sim_output"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
ant_sim_python="$ant_sim_root/.venv/bin/python"
"$ant_sim_python" -m ant_homing render --seed 0 --output "$ant_sim_output/arena"
"$ant_sim_python" -m ant_homing demo --method normalized --output "$ant_sim_output/normalized-demo"
"$ant_sim_python" -m ant_homing demo --method infonce --output "$ant_sim_output/infonce-demo"
"$ant_sim_python" -m ant_homing benchmark --seeds 0,1 --noise configs/noise-mild.json --output "$ant_sim_output/benchmark"
echo "Smoke workflow completed. Inspect $ant_sim_output for artifacts."
