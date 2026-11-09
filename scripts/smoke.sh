#!/usr/bin/env bash
# End-to-end infrastructure check; these tiny runs are not research results.
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
"$ant_sim_python" -m ant_homing collect --scenes configs/scenes-smoke.json --locations 8 --seed 7 --output "$ant_sim_output/data"
"$ant_sim_python" -m ant_homing train --data "$ant_sim_output/data" --epochs 2 --seed 7 --output "$ant_sim_output/model"
"$ant_sim_python" -m ant_homing evaluate --data "$ant_sim_output/data" --checkpoint "$ant_sim_output/model/model.pt" --output "$ant_sim_output/retrieval.json"
"$ant_sim_python" -m ant_homing demo --method raw --output "$ant_sim_output/visual-demo"
"$ant_sim_python" -m ant_homing demo --method cnn --checkpoint "$ant_sim_output/model/model.pt" --output "$ant_sim_output/cnn-demo"
"$ant_sim_python" -m ant_homing benchmark --seeds 0,1 --noise configs/noise-mild.json --output "$ant_sim_output/benchmark"
echo "Smoke workflow completed. Inspect $ant_sim_output for artifacts."
