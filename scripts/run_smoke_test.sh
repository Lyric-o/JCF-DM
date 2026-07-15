#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/synthetic"
OUT="${ROOT}/outputs/smoke_test"
mkdir -p "${OUT}" "${ROOT}/.mplconfig"
export MPLCONFIGDIR="${ROOT}/.mplconfig"

python -m compileall -q "${ROOT}/src" "${ROOT}/scripts"
cd "${OUT}"
python "${SRC}/uninformative_censoring_flow.py" \
  --n_train 64 --n_eval 32 --mc_runs 1 --mc_runs_extra 1 \
  --max_epochs 2 --patience 1 --K 1 --suffix smoke
python "${SRC}/informative_censoring_impl.py" --family spline --rho 1.0 \
  --mc_runs 1 --n_train 64 --n_eval 32 --K_eps 2 --max_epochs 2 \
  --out smoke_informative
python "${SRC}/comparison_lognormal_mixture_ic.py" --rho 1.0 --mc_runs 1 \
  --n_train 64 --n_eval 32 --K_mix 2 --max_epochs 2 --tag smoke
python "${SRC}/comparison_deep_lognormal_frank.py" --rho 1.0 --mc_runs 1 \
  --n_train 64 --n_eval 32 --max_epochs 2 --tag smoke
python "${SRC}/comparison_binned_km_ic.py" --rho 1.0 --mc_runs 1 \
  --n_train 64 --n_eval 32 --n_bins 4 --tag smoke

echo "Smoke test completed; artifacts written to ${OUT}"
