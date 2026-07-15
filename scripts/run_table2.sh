#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/synthetic"
OUT="${ROOT}/outputs/table2"
mkdir -p "${OUT}" "${ROOT}/.mplconfig"
export MPLCONFIGDIR="${ROOT}/.mplconfig"
cd "${OUT}"

python "${SRC}/informative_censoring_impl.py" --family spline --rho 1.0 \
  --mc_runs 50 --n_train 2048 --n_eval 1024 --K_eps 32 \
  --max_epochs 400 --out table2_joint
python "${SRC}/comparison_classical_ic.py" --rho 1.0 --mc_runs 50 \
  --n_train 2048 --n_eval 1024 --K_eps 32 --tag table2
python "${SRC}/ausset_informative.py" --dgp dgp1 --rho 1.0 --mc_runs 50 \
  --n_train 2048 --n_eval 1024 --max_epochs 400 --patience 50 --suffix table2
python "${SRC}/comparison_lognormal_mixture_ic.py" --rho 1.0 --mc_runs 50 \
  --n_train 2048 --n_eval 1024 --K_mix 4 --max_epochs 400 --tag table2
python "${SRC}/comparison_deep_lognormal_frank.py" --rho 1.0 --mc_runs 50 \
  --n_train 2048 --n_eval 1024 --max_epochs 400 --tag table2
python "${SRC}/comparison_binned_km_ic.py" --rho 1.0 --mc_runs 50 \
  --n_train 2048 --n_eval 1024 --n_bins 10 --tag table2

echo "Table 2 artifacts written to ${OUT}"
