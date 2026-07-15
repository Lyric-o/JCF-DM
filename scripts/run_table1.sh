#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/synthetic"
OUT="${ROOT}/outputs/table1"
mkdir -p "${OUT}" "${ROOT}/.mplconfig"
export MPLCONFIGDIR="${ROOT}/.mplconfig"
cd "${OUT}"

python "${SRC}/uninformative_censoring_flow.py" \
  --n_train 2048 --n_eval 1024 --mc_runs 50 --mc_runs_extra 30 \
  --max_epochs 400 --patience 50 --K 4 --suffix table1_joint
python "${SRC}/comparison_aft_uninf.py" \
  --n_train 2048 --n_eval 1024 --tag table1

for scenario in lognormal_base heavy_tail_t cox_t_lognormal_c cox_both; do
  mc=30
  if [[ "${scenario}" == "lognormal_base" ]]; then mc=50; fi
  python "${SRC}/comparison_ausset.py" --scenario "${scenario}" \
    --mc_runs "${mc}" --n_train 2048 --n_eval 1024 \
    --max_epochs 400 --patience 50 --K 4 --tag table1
done

python "${SRC}/comparison_lognormal_mixture.py" --scenario all \
  --n_train 2048 --n_eval 1024 --K_mix 4 --max_epochs 400 --tag table1
python "${SRC}/comparison_binned_km.py" --scenario all \
  --n_train 2048 --n_eval 1024 --n_bins 10 --tag table1

echo "Table 1 artifacts written to ${OUT}"
