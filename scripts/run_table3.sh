#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/real_data"
OUT="${ROOT}/outputs/table3"
mkdir -p "${OUT}"
export JCFDM_DATA_DIR="${JCFDM_DATA_DIR:-${ROOT}/data}"
python "${ROOT}/scripts/validate_data.py" --data-dir "${JCFDM_DATA_DIR}"
cd "${OUT}"

for dataset in support gbsg; do
  for rho in 0.0 1.0; do
    python "${SRC}/real_data_semisynthetic.py" --dataset "${dataset}" \
      --rho "${rho}" --mc_runs 10 --methods joint_flow,tonly_flow,coxph \
      --max_epochs 800 --hidden 64 --out "table3_${dataset}"
  done
done

echo "Table 3 artifacts written to ${OUT}"
