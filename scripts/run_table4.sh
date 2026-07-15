#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${ROOT}/src/real_data"
OUT="${ROOT}/outputs/table4"
mkdir -p "${OUT}"
export JCFDM_DATA_DIR="${JCFDM_DATA_DIR:-${ROOT}/data}"
python "${ROOT}/scripts/validate_data.py" --data-dir "${JCFDM_DATA_DIR}"
cd "${OUT}"

datasets=(support gbsg)
if [[ -f "${JCFDM_DATA_DIR}/seer.csv" ]]; then
  datasets=(seer "${datasets[@]}")
else
  echo "SEER is absent; reproducing the public SUPPORT and Rotterdam--GBSG columns only."
fi

for dataset in "${datasets[@]}"; do
  python "${SRC}/real_data_pure.py" --dataset "${dataset}" --mc_runs 5 \
    --methods joint_flow,tonly_flow,coxph,rsf,deephit \
    --max_epochs 600 --hidden 64 --test_frac 0.25 --out "table4_${dataset}"
done

echo "Table 4 artifacts written to ${OUT}"
