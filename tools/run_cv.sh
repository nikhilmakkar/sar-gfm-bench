#!/bin/bash
# Sequential CV queue for one backbone tag (single L40S, so runs are serial).
# Usage: tools/run_cv.sh dinov2l_frozen [start_fold]
#        tools/run_cv.sh r18 1            # resume queue from fold 1
set -u
TAG=${1:?usage: run_cv.sh <tag> [start_fold]}
START=${2:-0}
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."

for FOLD in $(seq "$START" 3); do
    CFG=configs/umbra_cv/orcnn_${TAG}_fold${FOLD}.py
    WD=work_dirs/orcnn_${TAG}_fold${FOLD}
    echo "=== [$(date +%H:%M:%S)] ${TAG} fold ${FOLD} ==="
    $PY tools/train.py "$CFG" --work-dir "$WD" > "work_dirs/${TAG}_fold${FOLD}.log" 2>&1
    RC=$?
    BEST=$(grep -o 'dota/mAP: [0-9.]*' "work_dirs/${TAG}_fold${FOLD}.log" | sort -t' ' -k2 -g | tail -1)
    echo "=== fold ${FOLD} done rc=${RC} best ${BEST:-none} ==="
done
echo "=== CV queue ${TAG} complete ==="
