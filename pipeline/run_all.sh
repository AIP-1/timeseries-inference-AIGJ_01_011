#!/bin/bash
# 합성 데이터 생성부터 과제 2~6 결과까지 한 번에 다시 만든다.
#   ./run_all.sh            전부
#   ./run_all.sh t4 t5      일부 과제만 (데이터·모델은 이미 있다고 가정)
set -e
cd "$(dirname "$0")"
PY=../.venv/bin/python
export PYTHONWARNINGS="ignore::UserWarning"
steps=("$@")
[ ${#steps[@]} -eq 0 ] && steps=(synth t2 t3 t4 t5)
mkdir -p results
for s in "${steps[@]}"; do
  echo "=== $s  $(date +%H:%M:%S)"
  case $s in
    synth) $PY synth/generate.py --out data/synth | head -1 ;;
    t2) $PY -u tasks/t2_baselines.py > results/t2.log 2>&1 ;;
    t3) $PY -u tasks/t3_forecast.py > results/t3.log 2>&1 ;;
    t4) $PY -u tasks/t4_anomaly.py > results/t4.log 2>&1 ;;
    t5) $PY -u tasks/t5_state.py > results/t5.log 2>&1 ;;
    t6) $PY -u tasks/t6_chronos.py > results/t6.log 2>&1 ;;
    *) echo "unknown step $s"; exit 1 ;;
  esac
done
echo "=== done $(date +%H:%M:%S)"
