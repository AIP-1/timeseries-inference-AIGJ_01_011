#!/bin/bash
# Mac의 코드를 Jetson에 올린다(코드만. 데이터·모델·결과·.env 는 건드리지 않음).
#   ./sync.sh          올리기
#   ./sync.sh pull     Jetson의 결과 보고서·수집 현황을 Mac으로 가져오기
set -e
cd "$(dirname "$0")"
if [ "$1" = "pull" ]; then
  mkdir -p pipeline/results collector/reports
  rsync -a --include='*/' --include='*.md' --include='*.json' --include='*.png' --exclude='*' \
    jetson:timeseries/pipeline/results/ pipeline/results/
  rsync -a jetson:timeseries/reports/ collector/reports/
  echo "pulled results"
  exit 0
fi
rsync -a collector/collector.py collector/config.yaml collector/status.py collector/report.py \
  collector/test_format_change.py collector/test_broker_restart.py collector/export_influx.py collector/compare_gaps.py jetson:timeseries/
rsync -a collector/collector.service jetson:.config/systemd/user/
rsync -a --exclude='results/' --exclude='data/' --exclude='artifacts/' --exclude='__pycache__/' \
  pipeline/ jetson:timeseries/pipeline/
echo "synced to jetson"
