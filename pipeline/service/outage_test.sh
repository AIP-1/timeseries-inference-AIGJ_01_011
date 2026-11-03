#!/bin/bash
# 과제 8 장애 시험: 데이터 공급(재생기, 실데이터면 수집기)을 10분 멈췄다가 다시 켠다.
# 기대: 멈춘 뒤 약 10분 안에 'data_gap' 경보 → 다시 켜면 몇 분 안에 경보 해제(off)·상태 ok 로 돌아옴
#   ./service/outage_test.sh [멈출 서비스=ts-replay] [멈출 분=10]
SVC=${1:-ts-replay}
MIN=${2:-10}
cd "$(dirname "$0")/.."
LOG=results/outage_test.log
api() { curl -s localhost:8000/latest | ../.venv/bin/python -c "
import json,sys; d=json.load(sys.stdin); s=d['spaces']
print(d['summary']['computed_at'][11:19], ' '.join(f\"{k}:{v.get('status')}/gap={v.get('rules',{}).get('data_gap')}/{v.get('minutes_since_data')}m\" for k,v in s.items()))"; }
{
echo "=== 장애 시험 시작 $(date '+%F %T') — $SVC 를 ${MIN}분 멈춤"
api
systemctl --user stop "$SVC"; echo "$(date +%T) $SVC stopped"
for i in $(seq 1 "$MIN"); do sleep 60; api; done
systemctl --user start "$SVC"; echo "$(date +%T) $SVC started"
for i in 1 2 3 4 5 6 7 8; do sleep 60; api; done
echo "=== 이 시간대 경보 기록"
curl -s "localhost:8000/alerts?limit=40" | ../.venv/bin/python -c "
import json,sys
for a in json.load(sys.stdin): print(a['space'], a['type'], a['state'], a['detail'], a.get('data_time'))"
echo "=== 끝 $(date '+%F %T')"
} 2>&1 | tee "$LOG"
