"""수집 현황 보고: 공간·항목별 건수, 마지막 수신 시각, 커버율(1분 단위로 데이터가 있는 비율).

사용법:  .venv/bin/python status.py [시간=24] [항목=co2,temperature,humidity]
"""
import os
import sqlite3
import sys
import time

import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
hours = float(sys.argv[1]) if len(sys.argv) > 1 else 24
metrics = (sys.argv[2] if len(sys.argv) > 2 else "co2,temperature,humidity").split(",")

with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
db = sqlite3.connect(f"file:{os.path.join(BASE, cfg['db_path'])}?mode=ro", uri=True)

now = int(time.time() * 1000)
since = now - int(hours * 3600 * 1000)
first = db.execute("SELECT MIN(ts_recv) FROM readings").fetchone()[0] or now
start = max(since, first)
minutes = now // 60000 - start // 60000 + 1  # 집계 구간에 걸친 1분 칸 수(양 끝 포함)

q = f"""
SELECT location, metric, COUNT(*), COUNT(DISTINCT device),
       COUNT(DISTINCT ts_recv / 60000), MAX(ts_recv), SUM(flag)
FROM readings
WHERE ts_recv >= ? AND metric IN ({','.join('?' * len(metrics))})
GROUP BY location, metric ORDER BY location, metric
"""
print(f"수집 현황 — 최근 {hours:g}시간 (실제 집계 구간 {minutes}분, 기준 시각 {time.strftime('%Y-%m-%d %H:%M:%S')})\n")
print(f"{'공간':<12}{'항목':<13}{'건수':>7}{'센서':>5}{'커버율':>8}{'범위밖':>7}  마지막 수신(지난 시간)")
for loc, metric, n, ndev, nmin, last, bad in db.execute(q, (start, *metrics)):
    ago = (now - last) / 60000
    warn = "  ⚠ 끊김" if ago > 5 else ""
    print(f"{loc:<12}{metric:<13}{n:>7}{ndev:>5}{100 * nmin / minutes:>7.1f}%{bad:>7}  "
          f"{time.strftime('%H:%M:%S', time.localtime(last / 1000))} ({ago:.1f}분 전){warn}")

dups = db.execute("SELECT COUNT(*) FROM raw_messages WHERE ts_recv >= ?", (start,)).fetchone()[0]
rows = db.execute("SELECT COUNT(DISTINCT msg_id) FROM readings WHERE ts_recv >= ?", (start,)).fetchone()[0]
bad = db.execute("SELECT COUNT(*) FROM unparsed WHERE ts_recv >= ?", (start,)).fetchone()[0]
skew = db.execute("""SELECT device, ROUND(AVG(ts_sensor - ts_recv) / 1000.0) AS s FROM readings
                     WHERE ts_recv >= ? AND source = 'lorawan' GROUP BY device HAVING ABS(s) > 30""",
                  (start,)).fetchall()
print(f"\n원본 메시지 {dups}건 → 서로 다른 메시지 {rows}건 (나머지는 재전송·중복), 해석 실패 {bad}건")
if skew:
    print("센서 시각과 수신 시각 차이가 30초 넘는 장치:", ", ".join(f"{d} {s:+.0f}s" for d, s in skew))
