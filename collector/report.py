"""과제 1 수집 현황 보고서(Markdown)를 만든다.

- 공간·항목별 커버율(1분 칸 중 데이터가 있는 비율), 건수, 마지막 수신
- 장치별 결측 구간(평소 보고 간격의 3배 또는 5분 넘게 빈 구간)과 전체 장치가 같이 빈 구간(수집기·네트워크 쪽 원인 후보)
- 재전송·중복, 해석 실패, 센서 시각 어긋남, 저장소 크기와 1년 뒤 추정

사용법:  .venv/bin/python report.py [시간=24]   → reports/collection_YYYYmmdd_HHMM.md
"""
import os
import sqlite3
import sys
import time

import pandas as pd
import yaml

BASE = os.path.dirname(os.path.abspath(__file__))
HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 24
METRICS = ["co2", "temperature", "humidity"]

with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
db_file = os.path.join(BASE, cfg["db_path"])
db = sqlite3.connect(f"file:{db_file}?mode=ro", uri=True)

now = int(time.time() * 1000)
first = db.execute("SELECT MIN(ts_recv) FROM readings").fetchone()[0] or now
start = max(now - int(HOURS * 3600e3), first)
df = pd.read_sql_query(
    "SELECT ts_recv, ts_sensor, source, location, device, metric, flag, msg_id FROM readings WHERE ts_recv >= ?",
    db, params=(start,))
df["t"] = pd.to_datetime(df.ts_recv, unit="ms", utc=True).dt.tz_convert("Asia/Seoul")
n_min = now // 60000 - start // 60000 + 1
fmt = lambda ms: pd.Timestamp(ms, unit="ms", tz="UTC").tz_convert("Asia/Seoul").strftime("%m-%d %H:%M")
out = []
w = out.append

w(f"# 수집 현황 보고\n")
w(f"- 집계 구간: {fmt(start)} ~ {fmt(now)} (KST, {n_min}분)")
w(f"- 원천: MQTT 실데이터(ChirpStack LoRaWAN 센서 + xflow 냉난방), 기준 시각: 수신 시각(ts_recv)\n")

# 1. 공간·항목별 커버율
env = df[(df.source == "lorawan") & df.metric.isin(METRICS)]
g = env.groupby(["location", "metric"])
tab = pd.DataFrame({
    "센서 수": g.device.nunique(),
    "건수": g.size(),
    "커버율(%)": (g.ts_recv.apply(lambda s: (s // 60000).nunique()) / n_min * 100).round(1),
    "범위 밖": g.flag.sum(),
    "마지막 수신": g.ts_recv.max().map(fmt),
}).reset_index().rename(columns={"location": "공간", "metric": "항목"})
w("## 1. 공간·항목별 커버율\n")
w(tab.to_markdown(index=False))
w("\n커버율 = 집계 구간의 1분 칸 중 해당 공간·항목 값이 하나라도 있는 칸의 비율. 기준 95%.\n")

# 2. 장치별 보고 간격과 결측 구간
w("## 2. 장치별 보고 간격과 결측 구간\n")
msgs = (df[df.source == "lorawan"].drop_duplicates("msg_id")
        .sort_values("ts_recv")[["device", "location", "ts_recv"]])
rows, gaps = [], []
for (dev, loc), s in msgs.groupby(["device", "location"]):
    d = s.ts_recv.diff().dropna() / 60000
    med = d.median() if len(d) else float("nan")
    limit = max(5, 3 * med) if len(d) else 5
    big = s.ts_recv.iloc[1:][d.values > limit] if len(d) else []
    for end in big:
        prev = s.ts_recv[s.ts_recv < end].max()
        gaps.append((dev, loc, fmt(prev), fmt(end), round((end - prev) / 60000, 1)))
    tail = (now - s.ts_recv.max()) / 60000
    if tail > limit:
        gaps.append((dev, loc, fmt(s.ts_recv.max()), "지금까지", round(tail, 1)))
    # 장치의 보고 간격으로 보면 받아야 했던 메시지 수 대비 실제 받은 비율
    span = (now - max(start, s.ts_recv.min())) / 60000
    expect = span / med + 1 if len(d) and med > 0 else float("nan")
    rows.append((dev, loc, len(s), round(med, 2), round(min(len(s) / expect, 1) * 100, 1)))
w(pd.DataFrame(rows, columns=["장치", "공간", "메시지", "보고 간격 중앙값(분)", "기대 대비 수신율(%)"])
  .to_markdown(index=False))
w("\n보고 간격이 10분인 장치는 1분 커버율이 구조적으로 10% 안팎이다. 이런 장치는 '기대 대비 수신율'로 판단한다.\n")
if gaps:
    w("**결측 구간** (평소 간격의 3배 또는 5분 초과)\n")
    w(pd.DataFrame(gaps, columns=["장치", "공간", "마지막 수신", "다음 수신", "길이(분)"]).to_markdown(index=False))
else:
    w("결측 구간 없음.")

# 전체가 같이 빈 1분 칸: 센서 하나가 아니라 수집기·네트워크·브로커 쪽 원인 후보
active = set((msgs.ts_recv // 60000).unique())
silent, run = [], None
for m in range(start // 60000, now // 60000):
    if m not in active:
        run = run or m
    elif run:
        if m - run >= 2:
            silent.append((fmt(run * 60000), fmt(m * 60000), m - run))
        run = None
w("\n**모든 센서가 동시에 2분 넘게 끊긴 구간** (수집기·네트워크·브로커 원인 후보)\n")
w(pd.DataFrame(silent, columns=["시작", "끝", "길이(분)"]).to_markdown(index=False) if silent else "없음.")

# 3. 품질 지표
raw = db.execute("SELECT COUNT(*) FROM raw_messages WHERE ts_recv >= ?", (start,)).fetchone()[0]
uniq = df.msg_id.nunique()
bad = pd.read_sql_query("SELECT reason, COUNT(*) AS n FROM unparsed WHERE ts_recv >= ? GROUP BY reason",
                        db, params=(start,))
skew = (df[df.source == "lorawan"].assign(s=lambda x: (x.ts_sensor - x.ts_recv) / 1000)
        .groupby("device").s.median().round(0))
w("\n## 3. 메시지 품질\n")
w(f"- 받은 원본 메시지 {raw}건, 서로 다른 메시지 {uniq}건 → 재전송·중복 {max(raw - uniq, 0)}건은 `msg_id`(ChirpStack deduplicationId)로 걸러 한 번만 저장")
w(f"- 해석 실패: {int(bad.n.sum()) if len(bad) else 0}건 " +
  (", ".join(f"{r.reason} {r.n}건" for r in bad.itertuples()) if len(bad) else ""))
off = skew[skew.abs() > 30]
w(f"- 센서 시각이 수신 시각과 30초 넘게 어긋난 장치: " +
  (", ".join(f"{d} {v:+.0f}초" for d, v in off.items()) if len(off) else "없음") +
  " → 일부 게이트웨이 시계가 앞서 있어 분석 기준 시각은 수신 시각으로 정함")

# 4. 저장소 크기
size = sum(os.path.getsize(db_file + ext) for ext in ("", "-wal") if os.path.exists(db_file + ext))
span_day = (now - start) / 86400e3
n_read = db.execute("SELECT COUNT(*) FROM readings WHERE ts_recv >= ?", (start,)).fetchone()[0]
raw_bytes = db.execute("SELECT COALESCE(SUM(LENGTH(payload)), 0) FROM raw_messages WHERE ts_recv >= ?",
                       (start,)).fetchone()[0]
READING_BYTES = 120  # 행 하나(값+태그 문자열)와 인덱스를 합친 대략의 크기
w("\n## 4. 저장소 크기\n")
w(f"- 현재 파일 크기 {size / 1e6:.1f} MB (WAL 포함, 첫 수신부터 {(now - first) / 3.6e6:.1f}시간)")
if span_day > 0:
    rd, rw = n_read / span_day, raw_bytes / span_day
    w(f"- 하루 유입: 측정값 약 {rd:,.0f}행(≈{rd * READING_BYTES / 1e6:.1f} MB), 원본 메시지 약 {rw / 1e6:.1f} MB")
    w(f"- 1년 뒤 추정: 측정값 {rd * READING_BYTES * 365 / 1e9:.1f} GB + 원본 {rw * 365 / 1e9:.1f} GB "
      f"(NVMe 여유 약 210 GB)")
    w("- 보관 정책 제안: 원본 메시지 30일, 1분 측정값 180일, 그 뒤로는 5분 평균만 영구 보관")

os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
path = os.path.join(BASE, "reports", time.strftime("collection_%Y%m%d_%H%M.md"))
with open(path, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
print(f"\n→ saved {path}")
