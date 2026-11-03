"""합성 데이터를 실시간처럼 흘려보내는 재생기 (실데이터 수집기 대신 서비스 시험용).

- 합성 데이터의 '검증+시험 기간'을 벽시계에 맞춰 1분에 1분씩 재생하고, 끝나면 처음으로 돌아간다
- 수집기와 같은 모양(readings 표)으로 data/replay.db 에 쓴다 → 서비스는 수집기 DB와 똑같이 읽는다
- 멈췄다가 다시 켜도 지나간 분은 채우지 않는다(실제 수집기가 꺼졌던 동안의 데이터가 없는 것과 같게)
- 데이터 시각(ts_sensor)은 합성 데이터의 시각. 시계 대응은 data/replay_clock.json 에 저장해 서비스와 공유
"""
import json
import os
import sqlite3
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from collector_schema import SCHEMA  # noqa: E402

SYNTH = os.path.join(ROOT, "data", "synth")
DB = os.path.join(ROOT, "data", "replay.db")
CLOCK = os.path.join(ROOT, "data", "replay_clock.json")
BACKFILL_MIN = 240  # 처음 켤 때만 4시간치를 미리 넣어 서비스가 바로 계산할 수 있게


def clock():
    """벽시계 → 재생 시각. 처음 한 번 정하고 파일로 공유."""
    meta = json.load(open(os.path.join(SYNTH, "meta.json")))
    if not os.path.exists(CLOCK):
        start = pd.Timestamp(meta["test_start"]) - pd.Timedelta(days=7)  # 검증 기간부터
        c = dict(wall_start=time.time(), sim_start=str(start), sim_end=meta["end"])
        json.dump(c, open(CLOCK, "w"))
    return json.load(open(CLOCK))


def sim_now(c, wall=None):
    wall = wall or time.time()
    start, end = pd.Timestamp(c["sim_start"]), pd.Timestamp(c["sim_end"])
    span = (end - start).total_seconds()
    return start + pd.Timedelta(seconds=(wall - c["wall_start"]) % span)


def main():
    c = clock()
    s = pd.read_csv(os.path.join(SYNTH, "sensors.csv"), parse_dates=["ts"])
    # 재생 시작 시각보다 BACKFILL_MIN 앞의 데이터도 남겨 둔다(처음 켤 때 미리 채우는 용도)
    s = s[s.ts >= pd.Timestamp(c["sim_start"]) - pd.Timedelta(minutes=BACKFILL_MIN)].set_index("ts").sort_index()
    db = sqlite3.connect(DB, isolation_level=None)
    db.executescript(SCHEMA)
    last = db.execute("SELECT MAX(ts_sensor) FROM readings").fetchone()[0]
    now = sim_now(c).floor("min")
    # 처음이면 몇 시간 미리 채우고, 재시작이면 지나간 분은 건너뛴다(끊긴 구간이 그대로 남도록)
    cursor = now - pd.Timedelta(minutes=BACKFILL_MIN) if last is None else now
    print(f"replay start sim={now} cursor={cursor}", flush=True)
    written = None  # 마지막으로 쓴 재생 시각
    while True:
        now = sim_now(c).floor("min")
        if written is not None and now < written:  # 재생 시각이 거꾸로 감 = 한 바퀴 돌아 처음으로
            print(f"wrap around at sim={now}", flush=True)
            db.execute("DELETE FROM readings")
            cursor = now - pd.Timedelta(minutes=BACKFILL_MIN)
        if now >= cursor:
            chunk = s.loc[cursor:now]
            wall_ms = int(time.time() * 1000)
            rows = [(wall_ms, int(ts.value // 1_000_000), "synth", r.space, None, f"synth-{r.space}", m, float(v), 0,
                     f"{ts.isoformat()}|{r.space}")
                    for ts, r in chunk.iterrows() for m, v in (("co2", r.co2), ("temperature", r.temperature),
                                                              ("humidity", r.humidity)) if pd.notna(v)]
            db.execute("BEGIN")  # 한 번에 커밋: 서비스가 반쯤 쓰인 분을 읽지 않게
            db.executemany("INSERT OR IGNORE INTO readings VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            db.execute("COMMIT")
            written, cursor = now, now + pd.Timedelta(minutes=1)
        time.sleep(5)


if __name__ == "__main__":
    main()
