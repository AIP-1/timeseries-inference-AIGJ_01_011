"""실데이터용 '추정 시간표' 만들기 — 실제 시간표·예약 자료가 없어서 학습 기간 CO2로 평소 사용 시간대를 찾는다.

방법 (학습 기간 데이터만 사용 — 검증·시험 기간은 보지 않는다)
  1. 공간별 5분 CO2에서 그날 새벽(03~06시) 중앙값을 빼 '사람 때문에 오른 양(초과분)'을 만든다
  2. 요일 × 30분 칸마다, 학습 기간 중 초과분이 기준(공간별 상위 분위수의 30%)을 넘은 날의 비율을 센다
  3. 그 비율이 50% 이상인 칸을 '평소 사용 시간'으로 보고, 이어진 칸을 묶어 시간표로 만든다
  4. 시간표를 전체 기간과 앞으로 60일에 펼친다(요일 반복). 예정 인원 대신 그 칸의 초과분 중앙값을 '예상 강도'로 쓴다
한계: 공휴일·방학·보강은 모른다. 매주 반복되는 패턴만 담긴다 → 보고서에 '추정 시간표'로 밝힌다.

사용법: ../.venv/bin/python real/estimate_schedule.py --influx ../data/influx
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tsi import data as D

SLOT = "30min"
MIN_FRAC = 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--influx", default=os.path.join(os.path.dirname(D.__file__), "..", "..", "data", "influx"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                  "data", "real"))
    a = ap.parse_args()
    raw = D.load_influx(a.influx)
    s, _ = D.clean(raw["sensors"])
    df = D.resample(s)
    val_start = pd.Timestamp(D.REAL_SPLIT["val_start"])
    train = df[(df.ts < val_start) & df.co2.notna()].copy()

    rows, tables = [], []
    for space, g in train.groupby("space"):
        g = g.copy()
        g["day"] = g.ts.dt.normalize()
        night = g[g.ts.dt.hour.between(3, 5)].groupby("day").co2.median()
        g["excess"] = g.co2 - g.day.map(night)
        g = g.dropna(subset=["excess"])
        th = 0.3 * g.excess.quantile(0.95)
        g["slot"] = g.ts.dt.floor(SLOT)
        sl = g.groupby("slot").excess.mean().to_frame()
        sl["wd"] = sl.index.weekday
        sl["tod"] = sl.index.hour * 60 + sl.index.minute
        sl["used"] = sl.excess > th
        tab = sl.groupby(["wd", "tod"]).agg(frac=("used", "mean"), level=("excess", "median"), n=("used", "size"))
        tab["space"] = space
        tables.append(tab.reset_index())
        on = tab[(tab.frac >= MIN_FRAC) & (tab.n >= 4)].reset_index()
        for wd, w in on.groupby("wd"):  # 이어진 칸을 하나의 일정으로 묶기
            tods = w.tod.values
            start = prev = tods[0]
            lv = [w.level.values[0]]
            for t, l in zip(tods[1:], w.level.values[1:]):
                if t - prev > 30:
                    rows.append((space, wd, start, prev + 30, float(np.median(lv))))
                    start, lv = t, []
                prev = t
                lv.append(l)
            rows.append((space, wd, start, prev + 30, float(np.median(lv))))
        print(f"{space}: 기준 초과분 {th:.0f}ppm, 평소 사용 칸 {len(on)}개", flush=True)

    tmpl = pd.DataFrame(rows, columns=["space", "weekday", "start_min", "end_min", "level"])
    days = pd.date_range(df.ts.min().normalize(), df.ts.max().normalize() + pd.Timedelta(days=60), freq="D")
    sched = []
    for d in days:
        for r in tmpl[tmpl.weekday == d.weekday()].itertuples():
            sched.append(dict(space=r.space, start=d + pd.Timedelta(minutes=int(r.start_min)),
                              end=d + pd.Timedelta(minutes=int(r.end_min)), kind="추정 사용",
                              planned_people=round(r.level / 100, 1)))  # 예상 강도: 초과분 100ppm = 1
    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(sched).to_csv(os.path.join(a.out, "schedule.csv"), index=False)
    tmpl.to_csv(os.path.join(a.out, "schedule_template.csv"), index=False)
    pd.concat(tables).to_csv(os.path.join(a.out, "usage_by_slot.csv"), index=False)
    wdn = "월화수목금토일"
    print("\n추정 시간표 (매주 반복)")
    for r in tmpl.sort_values(["space", "weekday", "start_min"]).itertuples():
        print(f"  {r.space} {wdn[r.weekday]} {r.start_min // 60:02d}:{r.start_min % 60:02d}"
              f"~{r.end_min // 60:02d}:{r.end_min % 60:02d}  강도 {r.level:.0f}ppm")


if __name__ == "__main__":
    main()
