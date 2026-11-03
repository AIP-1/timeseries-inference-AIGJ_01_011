"""과제 8 심화: 상시 서비스 2주 운영 기록으로 본 실제 성적.

입력(서비스가 남긴 기록만 사용, 모델·데이터를 다시 돌리지 않음)
  results/forecast_scores.jsonl : 서비스가 내놓은 30·60분 CO2 예측과 그 시각의 실제 5분 평균(스스로 채점)
  results/service_cycles.jsonl  : 주기마다 새로 켜진 경보
1) 실운영 예측 성적: 공간·주별 60분(30분) MAE를 '현재값 유지'(같은 기록의 h분 전 실제값)와 비교
2) 환기 경보의 실제 효과: 실제 5분 평균 CO2가 1,000ppm을 넘은 사건마다, 그 전 60분 안에 경보(ventilation_soon 또는
   co2_forecast_1000)가 먼저 났는지와 몇 분 앞섰는지. 경보 뒤 60분 안에 넘지 않은 경보는 '헛경보'
3) 제어 규칙 효과 추정(상한): "경보가 나면 바로 환기한다"고 할 때, 경보가 10분 이상 먼저 난 사건의 1,000ppm 초과
   시간은 막을 수 있었다고 보고(환기 효과는 즉시·충분하다고 가정 → 상한), 지난 1주 초과 시간이 얼마나 줄었을지 계산
산출: results/t8b/live.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tsi import common as C

RES = os.path.join(C.ROOT, "results")
LIMIT = 1000
LEAD_OK = 10


def main():
    sc = pd.read_json(os.path.join(RES, "forecast_scores.jsonl"), lines=True)
    sc["target"] = pd.to_datetime(sc.target)
    act = sc.drop_duplicates(["space", "target"]).set_index(["space", "target"]).actual
    sc["persist"] = [act.get((s, t - pd.Timedelta(minutes=h)), np.nan) for s, t, h in zip(sc.space, sc.target, sc.h)]
    sc = sc.dropna(subset=["persist", "actual"])
    start = sc.target.min().normalize()
    sc["week"] = ((sc.target - start).dt.days // 7).map(lambda w: f"{(start + pd.Timedelta(days=7 * w)):%m-%d}~")
    sc["e_model"], sc["e_persist"] = (sc.pred - sc.actual).abs(), (sc.persist - sc.actual).abs()
    t1 = (sc.groupby(["h", "space"])[["e_model", "e_persist"]].mean()
          .assign(개선=lambda x: ((1 - x.e_model / x.e_persist) * 100).round(1)).round(1)
          .rename(columns={"e_model": "서비스_MAE", "e_persist": "현재값_유지_MAE"}))
    allr = (sc.groupby("h")[["e_model", "e_persist"]].mean()
            .assign(개선=lambda x: ((1 - x.e_model / x.e_persist) * 100).round(1)).round(1)
            .rename(columns={"e_model": "서비스_MAE", "e_persist": "현재값_유지_MAE"}))
    t2 = (sc[sc.h == 60].groupby("week")[["e_model", "e_persist"]].mean()
          .assign(개선=lambda x: ((1 - x.e_model / x.e_persist) * 100).round(1)).round(1)
          .rename(columns={"e_model": "서비스_MAE", "e_persist": "현재값_유지_MAE"}))

    # 실제 CO2 5분 평균 시계열과 1,000ppm 초과 사건
    cy = pd.read_json(os.path.join(RES, "service_cycles.jsonl"), lines=True)
    cy["data_time"] = pd.to_datetime(cy.data_time)
    al = [(r.data_time, a.split(":")[1]) for r in cy.itertuples() for a in r.alerts
          if a.split(":")[0] in ("ventilation_soon", "co2_forecast_1000")]
    al = pd.DataFrame(al, columns=["t", "space"]).drop_duplicates()
    ev_rows, exceed_rows = [], []
    for sp, s in act.groupby(level=0):
        s = s.droplevel(0).sort_index()
        s = s[~s.index.duplicated()].asfreq("5min")
        above = s >= LIMIT
        a_sp = al[al.space == sp].t.sort_values()
        # 사건: 직전 30분이 모두 1,000 미만이었다가 넘은 시점
        prev = above.shift(1).rolling(6, min_periods=6).max()
        starts = s.index[above & (prev == 0)]
        for e in starts:
            w = a_sp[(a_sp >= e - pd.Timedelta(minutes=60)) & (a_sp <= e)]
            lead = (e - w.min()).total_seconds() / 60 if len(w) else None
            # 이 사건의 1,000 초과 길이(분)
            seg = above[e:]
            dur = (seg.idxmin() - e).total_seconds() / 60 if (~seg).any() else len(seg) * 5
            ev_rows.append(dict(공간=sp, 시작=e, 경보_앞선_분=lead, 초과_분=dur))
        # 헛경보: 경보 뒤 60분 안에 1,000을 넘지 않음 (이미 넘은 상태의 경보는 제외)
        for t in a_sp:
            if t in s.index and s.get(t, np.nan) >= LIMIT:
                continue
            nxt = s[t:t + pd.Timedelta(minutes=60)]
            exceed_rows.append(dict(공간=sp, t=t, 맞음=bool((nxt >= LIMIT).any())))
    ev = pd.DataFrame(ev_rows)
    fa = pd.DataFrame(exceed_rows)
    ev["10분_이상_먼저"] = ev.경보_앞선_분.fillna(-1) >= LEAD_OK
    t3 = ev.groupby("공간").agg(초과_사건=("시작", "size"), 경보_먼저=("경보_앞선_분", lambda x: x.notna().sum()),
                               _10분_이상_먼저=("10분_이상_먼저", "sum"),
                               앞선_분_중앙값=("경보_앞선_분", "median"), 초과_시간_분=("초과_분", "sum"))
    t3["막을_수_있던_초과_분(상한)"] = ev[ev["10분_이상_먼저"]].groupby("공간").초과_분.sum()
    t3 = t3.fillna(0)
    t3["줄어드는_비율(상한)"] = (t3["막을_수_있던_초과_분(상한)"] / t3.초과_시간_분.replace(0, np.nan) * 100).round(0)
    t4 = fa.groupby("공간").agg(경보=("맞음", "size"), 실제_초과=("맞음", "sum")).rename(columns={"실제_초과": "60분_안에_실제_초과"})
    t4["정밀도"] = (t4["60분_안에_실제_초과"] / t4.경보 * 100).round(1)
    last = ev[ev.시작 >= ev.시작.max() - pd.Timedelta(days=7)]
    tot, prev = last.초과_분.sum(), last[last["10분_이상_먼저"]].초과_분.sum()
    md = ["# 과제 8 심화. 상시 서비스 2주 운영 기록\n",
          f"> 기간 {sc.target.min():%Y-%m-%d %H:%M} ~ {sc.target.max():%Y-%m-%d %H:%M} (서비스가 실제로 내놓은 예측, 실데이터, "
          f"Jetson team4 MAXN_SUPER, ONNX Runtime CPU). 정답은 그 시각의 실제 5분 평균.\n",
          "## 1. 실운영 예측 성적 (CO2, ppm)\n", allr.to_markdown(), "", "공간별\n", t1.to_markdown(), "",
          "60분 예측, 주별\n", t2.to_markdown(), "",
          "## 2. 1,000ppm 초과 사건과 환기 경보\n",
          f"사건 = 직전 30분 동안 1,000 미만이다가 5분 평균이 1,000 이상이 된 시점. 경보 = 그 전 60분 안의 "
          f"ventilation_soon 또는 co2_forecast_1000.\n", t3.round(1).to_markdown(), "",
          "경보의 정밀도 (이미 1,000 이상인 때의 경보 제외)\n", t4.to_markdown(), "",
          "## 3. 제어 규칙 효과 추정 (상한)\n",
          f"규칙: '경보가 나면 바로 창문·환기 장치를 켠다'. 경보가 {LEAD_OK}분 이상 먼저 난 사건은 환기로 초과를 막을 수 있다고 "
          f"가정(환기가 즉시·충분히 효과가 있다는 낙관적 가정 → 상한).\n",
          f"지난 1주({last.시작.min():%m-%d}~{last.시작.max():%m-%d}) 1,000ppm 초과 {tot:.0f}분 중 "
          f"{prev:.0f}분({(prev / tot * 100) if tot else 0:.0f}%)은 경보가 {LEAD_OK}분 이상 먼저 났다 → 이 규칙으로 줄일 수 있는 "
          f"초과 시간의 상한.\n"]
    C.save(os.path.join(RES, "t8b"), "live", dict(events=len(ev)), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
