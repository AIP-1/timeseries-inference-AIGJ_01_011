"""과제 3 다시: 60분 CO2 개선을 늘려 보기 (검증 기간으로만 고르고, 시험은 마지막에 한 번 평가).

후보(모두 같은 검증·시험 시점 = 과거 4시간 창이 모두 있는 시점, 고장 구간 제외):
  A  기존 GRU (과거 2시간, 서비스 모델)
  B  기존 GRU·LSTM·TCN 평균 (과거 2시간)
  C  GRU 과거 4시간 × 시드 5개 평균 (새로 학습)
  D  B + C 평균
고르는 기준: 검증 기간 60분 CO2 MAE(ppm)가 가장 작은 후보. 시험 표는 고른 뒤 함께 보여 주되 선택에는 쓰지 않는다.
산출: results/t3c<tag>/improve.md, artifacts/forecast_gru4h_s<k><tag>.pt
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tasks.t2_baselines import baseline_preds, true_future
from tasks.t3_forecast import fit, predict
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
SEEDS = 5


def load(kind, tag):
    meta = json.load(open(os.path.join(ART, f"forecast_{kind}{tag}.json")))
    spec = FeatureSpec.from_dict(meta["spec"])
    m = make_forecaster(kind, len(spec.past), len(spec.future), D.HORIZON)
    m.load_state_dict(torch.load(os.path.join(ART, f"forecast_{kind}{tag}.pt")))
    return m.eval(), spec


def pred_with(L, model, spec, df, idx):
    D.LOOKBACK = L
    try:
        return predict(model, spec, df, idx)
    finally:
        D.LOOKBACK = 24


def errs(pred, df, idx):
    """60분 CO2 절대오차(시점별)와 시점별 60분 온도 절대오차."""
    return (np.abs(pred[:, -1, 0] - true_future(df, idx, "co2")[:, -1]),
            np.abs(pred[:, -1, 1] - true_future(df, idx, "temperature")[:, -1]))


def main():
    a = C.args("과제 3 개선", epochs=40, device="cuda", train_stride=1)
    raw, df, bounds, fmask = C.prepare(a)
    cond = C.conditions(raw, bounds)
    va = D.anchors(df, bounds, "val", exclude=fmask, lookback=48)
    te = D.anchors(df, bounds, "test", exclude=fmask, lookback=48)
    tr = D.anchors(df, bounds, "train", exclude=fmask, lookback=48)[::a.train_stride]
    train_df = df[D.period_mask(df.ts, bounds, "train")]
    print(f"anchors train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    preds = {p: {} for p in ("val", "test")}
    for kind in ("gru", "lstm", "tcn"):
        m, spec = load(kind, a.tag)
        for p, idx in (("val", va), ("test", te)):
            preds[p][kind.upper()] = pred_with(24, m, spec, df, idx)
    fits = []
    D.LOOKBACK = 48
    spec4 = FeatureSpec(train_df)
    c_models = []
    for s in range(SEEDS):
        t0 = time.time()
        m, val, ep = fit("gru", spec4, df, tr, va, a.epochs, a.device, seed=s)
        c_models.append(m)
        torch.save(m.state_dict(), os.path.join(ART, f"forecast_gru4h_s{s}{a.tag}.pt"))
        fits.append(dict(시드=s, 검증_오차=round(val, 4), 에폭=ep, 학습_s=round(time.time() - t0, 1)))
        print(fits[-1], flush=True)
    with open(os.path.join(ART, f"forecast_gru4h{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(kind="gru", spec=spec4.to_dict(), seeds=SEEDS, conditions=cond), f, ensure_ascii=False)
    D.LOOKBACK = 24
    for p, idx in (("val", va), ("test", te)):
        preds[p]["GRU4h"] = np.mean([pred_with(48, m, spec4, df, idx) for m in c_models], axis=0)

    cands = {
        "A 기존 GRU (2시간)": lambda P: P["GRU"],
        "B GRU·LSTM·TCN 평균 (2시간)": lambda P: (P["GRU"] + P["LSTM"] + P["TCN"]) / 3,
        "C GRU 4시간 × 시드 5개 평균": lambda P: P["GRU4h"],
        "D B + C 평균": lambda P: ((P["GRU"] + P["LSTM"] + P["TCN"]) / 3 + P["GRU4h"]) / 2,
    }
    base = {p: errs(np.stack([baseline_preds(df, idx, "co2")["현재값 유지"],
                              baseline_preds(df, idx, "temperature")["현재값 유지"]], axis=2), df, idx)
            for p, idx in (("val", va), ("test", te))}
    rows = []
    for name, f in cands.items():
        ev, tv = errs(f(preds["val"]), df, va)
        et, tt = errs(f(preds["test"]), df, te)
        rows.append(dict(후보=name, 검증_CO2_60분=round(float(np.nanmean(ev)), 2),
                         검증_개선=f"{(1 - np.nanmean(ev) / np.nanmean(base['val'][0])) * 100:.1f}%",
                         시험_CO2_60분=round(float(np.nanmean(et)), 2),
                         시험_개선=f"{(1 - np.nanmean(et) / np.nanmean(base['test'][0])) * 100:.1f}%",
                         시험_온도_60분=round(float(np.nanmean(tt)), 3)))
    tbl = pd.DataFrame(rows)
    chosen = tbl.loc[tbl.검증_CO2_60분.idxmin()]
    # 공간별(고른 후보, 시험)
    pick = cands[chosen.후보](preds["test"])
    e = errs(pick, df, te)[0]
    sp = df.space.values[te]
    by = pd.DataFrame([dict(공간=s, 현재값_유지=round(float(np.nanmean(base["test"][0][sp == s])), 1),
                            고른_후보=round(float(np.nanmean(e[sp == s])), 1),
                            개선=f"{(1 - np.nanmean(e[sp == s]) / np.nanmean(base['test'][0][sp == s])) * 100:.1f}%")
                       for s in np.unique(sp)])
    # 시간대별 오차(오차가 큰 순간)
    hr = pd.to_datetime(df.ts.values[te]).hour
    byh = pd.DataFrame(dict(h=hr, model=e, persist=base["test"][0])).groupby("h").mean().round(1)
    md = ["# 과제 3 다시. 60분 CO2 예측 개선 시도\n", C.cond_md(cond),
          f"검증 시점 {len(va):,}개, 시험 시점 {len(te):,}개(과거 4시간 창이 모두 있는 시점, 고장 구간 제외 — 과제 3 표(16,864개)와 "
          f"시점이 조금 다름). 현재값 유지 60분 CO2 MAE: 검증 {np.nanmean(base['val'][0]):.1f}ppm, 시험 "
          f"{np.nanmean(base['test'][0]):.1f}ppm.\n",
          "**고르는 기준은 검증 기간 60분 CO2 MAE**. 시험 열은 고른 뒤 함께 적은 것이며 선택에 쓰지 않았다.\n",
          tbl.to_markdown(index=False), "",
          f"검증으로 고른 후보: **{chosen.후보}** → 시험 60분 CO2 {chosen.시험_CO2_60분}ppm, 현재값 유지 대비 {chosen.시험_개선} "
          f"(기준 30% {'✅' if float(chosen.시험_개선[:-1]) >= 30 else '❌'})\n",
          "## C 학습 기록 (GRU 과거 4시간)\n", pd.DataFrame(fits).to_markdown(index=False), "",
          "## 공간별 (고른 후보, 시험)\n", by.to_markdown(index=False), "",
          "## 시간대별 60분 CO2 MAE (고른 후보 vs 현재값 유지, 시험)\n", byh.T.to_markdown(), ""]
    C.save(os.path.join(a.out, "t3c" + a.tag), "improve",
           dict(conditions=cond, rows=rows, chosen=chosen.후보, fits=fits), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
