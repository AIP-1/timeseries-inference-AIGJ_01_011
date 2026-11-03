"""과제 6 심화: 직접 학습한 모델(GRU)과 사전학습 모델(Chronos)을 섞은 예측이 둘 다보다 나은가.

후보: GRU, Chronos(과거 8h·24h, 중앙값), GRU와 Chronos 평균(같은 가중치).
고르는 기준: 검증 기간 60분 CO2 MAE. 시험 기간은 고른 뒤 함께 적는다(선택에 쓰지 않음).
시점: 검증·시험 기간을 4칸마다 솎음, 과거 24시간이 같은 공간 안에 다 있는 시점만(모든 후보 같은 시점).
산출: results/t6b<tag>/ensemble.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tasks.t2_baselines import baseline_preds, true_future
from tasks.t3_forecast import predict
from tasks.t6_chronos import chronos_predict
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

ART = os.path.join(C.ROOT, "artifacts")


def main():
    a = C.args("과제 6 심화", model="amazon/chronos-bolt-small", stride=4, device="cuda")
    from chronos import BaseChronosPipeline
    raw, df, bounds, fmask = C.prepare(a)
    cond = C.conditions(raw, bounds)
    co2 = df.co2.to_numpy(np.float32)
    space_start = df.groupby("space").cumcount().to_numpy()
    best = json.load(open(os.path.join(ART, f"forecast_best{a.tag}.json")))["model"]
    meta = json.load(open(os.path.join(ART, f"forecast_{best}{a.tag}.json")))
    spec = FeatureSpec.from_dict(meta["spec"])
    fm = make_forecaster(best, len(spec.past), len(spec.future), D.HORIZON)
    fm.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{a.tag}.pt"))); fm.eval()
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map=a.device, torch_dtype=torch.float32)
    res = {}
    for period in ("val", "test"):
        idx = D.anchors(df, bounds, period, exclude=fmask)[::a.stride]
        idx = idx[space_start[idx] >= 287]
        y = true_future(df, idx, "co2")[:, -1]
        P = {"현재값 유지": baseline_preds(df, idx, "co2")["현재값 유지"][:, -1],
             f"직접 학습({best.upper()})": predict(fm, spec, df, idx)[:, -1, 0]}
        for h in (8, 24):
            P[f"Chronos 과거 {h}h"] = chronos_predict(pipe, co2, idx, h * 60 // D.STEP_MIN)[:, -1, 1]
        for h in (8, 24):
            P[f"평균: {best.upper()} + Chronos {h}h"] = (P[f"직접 학습({best.upper()})"] + P[f"Chronos 과거 {h}h"]) / 2
        res[period] = (len(idx), {k: float(np.nanmean(np.abs(v - y))) for k, v in P.items()})
        print(period, res[period], flush=True)
    names = [k for k in res["val"][1] if k != "현재값 유지"]
    chosen = min(names, key=lambda k: res["val"][1][k])
    bv, bt = res["val"][1]["현재값 유지"], res["test"][1]["현재값 유지"]
    rows = [dict(방법=k, 검증_60분_MAE=round(res["val"][1][k], 2), 시험_60분_MAE=round(res["test"][1][k], 2),
                 시험_현재값_대비=f"{(1 - res['test'][1][k] / bt) * 100:.1f}%", 선택="✅" if k == chosen else "")
            for k in res["val"][1]]
    md = ["# 과제 6 심화. 직접 학습 모델 + Chronos 섞기\n", C.cond_md(cond),
          f"시점: 검증 {res['val'][0]:,}개, 시험 {res['test'][0]:,}개 (4칸마다, 과거 24시간이 있는 시점). "
          f"모델 {a.model}, 추가 학습 없음, 장치 {a.device}. 값은 60분 뒤 CO2 MAE(ppm).\n",
          "**검증 기간 MAE로 고른다.** 시험 열은 선택에 쓰지 않았다.\n",
          pd.DataFrame(rows).to_markdown(index=False), "",
          f"검증으로 고른 방법: **{chosen}** → 시험 {res['test'][1][chosen]:.1f}ppm "
          f"(현재값 유지 {bt:.1f}ppm 대비 {(1 - res['test'][1][chosen] / bt) * 100:.1f}%)\n"]
    C.save(os.path.join(a.out, "t6b" + a.tag), "ensemble", dict(conditions=cond, rows=rows, chosen=chosen), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
