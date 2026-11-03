"""과제 5 다시: 환기 선제 경보 재현율 (임계값은 다시 고르지 않는다).

기존 MLP(artifacts/vent.pt, 임계값은 저장된 값 그대로)에 과제 3 예측 모델을 함께 쓰는 방법을 후보로 둔다.
  M  MLP 혼자 (기존 결과)
  U  MLP 또는 예측 모델(30분 안 최대 예측 ≥ 1,000) 둘 중 하나라도 경보
  I  MLP 그리고 예측 모델 둘 다 경보 (참고: 정밀도 쪽)
고르는 기준(미리 정함, 기존 임계값 기준과 같음): 검증 기간 재현율 90% 이상인 후보 중 검증 macro-F1 최대.
시험 기간은 고른 뒤 한 번 평가. 기존 MLP 결과도 표에 그대로 둔다.
산출: results/t5c<tag>/vent.md
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tasks.t3_forecast import predict
from tasks.t5_state import cm_md, metrics, proba
from tsi import common as C
from tsi import data as D
from tsi import state as S
from tsi.features import FeatureSpec
from tsi.models import MLPClassifier, make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
VAL_RECALL = 0.90


def main():
    a = C.args("과제 5 환기 경보 다시")
    raw, df, bounds, fmask = C.prepare(a)
    cond = C.conditions(raw, bounds)
    pm = lambda n: D.period_mask(df.ts, bounds, n).values & ~fmask & df.co2.notna().values
    V = S.vent_features(df)
    vy = S.vent_label(df).values
    vok = np.isfinite(vy) & V[["co2_d30", "co2_above_min2h"]].notna().all(axis=1).values
    meta = json.load(open(os.path.join(ART, f"vent{a.tag}.json")))
    m = MLPClassifier(len(meta["cols"])); m.load_state_dict(torch.load(os.path.join(ART, f"vent{a.tag}.pt"))); m.eval()
    X = np.nan_to_num((V[meta["cols"]].values - np.array(meta["mu"])) / np.array(meta["sd"])).astype(np.float32)
    th = meta["threshold"]
    best = json.load(open(os.path.join(ART, f"forecast_best{a.tag}.json")))["model"]
    fmeta = json.load(open(os.path.join(ART, f"forecast_{best}{a.tag}.json")))
    spec = FeatureSpec.from_dict(fmeta["spec"])
    fm = make_forecaster(best, len(spec.past), len(spec.future), D.HORIZON)
    fm.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{a.tag}.pt"))); fm.eval()

    res, sets = [], {}
    for period in ("val", "test"):
        idx = np.where(pm(period) & vok)[0]
        idx = idx[(idx >= D.LOOKBACK) & (idx + D.HORIZON < len(df))]
        mlp = proba(m, X[idx]) >= th
        fc = predict(fm, spec, df, idx)[:, :S.VENT_AHEAD, 0].max(axis=1) >= S.VENT_LIMIT
        sets[period] = (vy[idx].astype(int), {"M  MLP 혼자 (기존)": mlp, "U  MLP 또는 예측 모델": mlp | fc,
                                              "I  MLP 그리고 예측 모델": mlp & fc,
                                              f"참고: 예측 모델({best.upper()}) 혼자": fc})
    yv, cv = sets["val"]
    val_rows = {k: metrics(yv, p.astype(int), k) for k, p in cv.items()}
    ok = [k for k in cv if not k.startswith("참고") and val_rows[k]["재현율"] >= VAL_RECALL * 100]
    chosen = max(ok, key=lambda k: val_rows[k]["macro_F1"]) if ok else "M  MLP 혼자 (기존)"
    yt, ct = sets["test"]
    rows = []
    for k in cv:
        v, t = val_rows[k], metrics(yt, ct[k].astype(int), k)
        rows.append(dict(후보=k, 검증_재현율=v["재현율"], 검증_macro_F1=v["macro_F1"], 시험_재현율=t["재현율"],
                         시험_정밀도=t["정밀도"], 시험_F1=t["F1"], 시험_macro_F1=t["macro_F1"],
                         선택="✅" if k == chosen else ""))
    tr = next(r for r in rows if r["후보"] == chosen)
    md = ["# 과제 5 다시. 환기 선제 경보\n", C.cond_md(cond),
          f"MLP 임계값은 저장된 값({th:.2f}) 그대로 쓰고 다시 고르지 않았다. 예측 모델은 과제 3 대표 모델({best.upper()}), "
          f"30분 안 최대 예측이 {S.VENT_LIMIT:.0f}ppm 이상이면 경보. 판단 시점은 과거 창·미래 60분이 모두 있는 시점.\n",
          f"**고르는 기준(미리 정함)**: 검증 기간 재현율 {VAL_RECALL * 100:.0f}% 이상 중 검증 macro-F1 최대. "
          "시험 열은 고른 뒤 한 번 평가한 값이다.\n",
          pd.DataFrame(rows).to_markdown(index=False), "",
          f"고른 후보: **{chosen}** → 시험 재현율 {tr['시험_재현율']}% (기준 85% {'✅' if tr['시험_재현율'] >= 85 else '❌'}), "
          f"macro-F1 {tr['시험_macro_F1']}%\n",
          "고른 후보의 시험 혼동 행렬\n", cm_md(yt, ct[chosen].astype(int), ["괜찮음", "곧 환기 필요"]), ""]
    C.save(os.path.join(a.out, "t5c" + a.tag), "vent", dict(conditions=cond, rows=rows, chosen=chosen), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
