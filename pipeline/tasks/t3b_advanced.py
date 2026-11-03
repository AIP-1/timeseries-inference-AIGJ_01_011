"""과제 3 심화(B).
  1) 과거 길이 비교: 과거 1·2·4시간(5분 × 12·24·48)으로 GRU를 학습해 같은 시험 시점에서 비교
  2) 처음 보는 공간: 한 공간을 학습에서 완전히 빼고(그 공간의 공간 표시도 쓰지 않음) 그 공간을 예측
모든 비교는 가장 긴 창(4시간)이 들어가는 같은 시험 시점에서 한다(공정한 비교).
산출: results/t3b<tag>/advanced.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tasks.t2_baselines import baseline_preds, true_future
from tasks.t3_forecast import fit, predict
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec

LOOKBACKS = [12, 24, 48]


def mae60(pred, df, idx):
    yt = true_future(df, idx, "co2")[:, -1]
    return float(np.nanmean(np.abs(pred[:, -1, 0] - yt)))


def main():
    a = C.args("과제 3 심화", epochs=40, device="cpu", train_stride=1, kind="gru")
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t3b" + a.tag)
    cond = C.conditions(raw, bounds)
    Lmax = max(LOOKBACKS)
    te = D.anchors(df, bounds, "test", exclude=fmask, lookback=Lmax)
    va = D.anchors(df, bounds, "val", exclude=fmask, lookback=Lmax)
    tr = D.anchors(df, bounds, "train", exclude=fmask, lookback=Lmax)[::a.train_stride]
    train_df = df[D.period_mask(df.ts, bounds, "train")]
    persist = baseline_preds(df, te, "co2")["현재값 유지"]
    base60 = float(np.nanmean(np.abs(persist[:, -1] - true_future(df, te, "co2")[:, -1])))
    md = ["# 과제 3 심화\n", C.cond_md(cond),
          f"시험 시점 {len(te):,}개(과거 4시간 창이 모두 있는 시점, 고장 구간 제외). 현재값 유지 60분 CO2 MAE {base60:.1f}ppm\n"]

    # 1) 과거 길이
    rows = []
    for L in LOOKBACKS:
        D.LOOKBACK = L  # features.windows 가 이 값을 읽는다
        spec = FeatureSpec(train_df)
        m, val, ep = fit(a.kind, spec, df, tr, va, a.epochs, a.device)
        p = predict(m, spec, df, te)
        rows.append(dict(과거=f"{L * D.STEP_MIN // 60}시간", 검증_오차=round(val, 4), 에폭=ep,
                         CO2_60분_MAE=round(mae60(p, df, te), 1),
                         현재값_대비_개선=f"{(1 - mae60(p, df, te) / base60) * 100:.1f}%"))
        print(rows[-1], flush=True)
    D.LOOKBACK = 24
    best = min(rows, key=lambda r: r["검증_오차"])
    md += ["## 1. 과거 길이 비교 (" + a.kind.upper() + ")\n", pd.DataFrame(rows).to_markdown(index=False), "",
           f"검증 오차로 고르면 **과거 {best['과거']}**. 시험 결과로 고르지 않는다.\n"]

    # 2) 처음 보는 공간
    spaces = sorted(df.space.unique())
    rows2 = []
    spec_all = FeatureSpec(train_df)
    m_all, _, _ = fit(a.kind, spec_all, df, tr, va, a.epochs, a.device)
    for sp in spaces:
        keep = lambda idx: idx[df.space.values[idx] != sp]
        only = lambda idx: idx[df.space.values[idx] == sp]
        # 공간 표시(one-hot)를 빼야 '처음 보는 공간'이 된다
        sub = df.copy()
        for c in [c for c in df.columns if c.startswith("space_")]:
            sub[c] = 0.0
        spec = FeatureSpec(sub[D.period_mask(sub.ts, bounds, "train") & (sub.space != sp)])
        m, _, _ = fit(a.kind, spec, sub, keep(tr), keep(va), a.epochs, a.device)
        t_sp = only(te)
        unseen = mae60(predict(m, spec, sub, t_sp), sub, t_sp)
        seen = mae60(predict(m_all, spec_all, df, t_sp), df, t_sp)
        pers = float(np.nanmean(np.abs(persist[df.space.values[te] == sp][:, -1]
                                       - true_future(df, t_sp, "co2")[:, -1])))
        rows2.append(dict(시험_공간=sp, 시점=len(t_sp), 현재값_유지=round(pers, 1), 함께_학습=round(seen, 1),
                          처음_보는_공간=round(unseen, 1),
                          성능_손실=f"{(unseen / seen - 1) * 100:+.0f}%"))
        print(rows2[-1], flush=True)
    md += ["## 2. 처음 보는 공간 일반화\n",
           "한 공간을 학습에서 완전히 빼고(공간 표시도 0으로) 그 공간의 60분 CO2를 예측. "
           "'함께 학습'은 모든 공간으로 학습한 같은 구조 모델.\n",
           pd.DataFrame(rows2).to_markdown(index=False), "",
           "처음 보는 공간 오차가 현재값 유지보다 작으면, 새 공간에 센서만 달아도 학습 없이 바로 쓸 수 있다는 뜻이다. "
           "공간마다 부피·환기량·사용 인원이 달라 CO2가 오르는 속도가 다르므로, 손실이 큰 공간은 며칠치 데이터로 "
           "추가 학습하거나 과제 6의 사전학습 모델을 임시로 쓰는 것이 낫다."]
    C.save(out, "advanced", dict(conditions=cond, lookback=rows, unseen=rows2), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
