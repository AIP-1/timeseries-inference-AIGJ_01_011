"""과제 5 심화(B) — 합성 데이터(직접 관찰 정답이 있는 데이터).
  1) 처음 보는 공간: 한 공간을 학습에서 완전히 빼고(학습·임계값 선택 모두 나머지 공간) 그 공간의 사용 여부를 추정.
     같은 시험 칸에서 '모든 공간으로 학습'한 모델과 비교
  2) 노쇼·예약 없는 사용 찾기 규칙
     - 노쇼: 일정(수업·예약)이 시작된 뒤 15분(5분 칸 3개) 동안 모델이 계속 '비어있음'
     - 예약 없는 사용: 일정이 없는데 모델이 15분 이상 계속 '사용중'
     정답: 같은 칸의 직접 관찰(실제 인원) — 시험 기간 평일 09~19시
산출: results/t5b/advanced.md
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from tasks.t5_state import Norm, metrics, proba, train_clf
from tsi import common as C
from tsi import data as D
from tsi import state as S

RUN = 3  # 15분 = 5분 칸 3개


def runs(mask, space):
    """공간별로 mask 가 RUN칸 이상 이어진 구간의 시작 위치."""
    out = []
    for sp in np.unique(space):
        idx = np.where(space == sp)[0]
        m = mask[idx]
        k = 0
        for j, v in enumerate(m):
            k = k + 1 if v else 0
            if k == RUN:
                out.append(idx[j - RUN + 1])
    return np.array(out, int)


def main():
    a = C.args("과제 5 심화")
    raw, df, bounds, fmask = C.prepare(a)
    cond = C.conditions(raw, bounds)
    pm = lambda n: D.period_mask(df.ts, bounds, n).values & ~fmask & df.co2.notna().values
    F = S.occupancy_features(df)
    sched_y = df.sched_active.values.astype(int)
    obs_y = S.occupancy_truth(df, raw["occupancy"])
    ok = F[["co2_d30", "co2_above_min2h"]].notna().all(axis=1).values
    tr, va, te = pm("train") & ok, pm("val") & ok, pm("test") & ok
    hours, wd = df.ts.dt.hour.values, df.ts.dt.weekday.values
    observed = te & (wd < 5) & (hours >= 9) & (hours < 19)
    space = df.space.values

    def fit_eval(train_mask, val_mask):
        norm = Norm(F.values[train_mask])
        X = norm(F.values)
        m = train_clf(X[train_mask], sched_y[train_mask])
        pv = proba(m, X[val_mask])
        th = max(np.linspace(0.2, 0.8, 25), key=lambda t: f1_score(sched_y[val_mask], pv >= t))
        return (proba(m, X) >= th).astype(int)

    p_all = fit_eval(tr, va)
    rows = []
    for sp in np.unique(space):
        out_sp = space != sp
        p_lo = fit_eval(tr & out_sp, va & out_sp)
        m = observed & (space == sp)
        a_ = metrics(obs_y[m], p_all[m], "")
        b_ = metrics(obs_y[m], p_lo[m], "")
        rows.append(dict(시험_공간=sp, 칸=int(m.sum()), 함께_학습_F1=a_["F1"], 처음_보는_공간_F1=b_["F1"],
                         함께_학습_빈방오판=a_["빈방을_사용중으로"], 처음_보는_공간_빈방오판=b_["빈방을_사용중으로"],
                         F1_차이=round(b_["F1"] - a_["F1"], 1)))
        print(rows[-1], flush=True)

    # 2) 노쇼·예약 없는 사용 (모든 공간으로 학습한 모델 p_all 사용)
    in_obs = observed
    sched_start = np.r_[False, (sched_y[1:] == 1) & ((sched_y[:-1] == 0) | (space[1:] != space[:-1]))]
    ev = []
    for i in np.where(sched_start & in_obs)[0]:
        j = np.arange(i, i + RUN)
        if j[-1] >= len(df) or (space[j] != space[i]).any() or not in_obs[j].all():
            continue
        ev.append(dict(i=i, 규칙=bool((p_all[j] == 0).all()), 정답=bool((obs_y[j] == 0).all())))
    ns = pd.DataFrame(ev)
    unsched_rule = runs(in_obs & (sched_y == 0) & (p_all == 1), space)
    unsched_true = runs(in_obs & (sched_y == 0) & (obs_y == 1), space)
    hit_rule = np.isin(unsched_rule, np.concatenate([unsched_true + k for k in range(-RUN, RUN + 1)]))
    hit_true = np.isin(unsched_true, np.concatenate([unsched_rule + k for k in range(-RUN, RUN + 1)]))

    def pr(tp, n_pred, n_true):
        return (f"{tp / n_pred * 100:.0f}% ({tp}/{n_pred})" if n_pred else "-",
                f"{tp / n_true * 100:.0f}% ({tp}/{n_true})" if n_true else "-")

    tp_ns = int((ns.규칙 & ns.정답).sum())
    p_ns, r_ns = pr(tp_ns, int(ns.규칙.sum()), int(ns.정답.sum()))
    p_un, _ = pr(int(hit_rule.sum()), len(unsched_rule), len(unsched_true))
    _, r_un = pr(int(hit_true.sum()), len(unsched_rule), len(unsched_true))
    t2 = pd.DataFrame([dict(규칙="노쇼: 일정 시작 뒤 15분 계속 '비어있음'", 일정_시작_수=len(ns), 실제_노쇼=int(ns.정답.sum()),
                            정밀도=p_ns, 재현율=r_ns),
                       dict(규칙="예약 없는 사용: 일정 없는데 15분 이상 '사용중'", 일정_시작_수="-",
                            실제_노쇼=f"실제 예약 없는 사용 {len(unsched_true)}건", 정밀도=p_un, 재현율=r_un)])
    md = ["# 과제 5 심화\n", C.cond_md(cond),
          "정답: 시험 기간 평일 09~19시 직접 관찰(합성 데이터의 실제 인원). 모델·임계값은 학습·검증 기간으로만 정했다.\n",
          "## 1. 처음 보는 공간의 사용 여부 추정\n",
          "'처음 보는 공간' = 그 공간을 학습과 임계값 선택에서 모두 뺀 모델. 같은 시험 칸에서 비교. F1·빈방오판(%) = "
          "'비어있음'을 '사용중'으로 판단한 비율\n", pd.DataFrame(rows).to_markdown(index=False), "",
          "## 2. 노쇼·예약 없는 사용 찾기\n",
          "예약 없는 사용은 규칙이 찾은 구간과 실제 구간의 시작이 15분 안에 있으면 맞은 것으로 셌다.\n",
          t2.to_markdown(index=False), ""]
    C.save(os.path.join(a.out, "t5b"), "advanced", dict(conditions=cond, loo=rows, rules=t2.to_dict("records")),
           "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
