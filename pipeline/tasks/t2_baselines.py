"""과제 2. 데이터 진단과 기준선.

산출: results/t2/diagnosis.md (+ json, 그래프), results/t2/baselines.md
- 결측률·제거된 이상값, 평일 시간대별 평균 패턴, 일정 변화 뒤 CO2 반응 시간, 일정과 어긋난 날
- 기준선 3종(현재값 유지, 최근 1시간 평균, 어제 같은 시각)의 5·15·30·60분 MAE
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tsi import common as C
from tsi import data as D

plt.rcParams["font.family"] = ["NanumSquare", "Noto Sans CJK KR", "Noto Serif CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def baseline_preds(df, idx, target):
    """기준선 예측: 반환 {이름: (n, HORIZON)}"""
    y = df[target].values
    H = D.HORIZON
    persist = np.repeat(y[idx][:, None], H, axis=1)
    hour_mean = np.stack([np.nanmean(y[i - 11:i + 1]) for i in idx])[:, None].repeat(H, axis=1)
    day = 24 * 60 // D.STEP_MIN
    yest = np.stack([y[i + 1 - day:i + 1 + H - day] if i + 1 - day >= 0 else np.full(H, np.nan) for i in idx])
    # 어제 같은 시각이 다른 공간 행으로 넘어가지 않게: 같은 공간인지 확인
    same = df.space.values[np.clip(idx + 1 - day, 0, None)] == df.space.values[idx]
    yest[~same] = np.nan
    return {"현재값 유지": persist, "최근 1시간 평균": hour_mean, "어제 같은 시각": yest}


def true_future(df, idx, target):
    y = df[target].values
    return np.stack([y[i + 1:i + 1 + D.HORIZON] for i in idx])


def evaluate_baselines(df, idx, by_space=False):
    rows = []
    for target in D.TARGETS:
        yt = true_future(df, idx, target)
        for name, p in baseline_preds(df, idx, target).items():
            for hmin, k in D.EVAL_STEPS.items():
                err = np.abs(p[:, k - 1] - yt[:, k - 1])
                if by_space:
                    for sp in np.unique(df.space.values[idx]):
                        m = df.space.values[idx] == sp
                        rows.append(dict(model=name, target=target, h_min=hmin, space=sp, mae=float(np.nanmean(err[m]))))
                else:
                    rows.append(dict(model=name, target=target, h_min=hmin, mae=float(np.nanmean(err)),
                                     n=int(np.isfinite(err).sum())))
    return rows


def reaction_times(df1, schedule, rise=100, max_min=60):
    """일정 시작 뒤 CO2가 시작 시점보다 rise ppm 오르기까지 걸린 분(1분 데이터)."""
    out = []
    g = {sp: x.set_index("ts")["co2"] for sp, x in df1.groupby("space")}
    for r in schedule.itertuples():
        s = g.get(r.space)
        if s is None:
            continue
        base = s.loc[r.start - pd.Timedelta(minutes=5): r.start].median()
        after = s.loc[r.start: r.start + pd.Timedelta(minutes=max_min)]
        if np.isnan(base) or after.empty:
            continue
        hit = after[after > base + rise]
        out.append(dict(space=r.space, kind=r.kind, start=r.start,
                        minutes=(hit.index[0] - r.start).total_seconds() / 60 if len(hit) else np.nan))
    return pd.DataFrame(out)


def main():
    a = C.args("과제 2")
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t2" + a.tag)
    os.makedirs(out, exist_ok=True)
    cond = C.conditions(raw, bounds)
    md = [f"# 과제 2. 데이터 진단서\n", C.cond_md(cond)]

    # 1. 정제 규칙과 결측
    s1, removed = D.clean(raw["sensors"])
    t0, t1 = raw["sensors"].ts.min(), raw["sensors"].ts.max()
    expected = int((t1 - t0).total_seconds() // 60) + 1
    miss = (1 - s1.groupby("space")[D.SENSORS].count() / expected) * 100
    md += ["## 1. 정제 규칙\n",
           "| 상황 | 처리 | 이유 |\n|---|---|---|",
           f"| 물리적으로 불가능한 값 | 결측으로 바꿈 (CO2 {D.VALID['co2']}, 온도 {D.VALID['temperature']}, 습도 {D.VALID['humidity']}) | 0ppm·65535·-40℃는 센서·통신 오류. 그대로 두면 평균과 학습을 크게 왜곡 |",
           f"| 짧은 결측 (≤{D.MAX_FILL_STEPS * D.STEP_MIN}분) | 선형 보간 | CO2·온도는 몇 분 안에 급변하지 않아 보간 오차가 잡음 수준 |",
           f"| 긴 결측 (>{D.MAX_FILL_STEPS * D.STEP_MIN}분) | 결측 유지, 그 구간을 지나는 학습·평가 창은 버림 | 수 시간을 지어내면 모델이 가짜 패턴을 배움 |",
           f"| 센서마다 다른 기록 주기 | {D.STEP} 평균으로 맞춤 | 아래 '분석 간격' 참고 |\n",
           f"**제거된 이상값**: " + ", ".join(f"{k} {v}건" for k, v in removed.items()) + "\n",
           "**공간·항목별 결측률(1분 기준, %)**\n", miss.round(2).to_markdown(), "",
           "## 2. 분석 간격\n",
           f"{D.STEP}으로 모은다. 1분 값은 측정 잡음(CO2 ±8ppm)이 커서 변화량이 묻히고, "
           "15분 이상으로 모으면 수업 시작 직후 오르는 모습(아래 반응 시간 참고)이 뭉개진다. "
           "60분 예측이 12칸이 되어 모델 출력 크기도 적당하다.\n",
           "## 3. 학습·검증·시험 분할\n",
           "| 구간 | 기간 |\n|---|---|"] + [f"| {k} | {v} |" for k, v in cond["period"].items()] + [
           "\n시간 순서대로 자른다. 무작위로 섞으면 같은 수업 시간의 바로 앞뒤 5분이 학습과 시험에 나뉘어 들어가 "
           "모델이 '정답 바로 옆 값'을 외워 성능이 부풀려진다. 정규화 평균·표준편차는 학습 기간으로만 계산한다. "
           "시험 기간(마지막 주)에는 고장이 심어져 있으므로, 예측 평가에서는 고장 구간과 그 뒤 60분을 뺀다.\n"]

    # 2. 평일 시간대별 패턴
    wd = df[(df.ts.dt.weekday < 5) & D.period_mask(df.ts, bounds, "train")]
    pat = wd.groupby([wd.space, wd.ts.dt.hour + wd.ts.dt.minute / 60]).co2.mean().unstack(0)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    pat.plot(ax=ax[0]); ax[0].set_title("평일 시간대별 평균 CO2 (학습 기간)"); ax[0].set_xlabel("시"); ax[0].axhline(1000, ls="--", c="gray")
    wd.groupby([wd.space, wd.ts.dt.hour + wd.ts.dt.minute / 60]).temperature.mean().unstack(0).plot(ax=ax[1])
    ax[1].set_title("평일 시간대별 평균 온도"); ax[1].set_xlabel("시")
    fig.tight_layout(); fig.savefig(os.path.join(out, "weekday_pattern.png"), dpi=110); plt.close(fig)
    md += ["## 4. 평일 시간대별 패턴\n", "![평일 패턴](weekday_pattern.png)\n",
           "공간별 시간대 평균 CO2 (ppm)\n",
           pat.loc[[7, 9, 10.5, 12, 12.5, 13.5, 15, 17, 18.5, 20]].round(0).to_markdown(), ""]

    # 3. 일정 → CO2 반응
    rt = reaction_times(s1.dropna(subset=["co2"]), raw["schedule"])
    summ = rt.groupby(["space", "kind"]).minutes.agg(["count", "median", lambda x: x.isna().mean() * 100])
    summ.columns = ["일정 수", "반응 시간 중앙값(분)", "60분 안에 반응 없음(%)"]
    md += ["## 5. 일정 변화 뒤 CO2 반응\n",
           "일정 시작 시점보다 CO2가 100ppm 오를 때까지 걸린 시간\n", summ.round(1).to_markdown(), ""]
    # 일정과 실제가 어긋난 날: 수업이 있는데 오르지 않았거나(휴강·결석), 일정이 없는데 오른 날
    day = df[df.ts.dt.weekday < 5].assign(d=lambda x: x.ts.dt.date)
    dmax = day.groupby(["space", "d"]).co2.max()
    dsched = raw["schedule"].assign(d=lambda x: x.start.dt.date).groupby(["space", "d"]).size()
    j = pd.concat([dmax.rename("max_co2"), dsched.rename("n_sched")], axis=1).fillna({"n_sched": 0})
    odd = j[((j.n_sched > 0) & (j.max_co2 < 600)) | ((j.n_sched == 0) & (j.max_co2 > 800))]
    md += ["**일정과 실제가 어긋난 날** (일정이 있는데 최고 CO2 600 미만, 또는 일정이 없는데 800 초과)\n",
           odd.round(0).to_markdown() if len(odd) else "없음", "",
           f"반응 없음 비율은 노쇼·결석·조기 종료를, 반응 시간은 '일정 시작 시각'을 특징으로 쓸 때 "
           f"몇 칸 뒤부터 효과가 나타나는지를 알려 준다.\n"]

    # 4. 기준선
    idx_val = D.anchors(df, bounds, "val", exclude=fmask)
    idx_test = D.anchors(df, bounds, "test", exclude=fmask)
    rows = evaluate_baselines(df, idx_test)
    tab = C.mae_table(rows)
    by_sp = pd.DataFrame(evaluate_baselines(df, idx_test, by_space=True))
    sp60 = by_sp[by_sp.h_min == 60].pivot_table(index=["target", "model"], columns="space", values="mae").round(1)
    bmd = ["# 과제 2. 기준선 성능표\n", C.cond_md(cond),
           f"시험 기간, 고장 구간 제외, 평가 시점 {len(idx_test):,}개 (모든 모델이 같은 시점에서 평가). 값은 MAE (CO2 ppm, 온도 ℃)\n",
           tab.to_markdown(), "\n**60분 뒤 MAE, 공간별**\n", sp60.to_markdown(), ""]
    best = {t: min(((r["model"], r["mae"]) for r in rows if r["target"] == t and r["h_min"] == 60), key=lambda x: x[1])
            for t in D.TARGETS}
    bmd.append("**60분 뒤 가장 강한 기준선**: " + ", ".join(f"{t} → {m} ({v:.2f})" for t, (m, v) in best.items()))
    co2_60 = sp60.loc["co2"]
    ratio = (co2_60.loc["어제 같은 시각"] / co2_60.loc["현재값 유지"]).sort_values()
    bmd.append(f"\n'어제 같은 시각'은 요일·일정이 매일 달라(회의실 예약, 휴강·결석, 월요일의 어제=일요일) 가장 약하다. "
               f"현재값 유지 대비 오차 배율은 {', '.join(f'{k} {v:.1f}배' for k, v in ratio.items())}로, "
               f"매일 비슷한 시각에 비슷하게 붐비는 공간({ratio.index[0]})에서 가장 덜 나쁘다.")
    md += ["## 6. 기준선\n", "→ [baselines.md](baselines.md)"]
    C.save(out, "baselines", dict(conditions=cond, overall=rows, by_space=by_sp.to_dict("records"),
                                  n_anchors_test=len(idx_test), n_anchors_val=len(idx_val)), "\n".join(bmd) + "\n")
    C.save(out, "diagnosis", dict(conditions=cond, removed=removed, missing_pct=miss.round(3).to_dict()),
           "\n".join(md) + "\n")
    print("\n".join(bmd))


if __name__ == "__main__":
    main()
