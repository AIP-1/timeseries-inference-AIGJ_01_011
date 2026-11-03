"""과제 4. 이상 탐지 — 오토인코더(정상만 학습) + 규칙(튄 값·값 고정) + 예측 오차 탐지기(심화).

- 오토인코더는 학습 기간(고장 없음)으로만 학습
- 경보 기준(임계값)은 검증 기간(고장 없음)의 헛경보가 하루 목표 이하가 되도록 정함 → 시험 기간 고장으로 평가
- 고장 종류별 탐지율·지연·헛경보, 원인 센서 일치율, 임계값을 바꿀 때의 변화
산출: results/t4/anomaly.md, artifacts/ae.pt + ae.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tsi import anomaly as A
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import AutoEncoder, make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
FA_TARGET_PER_DAY = 0.3  # 탐지기 하나당 검증 기간 헛경보 목표(전체 공간 합). 규칙과 합쳐도 하루 1회 이하가 되게


def train_ae(X, epochs=60, seed=0):
    C.set_seed(seed)
    m = AutoEncoder(X.shape[1])
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xt = torch.tensor(X)
    for _ in range(epochs):
        perm = torch.randperm(len(Xt))
        for b in range(0, len(Xt), 256):
            j = perm[b:b + 256]
            loss = ((m(Xt[j]) - Xt[j]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def forecast_residual(df, spec_d, kind, idx):
    """30분 전에 내놓은 예측과 지금 값의 차이(정규화). 반환: 점수(N,), 센서별(N,3; 습도는 0)."""
    from tasks.t3_forecast import predict
    m = make_forecaster(kind, len(spec_d.past), len(spec_d.future), D.HORIZON)
    m.load_state_dict(torch.load(os.path.join(ART, f"forecast_{kind}.pt")))
    m.eval()
    k = 6
    src = idx - k  # 30분 전 시점
    ok = df.space.values[np.clip(src, 0, None)] == df.space.values[idx]
    src, idx = src[ok], idx[ok]
    pred = predict(m, spec_d, df, src)[:, k - 1, :]
    actual = df[D.TARGETS].to_numpy(np.float32)[idx]
    # 순간 오차(수업 시작 직후 등)는 정상에서도 크다. 30분 동안 '예측보다 계속 높게' 나오는지를 본다
    # (냉방·환기 정지와 영점 틀어짐은 모두 값을 올리는 고장)
    z = np.full((len(df), 2), np.nan, np.float32)
    z[idx] = (actual - pred) / spec_d.dsd
    zs = (pd.DataFrame(z).groupby(df.space.values).rolling(6, min_periods=4).mean()
          .reset_index(level=0, drop=True).sort_index().to_numpy(np.float32))[idx]
    zs = np.clip(np.nan_to_num(zs), 0, None)
    per = np.concatenate([zs, np.zeros((len(zs), 1), np.float32)], axis=1)
    return per.max(axis=1), per, idx


def alarms_from_scores(df, pos, score, per, th, name):
    hit = score > th
    sensors = np.array(D.SENSORS)[per[hit].argmax(axis=1)] if hit.any() else []
    return pd.DataFrame(dict(ts=df.ts.values[pos[hit]], space=df.space.values[pos[hit]], detector=name, sensor=sensors))


def pick_threshold(df, pos, score, per, name, val_days, target=FA_TARGET_PER_DAY):
    """검증 기간 헛경보가 목표 이하가 되는 가장 낮은 임계값."""
    for q in np.linspace(0.95, 0.99999, 400):
        th = float(np.quantile(score, q))
        ev = A.to_events(alarms_from_scores(df, pos, score, per, th, name))
        if len(ev) / val_days <= target:
            return th
    return float(score.max())


def evaluate(alarms, faults, days):
    ev = A.to_events(alarms) if len(alarms) else pd.DataFrame(columns=["space", "start", "end", "sensor"])
    res, fa = A.match(ev, faults)
    return res, fa / days


def main():
    a = C.args("과제 4", forecast_model="gru")
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t4")
    cond = C.conditions(raw, bounds)
    pm = lambda name: D.period_mask(df.ts, bounds, name).values
    days = {k: (v[1] - v[0]).total_seconds() / 86400 for k, v in bounds.items()}
    faults = raw["faults"]
    train_df = df[pm("train")]
    spec = FeatureSpec(train_df)

    # 오토인코더
    X, pos = A.ae_matrix(df, spec)
    in_p = lambda name: pm(name)[pos]
    ae = train_ae(X[in_p("train")])
    s_all, per_all = A.ae_scores(ae, X)
    va, te = in_p("val"), in_p("test")
    th_ae = pick_threshold(df, pos[va], s_all[va], per_all[va], "AE", days["val"])
    torch.save(ae.state_dict(), os.path.join(ART, "ae.pt"))
    with open(os.path.join(ART, "ae.json"), "w", encoding="utf-8") as f:
        json.dump(dict(n_in=X.shape[1], threshold=th_ae, spec=spec.to_dict(), steps=A.AE_STEPS), f)

    # 예측 오차 탐지기 (심화 B)
    spec_d = FeatureSpec.from_dict(json.load(open(os.path.join(ART, f"forecast_{a.forecast_model}.json")))["spec"])
    idx_all = np.where(df[D.TARGETS].notna().all(axis=1).values)[0]
    idx_all = idx_all[idx_all >= D.LOOKBACK + 6]
    fr_s, fr_per, fr_pos = forecast_residual(df, spec_d, a.forecast_model, idx_all)
    fva, fte = pm("val")[fr_pos], pm("test")[fr_pos]
    th_fr = pick_threshold(df, fr_pos[fva], fr_s[fva], fr_per[fva], "FR", days["val"])

    # 규칙 (1분 원자료, 범위 밖 값은 정제 후)
    s1, _ = D.clean(raw["sensors"])
    rules = A.rules_1min(s1)
    rules = rules.rename(columns={"rule": "detector"})

    def window(al, name):
        a0, a1 = bounds[name]
        return al[(al.ts >= a0) & (al.ts < a1)]

    ae_al = alarms_from_scores(df, pos[te], s_all[te], per_all[te], th_ae, "AE")
    fr_al = alarms_from_scores(df, fr_pos[fte], fr_s[fte], fr_per[fte], th_fr, "FR")
    rule_te = window(rules, "test")
    combos = {"모델(AE)": ae_al, "규칙": rule_te, "AE+규칙": pd.concat([ae_al, rule_te]),
              "예측오차(FR)": fr_al, "AE+FR+규칙": pd.concat([ae_al, fr_al, rule_te])}
    tables, summary = {}, []
    for name, al in combos.items():
        res, fa_day = evaluate(al, faults, days["test"])
        tables[name] = res
        summary.append(dict(탐지기=name, 탐지율=f"{res.detected.mean() * 100:.0f}% ({res.detected.sum()}/{len(res)})",
                            지연_중앙값_분=round(res.delay_min.median(), 1), 하루_헛경보=round(fa_day, 2),
                            원인_센서_일치=f"{res.cause_ok[res.detected].mean() * 100:.0f}%" if res.detected.any() else "-"))
    by_kind = pd.DataFrame({name: t.groupby("kind").detected.apply(lambda x: f"{x.sum()}/{len(x)}")
                            for name, t in tables.items()})
    delay_kind = pd.DataFrame({name: t.groupby("kind").delay_min.median().round(0) for name, t in tables.items()})
    rule_val_fa = len(A.to_events(window(rules, "val"))) / days["val"]

    # 임계값을 바꿀 때 (AE, 시험 기간)
    sweep = []
    for mult in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        al = alarms_from_scores(df, pos[te], s_all[te], per_all[te], th_ae * mult, "AE")
        res, fa_day = evaluate(al, faults, days["test"])
        sweep.append(dict(임계값_배율=mult, 임계값=round(th_ae * mult, 3), 탐지율=f"{res.detected.mean() * 100:.0f}%",
                          하루_헛경보=round(fa_day, 2)))

    best = tables["AE+규칙"]
    miss_model = sorted(set(tables["모델(AE)"].query("not detected").kind))
    miss_rule = sorted(set(tables["규칙"].query("not detected").kind))
    md = ["# 과제 4. 이상 탐지\n", C.cond_md(cond),
          f"시험 기간 고장 {len(faults)}건(5종). 경보는 공간별로 30분 안에 이어지면 한 사건으로 묶는다. "
          f"고장 시작~끝+30분 안의 경보를 탐지로, 어떤 고장과도 겹치지 않는 경보 사건을 헛경보로 센다.\n",
          "## 탐지 성적표\n", pd.DataFrame(summary).to_markdown(index=False), "",
          "**고장 종류별 탐지 건수**\n", by_kind.to_markdown(), "",
          "**고장 종류별 탐지 지연 중앙값(분)**\n", delay_kind.to_markdown(), "",
          "## 확인 기준\n",
          f"- 모델+규칙 탐지율 {best.detected.mean() * 100:.0f}% (기준 80%) {'✅' if best.detected.mean() >= 0.8 else '❌'}",
          f"- 헛경보 하루 {[s for s in summary if s['탐지기'] == 'AE+규칙'][0]['하루_헛경보']}회 (기준 1회 이하)",
          f"- 모델이 못 잡은 종류: {', '.join(miss_model) or '없음'} / 규칙이 못 잡은 종류: {', '.join(miss_rule) or '없음'}\n",
          "## 경보 기준(임계값) 정하는 법\n",
          f"오토인코더 복원 오차의 임계값은 **고장이 없는 검증 기간**에서 헛경보가 하루 {FA_TARGET_PER_DAY}회 이하가 되는 "
          f"가장 낮은 값({th_ae:.3f})으로 정했다. 임계값을 낮추면 더 일찍·더 많이 잡지만 헛경보가 늘고, 높이면 그 반대다. "
          f"규칙은 검증 기간 헛경보 하루 {rule_val_fa:.2f}회.\n",
          pd.DataFrame(sweep).to_markdown(index=False), "",
          "## 해석\n",
          "- **값 고정**: 정상 범위 값에서 멈추면 복원 오차가 거의 없어 모델은 못 본다. '같은 값이 N분' 규칙이 바로 잡는다.",
          "- **튄 값**: 한 칸(1분)만 튀면 5분 평균에서 1/5로 줄어 모델이 놓치기 쉽다. 1분 데이터의 급변 규칙이 잡는다.",
          "- **영점 틀어짐**: 몇 시간에 걸쳐 천천히 오르므로 급변 규칙은 못 잡는다. 시간대를 함께 넣은 모델은 "
          "'새벽 3~8시에 CO2가 외기(420)보다 수백 ppm 높다'는 이상함으로 잡는다.",
          "- **냉방·환기 정지**: 값 하나하나는 정상 범위라 규칙은 못 잡고, '수업 중인데 온도가 오른다', "
          "'평소보다 CO2가 가파르게 오른다'는 패턴을 모델·예측오차가 잡는다. 빈 방에서 환기가 멈추면 "
          "CO2가 오르지 않아 탐지할 수 없고, 사람이 없으니 급히 탐지할 필요도 낮다.",
          "- 원인 센서는 센서별 복원 오차가 가장 큰 센서로 가리킨다.\n",
          "## 운영 제안\n",
          "- 값 고정·데이터 끊김: 시설 담당자에게 즉시 알림(센서 점검)",
          "- 냉방·환기 정지 의심: 해당 공간 담당자 + 시설팀, 같은 사건은 30분 동안 다시 보내지 않음",
          "- 영점 틀어짐 의심: 새벽 시간대 판정이므로 아침 업무 시작 때 한 번 모아서 알림\n",
          "**고장별 결과(AE+규칙)**\n", best.to_markdown(index=False)]
    C.save(out, "anomaly", dict(conditions=cond, summary=summary, threshold_ae=th_ae, threshold_fr=th_fr,
                                per_fault={k: v.to_dict("records") for k, v in tables.items()}, sweep=sweep),
           "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
