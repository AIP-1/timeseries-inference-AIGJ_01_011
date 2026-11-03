"""과제 3. 60분 예측 (GRU·LSTM·TCN, 일정 포함/제외).

- 입력: 과거 2시간(5분 × 24) + 미래 60분의 시간대·일정, 출력: 60분 동안 5분 간격 CO2·온도 '변화량'
- 과제 2와 같은 시험 시점(고장 구간 제외)에서 기준선과 비교
- 학습 기간으로 학습, 검증 기간으로 조기 종료·모델 선택, 시험 기간은 마지막에 한 번만 평가
산출: results/t3/forecast.md, artifacts/forecast_<model>.pt + .json
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
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

ART = os.path.join(C.ROOT, "artifacts")


def fit(kind, spec, df, tr_idx, va_idx, epochs, device, seed=0):
    C.set_seed(seed)
    Xp, Xf, Y = (torch.tensor(a) for a in spec.windows(df, tr_idx))
    Vp, Vf, VY = (torch.tensor(a) for a in spec.windows(df, va_idx))
    model = make_forecaster(kind, Xp.shape[2], Xf.shape[2], D.HORIZON).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, bad = 1e9, None, 0
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Y))
        for b in range(0, len(Y), 256):
            j = perm[b:b + 256]
            loss = torch.nn.functional.smooth_l1_loss(model(Xp[j].to(device), Xf[j].to(device)), Y[j].to(device))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            v = torch.cat([model(Vp[b:b + 2048].to(device), Vf[b:b + 2048].to(device)).cpu()
                           for b in range(0, len(VY), 2048)])
        val = (v - VY).abs().mean().item()  # 정규화 단위 검증 오차 (모든 시점·두 대상 평균)
        if val < best - 1e-4:
            best, best_state, bad = val, {k: x.detach().cpu().clone() for k, x in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 5:
                break
    model.load_state_dict(best_state)
    return model.cpu().eval(), best, ep + 1


def predict(model, spec, df, idx):
    p, f = spec.windows(df, idx, with_target=False)
    with torch.no_grad():
        out = torch.cat([model(torch.tensor(p[b:b + 2048]), torch.tensor(f[b:b + 2048]))
                         for b in range(0, len(idx), 2048)]).numpy()
    now = df[D.TARGETS].to_numpy(np.float32)[idx]
    return spec.denorm(out, now)  # (N, H, 2)


def score(name, pred, df, idx, rows, by_space=None):
    for ti, target in enumerate(D.TARGETS):
        yt = true_future(df, idx, target)
        for hmin, k in D.EVAL_STEPS.items():
            err = np.abs(pred[:, k - 1, ti] - yt[:, k - 1])
            rows.append(dict(model=name, target=target, h_min=hmin, mae=float(np.nanmean(err))))
            if by_space is not None and hmin == 60:
                for sp in np.unique(df.space.values[idx]):
                    m = df.space.values[idx] == sp
                    by_space.append(dict(model=name, target=target, space=sp, mae=float(np.nanmean(err[m]))))


def main():
    a = C.args("과제 3", epochs=40, models="gru,lstm,tcn", device="cpu", train_stride=1)
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t3" + a.tag)
    cond = C.conditions(raw, bounds)
    tr = D.anchors(df, bounds, "train", exclude=fmask)[::a.train_stride]  # 실데이터는 시점이 많아 솎아서 학습
    va = D.anchors(df, bounds, "val", exclude=fmask)
    te = D.anchors(df, bounds, "test", exclude=fmask)
    train_df = df[D.period_mask(df.ts, bounds, "train")]
    print(f"anchors train {len(tr)} val {len(va)} test {len(te)}")

    rows, by_space, fits = [], [], []
    for name, p in baseline_preds(df, te, "co2").items():  # 기준선(과제 2와 같은 시점)
        pt = baseline_preds(df, te, "temperature")[name]
        score(name, np.stack([p, pt], axis=2), df, te, rows, by_space)

    os.makedirs(ART, exist_ok=True)
    variants = [(k, True) for k in a.models.split(",")] + [("gru", False)]  # 일정 제외 비교는 GRU로
    for kind, use_sched in variants:
        spec = FeatureSpec(train_df, use_schedule=use_sched)
        t0 = time.time()
        model, val, ep = fit(kind, spec, df, tr, va, a.epochs, a.device)
        name = f"{kind.upper()}" + ("" if use_sched else " (일정 제외)")
        pred = predict(model, spec, df, te)
        score(name, pred, df, te, rows, by_space)
        n_par = sum(p.numel() for p in model.parameters())
        fits.append(dict(model=name, val_loss=round(val, 4), epochs=ep, params=n_par, train_s=round(time.time() - t0, 1)))
        print(fits[-1])
        tag = f"forecast_{kind}" + ("" if use_sched else "_nosched") + a.tag
        torch.save(model.state_dict(), os.path.join(ART, f"{tag}.pt"))
        with open(os.path.join(ART, f"{tag}.json"), "w", encoding="utf-8") as f:
            json.dump(dict(kind=kind, spec=spec.to_dict(), val_loss=val, conditions=cond), f, ensure_ascii=False)

    # 검증 오차로 대표 모델을 고른다(시험 결과로 고르지 않는다)
    sched_fits = [f for f in fits if "일정 제외" not in f["model"]]
    best = min(sched_fits, key=lambda f: f["val_loss"])
    with open(os.path.join(ART, f"forecast_best{a.tag}.json"), "w", encoding="utf-8") as f:
        json.dump(dict(model=best["model"].lower()), f)

    tab = C.mae_table(rows)
    r = pd.DataFrame(rows)
    get = lambda m, t, h=60: r[(r.model == m) & (r.target == t) & (r.h_min == h)].mae.iloc[0]
    imp = {m["model"]: (1 - get(m["model"], "co2") / get("현재값 유지", "co2")) * 100 for m in fits}
    imp_t = {m["model"]: (1 - get(m["model"], "temperature") / get("현재값 유지", "temperature")) * 100 for m in fits}
    sched_gain = get("GRU (일정 제외)", "co2") - get("GRU", "co2")
    sp = pd.DataFrame(by_space)
    sp60 = sp[sp.target == "co2"].pivot_table(index="model", columns="space", values="mae").round(1)
    sp_ratio = (sp60.loc[best["model"]] / sp60.loc["현재값 유지"] * 100).round(0)
    bm = best["model"]
    md = [f"# 과제 3. 60분 예측\n", C.cond_md(cond),
          f"시험 평가 시점 {len(te):,}개(과제 2 기준선과 같은 시점, 고장 구간 제외). MAE: CO2 ppm, 온도 ℃\n",
          tab.to_markdown(), "",
          "## 확인 기준\n",
          "| 모델 | 60분 CO2 개선(현재값 유지 대비) | 기준 30% | 60분 온도 개선 | 기준 >0% |\n|---|---|---|---|---|"]
    md += [f"| {m} | {imp[m]:.1f}% | {'✅' if imp[m] >= 30 else '❌'} | {imp_t[m]:.1f}% | {'✅' if imp_t[m] > 0 else '❌'} |"
           for m in imp]
    md += ["\n## 학습 기록\n", pd.DataFrame(fits).to_markdown(index=False),
           f"\n검증 오차로 고른 대표 모델: **{bm}** (서비스·최적화에 사용)\n",
           "## 공간별 60분 CO2 MAE\n", sp60.to_markdown(),
           f"\n{bm}의 오차는 현재값 유지 대비 " + ", ".join(f"{k} {v:.0f}%" for k, v in sp_ratio.items()) + ".",
           "회의실은 예약 노쇼·연장·예약 없는 사용 때문에 일정이 실제와 자주 어긋나고, 부피가 작아 몇 명만 들어와도 "
           "CO2가 빠르게 변해 상대적으로 어렵다. 과거 센서값에는 '곧 누가 들어온다'는 정보가 없으므로 예약 정보가 핵심이다.\n",
           "## 일정 정보의 가치\n",
           f"일정(예정 인원)을 입력에서 빼면 GRU의 60분 CO2 MAE가 {get('GRU (일정 제외)', 'co2'):.1f} → "
           f"넣으면 {get('GRU', 'co2'):.1f}ppm. **일정 정보를 넣으면 60분 예측 오차가 {sched_gain:.1f}ppm "
           f"({sched_gain / get('GRU (일정 제외)', 'co2') * 100:.0f}%) 줄어든다.**\n",
           "## 설계\n",
           f"- 과거 {D.LOOKBACK * D.STEP_MIN}분, 미래 {D.HORIZON * D.STEP_MIN}분을 한 번에 출력(반복 예측은 오차가 쌓임)",
           "- 값이 아니라 '지금 값에서의 변화량'을 예측: 공간마다 다른 수준(라운지 450 vs 강의실 1400)을 모델이 외울 필요가 없고, "
           "출력 0이 곧 '현재값 유지' 기준선이라 학습이 그보다 나빠지기 어렵다",
           "- 미래 공변량: 미래 60분의 시간대와 예정 인원(일정은 미리 알 수 있음)"]
    C.save(out, "forecast", dict(conditions=cond, rows=rows, by_space=by_space, fits=fits, best=bm,
                                 schedule_gain_ppm=sched_gain), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
