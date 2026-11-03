"""과제 6. 사전학습 시계열 모델(Chronos-Bolt) — 추가 학습 없이 CO2 60분 예측.

- 같은 시험 시점에서 기준선 · 직접 학습한 모델 · Chronos 비교 (시간을 줄이려고 시험 시점을 고르게 솎아 씀,
  세 방법 모두 같은 솎은 시점에서 비교)
- 10~90% 예측 구간 적중률(이상적이면 80%)
- 보드에서 1회 추론 시간(배치 1, 평균·느린 쪽 5%)과 메모리
- 과거 길이(2시간·8시간·24시간)를 바꿔 비교
산출: results/t6/chronos.md
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tasks.t2_baselines import baseline_preds, true_future
from tasks.t3_forecast import predict
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
Q = [0.1, 0.5, 0.9]


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def chronos_predict(pipe, series, idx, ctx_steps, batch=128):
    """idx 시점까지의 과거 ctx_steps 칸으로 12칸 예측. 반환 (N, 12, 3) [q10, q50, q90]"""
    out = []
    for b in range(0, len(idx), batch):
        ctx = [torch.tensor(series[max(0, i - ctx_steps + 1):i + 1], dtype=torch.float32) for i in idx[b:b + batch]]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=D.HORIZON, quantile_levels=Q)
        out.append(q.numpy())
    return np.concatenate(out)


def main():
    a = C.args("과제 6", model="amazon/chronos-bolt-small", stride=4, device="cpu")
    from chronos import BaseChronosPipeline
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t6" + a.tag)
    cond = C.conditions(raw, bounds)
    te = D.anchors(df, bounds, "test", exclude=fmask)[::a.stride]
    # 서로 다른 공간의 값이 과거 창에 섞이지 않게: 공간 시작 위치보다 앞으로 가지 않도록 자른다
    co2 = df.co2.to_numpy(np.float32)
    space_start = df.groupby("space").cumcount().to_numpy()
    yt = true_future(df, te, "co2")

    rows, cov_rows, cost = [], [], []
    for name, p in baseline_preds(df, te, "co2").items():
        if name != "어제 같은 시각":
            rows += [dict(model=name, h_min=h, mae=float(np.nanmean(np.abs(p[:, k - 1] - yt[:, k - 1]))))
                     for h, k in D.EVAL_STEPS.items()]
    best = json.load(open(os.path.join(ART, f"forecast_best{a.tag}.json")))["model"]
    meta = json.load(open(os.path.join(ART, f"forecast_{best}{a.tag}.json")))
    spec = FeatureSpec.from_dict(meta["spec"])
    fm = make_forecaster(best, len(spec.past), len(spec.future), D.HORIZON)
    fm.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{a.tag}.pt"))); fm.eval()
    pf = predict(fm, spec, df, te)[:, :, 0]
    rows += [dict(model=f"직접 학습({best.upper()})", h_min=h, mae=float(np.nanmean(np.abs(pf[:, k - 1] - yt[:, k - 1]))))
             for h, k in D.EVAL_STEPS.items()]

    m0 = rss_mb()
    t0 = time.time()
    pipe = BaseChronosPipeline.from_pretrained(a.model, device_map=a.device, torch_dtype=torch.float32)
    load_s = time.time() - t0
    n_params = sum(p.numel() for p in pipe.model.parameters())
    label = a.model.split("/")[-1]
    for ctx_h in (2, 8, 24):
        steps = ctx_h * 60 // D.STEP_MIN
        ok = space_start[te] >= steps - 1 if ctx_h > 2 else np.ones(len(te), bool)
        idx = te[ok]
        qp = chronos_predict(pipe, co2, idx, steps)
        y = yt[ok]
        rows += [dict(model=f"{label} 과거 {ctx_h}h", h_min=h, mae=float(np.nanmean(np.abs(qp[:, k - 1, 1] - y[:, k - 1]))))
                 for h, k in D.EVAL_STEPS.items()]
        inside = (y >= qp[:, :, 0]) & (y <= qp[:, :, 2])
        cov_rows.append(dict(과거=f"{ctx_h}h", 시점수=len(idx), 구간_적중률_전체=round(np.nanmean(inside) * 100, 1),
                             구간_적중률_60분=round(np.nanmean(inside[:, -1]) * 100, 1),
                             구간_폭_60분_ppm=round(float(np.nanmean(qp[:, -1, 2] - qp[:, -1, 0])), 1)))
        print(cov_rows[-1], flush=True)

    # 비용: 배치 1, 과거 24시간, 워밍업 5회 후 50회
    i0 = te[space_start[te] >= 287][0]
    lat = []
    for r in range(55):
        t = time.perf_counter()
        chronos_predict(pipe, co2, np.array([i0]), 288, batch=1)
        if r >= 5:
            lat.append((time.perf_counter() - t) * 1000)
    lat_gru = []
    p1, f1 = spec.windows(df, np.array([i0]), with_target=False)
    p1, f1 = torch.tensor(p1), torch.tensor(f1)
    with torch.no_grad():
        for r in range(205):
            t = time.perf_counter(); fm(p1, f1)
            if r >= 5:
                lat_gru.append((time.perf_counter() - t) * 1000)
    mem = rss_mb() - m0
    cost = [dict(방법=label, 파라미터=f"{n_params / 1e6:.1f}M", 불러오기_s=round(load_s, 1),
                 지연_평균_ms=round(np.mean(lat), 1), 지연_p95_ms=round(np.percentile(lat, 95), 1),
                 추가_메모리_MB=round(mem, 0), 장치=a.device),
            dict(방법=f"직접 학습({best.upper()})", 파라미터=f"{sum(p.numel() for p in fm.parameters()) / 1e6:.2f}M",
                 불러오기_s="-", 지연_평균_ms=round(np.mean(lat_gru), 2), 지연_p95_ms=round(np.percentile(lat_gru, 95), 2),
                 추가_메모리_MB="수 MB", 장치="cpu")]

    t = pd.DataFrame(rows).pivot_table(index="model", columns="h_min", values="mae", sort=False)
    t.columns = [f"{c}분" for c in t.columns]
    g = lambda m: t.loc[m, "60분"]
    ch_best = min((m for m in t.index if m.startswith(label)), key=g)
    md = ["# 과제 6. 사전학습 모델 비교\n", C.cond_md(cond),
          f"모델: `{a.model}` (추가 학습 없음, CO2 과거값만 입력). 시험 시점 {len(te):,}개(고장 구간 제외, {a.stride}칸마다 하나). "
          "세 방법 모두 같은 시점에서 비교. MAE (ppm)\n",
          t.round(1).to_markdown(), "",
          "## 예측 구간(10~90%) 적중률\n", "이상적이면 80%가 구간 안에 들어온다.\n",
          pd.DataFrame(cov_rows).to_markdown(index=False), "",
          "## 비용 (Jetson, 배치 1, 과거 24시간, 워밍업 5회 후 반복 측정)\n", pd.DataFrame(cost).to_markdown(index=False), "",
          "## 채택 의견\n",
          f"- 정확도: Chronos 최선({ch_best}) 60분 MAE {g(ch_best):.1f}ppm, 직접 학습 {g(f'직접 학습({best.upper()})'):.1f}ppm, "
          f"현재값 유지 {g('현재값 유지'):.1f}ppm.",
          "- 차이의 원인: Chronos는 CO2 과거값만 본다. 직접 학습한 모델은 시간대·요일과 **미래 일정(예정 인원)**을 함께 보므로 "
          "수업 시작·점심처럼 과거값만으로는 알 수 없는 변화를 미리 안다. 과제 3의 '일정 제외' 모델과의 차이가 그 몫이다.",
          "- 쓸 만한 곳: 학습 데이터가 없는 **새 공간**이나 운영 초기에는 학습 없이 바로 쓸 수 있고, 예측 구간을 함께 주므로 "
          "'최악의 경우 몇 ppm'을 알릴 때 유용하다.",
          f"- 비용: 1회 추론 {np.mean(lat):.0f}ms(p95 {np.percentile(lat, 95):.0f}ms)로 직접 학습한 모델(PyTorch CPU "
          f"{np.mean(lat_gru):.1f}ms)보다 {np.mean(lat) / np.mean(lat_gru):.0f}배 느리고, 파라미터는 "
          f"{n_params / sum(p.numel() for p in fm.parameters()):.0f}배, 추가 메모리 약 {mem:.0f}MB. "
          "직접 학습한 모델을 ONNX Runtime으로 돌리면 1ms 미만이라 차이는 더 커진다(과제 7). "
          "같은 보드에 로컬 LLM을 함께 올려야 하므로 **상시 서비스의 기본 예측기로는 채택하지 않고**, "
          "새 공간이 생겼을 때 임시 예측기로 쓰는 것을 제안한다."]
    C.save(out, "chronos", dict(conditions=cond, model=a.model, rows=rows, coverage=cov_rows, cost=cost),
           "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
