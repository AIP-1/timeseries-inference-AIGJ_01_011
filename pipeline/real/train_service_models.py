"""실데이터로 서비스용 이상 탐지(AE)·사용 여부·환기 경보 모델을 학습하고 ONNX로 저장한다(꼬리표 _real).

과제 4·5의 평가(고장 정답·실제 인원)는 합성 데이터로 했다. 실데이터에는 그런 정답이 없으므로 여기서는
  - AE: 학습 기간으로 학습, 검증 기간 헛경보 하루 0.3회 이하가 되는 임계값. 시험 기간 경보 사례를 목록으로 남김
  - 환기 경보: 정답이 '30분 뒤 CO2'라 실데이터로 정확히 평가 가능 → 재현율·정밀도·macro-F1
  - 사용 여부: 실제 인원 기록이 없어 '추정 시간표'를 정답 대신 씀 → 일치율은 참고값
산출: results/real/service_models.md, artifacts/{ae,occ,vent,forecast}_real.(pt|json|onnx)
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score

from tasks.t4_anomaly import alarms_from_scores, pick_threshold, train_ae
from tasks.t5_state import Norm, metrics, proba, save_model, train_clf
from tsi import anomaly as A
from tsi import common as C
from tsi import data as D
from tsi import state as S
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
TAG = "_real"


def export_onnx(model, args, path, names):
    torch.onnx.export(model, args, path, input_names=names, output_names=["out"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in names + ["out"]}, dynamo=False)


def main():
    a = C.args("실데이터 서비스 모델")
    raw, df, bounds, _ = C.prepare(a)
    cond = C.conditions(raw, bounds)
    pm = lambda n: D.period_mask(df.ts, bounds, n).values
    days = {k: (v[1] - v[0]).total_seconds() / 86400 for k, v in bounds.items()}
    spec = FeatureSpec(df[pm("train")])
    md = ["# 실데이터 서비스 모델\n", C.cond_md(cond)]

    # ---- AE
    X, pos = A.ae_matrix(df, spec)
    inp = lambda n: pm(n)[pos]
    ae = train_ae(X[inp("train")][::2])
    s_all, per_all = A.ae_scores(ae, X)
    va, te = inp("val"), inp("test")
    th = pick_threshold(df, pos[va], s_all[va], per_all[va], "AE", days["val"])
    torch.save(ae.state_dict(), os.path.join(ART, f"ae{TAG}.pt"))
    json.dump(dict(n_in=X.shape[1], threshold=th, spec=spec.to_dict(), steps=A.AE_STEPS),
              open(os.path.join(ART, f"ae{TAG}.json"), "w"))
    export_onnx(ae.eval(), (torch.tensor(X[:2]),), os.path.join(ART, f"ae{TAG}.onnx"), ["x"])
    ev = A.to_events(alarms_from_scores(df, pos[te], s_all[te], per_all[te], th, "AE"))
    md += ["## 이상 탐지(AE)\n",
           f"임계값 {th:.4f} (검증 기간 헛경보 하루 0.3회 이하). 시험 기간 경보 사건 {len(ev)}건 "
           f"(하루 {len(ev) / days['test']:.2f}회). 실데이터에는 고장 정답이 없어 아래 사례를 사람이 확인해야 한다.\n",
           ev.assign(start=ev.start.astype(str), end=ev.end.astype(str)).to_markdown(index=False) if len(ev) else "경보 없음", ""]

    # ---- 환기 경보 (정답: 30분 뒤 CO2)
    V = S.vent_features(df)
    vy = S.vent_label(df).values
    ok = np.isfinite(vy) & V[["co2_d30", "co2_above_min2h"]].notna().all(axis=1).values
    vtr, vva, vte = pm("train") & ok, pm("val") & ok, pm("test") & ok
    vn = Norm(V.values[vtr]); VX = vn(V.values)
    pr = vy[vtr].mean()
    vent = train_clf(VX[vtr], vy[vtr], pos_weight=float((1 - pr) / max(pr, 1e-3)) ** 0.5)
    pv = proba(vent, VX[vva])
    cands = [t for t in np.linspace(0.05, 0.95, 91) if recall_score(vy[vva], pv >= t) >= 0.90]
    th_v = max(cands, key=lambda t: f1_score(vy[vva], pv >= t, average="macro")) if cands else 0.05
    save_model(vent, vn, list(V.columns), float(th_v), f"vent{TAG}", dict(limit=S.VENT_LIMIT, ahead_steps=S.VENT_AHEAD))
    export_onnx(vent.eval(), (torch.tensor(VX[:2]),), os.path.join(ART, f"vent{TAG}.onnx"), ["x"])
    vp = (proba(vent, VX) >= th_v).astype(int)
    rows = [metrics(vy[vte], vp[vte], "MLP 분류기"), metrics(vy[vte], (df.co2.values >= 850).astype(int)[vte], "규칙: 지금 ≥ 850ppm")]
    for r in rows:
        r.pop("빈방을_사용중으로")
    md += ["## 환기 선제 경보 (실데이터로 평가 가능)\n",
           f"양성 비율: 학습 {vy[vtr].mean() * 100:.1f}%, 시험 {vy[vte].mean() * 100:.1f}%. 임계값 {th_v:.2f}\n",
           pd.DataFrame(rows).to_markdown(index=False), ""]

    # ---- 사용 여부 (정답 대신 추정 시간표)
    F = S.occupancy_features(df)
    y = df.sched_active.values
    okf = F[["co2_d30", "co2_above_min2h"]].notna().all(axis=1).values & df.co2.notna().values
    tr, va2, te2 = pm("train") & okf, pm("val") & okf, pm("test") & okf
    on = Norm(F.values[tr]); OX = on(F.values)
    occ = train_clf(OX[tr], y[tr])
    po = proba(occ, OX[va2])
    th_o = max(np.linspace(0.2, 0.8, 25), key=lambda t: f1_score(y[va2], po >= t))
    save_model(occ, on, list(F.columns), float(th_o), f"occ{TAG}")
    export_onnx(occ.eval(), (torch.tensor(OX[:2]),), os.path.join(ART, f"occ{TAG}.onnx"), ["x"])
    op = (proba(occ, OX) >= th_o).astype(int)
    m = metrics(y[te2], op[te2], "MLP vs 추정 시간표")
    m.pop("빈방을_사용중으로")
    md += ["## 사용 여부 (참고)\n",
           "실제 인원 기록이 없어 추정 시간표를 정답 대신 썼다. 아래는 '평소 사용 시간과 얼마나 맞나'일 뿐 정확도가 아니다.\n",
           pd.DataFrame([m]).to_markdown(index=False), ""]

    # ---- 예측 모델 ONNX (과제 3 실데이터에서 검증 오차로 고른 모델)
    best = json.load(open(os.path.join(ART, f"forecast_best{TAG}.json")))["model"]
    fmeta = json.load(open(os.path.join(ART, f"forecast_{best}{TAG}.json")))
    fs = FeatureSpec.from_dict(fmeta["spec"])
    fm = make_forecaster(best, len(fs.past), len(fs.future), D.HORIZON)
    fm.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{TAG}.pt"))); fm.eval()
    idx = D.anchors(df, bounds, "test")[:2]
    p, f = fs.windows(df, idx, with_target=False)
    export_onnx(fm, (torch.tensor(p), torch.tensor(f)), os.path.join(ART, f"forecast{TAG}.onnx"), ["past", "future"])
    md.append(f"서비스용 예측 모델: {best.upper()} (artifacts/forecast{TAG}.onnx)")

    os.makedirs(os.path.join(a.out, "real"), exist_ok=True)
    C.save(os.path.join(a.out, "real"), "service_models", dict(conditions=cond, ae_threshold=th, vent=rows, occ=m,
                                                              ae_events=len(ev)), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
