"""과제 7 보완: TensorRT FP16 엔진의 출력을 PyTorch(FP32)와 원래 단위로 비교한다.

trtexec 로 시험 기간 실제 입력(배치 N)을 넣어 출력을 JSON으로 받고, 같은 입력의 PyTorch 출력과 비교:
  예측 → CO2·온도 최대 차이(ppm, ℃), 분류 → 결과가 바뀐 비율
산출: results/t7<tag>/trt_accuracy.md
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tasks.t7_optimize import load_models
from tsi import common as C
from tsi import data as D
from tsi import state as S

ART = os.path.join(C.ROOT, "artifacts")
N = 256


def trt_run(onnx_path, feeds, workdir, fp16=True):
    """feeds: {name: float32 array}. trtexec(FP16 또는 FP32)로 실행해 첫 출력(out)을 돌려준다."""
    trtexec = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"
    loads, shapes = [], []
    for name, x in feeds.items():
        p = os.path.join(workdir, f"{name}.bin")
        np.ascontiguousarray(x, np.float32).tofile(p)
        loads.append(f"{name}:{p}")
        shapes.append(f"{name}:" + "x".join(str(d) for d in x.shape))
    out_json = os.path.join(workdir, "out.json")
    r = subprocess.run([trtexec, f"--onnx={onnx_path}"] + (["--fp16"] if fp16 else []) + [f"--shapes={','.join(shapes)}",
                        f"--loadInputs={','.join(loads)}", f"--exportOutput={out_json}", "--iterations=1",
                        "--warmUp=0", "--duration=0"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-800:] + r.stderr[-800:])
    outs = json.load(open(out_json))
    o = next(x for x in outs if x["name"] == "out")
    return np.array(o["values"], np.float32).reshape([int(d) for d in o["dimensions"].split("x")])


def main():
    a = C.args("과제 7 TensorRT 정확도")
    raw, df, bounds, fmask = C.prepare(a)
    best, spec, fm, ae, ae_meta, cls = load_models(df[D.period_mask(df.ts, bounds, "train")], a.tag)
    te = D.anchors(df, bounds, "test", exclude=fmask)
    idx = te[np.linspace(0, len(te) - 1, N).astype(int)]
    rows = []
    with tempfile.TemporaryDirectory() as w:
        P, F = spec.windows(df, idx, with_target=False)
        with torch.no_grad():
            ref = fm(torch.tensor(P), torch.tensor(F)).numpy()
        got = trt_run(os.path.join(ART, f"forecast{a.tag}.onnx"), {"past": P, "future": F}, w)
        d = np.abs((got - ref) * spec.dsd).reshape(-1, 2)
        rows.append(dict(모델=f"예측({best.upper()}) TensorRT FP16", 지표="CO2 최대 차이(ppm)", 값=f"{d[:, 0].max():.3f}",
                         평균=f"{d[:, 0].mean():.4f}", 기준="≤ 1ppm", 판정="✅" if d[:, 0].max() <= 1 else "❌"))
        rows.append(dict(모델=f"예측({best.upper()})", 지표="온도 최대 차이(℃)", 값=f"{d[:, 1].max():.4f}",
                         평균=f"{d[:, 1].mean():.5f}", 기준="—", 판정=""))
        got32 = trt_run(os.path.join(ART, f"forecast{a.tag}.onnx"), {"past": P, "future": F}, w, fp16=False)
        d32 = np.abs((got32 - ref) * spec.dsd).reshape(-1, 2)
        rows.append(dict(모델=f"예측({best.upper()}) TensorRT FP32", 지표="CO2 최대 차이(ppm)", 값=f"{d32[:, 0].max():.4f}",
                         평균=f"{d32[:, 0].mean():.5f}", 기준="≤ 1ppm", 판정="✅" if d32[:, 0].max() <= 1 else "❌"))
        feats = {"occ": S.occupancy_features(df), "vent": S.vent_features(df)}
        for kind, (m, meta) in cls.items():
            X = np.nan_to_num((feats[kind][meta["cols"]].values[idx] - np.array(meta["mu"])) / np.array(meta["sd"])).astype(np.float32)
            with torch.no_grad():
                pr = torch.softmax(m(torch.tensor(X)), 1)[:, 1].numpy()
            lo = trt_run(os.path.join(ART, f"{kind}{a.tag}.onnx"), {"x": X}, w)
            pt = np.exp(lo[:, 1]) / np.exp(lo).sum(axis=1)
            flip = ((pr >= meta["threshold"]) != (pt >= meta["threshold"])).mean() * 100
            rows.append(dict(모델={"occ": "사용여부(MLP)", "vent": "환기경보(MLP)"}[kind], 지표="분류 결과가 바뀐 비율",
                             값=f"{flip:.2f}%", 평균=f"확률 최대 차이 {np.abs(pr - pt).max():.4f}", 기준="≤ 0.1%",
                             판정="✅" if flip <= 0.1 else "❌"))
    md = ["# 과제 7 보완. TensorRT FP16 변환 오차\n", C.cond_md(C.conditions(raw, bounds)),
          f"시험 기간 실제 입력 {N}개(고르게 뽑음)를 trtexec FP16 엔진과 PyTorch FP32에 똑같이 넣어 비교.\n",
          pd.DataFrame(rows).to_markdown(index=False)]
    out = os.path.join(a.out, "t7" + a.tag)
    C.save(out, "trt_accuracy", dict(rows=rows, n=N), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
