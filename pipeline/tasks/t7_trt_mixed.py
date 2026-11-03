"""과제 7 다시: TensorRT 혼합 정밀도로 CO2 예측 차이 1ppm 이하 만들기.

FP16 전체 엔진은 시험 입력에서 CO2 최대 차이 1.35ppm(기준 1ppm 초과). trtexec 는 GRU 부분을 한 덩어리(Myelin)로
합쳐 층별 정밀도 지정(--layerPrecisions)이 먹지 않으므로, 모델 안에서 부분별 자료형을 정한 ONNX를 만들고
--stronglyTyped 로 그 자료형 그대로 엔진을 만든다.
  fp16 전체 / GRU fp16 + 출력층 fp32 / GRU fp32 + 출력층 fp16 / fp32 전체 (fp32 쪽은 TF32 켬·끔 둘 다)
고르는 법: 검증 기간 입력 256개에서 최대 차이 1ppm 이하인 것 중 배치 1 지연이 가장 짧은 것 → 시험 입력 256개로 한 번만 확인.
산출: results/t7<tag>/trt_mixed.md
"""
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch import nn

from tasks.t7_optimize import load_models

from tsi import common as C
from tsi import data as D

ART = os.path.join(C.ROOT, "artifacts")
N = 256
TRTEXEC = shutil.which("trtexec") or "/usr/src/tensorrt/bin/trtexec"


class Mixed(nn.Module):
    """RNNForecaster 의 GRU 와 출력층(head)을 각각 정한 자료형으로 돌린다. 입출력은 fp32."""

    def __init__(self, fm, rnn_dtype, head_dtype):
        super().__init__()
        self.rnn = fm.rnn.to(rnn_dtype)
        self.head = fm.head.to(head_dtype)
        self.rd, self.hd, self.h, self.o = rnn_dtype, head_dtype, fm.h, fm.o

    def forward(self, past, fut):
        out, _ = self.rnn(past.to(self.rd))
        z = torch.cat([out[:, -1].to(self.hd), fut.flatten(1).to(self.hd)], dim=1)
        return self.head(z).float().view(-1, self.h, self.o)


def export(m, P, F, path):
    dev = "cuda"
    m = m.to(dev).eval()
    args = (torch.tensor(P[:2]).to(dev), torch.tensor(F[:2]).to(dev))
    torch.onnx.export(m, args, path, input_names=["past", "future"], output_names=["out"], opset_version=17,
                      dynamic_axes={"past": {0: "batch"}, "future": {0: "batch"}, "out": {0: "batch"}}, dynamo=False)


def latency(onnx_path, flags, P, F):
    """trtexec 로 배치 1 지연(GPU 계산 평균·p95, ms)."""
    shapes = f"past:1x{P.shape[1]}x{P.shape[2]},future:1x{F.shape[1]}x{F.shape[2]}"
    r = subprocess.run([TRTEXEC, f"--onnx={onnx_path}", *flags, f"--shapes={shapes}", "--warmUp=200",
                        "--iterations=2000", "--duration=0"], capture_output=True, text=True)
    m = re.search(r"GPU Compute Time: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms.*?percentile\(95%\) = ([\d.]+) ms",
                  r.stdout, re.S)
    return (float(m.group(1)), float(m.group(2))) if m else (None, None)


def main():
    a = C.args("과제 7 TensorRT 혼합 정밀도")
    raw, df, bounds, fmask = C.prepare(a)
    best, spec, fm, *_ = load_models(df[D.period_mask(df.ts, bounds, "train")], a.tag)
    pick = {}
    for period in ("val", "test"):
        an = D.anchors(df, bounds, period, exclude=fmask)
        pick[period] = an[np.linspace(0, len(an) - 1, N).astype(int)]
    data = {}
    for period, idx in pick.items():
        P, F = spec.windows(df, idx, with_target=False)
        with torch.no_grad():
            ref = fm(torch.tensor(P), torch.tensor(F)).numpy()
        data[period] = (P, F, ref)
    base_onnx = os.path.join(ART, f"forecast{a.tag}.onnx")
    rows, cand = [], []
    with tempfile.TemporaryDirectory() as w:
        P, F, _ = data["val"]
        variants = [("FP16 전체", base_onnx, ["--fp16"]), ("FP32 전체", base_onnx, []),
                    ("FP32 전체, TF32 끔", base_onnx, ["--noTF32"])]
        for name, rd, hd in [("GRU FP16 + 출력층 FP32", torch.float16, torch.float32),
                             ("GRU FP32 + 출력층 FP16", torch.float32, torch.float16)]:
            path = os.path.join(ART, f"forecast{a.tag}_{'g16h32' if rd == torch.float16 else 'g32h16'}.onnx")
            import copy
            export(Mixed(copy.deepcopy(fm), rd, hd), P, F, path)
            variants.append((name, path, ["--stronglyTyped"]))
            if rd == torch.float32:   # Orin(Ampere)은 FP32 곱셈을 TF32로 할 수 있어 끈 것도 본다
                variants.append((name + ", TF32 끔", path, ["--stronglyTyped", "--noTF32"]))
        for name, path, flags in variants:
            P, F, ref = data["val"]
            got = trt_run_flags(path, {"past": P, "future": F}, w, flags)
            val = np.abs((got - ref) * spec.dsd).reshape(-1, 2)[:, 0].max()
            lat, p95 = latency(path, flags, P, F)
            cand.append(dict(name=name, path=path, flags=flags, val=val, lat=lat, p95=p95))
            print(name, val, lat, p95, flush=True)
        ok = [c for c in cand if c["val"] <= 1 and c["lat"] is not None]
        chosen = min(ok, key=lambda c: c["lat"]) if ok else None
        for c in cand:
            if c is chosen or c["name"] == "FP16 전체":   # 시험 입력은 고른 것(과 원래 FP16 비교용)만 확인
                P, F, ref = data["test"]
                got = trt_run_flags(c["path"], {"past": P, "future": F}, w, c["flags"])
                d = np.abs((got - ref) * spec.dsd).reshape(-1, 2)
                c["test"], c["test_mean"], c["test_temp"] = d[:, 0].max(), d[:, 0].mean(), d[:, 1].max()
        for c in cand:
            rows.append(dict(방식=c["name"], 빌드=" ".join(c["flags"]) or "(기본 FP32)",
                             검증_CO2_최대차이_ppm=round(float(c["val"]), 3),
                             시험_CO2_최대차이_ppm=round(float(c["test"]), 3) if "test" in c else "-",
                             시험_평균_ppm=round(float(c["test_mean"]), 4) if "test" in c else "-",
                             시험_온도_최대차이=round(float(c["test_temp"]), 4) if "test" in c else "-",
                             배치1_지연_ms=c["lat"], p95_ms=c["p95"], 선택="✅" if c is chosen else ""))
    tbl = pd.DataFrame(rows)
    verdict = (f"선택: **{chosen['name']}** — 시험 입력 CO2 최대 차이 {chosen['test']:.3f}ppm "
               f"({'기준 1ppm 이하 ✅' if chosen['test'] <= 1 else '기준 초과 ❌'}), 배치 1 지연 {chosen['lat']:.3f}ms"
               if chosen else "검증 기준을 만족하는 방식이 없음")
    md = ["# 과제 7 다시. TensorRT 혼합 정밀도\n", C.cond_md(C.conditions(raw, bounds)),
          f"입력: 검증 기간 {N}개로 방식을 고르고, 시험 기간 {N}개(원래 trt_accuracy.md와 같은 시점)로 한 번만 확인. "
          "지연은 trtexec GPU 계산 시간(워밍업 200회, 2000회 반복, 배치 1). 차이는 PyTorch FP32 대비 원래 단위.\n",
          "FP16 전체 엔진은 GRU 부분을 한 덩어리로 합쳐 층별 정밀도 지정이 먹지 않는다. 그래서 모델 안에서 GRU와 "
          "출력층의 자료형을 나눠 ONNX로 내보내고 `--stronglyTyped`로 그 자료형 그대로 엔진을 만들었다.\n",
          tbl.to_markdown(index=False), "", verdict]
    C.save(os.path.join(a.out, "t7" + a.tag), "trt_mixed", dict(rows=rows, n=N), "\n".join(md) + "\n")
    print("\n".join(md))


def trt_run_flags(onnx_path, feeds, workdir, flags):
    """trt_run 과 같지만 빌드 옵션을 그대로 넘긴다(--stronglyTyped 등)."""
    import json
    loads, shapes = [], []
    for name, x in feeds.items():
        p = os.path.join(workdir, f"{name}.bin")
        np.ascontiguousarray(x, np.float32).tofile(p)
        loads.append(f"{name}:{p}")
        shapes.append(f"{name}:" + "x".join(str(d) for d in x.shape))
    out_json = os.path.join(workdir, "out.json")
    r = subprocess.run([TRTEXEC, f"--onnx={onnx_path}", *flags, f"--shapes={','.join(shapes)}",
                        f"--loadInputs={','.join(loads)}", f"--exportOutput={out_json}", "--iterations=1",
                        "--warmUp=0", "--duration=0"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stdout[-800:] + r.stderr[-800:])
    o = next(x for x in json.load(open(out_json)) if x["name"] == "out")
    return np.array(o["values"], np.float32).reshape([int(d) for d in o["dimensions"].split("x")])


if __name__ == "__main__":
    main()
