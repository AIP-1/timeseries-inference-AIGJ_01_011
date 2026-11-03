"""과제 7 심화(B).
  1) INT8: ONNX Runtime 동적 양자화(가중치 INT8)로 4개 모델을 바꿔, 변환 오차(원래 단위)와 CPU 지연을 FP32 ONNX와 비교
     (TensorRT INT8은 보정 데이터 없이 만들면 값이 의미 없고, 이 크기 모델은 GPU보다 CPU가 빨라 CPU 쪽으로 시험)
  2) 메모리: 한 프로세스에 4개 모델(ONNX Runtime CPU)을 함께 올렸을 때 늘어난 메모리(RSS)와 지금 서비스 프로세스 RSS
입력: 시험 기간 실제 창 256개(trt_accuracy.md와 같은 시점).
산출: results/t7<tag>/int8_mem.md, artifacts/*_int8<tag>.onnx
"""
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from tasks.t7_optimize import load_models
from tsi import anomaly as A
from tsi import common as C
from tsi import data as D
from tsi import state as S

ART = os.path.join(C.ROOT, "artifacts")
N = 256


def rss_mb():
    return int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) / 1024


def lat(sess, feeds, reps=500):
    for _ in range(50):
        sess.run(None, feeds)
    t = []
    for _ in range(reps):
        s = time.perf_counter(); sess.run(None, feeds); t.append((time.perf_counter() - s) * 1000)
    return float(np.mean(t)), float(np.percentile(t, 95))


def main():
    a = C.args("과제 7 심화")
    raw, df, bounds, fmask = C.prepare(a)
    cond = C.conditions(raw, bounds)
    best, spec, fm, ae, ae_meta, cls = load_models(df[D.period_mask(df.ts, bounds, "train")], a.tag)
    te = D.anchors(df, bounds, "test", exclude=fmask)
    idx = te[np.linspace(0, len(te) - 1, N).astype(int)]
    P, Fu = spec.windows(df, idx, with_target=False)
    feats = {"occ": S.occupancy_features(df), "vent": S.vent_features(df)}
    inputs = {"forecast": {"past": P, "future": Fu}}
    for kind, (m, meta) in cls.items():
        inputs[kind] = {"x": np.nan_to_num((feats[kind][meta["cols"]].values[idx] - np.array(meta["mu"]))
                                           / np.array(meta["sd"])).astype(np.float32)}
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    rows = []
    for name in ("forecast", "ae", "occ", "vent"):
        f32 = os.path.join(ART, f"{name}{a.tag}.onnx")
        i8 = os.path.join(ART, f"{name}_int8{a.tag}.onnx")
        quantize_dynamic(f32, i8, weight_type=QuantType.QInt8)
        s32 = ort.InferenceSession(f32, so, providers=["CPUExecutionProvider"])
        s8 = ort.InferenceSession(i8, so, providers=["CPUExecutionProvider"])
        if name == "ae":
            x = s32.get_inputs()[0]
            feeds = {x.name: np.random.default_rng(0).standard_normal((N, *x.shape[1:])).astype(np.float32) * 0.5}
        else:
            feeds = inputs[name]
        o32, o8 = s32.run(None, feeds)[0], s8.run(None, feeds)[0]
        if name == "forecast":
            d = np.abs((o8 - o32) * spec.dsd).reshape(-1, 2)
            err = f"CO2 최대 {d[:, 0].max():.2f}ppm (평균 {d[:, 0].mean():.2f})"
            ok = d[:, 0].max() <= 1
        elif name == "ae":
            e32 = ((o32 - feeds[x.name]) ** 2).mean(1); e8 = ((o8 - feeds[x.name]) ** 2).mean(1)
            err = f"복원 오차 최대 상대 차이 {np.max(np.abs(e8 - e32) / (e32 + 1e-9)) * 100:.1f}%"
            ok = None
        else:
            meta = cls[name][1]
            sm = lambda o: np.exp(o[:, 1]) / np.exp(o).sum(1)
            flip = ((sm(o32) >= meta["threshold"]) != (sm(o8) >= meta["threshold"])).mean() * 100
            err = f"분류가 바뀐 비율 {flip:.2f}%"
            ok = flip <= 0.1
        f1 = {k: v[:1] for k, v in feeds.items()}
        l32, l8 = lat(s32, f1), lat(s8, f1)
        rows.append(dict(모델=name, 크기_FP32_KB=round(os.path.getsize(f32) / 1024), 크기_INT8_KB=round(os.path.getsize(i8) / 1024),
                         지연_FP32_ms=round(l32[0], 3), 지연_INT8_ms=round(l8[0], 3), 변환_오차=err,
                         기준=("✅" if ok else "❌") if ok is not None else "-"))
        print(rows[-1], flush=True)

    # 2) 4개 모델 함께 올렸을 때 메모리 (새 세션으로)
    import gc
    gc.collect()
    m0 = rss_mb()
    sess = {n: ort.InferenceSession(os.path.join(ART, f"{n}{a.tag}.onnx"), providers=["CPUExecutionProvider"])
            for n in ("forecast", "ae", "occ", "vent")}
    sess["forecast"].run(None, {k: v[:4] for k, v in inputs["forecast"].items()})
    m1 = rss_mb()
    svc = None
    try:
        import subprocess
        pid = subprocess.run(["pgrep", "-f", "service/infer.py"], capture_output=True, text=True).stdout.split()[0]
        svc = int(open(f"/proc/{pid}/status").read().split("VmRSS:")[1].split()[0]) / 1024
    except Exception:
        pass
    md = ["# 과제 7 심화. INT8과 메모리\n", C.cond_md(cond),
          f"INT8: ONNX Runtime 동적 양자화(가중치 INT8, 활성값은 실행 중 양자화), CPU 1스레드, 배치 1 지연(워밍업 50, 500회). "
          f"오차는 같은 시험 입력 {N}개에서 FP32 ONNX 대비(오토인코더는 같은 크기 임의 입력).\n",
          pd.DataFrame(rows).to_markdown(index=False), "",
          "## 4개 모델 함께 올린 메모리\n",
          f"- 예측·이상 탐지·사용 여부·환기 경보 ONNX 4개를 한 프로세스에 올리고 한 번씩 실행: RSS {m1 - m0:.0f}MB 늘어남",
          f"- 지금 돌고 있는 추론 서비스(ts-infer) 프로세스 전체 RSS: {svc:.0f}MB (파이썬·pandas·API 서버 포함)" if svc else "",
          "- 로컬 LLM과 함께: 보드 메모리 7.5GB 중 시계열 서비스는 위 정도라, 1~3B급 4비트 LLM(약 1~2.5GB)과 "
          "임베딩 모델(약 1GB)을 함께 올릴 여유가 있다. GPU 메모리는 쓰지 않는다(ONNX Runtime CPU)."]
    C.save(os.path.join(a.out, "t7" + a.tag), "int8_mem", dict(rows=rows, mem_models_mb=m1 - m0, service_rss_mb=svc),
           "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
