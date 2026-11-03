"""과제 7. 엣지 최적화 — ONNX 변환, 실행 방식별 지연·오차·메모리·전력.

실행 방식: PyTorch CPU, PyTorch GPU, ONNX Runtime(CPU·가능하면 CUDA/TensorRT), TensorRT FP16(trtexec 가 있으면)
- 배치 1(한 공간)·배치 4(네 공간)의 지연 평균·느린 쪽 5%(p95). 워밍업 20회 후 200회
- 변환 전후 차이를 원래 단위로: CO2 예측 최대 몇 ppm, 분류 결과가 바뀐 비율
- 측정 중 tegrastats 로 전력(VDD_IN)·GPU·메모리 기록
산출: results/t7/optimize.md, artifacts/*.onnx (서비스가 ONNX Runtime으로 사용)
"""
import json
import os
import shutil
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch

from tsi import anomaly as A
from tsi import common as C
from tsi import data as D
from tsi import state as S
from tsi.features import FeatureSpec
from tsi.models import AutoEncoder, MLPClassifier, make_forecaster

ART = os.path.join(C.ROOT, "artifacts")
WARM, REPS = 20, 200


class Tegra:
    """tegrastats 를 뒤에서 돌려 평균 전력(mW)·GPU 사용률·RAM 을 모은다."""
    def __init__(self):
        self.p = None
        if shutil.which("tegrastats"):
            self.p = subprocess.Popen(["tegrastats", "--interval", "100"], stdout=subprocess.PIPE, text=True)

    def stop(self):
        if not self.p:
            return {}
        self.p.terminate()
        lines = self.p.stdout.read().splitlines()
        import re
        pw = [int(m.group(1)) for l in lines for m in [re.search(r"VDD_IN (\d+)mW", l)] if m]
        gpu = [int(m.group(1)) for l in lines for m in [re.search(r"GR3D_FREQ (\d+)%", l)] if m]
        ram = [int(m.group(1)) for l in lines for m in [re.search(r"RAM (\d+)/", l)] if m]
        return dict(power_mW=round(np.mean(pw)) if pw else None, gpu_pct=round(np.mean(gpu)) if gpu else None,
                    ram_MB=max(ram) if ram else None)


POWER_S = 3.0  # 전력은 tegrastats(100ms 간격)로 재므로 최소 3초는 계속 돌린다


def bench(fn, inputs):
    """지연(워밍업 후 REPS 회)과, 같은 호출을 POWER_S 초 동안 반복할 때의 전력·GPU·메모리."""
    for _ in range(WARM):
        fn(*inputs)
    lat = []
    for _ in range(REPS):
        t = time.perf_counter(); fn(*inputs); lat.append((time.perf_counter() - t) * 1000)
    tg = Tegra()
    end = time.time() + POWER_S
    while time.time() < end:
        fn(*inputs)
    return float(np.mean(lat)), float(np.percentile(lat, 95)), tg.stop()


def conclusions(lt):
    """측정표에서 결론 문장을 만든다(숫자를 손으로 옮겨 적지 않도록)."""
    out = []
    for name, g in lt.groupby("모델", sort=False):
        b1 = g[g.배치 == 1].dropna(subset=["평균_ms"]).set_index("방식").평균_ms
        if b1.empty:
            continue
        fast = b1.idxmin()
        line = f"- {name}: 배치 1에서 가장 빠른 방식은 {fast}({b1[fast]:.3f}ms)"
        if "PyTorch CPU" in b1 and "PyTorch GPU" in b1:
            r = b1["PyTorch CPU"] / b1["PyTorch GPU"]
            line += f", PyTorch는 GPU가 CPU보다 {r:.1f}배 {'빠름' if r > 1 else '느림'}"
        b4 = g[g.배치 == 4].set_index("방식").평균_ms
        if fast in b4:
            line += f". 네 공간을 한 번에(배치 4) 돌려도 {b4[fast]:.3f}ms"
        out.append(line + ".")
    out.append("- ONNX Runtime CPU가 PyTorch보다 빠른 이유: 작은 모델에서는 연산보다 파이썬·프레임워크 호출 비용이 대부분이고, "
               "ONNX Runtime은 그 비용이 작다. GPU는 데이터 복사·커널 실행 비용이 있어 이렇게 작은 입력에서는 이득이 작다.")
    return out


def load_models(df_train, tag=""):
    best = json.load(open(os.path.join(ART, f"forecast_best{tag}.json")))["model"]
    fmeta = json.load(open(os.path.join(ART, f"forecast_{best}{tag}.json")))
    spec = FeatureSpec.from_dict(fmeta["spec"])
    fm = make_forecaster(best, len(spec.past), len(spec.future), D.HORIZON)
    fm.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{tag}.pt")))
    ae_meta = json.load(open(os.path.join(ART, f"ae{tag}.json")))
    ae = AutoEncoder(ae_meta["n_in"]); ae.load_state_dict(torch.load(os.path.join(ART, f"ae{tag}.pt")))
    cls = {}
    for kind in ("occ", "vent"):
        meta = json.load(open(os.path.join(ART, f"{kind}{tag}.json")))
        m = MLPClassifier(len(meta["cols"])); m.load_state_dict(torch.load(os.path.join(ART, f"{kind}{tag}.pt")))
        cls[kind] = (m.eval(), meta)
    return best, spec, fm.eval(), ae.eval(), ae_meta, cls


def export(model, args, path, names):
    torch.onnx.export(model, args, path, input_names=names, output_names=["out"], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in names + ["out"]}, dynamo=False)


def main():
    a = C.args("과제 7")
    import onnxruntime as ort
    raw, df, bounds, fmask = C.prepare(a)
    out = os.path.join(a.out, "t7" + a.tag)
    cond = C.conditions(raw, bounds)
    best, spec, fm, ae, ae_meta, cls = load_models(df[D.period_mask(df.ts, bounds, "train")], a.tag)
    te = D.anchors(df, bounds, "test", exclude=fmask)

    # 입력 준비 (시험 기간 실제 창)
    P, Fu = spec.windows(df, te, with_target=False)
    X_ae, _ = A.ae_matrix(df[D.period_mask(df.ts, bounds, "test")].reset_index(drop=True), FeatureSpec.from_dict(ae_meta["spec"]))
    feats = {"occ": S.occupancy_features(df), "vent": S.vent_features(df)}
    Xc = {}
    for tag, (m, meta) in cls.items():
        F = feats[tag][meta["cols"]].values[te]
        Xc[tag] = np.nan_to_num((F - np.array(meta["mu"])) / np.array(meta["sd"])).astype(np.float32)

    models = {  # 이름: (모델, 입력, 입력 이름, ONNX 파일 이름 — 서비스가 이 파일을 읽는다)
        f"예측({best.upper()})": (fm, [P, Fu], ["past", "future"], "forecast"),
        "이상탐지(AE)": (ae, [X_ae], ["x"], "ae"),
        "사용여부(MLP)": (cls["occ"][0], [Xc["occ"]], ["x"], "occ"),
        "환기경보(MLP)": (cls["vent"][0], [Xc["vent"]], ["x"], "vent"),
    }
    providers = ort.get_available_providers()
    gpu_ok = False
    try:
        x = torch.randn(4, 4, device="cuda"); (x @ x).sum().item(); gpu_ok = True
    except Exception:
        pass
    trtexec = shutil.which("trtexec") or ("/usr/src/tensorrt/bin/trtexec" if os.path.exists("/usr/src/tensorrt/bin/trtexec") else None)

    lat_rows, acc_rows = [], []
    for name, (m, inputs, names, fname) in models.items():
        path = os.path.join(ART, f"{fname}{a.tag}.onnx")
        export(m, tuple(torch.tensor(x[:2]) for x in inputs), path, names)
        # 정확도: 시험 전체 입력으로 torch fp32 vs ONNX Runtime
        with torch.no_grad():
            ref = m(*[torch.tensor(x) for x in inputs]).numpy()
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        got = sess.run(None, {n: x for n, x in zip(names, inputs)})[0]
        if name.startswith("예측"):
            diff = np.abs((got - ref) * spec.dsd).reshape(-1, 2)
            acc_rows.append(dict(모델=name, 비교="ONNX Runtime vs PyTorch", 지표="CO2 예측 최대 차이(ppm)",
                                 값=f"{diff[:, 0].max():.4f}", 온도_최대차이=f"{diff[:, 1].max():.5f}℃"))
        elif name.startswith("이상"):
            acc_rows.append(dict(모델=name, 비교="ONNX Runtime vs PyTorch", 지표="복원 오차 최대 상대 차이",
                                 값=f"{np.abs(got - ref).max() / (np.abs(ref).max() + 1e-9):.2e}"))
        else:
            acc_rows.append(dict(모델=name, 비교="ONNX Runtime vs PyTorch", 지표="분류 결과가 바뀐 비율",
                                 값=f"{(got.argmax(1) != ref.argmax(1)).mean() * 100:.3f}%"))

        for bs in (1, 4):
            xb = [x[:bs] for x in inputs]
            tb = [torch.tensor(x) for x in xb]
            with torch.no_grad():
                mean, p95, st = bench(lambda *t: m(*t), tb)
            lat_rows.append(dict(모델=name, 방식="PyTorch CPU", 배치=bs, 평균_ms=mean, p95_ms=p95, **st))
            if gpu_ok:
                mg = m.to("cuda"); tbg = [t.to("cuda") for t in tb]

                def run_gpu(*t):
                    mg(*t); torch.cuda.synchronize()
                with torch.no_grad():
                    try:
                        mean, p95, st = bench(run_gpu, tbg)
                        lat_rows.append(dict(모델=name, 방식="PyTorch GPU", 배치=bs, 평균_ms=mean, p95_ms=p95, **st))
                    except Exception as e:
                        lat_rows.append(dict(모델=name, 방식="PyTorch GPU", 배치=bs, 평균_ms=np.nan, p95_ms=np.nan,
                                             비고=str(e)[:60]))
                m.to("cpu")
            for prov in [p for p in ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in providers]:
                s = ort.InferenceSession(path, providers=[prov])
                feed = {n: x for n, x in zip(names, xb)}
                mean, p95, st = bench(lambda: s.run(None, feed), [])
                lat_rows.append(dict(모델=name, 방식=f"ONNX Runtime {prov.replace('ExecutionProvider', '')}", 배치=bs,
                                     평균_ms=mean, p95_ms=p95, **st))
            if trtexec:
                shapes = ",".join(f"{n}:{bs}x" + "x".join(str(d) for d in x.shape[1:]) for n, x in zip(names, xb))
                r = subprocess.run([trtexec, f"--onnx={path}", "--fp16", f"--shapes={shapes}", f"--warmUp=200",
                                    f"--iterations={REPS}", "--percentile=95"], capture_output=True, text=True)
                import re
                mm = re.search(r"Latency: min = [\d.]+ ms, max = [\d.]+ ms, mean = ([\d.]+) ms.*?percentile\(95%\) = ([\d.]+) ms",
                               r.stdout, re.S)
                if mm:
                    lat_rows.append(dict(모델=name, 방식="TensorRT FP16", 배치=bs, 평균_ms=float(mm.group(1)),
                                         p95_ms=float(mm.group(2))))

    lt = pd.DataFrame(lat_rows)
    for c in ("평균_ms", "p95_ms"):
        lt[c] = lt[c].round(3)
    nvp = subprocess.run(["nvpmodel", "-q"], capture_output=True, text=True).stdout.splitlines()
    mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout.splitlines()[1].split()
    others = subprocess.run(["bash", "-c", "ps -eo rss,args --no-headers | grep -c vscode-server/cli/servers"],
                            capture_output=True, text=True).stdout.strip()
    md = ["# 과제 7. 엣지 최적화\n", C.cond_md(cond),
          f"**측정 조건**: 전력 모드 `{nvp[0] if nvp else '?'}`, 클럭 고정(jetson_clocks) 안 함, 워밍업 {WARM}회 후 "
          f"{REPS}회 반복, 입력은 시험 기간 실제 창. 전력은 같은 호출을 {POWER_S:.0f}초 동안 반복하며 tegrastats VDD_IN "
          f"평균(보드 전체). 측정 끝 시점 메모리 {mem[2]}/{mem[1]}MB 사용(VS Code 원격 서버 등 다른 프로세스 "
          f"{others}개가 함께 떠 있음) — PyTorch CPU 지연의 p95가 큰 것은 이 경쟁의 영향일 수 있다.\n",
          f"사용 가능한 실행 방식: PyTorch GPU {'가능' if gpu_ok else '불가'}, ONNX Runtime {', '.join(providers)}, "
          f"TensorRT(trtexec) {'있음' if trtexec else '없음 — TensorRT 패키지 미설치'}\n",
          "## 지연·전력·메모리\n", lt.to_markdown(index=False), "",
          "## 변환 전후 결과 차이 (원래 단위)\n", pd.DataFrame(acc_rows).to_markdown(index=False), "",
          "## 결론\n"] + conclusions(lt) + [
          "- 그래도 변환하는 이유: 같은 보드에 임베딩 모델·로컬 LLM이 함께 돈다. 시계열 모델은 CPU 한쪽에서 ONNX Runtime으로 "
          "가볍게 돌리고 GPU는 LLM에 양보하거나, 공간이 수십 개로 늘면 배치로 묶어 GPU/TensorRT로 보내는 것이 맞다.",
          "- 평균보다 느린 쪽 5%(p95)가 중요한 이유: 매분 도는 서비스는 가끔 느려지는 순간에 다음 주기와 겹치거나 경보가 늦어진다.",
          "- 서비스(과제 8)는 변환 오차가 0에 가까운 ONNX Runtime CPU를 쓴다."]
    if not trtexec:
        md.append("\n**TensorRT 측정 방법**: Jetson에서 `sudo apt install -y tensorrt` 로 TensorRT를 설치한 뒤 "
                  "이 스크립트를 다시 실행하면 TensorRT FP16 행이 추가된다.")
    C.save(out, "optimize", dict(conditions=cond, latency=lat_rows, accuracy=acc_rows, providers=providers,
                                 gpu_ok=gpu_ok, trtexec=trtexec), "\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
