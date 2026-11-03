"""작업 스크립트들이 같이 쓰는 도우미: 데이터 준비, 결과 저장, 측정 조건 기록."""
import argparse
import json
import os
import platform
import time

import numpy as np
import pandas as pd

from . import data as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def args(desc, **extra):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "synth"), help="합성 데이터 폴더")
    ap.add_argument("--real-db", default=None, help="실데이터로 바꿔 끼울 때: 수집기 sensors.db 경로")
    ap.add_argument("--influx", default=None, help="실데이터: InfluxDB에서 내려받은 parquet 폴더(../data/influx)")
    ap.add_argument("--tag", default="", help="결과·모델 이름 뒤에 붙일 꼬리표 (예: _real)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    for k, v in extra.items():
        ap.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    return ap.parse_args()


def prepare(a):
    if a.influx:
        raw = D.load_influx(a.influx, schedule_csv=os.path.join(ROOT, "data", "real", "schedule.csv"))
    elif a.real_db:
        raw = D.load_real(a.real_db)
    else:
        raw = D.load_synth(a.data)
    df = D.build_frame(raw)
    bounds = D.split_bounds(df, raw["meta"])
    fmask = D.fault_mask(df, raw["faults"])
    return raw, df, bounds, fmask


def conditions(raw, bounds):
    """보고서 결과표마다 함께 적을 측정 조건."""
    nvp = "unknown"
    try:
        nvp = open("/etc/nvpmodel.conf").read().split("PM_CONFIG DEFAULT=")[1].split()[0]
    except Exception:
        pass
    return dict(data=("합성(자체 생성기)" if raw["source"] == "synthetic" else "실데이터(InfluxDB, 추정 시간표)"),
                period={k: f"{v[0]:%Y-%m-%d} ~ {(v[1] - pd.Timedelta(minutes=1)):%Y-%m-%d}" for k, v in bounds.items()},
                step=D.STEP, lookback_min=D.LOOKBACK * D.STEP_MIN, horizon_min=D.HORIZON * D.STEP_MIN,
                host=platform.node(), machine=platform.machine(), nvpmodel_default=nvp,
                generated=time.strftime("%Y-%m-%d %H:%M"))


def save(out_dir, name, obj, md=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    if md is not None:
        with open(os.path.join(out_dir, f"{name}.md"), "w", encoding="utf-8") as f:
            f.write(md)


def cond_md(c):
    p = c["period"]
    return (f"> 측정 조건: 데이터 {c['data']} · 학습 {p['train']} · 검증 {p['val']} · 시험 {p['test']} · "
            f"간격 {c['step']} · 과거 {c['lookback_min']}분 → 미래 {c['horizon_min']}분 · "
            f"{c['host']} ({c['machine']}) · {c['generated']}\n")


def mae_table(rows):
    """rows: [{model, target, h_min, mae}] → 마크다운 표 (행: 모델, 열: 예측 시점)"""
    t = pd.DataFrame(rows).pivot_table(index=["target", "model"], columns="h_min", values="mae")
    t.columns = [f"{c}분" for c in t.columns]
    return t.round(2)


def set_seed(seed=0):
    import random
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
