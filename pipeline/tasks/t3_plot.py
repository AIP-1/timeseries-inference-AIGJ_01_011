"""과제 3 그림: 시험 기간 하루의 실제 CO2와 '60분 전에 내놓은 예측'을 공간별로 겹쳐 그린다.

  ../.venv/bin/python tasks/t3_plot.py --influx ../data/influx --tag _real --day 2026-10-14
산출: results/t3<tag>/forecast_day_<day>.png
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tasks.t3_forecast import predict
from tsi import common as C
from tsi import data as D
from tsi.features import FeatureSpec
from tsi.models import make_forecaster

plt.rcParams["font.family"] = ["NanumSquare", "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
ART = os.path.join(C.ROOT, "artifacts")


def main():
    a = C.args("과제 3 그림", day="2026-10-14")
    raw, df, bounds, fmask = C.prepare(a)
    best = json.load(open(os.path.join(ART, f"forecast_best{a.tag}.json")))["model"]
    meta = json.load(open(os.path.join(ART, f"forecast_{best}{a.tag}.json")))
    spec = FeatureSpec.from_dict(meta["spec"])
    m = make_forecaster(best, len(spec.past), len(spec.future), D.HORIZON)
    m.load_state_dict(torch.load(os.path.join(ART, f"forecast_{best}{a.tag}.pt"))); m.eval()

    day = pd.Timestamp(a.day)
    te = D.anchors(df, bounds, "test", exclude=fmask)
    spaces = sorted(df.space.unique())
    fig, axes = plt.subplots(len(spaces), 1, figsize=(12, 3.2 * len(spaces)), sharex=True)
    for ax, sp in zip(np.atleast_1d(axes), spaces):
        # 그날 06~22시에 '60분 뒤'가 떨어지는 예측 기준 시점들
        idx = te[(df.space.values[te] == sp) & (df.ts.values[te] >= np.datetime64(day + pd.Timedelta(hours=5)))
                 & (df.ts.values[te] < np.datetime64(day + pd.Timedelta(hours=21)))]
        g = df[(df.space == sp) & (df.ts >= day + pd.Timedelta(hours=6)) & (df.ts < day + pd.Timedelta(hours=22))]
        ax.plot(g.ts, g.co2, color="#14213D", lw=2, label="실제 CO₂")
        if len(idx):
            p = predict(m, spec, df, idx)[:, -1, 0]
            t60 = df.ts.values[idx] + np.timedelta64(60, "m")
            ax.plot(t60, p, color="#2F5FC4", lw=1.6, ls="--", label=f"60분 전 예측 ({best.upper()})")
            ax.plot(t60, df.co2.values[idx], color="#B4521A", lw=1, alpha=0.7, label="현재값 유지 (60분 전 값)")
        ax.axhline(1000, color="gray", ls=":", lw=1)
        sch = raw["schedule"]
        for r in sch[(sch.space == sp) & (sch.start.dt.normalize() == day)].itertuples():
            ax.axvspan(r.start, r.end, color="#2F5FC4", alpha=0.06)
        ax.set_ylabel(f"{sp}\nCO₂ (ppm)")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="upper left", fontsize=9, ncol=3)
    np.atleast_1d(axes)[0].set_title(f"{day:%Y-%m-%d (%a)} — 실제 CO₂와 60분 전에 내놓은 예측 (음영: 추정 사용 시간, 점선: 1,000ppm)")
    np.atleast_1d(axes)[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%H:%M"))
    fig.tight_layout()
    out = os.path.join(a.out, "t3" + a.tag)
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, f"forecast_day_{day:%Y%m%d}.png")
    fig.savefig(path, dpi=110)
    print("saved", path)


if __name__ == "__main__":
    main()
