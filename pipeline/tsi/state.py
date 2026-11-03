"""상태 추정(과제 5) 입력 특징. 서비스(과제 8)도 같은 함수를 쓴다.

사용 여부  : 센서값 변화만 쓴다(일정 정보는 쓰지 않는다 — 정답을 일정에서 만들었기 때문)
환기 경보  : 센서값 + 일정(지금·30분 안 예정 인원)을 쓸 수 있다
"""
import numpy as np
import pandas as pd

from . import data as D

VENT_LIMIT = 1000.0
VENT_AHEAD = 6  # 30분 = 5분 × 6


def _diff(g, col, k):
    return g[col] - g[col].shift(k)


def occupancy_features(df):
    out = []
    for space, g in df.groupby("space", sort=False):
        f = pd.DataFrame(index=g.index)
        f["co2"] = g.co2
        f["co2_d5"] = _diff(g, "co2", 1)
        f["co2_d15"] = _diff(g, "co2", 3)
        f["co2_d30"] = _diff(g, "co2", 6)
        f["co2_above_min2h"] = g.co2 - g.co2.rolling(24, min_periods=6).min()
        f["t_d15"] = _diff(g, "temperature", 3)
        f["t_d30"] = _diff(g, "temperature", 6)
        f["h_d15"] = _diff(g, "humidity", 3)
        out.append(f)
    f = pd.concat(out).loc[df.index]
    for c in ["tod_sin", "tod_cos", "weekend"] + [c for c in df.columns if c.startswith("space_")]:
        f[c] = df[c]
    return f


def vent_features(df):
    f = occupancy_features(df)
    f["sched_people"] = df.sched_people
    ahead = []
    for space, g in df.groupby("space", sort=False):
        # 앞으로 30분 안의 최대 예정 인원(일정은 미리 알 수 있음)
        s = g.sched_people[::-1].rolling(VENT_AHEAD, min_periods=1).max()[::-1].shift(-1)
        ahead.append(s)
    f["sched_ahead"] = pd.concat(ahead).loc[df.index].fillna(0)
    return f


def vent_label(df):
    """양성: 지금은 기준 미만인데 앞으로 30분 안에 기준을 넘는다. 이미 넘은 시점은 판단 대상에서 뺀다(NaN)."""
    lab = []
    for space, g in df.groupby("space", sort=False):
        fut = g.co2[::-1].rolling(VENT_AHEAD, min_periods=VENT_AHEAD).max()[::-1].shift(-1)
        y = (fut >= VENT_LIMIT).astype(float)
        y[g.co2 >= VENT_LIMIT] = np.nan
        y[fut.isna() | g.co2.isna()] = np.nan
        lab.append(y)
    return pd.concat(lab).loc[df.index]


def occupancy_truth(df, occupancy):
    """5분 칸 동안 실제로 사람이 있었던 비율이 절반 이상이면 사용중 (직접 관찰 정답)."""
    occ = occupancy.assign(used=(occupancy.people > 0).astype(float))
    occ["ts"] = occ.ts.dt.floor(D.STEP)
    o = occ.groupby(["space", "ts"]).used.mean().reset_index()
    m = df[["space", "ts"]].merge(o, on=["space", "ts"], how="left")
    return (m.used.values >= 0.5).astype(float)
