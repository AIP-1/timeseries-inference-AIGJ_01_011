"""과제 2~8 공통 데이터 처리. 학습·평가·서비스가 모두 이 모듈 하나만 쓴다(전처리 불일치 방지).

흐름: load_* → clean → resample(5분) → fill_short_gaps → add_time/schedule features → split → Scaler(train만)
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd

TARGETS = ["co2", "temperature"]
SENSORS = ["co2", "temperature", "humidity"]
STEP = "5min"           # 분석 간격
STEP_MIN = 5
LOOKBACK = 24           # 과거 2시간
HORIZON = 12            # 미래 60분 (5분 × 12)
EVAL_STEPS = {5: 1, 15: 3, 30: 6, 60: 12}  # 평가할 예측 시점(분) → 몇 번째 칸

# 정제 규칙: 물리적으로 불가능한 값은 결측으로 바꾼다
VALID = {"co2": (300, 5000), "temperature": (-10, 50), "humidity": (5, 100)}
MAX_FILL_STEPS = 3      # 15분 이하 결측만 선형 보간, 더 길면 결측으로 둔다


# ---------------------------------------------------------------- 불러오기
def load_synth(path):
    s = pd.read_csv(os.path.join(path, "sensors.csv"), parse_dates=["ts"])
    sched = pd.read_csv(os.path.join(path, "schedule.csv"), parse_dates=["start", "end"])
    occ = pd.read_csv(os.path.join(path, "occupancy.csv"), parse_dates=["ts"])
    faults = pd.read_csv(os.path.join(path, "faults.csv"), parse_dates=["start", "end"])
    with open(os.path.join(path, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    return dict(sensors=s, schedule=sched, occupancy=occ, faults=faults, meta=meta, source="synthetic")


def load_real(db_path, location_map=None, schedule_csv=None):
    """실데이터(수집기 SQLite)를 합성 데이터와 같은 모양으로 바꾼다. 실데이터로 바꿔 끼울 때 쓴다.
    location_map: {"실습실": "강의실A", ...} 처럼 수집기의 공간 이름을 분석용 이름으로."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    q = f"""SELECT ts_recv, location, metric, value FROM readings
            WHERE source='lorawan' AND flag=0 AND metric IN ({','.join('?' * len(SENSORS))})"""
    r = pd.read_sql_query(q, db, params=SENSORS)
    r["ts"] = pd.to_datetime(r.ts_recv, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
    r["space"] = r.location.map(location_map or {}).fillna(r.location)
    # 같은 공간에 센서가 여러 개면 1분 단위 평균
    r["ts"] = r.ts.dt.floor("min")
    s = r.pivot_table(index=["ts", "space"], columns="metric", values="value", aggfunc="mean").reset_index()
    sched = (pd.read_csv(schedule_csv, parse_dates=["start", "end"]) if schedule_csv
             else pd.DataFrame(columns=["space", "start", "end", "kind", "planned_people"]))
    return dict(sensors=s, schedule=sched, occupancy=None, faults=None, meta={}, source="real")


REAL_SPACES = ["사무실", "실습실", "회의실"]   # CO2 센서가 있는 공간
EXCLUDE_SPOTS = {"복도"}                    # 실습실 태그지만 복도에 있는 온습도 센서는 실내 평균에서 뺀다
REAL_SPLIT = dict(val_start="2026-09-01", test_start="2026-10-01")


def load_influx(path, spaces=REAL_SPACES, schedule_csv=None, split=REAL_SPLIT):
    """InfluxDB에서 내려받은 1분 평균(export_influx.py, data/influx/*.parquet) → 합성 데이터와 같은 모양.
    공간에 센서가 여러 개면 CO2는 중앙값(센서 하나가 고장 나도 덜 흔들림), 온습도는 평균."""
    files = sorted(f for f in os.listdir(path) if f.endswith(".parquet"))
    r = pd.concat([pd.read_parquet(os.path.join(path, f)) for f in files], ignore_index=True)
    r = r[r.location.isin(spaces) & ~r.spot.isin(EXCLUDE_SPOTS) & r.metric.isin(SENSORS)]
    r["ts"] = pd.to_datetime(r.ts, utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None).dt.floor("min")
    co2 = r[r.metric == "co2"].groupby(["ts", "location"]).value.median().rename("co2")
    th = r[r.metric != "co2"].pivot_table(index=["ts", "location"], columns="metric", values="value", aggfunc="mean")
    s = pd.concat([co2, th], axis=1).reset_index().rename(columns={"location": "space"})
    for c in SENSORS:
        if c not in s:
            s[c] = np.nan
    sched = (pd.read_csv(schedule_csv, parse_dates=["start", "end"]) if schedule_csv and os.path.exists(schedule_csv)
             else pd.DataFrame(columns=["space", "start", "end", "kind", "planned_people"]))
    return dict(sensors=s[["ts", "space"] + SENSORS], schedule=sched, occupancy=None, faults=None,
                meta=dict(split), source="real")


# ---------------------------------------------------------------- 정제
def clean(sensors):
    """범위 밖 값을 결측으로. 몇 개를 지웠는지 함께 돌려준다."""
    s = sensors.copy()
    removed = {}
    for c, (lo, hi) in VALID.items():
        if c in s:
            bad = ~s[c].between(lo, hi) & s[c].notna()
            removed[c] = int(bad.sum())
            s.loc[bad, c] = np.nan
    return s, removed


def resample(sensors, step=STEP):
    """공간별 5분 평균. 칸 안에 값이 하나도 없으면 결측."""
    out = []
    for space, g in sensors.groupby("space"):
        r = g.set_index("ts")[SENSORS].resample(step).mean()
        r["n_obs"] = g.set_index("ts")["co2"].resample(step).count()
        r["space"] = space
        out.append(r.reset_index())
    return pd.concat(out, ignore_index=True)


def fill_short_gaps(df, max_steps=MAX_FILL_STEPS):
    """짧은 결측만 선형 보간. observed=원래 관측이 있던 칸."""
    out = []
    for space, g in df.groupby("space"):
        g = g.sort_values("ts").copy()
        g["observed"] = g["co2"].notna()
        for c in SENSORS:
            g[c] = g[c].interpolate(limit=max_steps, limit_area="inside")
        out.append(g)
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 특징
def add_time_features(df):
    t = df.ts
    hour = t.dt.hour + t.dt.minute / 60
    df["tod_sin"] = np.sin(2 * np.pi * hour / 24)
    df["tod_cos"] = np.cos(2 * np.pi * hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * t.dt.weekday / 7)
    df["dow_cos"] = np.cos(2 * np.pi * t.dt.weekday / 7)
    df["weekend"] = (t.dt.weekday >= 5).astype(float)
    return df


def add_schedule_features(df, schedule):
    """일정에서 '이 칸에 예정된 인원'을 만든다. 미래에도 알 수 있는 값이라 예측 입력(미래 공변량)으로 쓸 수 있다."""
    df["sched_people"] = 0.0
    for r in schedule.itertuples():
        m = (df.space == r.space) & (df.ts >= r.start) & (df.ts < r.end)
        df.loc[m, "sched_people"] = np.maximum(df.loc[m, "sched_people"], r.planned_people)
    df["sched_active"] = (df.sched_people > 0).astype(float)
    return df


def build_frame(raw):
    """불러온 데이터 → 5분 분석용 표. 공간 × 시각이 빠짐없이 있는 직사각형 모양."""
    s, removed = clean(raw["sensors"])
    df = fill_short_gaps(resample(s))
    df = add_time_features(df)
    df = add_schedule_features(df, raw["schedule"])
    spaces = sorted(df.space.unique())
    for i, sp in enumerate(spaces):
        df[f"space_{i}"] = (df.space == sp).astype(float)
    df.attrs.update(removed=removed, spaces=spaces)
    return df.sort_values(["space", "ts"]).reset_index(drop=True)


# ---------------------------------------------------------------- 분할·정규화
def split_bounds(df, meta=None):
    """시간 순서로 학습 3주 / 검증 1주 / 시험 1주(고장이 심어진 마지막 주). 무작위로 섞지 않는다."""
    t0 = df.ts.min().normalize()
    test_start = pd.Timestamp(meta["test_start"]) if meta and "test_start" in meta else df.ts.max().normalize() - pd.Timedelta(days=7)
    val_start = pd.Timestamp(meta["val_start"]) if meta and "val_start" in meta else test_start - pd.Timedelta(days=7)
    return dict(train=(t0, val_start), val=(val_start, test_start), test=(test_start, df.ts.max() + pd.Timedelta(minutes=5)))


def period_mask(ts, bounds, name):
    a, b = bounds[name]
    return (ts >= a) & (ts < b)


class Scaler:
    """평균·표준편차를 학습 기간 데이터로만 계산한다."""
    def __init__(self, cols):
        self.cols = cols

    def fit(self, df):
        self.mean = df[self.cols].mean().values.astype(np.float32)
        self.std = (df[self.cols].std().values + 1e-6).astype(np.float32)
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def to_dict(self):
        return dict(cols=self.cols, mean=self.mean.tolist(), std=self.std.tolist())

    @classmethod
    def from_dict(cls, d):
        s = cls(d["cols"]); s.mean = np.array(d["mean"], np.float32); s.std = np.array(d["std"], np.float32)
        return s


# ---------------------------------------------------------------- 고장 구간
def fault_mask(df, faults, pad_after_min=60):
    """고장 구간(+끝난 뒤 영향이 남는 60분)을 표시. 예측 학습·평가에서 뺄 때 쓴다."""
    m = np.zeros(len(df), bool)
    if faults is None:
        return m
    for f in faults.itertuples():
        m |= ((df.space == f.space) & (df.ts >= f.start)
              & (df.ts < f.end + pd.Timedelta(minutes=pad_after_min))).values
    return m


# ---------------------------------------------------------------- 평가 시점
def anchors(df, bounds, period, exclude=None, lookback=LOOKBACK, horizon=HORIZON):
    """예측 기준 시점 목록. 과거 lookback 칸과 미래 horizon 칸이 모두 관측(또는 짧은 보간)된 시점만.
    기준선과 모델이 모두 이 같은 시점에서 평가된다. 반환: (공간별 행 위치 배열) 리스트"""
    out = []
    pm = period_mask(df.ts, bounds, period).values
    ok_val = df[TARGETS].notna().all(axis=1).values
    excl = exclude if exclude is not None else np.zeros(len(df), bool)
    for space, g in df.groupby("space"):
        pos = g.index.values
        ok = ok_val[pos] & ~excl[pos]
        # 길이 lookback+horizon 창 안이 모두 ok 인지 누적합으로 빠르게 확인
        cs = np.concatenate([[0], np.cumsum(ok)])
        for i in range(lookback - 1, len(pos) - horizon):
            if pm[pos[i]] and cs[i + horizon + 1] - cs[i - lookback + 1] == lookback + horizon:
                out.append(pos[i])
    return np.array(out, dtype=np.int64)
