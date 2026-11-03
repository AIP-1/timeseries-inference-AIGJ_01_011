"""이상 탐지 공통 부품. 평가(과제 4)와 서비스(과제 8)가 같이 쓴다.

규칙(1분 데이터, 실시간으로 쓸 수 있게 과거만 본다)
  spike : 직전 10분 중앙값에서 크게 벗어남
  stuck : 같은 값이 N분 넘게 그대로
  gap   : 마지막 수신 뒤 N분 넘게 데이터 없음 (서비스용)
  range : 물리 범위 밖 (서비스용)
모델(5분 데이터)
  AE    : 과거 1시간 창 + 시간대를 복원, 복원 오차가 크면 이상. 센서별 오차로 원인을 가리킨다
"""
import numpy as np
import pandas as pd

from . import data as D

SPIKE_TH = {"co2": 300.0, "temperature": 3.0, "humidity": 12.0}
STUCK_MIN = {"co2": 30, "temperature": 90, "humidity": 45}
GAP_MIN = 10
AE_STEPS = 12          # 5분 × 12 = 1시간
AE_TIME = ["tod_sin", "tod_cos", "weekend"]


def rule_spike(s1, sensor):
    """s1: 1분 간격 Series(결측 가능). 반환: bool Series (튄 칸에 표시, 실제 판정은 1분 뒤에 가능)
    한 칸만 크게 뛰었다가 바로 돌아오는 경우만 튄 값으로 본다. 수업 시작처럼 꾸준히 오르는 변화는 제외."""
    th = SPIKE_TH[sensor]
    x = s1.dropna()  # 바로 앞뒤 1분이 결측이어도 그 다음 관측값과 비교한다
    d1 = x - x.shift(1)
    d2 = x.shift(-1) - x
    gap = (x.index.to_series().diff(-1).abs() <= pd.Timedelta(minutes=5)) & \
          (x.index.to_series().diff().abs() <= pd.Timedelta(minutes=5))
    hit = (d1.abs() > th) & (d2.abs() > 0.6 * th) & (np.sign(d1) != np.sign(d2)) & gap
    return hit.reindex(s1.index, fill_value=False)


NIGHT = (1, 7)          # 영점 틀어짐 판정 시간대: 사람이 없는 새벽
NIGHT_EXCESS = 120.0    # 외기(420)보다 이만큼 높으면 의심
NIGHT_RUN_MIN = 30


def rule_night_drift(s1, limit=420 + NIGHT_EXCESS):
    """새벽(사람 없음)의 CO2는 평소 새벽 수준이어야 한다. 30분 평균이 limit 넘게 이어지면 영점 틀어짐 의심.
    limit 기본값은 외기(420)+120. 실데이터는 공간마다 학습 기간 새벽 값의 상위 1%로 정한다(service/config.yaml)."""
    h = s1.index.hour
    night = (h >= NIGHT[0]) & (h < NIGHT[1])
    m = s1.rolling(NIGHT_RUN_MIN, min_periods=NIGHT_RUN_MIN // 2).mean()
    return pd.Series(night, index=s1.index) & (m > limit)


def rule_stuck(s1, sensor, min_count=1):
    """같은 값이 STUCK_MIN 분 넘게, 그리고 min_count 번 넘게 이어진 칸(결측은 건너뛰고 셈).
    10분마다 보내는 센서는 몇 번만 같아도 시간이 차므로 횟수 조건을 함께 본다."""
    x = s1.dropna()
    run_id = (x != x.shift()).cumsum()
    run_start = x.index.to_series().groupby(run_id.values).transform("first")
    dur = (x.index.to_series() - run_start).dt.total_seconds() / 60
    count = x.groupby(run_id.values).cumcount() + 1
    flag = pd.Series(False, index=s1.index)
    flag.loc[x.index] = ((dur >= STUCK_MIN[sensor]) & (count.values >= min_count)).values
    return flag


def rules_1min(sensors_1min, night_limit=None, stuck_min_count=1):
    """공간별·센서별 규칙 경보 표: columns = ts, space, rule, sensor
    night_limit: {공간: 새벽 CO2 기준 ppm} (없으면 기본값), stuck_min_count: 값 고정으로 볼 최소 연속 횟수"""
    out = []
    for space, g in sensors_1min.groupby("space"):
        g = g.set_index("ts").sort_index()
        g = g[~g.index.duplicated()].asfreq("1min")
        for sensor in D.SENSORS:
            for rule in ("spike", "stuck"):
                f = rule_spike(g[sensor], sensor) if rule == "spike" else rule_stuck(g[sensor], sensor, stuck_min_count)
                ts = f.index[f.fillna(False).values]
                out.append(pd.DataFrame(dict(ts=ts, space=space, rule=rule, sensor=sensor)))
        f = rule_night_drift(g["co2"], (night_limit or {}).get(space, 420 + NIGHT_EXCESS))
        out.append(pd.DataFrame(dict(ts=f.index[f.values], space=space, rule="night_drift", sensor="co2")))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------- 오토인코더 입력
def ae_matrix(df, spec):
    """5분 표 → 창마다 [센서 3개 × 12칸(정규화) + 끝 시점 시간대 3개] 벡터와 그 끝 행 위치."""
    x = ((df[D.SENSORS].to_numpy(np.float32) - spec.mu) / spec.sd)
    t = df[AE_TIME].to_numpy(np.float32)
    ok = np.isfinite(x).all(axis=1)
    rows, pos = [], []
    for space, g in df.groupby("space"):
        p = g.index.values
        cs = np.concatenate([[0], np.cumsum(ok[p])])
        for i in range(AE_STEPS - 1, len(p)):
            if cs[i + 1] - cs[i + 1 - AE_STEPS] == AE_STEPS:
                w = x[p[i - AE_STEPS + 1:i + 1]]
                rows.append(np.concatenate([w.T.ravel(), t[p[i]]]))
                pos.append(p[i])
    return np.array(rows, np.float32), np.array(pos, np.int64)


def ae_scores(model, X):
    """반환: 전체 점수(N,), 센서별 오차(N, 3)"""
    import torch
    with torch.no_grad():
        R = model(torch.tensor(X)).numpy()
    e = (R - X) ** 2
    per = e[:, :3 * AE_STEPS].reshape(len(X), 3, AE_STEPS).mean(axis=2)
    return per.mean(axis=1), per


# ---------------------------------------------------------------- 경보를 사건으로 묶기
def to_events(alarms, merge_min=30):
    """alarms: DataFrame(ts, space, detector, sensor) → 같은 공간에서 merge_min 안에 이어지는 경보를 한 사건으로."""
    ev = []
    for (space,), g in alarms.sort_values("ts").groupby(["space"]):
        start = prev = None
        sensors = {}
        for r in g.itertuples():
            if prev is not None and (r.ts - prev) > pd.Timedelta(minutes=merge_min):
                ev.append(dict(space=space, start=start, end=prev, sensor=max(sensors, key=sensors.get)))
                start, sensors = None, {}
            start = start or r.ts
            sensors[r.sensor] = sensors.get(r.sensor, 0) + 1
            prev = r.ts
        if start is not None:
            ev.append(dict(space=space, start=start, end=prev, sensor=max(sensors, key=sensors.get)))
    return pd.DataFrame(ev, columns=["space", "start", "end", "sensor"])


def match(events, faults, tol_after=30):
    """고장마다 첫 경보 시각(탐지 지연), 고장과 겹치지 않는 경보 사건 수(헛경보)."""
    res, used = [], np.zeros(len(events), bool)
    for f in faults.itertuples():
        m = ((events.space == f.space) & (events.end >= f.start)
             & (events.start <= f.end + pd.Timedelta(minutes=tol_after))).values
        used |= m
        if m.any():
            first = events[m].start.min()
            cause = events[m].sort_values("start").sensor.iloc[0]
            res.append(dict(fault_id=f.fault_id, kind=f.kind, space=f.space, detected=True,
                            delay_min=max(0.0, (first - f.start).total_seconds() / 60), cause_ok=cause == f.sensor))
        else:
            res.append(dict(fault_id=f.fault_id, kind=f.kind, space=f.space, detected=False, delay_min=np.nan,
                            cause_ok=False))
    return pd.DataFrame(res), int((~used).sum())
