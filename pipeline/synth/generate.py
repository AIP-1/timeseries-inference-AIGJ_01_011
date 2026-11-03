"""과제 설명에 맞춘 합성 데이터 생성기 (제공 키트를 받지 못해 직접 만든 것).

공간 4곳(강의실A, 강의실B, 회의실, 라운지)의 1분 CO2·온도·습도를
수업 시간표·회의 예약·라운지 이용에 따라 물리적으로 만든다.

  CO2   : 질량 보존 dC/dt = N·G/V − (Q/V)(C − C_out)
  온도  : 냉난방 가동 중에는 설정 온도로 수렴, 꺼지면 외기 쪽으로 흐름 + 사람 발열
  습도  : 기준값 + 사람 수분 − 냉방 제습 + 잡음

현장의 불완전함(짧은·긴 결측, 0ppm 같은 깨진 값, 측정 잡음)과
마지막 1주에 정답을 아는 고장 5종(튄 값, 값 고정, 영점 틀어짐, 냉방 정지, 환기 정지)을 심는다.

출력 (out 폴더):
  sensors.csv     ts, space, co2, temperature, humidity      (측정값, 결측은 행 없음)
  schedule.csv    space, start, end, kind, planned_people    (운영 자료. 미래도 알 수 있음)
  occupancy.csv   ts, space, people                          (실제 인원. '직접 관찰' 정답용)
  faults.csv      fault_id, space, kind, sensor, start, end  (고장 정답)
  meta.json       생성 조건

사용법:  python synth/generate.py --out data/synth --weeks 5 --seed 42
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SPACES = {
    #            부피 m3, 정원, 환기(ACH) 냉난방 켜짐/꺼짐, 냉난방 응답 시간상수(분)
    "강의실A": dict(volume=210, capacity=30, ach_on=1.6, ach_off=0.3, tau_hvac=35),
    "강의실B": dict(volume=180, capacity=24, ach_on=1.4, ach_off=0.3, tau_hvac=35),
    "회의실":  dict(volume=60,  capacity=8,  ach_on=1.2, ach_off=0.25, tau_hvac=25),
    "라운지":  dict(volume=150, capacity=15, ach_on=2.0, ach_off=0.5, tau_hvac=45),
}
C_OUT = 420.0          # 외기 CO2 ppm
G_PERSON = 0.0052 * 60 / 1000  # 앉아 있는 사람 1명 CO2 배출 m3/min (0.0052 L/s)
HEAT_PER_PERSON = 0.0022       # 사람 1명이 1분에 올리는 온도(℃·m3)/부피 — 대략값
SETPOINT = 24.0
HVAC_HOURS = (8.5, 18.5)       # 평일 냉난방 가동 시간


def lecture_schedule(days, rng):
    """평일 오전 9~12시, 오후 13~18시 수업. 가끔 휴강·보강."""
    rows = []
    for d in days:
        if d.weekday() >= 5:
            continue
        for space in ("강의실A", "강의실B"):
            cap = SPACES[space]["capacity"]
            if rng.random() < 0.05:  # 휴강
                continue
            for sh, eh in ((9, 12), (13, 18)):
                if space == "강의실B" and sh == 13 and rng.random() < 0.3:
                    eh = 16  # B는 오후 수업이 일찍 끝나는 날이 있다
                rows.append(dict(space=space, start=d + pd.Timedelta(hours=sh), end=d + pd.Timedelta(hours=eh),
                                 kind="수업", planned_people=cap))
    return rows


def meeting_schedule(days, rng):
    rows = []
    for d in days:
        if d.weekday() >= 5:
            continue
        t = d + pd.Timedelta(hours=9)
        while t < d + pd.Timedelta(hours=18):
            t += pd.Timedelta(minutes=int(rng.choice([0, 30, 60, 90, 120])))
            dur = int(rng.choice([30, 60, 60, 90, 120]))
            if t + pd.Timedelta(minutes=dur) > d + pd.Timedelta(hours=19):
                break
            if rng.random() < 0.55:
                rows.append(dict(space="회의실", start=t, end=t + pd.Timedelta(minutes=dur), kind="회의",
                                 planned_people=int(rng.integers(2, 8))))
            t += pd.Timedelta(minutes=dur)
    return rows


def lounge_schedule(days):
    """라운지는 예약이 없고 운영 자료로 '점심·쉬는 시간 혼잡 예상'만 있다."""
    rows = []
    for d in days:
        if d.weekday() >= 5:
            continue
        for sh, sm, eh, em, n in ((10, 50, 11, 10, 8), (12, 0, 13, 0, 12), (15, 50, 16, 10, 8)):
            rows.append(dict(space="라운지", start=d + pd.Timedelta(hours=sh, minutes=sm),
                             end=d + pd.Timedelta(hours=eh, minutes=em), kind="휴식", planned_people=n))
    return rows


def actual_people(idx, sched, space, rng):
    """일정에서 실제 인원을 만든다: 결석, 지각·조퇴, 노쇼, 연장, 예약 없는 사용."""
    n = np.zeros(len(idx))
    t0 = idx[0]
    to_i = lambda t: int((t - t0) / pd.Timedelta(minutes=1))
    cap = SPACES[space]["capacity"]
    for r in sched[sched.space == space].itertuples():
        s, e = to_i(r.start), to_i(r.end)
        if r.kind == "회의":
            if rng.random() < 0.15:          # 노쇼
                continue
            e += int(rng.choice([0, 0, 0, 10, 20, 30]))  # 연장
            s += int(rng.integers(0, 10))
            people = max(1, r.planned_people + int(rng.integers(-2, 2)))
        elif r.kind == "수업":
            people = int(cap * rng.uniform(0.6, 1.0))   # 결석
            s += int(rng.integers(-10, 5))              # 일찍 오거나 지각
            # 쉬는 시간마다(50분 수업 10분 휴식) 절반이 나간다
        else:  # 라운지 혼잡 시간
            people = int(rng.poisson(r.planned_people))
        s, e = max(s, 0), min(e, len(n))
        seg = np.full(e - s, float(people))
        if r.kind == "수업":
            mins = np.arange(s, e) - s
            seg[(mins % 60) >= 50] *= 0.5
            if rng.random() < 0.08:  # 일찍 끝나는 날
                seg[int(len(seg) * rng.uniform(0.5, 0.85)):] = 0
        if r.kind == "휴식":
            seg = seg * rng.uniform(0.6, 1.2, size=len(seg)).round(1)
        n[s:e] = np.maximum(n[s:e], seg)
    # 예약 없는 사용 / 라운지 수시 이용
    days = pd.Series(idx.normalize().unique())
    for d in days:
        if d.weekday() >= 5:
            continue
        k = {"회의실": rng.poisson(0.4), "라운지": rng.poisson(5)}.get(space, 0)
        for _ in range(k):
            s = to_i(d + pd.Timedelta(hours=float(rng.uniform(9, 18.5))))
            dur = int(rng.integers(10, 60))
            n[s:s + dur] = np.maximum(n[s:s + dur], rng.integers(1, 4))
        if space in ("강의실A", "강의실B") and rng.random() < 0.15:  # 수업 끝나고 남아서 공부
            s = to_i(d + pd.Timedelta(hours=18))
            n[s:s + int(rng.integers(30, 120))] = rng.integers(2, 6)
    return n


def simulate_space(idx, people, space, outdoor_t, outdoor_rh, rng, vent_fail=None, cool_fail=None):
    """1분 간격으로 CO2·온도·습도를 적분한다. vent_fail/cool_fail: 고장 구간 bool 배열."""
    p = SPACES[space]
    V = p["volume"]
    hours = idx.hour + idx.minute / 60
    hvac_on = (idx.weekday < 5) & (hours >= HVAC_HOURS[0]) & (hours < HVAC_HOURS[1])
    co2 = np.empty(len(idx)); temp = np.empty(len(idx)); rh = np.empty(len(idx))
    c, t = C_OUT + 30, outdoor_t[0] + 3
    window = 0
    for i in range(len(idx)):
        ach = p["ach_on"] if hvac_on[i] else p["ach_off"]
        if vent_fail is not None and vent_fail[i]:
            ach *= 0.45   # 환기 장치 고장: 사용 중 CO2 상승 폭이 평소의 약 1.8배
        if window > 0:    # 창문 개방
            ach += 3.0; window -= 1
        elif people[i] > 0 and c > 1100 and rng.random() < 0.01:
            window = int(rng.integers(5, 20))
        q = ach * V / 60  # m3/min
        c += (people[i] * G_PERSON * 1e6 / V) - (q / V) * (c - C_OUT)
        cooling = hvac_on[i] and not (cool_fail is not None and cool_fail[i])
        gain = people[i] * HEAT_PER_PERSON * 60 / V
        if cooling:
            t += (SETPOINT - t) / p["tau_hvac"] + gain * 0.25
        else:
            t += (outdoor_t[i] + 4 - t) / 400 + gain  # 건물 단열: 천천히 외기 쪽으로
        co2[i], temp[i] = c, t
        rh[i] = (outdoor_rh[i] * (0.75 if cooling else 0.95) + people[i] * 0.25
                 - (t - 22) * 0.8)
    return co2, temp, np.clip(rh, 15, 95)


def outdoor(idx, rng):
    doy = idx.dayofyear.values
    hours = idx.hour.values + idx.minute.values / 60
    daily = 6 * np.sin(2 * np.pi * (hours - 9) / 24)
    weather = np.cumsum(rng.normal(0, 0.02, len(idx)))
    weather -= pd.Series(weather).rolling(60 * 24 * 3, min_periods=1, center=True).mean().values
    t = 27 - 0.05 * (doy - doy[0]) + daily + weather
    rh = 65 - 1.5 * daily + rng.normal(0, 1, len(idx))
    return t, rh


def plant_faults(df, idx, test_start, rng):
    """마지막 1주(test 기간)에 고장 5종을 심고 정답표를 돌려준다."""
    faults = []
    test_mask = idx >= test_start
    test_days = [d for d in pd.Series(idx[test_mask].normalize().unique()) if d.weekday() < 5]

    def at(day, h):
        return day + pd.Timedelta(hours=h)

    plan = [  # (종류, 공간, 센서, 날짜 순번, 시작 시각, 길이 분)
        ("튄 값", "강의실A", "co2", 0, 10.3, 1), ("튄 값", "라운지", "temperature", 2, 14.1, 1),
        ("튄 값", "회의실", "co2", 3, 11.7, 1),
        ("값 고정", "강의실B", "co2", 1, 9.5, 180), ("값 고정", "라운지", "humidity", 3, 13.0, 90),
        ("영점 틀어짐", "강의실A", "co2", 2, 3.0, 600),
        ("냉방 정지", "강의실B", "temperature", 2, 13.0, 240), ("냉방 정지", "강의실A", "temperature", 4, 13.0, 240),
        ("환기 정지", "회의실", "co2", 1, 9.0, 480), ("환기 정지", "강의실A", "co2", 3, 9.0, 300),
    ]
    for k, (kind, space, sensor, di, h, dur) in enumerate(plan):
        day = test_days[min(di, len(test_days) - 1)]
        s = at(day, h).floor("min")
        e = s + pd.Timedelta(minutes=dur)
        faults.append(dict(fault_id=k + 1, space=space, kind=kind, sensor=sensor, start=s, end=e))
    return pd.DataFrame(faults)


def apply_sensor_faults(df, faults, rng):
    """센서 고장(튄 값·값 고정·영점 틀어짐)을 측정값에 반영한다. 설비 고장은 시뮬레이션에서 이미 반영."""
    for f in faults.itertuples():
        m = (df.space == f.space) & (df.ts >= f.start) & (df.ts < f.end)
        if f.kind == "튄 값":
            jump = {"co2": rng.choice([-1, 1]) * rng.uniform(600, 1200), "temperature": rng.uniform(6, 10),
                    "humidity": rng.uniform(20, 30)}[f.sensor]
            # 물리적으로 가능한 범위 안에서 튀게 한다(불가능한 값은 '깨진 값'으로 따로 만든다)
            if f.sensor == "co2" and (df.loc[m, "co2"] + jump < 350).any():
                jump = abs(jump)
            df.loc[m, f.sensor] += jump
        elif f.kind == "값 고정":
            first = df.loc[m, f.sensor]
            if len(first):
                df.loc[m, f.sensor] = first.iloc[0]
        elif f.kind == "영점 틀어짐":
            n = int(m.sum())
            df.loc[m, f.sensor] += np.linspace(0, rng.uniform(250, 400), n)
            # 고장 이후 정비 전까지 틀어진 값이 유지되는 대신, 정답 구간을 명확히 하려고 끝나면 보정됨
    return df


def add_noise(df, rng):
    """측정 잡음. 센서 고장(값 고정 등)보다 먼저 더해야 고정된 값이 다시 흔들리지 않는다."""
    n = len(df)
    df["co2"] += rng.normal(0, 8, n)
    df["temperature"] += rng.normal(0, 0.08, n)
    df["humidity"] += rng.normal(0, 0.6, n)
    return df


def degrade(df, rng, start, end):
    """현장의 불완전함: 결측(짧은·긴), 깨진 값, 해상도."""
    n = len(df)
    keep = np.ones(n, bool)
    ts = df.ts.values
    # 짧은 결측: 센서마다 메시지 3% 유실
    keep &= rng.random(n) > 0.03
    # 게이트웨이 끊김: 전체 공간 동시 결측 1~6시간, 4회
    for _ in range(4):
        s = start + pd.Timedelta(minutes=int(rng.integers(0, int((end - start).total_seconds() // 60) - 400)))
        keep &= ~((ts >= np.datetime64(s)) & (ts < np.datetime64(s + pd.Timedelta(hours=float(rng.uniform(1, 6))))))
    # 한 공간만 수십 분 끊김
    for space in SPACES:
        for _ in range(3):
            s = start + pd.Timedelta(minutes=int(rng.integers(0, int((end - start).total_seconds() // 60) - 100)))
            keep &= ~((df.space.values == space) & (ts >= np.datetime64(s))
                      & (ts < np.datetime64(s + pd.Timedelta(minutes=int(rng.integers(10, 60))))))
    df = df[keep].copy()
    # 깨진 값: 0ppm, 65535, 온도 -40 등 (고장과 별개로 무작위로 드물게)
    bad = rng.random(len(df)) < 0.0015
    df.loc[bad, "co2"] = rng.choice([0, 0, 65535], bad.sum())
    bad_t = rng.random(len(df)) < 0.0005
    df.loc[bad_t, "temperature"] = rng.choice([-40.0, 85.0], bad_t.sum())
    df["co2"] = df.co2.round(0)
    df["temperature"] = df.temperature.round(1)
    df["humidity"] = df.humidity.round(1)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synth")
    ap.add_argument("--weeks", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--start", default="2026-08-31")  # 월요일
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    start = pd.Timestamp(a.start)
    end = start + pd.Timedelta(weeks=a.weeks)
    idx = pd.date_range(start, end, freq="1min", inclusive="left")
    test_start = end - pd.Timedelta(weeks=1)
    days = list(pd.date_range(start, end, freq="D", inclusive="left"))

    sched = pd.DataFrame(lecture_schedule(days, rng) + meeting_schedule(days, rng) + lounge_schedule(days))
    faults = plant_faults(None, idx, test_start, rng)
    out_t, out_rh = outdoor(idx, rng)

    frames, occ = [], []
    for space in SPACES:
        people = actual_people(idx, sched, space, rng)
        mask = lambda kind: np.any([(idx >= f.start) & (idx < f.end) for f in faults.itertuples()
                                    if f.space == space and f.kind == kind] or [np.zeros(len(idx), bool)], axis=0)
        # 환기 정지·냉방 정지는 설비 고장: 물리 시뮬레이션에 반영
        co2, temp, rh = simulate_space(idx, people, space, out_t, out_rh, rng,
                                       vent_fail=mask("환기 정지"), cool_fail=mask("냉방 정지"))
        frames.append(pd.DataFrame(dict(ts=idx, space=space, co2=co2, temperature=temp, humidity=rh)))
        occ.append(pd.DataFrame(dict(ts=idx, space=space, people=people.astype(int))))

    df = pd.concat(frames, ignore_index=True)
    df = add_noise(df, rng)
    df = apply_sensor_faults(df, faults, rng)
    df = degrade(df, rng, start, end).sort_values(["ts", "space"])

    os.makedirs(a.out, exist_ok=True)
    df.to_csv(os.path.join(a.out, "sensors.csv"), index=False)
    sched.sort_values(["start", "space"]).to_csv(os.path.join(a.out, "schedule.csv"), index=False)
    pd.concat(occ).to_csv(os.path.join(a.out, "occupancy.csv"), index=False)
    faults.to_csv(os.path.join(a.out, "faults.csv"), index=False)
    meta = dict(generator="team4 synth v1 (자체 제작)", seed=a.seed, start=str(start), end=str(end),
                test_start=str(test_start), spaces=SPACES, c_out=C_OUT, setpoint=SETPOINT,
                hvac_hours=HVAC_HOURS, rows=len(df))
    with open(os.path.join(a.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1, default=str)
    print(f"saved {len(df):,} rows to {a.out}  ({start.date()} ~ {end.date()}, test from {test_start.date()})")
    print(df.groupby("space")[["co2", "temperature", "humidity"]].describe().T.round(1).to_string())


if __name__ == "__main__":
    main()
