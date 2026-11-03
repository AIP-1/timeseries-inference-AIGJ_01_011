"""과제 8. 상시 추론 서비스 — 매분 공간마다 예측·이상·상태를 계산해 MQTT 발행 + 조회 API 로 제공.

- 학습과 똑같은 전처리(tsi.data / tsi.features / tsi.state / tsi.anomaly)를 쓴다
- 모델은 ONNX Runtime(CPU)으로 돌린다(과제 7에서 변환 오차 ≈ 0 확인)
- 막 시작된 5분 칸(아직 1~2분치뿐)은 쓰지 않는다: 마지막으로 '다 찬' 5분 칸을 기준 시점으로
- 한 공간·한 계산이 실패해도 다른 공간과 다음 주기는 계속된다. 데이터가 끊기면 '데이터 끊김' 경보
- 같은 경보는 상태가 바뀌거나 30분이 지나야 다시 보낸다

설정: service/config.yaml   실행: ../.venv/bin/python service/infer.py
"""
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import onnxruntime as ort  # noqa: E402

from tsi import anomaly as A  # noqa: E402
from tsi import data as D  # noqa: E402
from tsi import state as S  # noqa: E402
from tsi.features import FeatureSpec  # noqa: E402

ART = os.path.join(ROOT, "artifacts")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
log = logging.getLogger("infer")
HISTORY_MIN = 240           # 계산에 쓰는 과거 데이터 길이(특징 중 가장 긴 것이 2시간)
REALERT_MIN = 30


class Models:
    def __init__(self, tag=""):
        """tag: 모델 파일 꼬리표. '' = 합성 데이터로 학습한 모델, '_real' = 실데이터로 학습한 모델"""
        best = json.load(open(os.path.join(ART, f"forecast_best{tag}.json")))["model"]
        fmeta = json.load(open(os.path.join(ART, f"forecast_{best}{tag}.json")))
        self.forecast_name = best
        self.spec = FeatureSpec.from_dict(fmeta["spec"])
        self.ae_meta = json.load(open(os.path.join(ART, f"ae{tag}.json")))
        self.ae_spec = FeatureSpec.from_dict(self.ae_meta["spec"])
        self.occ = json.load(open(os.path.join(ART, f"occ{tag}.json")))
        self.vent = json.load(open(os.path.join(ART, f"vent{tag}.json")))
        opt = ort.SessionOptions(); opt.intra_op_num_threads = 2
        self.sess = {k: ort.InferenceSession(os.path.join(ART, f"{k}{tag}.onnx"), opt, providers=["CPUExecutionProvider"])
                     for k in ("forecast", "ae", "occ", "vent")}

    def run(self, k, **feed):
        return self.sess[k].run(None, feed)[0]


class Service:
    def __init__(self, cfg):
        self.cfg = cfg
        self.m = Models(cfg.get("artifact_tag", ""))
        self.latest, self.alerts = {}, deque(maxlen=500)
        self.sent = {}  # (space, alert type) → (state, last sent time)
        self.status = dict(started=time.strftime("%Y-%m-%d %H:%M:%S"), cycles=0, failures=0, last_error=None)
        self.mqtt = self._mqtt()
        sched = cfg.get("schedule_csv")
        self.schedule = (pd.read_csv(os.path.join(ROOT, sched), parse_dates=["start", "end"]) if sched
                         else pd.DataFrame(columns=["space", "start", "end", "kind", "planned_people"]))
        self.space_map = cfg.get("space_map") or {}
        self.log_path = os.path.join(ROOT, "results", "service_cycles.jsonl")
        self.score_path = os.path.join(ROOT, "results", "forecast_scores.jsonl")
        self.pending = {}   # (space, 예측 대상 칸 시각, 몇 분 뒤) → 예측 CO2. 실제 값이 들어오면 채점
        self.scores = deque(maxlen=20000)  # (채점 시각, space, 몇 분 뒤, 절대 오차)

    # ------------------------------------------------------------ 자기 채점 (심화)
    def self_check(self, space, g, anchor, forecast):
        """지난 예측 중 이제 실제 값이 나온 것을 채점하고, 이번 예측을 채점 대기열에 넣는다."""
        actual = g.set_index("ts").co2
        for key in [k for k in self.pending if k[0] == space and k[1] <= anchor]:
            pred = self.pending.pop(key)
            y = actual.get(key[1])
            if y is not None and np.isfinite(y):
                self.scores.append((anchor, space, key[2], abs(float(y) - pred)))
                with open(self.score_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(target=str(key[1]), space=space, h=key[2], pred=round(pred, 1),
                                            actual=round(float(y), 1)), ensure_ascii=False) + "\n")
        if forecast:
            for h in (30, 60):
                k = h // D.STEP_MIN - 1
                self.pending[(space, pd.Timestamp(forecast["times"][k]), h)] = float(forecast["co2"][k])
        recent = [(h, e) for t, s, h, e in self.scores if s == space and t >= anchor - pd.Timedelta(hours=24)]
        out = {}
        for h in (30, 60):
            errs = [e for hh, e in recent if hh == h]
            out[f"mae{h}_24h"] = round(float(np.mean(errs)), 1) if errs else None
            out[f"n{h}"] = len(errs)
        ref = self.cfg.get("expected_mae60")
        out["expected_mae60"] = ref
        out["degraded"] = bool(ref and out["mae60_24h"] is not None and out["n60"] >= 120  # 2시간치 이상 채점된 뒤에만
                               and out["mae60_24h"] > 1.5 * ref)
        return out

    # ------------------------------------------------------------ 시각
    def now(self):
        """서비스 기준 시각. 재생 모드는 합성 데이터 시각, 실데이터 모드는 지금(KST)."""
        if self.cfg["mode"] == "replay":
            from service.replay import clock, sim_now
            return sim_now(clock())
        return pd.Timestamp.now().tz_localize(None)

    # ------------------------------------------------------------ 데이터
    def load(self, now):
        db = sqlite3.connect(f"file:{os.path.join(ROOT, self.cfg['db'])}?mode=ro", uri=True, timeout=10)
        col = "ts_sensor" if self.cfg["mode"] == "replay" else "ts_recv"
        lo = int((now - pd.Timedelta(minutes=HISTORY_MIN)).tz_localize("Asia/Seoul" if col == "ts_recv" else "UTC")
                 .value // 1_000_000)
        src = "synth" if self.cfg["mode"] == "replay" else "lorawan"  # 실데이터: 냉난방(xflow) 행은 빼고 센서만
        r = pd.read_sql_query(f"""SELECT {col} AS t, location, spot, metric, value FROM readings
                                  WHERE {col} >= ? AND flag = 0 AND source = ?
                                  AND metric IN ('co2','temperature','humidity')""",
                              db, params=(lo, src))
        db.close()
        ts = pd.to_datetime(r.t, unit="ms")
        if col == "ts_recv":
            ts = ts.dt.tz_localize("UTC").dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
        r["ts"] = ts.dt.floor("min")
        r["space"] = r.location.map(self.space_map).fillna(r.location)
        r = r[r.space.isin(self.cfg["spaces"]) & ~r.spot.isin(D.EXCLUDE_SPOTS)]
        # 학습(tsi.data.load_influx)과 같은 규칙: 센서가 여러 개면 CO2는 중앙값, 온습도는 평균
        co2 = r[r.metric == "co2"].groupby(["ts", "space"]).value.median().rename("co2")
        th = r[r.metric != "co2"].pivot_table(index=["ts", "space"], columns="metric", values="value", aggfunc="mean")
        s = pd.concat([co2, th], axis=1).reset_index()
        for c in D.SENSORS:
            if c not in s:
                s[c] = np.nan
        return s[s.ts <= now]

    def frame(self, s1, now):
        """1분 데이터 → 학습과 같은 5분 표. 미래 60분 칸(시간대·일정만)도 붙인다."""
        anchor = now.floor(D.STEP) - pd.Timedelta(minutes=D.STEP_MIN)  # 마지막으로 다 찬 5분 칸
        start = anchor - pd.Timedelta(minutes=HISTORY_MIN - D.STEP_MIN)
        grid = pd.date_range(start, anchor + pd.Timedelta(minutes=D.HORIZON * D.STEP_MIN + D.STEP_MIN - 1), freq="1min")
        filler = pd.DataFrame([(t, sp) for sp in self.cfg["spaces"] for t in grid], columns=["ts", "space"])
        s = filler.merge(s1[s1.ts <= anchor + pd.Timedelta(minutes=D.STEP_MIN - 1)], on=["ts", "space"], how="left")
        df = D.build_frame(dict(sensors=s, schedule=self.schedule))
        # 10분마다 보내는 센서는 기준 5분 칸이 비는 때가 절반이다. 직전 10분(2칸) 안의 값만 이어 쓴다
        # (학습 때는 앞뒤 값으로 보간했지만, 지금 시점에는 다음 값이 아직 없다)
        past = df.ts <= anchor
        filled = df[past].groupby("space")[D.SENSORS].ffill(limit=2)
        df.loc[past, D.SENSORS] = filled
        return df, anchor

    # ------------------------------------------------------------ 계산
    def compute(self):
        t0 = time.time()
        now = self.now()
        s1 = self.load(now)
        df, anchor = self.frame(s1, now)
        res, new_alerts = {}, []
        for space in self.cfg["spaces"]:
            try:
                res[space] = self.compute_space(df, s1, space, anchor, now)
            except Exception as e:  # 한 공간이 실패해도 다른 공간은 계속
                log.exception("space %s failed", space)
                res[space] = dict(space=space, status="error", error=str(e)[:200])
            new_alerts += self.alerts_for(res[space], now)
        took = (time.time() - t0) * 1000
        summary = dict(computed_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), data_time=str(anchor + pd.Timedelta(minutes=D.STEP_MIN)),
                       mode=self.cfg["mode"], took_ms=round(took, 1), spaces=list(res))
        self.latest = dict(summary=summary, spaces=res)
        self.publish(res, summary, new_alerts)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(**summary, alerts=[a["type"] + ":" + a["space"] for a in new_alerts],
                                    status={k: v.get("status") for k, v in res.items()}), ensure_ascii=False) + "\n")
        return took

    def compute_space(self, df, s1, space, anchor, now):
        g = df[df.space == space]
        i = g.index[g.ts == anchor][0]
        last_obs = s1[(s1.space == space) & s1.co2.notna()].ts.max()
        gap_min = (now - last_obs).total_seconds() / 60 if pd.notna(last_obs) else None
        out = dict(space=space, data_time=str(anchor + pd.Timedelta(minutes=D.STEP_MIN)),
                   last_observation=str(last_obs) if pd.notna(last_obs) else None,
                   minutes_since_data=round(gap_min, 1) if gap_min is not None else None)
        rules = {"data_gap": gap_min is None or gap_min > self.cfg.get("gap_min", A.GAP_MIN)}
        cur = g.loc[i, D.SENSORS].to_dict()
        out["current"] = {k: (None if pd.isna(v) else round(float(v), 2)) for k, v in cur.items()}

        # 규칙: 1분 데이터
        g1 = s1[s1.space == space]
        recent = now - pd.Timedelta(minutes=10)
        rr = (A.rules_1min(g1, self.cfg.get("night_drift_limit"), self.cfg.get("stuck_min_count", 1)) if len(g1)
              else pd.DataFrame(columns=["ts", "space", "rule", "sensor"]))
        rr = rr[rr.ts >= recent]
        for rule in ("spike", "stuck", "night_drift"):
            hit = rr[rr.rule == rule]
            rules[rule] = sorted(hit.sensor.unique().tolist()) if len(hit) else []
        last10 = g1[g1.ts >= recent]
        rules["out_of_range"] = [c for c, (lo, hi) in D.VALID.items() if ((last10[c] < lo) | (last10[c] > hi)).any()]
        out["rules"] = rules

        enough = (D.LOOKBACK <= g.loc[:i].co2.notna().tail(D.LOOKBACK).sum() + D.MAX_FILL_STEPS
                  and not rules["data_gap"] and g.loc[i, D.TARGETS].notna().all())  # 기준 칸 값이 있어야 변화량을 더할 수 있다
        if not enough:
            out["self_check"] = self.self_check(space, g, anchor, None)  # 지난 예측 채점은 계속
            out["status"] = "degraded"
            out["note"] = "최근 2시간 데이터가 부족하거나 끊겨 모델 계산을 건너뜀(규칙 경보만 제공)"
            return out

        # 예측
        p, f = self.m.spec.windows(df, np.array([i]), with_target=False)
        pr = self.m.run("forecast", past=p, future=f)
        now_v = df[D.TARGETS].to_numpy(np.float32)[[i]]
        pred = self.m.spec.denorm(pr, now_v)[0]
        times = [str(anchor + pd.Timedelta(minutes=D.STEP_MIN * (k + 1))) for k in range(D.HORIZON)]
        over = np.where(pred[:, 0] >= S.VENT_LIMIT)[0]
        out["forecast"] = dict(model=self.m.forecast_name, step_min=D.STEP_MIN,
                               times=times, co2=[round(float(x)) for x in pred[:, 0]],
                               temperature=[round(float(x), 2) for x in pred[:, 1]],
                               co2_exceeds_1000_in_min=int((over[0] + 1) * D.STEP_MIN) if len(over) else None)

        # 이상 (오토인코더)
        X, pos = A.ae_matrix(g.reset_index(drop=True), self.m.ae_spec)
        j = np.where(pos == (i - g.index[0]))[0]
        if len(j):
            x = X[j]
            rec = self.m.run("ae", x=x)
            per = ((rec - x) ** 2)[:, :3 * A.AE_STEPS].reshape(1, 3, A.AE_STEPS).mean(axis=2)[0]
            score = float(per.mean())
            out["anomaly"] = dict(score=round(score, 4), threshold=round(self.m.ae_meta["threshold"], 4),
                                  is_anomaly=score > self.m.ae_meta["threshold"],
                                  cause_sensor=D.SENSORS[int(per.argmax())],
                                  per_sensor={k: round(float(v), 4) for k, v in zip(D.SENSORS, per)})

        # 상태
        for tag, feats in (("occ", S.occupancy_features), ("vent", S.vent_features)):
            meta = getattr(self.m, tag)
            F = feats(g)[meta["cols"]].loc[[i]].values
            x = np.nan_to_num((F - np.array(meta["mu"])) / np.array(meta["sd"])).astype(np.float32)
            logit = self.m.run(tag, x=x)[0]
            prob = float(np.exp(logit[1]) / np.exp(logit).sum())
            out["occupied" if tag == "occ" else "ventilation_alert"] = dict(
                probability=round(prob, 3), value=prob >= meta["threshold"], threshold=round(meta["threshold"], 3))
        if out["current"]["co2"] is not None and out["current"]["co2"] >= S.VENT_LIMIT:
            out["ventilation_alert"]["note"] = "이미 1,000ppm 이상 — 지금 환기 필요"
        out["self_check"] = self.self_check(space, g, anchor, out["forecast"])
        out["status"] = "ok"
        return out

    # ------------------------------------------------------------ 경보·발행
    def alerts_for(self, r, now):
        active = {}
        rules = r.get("rules", {})
        if rules.get("data_gap"):
            active["data_gap"] = f"{r.get('minutes_since_data')}분째 데이터 없음"
        for k in ("spike", "stuck", "night_drift", "out_of_range"):
            if rules.get(k):
                active[k] = ",".join(rules[k])
        if r.get("anomaly", {}).get("is_anomaly"):
            active["anomaly"] = r["anomaly"]["cause_sensor"]
        if r.get("ventilation_alert", {}).get("value"):
            active["ventilation_soon"] = f"30분 안에 1,000ppm 초과 예상 (p={r['ventilation_alert']['probability']})"
        mins = r.get("forecast", {}).get("co2_exceeds_1000_in_min")
        if mins is not None:
            active["co2_forecast_1000"] = f"{mins}분 뒤 1,000ppm 초과 예측"
        sc = r.get("self_check") or {}
        if sc.get("degraded"):
            active["forecast_degraded"] = (f"최근 24시간 60분 예측 오차 {sc['mae60_24h']}ppm "
                                           f"(평소 {sc['expected_mae60']}ppm의 1.5배 초과)")
        new = []
        wall = time.time()
        for typ in set(active) | {k[1] for k in self.sent if k[0] == r["space"]}:
            key = (r["space"], typ)
            on = typ in active
            prev_on, prev_t = self.sent.get(key, (False, 0))
            if on and (not prev_on or wall - prev_t >= REALERT_MIN * 60):
                new.append(dict(space=r["space"], type=typ, state="on", detail=active[typ], data_time=r.get("data_time")))
                self.sent[key] = (True, wall)
            elif not on and prev_on:
                new.append(dict(space=r["space"], type=typ, state="off", detail="정상으로 돌아옴", data_time=r.get("data_time")))
                self.sent[key] = (False, wall)
        self.alerts.extend(new)
        return new

    def _mqtt(self):
        mc = self.cfg.get("mqtt") or {}
        if not mc.get("enabled"):
            return None
        import paho.mqtt.client as mqtt
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=mc.get("client_id", "team4-infer"))
        if mc.get("username_env"):
            c.username_pw_set(os.environ.get(mc["username_env"]), os.environ.get(mc.get("password_env", ""), None))
        c.reconnect_delay_set(1, 30)
        c.connect_async(mc["host"], int(mc.get("port", 1883)), 60)
        c.loop_start()
        return c

    def publish(self, res, summary, new_alerts):
        if not self.mqtt:
            return
        p = self.cfg["mqtt"]["topic_prefix"]
        for sp, r in res.items():  # retained: 새로 구독한 쪽도 최신값을 바로 받는다
            self.mqtt.publish(f"{p}/state/{sp}", json.dumps(r, ensure_ascii=False), qos=1, retain=True)
        self.mqtt.publish(f"{p}/summary", json.dumps(summary, ensure_ascii=False), qos=1, retain=True)
        for a in new_alerts:
            self.mqtt.publish(f"{p}/alerts", json.dumps(a, ensure_ascii=False), qos=1)

    # ------------------------------------------------------------ 반복
    def loop(self):
        while True:
            t = time.time()
            try:
                took = self.compute()
                self.status.update(cycles=self.status["cycles"] + 1, last_ok=time.strftime("%H:%M:%S"),
                                   last_took_ms=round(took, 1))
            except Exception as e:  # 이번 주기가 실패해도 다음 주기는 다시 시도
                log.exception("cycle failed")
                self.status.update(failures=self.status["failures"] + 1, last_error=str(e)[:200])
            time.sleep(max(1, 60 - (time.time() - t)))


def make_api(svc):
    from fastapi import FastAPI, HTTPException

    app = FastAPI(title="team4 시계열 추론 서비스", version="1")

    @app.get("/health")
    def health():
        return svc.status

    @app.get("/spaces")
    def spaces():
        return svc.cfg["spaces"]

    @app.get("/latest")
    def latest():
        return svc.latest

    @app.get("/latest/{space}")
    def latest_space(space: str):
        r = svc.latest.get("spaces", {}).get(space)
        if r is None:
            raise HTTPException(404, f"모르는 공간: {space}")
        return r

    @app.get("/forecast/{space}")
    def forecast(space: str, minutes: int = 60):
        """예: /forecast/회의실?minutes=30 → 30분 뒤 CO2·온도"""
        r = latest_space(space)
        f = r.get("forecast")
        if not f:
            raise HTTPException(503, "지금은 예측이 없음(데이터 부족·끊김). /latest/{space} 의 note 참고")
        k = min(max(minutes // D.STEP_MIN, 1), D.HORIZON) - 1
        return dict(space=space, minutes=(k + 1) * D.STEP_MIN, time=f["times"][k], co2=f["co2"][k],
                    temperature=f["temperature"][k], data_time=r["data_time"])

    @app.get("/alerts")
    def alerts(limit: int = 50):
        return list(svc.alerts)[-limit:]

    return app


def main():
    with open(os.path.join(ROOT, "service", "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, "..", ".env"))
    svc = Service(cfg)
    threading.Thread(target=svc.loop, daemon=True).start()
    import uvicorn
    uvicorn.run(make_api(svc), host=cfg["api"]["host"], port=int(cfg["api"]["port"]), log_level="warning")


if __name__ == "__main__":
    main()
