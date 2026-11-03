"""MQTT 센서 메시지를 받아 SQLite 에 공간·항목·시각별로 저장하는 상시 수집기.

- 해석 규칙은 config.yaml 에 있다. 형식이 바뀌면 설정만 고친다.
- 센서가 찍은 시각(ts_sensor)과 받은 시각(ts_recv)을 둘 다 저장한다.
  일부 게이트웨이 시계가 약 5분 앞서 있어 분석 기준 시각은 ts_recv 를 쓴다.
- 같은 메시지가 두 번 오면(재전송) msg_id 로 걸러 한 번만 저장한다.
- 브로커가 끊기면 스스로 다시 붙고, 죽으면 systemd 가 다시 띄운다.
"""
import fnmatch
import json
import logging
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime

import paho.mqtt.client as mqtt
import yaml
from dotenv import load_dotenv

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    stream=sys.stdout)
log = logging.getLogger("collector")

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS readings (
    ts_recv   INTEGER NOT NULL,   -- 받은 시각 (epoch ms, UTC)
    ts_sensor INTEGER,            -- 센서/서버가 찍은 시각 (epoch ms, UTC)
    source    TEXT NOT NULL,
    location  TEXT NOT NULL,
    spot      TEXT,
    device    TEXT NOT NULL,
    metric    TEXT NOT NULL,
    value     REAL NOT NULL,
    flag      INTEGER NOT NULL DEFAULT 0,  -- 1: valid_range 밖
    msg_id    TEXT NOT NULL,
    UNIQUE (msg_id, device, metric)
);
CREATE INDEX IF NOT EXISTS idx_loc_metric_ts ON readings(location, metric, ts_recv);
CREATE TABLE IF NOT EXISTS raw_messages (
    ts_recv INTEGER NOT NULL,
    topic   TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS unparsed (
    ts_recv INTEGER NOT NULL,
    topic   TEXT NOT NULL,
    reason  TEXT NOT NULL
);
"""


CLOCK_SANE = 1_767_225_600  # 2026-01-01 UTC. 이보다 이르면 시스템 시계가 아직 안 맞은 것


def mqtt_match(pattern, topic):
    """MQTT 와일드카드(+, #) 비교."""
    p, t = pattern.split("/"), topic.split("/")
    for i, part in enumerate(p):
        if part == "#":
            return True
        if i >= len(t) or (part != "+" and part != t[i]):
            return False
    return len(p) == len(t)


def iso_to_ms(s):
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp() * 1000)


def numeric(v):
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return None


def pick(values, src):
    """설정의 rename/metrics 를 적용해 저장할 (metric, value) 목록을 만든다."""
    out = []
    for k, v in values.items():
        name = src.get("rename", {}).get(k, k)
        if src.get("metrics") and name not in src["metrics"]:
            continue
        x = numeric(v)
        if x is not None:
            out.append((name, x))
    return out


def parse_chirpstack(d, src, topic):
    di = d.get("deviceInfo", {})
    tags = di.get("tags") or {}
    obj = d.get("object")
    if not obj:
        raise ValueError("no object field")
    location = tags.get(src.get("location_tag", "location")) or "unknown"
    device = di.get("deviceName") or di.get("devEui") or topic.split("/")[3]
    msg_id = d.get("deduplicationId") or f"{di.get('devEui')}:{d.get('fCnt')}"
    ts_sensor = iso_to_ms(d["time"]) if d.get("time") else None
    return [dict(location=location, spot=tags.get(src.get("spot_tag", "spot")), device=device,
                 ts_sensor=ts_sensor, msg_id=msg_id, metrics=pick(obj, src))]


def parse_xflow(d, src, topic):
    rows = []
    unit_id = topic.split("/")[-2]
    for place, fields in (d.get("values") or {}).items():
        rows.append(dict(location=place, spot=None, device=f"{topic.split('/')[1]}-{unit_id}",
                         ts_sensor=d.get("ts"), msg_id=f"{unit_id}:{d.get('ts')}",
                         metrics=pick(fields, src)))
    return rows


def parse_json_flat(d, src, topic):
    """새 센서용 범용 해석기: {"ts": ..., "co2": 512, ...} 처럼 값이 한 단계로 펼쳐진 JSON.
    위치·장치는 토픽의 몇 번째 칸에서, 또는 payload 필드에서 가져온다(설정으로 지정)."""
    parts = topic.split("/")
    loc = d.get(src["location_field"]) if src.get("location_field") else parts[src.get("location_from_topic", 1)]
    dev = d.get(src["device_field"]) if src.get("device_field") else parts[src.get("device_from_topic", -1)]
    ts = d.get(src.get("ts_field", "ts"))
    if isinstance(ts, str):
        ts = iso_to_ms(ts)
    elif isinstance(ts, (int, float)) and ts < 1e12:  # 초 단위면 ms 로
        ts = int(ts * 1000)
    msg_id = str(d.get(src["id_field"])) if src.get("id_field") else f"{dev}:{ts}"
    return [dict(location=loc, spot=None, device=dev, ts_sensor=ts, msg_id=msg_id, metrics=pick(d, src))]


PARSERS = {"chirpstack": parse_chirpstack, "xflow": parse_xflow, "json_flat": parse_json_flat}


class Collector:
    def __init__(self, cfg):
        self.cfg = cfg
        db_path = os.path.join(BASE, cfg["db_path"])
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.db = sqlite3.connect(db_path, isolation_level=None, check_same_thread=False)
        self.db.executescript(SCHEMA)
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.ranges = cfg.get("valid_range", {})
        self.stats = {"msgs": 0, "rows": 0, "dups": 0, "unparsed": 0}
        self.last_report = time.time()

        m = cfg["mqtt"]
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=m["client_id"],
                                  clean_session=False)
        self.client.username_pw_set(os.environ["MQTT_USER"], os.environ["MQTT_PASS"])
        self.client.reconnect_delay_set(m.get("reconnect_min_s", 1), m.get("reconnect_max_s", 30))
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.client.on_message = self.on_message

    def on_connect(self, client, userdata, flags, rc, props=None):
        log.info("connected rc=%s session_present=%s", rc, flags.session_present)
        for src in self.cfg["sources"]:
            client.subscribe(src["topic"], qos=1)
            log.info("subscribed %s (%s)", src["topic"], src["parser"])

    def on_disconnect(self, client, userdata, flags, rc, props=None):
        log.warning("disconnected rc=%s — reconnecting automatically", rc)

    def on_message(self, client, userdata, msg):
        now = int(time.time() * 1000)
        self.stats["msgs"] += 1
        payload = msg.payload.decode("utf-8", "replace")
        try:
            if self.cfg.get("keep_raw"):
                self.db.execute("INSERT INTO raw_messages VALUES (?,?,?)", (now, msg.topic, payload))
            src = next((s for s in self.cfg["sources"] if mqtt_match(s["topic"], msg.topic)), None)
            if src is None:
                raise ValueError("no source matches topic")
            rows = PARSERS[src["parser"]](json.loads(payload), src, msg.topic)
            self.store(now, src["name"], rows)
        except Exception as e:  # 한 메시지가 깨져도 수집은 계속한다
            self.stats["unparsed"] += 1
            self.db.execute("INSERT INTO unparsed VALUES (?,?,?)", (now, msg.topic, str(e)[:300]))
        if time.time() - self.last_report > 600:
            log.info("last 10 min: %s", self.stats)
            self.stats = {k: 0 for k in self.stats}
            self.last_report = time.time()

    def store(self, now, source, rows):
        for r in rows:
            for metric, value in r["metrics"]:
                lo, hi = self.ranges.get(metric, (float("-inf"), float("inf")))
                cur = self.db.execute(
                    "INSERT OR IGNORE INTO readings VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (now, r["ts_sensor"], source, r["location"], r["spot"], r["device"], metric,
                     value, int(not lo <= value <= hi), r["msg_id"]))
                if cur.rowcount:
                    self.stats["rows"] += 1
                else:
                    self.stats["dups"] += 1

    def run(self):
        m = self.cfg["mqtt"]
        # Jetson에는 시계 배터리가 없어 부팅 직후에는 1970년 근처에서 시작한다. 시계가 맞춰지기 전에 받은
        # 메시지는 수신 시각이 엉터리가 되므로(실제로 재부팅 뒤 269행), 시계가 맞을 때까지 접속을 미룬다.
        # 브로커가 세션을 기억하고 있어(QoS 1) 기다리는 동안의 메시지는 접속 뒤에 받는다.
        while time.time() < CLOCK_SANE:
            log.warning("system clock not set yet (%s) — waiting for time sync", time.strftime("%F %T"))
            time.sleep(5)
        while True:  # 첫 접속이 실패해도(네트워크 미준비 등) 계속 시도한다
            try:
                self.client.connect(os.environ["MQTT_HOST"], int(os.environ.get("MQTT_PORT", 1883)),
                                    m.get("keepalive", 60))
                break
            except OSError as e:
                log.warning("connect failed: %s — retry in 10s", e)
                time.sleep(10)
        self.client.loop_forever(retry_first_connection=True)


def main():
    with open(os.path.join(BASE, "config.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    c = Collector(cfg)
    signal.signal(signal.SIGTERM, lambda *a: (c.client.disconnect(), sys.exit(0)))
    c.run()


if __name__ == "__main__":
    main()
