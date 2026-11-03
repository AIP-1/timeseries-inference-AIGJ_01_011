"""메시지 형식이 바뀌어도 코드를 고치지 않고 설정(config)만 바꿔 대응되는지 확인한다.

실행:  .venv/bin/python -m pytest -q test_format_change.py
브로커에는 아무것도 보내지 않고, 수집기의 해석·저장 부분만 임시 DB로 시험한다.
"""
import json

import pytest

import collector as C

CHIRP_TOPIC = "application/app/device/24e1240000000001/event/up"


def chirp(obj, tags=None, dedup="m1"):
    return {"deduplicationId": dedup, "time": "2026-10-20T01:00:00+00:00",
            "deviceInfo": {"deviceName": "NEW-CO2-0001", "devEui": "24e1240000000001",
                           "tags": tags or {"location": "회의실", "spot": "벽"}},
            "fCnt": 1, "object": obj}


def make(sources, tmp_path):
    cfg = {"db_path": str(tmp_path / "t.db"), "keep_raw": False, "sources": sources,
           "mqtt": {"client_id": "test"}, "valid_range": {"co2": [300, 10000]}}
    c = C.Collector.__new__(C.Collector)  # MQTT 접속 없이 해석·저장만 쓴다
    c.cfg, c.ranges = cfg, cfg["valid_range"]
    c.stats = {"msgs": 0, "rows": 0, "dups": 0, "unparsed": 0}
    import sqlite3
    c.db = sqlite3.connect(cfg["db_path"], isolation_level=None)
    c.db.executescript(C.SCHEMA)
    return c


def feed(c, topic, payload):
    class M:  # paho 메시지 흉내
        pass
    m = M(); m.topic, m.payload = topic, json.dumps(payload).encode()
    c.last_report = float("inf")
    c.on_message(None, None, m)
    return c.db.execute("SELECT location, metric, value, flag FROM readings ORDER BY metric").fetchall()


LORA = dict(name="lorawan", topic="application/+/device/+/event/up", parser="chirpstack",
            location_tag="location", spot_tag="spot", metrics=["co2", "temperature", "humidity"], rename={})


def test_case1_new_field_names_need_only_rename(tmp_path):
    """상황 1: 새 센서 모델이 co2 대신 carbon_dioxide, temperature 대신 temp 로 보낸다."""
    msg = chirp({"carbon_dioxide": 812, "temp": 23.4, "humidity": 55})
    (tmp_path / "a").mkdir(); (tmp_path / "b").mkdir()
    before = feed(make([LORA], tmp_path / "a"), CHIRP_TOPIC, msg)
    assert before == [("회의실", "humidity", 55.0, 0)]  # 이름이 달라 co2·온도를 놓친다

    fixed = dict(LORA, rename={"carbon_dioxide": "co2", "temp": "temperature"})  # 설정만 추가
    after = feed(make([fixed], tmp_path / "b"), CHIRP_TOPIC, msg)
    assert after == [("회의실", "co2", 812.0, 0), ("회의실", "humidity", 55.0, 0),
                     ("회의실", "temperature", 23.4, 0)]


def test_case2_new_vendor_topic_and_flat_json(tmp_path):
    """상황 2: 다른 회사 센서가 sensors/<공간>/<장치>/env 토픽으로 펼친 JSON을 보낸다."""
    new_src = dict(name="new_vendor", topic="sensors/+/+/env", parser="json_flat",
                   location_from_topic=1, device_from_topic=2, ts_field="timestamp",
                   metrics=["co2", "temperature", "humidity"],
                   rename={"carbon_dioxide": "co2", "temp": "temperature", "rh": "humidity"})
    c = make([LORA, new_src], tmp_path)
    rows = feed(c, "sensors/라운지/LG-01/env",
                {"timestamp": 1792461600, "carbon_dioxide": 640, "temp": 22.1, "rh": 48, "rssi": -70})
    assert rows == [("라운지", "co2", 640.0, 0), ("라운지", "humidity", 48.0, 0),
                    ("라운지", "temperature", 22.1, 0)]
    ts = c.db.execute("SELECT DISTINCT ts_sensor FROM readings").fetchone()[0]
    assert ts == 1792461600 * 1000  # 초 단위 시각을 ms 로 맞춰 저장


def test_duplicate_and_broken_messages(tmp_path):
    """재전송된 같은 메시지는 한 번만, 깨진 값은 저장하되 표시, 깨진 JSON은 수집을 멈추지 않는다."""
    c = make([LORA], tmp_path)
    feed(c, CHIRP_TOPIC, chirp({"co2": 0}, dedup="d1"))
    feed(c, CHIRP_TOPIC, chirp({"co2": 0}, dedup="d1"))  # 재전송
    rows = c.db.execute("SELECT value, flag FROM readings").fetchall()
    assert rows == [(0.0, 1)]  # 0ppm 은 범위 밖으로 표시
    assert c.stats["dups"] == 1

    class M: topic, payload = CHIRP_TOPIC, b"{not json"
    c.on_message(None, None, M())
    assert c.db.execute("SELECT COUNT(*) FROM unparsed").fetchone()[0] == 1
