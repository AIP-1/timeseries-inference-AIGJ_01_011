"""과제 1 확인: 브로커를 껐다 켜도 1분 안에 수집이 다시 시작되는가.

학원 브로커는 우리가 끌 수 없으므로, Jetson 안에 시험용 브로커(amqtt, 127.0.0.1:18883)를 따로 띄우고
운영 중인 수집기와 같은 코드(collector.Collector, 같은 재연결 설정)를 시험용 DB로 붙인다.
  1) 2초마다 ChirpStack 형식 메시지 발행 → 저장 확인
  2) 브로커 종료 20초 → 다시 시작
  3) 다시 시작한 뒤 첫 메시지가 DB에 저장될 때까지 걸린 시간
운영 중인 수집기·DB·브로커는 건드리지 않는다.
실행: .venv/bin/python test_broker_restart.py
"""
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time

import paho.mqtt.client as mqtt
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import collector as C

PORT = 18883
TOPIC = "application/test/device/24e1240000000099/event/up"


def broker(cfg_path):
    amqtt = os.path.join(os.path.dirname(sys.executable), "amqtt")
    return subprocess.Popen([amqtt, "-c", cfg_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    w = tempfile.mkdtemp(prefix="brk_")
    bcfg = os.path.join(w, "broker.yaml")
    yaml.safe_dump({"listeners": {"default": {"type": "tcp", "bind": f"127.0.0.1:{PORT}"}},
                    "plugins": {"amqtt.plugins.authentication.AnonymousAuthPlugin": {"allow_anonymous": True}}},
                   open(bcfg, "w"))
    cfg = yaml.safe_load(open(os.path.join(C.BASE, "config.yaml"), encoding="utf-8"))
    cfg["db_path"] = os.path.join(w, "test.db")
    cfg["mqtt"]["client_id"] = "team4-collector-restart-test"
    os.environ.update(MQTT_HOST="127.0.0.1", MQTT_PORT=str(PORT), MQTT_USER="test", MQTT_PASS="test")
    b = broker(bcfg); time.sleep(3)
    col = C.Collector(cfg)
    threading.Thread(target=col.run, daemon=True).start()
    pub = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="restart-test-pub")
    pub.reconnect_delay_set(1, 2)
    pub.connect_async("127.0.0.1", PORT); pub.loop_start()
    stop = threading.Event()
    sent = []

    def publish():
        k = 0
        while not stop.is_set():
            k += 1
            msg = {"deduplicationId": f"rt{k}", "time": time.strftime("%Y-%m-%dT%H:%M:%S+09:00"),
                   "deviceInfo": {"deviceName": "TEST-CO2", "devEui": "24e1240000000099",
                                  "tags": {"location": "시험실", "spot": "벽"}},
                   "fCnt": k, "object": {"co2": 600 + k, "temperature": 22.0, "humidity": 50}}
            if pub.is_connected():
                pub.publish(TOPIC, json.dumps(msg), qos=1)
                sent.append((time.time(), k))
            time.sleep(2)

    threading.Thread(target=publish, daemon=True).start()
    db = lambda: sqlite3.connect(cfg["db_path"])
    rows = lambda: db().execute("select count(*) from readings where device='TEST-CO2'").fetchone()[0]
    time.sleep(20)
    before = rows()
    t_kill = time.time(); b.terminate(); b.wait()
    time.sleep(20)
    t_up = time.time(); b = broker(bcfg)
    base = rows(); t_first = None
    while time.time() - t_up < 120:
        if rows() > base:
            t_first = time.time(); break
        time.sleep(0.5)
    time.sleep(10)
    after = rows()
    stop.set(); b.terminate()
    res = dict(stored_before_stop=before, broker_down_s=round(t_up - t_kill, 1),
               resume_s=round(t_first - t_up, 1) if t_first else None, stored_end=after, pass_=bool(t_first and t_first - t_up <= 60))
    print(json.dumps(res, ensure_ascii=False))


if __name__ == "__main__":
    main()
