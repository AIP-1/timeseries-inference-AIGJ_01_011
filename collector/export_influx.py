"""InfluxDB(iot-bucket)의 과거 CO2·온도·습도를 1분 평균으로 내려받아 data/influx/<YYYY-MM>.parquet 로 저장한다(읽기 전용).

  INFLUX_URL=http://localhost:8086 .venv/bin/python export_influx.py [시작=2025-09-01] [끝=오늘]
이미 받은 달은 건너뛴다(이번 달은 항상 다시 받음). 열: ts(UTC), location, spot, device, metric, value, n
"""
import os
import sys
import time

import pandas as pd
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

BASE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE, ".env"))
OUT = os.path.join(BASE, "data", "influx")
METRICS = ["co2", "temperature", "humidity"]

start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2025-09-01", tz="UTC")
end = pd.Timestamp(sys.argv[2], tz="UTC") if len(sys.argv) > 2 else pd.Timestamp.now(tz="UTC").ceil("D")
os.makedirs(OUT, exist_ok=True)
c = InfluxDBClient(url=os.environ["INFLUX_URL"], token=os.environ["INFLUX_TOKEN"], timeout=900_000)
org = c.buckets_api().find_buckets(name="iot-bucket").buckets[0].org_id
qa = c.query_api()
this_month = pd.Timestamp.now(tz="UTC").strftime("%Y-%m")

for m0 in pd.date_range(start.normalize().replace(day=1), end, freq="MS"):
    m1 = m0 + pd.offsets.MonthBegin(1)
    path = os.path.join(OUT, f"{m0:%Y-%m}.parquet")
    if os.path.exists(path) and f"{m0:%Y-%m}" != this_month:
        print(f"{m0:%Y-%m} skip", flush=True)
        continue
    t = time.time()
    flt = " or ".join(f'r._measurement=="{x}"' for x in METRICS)
    q = f"""import "types"
from(bucket:"iot-bucket") |> range(start: {m0:%Y-%m-%dT%H:%M:%SZ}, stop: {min(m1, end):%Y-%m-%dT%H:%M:%SZ})
 |> filter(fn:(r)=> {flt})
 |> filter(fn:(r)=> r._field == "value" and (types.isType(v: r._value, type: "float") or types.isType(v: r._value, type: "int")))
 |> map(fn:(r)=> ({{r with _value: float(v: r._value)}}))
 |> group(columns:["_measurement","location","spot","device_id"])
 |> aggregateWindow(every: 1m, fn: mean, createEmpty: false)
 |> keep(columns:["_time","_value","_measurement","location","spot","device_id"])"""
    frames = qa.query_data_frame(q, org=org)
    df = pd.concat(frames) if isinstance(frames, list) else frames
    if df is None or df.empty:
        print(f"{m0:%Y-%m} empty", flush=True)
        continue
    df = df.rename(columns={"_time": "ts", "_value": "value", "_measurement": "metric", "device_id": "device"})
    df = df.reindex(columns=["ts", "location", "spot", "device", "metric", "value"])  # 오래된 데이터에는 spot 태그가 없다
    # aggregateWindow 는 구간 끝 시각을 찍는다 → 1분 앞당겨 '그 1분의 시작' 시각으로 맞춘다
    df["ts"] = pd.to_datetime(df.ts, utc=True) - pd.Timedelta(minutes=1)
    df.to_parquet(path, index=False)
    print(f"{m0:%Y-%m} {len(df):,} rows {time.time() - t:.0f}s", flush=True)
print("done")
