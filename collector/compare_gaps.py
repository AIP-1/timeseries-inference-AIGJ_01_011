"""과제 1: 우리 수집기의 결측 구간을 학원 InfluxDB 기록과 대조해 원인을 나눈다.

장치마다 우리 수집기에서 '평소 보고 간격의 3배 또는 5분'보다 길게 빈 구간을 찾고, 같은 시간에 InfluxDB에는 그 장치 값이 있었는지 본다.
  - InfluxDB에도 없음 → 센서·게이트웨이 쪽 (우리 잘못 아님)
  - InfluxDB에는 있음 → 우리 쪽(Jetson·네트워크·수집기)
InfluxDB의 시각은 센서 시각이라 수신 시각과 수 분 어긋날 수 있어, 앞뒤 6분의 여유를 둔다.

  .venv/bin/python compare_gaps.py [시작 '2026-10-20 10:06'] [끝=지금]   → reports/gap_compare_*.md
InfluxDB 데이터는 export_influx.py 로 data/influx/ 에 받아 둔 것을 쓴다(먼저 이번 달을 다시 받을 것).
"""
import os
import sqlite3
import sys
import time

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TOL = pd.Timedelta(minutes=6)
start = pd.Timestamp(sys.argv[1] if len(sys.argv) > 1 else "2026-10-20 10:06")
end = pd.Timestamp(sys.argv[2]) if len(sys.argv) > 2 else pd.Timestamp.now().floor("min")

# 우리 수집기: 장치별 수신 시각(KST)
db = sqlite3.connect(f"file:{os.path.join(BASE, 'data', 'sensors.db')}?mode=ro", uri=True)
ms = lambda t: int(t.tz_localize("Asia/Seoul").value // 1_000_000)
ours = pd.read_sql_query("""SELECT DISTINCT msg_id, device, location, ts_recv FROM readings
                            WHERE source='lorawan' AND ts_recv BETWEEN ? AND ?""", db, params=(ms(start), ms(end)))
ours["t"] = pd.to_datetime(ours.ts_recv, unit="ms", utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
ours["key"] = ours.device.str[-6:]  # EM500-CO2-152590 ↔ devEui 24e124126d152590 (끝 6자리)

# InfluxDB: 장치별 기록 시각(UTC → KST), 1분 평균이라 1분에 한 줄
months = sorted({f"{d:%Y-%m}" for d in pd.date_range(start.normalize(), end, freq="D")})
inf = pd.concat([pd.read_parquet(os.path.join(BASE, "data", "influx", f"{m}.parquet")) for m in months])
inf["t"] = pd.to_datetime(inf.ts, utc=True).dt.tz_convert("Asia/Seoul").dt.tz_localize(None)
inf = inf[(inf.t >= start - TOL) & (inf.t <= end + TOL)]
inf["key"] = inf.device.astype(str).str[-6:]
inf_times = {k: g.t.drop_duplicates().sort_values().values for k, g in inf.groupby("key")}

rows, gaps, skipped = [], [], []
total_min = (end - start).total_seconds() / 60
for (key, dev, loc), g in ours.groupby(["key", "device", "location"]):
    # 비교 대상 아님: 문 열림처럼 일이 생길 때만 보내는 장치(메시지 10건 미만),
    # InfluxDB에서 내려받지 않은 항목만 보내는 장치(소음 센서 등)
    if len(g) < 10 or key not in inf_times:
        skipped.append(f"{loc} {dev} ({len(g)}건, " + ("이벤트형 장치" if len(g) < 10 else "CO2·온습도 없는 장치") + ")")
        continue
    t = g.t.sort_values().reset_index(drop=True)
    d = t.diff().dt.total_seconds() / 60
    med = d.median()
    limit = max(5, 3 * med)
    edges = [(start, t.iloc[0])] if (t.iloc[0] - start).total_seconds() / 60 > limit else []
    edges += [(t.iloc[i - 1], t.iloc[i]) for i in d.index[d > limit]]
    if (end - t.iloc[-1]).total_seconds() / 60 > limit:
        edges.append((t.iloc[-1], end))
    it = pd.Series(inf_times.get(key, []))
    n_inf = len(it[(it >= start) & (it <= end)])
    for a, b in edges:
        inside = it[(it > a + TOL) & (it < b - TOL)] if len(it) else []
        cause = "우리 쪽 (InfluxDB에는 있음)" if len(inside) else "센서·게이트웨이 쪽 (InfluxDB에도 없음)"
        gaps.append(dict(공간=loc, 장치=dev, 시작=f"{a:%m-%d %H:%M}", 끝=f"{b:%m-%d %H:%M}",
                         길이_분=round((b - a).total_seconds() / 60, 1), InfluxDB_값=len(inside), 원인=cause))
    expected = total_min / med if med > 0 else float("nan")
    rows.append(dict(공간=loc, 장치=dev, 간격_분=round(med, 1), 우리_수신=len(t),
                     기대_대비=f"{min(len(t) / expected, 1) * 100:.1f}%", InfluxDB_분=n_inf,
                     결측_구간=sum(1 for x in gaps if x["장치"] == dev)))

gp = pd.DataFrame(gaps)
md = [f"# 결측 구간 원인 대조 (수집기 vs 학원 InfluxDB)\n",
      f"- 구간: {start:%Y-%m-%d %H:%M} ~ {end:%Y-%m-%d %H:%M} KST ({total_min / 60:.1f}시간)",
      f"- 결측 구간 = 장치별로 평소 보고 간격의 3배 또는 5분보다 길게 빈 구간. InfluxDB 시각은 센서 시각이라 앞뒤 6분 여유\n",
      "## 장치별 수신\n", pd.DataFrame(rows).sort_values(["공간", "장치"]).to_markdown(index=False), "",
      f"비교에서 뺀 장치: {', '.join(skipped) or '없음'}\n",
      "## 결측 구간과 원인\n", gp.to_markdown(index=False) if len(gp) else "결측 구간 없음", ""]
if len(gp):
    s = gp.groupby("원인").agg(구간=("길이_분", "size"), 합계_분=("길이_분", "sum")).round(1)
    md += ["## 원인별 합계\n", s.to_markdown(), ""]
os.makedirs(os.path.join(BASE, "reports"), exist_ok=True)
path = os.path.join(BASE, "reports", time.strftime("gap_compare_%Y%m%d_%H%M.md"))
open(path, "w", encoding="utf-8").write("\n".join(md) + "\n")
print("\n".join(md))
print("→ saved", path)
