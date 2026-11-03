"""과제 8. 상시 서비스 가동 기록 요약 — results/service_cycles.jsonl + 장애 시험 기록 → results/t8/service.md

사용법:  ../.venv/bin/python tasks/t8_summary.py [시간=12]
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from tsi import common as C

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 12
RES = os.path.join(C.ROOT, "results")


def main():
    rows = [json.loads(l) for l in open(os.path.join(RES, "service_cycles.jsonl"), encoding="utf-8")]
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df.computed_at.str[:19])
    end = df.t.max()
    df = df[df.t >= end - pd.Timedelta(hours=HOURS)].copy()
    span_h = (df.t.max() - df.t.min()).total_seconds() / 3600
    expected = int(span_h * 60) + 1
    gaps = df.t.diff().dt.total_seconds().div(60)
    long_gaps = df[gaps > 2.5][["t"]].assign(빈_시간_분=gaps[gaps > 2.5].round(1))
    st = pd.DataFrame(list(df.status)).apply(pd.Series.value_counts).fillna(0).astype(int).T
    alerts = [a for lst in df.alerts for a in lst]
    al = pd.Series(alerts).str.split(":", expand=True) if alerts else pd.DataFrame(columns=[0, 1])
    al_tab = al.groupby([0, 1]).size().unstack(fill_value=0) if len(al) else pd.DataFrame()

    unit = subprocess.run(["systemctl", "--user", "show", "ts-infer", "-p", "NRestarts", "-p", "ActiveEnterTimestamp"],
                          capture_output=True, text=True).stdout.strip().replace("\n", ", ")
    outage = open(os.path.join(RES, "outage_test.log"), encoding="utf-8").read() if os.path.exists(
        os.path.join(RES, "outage_test.log")) else "(장애 시험 기록 없음)"

    md = ["# 과제 8. 상시 추론 서비스 가동 기록\n",
          f"> 구간 {df.t.min():%Y-%m-%d %H:%M} ~ {df.t.max():%Y-%m-%d %H:%M} ({span_h:.1f}시간), "
          f"모드 `{df['mode'].iloc[-1]}`(재생 모드는 합성 데이터를 실시간처럼 재생), 장비 Jetson team4, ONNX Runtime CPU\n",
          "## 요약\n",
          "| 항목 | 값 |\n|---|---|",
          f"| 완료한 주기 | {len(df):,} / 기대 {expected:,} ({len(df) / expected * 100:.1f}%) |",
          f"| 주기 처리 시간 | 평균 {df.took_ms.mean():.0f}ms · p95 {df.took_ms.quantile(.95):.0f}ms · 최대 {df.took_ms.max():.0f}ms |",
          f"| 2.5분 넘게 주기가 빈 횟수 | {len(long_gaps)} |",
          f"| systemd | {unit} |",
          f"| 경보 발생(켜짐·꺼짐 포함) | {len(alerts)}건 |\n",
          "## 공간별 상태 횟수\n", st.to_markdown(), "",
          "## 경보 종류별 건수\n", al_tab.to_markdown() if len(al_tab) else "없음", ""]
    if len(long_gaps):
        md += ["## 주기가 빈 구간\n", long_gaps.to_markdown(index=False), ""]
    md += ["## 장애 시험 (데이터 공급 10분 중단)\n", "```", outage.strip(), "```", "",
           "확인 기준: 중단 뒤 `data_gap` 경보(켜짐)가 나고, 다시 켜면 경보가 꺼지고 상태가 `ok`로 돌아와야 한다."]
    os.makedirs(os.path.join(RES, "t8"), exist_ok=True)
    with open(os.path.join(RES, "t8", "service.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
