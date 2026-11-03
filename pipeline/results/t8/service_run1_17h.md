# 과제 8. 상시 추론 서비스 가동 기록

> 구간 2026-10-20 16:15 ~ 2026-10-21 09:15 (17.0시간), 모드 `real`(재생 모드는 합성 데이터를 실시간처럼 재생), 장비 Jetson team4, ONNX Runtime CPU

## 요약

| 항목 | 값 |
|---|---|
| 완료한 주기 | 1,021 / 기대 1,020 (100.1%) |
| 주기 처리 시간 | 평균 2444ms · p95 2681ms · 최대 4055ms |
| 2.5분 넘게 주기가 빈 횟수 | 0 |
| systemd | NRestarts=0, ActiveEnterTimestamp=Tue 2026-10-20 16:17:30 KST |
| 경보 발생(켜짐·꺼짐 포함) | 105건 |

## 공간별 상태 횟수

|     |   degraded |   ok |
|:----|-----------:|-----:|
| 사무실 |          0 | 1021 |
| 실습실 |          0 | 1021 |
| 회의실 |        521 |  500 |

## 경보 종류별 건수

| 0                 |   사무실 |   실습실 |   회의실 |
|:------------------|------:|------:|------:|
| co2_forecast_1000 |     5 |    10 |     0 |
| data_gap          |     0 |     0 |     4 |
| forecast_degraded |     4 |     4 |     0 |
| night_drift       |    11 |    21 |     0 |
| stuck             |     0 |     0 |    22 |
| ventilation_soon  |    10 |    10 |     4 |

## 장애 시험 (데이터 공급 10분 중단)

```
=== 장애 시험 시작 2026-10-20 14:58:05 — ts-replay 를 10분 멈춤
14:57:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
14:58:05 ts-replay stopped
14:58:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
14:59:06 강의실A:ok/gap=False/1.2m 강의실B:ok/gap=False/1.2m 라운지:ok/gap=False/1.2m 회의실:ok/gap=False/1.2m
15:00:06 강의실A:ok/gap=False/2.2m 강의실B:ok/gap=False/2.2m 라운지:ok/gap=False/2.2m 회의실:ok/gap=False/2.2m
15:01:06 강의실A:ok/gap=False/3.2m 강의실B:ok/gap=False/3.2m 라운지:ok/gap=False/3.2m 회의실:ok/gap=False/3.2m
15:02:06 강의실A:ok/gap=False/4.2m 강의실B:ok/gap=False/4.2m 라운지:ok/gap=False/4.2m 회의실:ok/gap=False/4.2m
15:03:06 강의실A:degraded/gap=True/5.2m 강의실B:degraded/gap=True/5.2m 라운지:degraded/gap=True/5.2m 회의실:degraded/gap=True/5.2m
15:04:06 강의실A:degraded/gap=True/6.2m 강의실B:degraded/gap=True/6.2m 라운지:degraded/gap=True/6.2m 회의실:degraded/gap=True/6.2m
15:05:06 강의실A:degraded/gap=True/7.2m 강의실B:degraded/gap=True/7.2m 라운지:degraded/gap=True/7.2m 회의실:degraded/gap=True/7.2m
15:06:06 강의실A:degraded/gap=True/8.2m 강의실B:degraded/gap=True/8.2m 라운지:degraded/gap=True/8.2m 회의실:degraded/gap=True/8.2m
15:08:06 강의실A:degraded/gap=True/10.2m 강의실B:degraded/gap=True/10.2m 라운지:degraded/gap=True/10.2m 회의실:degraded/gap=True/10.2m
15:08:06 ts-replay started
15:09:06 강의실A:degraded/gap=False/1.2m 강의실B:degraded/gap=False/0.2m 라운지:degraded/gap=False/0.2m 회의실:degraded/gap=False/0.2m
15:09:06 강의실A:degraded/gap=False/1.2m 강의실B:degraded/gap=False/0.2m 라운지:degraded/gap=False/0.2m 회의실:degraded/gap=False/0.2m
15:11:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
15:11:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
15:12:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
15:14:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
15:15:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
15:16:06 강의실A:ok/gap=False/0.2m 강의실B:ok/gap=False/0.2m 라운지:ok/gap=False/0.2m 회의실:ok/gap=False/0.2m
=== 이 시간대 경보 기록
강의실A data_gap on 5.2분째 데이터 없음 2026-09-21 00:05:00
강의실B data_gap on 5.2분째 데이터 없음 2026-09-21 00:05:00
라운지 data_gap on 5.2분째 데이터 없음 2026-09-21 00:05:00
회의실 data_gap on 5.2분째 데이터 없음 2026-09-21 00:05:00
강의실A data_gap off 정상으로 돌아옴 2026-09-21 00:10:00
강의실B data_gap off 정상으로 돌아옴 2026-09-21 00:10:00
라운지 data_gap off 정상으로 돌아옴 2026-09-21 00:10:00
회의실 data_gap off 정상으로 돌아옴 2026-09-21 00:10:00
=== 끝 2026-10-20 15:16:07
```

확인 기준: 중단 뒤 `data_gap` 경보(켜짐)가 나고, 다시 켜면 경보가 꺼지고 상태가 `ok`로 돌아와야 한다.
