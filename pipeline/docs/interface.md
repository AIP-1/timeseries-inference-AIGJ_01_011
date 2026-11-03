# team4 시계열 추론 서비스 — 결과 형식 (v1.1, 2026-11-03 실데이터 운영 기준으로 고침)

교육장 공간마다 **매분** 아래 세 가지를 계산해 MQTT로 발행하고, HTTP API로 조회할 수 있게 한다.

1. 60분 예측(CO₂·온도, 5분 간격)과 "몇 분 뒤 1,000ppm 초과" 경보
2. 이상 여부·이상 점수·원인 센서, 규칙 경보(데이터 끊김·값 고정·튄 값·범위 밖·새벽 영점 틀어짐)
3. 사용 여부, 환기 선제 경보(30분 안에 1,000ppm 초과)

| 항목 | 값 |
|---|---|
| 주소 | Jetson `team4` — Tailscale `<JETSON_IP>`, 같은 네트워크면 Jetson IP |
| 갱신 주기 | 1분 (계산 기준은 마지막으로 다 찬 5분 칸) |
| 시각 | KST, `YYYY-MM-DD HH:MM:SS`, 수신 시각 기준(일부 게이트웨이 시계가 약 5분 앞서 있어 센서 시각은 쓰지 않음) |
| 단위 | CO₂ ppm, 온도 ℃, 습도 %, 시간 분 |
| 공간 | `사무실`, `실습실`, `회의실` (`GET /spaces`, 지금 운영 모드 `real`). 합성 데이터 재생 모드(`replay`)로 바꾸면 `강의실A`, `강의실B`, `라운지`, `회의실` |
| 운영 모드 | `real`: 수집기 DB의 실데이터, 실데이터로 학습한 모델(`*_real`). 2026-10-20부터 |

## 1. MQTT

브로커 `<Jetson 주소>:1883` (익명 구독 가능)

| 토픽 | retain | 내용 |
|---|---|---|
| `team4/inference/state/<공간>` | ✅ | 공간 하나의 최신 결과(아래 '공간 결과'). 새로 구독해도 최신값을 바로 받는다 |
| `team4/inference/summary` | ✅ | 이번 주기 요약(계산 시각, 처리 시간) |
| `team4/inference/alerts` | ❌ | 경보가 **켜지거나 꺼질 때만** 발행. 같은 경보가 계속되면 30분마다 한 번 다시 발행 |

```bash
mosquitto_sub -h <JETSON_IP> -t 'team4/inference/#' -v
```

## 2. HTTP API

`http://<Jetson 주소>:8000` — 문서 화면: `/docs`

| 요청 | 설명 |
|---|---|
| `GET /health` | 서비스 상태(가동 시작, 성공·실패 주기 수, 마지막 처리 시간) |
| `GET /spaces` | 공간 목록 |
| `GET /latest` | 모든 공간의 최신 결과 + 요약 |
| `GET /latest/{공간}` | 공간 하나의 최신 결과 |
| `GET /forecast/{공간}?minutes=30` | "회의실 30분 뒤 CO₂는?" — 5분 단위로 반올림, 5~60분 |
| `GET /alerts?limit=50` | 최근 경보 기록 |

```bash
curl -s "http://<JETSON_IP>:8000/forecast/회의실?minutes=30"
# {"space":"회의실","minutes":30,"time":"2026-11-03 09:45:00","co2":742,"temperature":15.3,"data_time":"2026-11-03 09:20:00"}
# time = 30분 뒤가 끝인 5분 칸의 시작 시각 (data_time + 25분 ~ + 30분 칸)
```

## 3. 공간 결과 (`/latest/{공간}`, `state/<공간>` 토픽)

| 필드 | 형식 | 설명 |
|---|---|---|
| `space` | 문자열 | 공간 이름 |
| `status` | `ok` / `degraded` / `error` | `degraded`: 데이터가 끊기거나 부족해 모델 계산을 건너뛰고 규칙 경보만 제공. `note`에 이유 |
| `data_time` | 시각 | 이 결과가 기준으로 삼은 데이터 시각(마지막으로 다 찬 5분 칸의 끝) |
| `last_observation`, `minutes_since_data` | 시각, 분 | 마지막으로 값이 들어온 시각과 지금까지 지난 시간 |
| `current.co2` / `temperature` / `humidity` | 수 | 기준 시각의 5분 평균(정제 후) |
| `forecast.times[12]` | 시각 목록 | 예측한 5분 칸의 시작 시각. `times[0]` = `data_time` (`data_time`~+5분 칸 = 5분 뒤), `times[11]` = +55분 (60분 뒤 칸) |
| `forecast.co2[12]`, `forecast.temperature[12]` | 수 목록 | 각 시각의 예측값 |
| `forecast.co2_exceeds_1000_in_min` | 분 또는 `null` | 예측상 처음 1,000ppm 이상이 되는 시점. 없으면 `null` |
| `anomaly.score`, `threshold`, `is_anomaly` | 수, 수, 참/거짓 | 오토인코더 복원 오차와 경보 기준 |
| `anomaly.cause_sensor`, `per_sensor` | 문자열, {센서: 수} | 오차가 가장 큰 센서(원인 후보)와 센서별 오차 |
| `rules.data_gap` | 참/거짓 | 설정한 시간 넘게 데이터 없음. `real` 모드 15분(회의실 센서가 10분 간격), `replay` 모드 5분 |
| `rules.stuck` / `spike` / `out_of_range` / `night_drift` | 센서 목록 | 최근 10분 안에 해당 규칙에 걸린 센서 |
| `occupied.value`, `probability` | 참/거짓, 0~1 | 센서값으로 추정한 사용 여부(일정은 쓰지 않음) |
| `ventilation_alert.value`, `probability` | 참/거짓, 0~1 | 30분 안에 1,000ppm을 넘을 것 같은지. 이미 넘었으면 `note`에 표시 |
| `self_check.mae30_24h`, `mae60_24h` | ppm | 서비스가 최근 24시간 동안 내놓은 30·60분 예측을 실제 값과 비교한 평균 오차(스스로 채점) |
| `self_check.n30`, `n60` | 수 | 채점한 예측 개수 |
| `self_check.degraded` | 참/거짓 | 60분 오차가 평소(`expected_mae60`)의 1.5배를 넘음 → 예측을 믿지 말 것 |

예시 (2026-11-03 09:20 실제 응답, 예측 목록 줄임):

```json
{
  "space": "회의실", "status": "ok", "data_time": "2026-11-03 09:20:00",
  "last_observation": "2026-11-03 09:17:00", "minutes_since_data": 4.4,
  "current": {"co2": 575.0, "temperature": 14.2, "humidity": 57.5},
  "rules": {"data_gap": false, "spike": [], "stuck": [], "night_drift": [], "out_of_range": []},
  "forecast": {"model": "gru", "step_min": 5, "times": ["2026-11-03 09:20:00", "...", "2026-11-03 10:15:00"],
               "co2": [578, 586, 595, "...", 742], "temperature": [14.21, "...", 15.34], "co2_exceeds_1000_in_min": null},
  "anomaly": {"score": 0.0001, "threshold": 0.0575, "is_anomaly": false, "cause_sensor": "co2",
              "per_sensor": {"co2": 0.0002, "temperature": 0.0001, "humidity": 0.0001}},
  "occupied": {"probability": 0.0, "value": false, "threshold": 0.525},
  "ventilation_alert": {"probability": 0.31, "value": true, "threshold": 0.12},
  "self_check": {"mae30_24h": 49.1, "n30": 288, "mae60_24h": 91.9, "n60": 283, "expected_mae60": 42.8, "degraded": true}
}
```

## 4. 경보 (`alerts` 토픽, `GET /alerts`)

```json
{"space": "회의실", "type": "ventilation_soon", "state": "on", "detail": "30분 안에 1,000ppm 초과 예상 (p=0.31)", "data_time": "2026-11-03 09:20:00"}
```

| type | 뜻 | 받을 사람(제안) |
|---|---|---|
| `data_gap` | 설정한 시간(`real` 15분, `replay` 5분) 넘게 데이터 없음 | 시설·운영 |
| `stuck`, `spike`, `out_of_range` | 센서 값 고정·튄 값·범위 밖 | 시설(센서 점검) |
| `night_drift` | 새벽 CO₂가 외기보다 높게 유지 → 영점 틀어짐 의심 | 시설(아침에 모아서) |
| `anomaly` | 오토인코더 이상(냉방·환기 정지 등 의심) | 해당 공간 담당 + 시설 |
| `ventilation_soon`, `co2_forecast_1000` | 곧 환기 필요 | 강사·공간 사용자 |
| `forecast_degraded` | 최근 24시간 예측 오차가 평소의 1.5배 초과(센서 이상·공간 사용 방식 변화 의심) | 서비스 운영자(재학습 검토) |

`state`가 `off`인 메시지는 그 경보가 정상으로 돌아왔다는 뜻이다.

## 5. 다른 시스템이 쓸 때 주의

- `status`가 `ok`가 아니면 `forecast`·`anomaly`·`occupied`가 없을 수 있다. 필드가 없을 때를 처리할 것.
- 예측은 5분 평균 기준이다. 1분 원시값과 몇 ppm 차이가 날 수 있다.
- 로컬 LLM 에이전트가 도구로 쓸 때는 `/forecast/{공간}?minutes=` 와 `/latest/{공간}` 두 개면 충분하다.
  응답에 `data_time`이 있으니 "몇 분 전 기준 값인지"를 함께 답하게 할 것.
