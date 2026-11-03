# team4 시계열 추론 — 과제 2~8 파이프라인

Jetson `~/timeseries/pipeline`. 제공 키트를 받지 못해 **합성 데이터 생성기를 과제 설명대로 직접 만들어** 먼저 전 과정을 완성했다.
실데이터는 수집기(`~/timeseries/collector.py`, 과제 1)가 계속 쌓고 있고, 충분히 쌓이면 입력만 바꿔 같은 코드를 다시 돌린다.

## 구성

| 경로 | 내용 |
|---|---|
| `synth/generate.py` | 합성 데이터 생성기: 4개 공간, 5주, 일정 기반 물리 모델, 결측·깨진 값·잡음, 마지막 주 고장 5종 |
| `tsi/data.py` | 공통 전처리: 정제 규칙, 5분 집계, 짧은 결측 보간, 시간·일정 특징, 기간 분할, 평가 시점 |
| `tsi/features.py` | 예측 모델 입력·정규화(학습 기간 값만). 학습과 서비스가 같은 코드 사용 |
| `tsi/anomaly.py`, `tsi/state.py` | 이상 탐지 규칙·오토인코더 입력, 사용 여부·환기 경보 특징 |
| `tsi/models.py` | GRU·LSTM·TCN 예측, 오토인코더, MLP 분류기 |
| `tasks/t2_baselines.py` ~ `t7_optimize.py` | 과제별 실험. 결과는 `results/t2` ~ `t7` 의 `.md`(보고서용 표) + `.json` |
| `service/` | 과제 8: 재생기(`replay.py`), 추론 서비스(`infer.py`), 설정, 로컬 MQTT 브로커 설정 |
| `docs/interface.md` | 결과 형식 문서(다른 팀용) |
| `artifacts/` | 학습된 모델(`.pt`), 정규화 값(`.json`), 변환된 모델(`.onnx`) |

## 다시 돌리기

```bash
cd ~/timeseries/pipeline
./run_all.sh                 # 합성 데이터 생성 → 과제 2, 3, 4, 5
./run_all.sh t6              # 사전학습 모델 비교(처음에 모델을 내려받음)
../.venv/bin/python tasks/t7_optimize.py   # ONNX 변환·지연 측정 (서비스가 쓰는 .onnx 생성)
```

## 상시 서비스 (과제 8)

```bash
systemctl --user status ts-broker ts-replay ts-infer
curl -s localhost:8000/latest/강의실A
journalctl --user -u ts-infer -f
```

- `ts-replay`: 합성 데이터 검증·시험 기간을 실시간처럼 재생 → `data/replay.db` (수집기 대신)
- `ts-infer`: 매분 계산 → MQTT `team4/inference/...` 발행 + API `:8000`
- `ts-broker`: 결과 발행용 로컬 MQTT 브로커 `:1883`
- 주기별 기록: `results/service_cycles.jsonl` (12시간 가동 기록 요약에 사용)

## 실데이터로 바꿔 끼우기

1. 공간 이름 대응과 일정 표 준비: `data/real/schedule.csv` (`space,start,end,kind,planned_people`)
2. 실험: `../.venv/bin/python tasks/t3_forecast.py --real-db ../data/sensors.db` (각 과제 스크립트 공통 옵션)
   - 실데이터에는 고장 정답이 없으므로 과제 4는 검증 기간 헛경보와 사람이 확인한 사례로 평가
3. 서비스: `service/config.yaml` 의 `mode: real`, `db: ../data/sensors.db`, `space_map` 설정 후 `systemctl --user restart ts-infer`

## 측정 조건

모든 결과표 첫 줄에 데이터 종류(합성/실데이터), 학습·검증·시험 기간, 간격, 장비가 자동으로 적힌다.
