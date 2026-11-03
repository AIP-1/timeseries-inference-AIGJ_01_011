# timeseries-inference — team4 (AIGJ_01_011)

AIP-1 「시계열 데이터 추론 실습」 과제 결과. 팀 Jetson `team4`(Orin Nano Super 8GB)에서 센서 수집 → 60분 예측 · 이상 탐지 · 공간 사용 여부/환기 경보 추정 → 매분 추론 서비스까지 운영했다.

## 제출물

| 제출물 | 위치 |
|---|---|
| 결과 보고서 (과제 1~8, 측정 조건 포함) | [pipeline/docs/final_report.md](pipeline/docs/final_report.md) |
| 결과 형식 문서 (MQTT 토픽 · HTTP API, v1.1) | [pipeline/docs/interface.md](pipeline/docs/interface.md) |
| 발표 자료 (5분) | [pipeline/docs/presentation_slides.md](pipeline/docs/presentation_slides.md) · 대본 [presentation_script.md](pipeline/docs/presentation_script.md) |
| 발표 슬라이드(웹) | https://claude.ai/artifact/5qii6ghg5qM6Ccdkw8Q9F2 — **비공개 링크**(작성자 계정에서만 열림). 같은 내용이 `presentation_slides.md`에 있다 |
| 상시 서비스 | 팀 Jetson에서 systemd 사용자 서비스로 가동: `collector`, `ts-broker`, `ts-replay`, `ts-infer`(API :8000) |

`interface.md`의 `<JETSON_IP>`는 팀 Jetson 주소다(공개 저장소라 실제 주소는 지움).

## 폴더

| 폴더 | 내용 |
|---|---|
| `collector/` | 과제 1 MQTT 수집기(`collector.py` + `config.yaml`), 장애 시험, 수집 현황 보고서 `reports/` |
| `pipeline/synth/` | 합성 데이터 생성기(제공 키트를 받지 못해 직접 만듦) |
| `pipeline/tsi/` | 공통 모듈(데이터 적재, 특징, 모델, 이상 탐지, 상태 추정) |
| `pipeline/tasks/` | 과제 2~8 실험 스크립트 (`t2_*` ~ `t8b_*`) |
| `pipeline/real/` | 실데이터용 시간표 추정 · 서비스 모델 학습 |
| `pipeline/service/` | 과제 8 상시 추론 서비스 (replay → broker → infer, systemd 유닛) |
| `pipeline/results/` | 과제별 결과표(.md/.json)와 그래프 |

실행 방법은 [pipeline/README.md](pipeline/README.md)를 본다.

## 공개 저장소에서 뺀 것

- 접속 정보 `.env`(MQTT 계정, InfluxDB 토큰) — 실행하려면 Jetson의 `~/timeseries/.env`에 `MQTT_USER`, `MQTT_PASS`, `INFLUX_URL`, `INFLUX_TOKEN`을 직접 넣는다
- 수집 DB(`sensors.db`), 실데이터 원본, 학습된 모델(`pipeline/artifacts/`), 로그 — 용량·데이터 보호
- 작업 대화 기록, 개인 메모

결과표에 "(합성)"이 붙은 것은 자체 생성기 데이터, 그 밖은 학원 InfluxDB 실데이터(2025-09 ~ 2026-10) 결과다.
