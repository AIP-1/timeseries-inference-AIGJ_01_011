# 과제 7. 엣지 최적화

> 측정 조건: 데이터 실데이터(InfluxDB, 추정 시간표) · 학습 2025-09-03 ~ 2026-08-31 · 검증 2026-09-01 ~ 2026-09-30 · 시험 2026-10-01 ~ 2026-10-20 · 간격 5min · 과거 120분 → 미래 60분 · team4 (aarch64) · 2026-10-20 16:36

**측정 조건**: 전력 모드 `NV Power Mode: MAXN_SUPER`, 클럭 고정(jetson_clocks) 안 함, 워밍업 20회 후 200회 반복, 입력은 시험 기간 실제 창. 전력은 같은 호출을 3초 동안 반복하며 tegrastats VDD_IN 평균(보드 전체). 측정 끝 시점 메모리 4735/7485MB 사용(VS Code 원격 서버 등 다른 프로세스 30개가 함께 떠 있음) — PyTorch CPU 지연의 p95가 큰 것은 이 경쟁의 영향일 수 있다.

사용 가능한 실행 방식: PyTorch GPU 가능, ONNX Runtime AzureExecutionProvider, CPUExecutionProvider, TensorRT(trtexec) 있음

## 지연·전력·메모리

| 모델        | 방식               |   배치 |   평균_ms |   p95_ms |   power_mW |   gpu_pct |   ram_MB |
|:----------|:-----------------|-----:|--------:|---------:|-----------:|----------:|---------:|
| 예측(GRU)   | PyTorch CPU      |    1 |  98.725 |  133.752 |       8109 |         0 |     5315 |
| 예측(GRU)   | PyTorch GPU      |    1 |   0.995 |    1.09  |       7540 |        39 |     5426 |
| 예측(GRU)   | ONNX Runtime CPU |    1 |   0.32  |    0.261 |       9246 |         0 |     5513 |
| 예측(GRU)   | TensorRT FP16    |    1 |   0.446 |    0.453 |        nan |       nan |      nan |
| 예측(GRU)   | PyTorch CPU      |    4 |  85.031 |  124.095 |       7876 |         4 |     6217 |
| 예측(GRU)   | PyTorch GPU      |    4 |   0.987 |    1.067 |       8376 |        35 |     6166 |
| 예측(GRU)   | ONNX Runtime CPU |    4 |   1.162 |    5.626 |       9695 |        24 |     6121 |
| 예측(GRU)   | TensorRT FP16    |    4 |   0.705 |    0.73  |        nan |       nan |      nan |
| 이상탐지(AE)  | PyTorch CPU      |    1 |   4.878 |    8.757 |       7788 |         0 |     4581 |
| 이상탐지(AE)  | PyTorch GPU      |    1 |   0.524 |    0.561 |       7566 |        15 |     4549 |
| 이상탐지(AE)  | ONNX Runtime CPU |    1 |   0.057 |    0.029 |       7523 |         0 |     4546 |
| 이상탐지(AE)  | TensorRT FP16    |    1 |   0.052 |    0.056 |        nan |       nan |      nan |
| 이상탐지(AE)  | PyTorch CPU      |    4 |   2.091 |    4.65  |       7755 |         0 |     4733 |
| 이상탐지(AE)  | PyTorch GPU      |    4 |   0.521 |    0.558 |       7517 |        17 |     4733 |
| 이상탐지(AE)  | ONNX Runtime CPU |    4 |   0.034 |    0.041 |       7544 |         0 |     4749 |
| 이상탐지(AE)  | TensorRT FP16    |    4 |   0.07  |    0.076 |        nan |       nan |      nan |
| 사용여부(MLP) | PyTorch CPU      |    1 |   2.823 |    5.13  |       7790 |         0 |     4703 |
| 사용여부(MLP) | PyTorch GPU      |    1 |   0.464 |    0.514 |       7571 |        14 |     4703 |
| 사용여부(MLP) | ONNX Runtime CPU |    1 |   0.045 |    0.025 |       7457 |         0 |     4707 |
| 사용여부(MLP) | TensorRT FP16    |    1 |   0.047 |    0.052 |        nan |       nan |      nan |
| 사용여부(MLP) | PyTorch CPU      |    4 |   1.048 |    1.935 |       7673 |         0 |     4691 |
| 사용여부(MLP) | PyTorch GPU      |    4 |   0.441 |    0.464 |       7523 |        15 |     4719 |
| 사용여부(MLP) | ONNX Runtime CPU |    4 |   0.035 |    0.03  |       7405 |         0 |     4750 |
| 사용여부(MLP) | TensorRT FP16    |    4 |   0.066 |    0.072 |        nan |       nan |      nan |
| 환기경보(MLP) | PyTorch CPU      |    1 |   3.127 |    7.135 |       7747 |         0 |     4742 |
| 환기경보(MLP) | PyTorch GPU      |    1 |   0.446 |    0.472 |       7548 |        14 |     4789 |
| 환기경보(MLP) | ONNX Runtime CPU |    1 |   0.035 |    0.027 |       7451 |         0 |     4778 |
| 환기경보(MLP) | TensorRT FP16    |    1 |   0.049 |    0.06  |        nan |       nan |      nan |
| 환기경보(MLP) | PyTorch CPU      |    4 |   1.025 |    1.917 |       7691 |         0 |     4696 |
| 환기경보(MLP) | PyTorch GPU      |    4 |   0.441 |    0.463 |       7485 |        15 |     4728 |
| 환기경보(MLP) | ONNX Runtime CPU |    4 |   0.025 |    0.025 |       7451 |         0 |     4723 |
| 환기경보(MLP) | TensorRT FP16    |    4 |   0.062 |    0.068 |        nan |       nan |      nan |

## 변환 전후 결과 차이 (원래 단위)

| 모델        | 비교                      | 지표                | 값        | 온도_최대차이   |
|:----------|:------------------------|:------------------|:---------|:----------|
| 예측(GRU)   | ONNX Runtime vs PyTorch | CO2 예측 최대 차이(ppm) | 0.0006   | 0.00000℃  |
| 이상탐지(AE)  | ONNX Runtime vs PyTorch | 복원 오차 최대 상대 차이    | 3.75e-07 | nan       |
| 사용여부(MLP) | ONNX Runtime vs PyTorch | 분류 결과가 바뀐 비율      | 0.000%   | nan       |
| 환기경보(MLP) | ONNX Runtime vs PyTorch | 분류 결과가 바뀐 비율      | 0.000%   | nan       |

## 결론

- 예측(GRU): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.320ms), PyTorch는 GPU가 CPU보다 99.2배 빠름. 네 공간을 한 번에(배치 4) 돌려도 1.162ms.
- 이상탐지(AE): 배치 1에서 가장 빠른 방식은 TensorRT FP16(0.052ms), PyTorch는 GPU가 CPU보다 9.3배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.070ms.
- 사용여부(MLP): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.045ms), PyTorch는 GPU가 CPU보다 6.1배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.035ms.
- 환기경보(MLP): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.035ms), PyTorch는 GPU가 CPU보다 7.0배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.025ms.
- ONNX Runtime CPU가 PyTorch보다 빠른 이유: 작은 모델에서는 연산보다 파이썬·프레임워크 호출 비용이 대부분이고, ONNX Runtime은 그 비용이 작다. GPU는 데이터 복사·커널 실행 비용이 있어 이렇게 작은 입력에서는 이득이 작다.
- 그래도 변환하는 이유: 같은 보드에 임베딩 모델·로컬 LLM이 함께 돈다. 시계열 모델은 CPU 한쪽에서 ONNX Runtime으로 가볍게 돌리고 GPU는 LLM에 양보하거나, 공간이 수십 개로 늘면 배치로 묶어 GPU/TensorRT로 보내는 것이 맞다.
- 평균보다 느린 쪽 5%(p95)가 중요한 이유: 매분 도는 서비스는 가끔 느려지는 순간에 다음 주기와 겹치거나 경보가 늦어진다.
- 서비스(과제 8)는 변환 오차가 0에 가까운 ONNX Runtime CPU를 쓴다.
