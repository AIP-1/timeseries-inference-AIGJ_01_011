# 과제 7. 엣지 최적화

> 측정 조건: 데이터 합성(자체 생성기) · 학습 2026-08-31 ~ 2026-09-20 · 검증 2026-09-21 ~ 2026-09-27 · 시험 2026-09-28 ~ 2026-10-04 · 간격 5min · 과거 120분 → 미래 60분 · team4 (aarch64) · 2026-10-20 14:29

**측정 조건**: 전력 모드 `NV Power Mode: 25W`, 클럭 고정(jetson_clocks) 안 함, 워밍업 20회 후 200회 반복, 입력은 시험 기간 실제 창. 전력은 같은 호출을 3초 동안 반복하며 tegrastats VDD_IN 평균(보드 전체). 측정 끝 시점 메모리 6904/7485MB 사용(VS Code 원격 서버 등 다른 프로세스 28개가 함께 떠 있음) — PyTorch CPU 지연의 p95가 큰 것은 이 경쟁의 영향일 수 있다.

사용 가능한 실행 방식: PyTorch GPU 가능, ONNX Runtime AzureExecutionProvider, CPUExecutionProvider, TensorRT(trtexec) 없음 — TensorRT 패키지 미설치

## 지연·전력·메모리

| 모델        | 방식               |   배치 |   평균_ms |   p95_ms |   power_mW |   gpu_pct |   ram_MB |
|:----------|:-----------------|-----:|--------:|---------:|-----------:|----------:|---------:|
| 예측(GRU)   | PyTorch CPU      |    1 | 121.889 |  162.854 |       6782 |         0 |     6829 |
| 예측(GRU)   | PyTorch GPU      |    1 |   1.4   |    1.789 |       6780 |        30 |     6849 |
| 예측(GRU)   | ONNX Runtime CPU |    1 |   0.319 |    0.387 |       7677 |         0 |     6847 |
| 예측(GRU)   | PyTorch CPU      |    4 |  85.529 |  117.713 |       6802 |         0 |     6820 |
| 예측(GRU)   | PyTorch GPU      |    4 |   1.186 |    1.271 |       6807 |        32 |     6833 |
| 예측(GRU)   | ONNX Runtime CPU |    4 |   1.433 |    5.035 |       7705 |         0 |     6834 |
| 이상탐지(AE)  | PyTorch CPU      |    1 |   4.856 |    9.064 |       6804 |         0 |     6647 |
| 이상탐지(AE)  | PyTorch GPU      |    1 |   0.65  |    0.675 |       6472 |        12 |     6654 |
| 이상탐지(AE)  | ONNX Runtime CPU |    1 |   0.036 |    0.04  |       6596 |         0 |     6655 |
| 이상탐지(AE)  | PyTorch CPU      |    4 |   2.317 |    5.077 |       6747 |         0 |     6671 |
| 이상탐지(AE)  | PyTorch GPU      |    4 |   0.655 |    0.691 |       6655 |        13 |     6674 |
| 이상탐지(AE)  | ONNX Runtime CPU |    4 |   0.076 |    0.038 |       6633 |         0 |     6693 |
| 사용여부(MLP) | PyTorch CPU      |    1 |   3.427 |    6.744 |       6786 |         0 |     6708 |
| 사용여부(MLP) | PyTorch GPU      |    1 |   0.574 |    0.603 |       6689 |        11 |     6700 |
| 사용여부(MLP) | ONNX Runtime CPU |    1 |   0.033 |    0.031 |       6650 |         0 |     6702 |
| 사용여부(MLP) | PyTorch CPU      |    4 |   1.436 |    3.124 |       6789 |         0 |     6719 |
| 사용여부(MLP) | PyTorch GPU      |    4 |   0.571 |    0.593 |       6703 |        12 |     6729 |
| 사용여부(MLP) | ONNX Runtime CPU |    4 |   0.032 |    0.033 |       6656 |         0 |     6728 |
| 환기경보(MLP) | PyTorch CPU      |    1 |   4.126 |    8.165 |       6789 |         0 |     6716 |
| 환기경보(MLP) | PyTorch GPU      |    1 |   0.583 |    0.628 |       6820 |        11 |     6717 |
| 환기경보(MLP) | ONNX Runtime CPU |    1 |   0.123 |    0.048 |       7302 |         0 |     6614 |
| 환기경보(MLP) | PyTorch CPU      |    4 |   3.438 |    7.076 |       7224 |         0 |     6695 |
| 환기경보(MLP) | PyTorch GPU      |    4 |   0.594 |    0.643 |       7257 |        12 |     6781 |
| 환기경보(MLP) | ONNX Runtime CPU |    4 |   0.061 |    0.155 |       7252 |         0 |     6836 |

## 변환 전후 결과 차이 (원래 단위)

| 모델        | 비교                      | 지표                | 값        | 온도_최대차이   |
|:----------|:------------------------|:------------------|:---------|:----------|
| 예측(GRU)   | ONNX Runtime vs PyTorch | CO2 예측 최대 차이(ppm) | 0.0002   | 0.00000℃  |
| 이상탐지(AE)  | ONNX Runtime vs PyTorch | 복원 오차 최대 상대 차이    | 4.94e-07 | nan       |
| 사용여부(MLP) | ONNX Runtime vs PyTorch | 분류 결과가 바뀐 비율      | 0.000%   | nan       |
| 환기경보(MLP) | ONNX Runtime vs PyTorch | 분류 결과가 바뀐 비율      | 0.000%   | nan       |

## 결론

- 예측(GRU): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.319ms), PyTorch는 GPU가 CPU보다 87.1배 빠름. 네 공간을 한 번에(배치 4) 돌려도 1.433ms.
- 이상탐지(AE): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.036ms), PyTorch는 GPU가 CPU보다 7.5배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.076ms.
- 사용여부(MLP): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.033ms), PyTorch는 GPU가 CPU보다 6.0배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.032ms.
- 환기경보(MLP): 배치 1에서 가장 빠른 방식은 ONNX Runtime CPU(0.123ms), PyTorch는 GPU가 CPU보다 7.1배 빠름. 네 공간을 한 번에(배치 4) 돌려도 0.061ms.
- ONNX Runtime CPU가 PyTorch보다 빠른 이유: 작은 모델에서는 연산보다 파이썬·프레임워크 호출 비용이 대부분이고, ONNX Runtime은 그 비용이 작다. GPU는 데이터 복사·커널 실행 비용이 있어 이렇게 작은 입력에서는 이득이 작다.
- 그래도 변환하는 이유: 같은 보드에 임베딩 모델·로컬 LLM이 함께 돈다. 시계열 모델은 CPU 한쪽에서 ONNX Runtime으로 가볍게 돌리고 GPU는 LLM에 양보하거나, 공간이 수십 개로 늘면 배치로 묶어 GPU/TensorRT로 보내는 것이 맞다.
- 평균보다 느린 쪽 5%(p95)가 중요한 이유: 매분 도는 서비스는 가끔 느려지는 순간에 다음 주기와 겹치거나 경보가 늦어진다.
- 서비스(과제 8)는 변환 오차가 0에 가까운 ONNX Runtime CPU를 쓴다.

**TensorRT 측정 방법**: Jetson에서 `sudo apt install -y tensorrt` 로 TensorRT를 설치한 뒤 이 스크립트를 다시 실행하면 TensorRT FP16 행이 추가된다.
