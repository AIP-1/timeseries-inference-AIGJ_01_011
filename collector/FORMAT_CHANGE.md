# 과제 1 산출물 — 메시지 형식이 바뀌는 상황과 대응 기록

수집기(`collector.py`)는 메시지 해석 규칙을 코드가 아니라 `config.yaml` 에 둔다.
아래 두 상황을 가정하고, **코드는 한 줄도 고치지 않고 설정만 바꿔** 대응되는지 `test_format_change.py` 로 확인했다(3건 모두 통과).

## 상황 1. 새 센서 모델이 항목 이름을 다르게 보낸다
- 가정: 회의실에 새로 단 CO₂ 센서가 `co2` 대신 `carbon_dioxide`, `temperature` 대신 `temp` 로 보낸다
- 바꾸기 전: 수집기는 이름이 달라 CO₂·온도를 버리고 습도만 저장한다(시험으로 확인)
- 대응: `config.yaml` 의 해당 source 에 이름 대응 한 줄 추가

```yaml
    rename: {carbon_dioxide: co2, temp: temperature}
```
- 결과: 같은 메시지에서 `co2=812`, `temperature=23.4`, `humidity=55` 가 표준 이름으로 저장됨

## 상황 2. 다른 회사 센서가 새 토픽·새 구조로 들어온다
- 가정: 라운지에 다른 회사 센서를 달았고, `sensors/<공간>/<장치>/env` 토픽에 값을 한 단계로 펼친 JSON을 보낸다
  (`{"timestamp": 1792461600, "carbon_dioxide": 640, "temp": 22.1, "rh": 48, "rssi": -70}`)
- 대응: 범용 해석기(`json_flat`)를 쓰는 source 를 설정에 추가

```yaml
  - name: new_vendor
    topic: sensors/+/+/env
    parser: json_flat
    location_from_topic: 1      # 토픽 두 번째 칸 = 공간
    device_from_topic: 2        # 세 번째 칸 = 장치
    ts_field: timestamp         # 초 단위여도 ms 로 맞춰 저장
    metrics: [co2, temperature, humidity]
    rename: {carbon_dioxide: co2, temp: temperature, rh: humidity}
```
- 결과: 공간 `라운지`, 장치 `LG-01` 로 CO₂·온도·습도가 저장되고, 저장하지 않기로 한 `rssi` 는 버려진다
- `systemctl --user restart collector` 한 번으로 반영(재시작해도 브로커가 세션을 기억해 끊긴 동안의 메시지 일부를 다시 받음)

## 함께 확인한 것
- 같은 메시지가 두 번 오면(재전송) `msg_id`(ChirpStack deduplicationId)로 한 번만 저장
- 0ppm 같은 깨진 값은 저장하되 `flag=1` 로 표시(지우지 않고 정제 단계에서 판단)
- JSON이 깨진 메시지가 와도 수집은 멈추지 않고 `unparsed` 표에 이유를 남김

## 실제로 겪은 형식 차이
- 장치마다 보내는 항목이 다르다: AM103(CO₂·온습도·배터리), EM300-TH(온습도만), WS302(소음), xflow 냉난방(중첩 JSON)
  → source 3개(`lorawan`, `hvac`, `hvac_controller`)와 해석기 2종으로 모두 설정에서 처리
- 소음 센서(WS302)는 가끔 측정값 없이 상태만 보낸다 → `unparsed` 에 "no object field" 로 기록, 데이터 손실 아님
- InfluxDB의 오래된 데이터에는 `spot` 태그가 없다 → 내려받을 때 빈칸으로 채움(`export_influx.py`)
