# 004 탐지: 방법

```
탐지 동글 (모니터 모드) ─▶ pi/detect.py ─▶ 기기별 통계 ─▶ core/score.py ─▶ 의심 기기 표
                                                              │
                                              core/identify.py (걷기 반응, 필요할 때)
ESP32 ─▶ 찌르기 → ACK → CSI ─▶ 001 방향 찾기
```

- 점수의 증거와 가중치는 [ai 스펙 001](https://github.com/DeteCam/ai/tree/main/specs/001-suspicion-score)에서 정하고 다듬는다. 설계 노트는 [ai/docs/의심 점수](https://github.com/DeteCam/ai/blob/main/docs/의심%20점수.md).
- ESP32 는 라즈베리파이와 전선 4가닥(전원 2, 직렬 통신 2)으로 잇는다. 파이가 "이 MAC 을 이 채널에서 찔러라"를 보내고, ESP32 는 응답마다 CSI 한 줄을 돌려준다.
- ESP32-C5 는 2.4·5GHz 를 둘 다 듣는다. 같은 크기의 C3 로 바꾸면 2.4GHz 만 된다.

## 무선 부품이 넷인 까닭

| 부품 | 일 | 하나로 못 줄이는 까닭 |
|---|---|---|
| 파이 내장 Wi-Fi | 영상 프레임의 CSI | CSI 를 재는 동안 한 채널에 묶인다 |
| ESP32 | ACK 의 CSI, 찌르기 | 내장 Wi-Fi 는 ACK 에서 CSI 를 내지 않는다 |
| 탐지 동글 | 채널을 돌며 엿듣기 | 위 둘은 한 채널에 묶여 있다 |
| 통신 동글 | 폰과 잇기 | 위 셋은 듣는 중이다 |
