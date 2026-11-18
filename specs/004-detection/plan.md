# 004 탐지: 방법

```
탐지 동글 (모니터 모드) ─▶ pi/detect.py ─▶ 기기별 통계 ─▶ core/score.py ─▶ 의심 기기 표
                                                              │
                                              core/identify.py (걷기 반응, 필요할 때)
패널의 ESP32-C5 ─▶ 찌르기 → ACK → CSI 세기 ─▶ 006 방향 재기
```

- 점수의 증거와 가중치는 [ai 스펙 001](https://github.com/DeteCam/ai/tree/main/specs/001-suspicion-score)에서 정하고 다듬는다. 설계 노트는 [ai/docs/의심 점수](https://github.com/DeteCam/ai/blob/main/docs/의심%20점수.md).
- 패널의 ESP32-C5 는 허브와 USB 또는 ESP-NOW 로 잇는다. 허브가 "이 MAC 을 이 채널에서 찔러라"를 보내고, ESP32 는 응답마다 세기 한 줄을 돌려준다.
- ESP32-C5 는 2.4·5GHz 를 둘 다 듣는다. 같은 크기의 C3 로 바꾸면 2.4GHz 만 된다.

## 무선 부품이 셋인 까닭

| 부품 | 일 | 하나로 못 줄이는 까닭 |
|---|---|---|
| 패널의 ESP32-C5 | 기기가 보낼 때와 ACK 의 CSI 세기 (방향), 찌르기 | 재는 동안 한 채널에 묶인다 |
| 탐지 동글 | 채널을 돌며 엿듣기 | 패널은 한 채널에 묶여 있다. 꼭 산다 (2026-11-18: 5 GHz 는 탐지만 하므로 탐지를 패널에 기대지 않는다) |
| 통신 동글 (또는 파이 내장 Wi-Fi) | 폰과 잇기 | 위 둘은 듣는 중이다 |

돌판 때는 파이 내장 Wi-Fi(nexmon)가 영상 프레임의 CSI 를 쟀다. 그 코드는 receiver-legacy 에 있다.
