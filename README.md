# DeteCam receiver — 수신기

방 모서리에 두는 손바닥 크기 수신기입니다. 지름 약 10 cm 원 위에 세운 막대 안테나 8개를 전자 스위치로 번갈아 연결하고, 고정 기준 안테나와 함께 Intel AX210 한 장으로 받아 숨은 Wi-Fi 카메라 신호의 방향을 잽니다. 움직이는 부품이 없어 조용하고, 쇼핑몰 부품을 볼트와 SMA 커넥터로 조립합니다.

> 상태(2026-11-24): **명세 · 검증 전.** 실측한 성능 숫자는 없습니다. 작업 기준은 [docs/AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md)입니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| [`specs/`](specs/README.md) | 스펙. 지금은 [`002-switch-ring-receiver`](specs/002-switch-ring-receiver/spec.md) |
| [`sim/`](sim/README.md) | 시뮬레이션 코드: v6 독립 검증(광선 추적 · 안테나 전자기 해석), 회전 수신기 계산. 스위치 고리 계산을 이어서 추가 |

조립 · 펌웨어 · 3D 모델은 스위치 고리 계산과 첫 실측(V2) 뒤에 추가합니다.

## 이전 설계

v6(84 mm 칩 안테나 PCB 수신기 2기)의 CAD · 출력 파일 · 펌웨어 · 시뮬레이터는 태그 [`v6-nano2`](https://github.com/DeteCam/receiver/tree/v6-nano2)에 있습니다. 독립 검증에서 성능이 크게 떨어져 내렸고, 그 문서의 숫자(1 m 이내 96.4% 등)는 쓰지 않습니다. 이유는 [docs/기술검증_보고서.md](https://github.com/DeteCam/docs/blob/main/기술검증_보고서.md)에 있습니다.

부품과 구매 순서는 [docs/구매_부품_목록.md](https://github.com/DeteCam/docs/blob/main/구매_부품_목록.md)를 봅니다.
