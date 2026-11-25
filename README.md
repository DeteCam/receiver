# DeteCam receiver — 수신기

방 모서리에 두는 손바닥 크기 수신기(`SPEC-002+ Rev.2`, 약 9.9 × 9.9 × 18 cm)입니다. 지름 8.5 cm 원 위에 세운 정품 2.4/5 GHz 듀얼밴드 다이폴 8개(`dz=0`)를 단일칩 흡수형 `PE42482` SP8T 스위치로 번갈아 연결하고, 중심 기준 안테나 1개(9번째 가상 소자 겸용)와 함께 Intel AX210NGW 한 장으로 받아 숨은 Wi-Fi 카메라 신호의 방향을 잽니다. 팬리스 라즈베리파이 5(`VesperCSI` + Linux 6.12 LTS, 호스트 GPIO 직결)와 Burst-TDM 무선 제어로 무소음(R1) · 무PCB(R5) · 무선(R9) 조건을 지킵니다.

> 상태(2026-11-25): **V0 문서 검증 및 V1 6종 공간 시뮬레이션 완료, V2~V4 실측 전.** 실측한 성능 숫자는 없습니다. 작업 기준은 [docs/AGENTS.md](https://github.com/DeteCam/docs/blob/main/AGENTS.md)입니다.

## 폴더

| 폴더 | 내용 |
|---|---|
| [`specs/`](specs/README.md) | 스펙. 지금은 [`002-switch-ring-receiver`](specs/002-switch-ring-receiver/spec.md) (`SPEC-002+`) |
| [`sim/`](sim/README.md) | 시뮬레이션 코드: **V1 6종 공간 스위치 고리 검증(`plan_20261125.md`, `v1_ring/RESULTS_20261125.md`)**, NEC2 안테나 해석(`em/ring_nec2.py`), v6 독립 검증, 회전 수신기 계산 |

조립 · 펌웨어 · 3D 모델은 스위치 고리 계산과 첫 실측(V2) 뒤에 추가합니다.

## 이전 설계

v6(84 mm 칩 안테나 PCB 수신기 2기)의 CAD · 출력 파일 · 펌웨어 · 시뮬레이터는 브랜치 [`archive/v6-nano2`](https://github.com/DeteCam/receiver/tree/archive/v6-nano2)에 있습니다. 독립 검증에서 성능이 크게 떨어져 내렸고, 그 문서의 숫자(1 m 이내 96.4% 등)는 쓰지 않습니다. 이유는 [docs/기술검증_보고서.md](https://github.com/DeteCam/docs/blob/main/기술검증_보고서.md)에 있습니다.

부품과 구매 순서는 [docs/구매_부품_목록.md](https://github.com/DeteCam/docs/blob/main/구매_부품_목록.md)를 봅니다.
