# TASKS-002 (`SPEC-002+`)

## V0 문서 · 소스 코드 · 사례 확인 (완료 — 2026-11-25)
- [x] RF 스위치 데이터시트 비교: `PE42482` 단일 SP8T vs `HMC7992` SP4T × 3 (손실 · 격리 · All-Off 50 Ω · 3.3 V 제어 · 6 GHz 위상 비대칭 · 위조 함정)
- [x] 듀얼밴드 SMA-Male 막대 안테나 사양 및 58 mm 단일대역 함정 확인 (`TE/Linx ANT-DB1-RAF-SMA` 등, 9개 국내 일괄 재고는 구매 직전 재확인)
- [x] CSI 소프트웨어 및 ARM64 호스트 조사: `FeitCSI` 4대 버그 소스 코드 확인 → `VesperCSI` + `#21` 패치 + Linux 6.12 LTS(RPi CM4/5 ARM64 검증) 채택
- [x] AX210 두 단자 위상 안정성 문헌 조사: 상태 내 0.40° 정밀도 + 패킷당 $\pi$-반전 · RX0↔RX1 스왑 · 독립 AGC 위험 확인 → 수동 RF 부호 앵커 설계 및 V2 5대 관문 확정
- [x] R9 무선 연결 충돌 조사: `FeitCSI-iwlwifi` `cfg80211` 충돌 + 10 cm RF 포화 + USB 3.0 잡음 확인 → Burst-TDM + UART BLE/내부 UART ESP32 확정

## V1 계산 (`sim/`, AGENTS.md 5장)
- [x] `sim/plan_20261125.md`: 6종 공간 × 2변형 + 최악 금속방(13장면), 수신기 · 카메라 자리, 개발/평가 분리, 사다리 표 사전 등록
- [x] `sim/em/ring_nec2.py`: `SPEC-002`, `SPEC-002+(8+1)`, `SPEC-002++(12+1)`, 흡수형 vs 개방 반사형 NEC2 3D 패턴 계산
- [x] `sim/v1_ring/`: 13장면 Sionna RT 2.2 생성 및 `run_eval.py` · `RESULTS_20261125.md` 완료 (사전 등록 평가 + L0~L5 사다리 + 대조군 + 사후 후보 `D85/D100+Conj+6Wall`)

## V2~V4 실측
- [ ] V2 (3.5~5만 원): 집 PC + AX210 + 광대역 SMA 분배기 + 6/10 dB 감쇠기 5대 관문 (2.4 & 5 GHz)
- [ ] V3 (1기 완성): RPi 5 팬리스 + M.2 E-key HAT + `PE42482` + 듀얼밴드 다이폴 9개, 가드밴드 · 보정표 · 빈 방 3 m 실측
- [ ] V4: 교실(40명+금속 칠판) · 반사 많은 방에서 DiffLoc 재현 장비와 교대 측정 (20곳 이상, 실패율 포함)
