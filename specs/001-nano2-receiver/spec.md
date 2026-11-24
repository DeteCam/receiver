# SPEC-001: DeteCam 6.0 (`Nano-2 AI`) 초소형 포켓 2기 벽면 수신기 사양

- **시스템 구성**: `Nano-2` 포켓 벽면 스위블 노드 **단 2기** (`96.0 × 96.0 × 36.0 mm`, `219.9 g/기`, 총 `440 g`) + 스마트폰 AR 앱
- **RF 안테나 배열**: `84×84 mm` 4층 FR-4 PCB, 테두리 12소자 + 기준 1소자(`Abracon AANI-CH-0078`), `d_min = 20.1 mm < λ/2 (5.85 GHz)` — 5 GHz 전 대역 그레이팅 로브 `0%`, `PE42442×4`, `AD8317`, 온보드 `RP2040` QFN-56, 후면 `8.35 mm` Ni-Cu 차폐 공극
- **컴퓨팅 / Wi-Fi CSI**: 무개조 `FriendlyElec NanoPi R5C` (4GB/32GB eMMC, RK3568B2 + 0.8T NPU, 58×58 mm, 38 g, USB-C 5V) + 네이티브 M.2 `Intel AX210NGW` (2.45 GHz 40 MHz / 5.22 GHz 160 MHz)
- **온디바이스 PyTorch AI**: `ReconCovNet-v4` (`0.024 ms`) + `CamGuard-Net` (`0.12 ms`, F1 `98.88%`) + Bayesian `MA-PINN` 가상 거울 앵커 + `ΔP_12` 전력비 융합 + `AR-OPA-2` + `RAPS` 90% AR
- **3D 프린팅 (Bambu X1C 실측)**: `1판_포켓노드_1기전체.gcode.3mf` = `4시간 00분 / 124.9 g`, 서포트 `0%`, `print_check.py` 문제 `0개`, 9자세 기계 충돌 `0.0 mm³`
- **신품 BOM 총액 (2기 완성)**: `346,000원` (보조배터리 2개 포함, 중고 0%)
