# DeteCam 6.0 (`Nano-2 AI`) — 초소형 포켓 2기 벽면 부착·PyTorch AI 수신기

> **v4.0(42 cm 대형 스탠드 3기 · 84만 원) → v5.0(12 cm 3기 · 46.7만 원) → v6.0 `Nano-2 AI` (9.6 cm 포켓 크기 단 2기 · 100% 신품 34.6만 원 · PyTorch AI + 가상 거울 앵커로 3기보다 더 정확!)**

![DeteCam 6.0 Nano-2 벽면 스위블](docs/images/nano2_wall_swivel.png)

## 1. 왜 `9.6 cm` 단 2기 + Physics-Informed AI 가 `12 cm` 3기보다 더 정확한가?
1. **5 GHz 그레이팅 로브 0% 물리 구조 (`84×84 mm` 4층 통합 RF PCB)**:
   - 테두리 12소자(`Abracon AANI-CH-0078`) 최소 간격 `20.1 mm < λ/2 (25.6 mm @ 5.85 GHz)` — 5 GHz 전 대역 공간 앨리어싱(허위 빔)을 물리적으로 완전 제거.
   - `PE42442×4` + `AD8317` + 온보드 `RP2040` QFN-56을 단일 4층 PCB에 통합하고, 후면 `8.35 mm` 공극 + `Ni-Cu` 차폐 시트로 콘크리트·젖은 타일·철제 사물함 벽면 간섭 차단.
2. **하드웨어 축소를 AI로 초월하는 4대 SW/AI 엔진 (`NanoPi R5C` RK3568B2 + 0.8T NPU 온디바이스 구동)**:
   - **`ReconCovNet-v4` (PyTorch, 457K params, `0.024 ms/frame` CPU)**: 다중 서브밴드 공분산 `[R_all, R_low, R_high]`에서 Differentiable-Capon 손실로 학습한 물리 잔차 신경망. **100% 미학습 4개 공간**에서 2.45 GHz P90 오차 `-16.8%` (`8.98°→7.47°`), 5.22 GHz P90 `-11.1%` (`5.52°→4.91°`) 개선!
   - **`MA-PINN` (Bayesian Log-Sum-Exp 3D 가상 거울 앵커 + `P_tx` 불변 `ΔP_12` 전력비 융합)**: 스마트폰 AR 스캔 3D 장애물 지도로 벽·거울·칠판·사물함의 1차 반사를 **공짜 가상 거울 앵커 6~12개**로 전환 → **단 2기**로 L자 코너·목욕탕 기둥·화장실 칸막이 사각지대 돌파!
   - **`CamGuard-Net` (PyTorch 1D-CNN + Self-Attention + 5초 `CamLoPA` 인과 자극, `0.12 ms`)**: 라이브 방송·클라우드 백업 하드 네거티브 800대 평가에서 수동 분석 `F1 75.64%` → 인과 융합 **`F1 98.88% · 재현율 99.00% · 오경보 1.25%`**!
   - **`AR-OPA-2` + `RAPS` Conformal AR**: 직선 가시선 + 1차 가상 거울 가시성을 동시 최적화해 2기 벽면 위치와 Pan(±30°)/Tilt(0°/-15°/-25°) 클릭 각도를 0.1초 추천, `RAPS`로 통계적 90% 신뢰 구역 보장.

## 2. 10대 실공간 3D 광선추적 + PyTorch AI 검증 결과 (`[시뮬레이션 · 미학습 공간 포함]`)

![10대 실공간 벤치마크](docs/images/nano2_10space_ai_benchmark.png)

| 검증 구간 (2.45 GHz 40 MHz / 5.22 GHz 160 MHz) | v5.0 3기 `Palm-12` (12 cm) | **v6.0 단 2기 `Nano-2 AI` (9.6 cm)** | **v6.0 2기 AI + 0원 폰보행** | 1기 `Nano-1 AI` (한계) |
|---|---:|---:|---:|---:|
| **전체 10공간 (5.22 GHz)** `Med / P90 / ≤1m` | 0.20 / 0.84 m / 90.7% | **0.19 / 0.54 m / 96.4%** | **0.19 / 0.50 m / 98.2%** | 0.71 / 3.06 m / 61.8% |
| **미학습 4공간 (5.22 GHz)** `Med / P90 / ≤1m` | 0.24 / 1.86 m / 84.8% | **0.21 / 0.64 m / 95.5%** | **0.21 / 0.62 m / 96.4%** | 0.86 / 4.59 m / 53.6% |
| **사각지대 NLoS (5.22 GHz)** `Med / P90 / ≤1m` | 0.21 / 1.90 m / 86.3% | **0.22 / 0.63 m / 95.1%** | **0.21 / 0.62 m / 97.1%** | 0.86 / 4.55 m / 56.9% |
| **전체 10공간 (2.45 GHz)** `Med / P90 / ≤1m` | 0.20 / 0.61 m / 94.3% | **0.24 / 0.87 m / 92.9%** | **0.23 / 0.65 m / 96.4%** | 0.79 / 2.79 m / 56.8% |
| **사각지대 NLoS (2.45 GHz)** `Med / P90 / ≤1m` | 0.22 / 1.34 m / 87.3% | **0.30 / 1.06 m / 87.3%** | **0.25 / 0.80 m / 96.1%** | 0.96 / 5.24 m / 51.0% |

## 3. 하드웨어 · 3D 출력 (`Bambu Studio 2.9.1` 실측) · 비용 (100% 신품)
- **외형 / 무게**: `96.0 × 96.0 × 36.0 mm` (무개조 `NanoPi R5C` `58×58 mm` + `AX210` 완전 수납), 노드당 `219.9 g` (**2기 전체 `440 g`, 부피 `664 cm³` — v5.0 대비 -48%!**)
- **기계 간섭 검증 (`cad/checks/check_nano2.py`)**: 9개 극한 각도(Pan `±30°` × Tilt `0°/-15°/-25°`) 부품 겹침 `0.0 mm³`, 벽면 충돌 `0.0 mm³`, 문제 `0개`.
- **Bambu Lab X1 Carbon 실측 슬라이싱 (`print/gcode/1판_포켓노드_1기전체.gcode.3mf`)**:
  - **노드 1기 = 단 1판 (`4시간 00분`, PLA `124.9 g`, 서포트 `0%`, 처짐/섬/외팔 `0`, `print_check.py` 문제 `0개`)**
  - **2기 전체 제작 = 단 2판 (`8시간 00분`, PLA `249.8 g`, v5.0 대비 출력 시간 `-45%`, 필라멘트 `-50%`)**
- **100% 신품 총 제작비 (중고 부품 0개!)**: **약 34.6만 원** (5V 보조배터리 2개 포함; v4.0 84만 원 대비 **-59%**, v5.0 46.7만 원 대비 **-26%**).

## 4. 저장소 구조 및 실행 방법
- [`cad/`](cad/README.md) — `96×96×36 mm` 포켓 CAD (`nano2_node.py`), STL 내보내기, 9자세 충돌 검사, Bambu Studio 슬라이싱
- [`stl/`](stl/) & [`print/`](print/) — 수밀성·오버행 `0.0` 검증 STL 6종 및 X1C 바로 출력용 `.gcode.3mf` / `.3mf`
- [`sim/`](sim/README.md) — 10대 실공간 3D 다중반사 광선추적 엔진(`raytrace_engine.py`), PyTorch 학습(`train_ai_models.py`), 10공간 종합 벤치마크(`benchmark_10spaces_ai.py`)
- [`firmware/`](firmware/README.md) — `Nano2AIEngine` (`ReconCovNet-v4` + `CamGuard-Net` 학습 가중치 포함), `TriF v13`, `AR-OPA-2`, `pytest` 6/6 통과
- [`docs/부품_목록.md`](docs/부품_목록.md) · [`docs/조립_및_출력_안내.md`](docs/조립_및_출력_안내.md) · [`docs/현장_최적배치_가이드.md`](docs/현장_최적배치_가이드.md)
- 조립/동작 3D 시뮬레이션 영상: [`docs/videos/조립_시뮬레이션.mp4`](docs/videos/조립_시뮬레이션.mp4), [`docs/videos/배치_동작_시뮬레이션.mp4`](docs/videos/배치_동작_시뮬레이션.mp4)

```bash
pytest -v firmware/tests/test_detecam_v6.py          # PyTorch AI + AR-OPA-2 + RAPS 6/6 검증
python3 cad/checks/check_nano2.py                    # 9개 극한 각도 CAD 충돌 0.0 mm³ 검증
python3 cad/slicing/print_check.py stl print         # Bambu X1C G-code 무결성 0 문제 검증
```
