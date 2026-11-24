# DeteCam 5.0 (Palm-12) 펌웨어 핵심 엔진 (`firmware/`)

`Radxa ROCK 3A` (Linux 6.12 + `VesperCSI` ARM64, 또는 x86 `Radxa X4`)에서 동작하는 DeteCam 5.0 핵심 신호처리·판별·배치 알고리즘 및 자동 검증 테스트입니다.

| 파일 | 역할 |
|---|---|
| [`core/trif_v12.py`](core/trif_v12.py) | `Palm-12` 12소자 테두리 배열 `TriF v12` (이중 체인 비 + **CM-RCLPA** + **상호결합 역행렬 `C⁻¹`** + Capon + `CV_f` + 10면 가상 거울 앵커 + RAPS 90% AR) |
| [`core/score.py`](core/score.py) | 3단계 카메라 판별 (수동 무늬 + 5초 인과 자극 `CamLoPA` + 3D 방 경계 필터) + 3노드 `AD8317` LTE/5G 교차 차감 + 무료 BLE 구역 추정 |
| [`core/ar_opa.py`](core/ar_opa.py) | 손 닿는 벽면(`1.1 m / 2.0 m`) 부착 + 15° 클릭 스위블(`Pan ±30°`, `Tilt 0°/-15°/-25°`) `AR-OPA` 최적 배치 및 3초 `Auto-Verify` |
| [`tests/test_detecam_v5.py`](tests/test_detecam_v5.py) | 전체 핵심 알고리즘 자동 단위 검증 (`pytest`, 5/5 통과) |

## 테스트 실행

```bash
pytest -v firmware/tests/test_detecam_v5.py
```
