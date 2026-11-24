# DeteCam 6.0 (`Nano-2 AI`) 온디바이스 펌웨어 & PyTorch Edge-AI

`FriendlyElec NanoPi R5C` (RK3568B2 ARM64 + 0.8T NPU)에서 실시간 실행되는 경량 신호처리·PyTorch 추론 코어:

- `core/ai_engine.py`: 실제 학습된 `weights/recon_covnet.pt` (`ReconCovNet-v4`) & `weights/camguard_net.pt` (`CamGuard-Net`) 로드 + Bayesian Log-Sum-Exp `MA-PINN` 가상 거울 앵커 + `ΔP_12` 전력비 융합 (`0.024 ms / 0.12 ms`).
- `core/trif_v12.py`: `NANO12_EDGE_M` (84×84 mm, `d_min=20.1 mm < λ/2 @ 5.85 GHz`), `CM-RCLPA` 180° 위상 반전 복구, `C⁻¹` 상호결합 보정, `RAPS` 90% Conformal AR.
- `core/ar_opa.py`: `AR-OPA-2` 단 2기 벽면 부착·스위블 각도 최적 배치(직선 LoS + 1차 가상 거울 가시성 동시 최적화) + 3초 `Auto-Verify`.
- `tests/test_detecam_v6.py`: `pytest` 6/6 통과.
