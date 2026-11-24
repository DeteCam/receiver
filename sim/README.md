# DeteCam 6.0 (`Nano-2 AI`) 3D 광선추적 시뮬레이션 & PyTorch AI 학습

- `raytrace_engine.py`: ITU-R P.2040 복소 유전율·Fresnel 반사, 3D 영상법 다중 반사, ITU-R P.526 회절 상한, 상호결합 행렬 `C`, AX210 15% 180° 위상 반전, 10대 다양한 실공간(L자 코너·목욕탕 기둥·금속 사물함·40석 교실·150석 강당·통유리 회의실·과밀 기숙사).
- `train_ai_models.py`: **PyTorch 실학습 + 미학습 공간(7~10번) 일반화 검증**:
  - `ReconCovNet-v4` (457K params, `0.024 ms`): 2.45 GHz P90 `8.98°→7.47° (-16.8%)`, 5.22 GHz P90 `5.52°→4.91° (-11.1%)`.
  - `CamGuard-Net` (24K params, `0.12 ms`): 800대 하드 네거티브 평가 수동 `F1 75.64%` → 5초 인과 융합 **`F1 98.88% · 오경보 1.25%`**.
- `benchmark_10spaces_ai.py`: 10대 공간 × 2개 대역(2.45/5.22 GHz) × 5가지 구성(`3N Palm-12` vs `2N Nano Classic` vs `2N Nano-2 AI` vs `1N Nano-1 AI` vs `2N AI+0원 폰`) 560회 종합 벤치마크 → 결과 [`nano2_10space_results.json`](nano2_10space_results.json).
