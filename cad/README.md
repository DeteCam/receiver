# DeteCam 5.0 (Palm-12) CAD · STL · Bambu Studio 2.9.1 파이프라인

- `model/palm_node.py`: `120 × 120 × 30.6 mm` 손바닥 크기 벽면 부착·직교 2축 스위블 노드 파라메트릭 CAD (`manifold3d`).
- `export/export_stl.py`: 6종(노드당 7개) 수밀·무처짐(`0.0 mm²`) STL 생성 → `stl/`.
- `checks/check_palm.py`: `Pan ∈ {-30°, 0°, +30°} × Tilt ∈ {0°, -15°, -25°}` 전 9개 자세 부품 간섭 및 벽면 충돌 `0.0 mm³` 검사 → `cad/results/mech_check.json`.
- `slicing/slice_bambu.py` & `slicing/print_check.py`: 공식 Bambu Studio 2.9.1 CLI로 **X1 Carbon 단 1판(`4시간 52분`, `165.0 g`, 서포트 0%)** 슬라이싱 및 층별 G코드 무결성 검사 → `print/`.
- `figures/render_figs.py`, `anim_assembly.py`, `anim_operation.py`: 3D 렌더, 벤치마크 차트, 조립·각도조절 영상 및 현장 동작 영상 생성 → `docs/images/`, `docs/videos/`.
