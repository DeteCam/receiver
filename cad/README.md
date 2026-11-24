# DeteCam 6.0 (`Nano-2 AI`) CAD · STL · Bambu Studio 슬라이싱

- **모델 (`model/nano2_node.py`)**: `96.0 × 96.0 × 36.0 mm` 포켓 크기 벽면 부착·직교 2축 스위블 노드 (무개조 `FriendlyElec NanoPi R5C` + `Intel AX210NGW` + `84×84 mm` 4층 RF PCB + `8.35 mm` Ni-Cu 차폐 공극).
- **STL 출력 (`export/export_stl.py`)**: 6종 부품(노드당 7개) 수밀성 `True`, 덩어리 `1`, 처짐 `0.0/0.0 mm²`.
- **9자세 충돌 검증 (`checks/check_nano2.py`)**: Pan `±30°` × Tilt `0°/-15°/-25°` 9개 조합 부품·벽면 충돌 `0.0 mm³`, 노드 중량 `219.9 g`.
- **Bambu Studio 2.9.1 실측 슬라이싱 (`slicing/slice_bambu.py`, `print_check.py`)**:
  - `1판_포켓노드_1기전체.gcode.3mf`: **4시간 00분 · PLA 124.9 g · 서포트 0% · `print_check.py` 문제 0개** (2기 전체 2판 = 8시간 00분 / 249.8 g).
