# cad: 3D 모델과 검사

모델은 파이썬 코드로 만듭니다 (manifold3d). 치수는 `p2.py` 한 곳에, 파이와 모터의 치수는 `params.py` 에 있습니다. 결과 숫자는 `결과/` 에, 그림은 `../docs/그림/` 에 씁니다.

```
pip install manifold3d trimesh numpy shapely          # 그림까지: pyvista pillow matplotlib
python3 export2.py                                    # ../stl 다시 만들기
```

| 스크립트 | 하는 일 | 걸리는 시간 |
|---|---|---|
| `p2.py`, `params.py` | 치수 | |
| `parts2.py` | 부품 모양과 조립 | |
| `export2.py` | STL 29개를 내보내고 다시 읽어 검사 | 1분 |
| `check2.py`, `check_extra.py` | 전 범위에서 겹치는 곳 찾기 (나사 머리, 볼트, 동전 포함) | 3~4분 |
| `dongle_gap.py` | USB 동글 두 개의 틈 (서로, 요크와, 받침과) | 1분 |
| `audit_sweep.py` | 판을 돌리며 요크·받침과의 틈을 촘촘히 | 15분 |
| `audit_screws.py` | 나사 53개: 구멍, 물리는 길이, 드라이버가 들어갈 길 | 1분 |
| `audit_assembly.py` | 조립 순서대로 부품을 밀어 넣는 길 | 2분 |
| `audit_motion.py` | 기어 물림, 멈춤턱, 자세별 틈 | 1분 |
| `mass.py` | 무게, 균형추 동전 수, 끄덕임에 걸리는 힘 | |
| `strength.py` | 튼튼함 (손계산 44곳) | |
| `fem_rotor.py` | 회전 막대를 유한요소법으로 (`pip install pytetwild`, CalculiX `ccx` 가 있어야 합니다) | 10분 |
| `tipmodel.py`, `tipover.py` | 넘어짐: 무게중심 계산 | |
| `tip_mujoco.py` | 넘어짐: 물리 엔진으로 움직여 보기 (`pip install mujoco`) | 2분 |
| `room_check.py`, `cable_vis.py` | 부품 자리, 전선 길이, 겉으로 보이는 전선 | |
| `marker.py` | AR 마커와 틀 (`opencv-python`) | |
| `figs.py`, `fig_*.py`, `anim2.py` | 문서의 그림 | |
| `slicing/slice_bambu.py ../stl ../슬라이스 --parts` | Bambu Studio 로 슬라이싱 (아래 참고) | 15분 |
| `slicing/print_check.py ../stl ../슬라이스` | 슬라이싱한 파일을 다시 읽어 검사 | 10분 |

치수를 바꿨으면 `export2.py` → `check2.py` → `audit_screws.py` → `audit_assembly.py` → `mass.py` 순서로 돌려 보고, 괜찮으면 슬라이싱을 다시 합니다. 자세한 것은 [조립 안내](../docs/조립%20안내.md) 11절.

**슬라이싱.** Bambu Studio 를 명령줄로 돌립니다. 화면이 있는 컴퓨터에서는 `BAMBU_STUDIO=/경로/bambu-studio` 를 주면 됩니다. 화면이 없는 리눅스에서는 `slicing/bs.sh` 가 가짜 화면을 띄워 돌립니다: `BAMBU_STUDIO_DIR` 에 AppImage 를 푼 폴더(`squashfs-root`)를 적고, `weston` 과 `libosmesa6` 을 깔고, `gcc -shared -fPIC -o slicing/shim/glshim.so slicing/shim/glshim.c` 를 한 번 해 둡니다.

**마커.** `marker.py` 가 만드는 `../마커/marker.json` 은 `../firmware/pi/marker.json` 과 같은 파일이어야 합니다 (등판이나 마커 자리를 바꾸면 그쪽에도 복사합니다. `firmware/tests/test_marker.py` 가 확인합니다).

**유한요소 해석.** `fem_rotor.py` 는 그물과 풀이 파일을 `fem/` 폴더에 남깁니다 (100MB 넘게 나와서 저장소에는 넣지 않습니다). `fig_fem.py` 는 그 파일을 읽어 그립니다.
