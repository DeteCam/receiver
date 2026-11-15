# cad: 3D 모델과 검사

모델은 파이썬 코드로 만듭니다 (manifold3d). 고칠 치수는 거의 다 `model/p2.py` 한 곳에, 파이와 모터의 치수는 `model/params.py` 에 있습니다. 결과 숫자는 `results/` 에, 그림은 `../docs/images/` 에, STL 은 `../stl/` 에 씁니다. 파일 하나하나는 [폴더와 파일](../docs/폴더와%20파일.md).

```
pip install manifold3d trimesh numpy shapely          # 그림까지: pyvista pillow matplotlib
python3 export/export2.py                             # ../stl 다시 만들기
```

| 폴더 | 하는 일 | 스크립트 (걸리는 시간) |
|---|---|---|
| `model/` | 치수와 부품 모양 | `p2.py` (치수), `params.py`, `parts2.py` (부품과 조립), `parts.py` (사는 부품), `cadlib.py`, `paths.py` |
| `export/` | 내보내기 | `export2.py` (STL 31개를 내보내고 다시 읽어 검사, 1분), `export_assembly.py` (3D 보기 파일), `marker.py` (AR 마커, `opencv-python`) |
| `checks/` | 검사 | `check2.py`·`check_extra.py` (전 범위 겹침, 나사 머리·볼트·동전·크기가 다른 부품까지, 10분쯤), `audit_screws.py` (나사 55개, 1분), `audit_assembly.py` (조립 순서대로 밀어 넣기, 1분), `audit_motion.py` (기어, 멈춤턱), `audit_sweep.py` (판을 돌리며 촘촘히, 30분~1시간), `audit_holes.py` (나사 구멍 둘레 살, 몇 분), `plate_play.py` (판의 움직임, 1분), `dongle_gap.py`, `room_check.py`, `cable_vis.py` |
| `analysis/` | 해석 | `mass.py` (무게, 동전 수, 모터 힘), `strength.py` (튼튼함 손계산), `fem_rotor.py` (유한요소: `pip install pytetwild` 와 CalculiX `ccx`, 10분), `tipmodel.py`·`tipover.py` (넘어짐 계산), `tip_mujoco.py` (물리 엔진: `pip install mujoco`, 2분) |
| `figures/` | 문서의 그림 | `figs.py`, `fig_*.py`, `anim2.py` (그리기는 `render.py`, `xvfb-run -a` 로) |
| `slicing/` | Bambu Studio 슬라이싱과 검사 | `slice_bambu.py ../stl ../print --parts` (15분), `print_check.py ../stl ../print` (10분) |
| `results/` | 위 스크립트들이 쓴 숫자 (json) | |

스크립트는 어느 폴더에서 돌려도 됩니다 (예: `python3 checks/check2.py`). 서로의 모듈은 저마다 첫 줄에서 `cad` 의 하위 폴더를 모두 찾습니다.

치수를 바꿨으면 `export/export2.py` → `checks/check2.py` → `checks/audit_screws.py` → `checks/audit_assembly.py` → `analysis/mass.py` 순서로 돌려 보고, 괜찮으면 슬라이싱을 다시 합니다. 자세한 것은 [조립 안내](../docs/조립%20안내.md) 11절.

**슬라이싱.** Bambu Studio 를 명령줄로 돌립니다. 화면이 있는 컴퓨터에서는 `BAMBU_STUDIO=/경로/bambu-studio` 를 주면 됩니다. 화면이 없는 리눅스에서는 `slicing/bs.sh` 가 가짜 화면을 띄워 돌립니다: `BAMBU_STUDIO_DIR` 에 AppImage 를 푼 폴더(`squashfs-root`)를 적고, `weston` 과 `libosmesa6` 을 깔고, `gcc -shared -fPIC -o slicing/shim/glshim.so slicing/shim/glshim.c` 를 한 번 해 둡니다.

**마커.** `export/marker.py` 가 만드는 `../marker/marker.json` 은 `../firmware/pi/marker.json` 과 같은 파일이어야 합니다 (등판이나 마커 자리를 바꾸면 그쪽에도 복사합니다. `firmware/tests/test_marker.py` 가 확인합니다).

**유한요소 해석.** `analysis/fem_rotor.py` 는 그물과 풀이 파일을 `_tmp/fem/` 에 남깁니다 (100MB 넘게 나와서 저장소에는 넣지 않습니다). `figures/fig_fem.py` 는 그 파일을 읽어 그립니다.
