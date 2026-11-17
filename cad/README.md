# cad: 3D 모델과 검사

모델은 파이썬 코드로 만듭니다 (manifold3d). 치수와 부품 모양은 모두 `model/panel.py` 한 파일에 있습니다. 부품을 받은 뒤 재서 고칠 값에는 `[실측]` 이라고 적어 두었습니다.

```
pip install manifold3d trimesh numpy shapely scipy pillow          # 그림과 영상까지: pyvista matplotlib, ffmpeg, xvfb
python3 export/export2.py                             # ../stl 다시 만들기 (부품 8개, 다시 읽어 검사)
python3 checks/panel_check.py                         # 겹침, 틈, 무게, 넘어지는 각, 자석
```

| 폴더 | 하는 일 | 스크립트 |
|---|---|---|
| `model/` | 치수와 부품 모양 | `panel.py` (패널 통·앞덮개·자석 걸이판·받침 발·누름쇠, 허브 통·선반·뚜껑, 사 오는 부품의 모형), `cadlib.py`, `paths.py` |
| `export/` | 내보내기 | `export2.py` (STL), `partlist.py` (`docs/부품표.md`·`.csv`), `marker_panel.py` (`marker/marker.json`) |
| `checks/` | 검사 | `panel_check.py` → `results/panel_check.json` |
| `figures/` | 그림과 영상 | `view_panel.py` (확인용 그림), `anim_panel.py` (`docs/videos/조립_시뮬레이션.mp4`), 그리기는 `render.py`. `xvfb-run -a` 로 돌립니다. `design_figs.py` 는 설계도와 데모 환경의 그림 일곱 장을 그립니다 (화면 없이) |
| `slicing/` | Bambu Studio 슬라이싱과 검사 | `slice_bambu.py ../stl ../print --parts` (5분), `print_check.py ../stl ../print` (1분) |
| `results/` | 위 스크립트들이 쓴 숫자 (json) | `parts.json`, `panel_check.json`, `gates.json` (관문마다 사는 것과 값) |

치수를 바꿨으면 `export/export2.py` → `checks/panel_check.py` → 슬라이싱 → `slicing/print_check.py` → `export/partlist.py` → `figures/anim_panel.py` 순서로 돌립니다.

**슬라이싱.** Bambu Studio 를 명령줄로 돌립니다. 화면이 있는 컴퓨터에서는 `BAMBU_STUDIO=/경로/bambu-studio` 를 주면 됩니다. 화면이 없는 리눅스에서는 `slicing/bs.sh` 가 가짜 화면을 띄워 돌립니다: `BAMBU_STUDIO_DIR` 에 AppImage 를 푼 폴더(`squashfs-root`)를 적고, `weston` 과 `libosmesa6` 을 깔고, `gcc -shared -fPIC -o slicing/shim/glshim.so slicing/shim/glshim.c` 를 한 번 해 둡니다.

**마커.** 마커 띠(`../marker/AR_마커.pdf`, 34 mm 마커 둘)는 앞덮개의 아래쪽 표시선 안에 붙입니다. 귀의 좌표는 `../marker/marker.json` 에 있습니다 (패널 좌표).

돌판 수신기의 모델과 검사(`parts2.py`, `checks/`, `analysis/`)는 태그 `plate-v2` 에 있습니다. 다만 `model/p2.py` 와 `model/params.py` (돌판의 치수)는 남겨 두었습니다. `sim/em` 과 `sim/rt/rt_eval.py`, `console/src/viewer_data3.py` 가 돌판의 치수를 여기서 읽기 때문입니다.
