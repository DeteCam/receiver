# 조립 영상 · 그림을 그리는 코드

`hw/조립_시뮬레이션.mp4` · `.gif`, `hw/img/`의 그림, [docs 구매 부품 목록](https://github.com/DeteCam/docs/blob/main/구매_부품_목록.md)의 부품 그림을 만든 코드다. **설계값으로 그린 그림이며 실물 사진이나 측정이 아니다.**

| 파일 | 내용 |
|---|---|
| `make_video.py` | 조립 영상(2026-11-27 외형). 출력 부품은 `hw/cad/parts.py`에서 그대로 가져오고, 전자 부품 · 나사 · 선은 단순한 모양으로 그린다. 단계 43개를 단계마다 mp4로 만든 뒤 잇는다. VTK로 그리고 PIL로 한글 자막을 넣는다 |
| `make_stills.py` | 조립된 모습 그림(`hw/img/수신기_외형_*.png`, 투명 배경판 포함) |
| `make_wiring.py` | 배선 그림(`hw/img/배선도.png`) |
| `model.py` | 머리(고리 · 날개 · 기둥 · 레일 · 상자 · 축 자리)의 기준 치수. `hw/cad/check.py`가 출력 부품을 이 값과 견준다 |
| `r3d.py`, `make_parts.py` | 작은 3D 그리기 도구와 부품 그림 19장(구매 목록용) |

```bash
pip install vtk pillow cadquery==2.8 trimesh matplotlib    # ffmpeg, xvfb, Noto Sans CJK 글꼴이 필요하다
xvfb-run -a -s "-screen 0 1280x720x24" python3 make_video.py test   # 단계마다 마지막 장면 한 장씩 out/test_*.png
xvfb-run -a -s "-screen 0 1280x720x24" python3 make_video.py all    # out/assembly.mp4 (1280 × 720, 30 fps)
xvfb-run -a -s "-screen 0 1280x720x24" python3 make_video.py C13    # 한 단계만
xvfb-run -a -s "-screen 0 1600x1200x24" python3 make_stills.py
python3 make_wiring.py
python3 make_parts.py out/parts
```

이미 만든 단계 파일(`out/sNN_*.mp4`)이 있으면 `all`은 그 단계를 건너뛰고 잇기만 한다. 다시 그리려면 그 파일을 지운다.
