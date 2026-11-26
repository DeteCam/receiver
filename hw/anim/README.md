# 조립 순서 그림을 그리는 코드

`hw/조립_시뮬레이션.mp4` · `.gif`와 [docs 구매 부품 목록](https://github.com/DeteCam/docs/blob/main/구매_부품_목록.md)의 부품 그림을 만든 코드다. **설계값으로 그린 그림이며 실물 사진이나 측정이 아니다.**

| 파일 | 내용 |
|---|---|
| `r3d.py` | 작은 3D 그리기 도구(numpy 깊이 버퍼, 선 그리기, 궤도 카메라) |
| `model.py` | 부품 모양과 자리. 치수는 [조립 순서](../조립_순서.md) 1절과 같다. `LAYOUT`은 케이블 묶음 자리(`k2i` = 고리 안쪽 뒤, 지금 배선) |
| `make_video.py` | 장면 8개를 그려 장면마다 mp4로 만든 뒤 잇는다 |
| `make_parts.py` | 부품 그림 19장 |

```bash
pip install numpy pillow          # ffmpeg과 Noto Sans CJK 글꼴이 필요하다
python3 make_video.py test        # 장면마다 한 장씩 out/test_*.png
python3 make_video.py all         # out/assembly.mp4 (1280 × 720, 30 fps, 약 39초)
python3 make_parts.py out/parts   # 부품 그림
ffmpeg -i out/assembly.mp4 -vf "fps=8,scale=560:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse" out/assembly.gif
```
