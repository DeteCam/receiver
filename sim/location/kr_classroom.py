"""한국 교실 규격으로 다시: 40명 교실에서 수신기를 어디에 두면 카메라 자리를 맞히나.   python3 kr_classroom.py [횟수]   결과: kr_classroom_result.json

교실 (근거와 가정)
  - 크기 7.5 (폭, 칠판 벽) x 9.0m (앞뒤). 1962~1992 표준설계도의 '가로 7.5m 세로 9m' (67.5m²). 요즘 교실도 60~63m² 로 비슷하다.
  - 천장 2.7m (가정. 3.0m 도 본다). 벽 콘크리트, 바닥 타일(데코타일), 천장 텍스(석고 판으로 본다).
  - 칠판: 1200 x 4800mm (학교용 대형 칠판의 흔한 크기), 아래끝 0.9m -> 위끝 2.1m, 앞벽 가운데. 철판(자석 칠판)으로 본다.
  - 창: 한쪽 긴 벽(x = 0)에 아래 0.9m ~ 위 2.4m 유리. 복도 쪽 긴 벽(x = 7.5)은 콘크리트에 앞문·뒷문(나무, 높이 2.1m)과 위쪽 복도 창(유리 1.8~2.4m).
  - 뒷벽: 사물함 (철제, 높이 1.2m, 폭 전체). 철제가 가장 나쁜 경우라 그렇게 둔다.
자리와 사람
  - 낱책상 650 x 450mm, 높이 0.74m. 6칸 x 7줄 = 42자리에서 마지막 줄 양끝 2자리를 비워 40명. 칸 간격 1.1m, 줄 간격 0.95m, 첫 줄은 칠판에서 2.0m.
  - 앉은 사람: 머리 꼭대기 1.32m = 의자 좌면 0.44m + 앉은키 0.88m (성인 남녀 평균 어림). 어깨너비 약 0.42m 라 반지름 0.2m 기둥.
수신기: 앞·뒤 모서리, 옆벽 가운데, 칠판 위, 뒤 사물함 위, 천장 가운데(매단 것), 교실 밖 복도. 자리 고르기와 전파 계산은 classroom.py, rx_places.py 와 같다.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
import classroom as C
import rx_places as RP                        # 복도 밖 수신기를 다루는 World (불러오면 C.ClassWorld 를 바꾼다)

HERE = os.path.dirname(os.path.abspath(__file__))
W, D = 7.5, 9.0


def setup(ceil=2.7):
    room.ROOMS["교실"] = dict(L=(W, D, ceil), wall="concrete", floor="tile", ceil="plaster", clutter=(-16.0, -8.0), absorb=(0.5, 0.9),
                            patches=[dict(wall="y0", rect=(1.35, 6.15, 0.9, 2.1), mat="metal"),          # 칠판 4.8 x 1.2m
                                     dict(wall="x0", rect=(0.5, 8.5, 0.9, 2.4), mat="glass6"),           # 바깥 창
                                     dict(wall="x1", rect=(0.4, 1.3, 0.0, 2.1), mat="wood40"),           # 앞문
                                     dict(wall="x1", rect=(7.7, 8.6, 0.0, 2.1), mat="wood40"),           # 뒷문
                                     dict(wall="x1", rect=(1.6, 7.4, 1.8, 2.4), mat="glass6"),           # 복도 창
                                     dict(wall="y1", rect=(0.0, W, 0.0, 1.2), mat="metal")],            # 뒷벽 사물함 (철제)
                            heights=((0.75, 1.0),))
    xs = 0.6 + np.arange(6) * 1.1 + 0.025                  # 6칸: 0.625 ~ 6.125 (양쪽 벽에서 0.6m 안팎)
    ys = 2.0 + np.arange(7) * 0.95                         # 7줄: 2.0 ~ 7.7m
    seats = [(x, y) for j, y in enumerate(ys) for i, x in enumerate(xs) if not (j == 6 and i in (0, 5))]
    C.COLS, C.ROWS = 6, 7
    C.SEATS = np.array(seats); C.SEAT_X = xs; C.SEAT_Y = ys
    C.BODY_H, C.BODY_R = 1.32, 0.20
    # 자리 번호 -> (줄, 칸): 마지막 줄은 두 자리가 빠졌으므로 좌표로 계산한다
    C.rc = lambda k: (int(np.argmin(np.abs(ys - C.SEATS[int(k)][1]))), int(np.argmin(np.abs(xs - C.SEATS[int(k)][0]))))


H = 2.4
PLACES = {
    "앞 모서리 (창 쪽)": [(0.5, 0.5, H)], "앞 모서리 (복도 쪽)": [(7.0, 0.5, H)],
    "뒤 모서리 (창 쪽, 사물함 위 벽)": [(0.5, 8.6, H)], "뒤 모서리 (복도 쪽)": [(7.0, 8.6, H)],
    "창 쪽 벽 가운데": [(0.3, 4.8, H)], "복도 쪽 벽 가운데": [(7.2, 4.8, H)],
    "칠판 위 가운데": [(3.75, 0.25, H)], "뒤 사물함 위 가운데 (1.3m)": [(3.75, 8.7, 1.3)],
    "천장 가운데 (매달아 2.5m)": [(3.75, 4.8, 2.5)],
    "복도 · 복도 창 앞 2.2m": [(8.3, 4.5, 2.2)],
    "뒤 두 모서리": [(0.5, 8.6, H), (7.0, 8.6, H)], "네 모서리": [(0.5, 0.5, H), (7.0, 0.5, H), (0.5, 8.6, H), (7.0, 8.6, H)],
}


def main(n=120):
    out = {}
    for ceil in (2.7, 3.0):
        setup(ceil)
        for name, pos in PLACES.items():
            if ceil != 2.7 and name not in ("네 모서리", "천장 가운데 (매달아 2.5m)", "뒤 두 모서리"):
                continue
            p = [(x, y, min(z, ceil - 0.2)) if "천장" in name else (x, y, z) for x, y, z in pos]
            if ceil == 3.0 and "천장" in name:
                p = [(x, y, 2.8) for x, y, z in pos]
            C.RX[name] = p
            for wh in ("책상 위", "가슴"):
                t0 = time.time(); s = C.summ(C.run(n, name, wh, True)); key = "천장 %.1fm | %s | %s" % (ceil, name, wh); out[key] = s
                last = max((k for k in s if k.isdigit()), key=int)
                print("%-44s %-4s %s대: 정확 %4.1f%% 옆자리까지 %4.1f%% 3후보 %4.1f%% | 방향 %s (%.0f초)" % (
                    "천장 %.1fm | %s" % (ceil, name), wh, last, s[last]["exact"], s[last]["near"], s[last]["top3"], s["dir_med"], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=out), open(os.path.join(HERE, "kr_classroom_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
