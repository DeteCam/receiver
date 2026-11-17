"""사각지대 그림.  python3 blind_fig.py   ->  docs/images/사각지대_자리지도.png

숫자는 blind.json (../rt/blind_study.py 가 낸 것)에서 읽는다.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE)); OUT = os.path.join(ROOT, "docs", "images")
for f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(f): fm.fontManager.addfont(f)
plt.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]; plt.rcParams["axes.unicode_minus"] = False
NAVY, TEAL, AMBER, RED, GRAY, INK, LINE = "#10304F", "#0E8F82", "#B36B00", "#E0483A", "#6B7780", "#0E1B26", "#C9D3DC"
CM = LinearSegmentedColormap.from_list("t", ["#F3F6F8", "#BFE3DE", "#5DB8AC", TEAL, "#0A5F57"])
B = json.load(open(os.path.join(HERE, "blind.json"), encoding="utf-8"))
XY = np.array(B["seats_xy"]); SP = np.array(B["alt"]["spots"]); W, L = 7.5, 9.0


def room(ax, vals, spots, title, sub):
    ax.add_patch(Rectangle((0, 0), W, L, fill=False, lw=1.2, ec=GRAY))
    ax.add_patch(Rectangle((1.95, -0.06), 3.6, 0.12, color=NAVY)); ax.text(3.75, -0.32, "칠판", ha="center", va="top", fontsize=8.5, color=NAVY)
    ax.text(0, L + 0.18, "창 쪽", ha="left", va="top", fontsize=9, color=GRAY); ax.text(W, L + 0.18, "복도 쪽", ha="right", va="top", fontsize=9, color=GRAY)
    for (x, y), v in zip(XY, vals):
        y = y + 0.125
        ax.add_patch(Rectangle((x - 0.45, y - 0.37), 0.9, 0.74, color=CM(v / 100.0), ec=RED if v < 40 else "white", lw=2.2 if v < 40 else 0.6))
        ax.text(x, y, "%d" % v, ha="center", va="center", fontsize=8.6, color="white" if v >= 62 else INK, fontweight="bold" if v < 40 else None)
    for i, s in enumerate(spots):
        x, y = SP[s][0], SP[s][1]
        if x < 1 or x > W - 1:                                                  # 옆벽: 벽에 붙인 막대, 글자는 벽 밖에
            ax.add_patch(Rectangle((x - 0.17, y - 0.30), 0.34, 0.60, color=AMBER, zorder=5)); lx, ly = (-0.5 if x < 1 else W + 0.5), y
        else:                                                                   # 앞벽 (칠판 윗변): 글자는 막대 밑에
            ax.add_patch(Rectangle((x - 0.30, y - 0.17), 0.60, 0.34, color=AMBER, zorder=5)); lx, ly = x, y + 0.62
        ax.text(lx, ly, "ABC"[i], ha="center", va="center", fontsize=10, color=AMBER, fontweight="bold", zorder=6)
    ax.set_xlim(-1.0, W + 1.0); ax.set_ylim(L + 0.75, -0.8); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=16); ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=8.6, color=GRAY, va="bottom")


now = B["now"]["K"]; alt = B["alt"]["combos"]
P = [(now["1"]["seat_exact"], (7, 17), "지금 거는 곳, 한 번 잴 때", "창 유리 가운데(A) + 복도 창턱 앞쪽(B). 평균 %.0f%%, 40%% 아래 %d자리" % (now["1"]["all"][0], now["1"]["below40"])),
     (now["4"]["seat_exact"], (7, 17), "같은 곳, 네 번 모았을 때", "평균 %.0f%%, 40%% 아래 %d자리" % (now["4"]["all"][0], now["4"]["below40"])),
     (alt["6+17"]["1"]["seat_exact"], (6, 17), "A 를 창 유리 앞쪽으로 옮기면, 한 번", "평균 %.0f%%, 40%% 아래 %d자리 (약한 곳이 뒤로 간다)" % (alt["6+17"]["1"]["all"][0], alt["6+17"]["1"]["below40"])),
     (alt["1+7+17"]["1"]["seat_exact"], (7, 17, 1), "칠판 윗변 가운데(C)에 하나 더, 한 번", "평균 %.0f%%, 40%% 아래 %d자리" % (alt["1+7+17"]["1"]["all"][0], alt["1+7+17"]["1"]["below40"]))]
fig, axs = plt.subplots(1, 4, figsize=(15.5, 5.6), dpi=150)
for ax, (v, sp, t, s) in zip(axs, P):
    room(ax, v, sp, t, s)
fig.suptitle("교실 40석: 자리마다 카메라의 자리를 정확히 맞힌 비율 (%)", fontsize=15, fontweight="bold", color=INK, x=0.012, ha="left", y=0.995)
fig.text(0.012, 0.015, "책상 위·가슴·안경 카메라를 합친 값. 빨간 테두리 = 40%% 아래. 전파 경로 계산 (Sionna RT), 카메라 %d대(지금 거는 곳) · %d대(다른 곳) × 사람 자세 8순간. 노란 막대 = 패널" % (B["now"]["cameras"], B["alt"]["cameras"]),
         fontsize=8.8, color=GRAY)
fig.subplots_adjust(left=0.01, right=0.995, top=0.86, bottom=0.06, wspace=0.06)
os.makedirs(OUT, exist_ok=True); fig.savefig(os.path.join(OUT, "사각지대_자리지도.png"), facecolor="white"); plt.close(fig); print("ok 사각지대_자리지도.png")
