"""방 전체 전파 계산 그림 (그림 폴더의 33_방_전파계산.png).   python3 fig_room2d.py [저장할 곳]
room2d.json (gprMax, 2차원) 을 읽는다: 방의 평면도에 카메라 후보 자리를 방향 오차로 칠하고, 오차 분포를 견준다."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import font_manager as fm
import room2d as R2
HERE = os.path.dirname(os.path.abspath(__file__))
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name(); plt.rcParams["axes.unicode_minus"] = False
INK, SUB, BG, GRID = "#22262B", "#5B6470", "#F4F2EE", "#D9D5CC"
OK, MID, BAD, UNSEEN = "#1baf7a", "#e8a33d", "#d64545", "#b9b6ae"
WALL = {"brick": "#b98b6e", "tile": "#9fb4c4", None: BG}
ITEM = {"wood_hollow": "#c9a57a", "pec": "#4a4f57", "glass": "#a9d4e6", "wood": "#c9a57a"}
J = json.load(open(os.path.join(HERE, "room2d.json"), encoding="utf-8"))
SHOW = [k for k in ("화장실 · 수신기 1", "화장실 · 수신기 2", "보통 방 · 수신기 1", "보통 방 · 수신기 2") if k in J]


def col(r):
    if not r["seen"]:
        return UNSEEN
    return OK if r["err"] <= 10 else (MID if r["err"] <= 30 else BAD)


fig = plt.figure(figsize=(15, 8.6), dpi=150, facecolor=BG)
n = len(SHOW)
for i, key in enumerate(SHOW):
    name = key.split(" · ")[0]; R = R2.ROOMS[name]; W, H = R["size"]; t = R["t"]
    ax = fig.add_axes([0.035 + i * (0.93 / max(n, 1)), 0.30, 0.93 / max(n, 1) - 0.02, 0.50]); ax.set_facecolor(BG)
    ax.add_patch(Rectangle((-t, -t), W + 2 * t, H + 2 * t, color=WALL[R["wall"]], zorder=0)); ax.add_patch(Rectangle((0, 0), W, H, color="#ffffff", zorder=1))
    for k, x0, y0, x1, y1, mat in R["items"]:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, color=ITEM[mat], zorder=2, alpha=0.95 if mat != "wood_hollow" else 0.55))
    rows = J[key]["rows"]
    xs = [r["cam"][0] for r in rows]; ys = [r["cam"][1] for r in rows]
    ax.scatter(xs, ys, s=26, c=[col(r) for r in rows], zorder=3, edgecolors="none")
    ax_, ay_, _ = J[key]["rx"]
    ax.scatter([ax_], [ay_], s=120, marker="*", color="#22305E", zorder=4)
    ax.set_xlim(-t - 0.05, W + t + 0.05); ax.set_ylim(-t - 0.05, H + t + 0.05); ax.set_aspect("equal"); ax.axis("off")
    S = J[key]["summary"]
    ax.set_title("%s (수신기 %s)\n가운데값 %.0f° · 골이 보인 곳 %.0f° · 30° 넘게 %.0f%%" % (name, key[-1], S["med"], S.get("med_seen", S["med"]), S["over30"]), fontsize=11, color=INK, loc="left")
# 아래: 오차 분포 (누적)
ax = fig.add_axes([0.06, 0.07, 0.55, 0.17]); ax.set_facecolor(BG)
for key, c in zip(["빈 공간 · 수신기 1"] + SHOW, ["#8E949C", "#2a78d6", "#5aa0e8", "#eb6834", "#f2a07a"]):
    if key not in J:
        continue
    e = np.sort([r["err"] for r in J[key]["rows"]]); ax.plot(e, np.arange(1, len(e) + 1) / len(e) * 100, color=c, lw=2, label=key)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=SUB, labelsize=9); ax.grid(True, color=GRID, lw=0.6); ax.set_xlim(0, 60); ax.set_ylim(0, 100)
ax.set_xlabel("방향 오차 (도)", color=INK, fontsize=10); ax.set_ylabel("카메라 자리 (%)", color=INK, fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc="lower right")
fig.text(0.66, 0.205, "점의 색: 카메라를 그 자리에 두었을 때 수신기가 찾은 방향의 오차", fontsize=10, color=INK)
for j, (c, lab) in enumerate(((OK, "10° 안"), (MID, "10~30°"), (BAD, "30° 넘게"), (UNSEEN, "골이 안 보임 (재지 못함으로 표시)"))):
    fig.text(0.665 + (j % 2) * 0.14, 0.165 - (j // 2) * 0.035, "●", color=c, fontsize=13); fig.text(0.68 + (j % 2) * 0.14, 0.168 - (j // 2) * 0.035, lab, fontsize=9.5, color=SUB)
fig.text(0.665, 0.085, "★ 수신기 (판이 반지름 80mm 로 돈다)   벽 색: 벽돌 / 타일   짙은 회색: 금속(거울, 철제 서랍장)", fontsize=9, color=SUB)
fig.text(0.035, 0.955, "방 전체를 전파 계산으로 풀어 수신기의 실제 코드로 방향을 찾은 결과 (gprMax 4.0.1, 시간 영역 2차원, 2.427~2.447GHz 16개 주파수)", fontsize=14, weight="bold", color=INK)
fig.text(0.035, 0.912, "판 각 5°마다 한 번씩 72번 풀고, 상반성으로 방 안의 카메라 후보 자리 전부를 한 번에 얻는다. 곡선은 firmware/core 의 fusion.trough 로 읽는다. 2차원이라 판이 위아래로 끝없이 길어 골이 실제보다 깊다.", fontsize=10, color=SUB)
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(empaths.IMG, "33_방_전파계산.png")
fig.savefig(out, facecolor=BG); print("saved", out)
