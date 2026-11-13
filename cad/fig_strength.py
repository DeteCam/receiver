"""튼튼함 검사(strength.py)의 결과 그림.   python3 fig_strength.py   ->  15_튼튼함.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import paths

plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, SUB, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, WARN = "#2a78d6", "#c8501e"

J = json.load(open(os.path.join(paths.RES, "strength.json"), encoding="utf-8")); rows = J["rows"]
key = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
print("칸:", list(rows[0].keys()))
mk = "sf"
rows = sorted(rows, key=lambda r: r[mk]); show = rows[:14]; rest = rows[14:]
part = lambda r: r.get("part") or r.get("부품"); case = lambda r: r.get("case") or r.get("경우"); what = lambda r: r.get("what") or r.get("무엇")
fig = plt.figure(figsize=(13.2, 7.6), dpi=160, facecolor=SURF)
fig.text(0.035, 0.965, "출력물이 부러지지 않나: 힘이 걸리는 자리 %d곳의 여유" % len(J["rows"]), fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.035, 0.905, "여유 = 재료가 버티는 힘 ÷ 실제로 걸리는 힘. 2배보다 작으면 조심할 곳이다. 무광 PLA 의 약한 쪽 값으로 계산했다 (층 사이는 층 안의 1/3 남짓).\n여유가 작은 순서로 14곳만 그렸다. 나머지 %d곳은 모두 %.0f배 이상이다. 실물로 부러뜨려 본 것이 아니라 단면과 힘으로 계산한 값이다." % (len(rest), min(r[mk] for r in rest)),
         fontsize=11.5, color=SUB, va="top", linespacing=1.5)
ax = fig.add_axes([0.40, 0.09, 0.56, 0.69]); ax.set_facecolor(SURF)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, labelsize=11, length=0); ax.grid(True, axis="x", color=GRID, lw=1.0); ax.set_axisbelow(True)
y = np.arange(len(show))[::-1]; v = [r[mk] for r in show]
ax.barh(y, v, height=0.56, color=[WARN if x < 2 else C1 for x in v], zorder=3)
for yy, r in zip(y, show):
    ax.text(r[mk] * 1.07, yy, "%.1f배" % r[mk] + ("  ← 조심: 파이는 똑바로 눌러 꽂는다" if r[mk] < 2 else ""), va="center", ha="left", fontsize=11, color=INK)
ax.set_xscale("log"); ax.set_xlim(1, 60); ax.set_xticks([1, 2, 5, 10, 20, 50]); ax.set_xticklabels(["1배", "2배", "5배", "10배", "20배", "50배"])
ax.axvline(2, color=WARN, lw=1.2, ls=(0, (4, 3)), zorder=2); ax.text(2.05, len(show) - 0.35, "2배 선", fontsize=10.5, color=SUB, va="bottom")
ax.set_ylim(-0.7, len(show) - 0.1); ax.set_yticks(y)
ax.set_yticklabels(["%s · %s\n%s" % (part(r), what(r), case(r)) for r in show], fontsize=10.5, color=INK, linespacing=1.25)
fig.savefig(os.path.join(paths.IMG, "15_튼튼함.png"), facecolor=SURF); plt.close(fig)
print("saved 15_튼튼함.png", [(part(r), round(r[mk], 1)) for r in show[:4]])
