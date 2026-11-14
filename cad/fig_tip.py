"""넘어짐 검사 그림 (16_넘어짐.png).   python3 fig_tip.py      (먼저 tipover.py, tip_mujoco.py)
   왼쪽: 위에서 본 그림 (받침, 무게중심이 움직이는 범위, 판이 닿는 범위, 비워 둘 둘레)
   오른쪽: 받침 끝까지 남는 거리, 넘어가는 기울기, 넘어뜨리는 힘 (계산과 물리 엔진), 모터가 스스로 미는 힘"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import paths
from p2 import BASE

plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, SUB, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # 검증한 색 순서 (파랑, 주황, 청록)
ST = json.load(open(os.path.join(paths.RES, "tipover.json"), encoding="utf-8")); MJ = json.load(open(os.path.join(paths.RES, "tip_mujoco.json"), encoding="utf-8"))
CASES = [("S", True, "작은 받침 S + 배터리", C1), ("S", False, "작은 받침 S, 배터리 없이", C2), ("L", True, "큰 받침 L + 배터리", C3)]
skey = lambda size, batt: [k for k, v in ST.items() if v.get("size") == size and v.get("battery") == batt and v.get("plate_t") == 1.0 and v.get("length") == 150.0][0]
mkey = lambda size, batt: "받침 %s, 배터리 %s" % (size, "있음" if batt else "없음")
S = [ST[skey(s, b)] for s, b, _, _ in CASES]; M = [MJ[mkey(s, b)] for s, b, _, _ in CASES]
RCH = ST["reach"]["150"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, labelsize=10.5, length=0); ax.grid(True, axis="x", color=GRID, lw=1.0); ax.set_axisbelow(True)


fig = plt.figure(figsize=(15.0, 9.0), dpi=160, facecolor=SURF)
fig.text(0.03, 0.972, "혼자 넘어지나: 무게중심, 바닥 기울기, 미는 힘, 벽", fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.928, "관절 세 개가 움직이는 전 범위에서 가장 나쁜 자세를 골라 본 값이다 (알루미늄 판 100×150×1T, 동전 8개). 무게는 슬라이서가 계산한 부품 무게와 사는 부품의 어림 무게를 썼다.\n"
                     "막대 = 무게중심으로 계산한 값, 검은 점 = 물리 엔진(MuJoCo)에서 실제 속도로 움직여 본 값. 둘이 맞는지 서로 확인했다.", fontsize=11.5, color=SUB, va="top", linespacing=1.5)

# ------------------------------------------------------------------ 왼쪽: 위에서 본 그림
ax = fig.add_axes([0.02, 0.105, 0.40, 0.685]); ax.set_facecolor(SURF); ax.set_aspect("equal"); ax.axis("off")
clear = RCH["clear_cm"] * 10.0; reach = RCH["max_mm"]; lim = clear + 26
ax.add_patch(Circle((0, 0), clear, facecolor="#f3f1ec", edgecolor=SUB, lw=1.4, ls=(0, (5, 4)), zorder=1))
ax.add_patch(Circle((0, 0), reach, facecolor="#e9eef7", edgecolor=C1, lw=1.2, zorder=2))
ax.add_patch(Circle((0, 0), BASE["L"]["R"], facecolor="none", edgecolor=C3, lw=1.8, zorder=3))
ax.add_patch(Circle((0, 0), BASE["S"]["R"], facecolor="#1B2A57", edgecolor="#1B2A57", lw=1.0, zorder=4, alpha=0.92))
ax.add_patch(Circle((0, 0), S[1]["worst"]["off"], facecolor=C2, edgecolor="none", zorder=5, alpha=0.95))
ax.add_patch(Circle((0, 0), S[0]["worst"]["off"], facecolor="#ffffff", edgecolor="none", zorder=6))
ax.plot([0], [0], marker="+", color=INK, ms=9, mew=1.4, zorder=7)


def note(r, ang, text, tx, ty, ha="left"):
    a = np.radians(ang); x, y = r * np.cos(a), r * np.sin(a)
    ax.annotate(text, xy=(x, y), xytext=(tx, ty), fontsize=10.8, color=INK, ha=ha, va="center", linespacing=1.35, zorder=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=0.9, shrinkA=2, shrinkB=0), bbox=dict(facecolor=SURF, edgecolor="#c9ccd1", boxstyle="round,pad=0.35", lw=0.9))


note(clear, 128, "비워 둘 둘레: 가운데에서 %d cm\n(벽, 물건, 사람 손)" % RCH["clear_cm"], -lim + 4, lim - 22)
note(reach, 52, "판과 균형추가 닿는 가장 먼 곳\n가운데에서 %.1f cm" % (reach / 10.0), lim - 4, lim - 22, "right")
note(BASE["L"]["R"], -38, "큰 받침 L (지름 %d cm)" % (BASE["L"]["R"] / 5), lim - 4, -lim + 30, "right")
note(BASE["S"]["R"], 178, "작은 받침 S\n(지름 %d cm)" % (BASE["S"]["R"] / 5), -lim + 4, 40)
note(S[0]["worst"]["off"] * 0.5, 235, "무게중심이 움직이는 범위\n배터리가 있으면 가운데에서 %.1f cm (흰 원),\n없으면 %.1f cm (주황 띠) 까지" % (S[0]["worst"]["off"] / 10, S[1]["worst"]["off"] / 10), -lim + 4, -lim + 30)
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
fig.text(0.03, 0.845, "위에서 본 그림 (실제 비율)", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.03, 0.815, "무게중심이 받침 원 안에 있으면 넘어지지 않는다. 어느 자세에서도 한참 안쪽이다.", fontsize=10.8, color=SUB, va="top")

# ------------------------------------------------------------------ 오른쪽 위: 세 가지 값
labs = [c[2] for c in CASES]; cols = [c[3] for c in CASES]; y = np.arange(3)[::-1]
panels = [("받침 끝까지 남는 거리", "mm", [s["worst"]["margin"] for s in S], [m["zmp"]["scan"]["margin_mm"] for m in M], "%.0f", "재는 동안 눌림 중심"),
          ("넘어가는 바닥 기울기", "°", [s["worst"]["tip_deg"] for s in S], [m["slope_tip"]["angle_deg"] for m in M], "%.0f", "마찰이 충분할 때"),
          ("꼭대기를 밀어 넘어뜨리는 힘", "N", [s["worst"]["push_top_N"] for s in S], [m["push_axis"]["force_N"] for m in M], "%.1f", "회전축 위를 밀 때")]
for i, (title, unit, calc, sim, fmt, simnote) in enumerate(panels):
    ax = fig.add_axes([0.575 + (i % 3) * 0.142, 0.605, 0.118, 0.16]); style(ax)
    ax.barh(y, calc, height=0.56, color=cols, zorder=3)
    ax.scatter(sim, y, s=42, color=INK, zorder=5, edgecolors=SURF, linewidths=1.2)
    top = max(max(calc), max(sim)) * 1.42
    for yy, c, s_ in zip(y, calc, sim):
        ax.text(max(c, s_) + top * 0.04, yy, (fmt % c) + unit, va="center", ha="left", fontsize=10.5, color=INK)
    ax.set_xlim(0, top); ax.set_ylim(-0.6, 2.6); ax.set_yticks(y); ax.set_yticklabels(labs if i == 0 else [""] * 3, fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.text(0.0, -0.27, "점: " + simnote, transform=ax.transAxes, fontsize=9.3, color=SUB, va="top")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3)); ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _p, u=unit: ("%g" % v) + u))
fig.text(0.445, 0.845, "얼마나 버티나", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.445, 0.815, "클수록 잘 버틴다. 보통 바닥(마찰 0.35)에서는 넘어가기 전에 먼저 미끄러진다 (S %.0f°, L %.0f° 기울기부터)." % (M[0]["slope_slide"]["angle_deg"], M[2]["slope_slide"]["angle_deg"]), fontsize=10.8, color=SUB, va="top")

# ------------------------------------------------------------------ 오른쪽 아래: 모터가 스스로 미는 힘
ax = fig.add_axes([0.615, 0.272, 0.36, 0.205]); style(ax)
sit = [("spin", "판이 무엇에 걸림 (판 모터)"), ("yaw", "받침이 돌다 판이 걸림 (받침 모터)"), ("nod", "머리를 숙이다 판 위끝이 벽에 닿음\n(끄덕임 모터)")]
yy = np.arange(len(sit))[::-1]; h = 0.25
for j, (s_, (_, _, lab, col)) in enumerate(zip(S, CASES)):
    v = [s_["self_push"][k]["pct"] for k, _ in sit]; off = (1 - j) * (h + 0.03)
    ax.barh(yy + off, v, height=h, color=col, zorder=3, label=lab)
    for a, b in zip(yy, v):
        ax.text(b + 3, a + off, "%.0f%%" % b, va="center", ha="left", fontsize=9.8, color=INK)
ax.axvline(100, color="#c8501e", lw=1.3, ls=(0, (4, 3)), zorder=2); ax.text(103, len(sit) - 0.5, "100% 를 넘으면 스스로 넘어갈 수 있다", fontsize=9.8, color=SUB, va="center")
ax.set_yticks(yy); ax.set_yticklabels([s[1] for s in sit], fontsize=10.3, color=INK, linespacing=1.25); ax.set_xlim(0, 200); ax.set_ylim(-0.6, len(sit) - 0.2)
ax.set_xticks([0, 50, 100, 150]); ax.set_xticklabels(["0%", "50%", "100%", "150%"])
ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.52), frameon=False, fontsize=9.8, handlelength=1.0, borderaxespad=0.2)
fig.text(0.445, 0.538, "모터가 스스로 넘어뜨릴 수 있나", fontsize=14.5, fontweight="bold", color=INK, va="top")
fig.text(0.445, 0.508, "모터가 낼 수 있는 가장 큰 힘 ÷ 넘어뜨리는 데 드는 힘 (계산). 아무것에도 닿지 않으면 0% 다.", fontsize=10.8, color=SUB, va="top")
# 벽을 세워 두고 머리를 숙여 본 결과 (물리 엔진): 받침에서 벽까지의 틈마다
fig.text(0.445, 0.222, "벽 옆에서 머리를 85° 까지 숙여 봤을 때 (물리 엔진, 바닥 마찰 0.35 와 0.8)", fontsize=11.8, fontweight="bold", color=INK, va="top")
cols = [0.445, 0.635, 0.745, 0.862]; y0 = 0.188
gaps = sorted(set(w["gap_mm"] for w in M[0]["wall"]))
heads = ["", "받침에서 벽까지 %d cm" % (gaps[0] / 10), "%d cm" % (gaps[1] / 10), "가운데에서 %d cm (비워 둘 둘레)" % RCH["clear_cm"]]
for x, t in zip(cols, heads):
    fig.text(x, y0, t, fontsize=10.3, color=SUB, va="top")


def wall_word(m, i):
    g = sorted(set(w["gap_mm"] for w in m["wall"]))[i]; r = [w for w in m["wall"] if w["gap_mm"] == g]
    if all(w["result"] == "넘어짐" for w in r):
        return "넘어짐", "#c8501e"
    if any("들림" in w["result"] for w in r):
        return "받침이 %.0f~%.0f° 들림" % (min(w["tilt_deg"] for w in r), max(w["tilt_deg"] for w in r)), "#b26a00"
    if all(not w["touched"] for w in r):
        return "벽에 닿지 않음", "#1f8a4c"
    return " / ".join(sorted(set(w["result"] for w in r))), INK


for i, ((_, _, lab, _), m_) in enumerate(zip(CASES, M)):
    y = y0 - 0.034 * (i + 1)
    fig.text(cols[0], y, lab, fontsize=10.5, color=INK, va="top")
    for k in range(3):
        word, col = wall_word(m_, k); fig.text(cols[k + 1], y, word, fontsize=10.5, color=col, va="top", fontweight="bold")
fig.text(0.03, 0.062, "→ 탁 트인 곳에서는 스스로 넘어지지 않는다: 재는 동안 무게가 실리는 점은 받침 가운데에서 %.1f cm 를 벗어나지 않는다. 다만 벽이나 물건이 가까이 있으면 머리를 숙이다가 그것을 밀어\n    받침 크기와 상관없이 넘어갈 수 있다. 그래서 둘레 %d cm 를 비우고 쓴다 (폰 화면에도 이 안내가 뜬다). 평소에는 큰 받침 L 이 훨씬 든든하다: 넘어뜨리는 데 힘이 두 배 든다." % (
    M[0]["zmp"]["scan"]["zmp_max_mm"] / 10, RCH["clear_cm"]), fontsize=11.3, color=INK, va="top", linespacing=1.5)
out = os.path.join(paths.IMG, "16_넘어짐.png"); fig.savefig(out, facecolor=SURF); plt.close(fig)
print("saved", out)
for (s, b, lab, _), s_, m_ in zip(CASES, S, M):
    print("%-22s 남는 거리 %.1f (모의 %.1f), 기울기 %.1f (모의 %.1f, 미끄러짐 %.1f), 미는 힘 %.2f (모의 축 %.2f, 판 귀 %.2f), 스스로 %s" % (
        lab, s_["worst"]["margin"], m_["zmp"]["scan"]["margin_mm"], s_["worst"]["tip_deg"], m_["slope_tip"]["angle_deg"], m_["slope_slide"]["angle_deg"], s_["worst"]["push_top_N"], m_["push_axis"]["force_N"], m_["push"]["force_N"],
        {k: v["pct"] for k, v in s_["self_push"].items()}))
