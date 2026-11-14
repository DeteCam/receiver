"""위치 추정 방법 비교 그림 (26_위치_추정_방법.png).  python3 fig_locate.py   (먼저 locate3d.py, locate3d_var2.py 를 돌려 둔다)"""
import paths                                   # 계산 코드(core)를 찾는다
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import locate3d as L3
import locate as K
from figures import OUT, SURF, INK, SUB, GRID, C1, C2, C3

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "locate3d_result.json"), encoding="utf-8"))["summary"]
V2 = json.load(open(os.path.join(HERE, "locate3d_var2.json"), encoding="utf-8"))["summary"]


def pick_trial(seed=31, want=(0.9, 0.5)):
    """보여 줄 예 하나: 2자리와 3자리의 오차·구간 길이가 가운데값에 가까운 것"""
    best = None
    for i, T in enumerate(L3.trials(40, "보통 방", "W", "inside", seed=seed, npos=3)):
        g = T["grid"]; rows = []
        for m in (1, 2, 3):
            post = K.posterior(g, T["Ms"][:m], None, K.Q_BAD, K.SIG_SCALE); c, idx, length, vol, conf, sec = K.zone(g, post)
            rows.append(dict(post=post, c=c, idx=idx, length=length, e=float(np.linalg.norm(c - T["cam"]))))
        inner = [b for b in T["boxes"] if b[0] > 0.3 and b[1] > 0.3 and b[3] < g.L[0] - 0.3 and b[4] < g.L[1] - 0.3]
        score = abs(rows[1]["length"] - want[0]) + abs(rows[2]["length"] - want[1]) + 2 * abs(rows[1]["e"] - 0.2) + 2 * abs(rows[2]["e"] - 0.12) + (0 if rows[2]["e"] < rows[1]["e"] else 1)
        if 0.8 < T["cam"][0] < g.L[0] - 0.8 and 0.9 < T["cam"][1] < g.L[1] - 0.9 and (best is None or score < best[0]):
            best = (score, T, rows)
    return best[1], best[2]


def panel(ax, T, rows, m, title):
    g = T["grid"]; L = g.L
    ax.set_facecolor(SURF); ax.set_aspect("equal"); ax.set_xlim(-0.12, L[0] + 0.12); ax.set_ylim(-0.12, L[1] + 0.12)
    ax.add_patch(Rectangle((0, 0), L[0], L[1], fill=False, ec=SUB, lw=1.6))
    for b in T["boxes"]:
        ax.add_patch(Rectangle((b[0], b[1]), b[3] - b[0], b[4] - b[1], fc="#d9d8d2", ec="#b9b8b2", lw=0.8))
    r = rows[m - 1]; X = g.X[r["idx"]]; w = r["post"][r["idx"]] / r["post"][r["idx"]].max()
    keys = {}
    for (x, y, _), ww in zip(X, w):
        k = (round(float(x), 2), round(float(y), 2)); keys[k] = max(keys.get(k, 0.0), float(ww))
    for (x, y), ww in keys.items():
        ax.add_patch(Rectangle((x - g.step / 2, y - g.step / 2), g.step, g.step, fc=C2, ec="none", alpha=0.18 + 0.6 * ww))
    for i, M in enumerate(T["Ms"][:m]):
        o = M["ref"]
        if M["d"] is not None:
            t = K.ray_exit(o, M["d"], np.zeros(3), L); e = o + M["d"] * min(t, 9.0)
            ax.plot([o[0], e[0]], [o[1], e[1]], color=C1, lw=1.2, alpha=0.75)
        ax.plot(o[0], o[1], "o", ms=13, color=C1, mec="white", mew=1.5); ax.text(o[0], o[1], str(i + 1), color="white", fontsize=9, fontweight="bold", ha="center", va="center")
    cam = T["cam"]; ax.plot(cam[0], cam[1], marker="X", ms=11, color=INK, mec="white", mew=1.2)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=12.5, color=INK, loc="left", pad=6, fontweight="bold")
    ax.text(0, -0.2, "구간 길이 %.1fm · 오차 %dcm" % (r["length"], round(100 * r["e"])) if m > 1 else "구간 길이 %.1fm (방향만 안다)" % r["length"], fontsize=11, color=SUB, va="top")


if __name__ == "__main__":
    plt.rcParams["font.family"] = "Noto Sans CJK JP"; plt.rcParams["axes.unicode_minus"] = False
    T, rows = pick_trial()
    fig = plt.figure(figsize=(16.5, 7.4), facecolor=SURF)
    fig.text(0.03, 0.965, "카메라가 방 한가운데 있어도: 방향 선을 '가능성 부피'로 합쳐 구간을 구한다", fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.905, "보통 방 3.5×4.5m, 카메라는 방 안쪽 물건(회색) 위에 있다고 놓은 시뮬레이션. 벽에 붙어 있다는 가정은 쓰지 않는다. 한 자리 = 판을 세 번 돌려 재는 것(약 1분).", fontsize=12.5, color=SUB, va="top")
    for k, (m, title) in enumerate(((1, "① 한 자리"), (2, "② 앱이 권한 둘째 자리까지"), (3, "③ 셋째 자리까지"))):
        ax = fig.add_axes([0.03 + k * 0.19, 0.17, 0.18, 0.64]); panel(ax, T, rows, m, title)
    fig.text(0.03, 0.085, "왼쪽 그림(예 하나): 파란 동그라미 = 수신기를 놓은 자리와 거기서 나온 방향, 주황 = 추정 구간(위에서 본 모습), 검은 X = 진짜 카메라, 회색 = 가구.", fontsize=11, color=SUB)

    ax = fig.add_axes([0.715, 0.22, 0.17, 0.54]); ax.set_facecolor(SURF)
    A, B = R["보통 방|inside|W"], R["보통 방|inside|V"]; A2, B2 = V2["보통 방|inside|W"]["기본"], V2["보통 방|inside|V"]["기본"]
    data = [("선 두 개가 만나는 곳\n아무 두 자리 (처음 생각)", A["tri_rand"]["med"], B["tri_rand"]["med"], A["tri_rand"]["in50"], B["tri_rand"]["in50"]),
            ("선 두 개가 만나는 곳\n둘째 자리를 골라서", A["tri_chosen"]["med"], B["tri_chosen"]["med"], A["tri_chosen"]["in50"], B["tri_chosen"]["in50"]),
            ("가능성 부피 · 2자리", A["2"]["med"], B["2"]["med"], A["2"]["in50"], B["2"]["in50"]),
            ("가능성 부피 · 3자리", A["3"]["med"], B["3"]["med"], A["3"]["in50"], B["3"]["in50"]),
            ("가능성 부피 · 4자리", A2["4"]["med"], B2["4"]["med"], A2["4"]["in50"], B2["4"]["in50"])]
    ys = np.arange(len(data))[::-1]
    for y, (name, a, b, pa, pb) in zip(ys, data):
        lo, hi = 100 * min(a, b), 100 * max(a, b)
        ax.barh(y, hi, height=0.42, color=C1 if "부피" in name else "#9aa3ad", zorder=3)
        ax.barh(y, lo, height=0.42, color=C1 if "부피" in name else "#9aa3ad", zorder=4)
        ax.barh(y, hi - lo, left=lo, height=0.42, color="white", alpha=0.45, zorder=5)
        ax.text(hi + 1.2, y, ("%d~%dcm" % (round(lo), round(hi))) if round(lo) != round(hi) else "%dcm" % round(hi), va="center", fontsize=12, color=INK, fontweight="bold")
        ax.text(-1.5, y, name, va="center", ha="right", fontsize=11.5, color=INK, linespacing=1.25)
        ax.text(68, y, "%d~%d%%" % (round(min(pa, pb)), round(max(pa, pb))), va="center", fontsize=12, color=INK)
    ax.set_xlim(0, 62); ax.set_ylim(-0.6, len(data) - 0.4); ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=SUB, labelsize=10.5, length=0); ax.grid(True, axis="x", color=GRID, lw=1.0); ax.set_axisbelow(True)
    ax.set_xlabel("위치 오차의 가운데값 (cm) · 짧을수록 좋다", color=SUB, fontsize=11.5)
    ax.text(0, len(data) - 0.28, "위치 오차", fontsize=12, color=SUB, fontweight="bold"); ax.text(68, len(data) - 0.28, "50cm 안에 든 비율", fontsize=12, color=SUB, fontweight="bold")
    fig.text(0.03, 0.045, "오른쪽 막대: 숫자 범위는 신호가 가려지는 모양(골)을 좋게 / 나쁘게 가정한 두 경우 (진한 색 = 좋게). 방법 비교는 200번, 4자리는 120번 돌린 값. 결과 파일: sim/locate3d_result.json, locate3d_var2.json", fontsize=11, color=SUB)
    fig.savefig(OUT + "26_위치_추정_방법.png", dpi=150, facecolor=SURF)
    print("saved", OUT + "26_위치_추정_방법.png", [(round(r["length"], 2), round(r["e"], 2)) for r in rows])
