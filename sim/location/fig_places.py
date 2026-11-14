"""곳별 시뮬레이션(places.py)과 판별 시뮬레이션(identify_sim.py)의 결과 그림.   python3 fig_places.py
   27_곳별_결과.png   변기 칸·욕실·모텔 방·강의실에서 얼마나 맞나
   28_판별_결과.png   움직임에 반응하는 카메라를 얼마나 잡나, 카메라가 아닌 기기를 얼마나 잘못 잡나"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = paths.IMG + "/"                                           # 저장소에서는 docs/images, 작업용 배치에서는 out2/img
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, SUB, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"            # 검증한 색 순서 (파랑, 주황, 청록)

ROWS = [("모텔 방", "모텔 방 (사람은 나가 있음)"), ("욕실", "욕실"), ("강의실", "강의실"),
        ("변기 칸 (합성수지 칸막이)", "변기 칸 (합성수지 칸막이)"), ("  같은 칸, 여섯 번 재기", "변기 칸 (합성수지) 여섯 번 재기"),
        ("변기 칸 (철판 칸막이)", "변기 칸 (철판 칸막이)"), ("  같은 칸, 문 열고 여섯 번", "변기 칸 (철판) 문 열고 여섯 번")]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=SUB, labelsize=11, length=0)
    ax.grid(True, axis="x", color=GRID, lw=1.0); ax.set_axisbelow(True)


def fig_places():
    R = json.load(open(os.path.join(HERE, "places_result.json"), encoding="utf-8")); S = R["summary"]
    rows = [(lab, key) for lab, key in ROWS if key + "|W" in S and key + "|V" in S]
    panels = [("방향을 크게 틀리는 일 (30° 넘게, %)", lambda v: v["dir_over30"], "%.0f", 70, "한 자리에서 잰 방향. 작을수록 좋다"),
              ("끝났을 때 위치 오차 (가운데값, cm)", lambda v: 100 * v["stop"]["med"], "%.0f", 62, "멈춤 규칙대로 2~4자리. 작을수록 좋다"),
              ("참 위치가 추정 구간 안 (%)", lambda v: v["stop"]["in_zone15"], "%.0f", 112, "구간에 15cm 여유를 두고. 클수록 좋다")]
    fig = plt.figure(figsize=(14.2, 7.4), dpi=160, facecolor=SURF)
    fig.text(0.03, 0.965, "곳마다 얼마나 맞나: 변기 칸 · 욕실 · 모텔 방 · 강의실 (시뮬레이션)", fontsize=20, fontweight="bold", color=INK, va="top")
    fig.text(0.03, 0.905, "곳마다 카메라를 숨길 만한 자리에 %d번씩 두고, 장치와 같은 계산으로 찾았다. 벽 재질(타일·합성수지·철판·유리)의 반사와 사람 몸, 폰이 자리를 재는 오차(2cm, 1.5°)를 넣었다.\n실물로 잰 것이 아니다. 골 모양을 두 가지로 가정했다: 주황 = 회절 모형 그대로(나쁜 쪽), 파랑 = 손으로 잰 곡선과 비슷한 모양(좋은 쪽). 실제는 그 사이일 것이다." % R["n"],
             fontsize=11.5, color=SUB, va="top", linespacing=1.5)
    y = np.arange(len(rows))[::-1]; h = 0.34
    for k, (title, fn, fmt, xmax, note) in enumerate(panels):
        ax = fig.add_axes([0.215 + k * 0.265, 0.115, 0.225, 0.62]); style(ax)
        for j, (shape, col, lab) in enumerate((("W", C2, "골 모양: 모형 그대로 (나쁜 쪽)"), ("V", C1, "골 모양: 손 측정과 비슷 (좋은 쪽)"))):
            v = [fn(S[key + "|" + shape]) for _, key in rows]
            ax.barh(y + (0.5 - j) * (h + 0.04), v, height=h, color=col, label=lab, zorder=3)
            for yy, vv in zip(y + (0.5 - j) * (h + 0.04), v):
                ax.text(vv + xmax * 0.015, yy, fmt % vv, va="center", ha="left", fontsize=10.5, color=INK)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.7, len(rows) - 0.3); ax.set_yticks(y)
        ax.set_yticklabels([lab for lab, _ in rows] if k == 0 else [""] * len(rows), fontsize=12, color=INK)
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=22, fontweight="bold")
        ax.text(0, 1.015, note, transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
        if k == 0:
            ax.legend(frameon=False, fontsize=11, loc="upper left", bbox_to_anchor=(-0.86, -0.07), ncol=2, labelcolor=SUB, handlelength=1.2, columnspacing=2.2)
    fig.savefig(OUT + "27_곳별_결과.png", facecolor=SURF); plt.close(fig)
    print("saved 27_곳별_결과.png", [k for _, k in rows])


def fig_identify():
    R = json.load(open(os.path.join(paths.HERE, "identify", "identify_result.json"), encoding="utf-8")); S = R["summary"]
    pick = [(k, lab, c) for k, lab, c in (("8초씩 3번 (장치 기본)", "8초씩 3번 (장치 기본, 55초)", C1), ("6초씩 5번", "6초씩 5번 (67초)", C2), ("나갔다 들어오기 (밖 20초, 안 15초) 2번", "방을 나갔다 들어오기 2번 (83초)", C3)) if k in S]
    gains = [g for g in ("1.1", "1.2", "1.3", "1.5", "2.0", "3.0")]
    fig = plt.figure(figsize=(13.6, 6.2), dpi=160, facecolor=SURF)
    fig.text(0.04, 0.955, "판별: 움직임에 전송량이 반응하는 카메라를 얼마나 잡나 (시뮬레이션, %d번씩)" % R["n"], fontsize=19, fontweight="bold", color=INK, va="top")
    fig.text(0.04, 0.89, "화면이 시키는 대로 멈췄다 움직였다 하는 동안 0.5초마다 전송량을 세어, 박자를 따라 오르는지 본다. 카메라마다 움직임에 반응하는 정도가 다르다 (가로축).\n반응이 약한 카메라(1.2배 아래)는 놓친다: 그때는 '의심 기기'로 두고 위치를 재러 간다. 숫자는 논문의 전송량 범위로 만든 가짜 자료에서 나온 것이다.",
             fontsize=11.5, color=SUB, va="top", linespacing=1.5)
    ax = fig.add_axes([0.065, 0.13, 0.50, 0.60]); style(ax); ax.grid(True, axis="y", color=GRID, lw=1.0)
    x = np.arange(len(gains))
    for k, lab, c in pick:
        v = [S[k]["detect"][g] for g in gains]
        ax.plot(x, v, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=2, label=lab, solid_capstyle="round", zorder=3)
    k0 = pick[0][0]
    for xi, g in zip(x, gains):
        ax.annotate("%.0f%%" % S[k0]["detect"][g], (xi, S[k0]["detect"][g]), textcoords="offset points", xytext=(0, 11), ha="center", fontsize=10.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(["%s배" % g for g in gains]); ax.set_ylim(0, 112); ax.set_yticks(range(0, 101, 20)); ax.set_xlim(-0.35, len(gains) - 0.65)
    ax.set_xlabel("움직일 때 전송량이 늘어나는 정도 (카메라의 반응)", color=SUB, fontsize=12); ax.set_title("카메라를 카메라라고 하는 비율 (%)", loc="left", fontsize=13, color=INK, pad=10, fontweight="bold")
    ax.legend(frameon=False, fontsize=11, loc="lower right", labelcolor=SUB)
    ax2 = fig.add_axes([0.66, 0.13, 0.31, 0.60]); style(ax2)
    kinds = ["반응 없는 카메라\n(다른 곳을 찍는 것)", "꾸준한 기기\n(영상 통화)", "뭉텅이 기기\n(웹·동영상 보기)", "다른 방 카메라\n(제멋대로 오른다)"]
    y = np.arange(len(kinds))[::-1]; h = 0.24
    for j, (k, lab, c) in enumerate(pick):
        v = [S[k]["detect"]["1.0"], S[k]["false"]["꾸준한 기기"], S[k]["false"]["뭉텅이 기기"], S[k]["false"]["다른 방 카메라"]]
        yy = y + (1 - j) * (h + 0.03)
        ax2.barh(yy, v, height=h, color=c, zorder=3)
        for a, b in zip(yy, v):
            ax2.text(b + 0.25, a, "%.1f" % b, va="center", ha="left", fontsize=10.5, color=INK)
    ax2.set_yticks(y); ax2.set_yticklabels(kinds, fontsize=11, color=INK); ax2.set_xlim(0, 10); ax2.set_ylim(-0.6, len(kinds) - 0.4)
    ax2.set_title("이 방을 찍지 않는데 '카메라'라고 하는 비율 (%)", loc="left", fontsize=13, color=INK, pad=22, fontweight="bold")
    ax2.text(0, 1.015, "색은 왼쪽 그림과 같다. 작을수록 좋다", transform=ax2.transAxes, fontsize=10.5, color=SUB, va="bottom")
    fig.savefig(OUT + "28_판별_결과.png", facecolor=SURF); plt.close(fig)
    print("saved 28_판별_결과.png", [k for k, _, _ in pick])


if __name__ == "__main__":
    fig_places()
    if os.path.exists(os.path.join(paths.HERE, "identify", "identify_result.json")):
        fig_identify()
