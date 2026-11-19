# -*- coding: utf-8 -*-
"""빠른 시험용 도안 (A4, 실제 크기).  python3 template.py  ->  도안.pdf

1쪽: 우드락에 붙일 안테나 9개 자리 (3 x 3, 33 mm, 45도). 가운데 1번이 재는 안테나, 둘레 8개는 '이웃'.
     패널 안쪽 안테나는 이웃이 여덟이라, 가장 나쁜 경우를 그대로 본다. 반사판 간격 막대 표.
2쪽: 돌리기용 각도판 (15도마다).
인쇄할 때 '실제 크기(100%)' 로. 1쪽 아래의 50 mm 자로 확인한다.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
for f in font_manager.findSystemFonts():
    if "NotoSansCJK-Regular" in f or "NanumGothic.ttf" in f:
        font_manager.fontManager.addfont(f)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=f).get_name()
        break

W, H = 210.0, 297.0
PITCH, FL, FW, ANG = 33.0, 34.9, 9.0, 45.0


def page():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    return fig, ax


def film(ax, cx, cy, n, main=False):
    a = np.radians(ANG); u = np.array([np.cos(a), np.sin(a)]); v = np.array([-u[1], u[0]])
    c = np.array([cx, cy]); pts = [c + s * FL / 2 * u + t * FW / 2 * v for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ax.add_patch(Polygon(pts, closed=True, fc="#fde9a8" if main else "#f2f2f2", ec="black", lw=0.8))
    ax.plot([cx], [cy], "k+", ms=6)
    ax.text(cx + 7, cy - 9, str(n), fontsize=11, weight="bold" if main else "normal", ha="center", va="center")


def page1(pdf):
    fig, ax = page()
    ax.text(W / 2, 282, "빠른 시험 도안 1/2: 안테나 자리 (실제 크기로 인쇄)", ha="center", fontsize=13, weight="bold")
    ax.text(W / 2, 274, "우드락(150 x 150 이상)에 이 종이를 붙이고, 노란 칸(1번)부터 필름 안테나를 겹쳐 붙인다.", ha="center", fontsize=8.5)
    ax.text(W / 2, 269, "+ 는 안테나 가운데(선이 나오는 곳). 선은 우드락에 구멍을 뚫어 뒤로 뺀다. 간격 33 mm, 45도 = 실제 패널과 같다.", ha="center", fontsize=8.5)
    cx0, cy0 = W / 2, 185
    ax.add_patch(Rectangle((cx0 - 75, cy0 - 75), 150, 150, fc="none", ec="#999999", lw=0.6, ls="--"))
    ax.text(cx0 - 73, cy0 + 70, "우드락 150 x 150 (반사판도 이 크기 이상)", fontsize=7, color="#666666")
    order = {(0, 0): 1, (-1, 1): 2, (0, 1): 3, (1, 1): 4, (-1, 0): 5, (1, 0): 6, (-1, -1): 7, (0, -1): 8, (1, -1): 9}
    for (i, j), n in order.items():
        film(ax, cx0 + i * PITCH, cy0 + j * PITCH, n, main=(n == 1))
    ax.annotate("", xy=(cx0 + 82, cy0 + 70), xytext=(cx0 + 82, cy0 + 50), arrowprops=dict(arrowstyle="-|>", lw=1.2))
    ax.text(cx0 + 84, cy0 + 58, "위", fontsize=8)
    # 간격 막대
    y = 92
    ax.text(18, y, "반사판 간격 (안테나 면에서 반사판까지)", fontsize=10, weight="bold")
    rows = [("15 mm", "우드락 5 mm 3장"), ("20 mm", "우드락 5 mm 4장"), ("31 mm", "우드락 5 mm 6장 + 두꺼운 종이 1 mm")]
    for k, (a, b) in enumerate(rows):
        ax.text(22, y - 8 - 6 * k, "%s  =  %s 를 네 귀퉁이에 기둥으로 (테이프)" % (a, b), fontsize=8.5)
    ax.text(18, y - 30, "반사판 = 두꺼운 종이 150 x 150 에 쿠킹 호일을 주름 없이 붙인 것 (호일 쪽이 안테나를 본다).", fontsize=8.5)
    ax.text(18, y - 36, "우드락이 5 mm 가 아니면 자로 재서 맞춘다. 1~2 mm 틀려도 판정에는 거의 영향이 없다.", fontsize=8.5)
    # 자
    ax.plot([18, 68], [30, 30], "k-", lw=1.2)
    for t in range(0, 51, 10):
        ax.plot([18 + t, 18 + t], [30, 33 if t % 50 else 35], "k-", lw=0.8)
    ax.text(18, 22, "이 선이 50 mm 가 아니면 인쇄 배율이 틀린 것 (프린터에서 '실제 크기')", fontsize=8)
    ax.text(W - 18, 22, "DeteCam receiver / firmware/bench/template.py", fontsize=6, ha="right", color="#888888")
    pdf.savefig(fig); plt.close(fig)


def page2(pdf):
    fig, ax = page()
    ax.text(W / 2, 282, "빠른 시험 도안 2/2: 각도판 (15도마다)", ha="center", fontsize=13, weight="bold")
    ax.text(W / 2, 274, "0 이 보내는 보드 쪽. 우드락 판의 가운데(1번 안테나 아래)를 + 에 맞춰 세우고, 판 앞면의 화살표를 각도에 맞춘다.", ha="center", fontsize=8.5)
    c = np.array([W / 2, 145]); R = 88
    ax.add_patch(Circle(c, R, fc="none", ec="black", lw=1.0)); ax.add_patch(Circle(c, 30, fc="none", ec="#999999", lw=0.6))
    for a in range(0, 360, 5):
        t = np.radians(90 - a); u = np.array([np.cos(t), np.sin(t)])
        L = 10 if a % 15 == 0 else 4
        ax.plot(*zip(c + (R - L) * u, c + R * u), "k-", lw=1.0 if a % 15 == 0 else 0.5)
        if a % 15 == 0:
            ax.text(*(c + (R + 7) * u), str(a), ha="center", va="center", fontsize=8, weight="bold" if a in (0, 180) else "normal")
            ax.plot(*zip(c + 30 * u, c + (R - 12) * u), color="#dddddd", lw=0.4)
    ax.plot(*c, "k+", ms=12)
    ax.annotate("보내는 보드 쪽 (2 m 앞)", xy=tuple(c + np.array([0, R + 14])), xytext=tuple(c + np.array([0, R + 30])), ha="center",
                fontsize=9, arrowprops=dict(arrowstyle="-|>"))
    pdf.savefig(fig); plt.close(fig)


def main():
    with PdfPages(os.path.join(HERE, "도안.pdf")) as pdf:
        page1(pdf); page2(pdf)
    print(os.path.join(HERE, "도안.pdf"))


if __name__ == "__main__":
    main()
