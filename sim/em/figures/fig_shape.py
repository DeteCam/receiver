"""판 모양 비교 그림 (31_판_모양.png).   python3 fig_shape.py
shape_summary.json (shape_eval.py --report) 을 읽는다: 판 모양마다 안테나 모형 4가지의 평균. 왼쪽 = 곡선 전체를 맞추는 읽기, 오른쪽 = 지금 장치의 '가장 깊은 골' 읽기."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from shape_eval import LABEL

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fm.fontManager.addfont(FONT); plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
INK, SUB, BG, GRID = "#22262B", "#5B6470", "#F4F2EE", "#D9D5CC"
C_NORM, C_BAD, C_BASE, HOT = "#8FA6E0", "#22305E", "#B9BEC6", "#EB6834"
S = json.load(open(os.path.join(empaths.STUDIES, "shape_summary.json"), encoding="utf-8"))
GROUPS = [("지금", ["w100_R80"]),
          ("구멍·틈", ["split10", "split20", "split40", "mesh10", "mesh20", "slotv60", "sloth60"]),
          ("접기·휨", ["veein20", "veeout20", "veeout40", "arc100", "flange15", "wings15"]),
          ("크기 (틀·끄덕임 범위에 안 들어감)", ["w100h200_R80", "w100_R100", "vee150_R80", "w150_R80"])]
SHORT = {"w100_R80": "지금: 평판 100×150, 팔 8cm", "split10": "가운데 세로 틈 10mm", "split20": "가운데 세로 틈 20mm", "split40": "가운데 세로 틈 40mm",
         "mesh10": "구멍 많은 판 (칸 10mm)", "mesh20": "구멍 많은 판 (칸 20mm)", "slotv60": "가운데 세로 슬롯 60mm", "sloth60": "가운데 가로 슬롯 60mm",
         "veein20": "V로 접기, 양옆 안쪽 20°", "veeout20": "V로 접기, 양옆 바깥 20°", "veeout40": "V로 접기, 양옆 바깥 40°", "arc100": "곡면 (축을 따라 휨)",
         "flange15": "양 끝 15mm 바깥으로 꺾기", "wings15": "양 끝 15mm 안쪽으로 꺾기", "w100h200_R80": "높이 200 (DiffLoc+ 크기)", "w100_R100": "팔 10cm",
         "vee150_R80": "폭 150, V 30°", "w150_R80": "폭 150"}


def icon(ax, pl):
    """위에서 본 판 (가운데 점 = 안테나, 위쪽 = 판이 있는 쪽)"""
    R, w = 80.0, 100.0
    if pl == "w100_R100":
        R = 100.0
    if pl in ("vee150_R80", "w150_R80"):
        w = 150.0
    ax.plot([0], [0], "o", ms=3.0, color=INK)
    lw = 2.2; c = C_BAD
    h = w / 2
    if pl.startswith("vee") or pl.startswith("veeout") or pl.startswith("veein"):
        a = np.radians({"veein20": -20, "veeout20": 20, "veeout40": 40, "vee150_R80": 30}[pl])
        for s in (1, -1):
            ax.plot([0, s * np.cos(a) * h], [R, R + np.sin(a) * h], color=c, lw=lw, solid_capstyle="butt")
    elif pl == "arc100":
        t = np.linspace(-w / 2 / R, w / 2 / R, 40); ax.plot(R * np.sin(t), R * np.cos(t), color=c, lw=lw)
    elif pl.startswith("split"):
        g = float(pl[5:]) / 2
        for s in (1, -1):
            ax.plot([s * g, s * h], [R, R], color=c, lw=lw, solid_capstyle="butt")
    elif pl.startswith("mesh"):
        ax.plot([-h, h], [R, R], color=c, lw=lw, ls=(0, (1.2, 1.0)))
    else:
        ax.plot([-h, h], [R, R], color=c, lw=lw, solid_capstyle="butt")
        if pl in ("flange15", "wings15"):
            d = 15.0 if pl == "flange15" else -15.0
            for s in (1, -1):
                ax.plot([s * h, s * h], [R, R + d], color=c, lw=lw)
        if pl in ("slotv60", "sloth60"):
            ax.plot([-6, 6], [R, R], color=BG, lw=lw + 0.5)
    ax.set_xlim(-82, 82); ax.set_ylim(-12, 130); ax.set_aspect("equal"); ax.axis("off")


rows = [(g, pl) for g, pls in GROUPS for pl in pls if pl in S and "보통|맞춤" in S[pl]]
n = len(rows) + len(GROUPS)
fig = plt.figure(figsize=(16.5, 0.52 * n + 2.6), dpi=150, facecolor=BG)
fig.text(0.02, 1 - 0.32 / (0.52 * n + 2.6) * 2.0, "알루미늄 판 모양을 바꾸면 방향이 더 정확해지나 (전파 계산 NEC2, 안테나 모형 4가지 평균, 카메라 120곳)", fontsize=17, weight="bold", color=INK)
top = 0.90; bot = 0.10; rowh = (top - bot) / n
axL = fig.add_axes([0.30, bot, 0.40, top - bot]); axR = fig.add_axes([0.80, bot, 0.17, top - bot])
for ax in (axL, axR):
    ax.set_facecolor(BG); ax.set_ylim(n - 0.5, -0.5); ax.tick_params(left=False, labelleft=False)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
base = S["w100_R80"]
y = 0
for g, pls in GROUPS:
    fig.text(0.02, top - (y + 0.5) * rowh, g, fontsize=12.5, weight="bold", color=SUB, va="center")
    y += 1
    for pl in pls:
        if pl not in S or "보통|맞춤" not in S[pl]:
            continue
        r = S[pl]; nm, bd = r["보통|맞춤"]["med"], r["나쁨|맞춤"]["med"]; o30 = r["나쁨|맞춤"]["over30"]; tg = r["보통|골"]["med"]
        yc = top - (y + 0.5) * rowh
        ia = fig.add_axes([0.02, yc - rowh * 0.48, 0.045, rowh * 0.96]); icon(ia, pl)
        fig.text(0.072, yc, SHORT[pl], fontsize=11.5, color=INK, va="center", weight="bold" if pl in ("w100_R80", "veeout20", "veeout40") else "normal")
        axL.barh(y - 0.17, nm, height=0.32, color=C_NORM); axL.barh(y + 0.17, bd, height=0.32, color=C_BAD)
        axL.text(bd + 0.2, y + 0.17, "%.1f°  (30° 넘게 틀림 %.0f%%)" % (bd, o30), fontsize=9.5, va="center", color=INK)
        axL.text(nm + 0.2, y - 0.17, "%.1f°" % nm, fontsize=9, va="center", color=SUB)
        axR.barh(y, tg, height=0.5, color=C_BASE if pl != "w100_R80" else HOT)
        axR.text(tg + 1.5, y, "%.0f°" % tg, fontsize=9, va="center", color=INK)
        y += 1
axL.axvline(base["나쁨|맞춤"]["med"], color=HOT, lw=1.2, ls="--"); axL.axvline(base["보통|맞춤"]["med"], color=HOT, lw=1.0, ls=":")
axL.set_xlim(0, 21); axL.set_xlabel("방향 오차 가운데값 (도)  ·  곡선 전체를 맞추는 읽기", fontsize=11, color=INK)
axL.text(0.0, -0.9, "■ 보통 방   ", color=C_NORM, fontsize=11, transform=axL.get_yaxis_transform(), va="center")
axL.text(0.22, -0.9, "■ 반사가 심한 방", color=C_BAD, fontsize=11, transform=axL.get_yaxis_transform(), va="center")
axL.text(0.62, -0.9, "--- 지금 판 (반사 심한 방)", color=HOT, fontsize=10.5, transform=axL.get_yaxis_transform(), va="center")
axR.set_xlim(0, 130); axR.set_xlabel("지금 장치의 '가장 깊은 골' 읽기\n보통 방 오차 가운데값 (도)", fontsize=10.5, color=INK)
fig.text(0.02, 0.025, "읽는 법: 판을 돌리며 받은 세기 곡선 전체를 판 모양마다의 기준 곡선과 맞추면 오차가 몇 도로 줄지만, 가장 깊은 골 하나만 읽으면 어느 판이든 크게 틀린다 (전파 계산에서는 판 가운데 뒤가 도로 밝아지는 '골 두 개' 모양이 나온다).\n"
         "판 모양 가운데서는 V로 바깥으로 접은 판만 반사가 심한 방에서 나아졌다 (12.2° → 8.2~8.8°). 그런데 접은 끝이 판 회전축에서 108~118mm 까지 나가 깊이 숙여 돌릴 때 요크에 닿는다. 구멍·틈·곡면·꺾은 끝은 나아지지 않았다.",
         fontsize=10.5, color=SUB, va="bottom")
out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(empaths.IMG, "31_판_모양.png")          # 저장소에서는 docs/images
fig.savefig(out, facecolor=BG); print("saved", out)
