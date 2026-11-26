"""Wiring diagram of the receiver (hw/배선표.md).  Design values, not checked on the real parts.
python3 hw/anim/make_wiring.py  ->  hw/img/배선도.png"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
FP = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
FB = fm.FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
C = {"9V": "#d62728", "5V": "#e4572e", "3V3": "#ff9f1c", "GND": "#222222", "SIG": "#2f6fd0", "MOT": "#6a3d9a", "USB": "#555555"}

fig, ax = plt.subplots(figsize=(16, 9), dpi=110)
ax.set_xlim(0, 160)
ax.set_ylim(0, 90)
ax.axis("off")


def blk(x, y, w, h, title, sub="", fc="#eef3fb", ec="#2c4a7a"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center", fontproperties=FB, fontsize=13, color="#13203a")
    if sub:
        ax.text(x + w / 2, y + h / 2 - 2.0, sub, ha="center", va="center", fontproperties=FP, fontsize=9.5, color="#33415c",
                linespacing=1.35)


def wire(pts, kind, text=None, at=0.5, dy=1.2, lw=2.4):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=C[kind], lw=lw, solid_capstyle="round")
    if text:
        i = max(0, min(len(pts) - 2, int(at * (len(pts) - 1))))
        (x0, y0), (x1, y1) = pts[i], pts[i + 1]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, text, ha="center", va="bottom", fontproperties=FP, fontsize=9, color=C[kind],
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))


ax.text(2, 87, "수신기 배선 (납땜 없음)  ·  받침 안 → 속 빈 축 → 머리 상자 안", fontproperties=FB, fontsize=17, color="#13203a", va="top")
ax.add_patch(FancyBboxPatch((1, 2), 38, 74, boxstyle="round,pad=0.3,rounding_size=2", fc="#f4f5f7", ec="#9aa3b2", lw=1, ls="--"))
ax.text(20, 74, "받침 (끄덕이지 않음)", ha="center", fontproperties=FB, fontsize=12, color="#555f70")
ax.add_patch(FancyBboxPatch((47, 2), 112, 74, boxstyle="round,pad=0.3,rounding_size=2", fc="#fbfbfd", ec="#9aa3b2", lw=1, ls="--"))
ax.text(103, 74, "머리 상자 안 (끄덕임)", ha="center", fontproperties=FB, fontsize=12, color="#555f70")
ax.add_patch(FancyBboxPatch((40.5, 30), 5, 16, boxstyle="round,pad=0.2,rounding_size=1", fc="#e8e8e8", ec="#666", lw=1))
ax.text(43, 45, "속 빈 축", ha="center", va="bottom", fontproperties=FB, fontsize=10, color="#333")

blk(4, 56, 30, 13, "보조배터리", "PD 20,000 mAh\n받침 배터리 칸")
blk(4, 34, 30, 14, "PD 트리거", "딥 스위치 9 V\n나사 단자 출력 + / −", fc="#fdeeee", ec="#a02020")
blk(4, 6, 30, 18, "모터 (NEMA17)", "웜 감속기 아래, 구동 탑\n2B · 2A · 1A · 1B")
wire([(19, 56), (19, 48)], "USB", "USB-C ↔ USB-C", dy=0.4)

blk(50, 52, 22, 12, "와고 A · B", "A = 9 V, B = GND\n+ 100 µF 35 V", fc="#f0f0f0", ec="#555")
blk(50, 32, 22, 13, "벅 9 → 5 V", "RCNUN 5 V 5 A", fc="#eaf0fb")
blk(50, 8, 22, 15, "와고 C · D", "C = 5 V, D = GND", fc="#f0f0f0", ec="#555")
blk(84, 50, 26, 20, "TMC2209", "EN · STEP · DIR · VDD · GND\nVM · GND(모터)\n2B · 2A · 1A · 1B", fc="#f3ecfa", ec="#6a3d9a")
blk(84, 8, 30, 20, "CM4-IO-BASE-A", "40핀 2 · 4번 = 5 V\n6 · 14번 = GND\nUSB-A → Pico", fc="#eaf6ee", ec="#1d6b3a")
blk(122, 30, 22, 30, "Pico H", "거꾸로, 핀 위\n4·5·6·7·9·10·11·12·14\n→ CTRL1~9\n3 → GND\n15·16·17·18 → TMC\n36 = 3.3 V(Y)\n19 SDA · 20 SCL · 38 GND",
    fc="#eaf6ee", ec="#1d6b3a")
blk(122, 64, 34, 10, "SP16T 제어 헤더 · XH", "CTRL1~9 + GND (3.3 V만) · XH 5 V", fc="#eaf6ee", ec="#1d6b3a")
blk(122, 6, 22, 16, "ADXL345", "VCC · GND\nSDA · SCL\n(I2C 0x53)", fc="#f3ecfa", ec="#6a3d9a")

# 9 V pair
wire([(34, 44), (40, 44), (40.5, 42), (45.5, 42), (47, 58), (50, 58)], "9V", "9 V (빨강 22 AWG)", at=0.0, dy=0.6)
wire([(34, 40), (40.5, 40), (45.5, 40), (47.5, 55), (50, 55)], "GND", "GND (검정)", at=0.2, dy=-2.6)
# motor wires
wire([(34, 15), (39, 15), (39, 31.5), (45.5, 31.5), (47, 27.5), (76, 27.5), (78, 52), (84, 52)], "MOT", "모터 선 4가닥", at=0.55, dy=0.5)
# 9 V to buck / TMC
wire([(61, 52), (61, 45)], "9V", "9 V", dy=0.3)
wire([(72, 61), (84, 61)], "9V", "VM", dy=0.4)
wire([(72, 56), (84, 56)], "GND", "GND", dy=0.4)
# buck -> C/D
wire([(61, 32), (61, 23)], "5V", "5 V", dy=0.3)
wire([(72, 18), (84, 18)], "5V", "2 · 4번", dy=0.4)
wire([(72, 13), (84, 13)], "GND", "6 · 14번", dy=0.4)
wire([(72, 10), (78, 10), (78, 3.5), (119.8, 3.5), (119.8, 66), (122, 66)], "5V", "XH 5 V(+ · −)", at=0.45, dy=0.6)
# USB CM4 -> Pico
wire([(114, 22), (118, 22), (118, 34), (122, 34)], "USB", "마이크로 USB", at=0.0, dy=0.5)
# Pico -> TMC
wire([(122, 52), (110, 52)], "SIG", "EN · STEP · DIR", dy=0.4)
wire([(122, 47), (116, 47), (116, 57), (110, 57)], "3V3", "3.3 V", at=0.0, dy=0.4)
# Pico -> SP16T
wire([(133, 60), (133, 64)], "SIG", None)
ax.text(134.5, 61.6, "CTRL1~9 · GND", fontproperties=FP, fontsize=9, color=C["SIG"])
# Pico -> ADXL
wire([(133, 30), (133, 22)], "SIG", None)
ax.text(134.5, 25.5, "SDA · SCL · 3.3 V · GND", fontproperties=FP, fontsize=9, color=C["SIG"])

for i, (k, t) in enumerate([("9V", "9 V"), ("5V", "5 V"), ("3V3", "3.3 V"), ("GND", "GND"), ("SIG", "신호"), ("MOT", "모터"),
                            ("USB", "USB")]):
    ax.plot([2 + 16 * i, 7 + 16 * i], [79.5, 79.5], color=C[k], lw=3)
    ax.text(8 + 16 * i, 79.5, t, va="center", fontproperties=FP, fontsize=10)
ax.text(159, -1.5, "설계값 · 실물로 확인 전  ·  핀 번호는 hw/배선표.md", ha="right", fontproperties=FP, fontsize=9, color="#667")
out = os.path.join(HERE, "..", "img", "배선도.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
print("wrote", out)
