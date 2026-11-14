"""알루미늄 판의 구멍 자리 그림 (11_판_구멍_자리.png).   python3 fig_holes.py
   왼쪽: 판(100 x 150)과 구멍 4개의 치수.   오른쪽: 구멍 뚫기 틀을 판의 귀에 걸고 자리를 찍는 방법."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import paths
from p2 import *
from PIL import Image, ImageDraw, ImageFont

OUT = paths.IMG + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE"
METAL, JIG, HOT, DIM = "#D4D8DD", "#2B3F7A", "#EB6834", "#3F78C9"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
L = 150.0; W_ = PLATE_W
HOLES = [(W_ / 2 + y, z - BL_Z0) for y, z in PLATE_BOLTS]                        # (옆 변에서, 아래 변에서) mm
xs = sorted(set(h[0] for h in HOLES)); zs = sorted(set(h[1] for h in HOLES))
Wd, Hd = 2360, 1560; m = 40; S = 5.4                                              # 1mm = 5.4 픽셀
im = Image.new("RGB", (Wd, Hd), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "알루미늄 판에 구멍 뚫기: 구멍 4개의 자리", font=font(50, True), fill=INK)
d.text((m, 100), "판 100 × 150mm, 두께 1T (0.5T 도 된다). 구멍은 판 한가운데에서 좌우 %gmm, 위아래 %gmm. 위아래·좌우가 대칭이라 판을 어느 쪽으로 돌려 대도 같은 자리가 나온다." % (
    (xs[1] - xs[0]) / 2, (zs[1] - zs[0]) / 2), font=font(28), fill=SUB)


def arrow(p, q, text, off=(0, 0), fs=30, col=DIM, side=None):
    """치수선: 두 점 사이에 선과 양 끝 화살, 가운데에 숫자"""
    d.line([p, q], fill=col, width=3)
    for a, b in ((p, q), (q, p)):
        dx, dy = b[0] - a[0], b[1] - a[1]; n = max(1e-6, (dx * dx + dy * dy) ** 0.5); ux, uy = dx / n, dy / n
        d.polygon([a, (a[0] + ux * 16 - uy * 6, a[1] + uy * 16 + ux * 6), (a[0] + ux * 16 + uy * 6, a[1] + uy * 16 - ux * 6)], fill=col)
    cx, cy = (p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1]
    tw = d.textlength(text, font=font(fs, True)); d.rectangle([cx - tw / 2 - 6, cy - fs * 0.68, cx + tw / 2 + 6, cy + fs * 0.68], fill=BG)
    d.text((cx, cy), text, font=font(fs, True), fill=col, anchor="mm")


# ------------------------------------------------------------------ 왼쪽: 치수
ox, oy = 330, 300                                                                 # 판의 왼쪽 위 귀
X = lambda v: ox + v * S; Y = lambda z: oy + (L - z) * S
d.text((m, 190), "구멍 자리 (판을 세워서 본 모습)", font=font(34, True), fill=INK)
d.rectangle([X(0), Y(L), X(W_), Y(0)], fill=METAL, outline=INK, width=3)
d.line([X(W_ / 2), Y(L) + 10, X(W_ / 2), Y(0) - 10], fill="#9AA0A8", width=1); d.line([X(0) + 10, Y(L / 2), X(W_) - 10, Y(L / 2)], fill="#9AA0A8", width=1)
for hx, hz in HOLES:
    r = 1.7 * S; d.ellipse([X(hx) - r, Y(hz) - r, X(hx) + r, Y(hz) + r], fill=BG, outline=HOT, width=4)
    d.line([X(hx) - 16, Y(hz), X(hx) + 16, Y(hz)], fill=HOT, width=2); d.line([X(hx), Y(hz) - 16, X(hx), Y(hz) + 16], fill=HOT, width=2)
yb = Y(0) + 70                                                                    # 아래쪽 치수 (옆 변에서)
for a, b_, t in ((0, xs[0], "%g" % xs[0]), (xs[0], xs[1], "%g" % (xs[1] - xs[0])), (xs[1], W_, "%g" % (W_ - xs[1]))):
    arrow((X(a), yb), (X(b_), yb), t)
for v in (0, xs[0], xs[1], W_):
    d.line([X(v), Y(0) + 6, X(v), yb + 18], fill=DIM, width=1)
arrow((X(0), yb + 80), (X(W_), yb + 80), "폭 %g" % W_); d.line([X(0), yb, X(0), yb + 98], fill=DIM, width=1); d.line([X(W_), yb, X(W_), yb + 98], fill=DIM, width=1)
xr = X(W_) + 80                                                                   # 오른쪽 치수 (아래 변에서)
for a, b_, t in ((0, zs[0], "%g" % zs[0]), (zs[0], zs[1], "%g" % (zs[1] - zs[0])), (zs[1], L, "%g" % (L - zs[1]))):
    arrow((xr, Y(a)), (xr, Y(b_)), t)
for v in (0, zs[0], zs[1], L):
    d.line([X(W_) + 6, Y(v), xr + 18, Y(v)], fill=DIM, width=1)
arrow((xr + 110, Y(0)), (xr + 110, Y(L)), "높이 %g" % L); d.line([xr, Y(0), xr + 128, Y(0)], fill=DIM, width=1); d.line([xr, Y(L), xr + 128, Y(L)], fill=DIM, width=1)
d.text((X(W_ / 2), Y(L) - 34), "위", font=font(26), fill=SUB, anchor="mm"); d.text((X(W_ / 2), Y(0) + 26), "아래", font=font(26), fill=SUB, anchor="mm")
d.text((X(0) - 250, Y(zs[1]) - 70), "구멍 4개\n지름 3.2~3.5mm\n(M3 볼트가\n지나간다)", font=font(28, True), fill=HOT, spacing=8)
d.line([X(0) - 60, Y(zs[1]) - 6, X(xs[0]) - 14, Y(zs[1])], fill=HOT, width=2)

# ------------------------------------------------------------------ 오른쪽: 틀을 귀에 걸기
ox2, oy2 = 1520, 300
X2 = lambda v: ox2 + v * S; Y2 = lambda z: oy2 + (L - z) * S
d.text((1250, 190), "구멍 뚫기 틀 쓰는 법 (0판에서 같이 뽑는다)", font=font(34, True), fill=INK)
d.rectangle([X2(0), Y2(L), X2(W_), Y2(0)], fill=METAL, outline=INK, width=3)
zt = zs[-1] + 7.0; x1 = xs[-1] + 6.0
# 틀: 아래 변의 띠, 옆 변의 띠, 세로 살 2개, 가로 살 2개 (parts2.drill_jig 와 같은 치수)
for (a0, a1, b0, b1) in [(-2.0, x1, -2.0, 8.0), (-2.0, 8.0, -2.0, zt)] + [(x - 6.0, x + 6.0, 0.0, zt) for x in xs] + [(-2.0, x1, z - 6.0, z + 6.0) for z in zs]:
    d.rectangle([X2(a0), Y2(b1), X2(a1), Y2(b0)], fill=JIG)
d.rectangle([X2(-2.0), Y2(-0.05), X2(x1), Y2(-2.0)], fill="#16213f"); d.rectangle([X2(-2.0), Y2(zt), X2(-0.05), Y2(-2.0)], fill="#16213f")      # 걸림 턱
for hx, hz in HOLES:
    r = 1.1 * S + 2; d.ellipse([X2(hx) - r, Y2(hz) - r, X2(hx) + r, Y2(hz) + r], fill=HOT, outline="#FFFFFF", width=2)


def tag(pt, text, tx, ty):
    lines = text.split("\n"); f = font(27, True); w = max(d.textlength(l, font=f) for l in lines) + 22; h = 38 * len(lines) + 14
    cx = min(max(pt[0], tx), tx + w); cy = min(max(pt[1], ty), ty + h)
    d.line([pt, (cx, cy)], fill=INK, width=2); d.ellipse([pt[0] - 6, pt[1] - 6, pt[0] + 6, pt[1] + 6], fill="#FFFFFF", outline=INK, width=2)
    d.rounded_rectangle([tx, ty, tx + w, ty + h], 8, fill="#FFFFFF", outline="#C9CCD1", width=2)
    for i, l in enumerate(lines):
        d.text((tx + 11, ty + 6 + i * 38), l, font=f, fill=INK)


tag((X2(40), Y2(-1.0)), "아래 턱을 판의 아래 변에 붙인다", X2(40) - 60, Y2(0) + 44)
tag((X2(-1.0), Y2(45)), "옆 턱을 판의\n옆 변에 붙인다", 1250, Y2(45) - 40)
tag((X2(xs[1]), Y2(zs[1])), "구멍 4곳에 송곳이나\n센터 펀치로 자리를 찍는다", X2(22), Y2(132))
d.text((X2(W_ / 2), Y2(L) + 40), "알루미늄 판", font=font(28, True), fill=SUB, anchor="mm")
steps = ["① 틀을 뒤집어 판 위에 얹는다 (턱이 판의 변에 걸린다).", "② 틀을 꼭 누른 채 구멍 4곳에 자리를 찍는다.", "③ 틀을 떼고, 판 밑에 나무를 받치고 3.2~3.5mm 날로 뚫는다.", "④ 구멍 둘레의 거스러미를 줄이나 큰 드릴 날로 다듬는다."]
for i, t in enumerate(steps):
    d.text((1250, Y2(0) + 150 + i * 46), t, font=font(28), fill=INK)
d.text((m, Hd - 62), "높이 200mm 판을 쓸 때도 같은 틀로 아래 변에서 %g, %gmm 자리에 뚫는다 (이때는 위아래가 대칭이 아니다: 틀을 건 쪽이 아래다). 동전은 1T 면 11개, 0.5T 면 5개." % (zs[0], zs[1]), font=font(26), fill=SUB)
im.save(OUT + "11_판_구멍_자리.png"); print("saved", OUT + "11_판_구멍_자리.png", "구멍:", HOLES)
