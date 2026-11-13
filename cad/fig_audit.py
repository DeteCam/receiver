"""전체 재검토에서 모양을 고친 두 곳을 전/후로 보여 주는 그림.   xvfb-run -a python3 fig_audit.py
결과: 그림 폴더의 10_검토에서_고친_곳.png"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import render, paths
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
BG, INK, SUB, BAD, GOOD = "#F3F2EE", "#1E2330", "#5B6170", "#D2452D", "#2E8B57"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"; FONTB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)


def old_blade(length=150.0):
    """고치기 전의 판: 꽂는 자리 위로 가운데 살이 그대로 올라간다"""
    hw = PLATE_W / 2; z0, z1 = BL_Z0, BL_Z0 + length
    x1 = R_FOIL; x0 = x1 - PANEL_T; rib = 3.0
    panel = box(x0, x1, -hw, hw, z0, z1)
    frame = box(x0 - rib, x0 + 0.1, -hw, hw, z0, z1) - box(x0 - rib - 1, x0 + 1, -hw + 2.4, hw - 2.4, z0 + 2.4, z1 - 2.4)
    zc = (BAR_Z0 + BAR_Z1) / 2
    ribs = [box(x0 - rib, x0 + 0.1, -hw, hw, z - 1.0, z + 1.0) for z in np.arange(z0 + 46.0, z1 - 20.0, 46.0)]
    ribs.append(box(x0 - rib, x0 + 0.1, -1.0, 1.0, z0, z1))
    collar = box(COLLAR_X0, x0 + 0.1, -9.0, 9.0, zc - 6.0, zc + 6.0)
    web = [box(x0 - 6.0, x0 + 0.1, -1.2, 1.2, zc - 26.0, zc + 26.0), box(x0 - 6.0, x0 + 0.1, -26.0, 26.0, zc - 1.2, zc + 1.2)]
    m = union([panel, frame, collar] + ribs + web)
    cut = [box(COLLAR_X0 - 1, x0 - 0.6, -TENON_W / 2 - 0.15, TENON_W / 2 + 0.15, BAR_Z0 - 0.15, BAR_Z1 + 0.15), cz(1.7, BAR_Z1, zc + 7.0, COLLAR_X0 + 3.6, 0)]
    return m - union(cut)


H = 760
zc = (BAR_Z0 + BAR_Z1) / 2; sx = COLLAR_X0 + 3.6
screw = cz(3.0, zc + 6.0, zc + 8.6, sx, 0)
drv = cz(3.0, zc + 9.0, zc + 9.0 + 70.0, sx, 0) + cz(8.0, zc + 79.0, zc + 79.0 + 40.0, sx, 0)
cam1 = dict(cam_az=212, cam_el=30, dist=175, focus=(70, 0, 60))
for tag, bl, col, gh in (("a0", old_blade(), BAD, ("drv",)), ("a1", P.blade(150.0), GOOD, ())):
    A = {"blade": (bl, "R", P.COL["body"], True), "bar": (P.rotor_bar(), "R", P.COL["accent"], True), "screw": (screw, "R", "#D8DADF", False), "drv": (drv, "R", col, False)}
    render.render(A, TMP + 'aud_%s.png' % tag, size=(H, H), ground=False, ghost=gh, ghost_opacity=0.5, **cam1)

# 드라이버가 들어가는 길: 머리 몸통의 오른쪽 절반을 위에서 똑바로 내려다본다 (뚜껑 나사 받이 한가운데 높이에서 자른 단면)
mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
cutz = box(-200, 200, -200, 200, -12.0, 200)
half = box(-6.0, 200, -200, 200, -200, 200)
def shell_scene(old):
    keep = (P.LID_SCREWS, dict(P.LID_BOSS_W), P.DRV_X0)
    if old:
        P.LID_SCREWS = [(-51.2, -12.0), (51.2, -12.0), (-51.2, -90.0), (51.2, -90.0)]; P.LID_BOSS_W = {-1: 2.7, 1: 2.7}; P.DRV_X0 = 15.0
    sh = (P.head_shell() - cutz) ^ half
    sxr = P.LID_SCREWS[1][0]; w = P.LID_BOSS_W[1]; x0b = sxr - w
    boss = box(x0b, HX - HW + 0.05, HYL, HYL + 7.0, -15.5, -11.7)
    d = ((db + dc).transform(P.driver_T(0)) - cutz).translate([0, -20.0, 0])
    A = {"shell": (sh, "H", P.COL["body"], True), "pi": ((pb - cutz) ^ half, "H", "#7FA08C", False), "pic": ((pc - cutz) ^ half, "H", "#B9BDC2", False),
         "boss": (boss, "H", BAD if old else GOOD, False), "drv": (d, "H", "#2FB36A", False)}
    pts = {"pi": (BX1, HYL + 3.5, -12.0), "boss": (x0b, HYL + 3.5, -12.0), "b0": (P.DRV_X0, 3.3 - 20.0 - 4.0, -12.0), "b1": (P.DRV_X0 + 35.0, 3.3 - 20.0 - 4.0, -12.0),
           "seat": (P.DRV_X0 + 17.5, 3.3, -12.0), "from": (P.DRV_X0 + 17.5, 3.3 - 20.0, -12.0)}
    gap = x0b - BX1
    P.LID_SCREWS, P.LID_BOSS_W, P.DRV_X0 = keep[0], keep[1], keep[2]
    return A, pts, gap
cam2 = dict(cam_az=-90, cam_el=89, dist=620, fov=5.6, focus=(30, -13, -12))
info = {}
for tag, old in (("b0", True), ("b1", False)):
    A, pts, gap = shell_scene(old)
    pix = render.render(A, TMP + 'aud_%s.png' % tag, size=(H, H), ground=False, label_pts=pts, **cam2)
    info[tag] = (pix, gap)

W = H * 4 + 5 * 30
sheet = Image.new('RGB', (W, H + 330), BG); d = ImageDraw.Draw(sheet)
d.text((40, 30), "전체 재검토에서 모양을 고친 두 곳", font=font(46, True), fill=INK)
caps = [("고치기 전", BAD, "판 고정 나사 바로 위를 가운데 살이 지나간다.\n나사 머리도 드라이버도 살에 걸린다."), ("고친 뒤", GOOD, "살을 두 줄로 갈랐다. 그 사이로 드라이버가\n내려간다. 위쪽 테두리에도 길을 팠다."),
        ("고치기 전", BAD, "뚜껑 나사 받이(빨강)가 길을 막는다.\n보드를 15° 기울여야만 들어갔다."), ("고친 뒤", GOOD, "받이(초록)를 벽 쪽으로 붙이고 얇게 했다.\n보드가 곧게 들어가고 옆이 0.5mm씩 남는다.")]
heads = ["① 판 고정 나사 (판 안쪽에서 본 모습)", "", "② 드라이버 보드가 들어가는 길 (위에서 본 단면)", ""]
for i, tag in enumerate(("a0", "a1", "b0", "b1")):
    x = 30 + i * (H + 30); y = 150
    if heads[i]:
        d.text((x + 6, 100), heads[i], font=font(32, True), fill=INK)
    sheet.paste(Image.open(TMP + 'aud_%s.png' % tag), (x, y))
    d.rectangle([x, y, x + H - 1, y + H - 1], outline="#D9D6CE", width=2)
    t, c, s2 = caps[i]
    d.text((x + 6, y + H + 16), t, font=font(34, True), fill=c)
    for j, l in enumerate(s2.split("\n")):
        d.text((x + 6, y + H + 66 + j * 38), l, font=font(27), fill=SUB)
    if tag in info:                                          # 치수선: 파이 가장자리 ~ 받이 사이(입구), 보드 폭, 들어가는 방향
        pix, gap = info[tag]
        P_ = lambda k: (x + pix[k][0], y + pix[k][1])
        (ax, ay), (bx, by) = P_("pi"), P_("boss"); yy = max(ay, by) + 70
        d.line([ax, ay, ax, yy + 12], fill=INK, width=2); d.line([bx, by, bx, yy + 12], fill=INK, width=2); d.line([ax, yy, bx, yy], fill=INK, width=3)
        d.text(((ax + bx) / 2, yy + 8), "입구 %.1fmm" % gap, font=font(30, True), fill=c, anchor="ma")
        (cx_, cy_), (dx_, dy_) = P_("b0"), P_("b1")
        d.line([cx_, cy_, dx_, dy_], fill="#0E6B3A", width=3)
        d.text(((cx_ + dx_) / 2, cy_ - 8), "보드 35mm", font=font(28, True), fill="#0E6B3A", anchor="mb")
        (fx, fy), (tx, ty) = P_("from"), P_("seat")
        d.line([fx, fy - 40, tx, ty + 30], fill=INK, width=3); d.polygon([(tx, ty + 14), (tx - 9, ty + 34), (tx + 9, ty + 34)], fill=INK)
        d.text((tx + 16, (fy + ty) / 2 - 10), "뒷벽으로", font=font(24), fill=INK, anchor="lm")
sheet.save(OUT + '10_검토에서_고친_곳.png')
print("saved", sheet.size)
