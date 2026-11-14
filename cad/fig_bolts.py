"""볼트로 조이는 곳 그림 (10_볼트_고정.png).   xvfb-run -a python3 fig_bolts.py
   왼쪽: 회전부를 풀어 놓은 그림 (알루미늄 판 + 볼트·너트, 동전 균형추 + 마개, 축 고정 나사)
   오른쪽 위: 머리 속 (파이와 드라이버를 기둥에 M2.5 나사로)     오른쪽 아래: 뒤에서 본 모습 (마커 틀, 핀 막이)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import render, paths
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont
from fig_change_view import HP_factory

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE"
HOT, STEEL, BODY, ACC = "#EB6834", "#C9CED4", P.COL["body"], P.COL["accent"]
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
HP = HP_factory(Y_T, S_T, H_T)


def tags(im, pix, items, fs=23):
    """그림 위에 이름표: (이름, 글, 이름표 상자의 왼쪽 위 x, y). 상자는 그림 밖으로 나가지 않게 민다"""
    d = ImageDraw.Draw(im); f = font(fs, True); W_, H_ = im.size
    for key, text, tx, ty in items:
        ax, ay = pix[key]; lines = text.split("\n")
        w = max(d.textlength(l, font=f) for l in lines) + 20; hh = int(fs * 1.34) * len(lines) + 12
        x0 = min(max(4, tx), W_ - w - 4); y0 = min(max(4, ty), H_ - hh - 4)
        cx_, cy_ = min(max(ax, x0), x0 + w), min(max(ay, y0), y0 + hh)              # 상자에서 가리키는 점과 가장 가까운 곳
        d.line([(ax, ay), (cx_, cy_)], fill=INK, width=2)
        d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill="#FFFFFF", outline=INK, width=2)
        d.rounded_rectangle([x0, y0, x0 + w, y0 + hh], 8, fill="#FFFFFF", outline="#C9CCD1", width=2)
        for i, l in enumerate(lines):
            d.text((x0 + 10, y0 + 5 + i * int(fs * 1.34)), l, font=f, fill=INK)
    return im


LAB_A = [("plate", "알루미늄 판 100 × 150 × 1T\n구멍 4개는 직접 뚫는다 (지름 3.2~3.5)", 250, 150), ("bolt", "M3 × 8 볼트 4개\n판 바깥에서\n꽂는다", 10, 600),
         ("nut", "M3 너트 4개: 판 받이 안쪽의 육각 자리에 끼운다\n(너트가 돌지 않아 드라이버 하나로 조인다)", 300, 960),
         ("setnut", "축 고정 나사의 너트\n위에서 떨어뜨려 넣는다", 640, 250), ("set", "축 고정 나사 M3 × 8\n모터 축의 납작한 면을 누른다", 610, 650),
         ("pod", "균형추 통\n(아래로 열린다)", 1130, 400), ("coins", "100원 동전 8개\n(0.5T 판이면 4개)", 1165, 690), ("sp", "채움 조각 4개\n(합쳐서 늘 12장)", 1165, 775),
         ("cap", "마개", 1165, 852), ("capscrew", "M3 × 8 나사 2개", 1165, 905)]
LAB_B = [("pi", "파이: M2.5 × 6 나사 4개", 20, 70), ("drv", "드라이버 한 장에 M2.5 × 6 나사 4개\n(세 장에 모두 12개)", 470, 510)]
LAB_C = [("frame", "마커 틀: M3 × 6 나사 4개\n종이를 사이에 끼워 덮는다", 560, 225), ("keep", "핀 막이\nM3 × 6 나사 1개\n(양쪽 팔에 하나씩)", 20, 300)]
# ------------------------------------------------------------------ 가) 회전부를 풀어 놓은 그림 (머리 좌표 그대로: Z 가 판 회전축)
xf = FL_X1 - FL_T; zs = M1_ZT + 7.0; EX = 34.0                                  # 판을 바깥으로 빼놓는 거리
A = {"bar": (P.rotor_bar(), "R", BODY, True), "plate": (P.al_plate(150.0, AL_T).translate([EX, 0, 0]), "R", STEEL, False)}
bolts = []; nuts = []
for y, z in PLATE_BOLTS:
    x0 = FL_X1 + AL_T + EX + 30.0
    bolts += [cx(2.8, x0, x0 + 2.4, y, z), cx(1.5, x0 - 8.0, x0, y, z)]
    nuts.append(prism_xy_to_yz(P.hex_cs(5.5), xf - 20.0, xf - 17.6).translate([0, y, z]))
A["rotor_hw"] = (union(bolts + nuts), "R", "#8E949C", False)
zc = POD_Z0 - 16.0                                                               # 동전 더미의 윗면 (통 아래로 빼놓았다)
A["coins"] = (cz(COIN_D / 2, zc - COIN_T * 8, zc, POD_XC, 0), "R", "#B9B3A6", False)
sp = [P.coin_spacer().translate([POD_XC, 0, zc - 12.0 - 9.0 - (k + 1) * SPACER_T - k * 0.6]) for k in range(4)]
A["spacers"] = (union(sp), "R", ACC, True)
zcap = zc - 12.0 - 9.0 - 4 * (SPACER_T + 0.6) - 12.0
cap = P.coin_cap(8); cz0 = POD_Z0 - P.coin_gap(8)                               # 마개 턱 윗면의 원래 높이
A["cap"] = (cap.translate([0, 0, zcap - cz0]), "R", HOT, True)
zsc = zcap - CAP_T - 14.0
A["cap_screws"] = (union([cz(2.8, zsc - 2.4, zsc, POD_XC, s * CAP_EAR_Y) + cz(1.5, zsc, zsc + 8.0, POD_XC, s * CAP_EAR_Y) for s in (1, -1)]), "R", "#8E949C", False)
ys = 11.6 + 16.0
A["set_screw"] = (cy(2.8, ys - 2.4, ys, 0, zs) + cy(1.5, ys - 10.4, ys - 2.4, 0, zs), "R", "#8E949C", False)
A["set_nut"] = (prism_xy_to_yz(P.hex_cs(5.5), 0.0, 2.4).rotate([0, 0, 90]).translate([0, 4.1, BAR_Z1 + 16.0]), "R", "#8E949C", False)
pts = {"plate": (FL_X1 + EX + 0.5, -34.0, 62.0), "bolt": (FL_X1 + AL_T + EX + 31.0, 19.0, 8.0), "nut": (xf - 19.0, -19.0, 26.0), "flange": (FL_X1, 22.0, 30.0),
       "coins": (POD_XC, -11.0, zc - 6.0), "sp": (POD_XC, -11.0, zc - 12.0 - 9.0 - 4.0), "cap": (POD_XC, -13.0, zcap - 1.2), "capscrew": (POD_XC, -CAP_EAR_Y, zsc - 1.2),
       "set": (0.0, ys - 1.2, zs), "setnut": (0.0, 5.3, BAR_Z1 + 16.0), "pod": (POD_XC, -13.5, 30.0), "hub": (3.0, -8.0, BAR_Z1)}
pix = render.render(A, TMP + 'bolt_a.png', cam_az=116, cam_el=14, dist=660, focus=(16, 0, 12), size=(1380, 1240), label_pts=pts, ground=False, fov=26.0, ghost=("plate",), ghost_opacity=0.42)
pa = tags(Image.open(TMP + 'bolt_a.png').convert("RGB"), pix, LAB_A)

# ------------------------------------------------------------------ 나) 머리 속: 파이와 드라이버의 나사 (뚜껑을 떼고 앞에서)
mb, mc = v1.motor_body_dummy(); b, cc = v1.pi_dummy(); db, dc = v1.driver_dummy()
B = {"head_shell": (P.head_shell(), "H", BODY, True), "pi": (b, "H", P.COL["pcb"], False), "pi_c": (cc, "H", "#9A9FA6", False)}
heads = []
for k in range(3):
    T = P.driver_T(k); bd = db.transform(T); B["d%d" % k] = (bd, "H", "#3B8F5E", False); B["dc%d" % k] = (dc.transform(T), "H", "#22262B", False)
    yb = float(np.asarray(bd.to_mesh().vert_properties)[:, 1].min())
    heads += [cy(2.3, yb - 1.7, yb, sx, sz) for sx, sz in P.driver_holes(k)]
ypi = float(np.asarray(b.to_mesh().vert_properties)[:, 1].min())            # 파이 기판의 앞면
heads += [cy(2.3, ypi - 1.7, ypi, BX0 + bq, BZ0 + a) for a, bq in PI_HOLES]
B["screws"] = (union(heads), "H", HOT, False)
d0 = P.driver_holes(0)[0]; d2 = P.driver_holes(2)[1]
ptsb = {"pi": (BX0 + PI_HOLES[2][1], ypi - 1.7, BZ0 + PI_HOLES[2][0]), "drv": (d0[0], -9.0, d0[1]), "drv2": (d2[0], -9.0, d2[1])}
pixb = render.render(B, TMP + 'bolt_b.png', cam_az=-80, cam_el=8, dist=330, focus=(2, 0, -46), size=(900, 600), label_pts=ptsb, ground=False, fov=26.0)
pb = tags(Image.open(TMP + 'bolt_b.png').convert("RGB"), pixb, LAB_B)

# ------------------------------------------------------------------ 다) 뒤에서: 마커 틀과 핀 막이 (재는 자세)
yaw, pitch = 0.0, TILT3
C = P.assembly(yaw, pitch, 0.0)
for k in ("marker_frame", "keeper_r", "keeper_l"):
    C[k] = (C[k][0], C[k][1], HOT, True)
C["head_hw"] = (P.pose_head(P.head_hw(), yaw, pitch), "H", "#8E949C", False)
kx, kz = P.keeper_xz()
ms = P.marker_screws()[0]
ptsc = {"frame": HP((ms[0], HY1 + MK_FRAME["t"] + 2.4, ms[1]), yaw, pitch), "keep": np.array([kx, -(ARM_Y1 + 2.0), H_T + kz]), "paper": HP((-27.0, HY1 + 0.5, -76.0), yaw, pitch)}
pixc = render.render(C, TMP + 'bolt_c.png', cam_az=205, cam_el=38, dist=560, focus=tuple(HP((0, 0, -60), yaw, pitch)), size=(900, 600), label_pts=ptsc, ground=False, fov=26.0)
pixc["keep"] = (303, 289)                                                       # 왼쪽 팔의 핀 막이 (그림에서 읽은 자리)
pc = tags(Image.open(TMP + 'bolt_c.png').convert("RGB"), pixc, LAB_C)

W, H = 2360, 1560; m = 40
im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "테이프 없이 전부 나사로: 어디에 무엇을 조이나", font=font(50, True), fill=INK)
d.text((m, 100), "나사는 모두 53개다: M3 6mm 18개, 8mm 11개(그 가운데 5개는 너트와 함께), 10mm 8개, M2.5 6mm 16개. 주황색 = 이번에 새로 생긴 고정 부품.", font=font(28), fill=SUB)
im.paste(pa, (m, 180)); im.paste(pb, (m + 1380 + 20, 180)); im.paste(pc, (m + 1380 + 20, 180 + 600 + 40))
d.text((m + 16, 190), "회전부 (풀어 놓은 그림)", font=font(32, True), fill=INK)
d.text((m + 1380 + 36, 190), "머리 속", font=font(32, True), fill=INK)
d.text((m + 1380 + 36, 180 + 600 + 50), "뒤에서 본 모습", font=font(32, True), fill=INK)
d.text((m, 180 + 1240 + 24), "조립 순서: 판에 구멍을 뚫는다(구멍 뚫기 틀) → 판을 판 받이에 볼트로 → 통을 위로 향하게 뒤집어 동전, 채움 조각, 마개 → 막대를 모터 3 축에 꽂고 축 고정 나사.", font=font(27), fill=INK)
d.text((m, 180 + 1240 + 66), "동전 수는 판 무게에 맞춘 것이다 (판 5g 마다 동전 1개쯤). 나사는 걸리는 느낌이 올 때까지만 조인다: 플라스틱에 직접 물리는 나사는 더 조이면 구멍이 뭉개진다.", font=font(27), fill=SUB)
im.save(OUT + '10_볼트_고정.png'); print("saved", OUT + '10_볼트_고정.png')
for nm, px_ in (("A", pix), ("B", pixb), ("C", pixc)):
    print(nm, {k: (int(v[0]), int(v[1])) for k, v in px_.items()})
