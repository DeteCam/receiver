"""볼트로 조이는 곳 그림 (10_볼트_고정.png).   xvfb-run -a python3 fig_bolts.py
   왼쪽: 회전부를 풀어 놓은 그림 (판 틀 + 알루미늄 판 + 누름쇠 + 접시머리 나사·너트, 동전 균형추 + 마개, 축 고정 나사)
   오른쪽 위: 머리 속 (파이는 M2.5, 드라이버는 M2 나사로 기둥에)     오른쪽 아래: 뒤에서 본 모습 (마커 틀, 핀 막이)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
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


NC = P.COINS[(150.0, AL_T)]; NS = COIN_SLOTS - NC
# 이름표 상자 자리 (그림 가 안의 픽셀). 가리키는 선이 다른 상자를 지나지 않게 둘레에 나눠 둔다: 위쪽 띠(누름쇠 나사·누름쇠·틀 뒷면 너트), 왼쪽(판, 판 틀 나사), 오른쪽(판 틀, 축 너트)
LAB_A = [("top", "M3 × 8 접시머리 2개\n누름쇠 → 판 틀 위 띠", 20, 64), ("clamp", "판 누름쇠: 판 위 변을 누른다", 330, 150),
         ("tnut", "M3 너트 2개: 판 틀 뒷면 위쪽의\n육각 자리에 끼운다", 720, 196),
         ("plate", "알루미늄 판 100 × 150 × %gT\n구멍을 뚫지 않는다" % AL_T, 10, 420),
         ("frame", "판 틀 (한 장)\n판이 들어가는 홈 + 아래 턱", 640, 396),
         ("bolt", "M3 × 6 접시머리 4개\n판 틀 → 판 받이 (판이 덮는다)", 10, 600),
         ("nut", "M3 너트 4개: 판 받이 안쪽의 육각 자리에 끼운다\n(너트가 돌지 않아 드라이버 하나로 조인다)", 300, 960),
         ("setnut", "축 고정 나사의 너트\n위에서 떨어뜨려 넣는다", 935, 428), ("set", "축 고정 나사 M3 × 8\n모터 축의 납작한 면을 누른다", 560, 700),
         ("pod", "균형추 통\n(아래로 열린다)", 1130, 520), ("coins", "100원 동전 %d개\n(통이 꽉 찬다)" % NC if NS == 0 else "100원 동전 %d개" % NC, 1150, 700)]
if NS > 0:
    LAB_A.append(("sp", "동전 채움 조각 %d개\n(합쳐서 늘 12장)" % NS, 1150, 790))
LAB_A += [("cap", "마개", 1165, 870), ("capscrew", "M3 × 8 나사 2개", 1165, 920)]
LAB_B = [("pi", "파이: M2.5 × 6 나사 4개", 20, 70), ("drv", "드라이버 한 장에 M2 × 6 나사 4개\n(세 장에 모두 12개)", 470, 510)]
LAB_C = [("frame", "마커 틀: M3 × 6 나사 4개\n종이를 사이에 끼워 덮는다", 560, 225), ("keep", "핀 막이\nM3 × 6 나사 1개\n(양쪽 팔에 하나씩)", 20, 300)]
# ------------------------------------------------------------------ 가) 회전부를 풀어 놓은 그림 (머리 좌표 그대로: Z 가 판 회전축)
xf = FL_XH - FL_T; zs = M1_ZT + 7.0
EB, EP, EF = 30.0, 58.0, 86.0                                                  # 판 틀, 판, 누름쇠를 바깥으로 빼놓는 거리
A = {"bar": (P.rotor_bar(), "R", BODY, True), "frame": (P.plate_frame().translate([EB, 0, 0]), "R", ACC, True),
     "plate": (P.al_plate(150.0, AL_T).translate([EP, 0, 0]), "R", STEEL, False), "clamp": (P.plate_clamp().translate([EF, 0, 0]), "R", ACC, True)}
screws = []; nuts = []
csk = lambda x0, y, z, L: [M.cylinder(1.7, 1.5, 3.0).rotate([0, 90, 0]).translate([x0 - 1.7, y, z]), cx(1.5, x0 - L, x0 - 1.7, y, z)]      # 접시머리 (머리 윗면이 x0)
for y, z in PLATE_BOLTS:
    screws += csk(FL_X1 + EB + 14.0, y, z, 6.0)
    nuts.append(prism_xy_to_yz(P.hex_cs(5.5), xf - 20.0, xf - 17.6).translate([0, y, z]))
XO = FL_X1 + AL_T + CL_T                                                         # 누름쇠 바깥면 (조립한 자리)
for y, z in P.clamp_screws():
    screws += csk(XO + EF + 16.0, y, z, 8.0)
    nuts.append(prism_xy_to_yz(P.hex_cs(5.5), FL_XH + EB - 14.0, FL_XH + EB - 11.6).translate([0, y, z]))
A["rotor_hw"] = (union(screws + nuts), "R", "#8E949C", False)
zc = POD_Z0 - 16.0                                                               # 동전 더미의 윗면 (통 아래로 빼놓았다)
A["coins"] = (cz(COIN_D / 2, zc - COIN_T * NC, zc, POD_XC, 0), "R", "#B9B3A6", False)
zsp = zc - COIN_T * NC - 9.0                                                     # 채움 조각을 늘어놓기 시작하는 높이
if NS > 0:
    A["spacers"] = (union([P.coin_spacer().translate([POD_XC, 0, zsp - (k + 1) * SPACER_T - k * 0.6]) for k in range(NS)]), "R", ACC, True)
zcap = zsp - NS * (SPACER_T + 0.6) - 12.0
cap = P.coin_cap(NC); cz0 = POD_Z0 - P.coin_gap(NC)                             # 마개 턱 윗면의 원래 높이
A["cap"] = (cap.translate([0, 0, zcap - cz0]), "R", HOT, True)
zsc = zcap - CAP_T - 14.0
A["cap_screws"] = (union([cz(2.8, zsc - 2.4, zsc, POD_XC, s * CAP_EAR_Y) + cz(1.5, zsc, zsc + 8.0, POD_XC, s * CAP_EAR_Y) for s in (1, -1)]), "R", "#8E949C", False)
ys = 11.6 + 16.0
A["set_screw"] = (cy(2.8, ys - 2.4, ys, 0, zs) + cy(1.5, ys - 10.4, ys - 2.4, 0, zs), "R", "#8E949C", False)
A["set_nut"] = (prism_xy_to_yz(P.hex_cs(5.5), 0.0, 2.4).rotate([0, 0, 90]).translate([0, 4.1, BAR_Z1 + 16.0]), "R", "#8E949C", False)
zt = P.clamp_screws()[0][1]
pts = {"plate": (FL_X1 + EP + 0.5, -30.0, 70.0), "clamp": (XO + EF, -20.0, zt + 2.0), "top": (XO + EF + 16.0, 24.0, zt), "tnut": (FL_XH + EB - 12.8, -24.0, zt),
       "frame": (FL_X1 + EB + FR_DEPTH, -51.0, 70.0), "bolt": (FL_X1 + EB + 14.0, 19.0, 8.0), "nut": (xf - 19.0, -19.0, 26.0),
       "coins": (POD_XC, -11.0, zc - 6.0), "cap": (POD_XC, -13.0, zcap - 1.2), "capscrew": (POD_XC, -CAP_EAR_Y, zsc - 1.2),
       "set": (0.0, ys - 1.2, zs), "setnut": (0.0, 5.3, BAR_Z1 + 16.0), "pod": (POD_XC, -13.5, 30.0), "hub": (3.0, -8.0, BAR_Z1)}
if NS > 0:
    pts["sp"] = (POD_XC, -11.0, zsp - 4.0)
pix = render.render(A, TMP + 'bolt_a.png', cam_az=116, cam_el=14, dist=780, focus=(40, 0, 14), size=(1380, 1240), label_pts=pts, ground=False, fov=26.0, ghost=("plate",), ghost_opacity=0.42)
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

W, H = 2360, 1610; m = 40


def wrap(d, text, f, width):
    """글을 그림 폭에 맞게 줄로 나눈다 (띄어쓰기 자리에서)"""
    out, cur = [], ""
    for w_ in text.split(" "):
        t_ = (cur + " " + w_).strip()
        if d.textlength(t_, font=f) > width and cur:
            out.append(cur); cur = w_
        else:
            cur = t_
    return out + [cur]
im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "테이프 없이 전부 나사로: 어디에 무엇을 조이나", font=font(50, True), fill=INK)
import json as _js
_rows = [r for r in _js.load(open(os.path.join(paths.RES, "audit_screws.json"), encoding="utf-8")) if not r.get("extra")]
_cnt = lambda m_, L_, nut=None: sum(1 for r in _rows if r["m"] == m_ and r["length"] == L_ and (nut is None or r["nut"] == nut))
d.text((m, 100), "나사는 모두 %d개다: M3 6mm %d개(그 가운데 %d개는 너트와 함께), 8mm %d개(너트와 %d개), 10mm %d개, M2.5 6mm %d개, M2 6mm %d개. 알루미늄 판에는 구멍을 뚫지 않는다." % (
    len(_rows), _cnt(3.0, 6), _cnt(3.0, 6, True), _cnt(3.0, 8), _cnt(3.0, 8, True), _cnt(3.0, 10), _cnt(2.5, 6), _cnt(2.0, 6)), font=font(28), fill=SUB)
im.paste(pa, (m, 180)); im.paste(pb, (m + 1380 + 20, 180)); im.paste(pc, (m + 1380 + 20, 180 + 600 + 40))
d.text((m + 16, 190), "회전부 (풀어 놓은 그림)", font=font(32, True), fill=INK)
d.text((m + 1380 + 36, 190), "머리 속", font=font(32, True), fill=INK)
d.text((m + 1380 + 36, 180 + 600 + 50), "뒤에서 본 모습", font=font(32, True), fill=INK)
yy = 180 + 1240 + 22
for txt, colr in (("조립 순서: 동전 %d개와 마개 → 판 받이에 너트 4개, 판 틀을 접시머리 6mm 4개로 → 판 아래 변을 아래 턱 밑에 넣고 눕힌다 → 틀 뒷면에 너트 2개, 누름쇠를 접시머리 8mm 2개로 → 막대를 모터 3 축에 꽂고 축 고정 나사." % NC, INK),
                  ("동전 수는 판 무게에 맞춘 것이다 (%gT 판이면 %d개%s). 나사는 걸리는 느낌이 올 때까지만 조인다: 플라스틱에 직접 물리는 나사는 더 조이면 구멍이 뭉개진다." % (AL_T, NC, "로 통이 꽉 찬다" if NS == 0 else ""), SUB)):
    for ln in wrap(d, txt, font(27), W - 2 * m):
        d.text((m, yy), ln, font=font(27), fill=colr); yy += 40
assert yy <= H - 10, ("아래 글이 그림 밖으로 나간다", yy, H)
im.save(OUT + '10_볼트_고정.png'); print("saved", OUT + '10_볼트_고정.png')
for nm, px_ in (("A", pix), ("B", pixb), ("C", pixc)):
    print(nm, {k: (int(v[0]), int(v[1])) for k, v in px_.items()})
