"""설명 그림을 만든다 (미드나잇 블루).   xvfb-run -a python3 figs.py [번호 ...]
  1 전체 모습   2 세 자세   3 속 보기   4 출력 부품   5 조립 순서   6 배선   7 출력판 나누기
결과: 그림 폴더의 01_*.png ... 08_*.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import parts2 as P
import parts as v1
import render, export2 as E
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont

import paths
ROOT = paths.RES + '/'
OUT = paths.IMG + '/'
TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#8A929C", "#F4F2EE"
HI = "#F08A3C"                     # 새로 붙이는 부품 표시색 (조립 순서 그림)
WANT = set(sys.argv[1:]) or set("1234567")
S = 1600


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)


def rz(p, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def YP(p, yaw=0.0):
    """요크 좌표의 점 -> 세계 좌표"""
    return rz(np.asarray(p, float), yaw)


def HP(p, yaw=0.0, pitch=0.0, spin=None):
    """머리 좌표의 점 -> 세계 좌표 (parts2.pose_head 와 같은 순서)"""
    p = np.asarray(p, float)
    if spin is not None:
        p = rz(p, spin)
    x, y, z = p[0], p[1] - Y_T, p[2] - S_T
    x, y = -y, x
    pr = np.radians(pitch)
    x, z = x * np.cos(pr) + z * np.sin(pr), -x * np.sin(pr) + z * np.cos(pr)
    return rz(np.array([x, y, z + H_T]), yaw)


def tube(pts, r=1.6):
    """점들을 잇는 굵은 선 (전선 표시용)"""
    out = []
    pts = [np.asarray(p, float) for p in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        z = d / L
        x = np.cross([0, 0, 1.0], z)
        x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-6 else x / np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.column_stack([x, y, z, a])
        out.append(M.cylinder(L, r, r, 16).transform(T))
        out.append(M.sphere(r, 16).translate(list(b)))
    return union(out)


def annotate(img_path, out_path, pix, labels, title=None, sub=None, pad_top=0, fs=0.0205, pad_bottom=0, foot=None, pad_side=0):
    """labels: [(키, 글, 'L'|'R', y픽셀)]. pad_side 만큼 양옆에 여백을 붙여 글이 그림과 겹치지 않게 한다. 글 크기는 원래 그림 폭 기준."""
    im = Image.open(img_path).convert('RGB')
    W0, H = im.size
    if pad_top or pad_bottom or pad_side:
        canvas = Image.new('RGB', (W0 + 2 * pad_side, H + pad_top + pad_bottom), BG); canvas.paste(im, (pad_side, pad_top)); im = canvas
    W = im.size[0]
    d = ImageDraw.Draw(im)
    f = font(W0 * fs); fb = font(W0 * fs, True)
    m = int(W0 * 0.03)
    for key, text, side, ty in labels:
        ax, ay = pix[key]; ax += pad_side; ay += pad_top; ty += pad_top
        lines = text.split('\n')
        w = max(d.textlength(l, font=fb if i == 0 else f) for i, l in enumerate(lines))
        lh = int(W0 * fs * 1.36)
        tx = m if side == 'L' else int(W - m - w)
        ex = tx + w + 10 if side == 'L' else tx - 10
        d.line([(ex, ty + lh * 0.55), (ax, ay)], fill=LINE, width=max(2, W0 // 700))
        r = max(4, W0 // 260)
        d.ellipse([ax - r, ay - r, ax + r, ay + r], fill="#FFFFFF", outline=INK, width=max(2, W0 // 700))
        for i, l in enumerate(lines):
            d.text((tx, ty + i * lh), l, font=fb if i == 0 else f, fill=INK if i == 0 else SUB)
    if title:
        d.text((m, int(W0 * 0.026)), title, font=font(W0 * 0.033, True), fill=INK)
    if sub:
        for i, l in enumerate(sub.split('\n')):
            d.text((m, int(W0 * 0.074) + i * int(W0 * 0.03)), l, font=font(W0 * 0.0205), fill=SUB)
    if foot:
        d.text((m, im.size[1] - int(W0 * 0.045)), foot, font=font(W0 * 0.019), fill=SUB)
    im.save(out_path)


def auto_labels(pix, texts, width, top, bottom, force=None):
    """점이 그림의 왼쪽 절반에 있으면 왼쪽 이름표, 아니면 오른쪽. 같은 쪽에서는 점의 높이 순서대로 고르게 놓는다 (선이 서로 엇갈리지 않게)."""
    force = force or {}
    side = {k: force.get(k, 'L' if pix[k][0] < width / 2 else 'R') for k in texts}
    out = []
    for sd in ('L', 'R'):
        ks = sorted([k for k in texts if side[k] == sd], key=lambda k: pix[k][1])
        for i, k in enumerate(ks):
            y = top if len(ks) == 1 else top + (bottom - top) * i / (len(ks) - 1)
            out.append((k, texts[k], sd, int(y)))
    return out


def vis_check(A, pts, cam_az, cam_el, dist, focus, hide=(), tag=""):
    """이름표가 가리키는 점이 카메라에서 실제로 보이는지 확인한다 (다른 부품에 가려 있으면 알려 준다)."""
    import pyvista as pv
    meshes = [render.to_pv(m) for k, (m, g, c, pr) in A.items() if k not in hide]
    allm = meshes[0].merge(meshes[1:]) if len(meshes) > 1 else meshes[0]
    az, el = np.radians(cam_az), np.radians(cam_el)
    cam = np.array(focus, float) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    bad = []
    for k, p in pts.items():
        p = np.asarray(p, float)
        hit, _ = allm.ray_trace(cam, p, first_point=True)
        if len(hit) and np.linalg.norm(p - cam) - np.linalg.norm(np.asarray(hit) - cam) > 4.0:
            bad.append((k, round(float(np.linalg.norm(p - cam) - np.linalg.norm(np.asarray(hit) - cam)), 1)))
    print("[%s] 가려진 이름표:" % tag, bad or "없음")
    return bad


def envelope(pitch, length=150.0):
    """그 자세에서 판이 한 바퀴 돌 때의 가장 높은 점과 가장 먼 점 (mm)"""
    zmax = 0; rmax = 0
    R = P.rotor_set(length)
    for spin in np.arange(0, 360, 10.0):
        b = P.pose_head(R, 0, pitch, spin).bounding_box()
        zmax = max(zmax, b[5]); rmax = max(rmax, np.hypot(max(abs(b[0]), abs(b[3])), max(abs(b[1]), abs(b[4]))))
    return zmax, rmax


# ---------------------------------------------------------------- 1 전체 모습
if "1" in WANT:
    yaw, pitch, spin = 0.0, TILT3, 30.0
    A = P.assembly(yaw, pitch, spin)
    px, pz = P.pin2_xz()
    a_sec = np.radians(166.0)
    pts = {
        "plate": HP((FL_X1 + 1.0, -15, 62), yaw, pitch, spin), "pod": HP((POD_XC, -13.5, 30), yaw, pitch, spin), "bolt": HP((FL_X1 + AL_T + CL_T, P.clamp_screws()[0][0], P.clamp_screws()[0][1]), yaw, pitch, spin),
        "crown": HP((33.1, -15.4, 21), yaw, pitch), "head": HP((HX, -14, -22), yaw, pitch),
        "sector": YP((66 * np.cos(a_sec), SEC_Y1, H_T + 66 * np.sin(a_sec)), yaw), "m2": YP((px, ARM_Y1 + 19.5, pz + M_OFFSET), yaw),
        "arm": YP((-40, ARM_Y1, 50), yaw), "box": YP((WING_X1, -22.0, 72.0), yaw), "base": np.array([68.0, 16.0, 14.0]),
        "pin": YP((0, ARM_Y1 + 2, H_T + 3), yaw),
    }
    cam = dict(cam_az=33, cam_el=16, dist=900, focus=(0, 0, 150))
    vis_check(A, pts, tag="1", **cam)
    pix = render.render(A, TMP + 'raw1.png', size=(S, S), label_pts=pts, ground=False, **cam)
    Image.open(TMP + 'raw1.png').crop((0, 150, S, S - 150)).save(TMP + 'raw1c.png')
    pix = {k: (v[0], v[1] - 150) for k, v in pix.items()}
    annotate(TMP + 'raw1c.png', OUT + '01_전체_모습.png', pix, [
        ("crown", "모터 3 (판 회전)\n머리 위 모터 자리에 들어간다", 'L', 190),
        ("pod", "회전 막대 + 동전 균형추\n100원 동전 %d개로 판 무게를 맞춘다" % P.COINS[(150.0, AL_T)], 'L', 420),
        ("head", "머리\n라즈베리파이 + 드라이버 3장", 'L', 660),
        ("box", "전선 상자\n모터 1(받침 회전, 4 : 1)과 전선이 이 안에 있다", 'L', 900),
        ("base", "받침 (지름 14cm)\n배터리가 들어가고 움직이지 않는다", 'L', 1110),
        ("plate", "알루미늄 판 100×150 (%gT) + 판 틀\n판 틀 홈에 넣고 아래 턱과 누름쇠로 잡는다 (구멍 없음)" % AL_T, 'R', 190),
        ("sector", "부채꼴 기어\n머리에 붙어 같이 끄덕인다", 'R', 440),
        ("pin", "끄덕임 축 핀\n각도 바늘이 달려 있다", 'R', 660),
        ("m2", "모터 2 (끄덕임)\n작은 기어 10T, 11.2 : 1", 'R', 880),
        ("arm", "요크\n팔 2개 + 회전판 + 전선 상자, 한 덩어리", 'R', 1100),
    ], title="DeteCam 수신기 v2 (관절 3개)", sub="머리를 54.7° 기울인 재는 자세 · 세웠을 때 높이 28.5cm · 전선은 상자와 가운데 통 속으로 지나간다 · 출력 부품은 모두 미드나잇 블루", pad_side=330, pad_top=50)

# ---------------------------------------------------------------- 2 세 자세
if "2" in WANT:
    h = 1000
    poses = [(0.0, "세움 0°", "옮길 때, 수평 방향만 볼 때"), (TILT3, "기울임 54.7°", "재는 자세. 받침을 120°씩 세 번 (좁은 방은 60°씩 여섯 번)"), (85.0, "깊이 숙임 85°", "바로 위·아래를 볼 때. 모터로 갈 수 있는 끝")]
    sheet = Image.new('RGB', (h * 3, h + 250), BG); d = ImageDraw.Draw(sheet)
    d.text((48, 36), "머리가 끄덕이는 범위", font=font(50, True), fill=INK)
    _J = json.load(open(ROOT + 'audit_sweep.json', encoding='utf-8'))
    SWP = {k: min(_J[b_][k] for b_ in ("받침 S · 판 150", "받침 L · 판 150") if b_ in _J) for k in ("gap_tilt", "gap_soft")}
    d.text((48, 108), "끄덕임 −5° ~ 85° (87°에서 멈춤 나사에 닿는다), 받침 회전 ±174°, 판은 계속 돈다. 판과 몸체 사이 틈은 재는 자세에서 %.0fmm, 85°에서 %.0fmm 남는다." % (SWP["gap_tilt"], SWP["gap_soft"]), font=font(30), fill=SUB)
    for i, (p, t1, t2) in enumerate(poses):
        A = P.assembly(20.0, p, 110.0)
        render.render(A, TMP + 'raw2_%d.png' % i, cam_az=-62, cam_el=10, size=(h, h), dist=900, focus=(0, 0, 150), ground_z=-0.4)
        sheet.paste(Image.open(TMP + 'raw2_%d.png' % i).crop((0, 110, h, h)), (i * h, 170))
        zmax, rmax = envelope(p)
        d.text((i * h + 48, h + 80), t1, font=font(44, True), fill=INK)
        d.text((i * h + 48, h + 142), t2, font=font(29), fill=SUB)
        d.text((i * h + 48, h + 186), "가장 높은 곳 %.0fmm · 가장 먼 곳 %.0fmm" % (zmax, rmax), font=font(29), fill=SUB)
    sheet.save(OUT + '02_세_자세.png')

# ---------------------------------------------------------------- 3 속 보기
if "3" in WANT:
    # (가) 머리 속: 뚜껑을 떼고 앞에서 본 모습. 머리 좌표 그대로 그린다.
    mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
    C = P.COL
    A = {"shell": (P.head_shell(), "H", C["body"], True), "crown": (P.crown(), "H", C["accent"], True),
         "pi_board": (pb, "H", C["pcb"], False), "pi_parts": (pc, "H", C["metal"], False), "plug": (P.usb_plug_dummy(), "H", C["dark"], False),
         "m3": (v1._m1(mb), "H", C["metal"], False), "m3c": (v1._m1(mc), "H", C["cap"], False), "ant": (v1.antenna_marker(), "H", C["ant"], False)}
    for k in range(3):
        A["d%d" % k] = (db.transform(P.driver_T(k)), "H", "#3B8F5E", False); A["dp%d" % k] = (dc.transform(P.driver_T(k)), "H", C["dark"], False)
    bpcb, bsw = P.button_dummy()
    A["btn_pcb"] = (bpcb, "H", "#B03A2E", False); A["btn_sw"] = (bsw, "H", C["dark"], False); A["dongle"] = (P.dongle_dummy(), "H", "#C9772E", False)
    A["dongle2"] = (P.dongle_dummy(DONGLE2), "H", "#1F9E8F", False)
    A["wbtn"] = (tube([(BTN_X, BTN_PCB_Y - 1.5, BTN_Z - 30), (BTN_X + 6, -22, -62), (-20, -24, -58), (4, -22, -30), (10.5, -12, -8.5)], 1.0), "H", "#8E5BD9", False)
    A["w3"] = (tube([(-25, 0, 16), (-21, -15, 9), (-21, -15, -6), (20, -8, -8), (DRV_X0 + 17.5, -7, DRV_ZC[0] - 9)], 1.5), "H", "#D9B13B", False)
    A["wpw"] = (tube([(-18, -24, -8), (8, -24, -10), (DRV_X0 + 6, -10, -6)], 1.2), "H", "#C0392B", False)
    pts = {"pi": (-20, -3, -50), "ant": (0, -2.5, 0), "plug": (BX0 - 5.7, -4, -10), "d3": (DRV_X0 + 17.5, -9.0, DRV_ZC[0]), "d2": (DRV_X0 + 17.5, -9.0, DRV_ZC[1]),
           "d1": (DRV_X0 + 17.5, -9.0, DRV_ZC[2]), "m3": (-8, -14, 20), "gpio": (BX0 + 52.5, -8, -28), "lan": (BX0 + 10.25, -14.0, BZ0 + 6), "dongle": (BX0 + DONGLE["x"], -PI_T / 2 - DONGLE["up"] - 4.0, BZ0 - 12), "dongle2": (BX0 + DONGLE2["x"], -PI_T / 2 - DONGLE2["up"] - 4.0, BZ0 - 12), "cable": (BX0 - 5.6, -4.2, -95),
           "win": (-HX, -4, -14), "hole": (-21, -15, 8), "btn": (BTN_X, BTN_PCB_Y - 5.0, BTN_Z), "pin56": (10.5, -12, -8.5)}
    cam = dict(cam_az=-78, cam_el=12, dist=470, focus=(0, 0, -40))
    vis_check(A, pts, tag="3a", **cam)
    pix = render.render(A, TMP + 'raw3a.png', size=(S, S), label_pts=pts, ground=False, **cam)
    Image.open(TMP + 'raw3a.png').crop((0, 0, S, S - 230)).save(TMP + 'raw3ac.png')
    annotate(TMP + 'raw3ac.png', OUT + '03_머리_속.png', pix, [
        ("m3", "모터 3\n전선은 바닥 구멍으로 내려온다", 'L', 230),
        ("win", "전원 플러그 창\nㄱ자 플러그가 벽에 안 걸린다", 'L', 420),
        ("plug", "전원 플러그 (ㄱ자 마이크로 5핀)\n선은 아래로 내려 받침 속 배터리로", 'L', 610),
        ("btn", "전원 버튼 모듈 (KY-004)\n받침 살에 얹고 뚜껑의 살이 눌러 고정한다", 'L', 800),
        ("pi", "라즈베리파이 3B+\n뒷벽 기둥 4개에 M2.5 나사로 조인다", 'L', 990),
        ("lan", "랜 단자 (개발할 때만)\n쓰려면 조립할 때 랜선을 꽂아 둔다", 'L', 1180),
        ("ant", "안테나\n판 회전축이 여기를 지난다", 'R', 230),
        ("pin56", "버튼 선 2가닥 (보라)\nGPIO 5번, 6번 핀에 꽂는다", 'R', 420),
        ("gpio", "GPIO 핀\n점퍼선 10cm로 바로 옆 드라이버에", 'R', 610),
        ("d3", "드라이버 3 (판 회전)", 'R', 760),
        ("d2", "드라이버 2 (끄덕임)", 'R', 870),
        ("d1", "드라이버 1 (받침 회전)\n셋 다 기둥 4개에 M2 나사로 조인다", 'R', 980),
        ("dongle", "무선 동글 (주황. 폰과 통신)\nGPIO 쪽 USB 의 위 칸", 'R', 1130),
        ("dongle2", "탐지 동글 (청록. 둘레 기기 엿듣기)\n가운데 USB 의 위 칸. 없어도 된다", 'R', 1280),
    ], foot="노란 선 = 모터 3 전선, 빨간 선 = 점퍼선 한 가닥(예), 보라 선 = 버튼 선. 실제로는 점퍼선 18가닥이 파이와 드라이버 사이를 오간다.", title="머리 속 (뚜껑을 떼고 앞에서 본 모습)", sub="안쪽 크기 108 × 33.6 × 108mm · 드라이버 자리 폭 37mm (보드 35mm) · 점퍼선을 꽂고도 뚜껑까지 5mm 넘게 남는다", pad_side=380)

    # (나) 전체: 껍데기를 반투명하게 하고 모터, 배터리, 전선 길을 보여 준다 (전선 길은 room_check.py 에서 길이를 잰 그 길)
    yaw, pitch, spin = 0.0, TILT3, 90.0
    A = P.assembly(yaw, pitch, spin)
    hp = lambda p: HP(p, yaw, pitch)
    sx, sy = P.m1_shaft(); ux, uy = M1_BODY_DIR
    m1c = np.array([sx + ux * (M_OFFSET + 17.0), sy + uy * (M_OFFSET + 17.0), DECK_Z1 + 8.0])       # 모터 1 전선이 나오는 곳 (상자 안)
    cx_, cz_ = P.m2_cap_xz(); m2c = np.array([cx_, ARM_Y1 + 8.0, cz_])                             # 모터 2 전선이 나오는 곳
    x0, x1, z0, z1 = P.M2_SLOT; slot = np.array([(x0 + x1) / 2, CHEEK_Y - 4.0, (z0 + z1) / 2])
    YEL, RED, GRN = "#E3B93A", "#D0402F", "#2FB37C"
    A["w_m1"] = (tube([m1c, m1c + [0, 0, 16], hp((DRV_X0 + 17.5, -12, HZ0 - 4)), hp((DRV_X0 + 17.5, -8, DRV_ZC[2] - 10))]), "X", YEL, False)
    A["w_m2"] = (tube([m2c, (cx_, ARM_Y0 - 3.0, cz_), slot, (-40.0, 30.0, 60.0), hp((DRV_X0 + 24, -14, HZ0 - 4)), hp((DRV_X0 + 24, -8, DRV_ZC[1] - 10))]), "X", YEL, False)
    R = BASE["S"]["R"]
    A["w_usb"] = (tube([(0.0, -R + BATT_IN - 8.0, 10.0), (20.0, -R + BATT_IN + 6.0, 24.0), (3.0, 0, 26.0), (3.0, 0, TUBE_Z1 + 6), hp((BX0 - 6.6, -4.2, HZ0 - 6)), hp((BX0 - 6.6, -4.2, BZ1 - 14))], 1.9), "X", RED, False)
    px, pz = P.pin2_xz(); T2 = P.motor2_T()
    m2body = (T2 @ np.array([-M_OFFSET, 0, -19.0, 1.0]))[:3]
    pts = {"m1": YP((sx + ux * M_OFFSET, sy + uy * M_OFFSET, 62), yaw), "m2": YP(m2body, yaw), "m3": hp((-8, 0, 28)), "batt": np.array([0.0, -40.0, 12.0]),
           "pi": hp((-25, -1, -45)), "drv": hp((DRV_X0 + 17.5, 2, DRV_ZC[1])), "usb": np.array([3.0, 0.0, TUBE_Z1 + 3]), "w_m2": slot, "plugA": np.array([0.0, -R + BATT_IN - 5.0, 12.0]),
           "cap": np.array([-R - 1.0, 0.0, 24.0]), "dongle": hp((BX0 + DONGLE["x"], -PI_T / 2 - DONGLE["up"], BZ0 - 14)), "w_m1": m1c + [0, 0, 14]}
    del pts["drv"]
    ghost = ("tub", "cover", "yoke", "head_shell", "head_lid", "crown", "sector", "rotor_bar", "plate", "plate_frame", "plate_clamp", "coin_cap", "coins", "rotor_hw", "marker_frame", "keeper_r", "keeper_l", "clip", "pin_r", "pin_l", "m2_ring")
    pix = render.render(A, TMP + 'raw3b.png', cam_az=138, cam_el=17, size=(S, S), dist=860, focus=(0, 0, 135), label_pts=pts, ground=False, ghost=ghost, ghost_opacity=0.16)
    Image.open(TMP + 'raw3b.png').crop((0, 0, S, S - 110)).save(TMP + 'raw3bc.png')
    texts = {
        "m3": "모터 3 → 드라이버 3\n머리 안에서만. 필요한 길이 9cm",
        "pi": "파이 + 드라이버 3장 + 전원 버튼\n모두 머리 안",
        "cap": "랜선 홈 마개\n랜선 없이 조립할 때 받침 뒤 홈을 막는다",
        "dongle": "무선 동글 (폰과 통신)\n머리 아래 USB. 치마 안에 숨는다",
        "usb": "전원선 (빨강)\n배터리에서 가운데 통(구멍 20.5mm)으로",
        "batt": "배터리 (받침 속)\n%.0f×%.0f×%.0fmm까지 (큰 받침은 %.0f×%.0f×%.0f)" % (BASE["S"]["bl"], BASE["S"]["bw"], BATT_T, BASE["L"]["bl"], BASE["L"]["bw"], BATT_T),
        "plugA": "배터리 쪽 ㄱ자 플러그\n창 안쪽으로 8mm 들어가 있다",
        "w_m2": "모터 2 전선 (노랑)\n팔 뒤 아래 틈 → 칸막이 구멍 → 상자 속. 17cm",
        "m2": "모터 2 (끄덕임)\n오른쪽 팔 바깥",
        "w_m1": "모터 1 전선 (노랑)\n상자 속에서 바로 머리 아래로. 9cm",
        "m1": "모터 1 (받침 회전)\n상자 속",
    }
    print("3b 이름표 자리:", {k: (int(v[0]), int(v[1])) for k, v in pix.items()})
    annotate(TMP + 'raw3bc.png', OUT + '04_전선_길.png', pix, auto_labels(pix, texts, S, 300, 1330),
             title="모터·배터리 자리와 전선 길 (뒤 왼쪽에서, 껍데기를 비치게 그렸다)",
             sub="혼자 동작할 때의 모습이다. 겉에서 보이는 전선은 모터 2에서 팔 안으로 들어가는 2cm뿐이다 · 모터 전선은 24cm라 셋 다 남는다\n받침과 요크 사이는 전원선이 가운데 축으로 지나가서 받침을 돌려도 감기지 않는다 · 폰과는 무선 동글로 통신한다 (랜선은 개발할 때만)", pad_side=330, pad_top=90)

# ---------------------------------------------------------------- 4 출력 부품
def slice_summary():
    """Bambu Studio 로 슬라이싱한 결과 (print/summary.json)"""
    return json.load(open(os.path.join(paths.SLICE, 'summary.json'), encoding='utf-8'))


def hm(sec):
    return ("%d분" % round(sec / 60.0)) if sec < 3600 else ("%d시간 %02d분" % (sec // 3600, (sec % 3600) // 60))


if "4" in WANT:
    SL = slice_summary()["parts"]
    tw = 400
    names = ["01_base_tub_S", "02_base_cover_S", "03_clip", "04_yoke", "05_pinion_fit2", "16_m2_ring", "06_sector_gear", "07_head_shell", "08_head_lid", "09_crown", "15_marker_frame",
             "10_rotor_bar_fit2", "11_coin_cap", "12_coin_spacer", "13_pin_fit2", "14_pin_keeper", "17_lan_cap", "18_plate_frame", "19_plate_clamp", "20_batt_fill", "90_test_coupon", "91_driver_check", "01_base_tub_L", "02_base_cover_L", "17_lan_cap_L"]
    short = {"01_base_tub_S": "받침 통 S", "02_base_cover_S": "받침 덮개 S", "05_pinion_fit2": "작은 기어 10T", "10_rotor_bar_fit2": "회전 막대", "13_pin_fit2": "끄덕임 축 핀",
             "01_base_tub_L": "받침 통 L (선택)", "02_base_cover_L": "받침 덮개 L (선택)", "11_coin_cap": "균형추 통 마개", "12_coin_spacer": "동전 채움 조각", "20_batt_fill": "배터리 채움 조각", "14_pin_keeper": "핀 막이",
             "15_marker_frame": "마커 틀", "18_plate_frame": "판 틀", "19_plate_clamp": "판 누름쇠", "91_driver_check": "보드 구멍 맞춤 판", "04_yoke": "요크", "17_lan_cap": "랜선 홈 마개 S", "17_lan_cap_L": "랜선 홈 마개 L (선택)"}
    spec = {s[0]: s for s in E.SPEC}
    cols = 8; nrow = int(np.ceil(len(names) / cols)); ch = tw + 122
    sheet = Image.new('RGB', (tw * cols, ch * nrow + 190), BG); d = ImageDraw.Draw(sheet)
    d.text((40, 30), "출력 부품 (출력판에 놓는 방향 그대로)", font=font(44, True), fill=INK)
    d.text((40, 98), "모두 받침(서포트) 없이 뽑는다. 숫자는 Bambu Studio로 슬라이싱한 값이다: 0.2mm 층, 벽 3겹, 채움 15% (회전 막대·마개·동전 채움 조각은 100%). 시간은 프린터가 준비하는 약 10분을 뺀 것.", font=font(26), fill=SUB)
    for i, n in enumerate(names):
        name, fn, rot, qty, ko, pick, infill = spec[n]
        m = E.oriented(fn, rot); bb = m.bounding_box(); span = max(bb[3] - bb[0], bb[4] - bb[1], (bb[5] - bb[2]) * 1.4)
        col = P.COL["accent"] if n[:2] in ("03", "05", "06", "09", "10", "11", "12", "13", "14", "15", "16", "17", "18", "20") else P.COL["body"]
        render.render({n: (m, "X", col, True)}, TMP + 'thumb.png', cam_az=-55, cam_el=32, size=(tw, tw), dist=max(span, 34) * 3.3, focus=(0, 0, (bb[5] - bb[2]) / 2), ground=True, ground_z=-0.05)
        x, y = (i % cols) * tw, (i // cols) * ch + 160
        sheet.paste(Image.open(TMP + 'thumb.png'), (x, y))
        d.text((x + 18, y + tw + 2), "%s %s%s" % (n[:2], short.get(n, ko), "" if qty == 1 else "  ×%d" % qty), font=font(25, True), fill=INK)
        d.text((x + 18, y + tw + 38), "%.0f × %.0f × %.0f mm" % (bb[3] - bb[0], bb[4] - bb[1], bb[5] - bb[2]), font=font(22), fill=SUB)
        sl = SL[n]
        d.text((x + 18, y + tw + 68), "%s · %.0f g" % (hm(sl["model_sec"] * qty), sl["g"] * qty) if sl["g"] * qty >= 9.5 else "%s · %.1f g" % (hm(sl["model_sec"] * qty), sl["g"] * qty), font=font(22), fill=SUB)
    sheet.save(OUT + '05_출력_부품.png')

# ---------------------------------------------------------------- 5 조립 순서
if "5" in WANT:
    h = 800
    mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
    C = P.COL; B, AC = C["body"], C["accent"]
    yaw, pitch, spin = 0.0, 0.0, 90.0
    Yk = lambda m: P.pose_yoke(m, yaw); Hd = lambda m: P.pose_head(m, yaw, pitch); Rt = lambda m: P.pose_head(m, yaw, pitch, spin)
    base = {"tub": P.tub("S"), "battery": P.battery_dummy("S"), "cover": P.cover("S"), "cap": P.lan_cap()}
    yoke = {"yoke": Yk(P.yoke()), "clip": P.clip(), "motor1": Yk((mb + mc).transform(P.motor1_T())), "pinion1": Yk(P.pinion1())}
    bpcb, bsw = P.button_dummy()
    headi = {"shell": P.head_shell(), "pi": pb + pc, "d0": (db + dc).transform(P.driver_T(0)), "d1": (db + dc).transform(P.driver_T(1)), "d2": (db + dc).transform(P.driver_T(2)), "plug": P.usb_plug_dummy(),
             "btn": bpcb + bsw, "dongle": P.dongle_dummy() + P.dongle_dummy(DONGLE2)}
    heado = {"crown": P.crown(), "motor3": v1._m1(mb) + v1._m1(mc), "sector": P.sector_head(), "marker_frame": P.marker_frame()}
    lid = {"lid": P.head_lid()}                                  # 뚜껑은 머리를 요크에 올리고 전선을 다 꽂은 뒤 마지막에 닫는다
    join = {"pin_r": Yk(P.pin(1)), "pin_l": Yk(P.pin(-1)), "keeper_r": Yk(P.pin_keeper(1)), "keeper_l": Yk(P.pin_keeper(-1)), "motor2": Yk((mb + mc).transform(P.motor2_T())), "pinion2": Yk(P.pinion2()), "m2_ring": Yk(P.m2_ring())}
    rot = {"rotor_bar": Rt(P.rotor_bar()), "plate": Rt(P.al_plate(150.0)), "plate_frame": Rt(P.plate_frame()), "plate_clamp": Rt(P.plate_clamp()), "coin_cap": Rt(P.coin_cap(P.COINS[(150.0, AL_T)])), "rotor_hw": Rt(P.rotor_hw(150.0, P.COINS[(150.0, AL_T)]))}
    gray = {"battery": C["batt"], "motor1": C["metal"], "motor2": C["metal"], "motor3": C["metal"], "pi": C["pcb"], "d0": "#3B8F5E", "d1": "#3B8F5E", "d2": "#3B8F5E", "plug": C["dark"], "plate": C["foil"], "rotor_hw": C["screw"], "btn": "#B03A2E", "dongle": "#5E6B85", "cap": AC}

    def scene(old, new, hi_names=None):
        A = {}
        for k, m in old.items():
            A[k] = (m, "X", gray.get(k, B), True)
        for k, m in new.items():
            A[k] = (m, "X", HI if (hi_names is None or k in hi_names) else gray.get(k, B), True)
        return A

    whole_head = dict(headi); whole_head.update(heado)
    posed_head = {k: Hd(m) for k, m in whole_head.items()}
    posed_lid = {k: Hd(m) for k, m in lid.items()}
    steps = [
        ("① 받침", "전원선을 덮개의 가운데 통으로 올려 둔다.\n통 뒤쪽 홈에 마개를 위에서 끼우고,\n덮개를 나사 4개로 덮는다.", scene({"tub": base["tub"], "battery": base["battery"]}, {"cover": base["cover"], "cap": base["cap"]}), dict(cam_az=-60, cam_el=28, dist=520, focus=(0, 0, 25))),
        ("② 요크", "모터 1을 상자 속 돋은 자리에 나사 2개로 달고,\n아래에서 작은 기어를 축에 끼운다. 전원선을 꿰며\n요크를 받침에 얹고, 고리를 뒤에서 밀어 잠근다.", scene(base, yoke), dict(cam_az=-60, cam_el=24, dist=640, focus=(-10, 0, 70))),
        ("③ 머리 속", "몸통 기둥에 파이(M2.5 나사 4)와 드라이버 3장(M2\n나사 12)을 조인다. 점퍼선, ㄱ자 젠더, 전원 버튼 모듈\n(왼쪽 받침 살)을 달고, 동글 두 개를 아래 USB 에 꽂는다.", scene({"shell": headi["shell"]}, {k: v for k, v in headi.items() if k != "shell"}), dict(cam_az=-72, cam_el=18, dist=430, focus=(0, 0, -46))),
        ("④ 머리 겉", "모터 3을 넣은 모터 자리(나사 4)를 얹고 전선을 드라이버\n3에 꽂는다. 부채꼴 기어(나사 2)를 붙이고, 등판에 마커\n종이를 대고 마커 틀(나사 4)로 덮는다. 뚜껑은 아직.", scene(headi, heado), dict(cam_az=-40, cam_el=20, dist=520, focus=(10, 0, -40))),
        ("⑤ 머리 올리기", "모터 2에 맞춤 링과 작은 기어를 끼워 오른쪽 팔 바깥에서\n넣는다. 머리를 내리며 이를 맞물리고 핀 2개를 민 뒤 핀 막이\n(나사 1개씩)를 단다. 전선을 꽂고 뚜껑(나사 4)을 닫는다.", scene({**base, **yoke, **posed_head}, {**join, **posed_lid}), dict(cam_az=62, cam_el=16, dist=820, focus=(0, 0, 120))),
        ("⑥ 판", "동전 %d개를 넣어 마개로 막고, 판 틀을 판 받이에 접시머리\n4개로 단다. 판 아래 변을 아래 턱 밑에 비스듬히 밀어 넣고\n눕힌 뒤 누름쇠(나사 2개)로 누른다. 회전 막대를 모터 3 축에\n꽂아 축 고정 나사를 조인다." % P.COINS[(150.0, AL_T)], scene({**base, **yoke, **posed_head, **posed_lid, **join}, rot, ("rotor_bar", "plate_frame", "plate", "plate_clamp", "coin_cap")), dict(cam_az=-58, cam_el=14, dist=900, focus=(0, 0, 150))),
    ]
    band = 186
    sheet = Image.new('RGB', (h * 3, (h + band) * 2 + 150), BG); d = ImageDraw.Draw(sheet)
    d.text((44, 32), "조립 순서 (주황색 = 그 단계에서 새로 붙이는 것)", font=font(46, True), fill=INK)
    for i, (t1, t2, A, cam) in enumerate(steps):
        render.render(A, TMP + 'raw5_%d.png' % i, size=(h, h), ground=False, **cam)
        x, y = (i % 3) * h, (i // 3) * (h + band) + 120
        sheet.paste(Image.open(TMP + 'raw5_%d.png' % i), (x, y))
        d.text((x + 44, y + h - 6), t1, font=font(38, True), fill=INK)
        for j, l in enumerate(t2.split('\n')):
            d.text((x + 44, y + h + 48 + j * 36), l, font=font(26), fill=SUB)
    sheet.save(OUT + '06_조립_순서.png')

# ---------------------------------------------------------------- 6 배선
if "6" in WANT:
    W, Hh = 2000, 1690
    im = Image.new('RGB', (W, Hh), "#FCFCFB"); d = ImageDraw.Draw(im)
    C1, C2, C3, CB = "#2A78D6", "#EB6834", "#1BAF7A", "#8E5BD9"; RED, BLK = "#C0392B", "#3A3F45"
    d.text((60, 40), "배선: 라즈베리파이 GPIO ↔ 드라이버 3장 + 전원 버튼", font=font(52, True), fill=INK)
    d.text((60, 116), "점퍼선(암-암) 20가닥: 드라이버 18가닥(10cm) + 버튼 2가닥(20cm). 색칠한 핀만 쓴다. 판 회전은 DiffLoc 코드와 같은 핀(GPIO 6, 13, 19, 26)이다.", font=font(28), fill=SUB)
    name = {1: "3.3V", 2: "5V", 4: "5V", 6: "GND", 9: "GND", 14: "GND", 17: "3.3V", 20: "GND", 25: "GND", 30: "GND", 34: "GND", 39: "GND",
            3: "GPIO2", 5: "GPIO3", 7: "GPIO4", 8: "GPIO14", 10: "GPIO15", 11: "GPIO17", 12: "GPIO18", 13: "GPIO27", 15: "GPIO22", 16: "GPIO23", 18: "GPIO24", 19: "GPIO10",
            21: "GPIO9", 22: "GPIO25", 23: "GPIO11", 24: "GPIO8", 26: "GPIO7", 27: "ID_SD", 28: "ID_SC", 29: "GPIO5", 31: "GPIO6", 32: "GPIO12", 33: "GPIO13", 35: "GPIO19",
            36: "GPIO16", 37: "GPIO26", 38: "GPIO20", 40: "GPIO21"}
    drv = [("드라이버 3", "판 회전 (J3)", C1, [31, 33, 35, 37], 2, 39), ("드라이버 2", "끄덕임 (J2)", C2, [11, 13, 15, 16], 4, 14), ("드라이버 1", "받침 회전 (J1)", C3, [32, 36, 38, 40], 4, 34)]
    use = {}
    for t, t2, col, pins, p5, pg in drv:
        for k, p in enumerate(pins):
            use[p] = (col, "%s · IN%d" % (t, k + 1))
        use[pg] = (BLK, "%s · −" % t)
    use[2] = (RED, "드라이버 3 · +"); use[4] = (RED, "드라이버 1과 2 · +  (둘로 나눈다)")
    use[5] = (CB, "전원 버튼 · S"); use[6] = (CB, "전원 버튼 · −")
    x0, y0, dy, pw = 600, 262, 52, 46
    d.rounded_rectangle([x0 - 24, y0 - 30, x0 + pw * 2 + 36, y0 + dy * 20 - 6], 18, outline="#C9CCD1", width=3, fill="#F1F3F1")
    d.text((x0 + pw + 6, y0 - 58), "파이 GPIO 핀 (SD카드 쪽이 위)", font=font(27, True), fill=INK, anchor="mm")
    for r in range(20):
        for c in range(2):
            p = r * 2 + 1 + c; cx = x0 + c * (pw + 12) + pw / 2; cy = y0 + r * dy + dy / 2 - 14
            col, tag = use.get(p, (None, None))
            d.ellipse([cx - 19, cy - 19, cx + 19, cy + 19], fill=col or "#FFFFFF", outline=col or "#A8ADB3", width=3)
            d.text((cx, cy), str(p), font=font(20, True), fill="#FFFFFF" if col else SUB, anchor="mm")
            tx = cx - 62 if c == 0 else cx + 62
            anc = "rm" if c == 0 else "lm"
            if col:
                w1 = d.textlength(name[p], font=font(24, True))
                d.text((tx, cy), name[p], font=font(24, True), fill=INK, anchor=anc)
                x2 = tx - w1 - 22 if c == 0 else tx + w1 + 22
                d.line([(x2 + (8 if c == 0 else -8), cy), (x2 + (-26 if c == 0 else 26), cy)], fill=col, width=4)
                d.text((x2 - 36 if c == 0 else x2 + 36, cy), tag, font=font(24), fill=INK, anchor=anc)
            else:
                d.text((tx, cy), name[p], font=font(22), fill="#A0A5AB", anchor=anc)
    bx, bw_, bh = 1330, 610, 296
    for k, (t, t2, col, pins, p5, pg) in enumerate(drv):
        by = 200 + k * (bh + 16)
        d.rounded_rectangle([bx, by, bx + bw_, by + bh], 18, outline=col, width=5, fill="#FFFFFF")
        d.ellipse([bx + 24, by + 22, bx + 50, by + 48], fill=col)
        d.text((bx + 64, by + 35), "%s · %s" % (t, t2), font=font(30, True), fill=INK, anchor="lm")
        rows = [("IN%d" % (i + 1), p) for i, p in enumerate(pins)] + [("+ (5V)", p5), ("− (GND)", pg)]
        for i, (lab, p) in enumerate(rows):
            ty = by + 82 + i * 32
            d.text((bx + 30, ty), lab, font=font(24, True), fill=INK, anchor="lm")
            d.text((bx + 190, ty), "←  %d번 핀 (%s)%s" % (p, name[p], "  · 둘로 나눈 선" if (lab.startswith("+") and p == 4) else ""), font=font(24), fill=INK, anchor="lm")
        d.text((bx + 30, by + bh - 24), "흰 5핀 소켓에는 모터 전선을 꽂는다", font=font(21), fill=SUB, anchor="lm")
    by = 200 + 3 * (bh + 16); bh2 = 176
    d.rounded_rectangle([bx, by, bx + bw_, by + bh2], 18, outline=CB, width=5, fill="#FFFFFF")
    d.ellipse([bx + 24, by + 22, bx + 50, by + 48], fill=CB)
    d.text((bx + 64, by + 35), "전원 버튼 · KY-004 모듈", font=font(30, True), fill=INK, anchor="lm")
    for i, (lab, txt) in enumerate((("S", "←  5번 핀 (GPIO3)"), ("−", "←  6번 핀 (GND)"))):
        ty = by + 82 + i * 32
        d.text((bx + 30, ty), lab, font=font(24, True), fill=INK, anchor="lm"); d.text((bx + 190, ty), txt, font=font(24), fill=INK, anchor="lm")
    d.text((bx + 30, by + bh2 - 24), "가운데 핀(+)은 비워 둔다", font=font(21), fill=SUB, anchor="lm")
    fy = y0 + dy * 20 + 40
    for i, l in enumerate(["· 5V 핀은 2번, 4번 두 개뿐이다. 4번에서 나온 선 하나를 둘로 갈라(Y자 선) 드라이버 1과 2의 +에 꽂는다. GND 핀은 넉넉해서 드라이버마다 따로 쓴다.",
                           "· 한 번에 모터 하나만 돌리고 멈춘 모터는 전류를 끈다(기어가 버틴다). 그래서 파이의 5V 핀으로 충분하다 (모터 1개 약 0.24A).",
                           "· 모터가 반대로 돌면 선을 바꾸지 말고 제어 코드의 방향 값(DIR)만 뒤집는다. 드라이버의 전원 점퍼(ON/OFF 캡)는 꽂아 둔다.",
                           "· 전원 버튼: 2초 누르면 안전하게 꺼지고, 꺼진 뒤 한 번 누르면 다시 켜진다. I2C는 꺼 둔다 (5번 핀을 같이 쓴다).",
                           "   /boot/config.txt 에 한 줄을 넣는다:  dtoverlay=gpio-shutdown,gpio_pin=3,debounce=2000",
                           "· 폰과의 통신은 무선 동글(파이 USB)로 한다. 내장 Wi-Fi 는 CSI 를 받는 데만 쓴다. 랜선은 개발할 때만 꽂는다."]):
        d.text((60, fy + i * 46), l, font=font(26), fill=INK)
    im.save(OUT + '07_배선.png')

# ---------------------------------------------------------------- 7 출력판 나누기
PLATE_KO = {"90_test_coupon": "시험 조각", "91_driver_check": "보드 맞춤 판", "16_m2_ring": "맞춤 링", "02_base_cover_S": "받침 덮개", "01_base_tub_S": "받침 통", "03_clip": "고리", "04_yoke": "요크", "17_lan_cap": "홈 마개",
            "07_head_shell": "머리 몸통", "08_head_lid": "머리 뚜껑", "09_crown": "모터 자리", "06_sector_gear": "부채꼴 기어", "11_coin_cap": "마개", "12_coin_spacer": "동전 채움 조각", "20_batt_fill": "배터리 채움 조각", "14_pin_keeper": "핀 막이",
            "15_marker_frame": "마커 틀", "18_plate_frame": "판 틀", "19_plate_clamp": "누름쇠", "01_base_tub_L": "받침 통 L", "02_base_cover_L": "받침 덮개 L", "17_lan_cap_L": "홈 마개 L"}


def plate_panel(row, px):
    """출력판 한 장을 위에서 본 그림 (앞 = 아래). Bambu Studio 프로젝트에 넣은 자리 그대로 그린다."""
    import trimesh
    from shapely.geometry import Polygon, MultiPolygon
    from shapely import affinity
    spec = {s_[0]: s_ for s_ in E.SPEC}
    BEDS = 256.0; mg = int(px * 0.06); sc = (px - 2 * mg) / BEDS
    im = Image.new('RGB', (px, px), BG); d = ImageDraw.Draw(im)
    X = lambda x: mg + x * sc; Y = lambda y: px - mg - y * sc
    d.rounded_rectangle([X(0), Y(BEDS), X(BEDS), Y(0)], 10, fill="#E4E2DC", outline="#B9B6AE", width=2)
    for g in (64, 128, 192):
        d.line([(X(g), Y(0)), (X(g), Y(BEDS))], fill="#D3D0C8", width=1); d.line([(X(0), Y(g)), (X(BEDS), Y(g))], fill="#D3D0C8", width=1)
    d.rectangle([X(18), Y(14), X(240), Y(0)], fill="#D9D2C2")                                   # 시작할 때 시험 줄을 긋는 자리
    d.rectangle([X(0), Y(28), X(18), Y(0)], fill="#D9A79B", outline="#B8553F", width=2)          # 앞 왼쪽 금지 구역
    d.text((X(129), Y(7)), "프린터가 시작할 때 줄을 긋는 자리", font=font(px * 0.026), fill="#8A8273", anchor="mm")
    body = P.COL["body"]
    small = []; once = set()
    for name, cx, cy, rot in row["items"]:
        m = E.oriented(spec[name][1], spec[name][2]); tm, _ = E.clean32(m)
        poly = trimesh.path.polygons.projected(tm, normal=[0, 0, 1])
        polys = list(poly.geoms) if isinstance(poly, MultiPolygon) else [poly]
        x0, y0, x1, y1 = poly.bounds
        for pg in polys:
            pg = affinity.translate(affinity.rotate(pg, rot, origin=(0, 0)), cx - (x0 + x1) / 2, cy - (y0 + y1) / 2)
            d.polygon([(X(a), Y(b)) for a, b in pg.exterior.coords], fill=body, outline="#0F1838")
            for h_ in pg.interiors:
                d.polygon([(X(a), Y(b)) for a, b in h_.coords], fill="#E4E2DC", outline="#0F1838")
        w = (x1 - x0) * sc
        m_ = None
        for key in ("13_pin_fit", "05_pinion_fit"):
            if name.startswith(key):
                m_ = name[-1]
        if name in ("12_coin_spacer", "14_pin_keeper", "20_batt_fill"):             # 여러 개 놓이는 작은 것: 이름은 한 번만 (아래에서 무리 옆에 적는다)
            continue
        if m_:
            d.text((X(cx), Y(cy - (y1 - y0) / 2) + px * 0.022), m_, font=font(px * 0.03, True), fill=INK, anchor="mm")
        elif name.startswith("10_rotor_bar"):
            d.text((X(cx), Y(cy + (y1 - y0) / 2) - px * 0.022), "회전 막대 (맞춤 %s)" % name[-1], font=font(px * 0.03, True), fill=INK, anchor="mm")
        else:
            lab = PLATE_KO.get(name, name); f = font(px * 0.03, True); tw_ = d.textlength(lab, font=f)
            if tw_ < w - 8 and (y1 - y0) * sc > px * 0.05:                         # 큰 부품: 가운데에 이름표 (구멍 위에서도 읽히게 바탕을 깐다)
                d.rounded_rectangle([X(cx) - tw_ / 2 - 10, Y(cy) - px * 0.024, X(cx) + tw_ / 2 + 10, Y(cy) + px * 0.024], 8, fill="#0F1838")
                d.text((X(cx), Y(cy)), lab, font=f, fill="#FFFFFF", anchor="mm")
            elif name == "91_driver_check":                                         # 작은 부품이라 판 안쪽 빈 곳에 두 줄로
                f2 = font(px * 0.024, True)
                d.text((X(cx), Y(cy) - px * 0.016), "보드", font=f2, fill=INK, anchor="mm"); d.text((X(cx), Y(cy) + px * 0.016), "맞춤 판", font=f2, fill=INK, anchor="mm")
            elif name == "16_m2_ring":                                              # 옆 부품의 이름과 겹치지 않게 아래에
                d.text((X(cx), Y(cy - (y1 - y0) / 2) + px * 0.024), lab, font=f, fill=INK, anchor="mm")
            else:
                d.text((X(cx), Y(cy + (y1 - y0) / 2) - px * 0.022), lab, font=f, fill=INK, anchor="mm")
    pins = [it for it in row["items"] if it[0].startswith("13_pin_fit")]; gears = [it for it in row["items"] if it[0].startswith("05_pinion_fit")]
    if pins:
        d.text((X(min(it[1] for it in pins) - 14), Y(pins[0][2])), "핀", font=font(px * 0.03, True), fill=INK, anchor="rm")
    if gears:
        d.text((X(min(it[1] for it in gears) - 14), Y(gears[0][2])), "작은 기어", font=font(px * 0.03, True), fill=INK, anchor="rm")
    its = [it for it in row["items"] if it[0] == "12_coin_spacer"]
    if its:                                                                         # 두 줄 아래 빈 곳에
        d.text((X(np.mean([it[1] for it in its])), Y(min(it[2] for it in its) - 21.0)), "동전 채움 조각 × %d" % len(its), font=font(px * 0.028, True), fill=INK, anchor="mm")
    its = [it for it in row["items"] if it[0] == "20_batt_fill"]
    if its:                                                                         # 무리 위에
        d.text((X(np.mean([it[1] for it in its])), Y(max(it[2] for it in its) + 14.0)), "배터리 채움 조각 × %d" % len(its), font=font(px * 0.028, True), fill=INK, anchor="mm")
    its = [it for it in row["items"] if it[0] == "14_pin_keeper"]
    if its:                                                                         # 왼쪽 가장자리에 있어서 이름은 위에 적는다
        d.text((X(np.mean([it[1] for it in its])), Y(max(it[2] for it in its) + 9.5)), "핀 막이 × %d" % len(its), font=font(px * 0.028, True), fill=INK, anchor="mm")
    d.text((X(BEDS / 2), px - mg * 0.45), "앞 (문 쪽)", font=font(px * 0.026), fill=SUB, anchor="mm")
    return im


if "7" in WANT:
    SUMM = slice_summary()["plates"]
    main = [("0판_시험조각", "0판 · 시험 조각과 작은 것들", "시험 조각, 보드 구멍 맞춤 판, 맞춤 링, 핀과 작은 기어\n맞춤 1·2·3 을 2개씩, 마커 틀, 핀 막이 2개, 고리"),
            ("1판_받침_S", "1판 · 받침 (작은 것)", "받침 통, 받침 덮개, 랜선 홈 마개,\n배터리 채움 조각 6개"),
            ("2판_요크", "2판 · 요크", "요크 (회전판 + 팔 + 전선 상자)"),
            ("3판_머리", "3판 · 머리", "머리 몸통, 머리 뚜껑, 모터 자리,\n부채꼴 기어"),
            ("4판_회전부_맞춤2", "4판 · 회전부 (채움 100%)", "회전 막대, 균형추 통 마개, 동전 채움 조각 %d개(여분).\n막대는 맞춤 1·2·3 파일이 따로 있다"
             % sum(1 for it in SUMM["4판_회전부_맞춤2"]["items"] if it[0] == "12_coin_spacer")),
            ("5판_판틀", "5판 · 판 틀", "판 틀(한 장)과 판 누름쇠.\n%gT 알루미늄 판을 구멍 없이 홈에 넣어 잡는다" % AL_T)]
    h = 640; sheet = Image.new('RGB', (h * len(main), h + 440), BG); d = ImageDraw.Draw(sheet)
    d.text((40, 30), "출력판 나누기 (Bambu X1 Carbon, 256×256mm 판 %d번)" % len(main), font=font(44, True), fill=INK)
    d.text((40, 96), "Bambu Studio 프로젝트 파일(print/project)에 넣어 둔 자리 그대로다. 붉은 칸은 앞 왼쪽 금지 구역. 시간과 무게는 Bambu Studio 2.5.1 로 슬라이싱한 값.", font=font(26), fill=SUB)
    total_s = total_g = 0
    for i, (key, title, what) in enumerate(main):
        row = SUMM[key]
        sheet.paste(plate_panel(row, h), (i * h, 150))
        total_s += row["sec"]; total_g += row["g"]
        d.text((i * h + 40, h + 156), title, font=font(32, True), fill=INK)
        d.text((i * h + 40, h + 202), "%s · %.0f g" % (hm(row["sec"]), row["g"]), font=font(28, True), fill="#1B2A57")
        for j, l in enumerate(what.split("\n")):
            d.text((i * h + 40, h + 246 + j * 34), l, font=font(24), fill=SUB)
    L = SUMM["선택_받침통_L"]; LC = SUMM["선택_받침덮개_L"]
    d.text((40, h + 330), "합계 %s, 필라멘트 %.0f g (1kg 한 통의 %.0f%%). 모두 0.2mm 층 · 벽 3겹 · 채움 15%% (4판만 100%%) · 서포트 없음 · Textured PEI 판 · PLA 220°C / 판 55°C." % (
        hm(total_s), total_g, total_g / 10), font=font(27), fill=INK)
    d.text((40, h + 372), "큰 받침(L, 지름 19cm. 잘 안 넘어진다)을 쓰면 1판 대신 통 %s · %.0f g + 덮개 %s · %.0f g 두 판을 뽑는다 (L 용 홈 마개는 덮개 판에 같이 있다). 알루미늄 판, 나사·너트, 100원 동전은 사는 것." % (
        hm(L["sec"]), L["g"], hm(LC["sec"]), LC["g"]), font=font(27), fill=SUB)
    sheet.save(OUT + '08_출력판_나누기.png')
    json.dump(dict(plates=[dict(key=k, title=t, files=[it[0] for it in SUMM[k]["items"]], sec=SUMM[k]["sec"], g=SUMM[k]["g"]) for k, t, w in main], total_sec=total_s, total_g=round(total_g, 1),
                   optional=[dict(key=k, sec=SUMM[k]["sec"], g=SUMM[k]["g"]) for k in ("선택_받침통_L", "선택_받침덮개_L")], slicer="Bambu Studio 2.5.1"),
              open(ROOT + 'plates.json', 'w'), ensure_ascii=False, indent=1)

print(sorted(os.listdir(OUT)))
