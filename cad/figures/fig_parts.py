"""부품 목록 그림 (docs/부품 목록.md).   xvfb-run -a python3 fig_parts.py [p b s]
  p  출력 부품마다 한 장: 출력판에 놓는 방향 그대로, 바깥 치수 상자 (가로 · 세로 · 높이)
  b  사는 부품마다 한 장: 모양과 치수 이름표 (치수는 공식 자료 또는 모델에 넣은 흔한 값, 조립 안내 2절과 같다)
  s  나사·너트 (모두 같은 배율), 전선과 소모품, AR 마커 종이
결과: 그림 폴더의 부품/*.png"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]
import numpy as np
import parts2 as P
import parts as v1
import render, export2 as E
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont
import paths

OUT = os.path.join(paths.IMG, "부품"); os.makedirs(OUT, exist_ok=True)
TMP = paths.TMP + "/"
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#9AA1AA", "#F4F2EE"
DIM = "#D9480F"                                    # 치수를 잰 모서리
PART = MIDNIGHT                                    # 출력 부품 색
METAL, BLUE, PCB, DARK, FOIL, BATT, COIN = "#A9AEB3", "#3F78C9", "#2F7D4F", "#2B2F35", "#8F979F", "#50565E", "#9DA2A8"
WANT = set(sys.argv[1:]) or set("pbs")
W, BAND = 1400, 172                                # 그림 폭, 위쪽 제목 띠 높이
SL = json.load(open(os.path.join(paths.SLICE, "summary.json"), encoding="utf-8"))["parts"]
SPEC = {s[0]: s for s in E.SPEC}


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)


def mm(v):
    return ("%.0f" % v) if abs(v - round(v)) < 0.05 else ("%.1f" % v)


def hm(sec):
    return ("%d분" % max(1, round(sec / 60.0))) if sec < 3600 else ("%d시간 %02d분" % (sec // 3600, (sec % 3600) // 60))


def wrap(d, text, f, maxw):
    """글을 maxw 픽셀 안에서 줄을 바꾼다 (띄어쓰기에서, 안 되면 글자에서)"""
    out = []
    for para in text.split("\n"):
        cur = ""
        for w in para.split(" "):
            t = (cur + " " + w).strip()
            if d.textlength(t, font=f) <= maxw:
                cur = t; continue
            if cur:
                out.append(cur)
            cur = w
            while d.textlength(cur, font=f) > maxw:                     # 한 낱말이 너무 길면 글자에서 자른다
                k = len(cur)
                while k > 1 and d.textlength(cur[:k], font=f) > maxw:
                    k -= 1
                out.append(cur[:k]); cur = cur[k:]
        out.append(cur)
    return out


def head(im, title, sub):
    """위쪽 제목 띠"""
    d = ImageDraw.Draw(im); w = im.size[0]
    d.text((44, 26), title, font=font(44, True), fill=INK)
    y = 88
    for l in wrap(d, sub, font(25), w - 88)[:2]:
        d.text((44, y), l, font=font(25), fill=SUB); y += 34
    d.line([(44, BAND - 8), (w - 44, BAND - 8)], fill="#DDD8D0", width=2)


def finish(im, title, sub, fname, minw=1000, mx=70, my=46):
    """그림 둘레의 빈 곳을 잘라 내고 (가로는 minw 이상) 위에 제목 띠를 붙여 저장한다"""
    a = np.asarray(im.convert("RGB")).astype(int)[BAND:]
    bg = np.array([int(BG[i:i + 2], 16) for i in (1, 3, 5)])
    ys, xs = np.nonzero(np.abs(a - bg).sum(2) > 18)
    xa, xb, ya, yb = xs.min(), xs.max(), ys.min(), ys.max()
    fw = int(max(xb - xa + 2 * mx, minw)); cx = (xa + xb) / 2
    x0 = int(min(max(cx - fw / 2, 0), max(0, im.size[0] - fw))); fw = min(fw, im.size[0])
    y0 = max(0, ya - my); y1 = min(a.shape[0], yb + my)
    out = Image.new("RGB", (fw, BAND + y1 - y0), BG)
    out.paste(im.crop((x0, BAND + y0, x0 + fw, BAND + y1)), (0, BAND))
    head(out, title, sub)
    out.save(os.path.join(OUT, fname))
    return out.size


def cam_basis(az, el):
    a, e = np.radians(az), np.radians(el)
    c = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])          # 초점 -> 카메라
    r = np.cross(-c, [0, 0, 1.0]); r /= np.linalg.norm(r)
    return c, r, np.cross(r, -c)


def fit_dist(pts, focus, az, el, w, h, frac, fov=28.0):
    """점들이 화면의 frac 안에 들어오는 카메라 거리 (원근 때문에 앞쪽이 커지는 만큼 깊이의 1/4 을 더한다)"""
    c, r, u = cam_basis(az, el); q = np.asarray(pts, float) - focus; t = np.tan(np.radians(fov / 2))
    ex, ey, ez = np.ptp(q @ r), np.ptp(q @ u), np.ptp(q @ c)
    return max(ey / (frac * 2 * t), ex / (frac * 2 * t * w / h)) + ez / 4


def box_pts(b):
    return {(i, j, k): (b[3] if i else b[0], b[4] if j else b[1], b[5] if k else b[2]) for i in (0, 1) for j in (0, 1) for k in (0, 1)}


def dashed(d, a, b, fill, width=2, dash=11, gap=7):
    a, b = np.asarray(a, float), np.asarray(b, float); L = np.linalg.norm(b - a)
    if L < 1:
        return
    u = (b - a) / L; s = 0.0
    while s < L:
        e = min(L, s + dash); d.line([tuple(a + u * s), tuple(a + u * e)], fill=fill, width=width); s = e + gap


def dim_label(d, p0, p1, center, text, f, pad=16, xlim=(0, W)):
    """화면의 모서리 p0-p1 에 치수선(주황)과 글. 글은 모서리 가운데에서 상자 바깥쪽으로 놓는다"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d.line([tuple(p0), tuple(p1)], fill=DIM, width=4)
    t = p1 - p0; L = np.linalg.norm(t)
    n = np.array([-t[1], t[0]]) / max(L, 1e-6) if L > 1e-6 else np.array([-1.0, 0])
    mid = (p0 + p1) / 2
    if np.dot(n, mid - np.asarray(center)) < 0:
        n = -n
    for p in (p0, p1):                                                  # 끝 표시
        d.line([tuple(p - n * 9), tuple(p + n * 9)], fill=DIM, width=4)
    bb = d.textbbox((0, 0), text, font=f); tw, th = bb[2] - bb[0], bb[3] - bb[1]
    off = pad + 0.5 * abs(n[0]) * tw + 0.5 * abs(n[1]) * th
    cx, cy = mid + n * off
    cx = min(max(cx, xlim[0] + tw / 2 + 10), xlim[1] - tw / 2 - 10)
    d.rectangle([cx - tw / 2 - 6, cy - th / 2 - 6, cx + tw / 2 + 6, cy + th / 2 + 8], fill=BG)
    d.text((cx - tw / 2 - bb[0], cy - th / 2 - bb[1]), text, font=f, fill=INK)


def boxed_card(items, title, sub, fname, az=-55.0, el=28.0, H=900, frac=0.80, gap=None):
    """items: [(형상, 색, 붙일 이름 또는 None)]. 여럿이면 칸을 나눠 나란히 (모두 같은 배율). 하나마다 바깥 치수 상자와 가로 · 세로 · 높이를 적는다"""
    n = len(items); Hr = H - BAND; PW = W // n
    placed = []
    for m, col, tag in items:
        b = m.bounding_box(); m2 = m.translate([-(b[0] + b[3]) / 2, -(b[1] + b[4]) / 2, -b[2]]); placed.append((m2, col, tag, m2.bounding_box()))
    dist = max(fit_dist(np.array(list(box_pts(b).values())), np.array([0, 0, (b[5] - b[2]) / 2]), az, el, PW, Hr, frac * (0.92 if n > 1 else 1.0))
               for _, _, _, b in placed)
    im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    c, r, u = cam_basis(az, el)
    hx, hy, hz = int(c[0] < 0), int(c[1] < 0), int(c[2] < 0)          # 카메라에서 가장 먼 꼭짓점 (그 꼭짓점에 붙은 모서리 셋은 가려진다)
    vx, vy = 1 - hx, 1 - hy
    f = font(30, True); ft = font(27, True)
    for i, (m, col, tag, b) in enumerate(placed):
        x0 = i * PW
        pix = render.render({"p": (m, "X", col, True)}, TMP + "pc.png", cam_az=az, cam_el=el, size=(PW, Hr), dist=dist, focus=(0, 0, (b[5] - b[2]) / 2),
                            ground=False, label_pts=box_pts(b))
        im.paste(Image.open(TMP + "pc.png").convert("RGB"), (x0, BAND))
        P_ = {k: (v[0] + x0, v[1] + BAND) for k, v in pix.items()}
        cen = np.mean(list(P_.values()), axis=0)
        edges = [(a, bk) for a in P_ for bk in P_ if a < bk and sum(x != y for x, y in zip(a, bk)) == 1]
        ex = ((0, vy, 0), (1, vy, 0)); ey = ((vx, 0, 0), (vx, 1, 0))
        zx = min([(hx, vy), (vx, hy)], key=lambda q: P_[(q[0], q[1], 0)][0]); ez = ((zx[0], zx[1], 0), (zx[0], zx[1], 1))
        for a, bk in edges:
            if (hx, hy, hz) in (a, bk) or {a, bk} in ({*ex}, {*ey}, {*ez}):
                continue
            dashed(d, P_[a], P_[bk], LINE)
        for e, v in ((ex, b[3] - b[0]), (ey, b[4] - b[1]), (ez, b[5] - b[2])):
            dim_label(d, P_[e[0]], P_[e[1]], cen, "%s mm" % mm(v), f, xlim=(x0, x0 + PW))
        if tag:
            top = min(P_.values(), key=lambda p: p[1])
            bb = d.textbbox((0, 0), tag, font=ft); tw = bb[2] - bb[0]
            tx = min(max(cen[0] - tw / 2, x0 + 20), x0 + PW - tw - 20)
            d.text((tx, max(BAND + 8, top[1] - 70)), tag, font=ft, fill=SUB)
    sz = finish(im, title, sub, fname)
    print(fname, sz, [tuple(round(v, 1) for v in (b[3] - b[0], b[4] - b[1], b[5] - b[2])) for _, _, _, b in placed])


def callout_card(items, callouts, title, sub, fname, az=-55.0, el=30.0, H=860, frac=0.66, RW=820, focus=None):
    """items: [(형상, 색)]. callouts: [(3D 점, 굵은 이름, 설명)]. 그림은 왼쪽 RW 픽셀, 이름표는 오른쪽 칸"""
    Hr = H - BAND
    allp = np.vstack([np.array(list(box_pts(m.bounding_box()).values())) for m, c in items])
    fc = np.asarray(focus, float) if focus is not None else (allp.min(0) + allp.max(0)) / 2
    dist = fit_dist(allp, fc, az, el, RW, Hr, frac)
    A = {"q%d" % i: (m, "X", col, True) for i, (m, col) in enumerate(items)}
    pix = render.render(A, TMP + "cc.png", cam_az=az, cam_el=el, size=(RW, Hr), dist=dist, focus=tuple(fc), ground=False,
                        label_pts={i: p for i, (p, a, b) in enumerate(callouts)})
    im = Image.new("RGB", (W, H), BG); im.paste(Image.open(TMP + "cc.png").convert("RGB"), (0, BAND)); d = ImageDraw.Draw(im)
    fb, fr = font(28, True), font(25)
    tx = RW + 30; maxw = W - tx - 40
    blocks = []
    for i, (p, name, desc) in enumerate(callouts):
        lines = wrap(d, desc, fr, maxw) if desc else []
        hgt = 40 + 34 * len(lines)
        blocks.append((pix[i][1] + BAND, i, name, lines, hgt))
    blocks.sort()
    tot = sum(b[4] for b in blocks); spare = (H - BAND - 40) - tot
    y = BAND + 20 + max(0, spare) / (len(blocks) + 1)
    for ay, i, name, lines, hgt in blocks:
        ax, ay = pix[i][0], pix[i][1] + BAND
        d.line([(ax, ay), (tx - 14, y + 20)], fill=LINE, width=2)
        d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill="#FFFFFF", outline=DIM, width=3)
        d.text((tx, y), name, font=fb, fill=INK)
        for k, l in enumerate(lines):
            d.text((tx, y + 40 + 34 * k), l, font=fr, fill=SUB)
        y += hgt + max(0, spare) / (len(blocks) + 1)
    print(fname, finish(im, title, sub, fname))


# ================================================================ 출력 부품
KINDS = [  # (번호, 그림 이름, 제목, [STL 이름], [붙일 이름], 덧붙임)
    ("01", "01_받침_통", "받침 통", ["01_base_tub_S", "01_base_tub_L"], ["S (작은 받침)", "L (큰 받침, 선택)"], "배터리 크기에 맞춰 하나만 뽑는다"),
    ("02", "02_받침_덮개", "받침 덮개", ["02_base_cover_S", "02_base_cover_L"], ["S", "L (선택)"], "받침 통과 같은 글자를 뽑는다"),
    ("03", "03_회전판_고리", "회전판 고리", ["03_clip"], None, ""),
    ("04", "04_요크", "요크 (회전판 + 팔 + 전선 상자)", ["04_yoke"], None, ""),
    ("05", "05_작은_기어", "작은 기어 10T", ["05_pinion_fit2"], None, "맞춤 1·2·3 세 가지를 2개씩 뽑아 모터 축에 맞는 것 2개를 쓴다 (축 구멍만 다르다)"),
    ("06", "06_부채꼴_기어", "부채꼴 기어", ["06_sector_gear"], None, ""),
    ("07", "07_머리_몸통", "머리 몸통", ["07_head_shell"], None, ""),
    ("08", "08_머리_뚜껑", "머리 뚜껑", ["08_head_lid"], None, ""),
    ("09", "09_모터_자리", "모터 자리", ["09_crown"], None, ""),
    ("10", "10_회전_막대", "회전 막대 (판 받이 + 균형추 통)", ["10_rotor_bar_fit2"], None, "맞춤 1·2·3 가운데 시험 조각이 맞은 번호 하나만 뽑는다 (축 구멍만 다르다)"),
    ("11", "11_균형추_통_마개", "균형추 통 마개", ["11_coin_cap"], None, ""),
    ("12", "12_동전_채움_조각", "동전 채움 조각", ["12_coin_spacer"], None, ""),
    ("13", "13_끄덕임_축_핀", "끄덕임 축 핀", ["13_pin_fit1"], None, "지름 8.0 · 7.9 · 7.8 세 가지를 2개씩 뽑아 맞는 것 2개를 쓴다"),
    ("14", "14_핀_막이", "핀 막이", ["14_pin_keeper"], None, ""),
    ("15", "15_마커_틀", "마커 틀", ["15_marker_frame"], None, ""),
    ("16", "16_모터2_맞춤_링", "모터 2 맞춤 링", ["16_m2_ring"], None, ""),
    ("17", "17_랜선_홈_마개", "랜선 홈 마개", ["17_lan_cap", "17_lan_cap_L"], ["작은 받침용", "큰 받침용"], "받침 크기에 맞는 것 하나"),
    ("18", "18_판_틀", "판 틀", ["18_plate_frame"], None, ""),
    ("19", "19_판_누름쇠", "판 누름쇠", ["19_plate_clamp"], None, ""),
    ("20", "20_배터리_채움_조각", "배터리 채움 조각", ["20_batt_fill"], None, ""),
    ("21", "21_ESP32_안테나_누름판", "ESP32 안테나 누름판", ["21_ant_clip"], None, ""),
    ("22", "22_ESP32_막음쇠", "ESP32 막음쇠", ["22_xiao_cap"], None, ""),
    ("90", "90_맞춤_시험_조각", "맞춤 시험 조각", ["90_test_coupon"], None, ""),
    ("91", "91_보드_구멍_맞춤_판", "보드 구멍 맞춤 판", ["91_driver_check"], None, ""),
]


def printed():
    only = os.environ.get("ONLY", "").split(",") if os.environ.get("ONLY") else None
    for no, fn, title, stls, tags, note in KINDS:
        if only and no not in only:
            continue
        items, info = [], []
        for k, n in enumerate(stls):
            name, build, rot, qty, ko, pick, infill = SPEC[n]
            items.append((E.oriented(build, rot), PART, tags[k] if tags else None))
            s = SL[n]; info.append((qty, s["model_sec"], s["g"], infill))
        qty, sec, g, infill = info[0]
        if len(stls) == 1:
            sub = "%d개 · 하나에 %s · %s g%s" % (qty, hm(sec), mm(g), " · 채움 %d%%" % infill if infill != 15 else "")
        else:
            sub = " / ".join("%s: %s · %s g" % (t.split(" ")[0], hm(i[1]), mm(i[2])) for t, i in zip(tags, info))
        if note:
            sub += "  —  " + note
        boxed_card(items, "%s  %s" % (no, title), sub, fn + ".png")


# ================================================================ 사는 부품
def bought():
    # 28BYJ-48: 축이 +Z, 축 자리가 원점, 몸통 가운데는 -X 로 8
    mb, mc = v1.motor_body_dummy(); sh = v1.motor_shaft_dummy()
    c = (-0.643, -0.766)
    callout_card([(mb, METAL), (mc, BLUE), (sh, METAL)],
                 [((0, 0, 9.5), "축", "지름 5, 양쪽이 납작 (폭 3), 길이 10"),
                  ((0, -4.5, 0.8), "축 자리", "몸통 가운데에서 8 비켜 있다"),
                  ((-M_OFFSET, -M_EAR - 2.5, -0.4), "귀 2개", "구멍 지름 4.2, 구멍 간격 35 (M3 × 6 나사)"),
                  ((-M_OFFSET + 14 * c[0], 14 * c[1], -11), "몸통", "지름 28, 높이 19"),
                  ((-25.0, 0, -10), "파란 덮개", "전선 5가닥이 나온다. 끝에 흰 플러그")],
                 "28BYJ-48 스텝모터 (5V)", "3개 (ULN2003 드라이버 보드와 한 세트로 산다) · 약 34 g · 모터 1: 요크 전선 상자 속, 모터 2: 오른쪽 팔, 모터 3: 모터 자리",
                 "구매_28BYJ-48_모터.png", az=-130, el=30)
    # ULN2003 보드
    db, dc = v1.driver_dummy()
    callout_card([(db, PCB), (dc, DARK)],
                 [((17.0, 2.0, 0.8), "기판", "34 × 31.7 (실측), 두께 1.6"),
                  ((15.0, -13.25, 1.6), "나사 구멍 4개", "지름 3, 간격 %.1f × %.1f (모델 값). 가는 M2 나사라 ±1 까지 들어간다" % DRV_HOLE),
                  ((-4.0, 0.0, 9.6), "부품 쪽", "높이 약 15 (핀·소켓 포함). 흰 소켓에 모터 플러그를 꽂는다")],
                 "ULN2003 드라이버 보드", "3장 · 약 8 g · 머리 몸통 오른쪽 기둥 (위에서부터 3, 2, 1번) · M2 × 6 나사 4개씩",
                 "구매_ULN2003_드라이버.png", az=-55, el=42)
    # 라즈베리파이 3B+ (눕혀서: 부품 면이 위)
    pb, pc = v1.pi_dummy(); pb, pc = pb.rotate([-90, 0, 0]), pc.rotate([-90, 0, 0])
    hx, hz = max(v1.pi_hole_xz(), key=lambda q: (q[0], -q[1]))
    callout_card([(pb, PCB), (pc, METAL)],
                 [((BX1, (BZ0 + BZ1) / 2, 0.0), "기판", "85 × 56, 두께 1.6. 단자까지 높이 약 17"),
                  ((hx, hz, 0.8), "나사 구멍 4개", "지름 2.7, 간격 58 × 49 (M2.5 × 6 나사)"),
                  ((BX0 + 29.0, BZ0 - 2.0, 12.0), "USB 2단 × 2", "위 칸 둘에 무선 동글과 탐지 동글"),
                  ((BX0 + 2.0, BZ1 - 10.6, 3.7), "전원 단자 (마이크로 5핀)", "ㄱ자 젠더나 ㄱ자 플러그를 꽂는다")],
                 "라즈베리파이 3B+", "1개 · 50 g · 머리 몸통 뒷벽 기둥 4개 (부품 면이 앞, SD카드 쪽이 위) · SD카드는 조립 전에 꽂는다",
                 "구매_라즈베리파이.png", az=-62, el=40)
    # 알루미늄 판
    boxed_card([(box(0, 100, 0, 150, 0, AL_T), FOIL, None)], "알루미늄 판 1T", "1장 · 40.5 g · 가지고 있는 판 그대로, 구멍을 뚫지 않는다 · 판 틀의 홈에 넣으면 아래 턱과 누름쇠가 잡는다",
               "구매_알루미늄_판.png", el=34)
    # 100원 동전
    one = cz(COIN_D / 2, 0, COIN_T, 0, 0, 96); stack = cz(COIN_D / 2, 0, COIN_T * N_COIN, 0, 0, 96)
    boxed_card([(one, COIN, "1개 (5.42 g)"), (stack, COIN, "12개를 쌓으면")], "100원 동전 (균형추)",
               "12개 · 하나에 5.42 g (한국은행) · 두께 1.5 안팎 (옛 동전 1.6) · 회전 막대의 균형추 통 12칸을 다 채운다", "구매_100원_동전.png", el=30, gap=30)
    # 보조배터리: 받침에 들어가는 가장 큰 크기
    bs, bl = BASE["S"], BASE["L"]
    boxed_card([(box(0, bs["bw"], 0, bs["bl"], 0, BATT_T), BATT, "작은 받침(S)에 들어가는 가장 큰 것"), (box(0, bl["bw"], 0, bl["bl"], 0, BATT_T), BATT, "큰 받침(L)")],
               "보조배터리 (가지고 있는 것)", "5V 2A 이상, 단자는 USB-C 든 USB-A 든 · 이 크기 안이면 들어간다. 작아서 흔들리면 배터리 채움 조각(20)을 끼운다",
               "구매_보조배터리.png", el=30, gap=45)
    # 전원 버튼 모듈 KY-004 (단순 모델: 기판 15 x 18.5, 버튼, 꺾인 핀 3개)
    pcb = box(-7.5, 7.5, 0, 18.5, 0, 1.6)
    sw = box(-3, 3, 8.5, 14.5, 1.6, 5.1) + cz(1.75, 5.1, 6.8, 0, 11.5)
    hd = box(-3.9, 3.9, 0.6, 3.1, 1.6, 4.1)
    pins = union([box(x - 0.32, x + 0.32, -6.0, 1.9, 2.53, 3.17) for x in (-2.54, 0.0, 2.54)])
    callout_card([(pcb, PCB), (sw, DARK), (hd, DARK), (pins, METAL)],
                 [((7.5, 5.0, 0.8), "기판", "15 × 18.5 (만든 곳마다 조금 달라 자리는 16.5 × 20.5 까지)"),
                  ((0.0, 11.5, 6.8), "버튼", "앞을 본다. 2초 누르면 꺼지고, 짧게 누르면 한 번 잰다"),
                  ((2.54, -6.0, 2.85), "핀 3개", "S → 파이 5번 핀, − → 파이 6번 핀, 가운데는 비운다 (20cm 점퍼선 2가닥)")],
                 "전원 버튼 모듈 KY-004", "1개 · 약 2 g · 머리 왼쪽 받침 살에 끼운다 (나사 없음)", "구매_전원_버튼.png", az=-60, el=45)
    # 무선 동글 두 개
    def dongle(L, Wd, T, plug=12.0):
        return box(0, L - plug, -Wd / 2, Wd / 2, 0, T) , box(L - plug, L, -6.0, 6.0, T / 2 - 2.25, T / 2 + 2.25)
    a1, a2 = dongle(34, 16, 8); b1, b2 = dongle(31, 16, 9)
    boxed_card([(a1 + a2, DARK, "폰과 통신: NETGEAR WNA3100M"), (b1 + b2, "#3A4A5C", "탐지 동글: 예 ASUS USB-AC51")],
               "무선 동글 2개", "WNA3100M: 34 × 16 × 8, 4 g (NETGEAR 공식 값, 가지고 있는 것) · 탐지 동글: 모니터 모드가 되는 듀얼밴드 MT7610U (예: ASUS USB-AC51, 31 × 16 × 9), 몸통 45 × 18 × 10 안",
               "구매_무선_동글.png", el=32, gap=26)
    # ESP32 (XIAO ESP32-C3, 핀 머리 납땜) + FPC 안테나: 기판이 눕고 부품 면이 위, 핀은 아래
    xp = box(0, 21.0, 0, 17.8, 0, 1.2); sh = box(3.0, 15.0, 3.0, 14.8, 1.2, 3.2); usb = box(-1.5, 6.0, 4.9, 12.9, 1.2, 4.4)
    hdr = union([box(2.0, 19.0, y - 1.27, y + 1.27, -2.5, 0) for y in (1.6, 16.2)]); pins = union([box(2.0 + 2.54 * i + 0.95, 2.0 + 2.54 * i + 1.59, y - 0.32, y + 0.32, -8.5, -2.5) for i in range(7) for y in (1.6, 16.2)])
    ufl = cz(1.0, 1.2, 2.4, 18.0, 15.0)
    fpc = box(32.0, 72.0, 4.0, 14.0, 0, 0.3); coax = box(18.0, 32.0, 14.6, 15.4, 1.6, 2.4)
    callout_card([(xp, PCB), (sh, METAL), (usb, METAL), (hdr, DARK), (pins, METAL), (ufl, METAL), (fpc, DARK), (coax, DARK)],
                 [((9.0, 3.0, 3.2), "XIAO ESP32-C3", "21 × 17.8, 부품 높이 3.5. 핀 머리가 납땜된 것을 산다"),
                  ((2.0, 1.6, -8.5), "핀 머리 7핀 × 2", "점퍼선 4가닥: 5V(2번 핀, Y 케이블), GND(9번), D6 TX(10번), D7 RX(8번)"),
                  ((52.0, 14.0, 0.3), "FPC 안테나", "45 × 10 안, 동축선 8cm 이상. 보드와 같이 들어 있다"),
                  ((18.0, 15.0, 2.4), "안테나 단자 (U.FL)", "동축선 끝을 꾹 눌러 끼운다")],
                 "ESP32 (XIAO ESP32-C3) + FPC 안테나", "1개 · 약 3 g + 1 g · 보드는 뚜껑 안쪽 홈 받침에 세워 끼우고 막음쇠(22)로, 안테나는 머리 안 천장 홈에 누름판(21)으로",
                 "구매_ESP32.png", az=-60, el=38)


# ================================================================ 나사·너트, 전선, 마커 (평면 그림)
PXMM = 21.0                                     # 나사 그림 배율: 1 mm = 21 픽셀


def screw_side(d, cx, ytop, m, L, kind, dk, k):
    """나사 옆모습. ytop = 머리 윗면. kind '둥근' | '접시'. L 은 파는 길이 (둥근머리는 머리 밑부터, 접시머리는 머리 윗면부터)"""
    s = PXMM; r = m / 2 * s; R = dk / 2 * s
    if kind == "둥근":
        hb = ytop + k * s
        d.rounded_rectangle([cx - R, ytop, cx + R, hb], radius=int(k * s * 0.55), fill="#C9CCD1", outline=INK, width=2)
        d.rectangle([cx - R + 2, hb - 6, cx + R - 2, hb], fill="#C9CCD1")
        d.line([(cx - R, hb), (cx + R, hb)], fill=INK, width=2)
        y0, y1 = hb, hb + L * s
    else:
        kk = (dk - m) / 2                                                # 90° 접시
        hb = ytop + kk * s
        d.polygon([(cx - R, ytop), (cx + R, ytop), (cx + r, hb), (cx - r, hb)], fill="#C9CCD1", outline=INK)
        y0, y1 = hb, ytop + L * s
    d.rectangle([cx - r, y0, cx + r, y1 - 0.4 * s], fill="#C9CCD1", outline=INK, width=2)
    d.polygon([(cx - r, y1 - 0.4 * s), (cx + r, y1 - 0.4 * s), (cx + r * 0.7, y1), (cx - r * 0.7, y1)], fill="#C9CCD1", outline=INK)
    pitch = {3.0: 0.5, 2.5: 0.45, 2.0: 0.4}[m] * s
    y = y0 + pitch * 0.6
    while y < y1 - 0.5 * s:
        d.line([(cx - r, y + pitch * 0.35), (cx + r, y - pitch * 0.35)], fill="#7D838B", width=2); y += pitch
    # 치수: 오른쪽에 길이, 위에 머리 지름
    xr = cx + max(R, r) + 26
    ya = y0 if kind == "둥근" else ytop
    d.line([(xr, ya), (xr, y1)], fill=DIM, width=3)
    for yy in (ya, y1):
        d.line([(xr - 10, yy), (xr + 10, yy)], fill=DIM, width=3)
        d.line([(cx + max(R, r) + 4, yy), (xr + 4, yy)], fill=LINE, width=1)
    f = font(28, True)
    d.text((xr + 14, (ya + y1) / 2 - 18), mm(L), font=f, fill=INK)
    yh = ytop - 30
    d.line([(cx - R, yh), (cx + R, yh)], fill=DIM, width=3)
    for xx in (cx - R, cx + R):
        d.line([(xx, yh - 9), (xx, yh + 9)], fill=DIM, width=3)
    return y1


def screws():
    H = 1440; ROW = 650; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    head(im, "나사와 너트 (모두 같은 배율)", "나사 55개, 너트 7개. 모두 보통 미터 나사(볼트)다. 둥근머리는 머리 밑에서 끝까지, 접시머리는 머리 윗면에서 끝까지가 길이다. 숫자는 mm")
    rows = [
        [("M3 × 6 둥근머리", 3.0, 6, "둥근", 5.6, 2.4, 18, "모터 귀 6, 뚜껑 4, 부채꼴 기어 2, 마커 틀 4, 핀 막이 2"),
         ("M3 × 8 둥근머리", 3.0, 8, "둥근", 5.6, 2.4, 7, "모터 자리 4, 균형추 마개 2, 축 고정 1 (너트)"),
         ("M3 × 10 둥근머리", 3.0, 10, "둥근", 5.6, 2.4, 8, "받침 덮개 4 (접시머리도 됨), 멈춤 3, 고정 1"),
         ("M3 × 6 접시머리", 3.0, 6, "접시", 6.0, 0, 4, "판 틀 → 판 받이 (너트)"),
         ("M3 × 8 접시머리", 3.0, 8, "접시", 6.0, 0, 2, "누름쇠 → 판 틀 (너트)")],
        [("M2.5 × 6 둥근머리", 2.5, 6, "둥근", 5.0, 2.0, 4, "라즈베리파이"),
         ("M2 × 6 둥근머리", 2.0, 6, "둥근", 4.0, 1.6, 12, "드라이버 보드 3장 (4개씩)")]]
    fb, fr, fq = font(27, True), font(23), font(30, True)
    colw = (W - 80) / 5
    for ri, row in enumerate(rows):
        ytop = BAND + 110 + ri * ROW
        for ci, (name, m, L, kind, dk, k, n, where) in enumerate(row):
            cx = 40 + colw * (ci + 0.5) - 25
            screw_side(d, cx, ytop, m, L, kind, dk, k)
            hd_lab = ("머리 %s" % ("5.5~6" if m == 3.0 and kind == "둥근" else (mm(dk) if m == 3.0 else "약 %s" % mm(dk)))) + ("" if kind == "접시" else ", 높이 %s" % mm(k))
            bb = d.textbbox((0, 0), hd_lab, font=fr); d.text((cx - (bb[2] - bb[0]) / 2, ytop - 72), hd_lab, font=fr, fill=SUB)
            yt = ytop + 330
            for j, (t, ff, col) in enumerate(((name, fb, INK), ("%d개" % n, fq, DIM))):
                bb = d.textbbox((0, 0), t, font=ff); d.text((cx - (bb[2] - bb[0]) / 2, yt + j * 40), t, font=ff, fill=col)
            for j, l in enumerate(wrap(d, where, fr, colw - 24)):
                bb = d.textbbox((0, 0), l, font=fr); d.text((cx - (bb[2] - bb[0]) / 2, yt + 88 + j * 31), l, font=fr, fill=SUB)
    # M3 너트: 위에서 본 육각 + 옆모습
    s = PXMM; af = 5.5; R = af / np.sqrt(3) * s; cx = 40 + colw * 2.62; cy = BAND + 110 + ROW + 70
    pts = [(cx + R * np.cos(np.radians(60 * i)), cy + R * np.sin(np.radians(60 * i))) for i in range(6)]
    d.polygon(pts, fill="#C9CCD1", outline=INK); d.ellipse([cx - 1.5 * s, cy - 1.5 * s, cx + 1.5 * s, cy + 1.5 * s], fill=BG, outline=INK, width=2)
    yd = cy + af / 2 * s
    d.line([(cx - R - 40, cy - af / 2 * s), (cx - R - 40, yd)], fill=DIM, width=3)
    for yy in (cy - af / 2 * s, yd):
        d.line([(cx - R - 50, yy), (cx - R - 30, yy)], fill=DIM, width=3); d.line([(cx - R - 30, yy), (cx - 0.5 * R, yy)], fill=LINE, width=1)
    d.text((cx - R - 96, cy - 18), "5.5", font=font(28, True), fill=INK)
    x2 = cx + R + 55; d.rectangle([x2, cy - R, x2 + 2.4 * s, cy + R], fill="#C9CCD1", outline=INK, width=2)
    d.line([(x2, cy + R + 22), (x2 + 2.4 * s, cy + R + 22)], fill=DIM, width=3)
    d.text((x2 + 2.4 * s + 12, cy + R + 2), "2.4", font=font(28, True), fill=INK)
    yt = BAND + 110 + ROW + 330
    for j, (t, ff, col) in enumerate((("M3 너트", fb, INK), ("7개", fq, DIM))):
        bb = d.textbbox((0, 0), t, font=ff); d.text((cx + 40 - (bb[2] - bb[0]) / 2, yt + j * 40), t, font=ff, fill=col)
    for j, l in enumerate(wrap(d, "판 받이 4, 판 틀 뒷면 2, 축 고정 1. 맞변 5.5, 두께 2.4", fr, colw + 60)):
        bb = d.textbbox((0, 0), l, font=fr); d.text((cx + 40 - (bb[2] - bb[0]) / 2, yt + 88 + j * 31), l, font=fr, fill=SUB)
    # 오른쪽 아래: 나사 고르는 법
    x0 = 40 + colw * 3.6; y0 = BAND + 110 + ROW - 40
    note = ["사는 이름 예: 'M3 볼트 둥근머리 6mm', 'M3 접시머리 볼트 8mm'.",
            "태핑 나사(피스)나 플라스틱 전용 나사는 사지 않는다.",
            "너트 없는 자리는 출력 부품의 작은 구멍(M3 2.6, M2.5 2.1, M2 1.6)에 그대로 돌려 넣는다: 나사가 스스로 나사산을 낸다.",
            "길이를 꼭 지킨다: 모터 귀·마커 틀·핀 막이는 6, 멈춤·고정 나사는 10."]
    y = y0
    for t in note:
        for l in wrap(d, "· " + t, fr, W - x0 - 40):
            d.text((x0, y), l, font=fr, fill=INK if t is note[0] else SUB); y += 32
        y += 10
    im.save(os.path.join(OUT, "나사_너트.png")); print("나사_너트.png")


def plug_lr(d, x, y, kind, left):
    """ㄱ자 플러그 (옆모습). kind: 'A' (USB-A/C 쪽) | 'm' (마이크로 5핀)"""
    sgn = -1 if left else 1
    if kind == "A":
        d.rounded_rectangle([x - 24, y - 26, x + 24, y + 26], radius=8, fill=DARK)
        d.rectangle([x - 14, y + 26, x + 14, y + 70], fill=METAL, outline=INK, width=2)
    else:
        d.rounded_rectangle([x - 18, y - 20, x + 18, y + 20], radius=7, fill=DARK)
        d.rectangle([x - 10, y + 20, x + 10, y + 46], fill=METAL, outline=INK, width=2)


def misc():
    H = 1330; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    head(im, "전선과 작은 소모품", "길이는 그림의 비율과 다르다 (전선은 줄여 그렸다). 숫자는 mm, 전선 길이는 cm")
    fb, fr = font(28, True), font(24)
    # 1) 전원 케이블
    y = BAND + 110; xa, xb = 150, 900
    d.line([(xa + 24, y), (xb - 18, y)], fill=DARK, width=10)
    plug_lr(d, xa, y, "A", True); plug_lr(d, xb, y, "m", False)
    d.line([(xa, y - 60), (xb, y - 60)], fill=DIM, width=3)
    for xx in (xa, xb):
        d.line([(xx, y - 70), (xx, y - 50)], fill=DIM, width=3)
    d.text(((xa + xb) / 2 - 40, y - 104), "50 cm", font=fb, fill=INK)
    d.text((xa - 60, y + 84), "배터리 쪽: ㄱ자 USB-C 또는 USB-A", font=fr, fill=SUB)
    d.text((xb - 250, y + 84), "파이 쪽: ㄱ자 마이크로 5핀", font=fr, fill=SUB)
    tx = 1000
    d.text((tx, y - 92), "전원 케이블 1개", font=fb, fill=INK)
    for i, l in enumerate(wrap(d, "받침에서 머리까지 약 34 cm. 파이 쪽이 곧은 케이블이면 마이크로 5핀 ㄱ자 젠더(수-암)를 더한다", fr, W - tx - 40)):
        d.text((tx, y - 50 + 32 * i), l, font=fr, fill=SUB)
    # 2) 점퍼선
    def jumper(y, L, col, label, n):
        x0 = 150; x1 = x0 + L
        d.line([(x0 + 30, y), (x1 - 30, y)], fill=col, width=6)
        for xx in (x0, x1 - 30):
            d.rectangle([xx, y - 8, xx + 30, y + 8], fill=DARK)
        d.text((x1 + 24, y - 18), label, font=fb, fill=INK)
        d.text((x1 + 24 + d.textlength(label, font=fb) + 14, y - 15), n, font=fr, fill=SUB)
    y = BAND + 330
    d.text((150, y - 70), "점퍼선 (암-암, 끝이 2.54 mm 간격 소켓)", font=fb, fill=INK)
    jumper(y, 300, "#C2410C", "10 cm", "20가닥: 드라이버 3장에 18가닥")
    jumper(y + 60, 600, "#2B6CB0", "20 cm", "6가닥: 전원 버튼 2 + ESP32 4")
    # Y 분배 케이블 (암 1 -> 암 2)
    y = BAND + 535
    d.text((150, y - 108), "Y 분배 케이블 2개 (듀폰, 암 1개 → 암 2개, 납땜 필요 없음)", font=fb, fill=INK)
    d.rectangle([150, y - 8, 180, y + 8], fill=DARK); d.line([(180, y), (330, y)], fill="#C53030", width=6)
    for dy in (-38, 38):
        d.line([(330, y), (420, y + dy)], fill="#C53030", width=6); d.line([(420, y + dy), (500, y + dy)], fill="#C53030", width=6)
        d.rectangle([500, y + dy - 8, 530, y + dy + 8], fill=DARK)
    d.text((560, y - 52), "→ 드라이버 1의 + (2번 핀 쪽은 드라이버 3의 +)", font=fr, fill=SUB); d.text((560, y + 24), "→ 드라이버 2의 + (2번 핀 쪽은 ESP32 5V)", font=fr, fill=SUB)
    d.text((150, y + 52), "파이 4번 핀(5V) 쪽 하나, 2번 핀(5V) 쪽 하나", font=fr, fill=SUB)
    # 3) 케이블 타이
    y = BAND + 700
    d.text((150, y - 70), "케이블 타이 3~4개", font=fb, fill=INK)
    d.rectangle([150, y - 4, 150 + 100 * 6, y + 4], fill="#E8E4DC", outline=INK, width=2)
    d.rectangle([150 - 30, y - 13, 150, y + 13], fill="#E8E4DC", outline=INK, width=2)
    d.text((150 + 100 * 6 + 24, y - 18), "폭 2.5, 길이 100", font=fb, fill=INK)
    d.text((150, y + 26), "머리 아래에서 전선 셋(모터 1, 모터 2, 전원)을 한데 묶는다", font=fr, fill=SUB)
    # 4) 비비탄
    y = BAND + 910
    d.text((150, y - 80), "6 mm 비비탄 45개쯤 (없으면 양초)", font=fb, fill=INK)
    for i in range(9):
        cx = 170 + i * 66
        d.ellipse([cx - 24, y - 24, cx + 24, y + 24], fill="#F3F1EC", outline=INK, width=2)
        d.ellipse([cx - 13, y - 15, cx - 3, y - 6], fill="#FFFFFF")
    d.line([(146, y + 44), (194, y + 44)], fill=DIM, width=3); d.text((204, y + 28), "지름 6", font=fb, fill=INK)
    for i, l in enumerate(wrap(d, "받침 덮개의 둥근 테 홈에 한 줄로 깔고 그 위에 요크를 얹는다: 구슬 위로 굴러 돈다 (베어링과 같다). 두세 개 빈자리를 두고, 기름은 넣지 않는다", fr, W - 190)):
        d.text((150, y + 72 + 32 * i), l, font=fr, fill=SUB)
    # 5) 선택
    y = BAND + 1090
    d.text((150, y), "선택: 얇은 랜선 1개 (개발할 때 장치 안에 넣어 둘 때만) · 고무줄 1개 (85° 가까이 깊이 숙이는 자세를 자주 쓸 때)", font=fr, fill=SUB)
    im.save(os.path.join(OUT, "전선_소모품.png")); print("전선_소모품.png")


def marker():
    src = Image.open(os.path.join(paths.MARKER, "AR_마커.png")).convert("RGB")
    H = 900; im = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(im)
    head(im, "AR 마커 종이", "marker/AR_마커.pdf 를 A4에 실제 크기(100%)로 인쇄한다. 명함지(160~200g)를 권한다. 바깥 선대로 오리고 네 귀의 나사 자리를 따 낸다")
    w = 1100; h = int(src.size[1] * w / src.size[0]); x0 = (W - w) // 2 - 30; y0 = BAND + 90
    im.paste(src.resize((w, h), Image.LANCZOS), (x0, y0))
    d.rectangle([x0, y0, x0 + w, y0 + h], outline=LINE, width=2)
    yd = y0 + h + 40; d.line([(x0, yd), (x0 + w, yd)], fill=DIM, width=4)
    for xx in (x0, x0 + w):
        d.line([(xx, yd - 12), (xx, yd + 12)], fill=DIM, width=4)
    f = font(30, True); d.text((x0 + w / 2 - 50, yd + 14), "108 mm", font=f, fill=INK)
    xd = x0 + w + 40; d.line([(xd, y0), (xd, y0 + h)], fill=DIM, width=4)
    for yy in (y0, y0 + h):
        d.line([(xd - 12, yy), (xd + 12, yy)], fill=DIM, width=4)
    d.text((xd + 18, y0 + h / 2 - 20), "44 mm", font=f, fill=INK)
    for i, l in enumerate(wrap(d, "마커 두 개 (0번·1번, DICT_4X4_50), 한 변 34 mm. 머리 등판 아래쪽에 대고 마커 틀(15)로 덮어 M3 × 6 나사 4개로 조인다 (조립 안내 ④-4)", font(24), W - x0 - 60)):
        d.text((x0, yd + 66 + 32 * i), l, font=font(24), fill=SUB)
    im.save(os.path.join(OUT, "구매_AR_마커_종이.png")); print("구매_AR_마커_종이.png")


if __name__ == "__main__":
    if "p" in WANT:
        printed()
    if "b" in WANT:
        bought()
    if "s" in WANT:
        screws(); misc(); marker()
