"""수신기 혼자 동작하는 구조 그림 (12_혼자_동작.png).   xvfb-run -a python3 fig_alone.py
왼쪽: 무선 동글이 꽂히는 자리(머리 아래 USB)와 버튼·상태 불.  오른쪽: 누가 무엇을 하는지, 쓰는 순서."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import render, paths
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont
from fig_change_view import HP_factory

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG, CARD = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE", "#FFFFFF"
BLUE, ORANGE, GREEN, NAVY = "#2A78D6", "#EB6834", "#1E8E5A", "#1B2A57"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
HP = HP_factory(Y_T, S_T, H_T)


def tags(im, pix, items, fs=24):
    d = ImageDraw.Draw(im); f = font(fs, True)
    for key, text, dx, dy in items:
        ax, ay = pix[key]; lines = text.split("\n")
        w = max(d.textlength(l, font=f) for l in lines); hh = int(fs * 1.34) * len(lines)
        tx, ty = ax + dx, ay + dy
        x0 = tx - (w + 20 if dx < 0 else 0); y0 = ty - hh / 2 - 6
        d.line([(ax, ay), (x0 + (w + 20 if dx < 0 else 0), ty)], fill=INK, width=2)
        d.ellipse([ax - 6, ay - 6, ax + 6, ay + 6], fill="#FFFFFF", outline=INK, width=2)
        d.rounded_rectangle([x0, y0, x0 + w + 20, y0 + hh + 12], 8, fill="#FFFFFF", outline="#C9CCD1", width=2)
        for i, l in enumerate(lines):
            d.text((x0 + 10, y0 + 5 + i * int(fs * 1.34)), l, font=f, fill=INK)
    return im


# ---------------------------------------------------------------- 왼쪽 위: 무선 동글 자리 (머리 껍데기를 비치게)
yaw, pitch, spin = 0.0, TILT3, 90.0
A = P.assembly(yaw, pitch, spin)
A["dongle"] = (A["dongle"][0], "H", ORANGE, False)
TEAL = "#1F9E8F"
A["dongle2"] = (A["dongle2"][0], "H", TEAL, False)
hp = lambda p: HP(p, yaw, pitch)
pts = {"dongle": hp((BX0 + DONGLE["x"], -PI_T / 2 - DONGLE["up"], BZ0 - 14)), "dongle2": hp((BX0 + DONGLE2["x"], -PI_T / 2 - DONGLE2["up"], BZ0 - 14)), "ant": hp((0, -2.5, 0)), "pi": hp((-20, -3, -45))}
ghost = ("head_shell", "head_lid", "crown", "sector", "blade", "rotor_bar", "foil", "yoke", "pin_r", "pin_l")
pix = render.render(A, TMP + 'alone_a.png', cam_az=152, cam_el=14, dist=520, focus=(-5, 0, 118), size=(900, 760), label_pts=pts, ground=False, ghost=ghost, ghost_opacity=0.2)
pa = tags(Image.open(TMP + 'alone_a.png').convert("RGB"), pix, [("dongle", "무선 동글 (주황)\n폰과 통신", 150, 120), ("dongle2", "탐지 동글 (청록)\n둘레 기기를 엿듣는다. 없어도 된다", 60, 230), ("ant", "내장 Wi-Fi 안테나\nCSI 를 받는 데만 쓴다", -90, -130), ("pi", "라즈베리파이\n캡처 · 계산 · 모터", -230, 60)], fs=22)

# ---------------------------------------------------------------- 왼쪽 아래: 버튼과 상태 불 (앞에서)
A2 = P.assembly(0.0, 0.0, 90.0)
c_btn = HP((BTN_X, HY0, BTN_Z), 0, 0); c_led = HP(((LED_WIN[0] + LED_WIN[1]) / 2, HY0, (LED_WIN[2] + LED_WIN[3]) / 2), 0, 0)
pix2 = render.render(A2, TMP + 'alone_b.png', cam_az=22, cam_el=12, dist=640, focus=(0, 0, 120), size=(900, 760), label_pts={"btn": c_btn, "led": c_led}, ground=False)
pb = tags(Image.open(TMP + 'alone_b.png').convert("RGB"), pix2, [("btn", "버튼\n짧게 = 한 번 재기\n2초 = 안전하게 끄기", -50, 190), ("led", "초록 불 (상태)\n천천히 깜빡 = 준비\n빠르게 = 재는 중, 켜짐 = 끝", 150, -120)], fs=22)

# ---------------------------------------------------------------- 한 장으로
W, H = 2360, 1720; m = 40
sheet = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(sheet)
d.text((m, 28), "수신기 혼자 동작: 노트북 없이 찾고, 재고, 계산하고, 폰에 보여 준다", font=font(50, True), fill=INK)
d.text((m, 100), "내장 Wi-Fi 는 CSI 를 받는 데만 쓴다. 폰과는 무선 동글로 통신하고, 둘레 기기는 탐지 동글로 엿듣는다. 랜선은 개발할 때만 꽂는다.", font=font(28), fill=SUB)
sheet.paste(pa, (m, 170)); sheet.paste(pb, (m, 170 + 760 + 20))


def box(x, y, w, h, title, lines, col=NAVY, fill=CARD):
    d.rounded_rectangle([x, y, x + w, y + h], 18, fill=fill, outline=col, width=4)
    d.text((x + 26, y + 18), title, font=font(34, True), fill=col)
    for i, l in enumerate(lines):
        d.text((x + 26, y + 72 + i * 42), l, font=font(27), fill=INK)


def arrow(x0, y0, x1, y1, col, text=None, dash=False, both=True):
    if dash:
        n = int(np.hypot(x1 - x0, y1 - y0) // 26)
        for i in range(n):
            a, b = i / n, (i + 0.55) / n
            d.line([(x0 + (x1 - x0) * a, y0 + (y1 - y0) * a), (x0 + (x1 - x0) * b, y0 + (y1 - y0) * b)], fill=col, width=6)
    else:
        d.line([(x0, y0), (x1, y1)], fill=col, width=6)
    for (px, py, qx, qy) in ((x1, y1, x0, y0),) + (((x0, y0, x1, y1),) if both else ()):
        u = np.array([px - qx, py - qy], float); u /= np.linalg.norm(u); n = np.array([-u[1], u[0]])
        d.polygon([(px, py), tuple(np.array([px, py]) - 24 * u + 12 * n), tuple(np.array([px, py]) - 24 * u - 12 * n)], fill=col)
    if text:
        tw = max(d.textlength(l, font=font(25, True)) for l in text.split("\n"))
        for i, l in enumerate(text.split("\n")):
            d.text(((x0 + x1) / 2 - tw / 2, (y0 + y1) / 2 - 78 + i * 32), l, font=font(25, True), fill=col)


X0 = m + 900 + 50; Y0 = 190
box(X0, Y0, 640, 560, "수신기 (라즈베리파이 3B+)", ["· 탐지 동글: 둘레 기기 엿듣기", "· 내장 Wi-Fi: 카메라 신호(CSI) 받기", "· 모터 3개로 판을 돌려 방향을 잰다", "· 여러 자리의 결과를 합쳐", "  추정 구간을 구한다", "· 켜면 저절로 시작한다 (node.py)", "· 전원: 받침 속 보조배터리", "· 버튼 하나, 초록 불 하나"])
box(X0 + 900, Y0, 430, 330, "폰", ["· 브라우저 (지금)", "· AR 앱 (만들 것)", "· 자리 알려 주기, 재기,", "  구간 보기"], col=BLUE)
box(X0 + 900, Y0 + 400, 430, 160, "노트북", ["· 개발할 때만 (SSH)"], col=SUB)
arrow(X0 + 640 + 14, Y0 + 165, X0 + 900 - 14, Y0 + 165, BLUE, "무선 동글의 Wi-Fi\nhttp://192.168.4.1")
arrow(X0 + 640 + 14, Y0 + 480, X0 + 900 - 14, Y0 + 480, SUB, "랜선", dash=True)
ys = Y0 + 640
d.text((X0, ys), "쓰는 순서", font=font(36, True), fill=INK)
steps = [("1", "버튼을 눌러 켠다", "초록 불이 천천히 깜빡이면 준비된 것 (약 40초)"),
         ("2", "폰을 수신기의 Wi-Fi 에 붙인다", "이름 DeteCam-RX. 브라우저에서 192.168.4.1"),
         ("3", "탐지: ‘둘레 기기 찾기’", "약 30초. 큰 프레임을 많이 내보내는 기기 = 후보"),
         ("4", "판별: 시키는 대로 걷다 멈추다", "55초. 전송량이 따라 변하면 이 방을 찍는 카메라"),
         ("5", "위치: 수신기를 놓고 ‘재기’", "한 자리 약 1분. 권하는 자리로 옮겨 보통 3자리"),
         ("6", "표시: 구간 안의 물건을 살펴본다", "AR 앱은 등판의 마커로 자리를 맞춘다")]
for i, (n, t1, t2) in enumerate(steps):
    x = X0 + (i % 2) * 680; y = ys + 70 + (i // 2) * 150
    d.ellipse([x, y + 4, x + 56, y + 60], fill=NAVY); d.text((x + 28, y + 32), n, font=font(30, True), fill="#FFFFFF", anchor="mm")
    d.text((x + 76, y), t1, font=font(31, True), fill=INK); d.text((x + 76, y + 46), t2, font=font(25), fill=SUB)
d.text((X0, ys + 70 + 3 * 150 + 20), "더 살 것은 탐지 동글 하나: 모니터 모드가 되는 USB 무선 동글 (칩 RT5370·MT7601U 등). 없어도 돈다.", font=font(27, True), fill=INK)
d.text((X0, ys + 70 + 3 * 150 + 64), "폰과의 통신은 가지고 있는 NETGEAR WNA3100M 으로 한다. ‘자기 Wi-Fi 만들기’를 못 하면 폰 핫스팟에 붙는다 (pi/link_setup.sh).", font=font(25), fill=SUB)
d.text((X0, ys + 70 + 3 * 150 + 104), "재는 동안에는 폰과의 통신을 줄인다: 동글이 CSI 안테나에서 9cm 떨어져 있어서, 동글이 보내는 전파가 측정을 흐릴 수 있다.", font=font(25), fill=SUB)
sheet.save(OUT + '12_혼자_동작.png')
print("saved", OUT + '12_혼자_동작.png', pix, pix2)
