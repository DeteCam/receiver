"""이번에 고친 곳 그림 (11_전선과_전원_고친_곳.png).   xvfb-run -a python3 fig_change.py
  ① 전선: 고치기 전 / 고친 뒤 (뒤에서 같은 자리로 본 모습. 전선은 주황색)
  ② 전원 버튼과 상태 불 창 (머리 뚜껑)   ③ 모터 2 맞춤 링   ④ 받침 뒤 랜선 홈
'고치기 전' 그림은 옛 모델로 그려 둔 것(asset/before_wiring.png)을 쓴다."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import render, paths
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont
from fig_change_view import CAM, POSE, CABLE, HP_factory, tube

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG = "#22262B", "#5B6470", "#8A929C", "#F4F2EE"
BAD, GOOD = "#C0392B", "#1E8E5A"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
HP = HP_factory(Y_T, S_T, H_T)


def tags(im, pix, items, fs=24):
    """그림 위에 바로 이름표를 단다. items: [(키, 글, dx, dy)] - 점에서 (dx, dy) 만큼 떨어진 곳에 글을 놓고 선으로 잇는다."""
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


# ---------------------------------------------------------------- ① 전선: 고친 뒤 (고치기 전과 같은 카메라)
yaw, pitch, spin = POSE
hp = lambda p: HP(p, yaw, pitch)
A = P.assembly(yaw, pitch, spin)
sx, sy = P.m1_shaft(); ux, uy = M1_BODY_DIR
m1c = np.array([sx + ux * (M_OFFSET + 17.0), sy + uy * (M_OFFSET + 17.0), DECK_Z1 + 8.0])
cx_, cz_ = P.m2_cap_xz(); m2c = np.array([cx_, ARM_Y1 + 8.0, cz_])
x0, x1, z0, z1 = P.M2_SLOT; slot = np.array([(x0 + x1) / 2, CHEEK_Y - 4.0, (z0 + z1) / 2]); R = BASE["S"]["R"]
A["w_m1"] = (tube([m1c, m1c + [0, 0, 16], hp((DRV_X0 + 17.5, -12, HZ0 - 4)), hp((DRV_X0 + 17.5, -8, DRV_ZC[2] - 10))], 2.2), "X", CABLE, False)
A["w_m2"] = (tube([m2c, (cx_, ARM_Y0 - 3.0, cz_), slot, (-40.0, 30.0, 60.0), hp((DRV_X0 + 24, -14, HZ0 - 4)), hp((DRV_X0 + 24, -8, DRV_ZC[1] - 10))], 2.2), "X", CABLE, False)
A["w_usb"] = (tube([(0.0, -R + BATT_IN - 8.0, 10.0), (20.0, -R + BATT_IN + 6.0, 24.0), (3.0, 0, 26.0), (3.0, 0, TUBE_Z1 + 6), hp((BX0 - 6.6, -4.2, HZ0 - 6)), hp((BX0 - 6.6, -4.2, BZ1 - 14))], 2.2), "X", CABLE, False)
pts = {"m2": m2c + [0, -3.0, 0], "cap": np.array([-R - 1.0, 0.0, 24.0]), "hood": np.array([WING_X0 + 12.0, -10.0, HOOD_Z0 + 12.0]), "rear": np.array([WING_X0, -30.0, 52.0])}
pix_a = render.render(A, TMP + 'chg_after.png', label_pts=pts, ground=True, ground_z=-0.4, **CAM)
pix_b = {k: tuple(v) for k, v in json.load(open(os.path.join(HERE, "asset", "before_wiring.png.json"))).items()}
before = tags(Image.open(os.path.join(HERE, "asset", "before_wiring.png")).convert("RGB"), pix_b, [
    ("m2", "모터 2 전선\n팔 위로 넘어온다", -60, -190), ("m1", "모터 1 전선\n그대로 보인다", -150, 130),
    ("usb", "전원 플러그가\n받침 밖으로 나온다", 30, -150), ("lan", "랜선이 머리 아래에서\n바닥으로 늘어진다", 120, -70)])
after = tags(Image.open(TMP + 'chg_after.png').convert("RGB"), pix_a, [
    ("m2", "모터 2 전선\n2cm만 보이고 팔 안으로", -70, -200), ("hood", "뒷가림 벽과 지붕\n전선은 이 상자 속", 180, -150),
    ("cap", "랜선 홈 마개\n랜선은 개발할 때만 쓴다", 160, 60)])

# ---------------------------------------------------------------- ② 전원 버튼과 상태 불 창
A2 = P.assembly(0.0, 0.0, 90.0)
c_btn = HP((BTN_X, HY0, BTN_Z)); c_led = HP(((LED_WIN[0] + LED_WIN[1]) / 2, HY0, (LED_WIN[2] + LED_WIN[3]) / 2))
foc = HP((-20.0, HY0, -30.0))
pix2 = render.render(A2, TMP + 'chg_btn.png', cam_az=14, cam_el=10, dist=300, focus=tuple(foc), size=(740, 660), label_pts={"btn": c_btn, "led": c_led}, ground=False)
p2 = tags(Image.open(TMP + 'chg_btn.png').convert("RGB"), pix2, [("btn", "전원 버튼", -150, 90), ("led", "상태 불 창\n(초록 불이 꺼지면 다 꺼진 것)", 50, -130)], fs=22)

# ---------------------------------------------------------------- ③ 모터 2 맞춤 링 (바깥으로 빼서 늘어놓은 모습)
mb, mc = v1.motor_body_dummy(); C = P.COL
px, pz = P.pin2_xz()
A3 = {"yoke": (P.yoke(), "Y", C["body"], True),
      "pinion": (P.pinion2().translate([0, 24, 0]), "Y", C["accent"], True), "ring": (P.m2_ring().translate([0, 34, 0]), "Y", "#F08A3C", True),
      "motor2": (mb.transform(P.motor2_T()).translate([0, 50, 0]), "Y", C["metal"], False), "motor2_cap": (mc.transform(P.motor2_T()).translate([0, 50, 0]), "Y", C["cap"], False),
      "shaft": (v1.motor_shaft_dummy().transform(P.motor2_T()).translate([0, 50, 0]), "Y", C["metal"], False)}
pts3 = {"ring": np.array([px, ARM_Y1 + 34 - 2, pz + 8.0]), "pinion": np.array([px, PIN2_Y1 + 24 - 3, pz + 7.0]), "hole": np.array([px + 8.0, ARM_Y1, pz]), "motor": np.array([px - 8.0, ARM_Y1 + 50 + 10, pz + 6.0])}
pix3 = render.render(A3, TMP + 'chg_ring.png', cam_az=128, cam_el=20, dist=235, focus=(px, ARM_Y1 + 28, pz + 2), size=(740, 660), label_pts=pts3, ground=False)
p3 = tags(Image.open(TMP + 'chg_ring.png').convert("RGB"), pix3, [("ring", "맞춤 링 (새 부품)", -60, -190), ("pinion", "작은 기어", 90, -150), ("hole", "팔 구멍 지름 16.5", 110, 130), ("motor", "모터 2", -110, 150)], fs=22)

# ---------------------------------------------------------------- ④ 받침 뒤 랜선 홈과 마개 (마개를 위로 조금 뺀 모습)
A4 = {k: v for k, v in P.assembly(0.0, 0.0, 90.0).items() if k in ("tub", "battery", "usb_a")}
A4["cap"] = (P.lan_cap().translate([0, 0, 15.0]), "B", "#EB6834", True)
pts4 = {"notch": np.array([-R - 1.0, 0.0, LAN_NOTCH[1] + 1.0]), "cap": np.array([-R - 1.0, 0.0, 40.0])}
pix4 = render.render(A4, TMP + 'chg_lan.png', cam_az=163, cam_el=22, dist=300, focus=(-R + 10, 0, 34), size=(740, 660), label_pts=pts4, ground=True, ground_z=-0.4)
p4 = tags(Image.open(TMP + 'chg_lan.png').convert("RGB"), pix4, [("notch", "랜선 홈 (폭 13mm)", 70, 120), ("cap", "홈 마개 (새 부품)\n덮개를 덮기 전에 위에서 끼운다", 40, -150)], fs=22)

# ---------------------------------------------------------------- 한 장으로
W = 2260; m = 30
sheet = Image.new('RGB', (W, 150 + 1000 + 190 + 660 + 150), BG); d = ImageDraw.Draw(sheet)
d.text((m + 10, 28), "전선과 전원: 처음 설계에서 고친 곳", font=font(50, True), fill=INK)
d.text((m + 10, 100), "① 전선 — 뒤에서 본 모습 (재는 자세 54.7°). 전선은 알아보기 쉽게 주황색으로 그렸다.", font=font(28), fill=SUB)
sheet.paste(before, (m, 150)); sheet.paste(after, (W - m - 1100, 150))
d.text((m + 10, 1160), "고치기 전", font=font(36, True), fill=BAD)
d.text((m + 10, 1210), "모터 전선 2가닥, 전원 플러그, 랜선이 모두 겉에 드러나 있었다.", font=font(27), fill=SUB)
d.text((W - m - 1090, 1160), "고친 뒤", font=font(36, True), fill=GOOD)
d.text((W - m - 1090, 1210), "요크에 전선 상자와 뒷가림 벽. 전원선은 가운데 통으로. 랜선은 없앴다 (폰과는 무선으로).", font=font(27), fill=SUB)
cv = json.load(open(os.path.join(paths.RES, "cable_vis.json"), encoding="utf-8"))["summary"]
d.text((m + 10, 1252), "보이는 전선 길이 (사방 16곳에서 본 평균):  세웠을 때 %.0f → %.0f mm,  재는 자세 %.0f → %.0f mm.  재는 자세에서 앞쪽은 머리와 앞벽 사이 1.5cm 틈으로 아직 보인다." % (
    cv["0"]["before_mean"], cv["0"]["after_mean"], cv["55"]["before_mean"], cv["55"]["after_mean"]), font=font(27, True), fill=INK)
y2 = 150 + 1000 + 190
caps = [("② 전원 버튼", "머리 뚜껑의 버튼: 2초 누르면 안전하게 꺼지고,\n꺼진 뒤 한 번 누르면 다시 켜진다."),
        ("③ 모터 2 맞춤 링", "링이 팔 구멍을 채워 모터 자리를 잡는다.\n기어 사이가 흔들리지 않는다."),
        ("④ 랜선 홈과 마개", "혼자 동작할 때는 랜선이 없다: 홈을 마개로 막는다.\n랜선을 쓰려면 조립할 때 가운데 통으로 넣는다 (마개 없이).")]
for i, (im, (t1, t2)) in enumerate(zip((p2, p3, p4), caps)):
    x = m + i * (740 + 5)
    sheet.paste(im, (x, y2))
    d.text((x + 10, y2 + 668), t1, font=font(32, True), fill=INK)
    for j, l in enumerate(t2.split("\n")):
        d.text((x + 10, y2 + 714 + j * 34), l, font=font(25), fill=SUB)
sheet.save(OUT + '11_전선과_전원_고친_곳.png')
print("saved", OUT + '11_전선과_전원_고친_곳.png', {k: (int(v[0]), int(v[1])) for k, v in pix_a.items()}, pix2, pix3, pix4)
