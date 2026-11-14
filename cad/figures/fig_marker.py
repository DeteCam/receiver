"""AR 마커 그림 (14_AR_마커.png): 마커 종이를 다는 자리와, 폰이 그것으로 수신기의 자리를 아는 순서.   xvfb-run -a python3 fig_marker.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import parts2 as P
import render, paths
import marker as MK
from cadlib import *
from p2 import *
from PIL import Image, ImageDraw, ImageFont
from fig_change_view import HP_factory

OUT = paths.IMG + '/'; TMP = paths.TMP + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, LINE, BG, NAVY = "#22262B", "#5B6470", "#B9BEC6", "#F4F2EE", "#1B2A57"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)


def tags(im, pix, items, fs=24):
    """그림 위에 이름표: (이름, 글, 이름표를 둘 곳의 x·y 차이)"""
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


HP = HP_factory(Y_T, S_T, H_T)
yaw, pitch, spin = 0.0, TILT3, 0.0
A = P.assembly(yaw, pitch, spin)
z0 = MK.Z_BOTTOM; c = MK.SIZE / 6; cells = []
for i, cx_ in zip(MK.IDS, (MK.CX, -MK.CX)):                      # 등판을 뒤에서 보면 머리의 +X 가 왼쪽: 0번 마커가 왼쪽
    b = MK.bits(i)
    for r in range(6):
        for q in range(6):
            if b[r, q]:
                cells.append(box(cx_ + MK.SIZE / 2 - (q + 1) * c, cx_ + MK.SIZE / 2 - q * c, HY1 + 0.3, HY1 + 0.6, MK.ZC + MK.SIZE / 2 - (r + 1) * c, MK.ZC + MK.SIZE / 2 - r * c))
A["mk_paper"] = (P.pose_head(box(-MK.STRIP_W / 2, MK.STRIP_W / 2, HY1, HY1 + 0.3, z0, z0 + MK.STRIP_H), yaw, pitch), "H", "#FFFFFF", False)
A["mk_cells"] = (P.pose_head(union(cells), yaw, pitch), "H", "#111111", False)
pts = {"mk": HP((0, HY1, MK.ZC + MK.SIZE / 2 + 3), yaw, pitch), "vent": HP((-14, HY1, -36), yaw, pitch), "blade": HP((80, -20, 20), yaw, pitch), "front": np.array([60.0, 0.0, 6.0])}
pix = render.render(A, TMP + 'marker_a.png', cam_az=205, cam_el=44, dist=600, focus=tuple(HP((0, 0, -55), yaw, pitch)), size=(1150, 1100), label_pts=pts, ground=False)
pa = tags(Image.open(TMP + 'marker_a.png').convert("RGB"), pix, [("mk", "AR 마커 종이 (108 × 44mm)\n틀로 덮고 M3 나사 4개로 조인다", 110, 265), ("vent", "통풍 틈은 가리지 않는다", 170, -150), ("blade", "판은 옆(0도)에 둔다", 150, -120)], fs=24)

W, H = 2360, 1560; m = 40
im = Image.new('RGB', (W, H), BG); d = ImageDraw.Draw(im)
d.text((m, 28), "AR 마커: 폰이 수신기가 놓인 자리와 방향을 알아내는 방법", font=font(50, True), fill=INK)
d.text((m, 100), "수신기는 ‘수신기에서 본’ 방향과 구간을 계산한다. 그것을 방 안에 그리려면 수신기가 방 어디에, 어느 쪽을 보고 놓였는지 알아야 한다.", font=font(28), fill=SUB)
im.paste(pa, (m, 180))
d.text((m + 10, 180 + 1100 + 16), "재는 자세(끄덕임 54.7°)에서 뒤쪽 위에서 본 모습. 등판이 위를 55° 올려다봐서, 서서 내려다보는 폰에 잘 보인다.", font=font(24), fill=SUB)

X0 = m + 1150 + 60
d.text((X0, 190), "마커 띠 (실제 크기로 인쇄: marker/AR_마커.pdf)", font=font(34, True), fill=INK)
strip = Image.open(os.path.join(paths.MARKER, "AR_마커.png")).convert("RGB"); sw = 900; sh = int(strip.size[1] * sw / strip.size[0])
strip = strip.resize((sw, sh), Image.LANCZOS); sx, sy = X0 + 40, 270; im.paste(strip, (sx, sy)); d.rectangle([sx, sy, sx + sw, sy + sh], outline=LINE, width=2)
d.line([(sx, sy + sh + 26), (sx + sw, sy + sh + 26)], fill=SUB, width=3)
for xx in (sx, sx + sw):
    d.line([(xx, sy + sh + 14), (xx, sy + sh + 38)], fill=SUB, width=3)
d.text((sx + sw / 2, sy + sh + 56), "108mm (마커 한 변 34mm, 번호 0과 1, ArUco 4×4)", font=font(25), fill=SUB, anchor="mm")
d.line([(sx + sw + 26, sy), (sx + sw + 26, sy + sh)], fill=SUB, width=3)
for yy in (sy, sy + sh):
    d.line([(sx + sw + 14, yy), (sx + sw + 38, yy)], fill=SUB, width=3)
d.text((sx + sw + 50, sy + sh / 2), "44mm", font=font(25), fill=SUB, anchor="lm")

ys = sy + sh + 120
d.text((X0, ys), "쓰는 순서", font=font(34, True), fill=INK)
steps = [("1", "수신기가 머리를 숙인다", "재고 난 뒤의 자세 그대로다. 따로 시키려면 POST /api/present"),
         ("2", "폰으로 마커 두 개를 비춘다", "50cm~1m 에서. 번쩍이는 테이프로 덮으면 못 읽는다"),
         ("3", "앱이 수신기에 묻는다: GET /api/marker", "지금 관절 각도에서 마커 귀 8개가 받침에서 어디인지 (m)"),
         ("4", "귀 8개로 수신기의 자리·방향을 계산한다", "OpenCV 의 ArUco 찾기 + solvePnP. AR 좌표로 옮긴다"),
         ("5", "POST /api/pose 로 자리를 알려 준다", "수신기가 추정 구간을 방 좌표로 계산해 돌려준다 (/api/zone)")]
for i, (n, t1, t2) in enumerate(steps):
    y = ys + 66 + i * 118
    d.ellipse([X0, y + 4, X0 + 56, y + 60], fill=NAVY); d.text((X0 + 28, y + 32), n, font=font(30, True), fill="#FFFFFF", anchor="mm")
    d.text((X0 + 76, y), t1, font=font(31, True), fill=INK); d.text((X0 + 76, y + 46), t2, font=font(25), fill=SUB)
y = ys + 66 + 5 * 118 + 10
d.text((X0, y), "마커가 머리에 붙어 있어도 된다: 수신기가 관절 각도를 알기 때문에 받침 기준 좌표를 계산해 준다.", font=font(25), fill=INK)
d.text((X0, y + 40), "아직 해 보지 않은 것: 실제 폰으로 읽기, 자리 오차 재기 (시뮬레이션에서는 2cm, 1.5°로 가정했다).", font=font(25), fill=SUB)
im.save(OUT + '14_AR_마커.png'); print("saved", OUT + '14_AR_마커.png', pix)
