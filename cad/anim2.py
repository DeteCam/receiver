"""재는 동작을 움직이는 그림(GIF)으로 만든다 (최종 구조, 미드나잇 블루).
    xvfb-run -a python3 anim2.py render 0 3      # 프레임 0, 3, 6 ... (셋으로 나눠 같이 돌린다)
    xvfb-run -a python3 anim2.py render 1 3
    xvfb-run -a python3 anim2.py render 2 3
    python3 anim2.py join                        # 그림 폴더에 09_재는_동작.gif
"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import paths

OUT = paths.IMG + '/'
TMP = os.path.join(paths.TMP, 'anim') + '/'
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, BG = "#22262B", "#5B6470", "#F4F2EE"
COLS = ["#2A78D6", "#EB6834", "#1BAF7A"]            # 스캔 1, 2, 3 의 면
TILT = float(np.degrees(np.arctan(np.sqrt(2))))
SIZE = 600
N_NOD, N_SCAN, N_YAW = 12, 20, 10
S0 = 0.0
YAWS = (-120.0, 0.0, 120.0)


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz, index=1)


def ease(t):
    return t * t * (3 - 2 * t)


def timeline():
    """프레임마다 (받침 각, 끄덕임 각, 판 각, 고리 진행 3개, 글, 머무는 시간 ms)"""
    F = [(0.0, 0.0, S0, (0, 0, 0), "세운 자세에서 시작", 900)]
    for i in range(1, N_NOD + 1):
        F.append((0.0, TILT * ease(i / N_NOD), S0, (0, 0, 0), "① 머리를 54.7° 기울인다", 80))
    for i in range(1, N_YAW + 1):
        F.append((YAWS[0] * ease(i / N_YAW), TILT, S0, (0, 0, 0), "② 받침을 −120°로 돌린다", 80))
    done = [0.0, 0.0, 0.0]
    for k in range(3):
        yaw = YAWS[k]
        for i in range(1, N_SCAN + 1):
            pr = list(done); pr[k] = i / N_SCAN
            F.append((yaw, TILT, S0 + 360.0 * i / N_SCAN, tuple(pr), "③ 판을 한 바퀴 돌려 잰다 (%d/3)" % (k + 1), 80))
        done[k] = 1.0
        if k < 2:
            for i in range(1, N_YAW + 1):
                F.append((yaw + 120.0 * ease(i / N_YAW), TILT, S0, tuple(done), "④ 받침을 120° 돌린다", 80))
    y, p, s, pr, _, _ = F[-1]
    F[-1] = (y, p, s, pr, "잰 면 세 개가 서로 직각이다", 2600)
    return F


def arc(a0, a1, r0=84.0, r1=90.0, h=1.4, z=17.0):
    """머리 좌표에서 판이 지나간 길 (부채꼴 띠)"""
    from manifold3d import CrossSection, Manifold
    n = max(2, int(abs(a1 - a0) / 5) + 1)
    t = np.radians(np.linspace(a0, a1, n))
    pts = [(r1 * np.cos(a), r1 * np.sin(a)) for a in t] + [(r0 * np.cos(a), r0 * np.sin(a)) for a in t[::-1]]
    return Manifold.extrude(CrossSection([pts]), 2 * h).translate([0, 0, z - h])


def frame(i, F):
    import parts2 as P, render
    from cadlib import cz
    yaw, pitch, spin, prog, text, _ = F[i]
    A = P.assembly(yaw, pitch, spin)
    for k, pr in enumerate(prog):
        if pr <= 0:
            continue
        g = (cz(90.0, 17 - 1.4, 17 + 1.4) - cz(84.0, 10, 24)) if pr >= 1 else arc(S0, S0 + 360.0 * pr)
        A["ring%d" % k] = (P.pose_head(g, YAWS[k], TILT), "X", COLS[k], False)
    f = TMP + 'raw_%03d.png' % i
    render.render(A, f, cam_az=-62, cam_el=22, size=(SIZE, SIZE), dist=980, focus=(0, 0, 150), ground=False)
    im = Image.open(f).convert('RGB'); d = ImageDraw.Draw(im)
    d.text((22, 16), text, font=font(27, True), fill=INK)
    for k in range(3):
        x = 22 + k * 30
        d.rectangle([x, SIZE - 40, x + 22, SIZE - 18], fill=COLS[k] if prog[k] > 0 else "#D5D2CB")
    d.text((22 + 3 * 30 + 6, SIZE - 45), "잰 면", font=font(21), fill=SUB)
    im.save(TMP + 'f_%03d.png' % i); os.remove(f)


if __name__ == "__main__":
    os.makedirs(TMP, exist_ok=True)
    F = timeline()
    if sys.argv[1] == "render":
        a, n = int(sys.argv[2]), int(sys.argv[3])
        for i in range(a, len(F), n):
            frame(i, F)
        print("done", a, len(F))
    else:
        ims = [Image.open(TMP + 'f_%03d.png' % i).convert('RGB') for i in range(len(F))]
        strip = Image.new('RGB', (SIZE * 6, SIZE)); pick = np.linspace(0, len(ims) - 1, 6).astype(int)      # 모든 프레임이 같은 색판을 쓰게 해서 깜빡임을 없앤다
        for j, i in enumerate(pick):
            strip.paste(ims[i], (j * SIZE, 0))
        pal = strip.quantize(colors=96, method=Image.Quantize.MEDIANCUT)
        q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in ims]
        q[0].save(OUT + '09_재는_동작.gif', save_all=True, append_images=q[1:], duration=[f[5] for f in F], loop=0, optimize=False, disposal=1)
        print("gif", len(q), "frames", os.path.getsize(OUT + '09_재는_동작.gif') // 1024, "KB")
