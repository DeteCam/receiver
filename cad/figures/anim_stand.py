"""스탠드 받이 영상: 치수를 보여 주고, 너트를 넣고, 받침 통 밑에 나사 4개로 조이고, 스탠드에 돌려 끼운다.

    xvfb-run -a python3 anim_stand.py              # docs/videos/스탠드_받이.mp4 (약 24초)
    xvfb-run -a python3 anim_stand.py still 3 9 16 # 그 시각(초)의 화면을 임시 폴더에 (확인용)
    python3 anim_stand.py join                     # 조립 영상 뒤에 이어 붙인다 (조립_시뮬레이션.mp4 를 고쳐 쓴다)

조립 영상(anim_assembly.py)과 같은 크기·속도(1280x720, 24fps)다. 치수는 p2.py 의 값이다.
"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import parts2 as P
import paths
from cadlib import *
from p2 import *
from render import to_pv

pv.OFF_SCREEN = True
W, H, FPS = 1280, 720, 24
OUT = os.path.join(paths.VID, "스탠드_받이.mp4"); MAIN = os.path.join(paths.VID, "조립_시뮬레이션.mp4")
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, BG, HOT, PART, STEEL, TUB = "#22262B", "#5B6470", "#F4F2EE", "#EB6834", "#27348B", "#C3C8CF", "#8A9BC4"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
T1, T2, T3, T4 = 6.0, 9.5, 5.5, 4.0                      # 치수, 조이기, 스탠드에 끼우기, 세운 모습
TOTAL = T1 + T2 + T3 + T4


def Tm(v=(0, 0, 0), rz=0.0):
    a = np.radians(rz); c, s = np.cos(a), np.sin(a); M = np.eye(4); M[:2, :2] = [[c, -s], [s, c]]; M[:3, 3] = v; return M


def ease(u):
    u = float(np.clip(u, 0, 1)); return u * u * (3 - 2 * u)


def screw():
    m = union([cz(1.5, 0.0, 10.0), cz(2.8, -2.4, 0.0)]) - union([box(-2.0, 2.0, -0.4, 0.4, -2.5, -1.2), box(-0.4, 0.4, -2.0, 2.0, -2.5, -1.2)])
    return m                                              # 머리 밑이 z=0, 몸통이 +z 로 10mm


class Scene:
    def __init__(self):
        p = pv.Plotter(off_screen=True, window_size=(W, H)); p.set_background(BG); self.p = p
        try:
            p.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        kw = dict(smooth_shading=True, specular=0.25, specular_power=18, ambient=0.3, diffuse=0.8)
        self.a = {}
        self.a["mount"] = p.add_mesh(to_pv(P.stand_mount()), color=PART, **kw)
        self.a["nut"] = p.add_mesh(to_pv(P.stand_nut()), color=STEEL, specular=0.7, specular_power=40, ambient=0.3, diffuse=0.7, smooth_shading=True)
        self.a["tub"] = p.add_mesh(to_pv(P.tub()), color=TUB, opacity=0.55, **kw)
        for i in range(4):
            self.a["s%d" % i] = p.add_mesh(to_pv(screw()), color=STEEL, specular=0.7, specular_power=40, ambient=0.3, diffuse=0.7, smooth_shading=True)
        pole = union([cz(9.0, -260.0, -30.0), cz(15.0, -30.0, -24.0), cz(3.175, -24.0, -12.5)])          # 스탠드 꼭대기: 기둥, 턱, 1/4인치 나사
        self.a["pole"] = p.add_mesh(to_pv(pole), color="#3A3F46", **kw)
        self.card = Image.open(os.path.join(paths.IMG, "부품", "구매_스탠드.png")).convert("RGB")

    def cam(self, az, el, dist, focus):
        a, e = np.radians(az), np.radians(el); f = np.asarray(focus, float)
        self.p.camera.position = tuple(f + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])); self.p.camera.focal_point = tuple(f); self.p.camera.up = (0, 0, 1); self.p.camera.view_angle = 26.0
        self.p.reset_camera_clipping_range()

    def proj(self, pt):
        r = self.p.renderer; r.SetWorldPoint(float(pt[0]), float(pt[1]), float(pt[2]), 1.0); r.WorldToDisplay(); x, y, _ = r.GetDisplayPoint(); return x, H - y

    def frame(self, t):
        A = self.a; show = {k: False for k in A}; M = {k: np.eye(4) for k in A}; dims = []; cap = ("", ""); fade = 0.0; note = None; A["tub"].prop.opacity = 0.55
        if t < T1:                                                                     # 1. 치수
            u = t / T1; show["mount"] = True; az = -60 + 50 * ease(u); el = -30 + 60 * ease(min(1, u * 1.5)); self.cam(az, el, 430, (0, 0, -6))
            cap = ("스탠드 받이 (23)", "수신기를 조명 스탠드 위에 올릴 때 받침 통 밑에 단다 · PLA, 채움 100%, 2시간, 83 g")
            ar = np.radians(az); sx, sy = -np.sin(ar), np.cos(ar)                       # 화면에서 가로로 보이는 방향
            if u < 0.45:
                dims = [((50, -24, -12), (50, 24, -12), "나사 구멍 48"), ((-50, -24, -12), (50, -24, -12), "100")]
            else:
                dims = [((-STAND_R * sx, -STAND_R * sy, 0), (STAND_R * sx, STAND_R * sy, 0), "지름 140"), ((STAND_R * sx, STAND_R * sy, -12), (STAND_R * sx, STAND_R * sy, 0), "두께 12"),
                        ((-5.7 * sx, -5.7 * sy, 0), (5.7 * sx, 5.7 * sy, 0), "")]
                note = "가운데: 1/4인치 너트 자리 (맞변 11.1 + 틈 0.3)"
        elif t < T1 + T2:                                                              # 2. 너트 넣고, 받침 통을 올리고, 아래에서 나사 4개
            tt = t - T1; show.update(mount=True, nut=True, tub=True); un = ease(tt / 1.6); ut = ease((tt - 1.8) / 1.6); uc = ease((tt - 3.5) / 1.2)
            M["nut"] = Tm((0, 0, 42 * (1 - un))); M["tub"] = Tm((0, 0, 95 * (1 - ut))); A["tub"].prop.opacity = 0.55 * min(1.0, tt / 1.2) if tt < 1.8 else 0.55
            for i, (x, y) in enumerate(POSTS):
                us = ease((tt - 4.9 - i * 1.05) / 0.95); M["s%d" % i] = Tm((x, y, -5.0 - 34.0 * (1 - us)), rz=720 * us); show["s%d" % i] = tt > 4.7
            self.cam(-58, 30 - 56 * uc, 560, (0, 0, 14 - 8 * uc))
            cap = ("너트를 넣고, 받침 통 밑에 나사 4개로", "① 1/4인치 육각 너트를 가운데 육각 자리에  ② 받침 통을 올리고  ③ M3 × 10 나사 4개를 아래에서 위로 (5 mm 물린다)")
        elif t < T1 + T2 + T3:                                                         # 3. 스탠드 꼭대기 나사에 돌려 끼우기
            tt = t - T1 - T2; u = ease(tt / (T3 - 1.0)); show.update(mount=True, nut=True, tub=True, s0=True, s1=True, s2=True, s3=True, pole=True)
            dz = 26.0 * (1 - u); rot = 900 * u
            M["tub"] = Tm((0, 0, dz), rot); M["mount"] = Tm((0, 0, dz), rot); M["nut"] = Tm((0, 0, dz), rot)
            for i, (x, y) in enumerate(POSTS):
                M["s%d" % i] = Tm((0, 0, dz), rot) @ Tm((x, y, -5.0))
            self.cam(-60, 8, 620, (0, 0, -40))
            cap = ("스탠드 꼭대기 나사에 통째로 돌려 끼운다", "카메라 삼각대와 같은 1/4인치 나사 · 손으로 꽉 조인다 (받침 모터가 돌 때 풀리지 않을 여유 11.5배)")
            dims = [((0, 0, -24), (0, 0, -12.5), "1/4인치 나사")]
        else:                                                                          # 4. 세운 모습 (그림)
            fade = ease((t - T1 - T2 - T3) / 0.8); show.update(mount=True, nut=True, tub=True, pole=True); self.cam(-60, 8, 620, (0, 0, -40))
            cap = ("안테나 높이 2.4 m 로 세운다", "다리에 모래주머니 3 kg 이상 · 다리 하나가 학생 쪽을 보게 · 없으면 꼭대기를 약 200 g 의 힘으로 밀어도 넘어진다")
        for k, act in A.items():
            act.SetVisibility(bool(show[k])); act.user_matrix = M[k]
        self.p.reset_camera_clipping_range(); self.p.render()
        img = Image.fromarray(np.asarray(self.p.screenshot(return_img=True))[:, :, :3]); d = ImageDraw.Draw(img)
        for a_, b_, txt in dims:                                                       # 치수 선
            (x0, y0), (x1, y1) = self.proj(a_), self.proj(b_); d.line([(x0, y0), (x1, y1)], fill=HOT, width=3)
            for (x, y) in ((x0, y0), (x1, y1)):
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=HOT)
            if not txt:
                continue
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2; tw = d.textlength(txt, font=font(26, True)); ox = 18 if abs(x1 - x0) < 40 else -tw / 2; oy = -44 if abs(x1 - x0) >= 40 else -16
            d.rectangle([mx + ox - 8, my + oy - 4, mx + ox + tw + 8, my + oy + 34], fill=BG); d.text((mx + ox, my + oy), txt, font=font(26, True), fill=HOT)
        if note:
            d.text((40, 96), note, font=font(24, True), fill=HOT)
        if fade > 0:
            card = self.card.resize((int(self.card.size[0] * 620 / self.card.size[1]), 620)); bgim = Image.new("RGB", (W, H), BG); bgim.paste(card, ((W - card.size[0]) // 2, 86)); img = Image.blend(img, bgim, fade); d = ImageDraw.Draw(img)
        d.rectangle([0, 0, W, 78], fill=BG); d.text((40, 14), "시험장에서 쓸 때", font=font(20), fill=SUB); d.text((40, 38), cap[0], font=font(30, True), fill=INK)
        d.rectangle([0, H - 54, W, H], fill=BG); d.text((40, H - 44), cap[1], font=font(21), fill=SUB)
        return img.convert("RGB")


def writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    if cmd == "join":
        tmp = os.path.join(paths.TMP, "asm_joined.mp4")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", MAIN, "-i", OUT, "-filter_complex", "[0:v][1:v]concat=n=2:v=1:a=0[v]", "-map", "[v]", "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-r", str(FPS),
                        "-movflags", "+faststart", tmp], check=True)
        os.replace(tmp, MAIN); print("이어 붙임:", MAIN)
    else:
        S = Scene(); S.frame(0.0)                                                      # 첫 화면은 버린다 (카메라가 자리 잡기 전)
        if cmd == "still":
            os.makedirs(paths.TMP, exist_ok=True)
            for ts in sys.argv[2:]:
                S.frame(float(ts)).save(os.path.join(paths.TMP, "stand_%s.png" % ts)); print("still", ts)
        else:
            w = writer(OUT)
            for i in range(int(TOTAL * FPS)):
                w.stdin.write(np.asarray(S.frame(i / FPS), dtype=np.uint8).tobytes())
            w.stdin.close(); w.wait(); print("영상", OUT, "%.1f초" % TOTAL)
