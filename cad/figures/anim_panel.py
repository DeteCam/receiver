"""조립 영상: 패널(통, 걸이판, RF 모듈, 보드, 앞덮개)과 허브를 차례로 조립하고 치수를 보여 준다.

    xvfb-run -a python3 anim_panel.py                 # docs/videos/조립_시뮬레이션.mp4 (약 56초)
    xvfb-run -a python3 anim_panel.py still 3 12 30   # 그 시각(초)의 화면을 임시 폴더에 (확인용)

1280x720, 24fps. 치수는 model/panel.py 의 값이고, 무게와 출력 시간은 print/summary.json 의 값이다.
"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import panel as P
import paths
from cadlib import *
from render import to_pv

pv.OFF_SCREEN = True
W, H, FPS = 1280, 720, 24
OUT = os.path.join(paths.VID, "조립_시뮬레이션.mp4")
FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'; FONTB = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'
INK, SUB, BG, HOT = "#22262B", "#5B6470", "#F4F2EE", "#EB6834"
font = lambda sz, bold=False: ImageFont.truetype(FONTB if bold else FONT, int(sz), index=1)
DUR = [7.0, 7.0, 9.0, 6.0, 7.0, 5.0, 9.0, 6.0]                # 완성 모습, 걸이판, 속 부품, 보드, 앞덮개, 받침 발, 허브, 한 벌
T0 = np.r_[0, np.cumsum(DUR)]; TOTAL = float(T0[-1])
SUMMARY = os.path.join(paths.SLICE, "summary.json")


def up(m):
    """패널 좌표의 형상을 세운다: 패널의 y(위) -> 화면의 z, 패널의 z(앞) -> 화면의 -y"""
    return m.rotate([90, 0, 0])


def wp(x, y, z):
    return np.array([x, -z, y], float)


def Tm(v=(0, 0, 0), spin=0.0, about=(0, 0, 0), axis="y"):
    """v 만큼 옮기고, about 을 지나는 축(axis) 둘레로 spin 도 돌린다"""
    a = np.radians(spin); c, s = np.cos(a), np.sin(a); R = np.eye(4)
    if axis == "y":
        R[0, 0], R[0, 2], R[2, 0], R[2, 2] = c, s, -s, c
    else:
        R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    A = np.eye(4); A[:3, 3] = about; B = np.eye(4); B[:3, 3] = -np.asarray(about, float); M = np.eye(4); M[:3, 3] = v
    return M @ A @ R @ B


def ease(u):
    u = float(np.clip(u, 0, 1)); return u * u * (3 - 2 * u)


def screw_front(x, y, L=10.0):
    """앞덮개 나사: 머리가 앞(z+), 몸통이 통 속으로"""
    return up(union([cz(1.5, -L, 0.0), cz(2.8, 0.0, 2.2)]).translate([x, y, P.ZC - 1.8]))


def screw_in(x, y):
    """걸이 나사: 통 안에서 뒤로"""
    return up(union([cz(1.5, -8.0, 0.0), cz(2.8, 0.0, 2.2)]).translate([x, y, P.ZF]))


class Scene:
    def __init__(self):
        p = pv.Plotter(off_screen=True, window_size=(W, H)); p.set_background(BG); self.p = p
        try:
            p.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        kw = dict(smooth_shading=True, specular=0.25, specular_power=18, ambient=0.3, diffuse=0.8); st = dict(smooth_shading=True, specular=0.7, specular_power=40, ambient=0.3, diffuse=0.7)
        A = {}
        def add(name, m, col, **k2):
            A[name] = p.add_mesh(to_pv(m), color=col, **(k2 or kw))
        add("tray", up(P.tray()), P.PRINTED); add("bezel", up(P.bezel()), P.LIGHT); add("mount", up(P.mount_plate()), P.HOT); add("foot", up(P.foot()), P.HOT)
        add("clip", up(P.xiao_clip()), P.HOT); add("board", up(P.board()), P.PCB); add("patches", up(P.patches()), P.GOLD, **st); add("xiao", up(P.xiao()), P.DARK)
        add("cell", up(P.cell()), P.STEEL, **st); add("rf", up(P.rf_modules()), P.STEEL, **st); add("ref", up(P.ref_antenna()), P.GOLD, **st); add("magnets", up(P.magnets()), P.STEEL, **st)
        for i, (x, y) in enumerate(P.BOSS):
            add("fs%d" % i, screw_front(x, y), P.STEEL, **st)
        for i, (x, y) in enumerate(P.MOUNT):
            add("ms%d" % i, screw_in(x, y), P.STEEL, **st)
        add("hub_box", P.hub_box(), P.PRINTED); add("hub_deck", P.hub_deck(), P.HOT); add("hub_lid", P.hub_lid(), P.LIGHT); add("battery", P.battery(), P.DARK)
        add("pi", P.pi_board(), P.PCB); add("dongles", P.dongles(), P.STEEL, **st)
        # 한 벌 모습에 쓰는 둘째 패널 (받침 발을 단 것)
        add("p2", up(P.tray()), P.PRINTED); add("p2b", up(P.bezel()), P.LIGHT); add("p2f", up(P.foot()), P.HOT)
        self.a = A
        s = json.load(open(SUMMARY, encoding="utf-8")) if os.path.exists(SUMMARY) else None
        if s:
            pr = s["parts"]; pl = s["plates"]
            self.g_panel = pr["01_panel_tray"]["g"] + pr["02_panel_bezel"]["g"] + pr["03_mount_plate"]["g"] + pr["05_xiao_clip"]["g"]
            sec = sum(pl[k]["sec"] * (2 if k[0] in "12" else 1) for k in pl); g = sum(pl[k]["g"] * (2 if k[0] in "12" else 1) for k in pl)
            self.tot = "출력 약 %d시간 · 필라멘트 약 %.1f kg" % (round(sec / 3600), g / 1000)
        else:
            self.g_panel, self.tot = 285.0, ""

    def cam(self, az, el, dist, focus):
        a, e = np.radians(az), np.radians(el); f = np.asarray(focus, float)
        self.p.camera.position = tuple(f + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])); self.p.camera.focal_point = tuple(f); self.p.camera.up = (0, 0, 1); self.p.camera.view_angle = 26.0
        self.p.reset_camera_clipping_range()

    def proj(self, pt):
        r = self.p.renderer; r.SetWorldPoint(float(pt[0]), float(pt[1]), float(pt[2]), 1.0); r.WorldToDisplay(); x, y, _ = r.GetDisplayPoint(); return x, H - y

    def frame(self, t):
        A = self.a; show = {k: False for k in A}; M = {k: np.eye(4) for k in A}; dims = []; cap = ("", ""); note = None
        k = int(np.searchsorted(T0, t, side="right") - 1); k = min(k, len(DUR) - 1); tt = t - T0[k]; u = tt / DUR[k]
        fwd = np.array([0, -1.0, 0]); yc = (P.OY0 + P.OY1) / 2
        PANEL = ("tray", "bezel", "mount", "magnets", "clip", "board", "patches", "xiao", "cell", "rf", "ref") + tuple("fs%d" % i for i in range(6)) + tuple("ms%d" % i for i in range(4))
        if k == 0:                                                                      # 1. 완성 모습과 치수
            for n in PANEL: show[n] = True
            az = -125 + 95 * ease(u); self.cam(az, 12, 760, wp(0, yc, -16))
            cap = ("패널: 모터가 없는 수신기", "안테나 16개를 전자 스위치로 차례로 고른다 · 돌아가는 것이 없어 소리가 나지 않는다 · 한 번 재는 데 0.3초")
            if u < 0.5:
                dims = [(wp(-P.OX, P.OY0 - 8, P.ZC), wp(P.OX, P.OY0 - 8, P.ZC), "가로 %.0f" % (2 * P.OX)), (wp(P.OX + 8, P.OY0, P.ZC), wp(P.OX + 8, P.OY1, P.ZC), "세로 %.0f" % (P.OY1 - P.OY0))]
            else:
                dims = [(wp(P.OX, P.OY1 + 6, P.ZO - P.MT_T), wp(P.OX, P.OY1 + 6, P.ZC), "두께 %.0f" % (P.ZC - P.ZO + P.MT_T))]
                note = "뽑는 부품 %.0f g + 보드·RF 모듈·배터리 = 약 0.6 kg" % self.g_panel
        elif k == 1:                                                                    # 2. 통 뒤에 자석 걸이판
            show.update(tray=True, mount=True, magnets=True); um = ease(tt / 2.0); M["mount"] = M["magnets"] = Tm(60 * (1 - um) * -fwd)
            for i in range(4):
                us = ease((tt - 2.4 - 0.7 * i) / 0.7); show["ms%d" % i] = tt > 2.3; x, y = P.MOUNT[i]; M["ms%d" % i] = Tm(45 * (1 - us) * fwd, spin=720 * us, about=wp(x, y, 0))
            az = 118 - 170 * ease((tt - 1.2) / 3.0); self.cam(az, 14, 800, wp(0, yc, -20))
            cap = ("① 통 뒤에 자석 걸이판", "통 안에서 M3 × 8 나사 4개 (너트는 걸이판에) · 자석 지름 20 × 4개로 칠판·철문·사물함에 붙고, 위 고리로 걸이에 건다")
            if tt < 2.2:
                dims = [(wp(-40, -40, P.ZO - P.MT_T), wp(40, -40, P.ZO - P.MT_T), "나사 간격 80"), (wp(-P.MT_W / 2, -P.MT_W / 2 - 6, P.ZO - P.MT_T), wp(P.MT_W / 2, -P.MT_W / 2 - 6, P.ZO - P.MT_T), "걸이판 %.0f" % P.MT_W)]
        elif k == 2:                                                                    # 3. 속 부품
            show.update(tray=True, mount=True, magnets=True, ms0=True, ms1=True, ms2=True, ms3=True, rf=True, xiao=True, cell=True, clip=True)
            for n, t0 in (("rf", 0.3), ("cell", 2.2), ("xiao", 3.6), ("clip", 5.2)):
                un = ease((tt - t0) / 1.3); M[n] = Tm(70 * (1 - un) * fwd); show[n] = tt > t0 - 0.1
            self.cam(-62, 16, 720, wp(0, yc, -10))
            cap = ("② 통 안에 부품을 넣는다", "RF 스위치 2개와 합침기를 묶음끈으로 · 18650 배터리 홀더 · ESP32-C5(XIAO)는 오른쪽 벽에, 누름쇠와 M2 × 8 나사 2개")
            if tt > 6.4:
                dims = [(wp(P.XIAO_C[0] - 10.5, P.XIAO_C[1] - 14, P.XIAO_Z + 3), wp(P.XIAO_C[0] + 10.5, P.XIAO_C[1] - 14, P.XIAO_Z + 3), "21 × 17.8")]; note = "USB-C 는 오른쪽 벽으로 나온다"
        elif k == 3:                                                                    # 4. 보드
            show.update(tray=True, mount=True, magnets=True, rf=True, xiao=True, cell=True, clip=True, board=True, patches=True)
            ub = ease(tt / 2.4); M["board"] = M["patches"] = Tm(90 * (1 - ub) * fwd); self.cam(-62 + 14 * ease(u), 16, 720, wp(0, yc, -6))
            cap = ("③ 안테나 배열 보드를 턱에 얹는다", "보드 가장자리 2.5 mm 만 닿는다 (보드에 구멍이 없어도 된다) · 패치 16개, 스위치 칩은 보드 뒷면에")
            if tt > 2.6:
                dims = [(wp(-75, -75 - 5, 0.5), wp(75, -75 - 5, 0.5), "보드 150"), (wp(-1.5 * P.PITCH, 1.5 * P.PITCH + 20, 0.5), wp(-0.5 * P.PITCH, 1.5 * P.PITCH + 20, 0.5), "간격 %.1f" % P.PITCH)]
        elif k == 4:                                                                    # 5. 앞덮개
            for n in ("tray", "mount", "magnets", "rf", "xiao", "cell", "clip", "board", "patches", "bezel", "ref"): show[n] = True
            ub = ease(tt / 2.2); M["bezel"] = M["ref"] = Tm(80 * (1 - ub) * fwd)
            for i in range(6):
                us = ease((tt - 2.5 - 0.5 * i) / 0.6); show["fs%d" % i] = tt > 2.4; x, y = P.BOSS[i]; M["fs%d" % i] = Tm(45 * (1 - us) * fwd, spin=720 * us, about=wp(x, y, 0))
            self.cam(-48 - 20 * ease(u), 14, 740, wp(0, yc, -4))
            cap = ("④ 앞덮개를 덮는다", "기준 안테나(XIAO 에 딸려 온 것)를 덮개 안쪽 위에 붙이고, M3 × 10 나사 6개 · 패치 앞 빈틈 %.0f mm, 덮개 살 %.1f mm" % (P.AIR, P.SKIN))
            if tt > 5.6:
                dims = [(wp(-P.MARK[0] / 2, P.MARK_C[1] + P.MARK[1] / 2 + 3, P.ZC), wp(P.MARK[0] / 2, P.MARK_C[1] + P.MARK[1] / 2 + 3, P.ZC), "AR 마커 띠 %.0f × %.0f" % P.MARK)]
        elif k == 5:                                                                    # 6. 받침 발
            for n in ("tray", "bezel", "foot") + tuple("fs%d" % i for i in range(6)): show[n] = True
            uf = ease(tt / 1.8); M["foot"] = Tm(60 * (1 - uf) * -fwd); self.cam(118 - 60 * ease(u), 12, 820, wp(0, yc - 20, -40))
            cap = ("창턱·교탁·사물함 위에는 받침 발", "걸이판 대신 단다 (M3 × 8 2개). 통이 발 위에 얹힌다 · 발 밑의 1/4인치 너트로 카메라 삼각대에도 올라간다")
            if tt > 2.0:
                dims = [(wp(P.FT_W / 2 + 4, P.OY0 - P.FT_GAP - P.FT_T, P.ZO - P.FT_T - P.FT_D), wp(P.FT_W / 2 + 4, P.OY0 - P.FT_GAP - P.FT_T, P.FT_FRONT), "발 %.0f" % (P.FT_D + P.FT_T - P.ZO + P.FT_FRONT))]
        elif k == 6:                                                                    # 7. 허브
            show.update(hub_box=True); zt = P.HFLOOR + P.HH
            for n, t0, dz in (("battery", 0.6, 90), ("hub_deck", 2.2, 80), ("pi", 3.6, 80), ("dongles", 3.6, 80), ("hub_lid", 6.0, 70)):
                un = ease((tt - t0) / 1.2); M[n] = Tm((0, 0, dz * (1 - un))); show[n] = tt > t0 - 0.1
            self.cam(-58 + 20 * ease(u), 30, 560, (0, 0, 30))
            cap = ("허브: 라즈베리파이 + 탐지용 동글 + 보조 배터리", "배터리를 밑에 넣고, 선반을 얹고, 파이를 M2.5 나사 4개로 · 뚜껑은 M3 × 10 나사 4개 · 패널과는 USB 선으로 잇는다")
            if tt > 7.4:
                ox, oy = P.HL / 2 + P.HWALL, P.HW_ / 2 + P.HWALL
                dims = [((-ox, -oy - 6, 0), (ox, -oy - 6, 0), "%.0f" % (2 * ox)), ((ox + 6, -oy, 0), (ox + 6, oy, 0), "%.0f" % (2 * oy)), ((ox + 6, oy, 0), (ox + 6, oy, zt + 2.4), "높이 %.0f" % (zt + 2.4))]
        else:                                                                           # 8. 한 벌
            for n in PANEL + ("p2", "p2b", "p2f", "hub_box", "hub_lid"): show[n] = True
            L = Tm((-190, 0, 0)); R = Tm((190, 40, 0), spin=18, axis="z"); Hm = Tm((0, -150, P.OY0 - P.FT_GAP - P.FT_T))
            for n in PANEL: M[n] = L
            M["p2"] = M["p2b"] = M["p2f"] = R; M["hub_box"] = M["hub_lid"] = Hm
            self.cam(-78 + 26 * ease(u), 16, 1250, (0, -40, -20))
            cap = ("수신기 한 벌 = 패널 2개 + 허브", "패널 하나는 칠판에 자석으로, 하나는 옆 창이나 벽에 · " + self.tot)
        for n, act in A.items():
            act.SetVisibility(bool(show[n])); act.user_matrix = M[n]
        self.p.reset_camera_clipping_range(); self.p.render()
        img = Image.fromarray(np.asarray(self.p.screenshot(return_img=True))[:, :, :3]); d = ImageDraw.Draw(img)
        for a_, b_, txt in dims:                                                        # 치수 선
            (x0, y0), (x1, y1) = self.proj(a_), self.proj(b_); d.line([(x0, y0), (x1, y1)], fill=HOT, width=3)
            for (x, y) in ((x0, y0), (x1, y1)):
                d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=HOT)
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2; tw = d.textlength(txt, font=font(26, True)); ox = 18 if abs(x1 - x0) < 40 else -tw / 2; oy = 10 if abs(x1 - x0) >= 40 else -16
            d.rectangle([mx + ox - 8, my + oy - 4, mx + ox + tw + 8, my + oy + 34], fill=BG); d.text((mx + ox, my + oy), txt, font=font(26, True), fill=HOT)
        if note:
            d.text((40, 96), note, font=font(24, True), fill=HOT)
        d.rectangle([0, 0, W, 78], fill=BG); d.text((40, 14), "패널 수신기 조립 (단위 mm)", font=font(20), fill=SUB); d.text((40, 38), cap[0], font=font(30, True), fill=INK)
        d.rectangle([0, H - 54, W, H], fill=BG); d.text((40, H - 44), cap[1], font=font(21), fill=SUB)
        return img.convert("RGB")


def writer(path):
    return subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", path], stdin=subprocess.PIPE)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "render"
    S = Scene(); S.frame(0.0)                                                           # 첫 화면은 버린다 (카메라가 자리 잡기 전)
    if cmd == "still":
        for ts in sys.argv[2:]:
            S.frame(float(ts)).save(os.path.join(paths.TMP, "asm_%s.png" % ts)); print("still", ts)
    else:
        w = writer(OUT)
        for i in range(int(TOTAL * FPS)):
            w.stdin.write(np.asarray(S.frame(i / FPS), dtype=np.uint8).tobytes())
        w.stdin.close(); w.wait(); print("영상", OUT, "%.1f초" % TOTAL)
