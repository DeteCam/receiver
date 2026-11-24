"""DeteCam 5.0 (Palm-12) 3D 조립 & 벽면 2축 각도 조절 시뮬레이션 영상 생성기.

    xvfb-run -a python3 cad/figures/anim_assembly.py

1280x720, 24fps, 34초:
  0:00~0:06  완성 모습 (120 x 120 x 30.6 mm 손바닥 크기 · X1C 단 1판 출력)
  0:06~0:13  ① 후면 전자부 (SBC 섀시 + Radxa ROCK 3A/X4 + Intel AX210 + 인셋 방열 커버)
  0:13~0:19  ② 전면 RF 공극 (Ni-Cu 벽면 차폐 시트 + 112mm 12소자 통합 RF PCB + 레이돔 쉘)
  0:19~0:25  ③ 벽면 부착·스위블 짐벌 (자석 4개 벽판 + 직교 요크 + M4 손조임 노브 2개)
  0:25~0:34  ④ 실제 벽면 부착 후 좌우 Pan ±30° · 상하 Tilt 0°→-25° 실시간 각도 조절 시연!
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyvista as pv
import palm_node as P
from cadlib import box
import paths
from render import to_pv

pv.OFF_SCREEN = True
W, H, FPS = 1280, 720, 24
OUT = os.path.join(paths.VID, "조립_시뮬레이션.mp4")
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONTB = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
INK, SUB, BG = "#1E2D5A", "#4F5A6B", "#F7F6F2"
DUR = [6.0, 7.0, 6.0, 6.0, 9.0]
T0 = np.r_[0.0, np.cumsum(DUR)]
TOTAL = float(T0[-1])


def load_font(sz, bold=False):
    p = FONTB if bold and os.path.exists(FONTB) else FONT
    return ImageFont.truetype(p, int(sz), index=1)


def up(m):
    return m.rotate([90, 0, 0])


def ease(u):
    u = float(np.clip(u, 0.0, 1.0))
    return u * u * (3.0 - 2.0 * u)


def mat_trans(dx=0.0, dy=0.0, dz=0.0):
    M = np.eye(4)
    M[0, 3], M[1, 3], M[2, 3] = dx, dy, dz
    return M


def mat_rot_z_render(deg, piv):
    """렌더 좌표계(Z=위, Y=-Z_node) 에서 수직 Z축(노드 Y축) 둘레 좌우 Pan 회전."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4); R[0, 0], R[0, 1], R[1, 0], R[1, 1] = c, -s, s, c
    A = mat_trans(*piv); B = mat_trans(-piv[0], -piv[1], -piv[2])
    return A @ R @ B


def mat_rot_x_render(deg, piv):
    """렌더 좌표계(Z=위, -Y=전면) 에서 수평 X축 둘레 상하 Tilt 회전 (음수=아래 숙임)."""
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4); R[1, 1], R[1, 2], R[2, 1], R[2, 2] = c, -s, s, c
    A = mat_trans(*piv); B = mat_trans(-piv[0], -piv[1], -piv[2])
    return A @ R @ B


class Scene:
    def __init__(self):
        p = pv.Plotter(off_screen=True, window_size=(W, H))
        p.set_background(BG)
        self.p = p
        try:
            p.enable_anti_aliasing("fxaa")
        except Exception:
            pass
        kw = dict(smooth_shading=True, specular=0.28, specular_power=20, ambient=0.32, diffuse=0.78)
        st = dict(smooth_shading=True, specular=0.65, specular_power=42, ambient=0.28, diffuse=0.72)

        A0 = P.node_assembly(0.0, 0.0)
        self.a = {}
        for k, (m, col, _) in A0.items():
            metal = k in ("nicu_reflector", "smd_antennas", "magnets")
            self.a[k] = p.add_mesh(to_pv(up(m)), color=col, **(st if metal else kw))

        wall = box(-120.0, 120.0, -95.0, 95.0, P.WALL_Z0 - 8.0, P.WALL_Z0 - 0.2)
        self.a["wall_demo"] = p.add_mesh(to_pv(up(wall)), color="#E2DFD7", smooth_shading=True, ambient=0.45, diffuse=0.65)

        summ_p = os.path.join(paths.SLICE, "summary.json")
        if os.path.exists(summ_p):
            s = json.load(open(summ_p, encoding="utf-8"))
            pl = s.get("plates", {})
            tot_s = sum(v["sec"] for v in pl.values() if v.get("ok"))
            tot_g = sum(v["g"] for v in pl.values() if v.get("ok"))
            self.spec_line = f"X1C 단 1판 출력 · {tot_s // 3600}시간 {(tot_s % 3600) // 60:02d}분 · PLA {tot_g:.0f} g · 서포트 0%"
        else:
            self.spec_line = "X1C 단 1판 출력 · 4시간 52분 · PLA 165 g · 서포트 0%"

    def cam(self, az, el, dist, focus=(0, 26, 0)):
        a, e = np.radians(az), np.radians(el)
        f = np.asarray(focus, float)
        self.p.camera.position = tuple(f + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)]))
        self.p.camera.focal_point = tuple(f)
        self.p.camera.up = (0, 0, 1)
        self.p.camera.view_angle = 27.0
        self.p.reset_camera_clipping_range()

    def frame(self, t):
        A = self.a
        show = {k: False for k in A}
        M = {k: np.eye(4) for k in A}
        k = min(int(np.searchsorted(T0, t, side="right") - 1), len(DUR) - 1)
        tt = t - T0[k]
        u = tt / DUR[k]

        PUCK_PARTS = ["rf_shell", "sbc_chassis", "rear_cover", "rf_pcb", "smd_antennas", "rf_ics", "nicu_reflector", "rock3a", "ax210", "knob_tilt"]
        ALL_PARTS = PUCK_PARTS + ["swivel_yoke", "wall_base", "knob_pan", "magnets"]

        # 렌더 좌표 피벗 (up([90,0,0]) 변환: (x, y, z)_node -> (x, -z, y)_render)
        pan_piv_r = (0.0, -P.PAN_PIVOT_Z, 0.0)
        tilt_piv_r = (0.0, -P.TILT_PIVOT_Z, 0.0)

        if k == 0:
            for n in ALL_PARTS:
                show[n] = True
            self.cam(-115 + 95 * ease(u), 16, 400, (0, 28, 0))
            title = "DeteCam 5.0 (Palm-12) — 손바닥 크기 벽면 부착·각도 조절 12소자 수신기"
            sub = f"외형 120 × 120 × 30.6 mm  |  대형 스탠드 제거  |  {self.spec_line}"
        elif k == 1:
            show["sbc_chassis"] = True
            for nm, ts, dy in [("rock3a", 0.5, 75.0), ("ax210", 1.8, 75.0), ("rear_cover", 3.6, 95.0)]:
                if tt > ts - 0.1:
                    show[nm] = True
                    M[nm] = mat_trans(0.0, dy * (1.0 - ease((tt - ts) / 1.6)), 0.0)
            self.cam(38 + 22 * ease(u), 18, 340, (0, 12, 0))
            title = "① 후면 전자부 조립 (팬리스 무소음 · 5V 보조배터리 10시간 구동)"
            sub = "Radxa ROCK 3A (M.2 E-Key 네이티브, X4 공용) + Intel AX210 + 인셋 방열·틸트 커버(104×104)"
        elif k == 2:
            for n in ("sbc_chassis", "rock3a", "ax210", "rear_cover"):
                show[n] = True
            for nm, ts, dy in [("nicu_reflector", 0.3, -65.0), ("rf_pcb", 1.7, -85.0), ("smd_antennas", 1.7, -85.0),
                               ("rf_ics", 1.7, -85.0), ("rf_shell", 3.6, -105.0)]:
                if tt > ts - 0.1:
                    show[nm] = True
                    M[nm] = mat_trans(0.0, dy * (1.0 - ease((tt - ts) / 1.5)), 0.0)
            self.cam(-62 + 22 * ease(u), 18, 370, (0, -8, 0))
            title = "② 전면 12소자 통합 RF PCB + 11.4 mm Ni-Cu 벽면 차폐 공극"
            sub = "테두리 12+1 SMD 안테나 + PE42442×4 + AD8317 단일 4층 PCB (케이블 39개 제거 · 벽면 간섭 차단)"
        elif k == 3:
            for n in PUCK_PARTS:
                show[n] = True
            for nm, ts, dy in [("swivel_yoke", 0.4, 85.0), ("knob_tilt", 1.5, 65.0), ("wall_base", 2.6, 100.0),
                               ("magnets", 3.4, 115.0), ("knob_pan", 4.1, 80.0)]:
                if tt > ts - 0.1:
                    show[nm] = True
                    if nm == "knob_tilt":
                        M[nm] = mat_trans(dy * (1.0 - ease((tt - ts) / 1.2)), 0.0, 0.0)
                    else:
                        M[nm] = mat_trans(0.0, dy * (1.0 - ease((tt - ts) / 1.3)), 0.0)
            self.cam(-32 + 20 * ease(u), 18, 410, (0, 32, 0))
            title = "③ 무타공 벽면 마운트 + 직교 2축 스위블 짐벌 체결"
            sub = "Ø15×3 mm 네오디뮴 자석 4개(사물함·칠판) + 재사용 나노겔 패드(타일·거울·벽지) + M4 손조임 노브"
        else:
            for n in ALL_PARTS + ["wall_demo"]:
                show[n] = True
            # 좌우 Pan (-30° -> +30°) & 상하 Tilt (0° -> -25°) 실시간 시연!
            if u < 0.45:
                pan = 30.0 * np.sin(2.0 * np.pi * (u / 0.45))
                tilt = 0.0
                lbl_ang = f"좌우 Pan {pan:+.0f}° · 상하 Tilt 0° (방 중앙 조향)"
            else:
                v = (u - 0.45) / 0.55
                pan = 25.0 * ease(min(1.0, v * 2.0))
                tilt = -25.0 * np.sin(np.pi * v)
                lbl_ang = f"좌우 Pan {pan:+.0f}° · 상하 Tilt {tilt:.0f}° (책상·바닥 사각지대 조향)"

            R_pan = mat_rot_z_render(pan, pan_piv_r)
            R_tilt = mat_rot_x_render(tilt, tilt_piv_r)
            M["swivel_yoke"] = R_pan
            M["knob_pan"] = R_pan
            for n in PUCK_PARTS:
                M[n] = R_pan @ R_tilt
            self.cam(-48 + 14 * np.sin(np.pi * u), 15, 430, (0, 28, 0))
            title = f"④ 벽면 부착 후 손으로 1초 각도 조절 — {lbl_ang}"
            sub = "벽에 착 붙이고 AR-OPA 안내대로 딸깍 돌리면 끝! (충돌 0.0 mm³ · 화장실 P90 -30% · 탈의실 -20% 개선)"

        for n, act in A.items():
            act.SetVisibility(bool(show[n]))
            if show[n]:
                act.user_matrix = M[n]
        self.p.render()
        img = Image.fromarray(self.p.screenshot(return_img=True))
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 96, W, H], fill=(255, 255, 255))
        d.rectangle([0, H - 96, W, H - 93], fill=(235, 104, 52))
        d.text((34, H - 82), title, fill=INK, font=load_font(25, bold=True))
        d.text((34, H - 44), sub, fill=SUB, font=load_font(17.5, bold=False))
        bw = int((W - 68) * (t / TOTAL))
        d.rectangle([34, 22, W - 34, 28], fill=(224, 222, 215))
        d.rectangle([34, 22, 34 + bw, 28], fill=(25, 194, 176))
        return np.asarray(img)


def main():
    os.makedirs(paths.VID, exist_ok=True)
    sc = Scene()
    n_frames = int(round(TOTAL * FPS))
    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", OUT,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    for i in range(n_frames):
        rgb = sc.frame(i / FPS)
        proc.stdin.write(rgb.tobytes())
    proc.stdin.close()
    proc.wait()
    sc.p.close()
    print(f"Palm-12 assembly video saved: {OUT} ({os.path.getsize(OUT) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
