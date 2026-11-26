"""Small pictures of the parts for the purchase list (own drawings from the design values, not seller photos).
Usage: python3 make_parts.py <output folder>"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r3d import Renderer, Mesh, box, cyl, orbit  # noqa: E402
import model as M  # noqa: E402
from make_video import font  # noqa: E402

W, H = 400, 300


def shifted(mesh, lo=None):
    V, F, C, E = mesh.arrays()
    c = 0.5 * (V.min(0) + V.max(0))
    return V - c, F, C, E


def film():
    m = Mesh()
    box(m, (-4.5, -0.3, -17.5), (4.5, 0.3, 17.5), M.FILM)
    box(m, (-3.5, 0.3, -15), (-0.8, 0.5, 15), (0.75, 0.45, 0.05), edges=False)
    box(m, (0.8, 0.3, -15), (3.5, 0.5, 15), (0.75, 0.45, 0.05), edges=False)
    cyl(m, (4.5, 0, 0), (40, 0, 0), 0.7, M.CABLE, n=6)
    cyl(m, (40, 0, 0), (44, 0, 0), 1.6, M.GOLD, n=8)
    return m


def adapter():
    m = Mesh()
    cyl(m, (0, 0, 0), (8, 0, 0), 4.0, M.GOLD, n=6, edges=True)
    cyl(m, (8, 0, 0), (13, 0, 0), 2.2, M.GOLD, n=12)
    cyl(m, (-6, 0, 0), (0, 0, 0), 3.1, M.GOLD, n=12)
    return m


def sp16t():
    return M.sp16t()


def wilkinson():
    m = Mesh()
    box(m, (0, 0, 0), (32, 24, 9), M.BLUE)
    cyl(m, (32, 12, 4.5), (42, 12, 4.5), 3.1, M.GOLD, n=10)
    cyl(m, (0, 5, 4.5), (-10, 5, 4.5), 3.1, M.GOLD, n=10)
    cyl(m, (0, 19, 4.5), (-10, 19, 4.5), 3.1, M.GOLD, n=10)
    return m


def attenuator():
    m = Mesh()
    cyl(m, (0, 0, 0), (6, 0, 0), 3.1, M.GOLD, n=12)
    cyl(m, (6, 0, 0), (9, 0, 0), 4.2, M.GOLD, n=6, edges=True)
    cyl(m, (9, 0, 0), (20, 0, 0), 4.8, M.STEEL, n=14)
    cyl(m, (20, 0, 0), (26, 0, 0), 4.0, M.GOLD, n=12)
    return m


def pigtail(color=M.CABLE, a_gold=True):
    m = Mesh()
    cyl(m, (0, 0, 0), (9, 0, 0), 3.1, M.GOLD, n=12)
    cyl(m, (9, 0, 0), (13, 0, 0), 4.0, M.GOLD, n=6, edges=True)
    pts = [(13, 0, 0), (30, 6, 0), (48, -6, 0), (66, 4, 0), (80, 0, 0)]
    for p, q in zip(pts[:-1], pts[1:]):
        cyl(m, p, q, 0.6, color, n=6)
    cyl(m, (80, 0, 0), (83, 0, 0), 1.3, M.GOLD, n=8)
    return m


def jumper():
    m = Mesh()
    pts = [(0, 0, 0), (25, 8, 0), (50, -8, 0), (75, 6, 0), (100, 0, 0)]
    for p, q in zip(pts[:-1], pts[1:]):
        cyl(m, p, q, 1.3, (0.75, 0.50, 0.30), n=8)
    for x0, s in ((0, -1), (100, 1)):
        cyl(m, (x0, 0, 0), (x0 + s * 6, 0, 0), 4.0, M.GOLD, n=6, edges=True)
        cyl(m, (x0 + s * 6, 0, 0), (x0 + s * 12, 0, 0), 3.1, M.GOLD, n=12)
    return m


def ax210():
    m = Mesh()
    box(m, (0, 0, 0), (22, 30, 1.0), M.PCB)
    box(m, (2, 4, 1.0), (20, 24, 3.0), M.ALU)
    for x in (5, 12):
        cyl(m, (x, 27.5, 1.0), (x, 27.5, 2.2), 1.3, M.GOLD, n=8)
    return m


def cm4():
    m = Mesh()
    box(m, (0, 0, 0), (55, 40, 1.5), M.PCB)
    box(m, (16, 12, 1.5), (34, 28, 3.6), M.BLACK)
    box(m, (38, 6, 1.5), (50, 18, 3.0), M.ALU)
    box(m, (4, 26, 1.5), (14, 36, 3.2), M.ALU)
    return m


def carrier():
    m = Mesh()
    box(m, (0, 0, 0), (85, 56, 1.6), M.PCB)
    box(m, (72, 4, 1.6), (85, 19, 14), M.STEEL)
    box(m, (72, 24, 1.6), (85, 39, 8), M.STEEL)
    box(m, (10, 12, 1.6), (65, 14, 4.5), M.BLACK, edges=False)
    box(m, (10, 44, 1.6), (65, 46, 4.5), M.BLACK, edges=False)
    for i in range(20):
        box(m, (8 + 2.54 * i, 51, 1.6), (9 + 2.54 * i, 54.5, 7), M.GOLD, edges=False)
    return m


def m2adapter():
    m = Mesh()
    box(m, (0, 0, 0), (30, 42, 1.2), M.PCB2)
    box(m, (4, 30, 1.2), (26, 34, 4.0), M.BLACK)
    box(m, (6, -3, 0.1), (24, 0, 1.1), M.GOLD, edges=False)
    return m


def heatsink():
    m = Mesh()
    box(m, (0, 0, 0), (55, 40, 2), M.ALU)
    for i in range(9):
        box(m, (1 + i * 6.2, 0, 2), (4 + i * 6.2, 40, 10), M.ALU, edges=False)
    return m


def pico():
    m = Mesh()
    box(m, (0, 0, 0), (51, 21, 1.0), M.PCB)
    box(m, (43, 6, 1.0), (51, 15, 4.0), M.STEEL)
    box(m, (20, 7, 1.0), (27, 14, 2.0), M.BLACK)
    return m


def motor():
    m = Mesh()
    box(m, (0, 0, 0), (42, 42, 40), (0.20, 0.20, 0.22))
    box(m, (-40, -8, -9), (0, 50, 49), (0.62, 0.64, 0.67))
    cyl(m, (-20, 50, 20), (-20, 70, 20), 4.0, M.STEEL, n=12)
    return m


def tmc2209():
    m = Mesh()
    box(m, (0, 0, 0), (20, 15, 1.2), (0.55, 0.10, 0.10))
    box(m, (6, 3.5, 1.2), (14, 11.5, 3.0), M.BLACK)
    box(m, (7, 4.5, 3.0), (13, 10.5, 9.0), M.BLUE)
    return m


def adxl345():
    m = Mesh()
    box(m, (0, 0, 0), (20, 15, 1.2), M.PCB2)
    box(m, (8, 5.5, 1.2), (12, 9.5, 2.2), M.BLACK)
    return m


def pdtrigger():
    m = Mesh()
    box(m, (0, 0, 0), (28, 11, 1.2), M.RED)
    box(m, (0, 1.2, 1.2), (8, 9.8, 4.4), M.STEEL)
    return m


def buck():
    m = Mesh()
    box(m, (0, 0, 0), (22, 17, 1.2), (0.12, 0.25, 0.60))
    box(m, (5, 3, 1.2), (17, 14, 7.5), M.BLACK)
    return m


def powerbank():
    m = Mesh()
    box(m, (0, 0, 0), (150, 70, 25), M.BLUE)
    return m


PICTURES = [
    ("film", film, "필름 안테나 (35 × 9 mm, 케이블 300 mm)", (-60, 18), 1.25),
    ("adapter", adapter, "SMA 수 – U.FL 수 어댑터", (-55, 22), 1.3),
    ("sp16t", sp16t, "SP16T 스위치 (기판 60 × 100 mm)", (-60, 40), 1.15),
    ("wilkinson", wilkinson, "윌킨슨 2분배기 (0.5~6 GHz)", (-55, 30), 1.25),
    ("attenuator", attenuator, "SMA 6 dB 감쇠기 (6 GHz)", (-60, 20), 1.3),
    ("pigtail", pigtail, "MHF4 – SMA 암 피그테일", (-90, 55), 1.2),
    ("jumper", jumper, "RG316 SMA 수–수 점퍼", (-90, 55), 1.2),
    ("ax210", ax210, "Intel AX210NGW (M.2 2230)", (-60, 40), 1.3),
    ("cm4", cm4, "Raspberry Pi CM4 (무선 포함)", (-60, 40), 1.25),
    ("carrier", carrier, "CM4-IO-BASE-A (85 × 56 mm)", (-60, 40), 1.2),
    ("m2adapter", m2adapter, "M.2 M-key → A-key 어댑터", (-60, 40), 1.3),
    ("heatsink", heatsink, "CM4 방열판", (-60, 35), 1.25),
    ("pico", pico, "Raspberry Pi Pico", (-60, 40), 1.25),
    ("motor", motor, "웜기어 스텝 모터 (NEMA17)", (-50, 25), 1.3),
    ("tmc2209", tmc2209, "TMC2209 드라이버", (-60, 35), 1.35),
    ("adxl345", adxl345, "ADXL345 가속도 센서", (-60, 40), 1.35),
    ("pdtrigger", pdtrigger, "USB-C PD 트리거 (9 V)", (-60, 35), 1.35),
    ("buck", buck, "5 V 5 A 벅 변환기", (-60, 35), 1.35),
    ("powerbank", powerbank, "PD 보조배터리 (가진 것)", (-55, 28), 1.25),
]


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "parts"
    os.makedirs(out, exist_ok=True)
    rend = Renderer(W, H, ss=3, bg_top=(1.0, 1.0, 1.0), bg_bottom=(0.94, 0.95, 0.97))
    for name, fn, title, (az, el), zoom in PICTURES:
        V, F, C, E = shifted(fn())
        size = float(np.linalg.norm(V.max(0) - V.min(0)))
        cam = orbit((0, 0, 0), 3.1 * size / zoom, az, el, fov_deg=24.0)
        rend.begin(cam)
        rend.draw(V, F, C, E)
        img = rend.image()
        d = ImageDraw.Draw(img)
        d.rectangle([0, H - 40, W, H], fill=(245, 246, 248))
        f = font(18)
        d.text(((W - d.textlength(title, font=f)) / 2, H - 34), title, font=f, fill=(25, 28, 34))
        img.save(os.path.join(out, f"{name}.png"), optimize=True)
        print(name, flush=True)


if __name__ == "__main__":
    main()
