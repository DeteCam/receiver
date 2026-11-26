"""Step-by-step assembly video of the DeteCam receiver (new exterior, 2026-11-27).  Design values, not measured.
Printed parts come from hw/cad/parts.py (CadQuery); electronics, screws and wires are simple shapes drawn here.
VTK renders off screen, PIL writes the Korean captions, ffmpeg encodes one file per step and joins them.
  xvfb-run -a -s "-screen 0 1280x720x24" python3 hw/anim/make_video.py test   # last frame of every step -> out/test_*.png
  xvfb-run -a -s "-screen 0 1280x720x24" python3 hw/anim/make_video.py all    # out/assembly.mp4 (1280 x 720, 30 fps)
  xvfb-run -a -s "-screen 0 1280x720x24" python3 hw/anim/make_video.py C3     # one step
"""
import math
import os
import subprocess
import sys

import numpy as np
import vtk
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "cad"))
import parts as P  # noqa: E402

OUT = os.path.join(HERE, "out")
W, H, FPS = 1280, 720, 30
FONT_R = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_B = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"

# ---------------------------------------------------------------- colours
WHITE_PLA = (0.94, 0.94, 0.92)
BASE_PLA = (0.30, 0.33, 0.38)
WALL_PLA = (0.62, 0.65, 0.69)
PCB = (0.10, 0.45, 0.27)
PCB_BLUE = (0.10, 0.25, 0.55)
PCB_RED = (0.75, 0.16, 0.14)
PCB_PURPLE = (0.42, 0.16, 0.55)
BLACK = (0.13, 0.13, 0.15)
GOLD = (0.86, 0.69, 0.26)
STEEL = (0.62, 0.64, 0.67)
DARK = (0.22, 0.22, 0.25)
FILM = (0.95, 0.58, 0.10)
NYLON = (0.97, 0.97, 0.95)
ORANGE_W = (0.95, 0.45, 0.10)
WAGO = (0.75, 0.77, 0.80)
WIRE = {"red": (0.85, 0.12, 0.10), "black": (0.08, 0.08, 0.08), "orange": (1.0, 0.55, 0.05), "yellow": (0.95, 0.80, 0.10),
        "green": (0.10, 0.65, 0.25), "blue": (0.15, 0.35, 0.85), "white": (0.95, 0.95, 0.95), "purple": (0.55, 0.25, 0.75),
        "grey": (0.55, 0.55, 0.58), "brown": (0.50, 0.30, 0.15), "cable": (0.30, 0.30, 0.32)}

ZP, Z_TOP, Z_RIM, FLOOR = P.ZP, P.Z_TOP, P.Z_RIM, P.FLOOR
BX, BY = P.BX, P.BY


# ---------------------------------------------------------------- geometry helpers (vtkPolyData, world mm)
def box(x0, x1, y0, y1, z0, z1):
    s = vtk.vtkCubeSource()
    s.SetBounds(x0, x1, y0, y1, z0, z1)
    s.Update()
    return s.GetOutput()


def cyl(p0, p1, r, n=18):
    ln = vtk.vtkLineSource()
    ln.SetPoint1(*map(float, p0))
    ln.SetPoint2(*map(float, p1))
    t = vtk.vtkTubeFilter()
    t.SetInputConnection(ln.GetOutputPort())
    t.SetRadius(r)
    t.SetNumberOfSides(n)
    t.CappingOn()
    t.Update()
    return t.GetOutput()


def tube(pts, r=0.7, n=8):
    pts = np.asarray(pts, float)
    p = vtk.vtkPoints()
    for q in pts:
        p.InsertNextPoint(*q)
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(pts))
    for i in range(len(pts)):
        line.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    pd = vtk.vtkPolyData()
    pd.SetPoints(p)
    pd.SetLines(cells)
    t = vtk.vtkTubeFilter()
    t.SetInputData(pd)
    t.SetRadius(r)
    t.SetNumberOfSides(n)
    t.CappingOn()
    t.Update()
    return t.GetOutput()


def merge(*pds):
    a = vtk.vtkAppendPolyData()
    for pd in pds:
        a.AddInputData(pd)
    a.Update()
    return a.GetOutput()


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def hexnut(c, axis, af=5.5, h=2.4):
    a = unit(axis)
    c = np.asarray(c, float)
    return cyl(c - a * h / 2, c + a * h / 2, af / math.sqrt(3.0), n=6)


def screw_pd(head, axis, length, kind="pan", d=3.0):
    """Screw whose head bears on the point `head`, shank going along `axis`."""
    a = unit(axis)
    h = np.asarray(head, float)
    shank = cyl(h, h + a * length, d / 2.0, n=10)
    if kind == "flat":
        hd = cyl(h + a * 1.7, h, d, n=16)                      # countersunk head drawn as a short disc
    else:
        hd = cyl(h - a * 2.2, h, d * 0.92, n=16)
    slot = box(-0.3, 0.3, -d * 0.7, d * 0.7, -0.5, 0.5)       # cross mark is not drawn (too small to see)
    del slot
    return merge(shank, hd)


def polyline_len(pts):
    pts = np.asarray(pts, float)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])


def partial(pts, f):
    """First fraction f (0..1) of a polyline."""
    pts = np.asarray(pts, float)
    L = polyline_len(pts)
    s = max(1e-3, f) * L[-1]
    k = int(np.searchsorted(L, s))
    if k >= len(pts):
        return pts
    k = max(1, k)
    t = (s - L[k - 1]) / max(1e-9, L[k] - L[k - 1])
    return np.vstack([pts[:k], pts[k - 1] + t * (pts[k] - pts[k - 1])])


def ease(t):
    t = min(1.0, max(0.0, t))
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- the scene
class Item:
    def __init__(self, name, pd, color, opacity=1.0, group="", spec=0.25):
        self.name, self.pd, self.color, self.group = name, pd, color, group
        self.mapper = vtk.vtkPolyDataMapper()
        self.mapper.SetInputData(pd)
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.mapper)
        pr = self.actor.GetProperty()
        pr.SetColor(*color)
        pr.SetOpacity(opacity)
        pr.SetSpecular(spec)
        pr.SetSpecularPower(25)
        pr.SetDiffuse(0.85)
        pr.SetAmbient(0.18)
        self.base_opacity = opacity
        self.actor.VisibilityOff()
        self.wire = None                       # polyline (for growing wires)
        self.wire_r = 0.7


SCENE = {}
REN = None


def add(name, pd, color, opacity=1.0, group="", spec=0.25):
    it = Item(name, pd, color, opacity, group, spec)
    SCENE[name] = it
    REN.AddActor(it.actor)
    return it


def add_wire(name, pts, color, r=0.7, group=""):
    it = add(name, tube(pts, r), color, group=group, spec=0.1)
    it.wire, it.wire_r = np.asarray(pts, float), r
    return it


def cad_poly(shape, tol=0.25):
    v, f = shape.tessellate(tol, 0.3)
    pts = vtk.vtkPoints()
    for p in v:
        pts.InsertNextPoint(p.x, p.y, p.z)
    cells = vtk.vtkCellArray()
    for a, b, c in f:
        cells.InsertNextCell(3)
        cells.InsertCellPoint(a)
        cells.InsertCellPoint(b)
        cells.InsertCellPoint(c)
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(cells)
    n = vtk.vtkPolyDataNormals()
    n.SetInputData(pd)
    n.SetFeatureAngle(35)
    n.SplittingOn()
    n.Update()
    return n.GetOutput()


# ---- fastener and part positions (world, nod 0) ------------------------------------------------------------
def ring_pt(k, r):
    a = 2.0 * math.pi * k / 16
    return r * math.cos(a), r * math.sin(a)


CORNERS = [(sx * 60.0, sy * 71.25) for sx in (-1, 1) for sy in (-1, 1)]
HUB_PTS = [(17.0 * math.cos(math.radians(a)), ZP + 17.0 * math.sin(math.radians(a))) for a in (45, 135, 225, 315)]
TRUN_PTS = [(15.0 * math.cos(math.radians(a)), ZP + 15.0 * math.sin(math.radians(a))) for a in (90, 210, 330)]
SLEEVE_PTS = [(19.0 * math.cos(math.radians(a)), 19.0 * math.sin(math.radians(a))) for a in (90, 210, 330)]
BUTTRESS_PTS = [(-45.0, -24.0), (-45.0, 1.5)]
CARRIER_PTS = [(-32.5, -70.5), (25.5, -70.5), (-32.5, -21.5), (25.5, -21.5)]
PICO_PTS = [(42.8, 7.0), (54.2, 7.0), (42.8, 54.0), (54.2, 54.0)]
COVER_PTS = [(-22.6, -101.0), (22.6, -101.0), (-22.6, -150.0), (22.6, -150.0)]
FEET = [(-50.0, -80.0), (50.0, -80.0), (-50.0, 130.0), (50.0, 130.0)]
Z_CARRIER = FLOOR + 9.0               # top of the carrier standoffs
Z_PICO = FLOOR + 4.0
Z_SHELF = -114.9                      # top of the SP16T shelf
YC_L = (P.TOWER_L_Y0 + P.TOWER_L_Y1) / 2.0


def pico_pin(n):
    """Pico H upside down on its standoffs (board x 38..59, y 5..56, USB at y = 5).  Top of pin n."""
    if n <= 20:
        x, y = 57.4, 7.0 + 2.54 * (n - 1)
    else:
        x, y = 39.6, 7.0 + 2.54 * (40 - n)
    return np.array([x, y, Z_PICO + 1.0 + 8.0])


TMC_ROW1 = ["EN", "MS1", "MS2", "PDN", "PDN2", "CLK", "STEP", "DIR"]
TMC_ROW2 = ["VM", "GND_M", "2B", "2A", "1A", "1B", "VDD", "GND_L"]


def tmc_pin(name):
    z = FLOOR + 11.0 + 8.0
    if name in TMC_ROW1:
        return np.array([41.1 + 2.54 * TMC_ROW1.index(name), 72.35, z])
    return np.array([41.1 + 2.54 * TMC_ROW2.index(name), 59.65, z])


ADXL_PINS = ["GND", "VCC", "CS", "INT1", "INT2", "SDO", "SDA", "SCL"]


def adxl_pin(name):
    return np.array([6.1 + 2.54 * ADXL_PINS.index(name), 63.3, FLOOR + 1.6 + 8.5])


def sp_pin(row, side):
    """SP16T control header (2 x 8) on the board, row 1..8, side 'L' / 'R'."""
    return np.array([4.0 if side == "L" else 6.54, -9.0 + 2.54 * (row - 1), Z_SHELF + 1.6 + 8.5])


SP_CTRL = {1: (2, "L"), 2: (4, "R"), 3: (3, "R"), 4: (3, "L"), 5: (2, "R"), 6: (4, "L"), 7: (5, "R"), 8: (5, "L"), 9: (6, "R")}


def hdr40_pin(n):
    """CM4-IO-BASE-A 40-pin header along the far long edge of the board (y -72.5), pin 1 at x -19.5."""
    col, row = (n - 1) // 2, (n - 1) % 2
    return np.array([-19.5 + 2.54 * col, -72.2 + 2.54 * row, Z_CARRIER + 1.6 + 8.5])


WAGO_X = {"A": -24.0, "B": -9.0, "C": 6.0, "D": 21.0}


def wago_port(w, k):
    return np.array([WAGO_X[w] + 6.5, 22.0 + 1.5 + 3.4 * k, FLOOR + 8.0])


def sp_ports():
    out = []
    for y in (-42, -30, -18, -6, 6, 18, 30, 42):
        out.append(((-30.0, float(y)), (-1.0, 0.0)))
    for x in (-22, -9, 4, 17):
        out.append(((float(x), -50.0), (0.0, -1.0)))
        out.append(((float(x), 50.0), (0.0, 1.0)))
    return out


# ---------------------------------------------------------------- build everything (hidden at first)
def build():
    W_ = P.all_world()
    head_print = ["H1_box_tray", "H2_box_lid", "H3_sp16t_bridge", "H4_hub_right", "H5_trunnion_left", "H6_trunnion_clip",
                  "H7_column", "H9_ring_plate", "H11_column_sleeve", "H12_rail_buttress", "H8_rail_1", "H8_rail_2"]
    for n in head_print:
        add(n, cad_poly(W_[n]), WHITE_PLA, group="head")
    for k in range(17):
        add(f"H10_fin_{k:02d}", cad_poly(W_[f"H10_fin_{k:02d}"]), WHITE_PLA, group="head")
    add("H14_fin_gauge", cad_poly(W_["H14_fin_gauge"].translate(P.cq.Vector(-6.05, 57.0, -1.0))), (0.95, 0.75, 0.30))
    for n in ["B1_plinth_tray", "B2_deck_towers", "B3_tower_cover", "B5_tower_cap", "B4_battery_door"]:
        add(n, cad_poly(W_[n]), BASE_PLA, group="base")
    add("W1_wall_bracket", cad_poly(W_["W1_wall_bracket"]), WALL_PLA, group="wall")
    # films (on the +tangent face of each fin) and the centre film
    for k in range(17):
        if k == 16:
            pd = box(0.9, 1.1, -4.5, 4.5, -17.5, 17.5)
        else:
            a = 2.0 * math.pi * k / 16
            c, s = math.cos(a), math.sin(a)
            pd = box(55.5, 64.5, 0.9, 1.1, -17.5, 17.5)
            tr = vtk.vtkTransform()
            tr.RotateZ(math.degrees(a))
            f = vtk.vtkTransformPolyDataFilter()
            f.SetInputData(pd)
            f.SetTransform(tr)
            f.Update()
            pd = f.GetOutput()
            del c, s
        add(f"film_{k:02d}", pd, FILM, group="head", spec=0.4)
    # ---------- electronics in the box
    zc = Z_CARRIER
    add("carrier", merge(box(-36, 49, -74, -18, zc, zc + 1.6),
                         box(36, 49, -72, -58, zc + 1.6, zc + 16), box(36, 49, -55, -41, zc + 1.6, zc + 16),   # USB-A x2
                         box(38, 49, -36, -20, zc + 1.6, zc + 14),                                             # RJ45
                         box(-20.8, 31.3, -73.4, -68.4, zc + 1.6, zc + 4.1)), PCB, group="head")               # header base
    add("hdr40_pins", merge(*[cyl(hdr40_pin(n) - [0, 0, 6.0], hdr40_pin(n), 0.32, 4) for n in range(1, 41)]), GOLD, group="head")
    add("cm4", merge(box(-18, 37, -66, -26, zc + 4.6, zc + 6.0), box(-2, 16, -56, -40, zc + 6.0, zc + 7.6)), PCB, group="head")
    add("heatsink", merge(box(-18, 37, -66, -26, zc + 7.6, zc + 9.0),
                          *[box(-16 + 7.6 * i, -13.5 + 7.6 * i, -66, -26, zc + 9.0, zc + 13.0) for i in range(7)]),
        STEEL, group="head", spec=0.6)
    add("ax210", merge(box(-30, 4, -60, -36, zc - 3.2, zc - 1.6), box(-26, -4, -58, -38, zc - 5.0, zc - 3.2)), PCB_BLUE, group="head")
    add("pigtails", merge(tube([(-28, -42, zc - 4), (-40, -30, zc - 4), (-52, -8, FLOOR + 5)], 0.6),
                          tube([(-28, -48, zc - 4), (-46, -40, zc - 4), (-56, 10, FLOOR + 5)], 0.6)), DARK, group="head")
    zp = Z_PICO
    add("pico", merge(box(38, 59, 5, 56, zp, zp + 1.0), box(44.5, 52.5, 4, 9.5, zp - 3.0, zp),            # USB under (upside down)
                      box(38.6, 40.6, 6, 56, zp + 1.0, zp + 3.5), box(56.4, 58.4, 6, 56, zp + 1.0, zp + 3.5)), PCB, group="head")
    add("pico_pins", merge(*[cyl(pico_pin(n) - [0, 0, 6.0], pico_pin(n), 0.32, 4) for n in range(1, 41)]), GOLD, group="head")
    zt = FLOOR + 10.0
    add("tmc", merge(box(39.85, 60.15, 58.4, 73.6, zt, zt + 1.0), box(43, 57, 60, 72, zt - 8.5, zt)), PCB_PURPLE, group="head")
    add("tmc_pins", merge(*[cyl(tmc_pin(nm) - [0, 0, 7.0], tmc_pin(nm), 0.32, 4) for nm in TMC_ROW1 + TMC_ROW2]), GOLD, group="head")
    za = FLOOR + 0.8
    add("adxl", box(5, 25, 49.5, 65, za, za + 1.6), PCB_PURPLE, group="head")
    add("adxl_pins", merge(*[cyl(adxl_pin(nm) - [0, 0, 7.0], adxl_pin(nm), 0.32, 4) for nm in ADXL_PINS]), GOLD, group="head")
    add("adxl_arrow", merge(box(8, 20, 55.5, 57, za + 1.6, za + 1.8), cyl((20, 56.25, za + 1.7), (23, 56.25, za + 1.7), 1.6, 3)),
        WHITE_PLA, group="head")
    add("buck", merge(box(-24, -2, 2, 19, FLOOR, FLOOR + 1.6), box(-20, -8, 5, 16, FLOOR + 1.6, FLOOR + 8.5)), PCB_BLUE, group="head")
    for w, x in WAGO_X.items():
        add(f"wago_{w}", merge(box(x, x + 13, 22, 40, FLOOR, FLOOR + 8.5),
                               *[box(x + 1, x + 12, 22.5 + 3.4 * k, 24.0 + 3.4 * k, FLOOR + 8.5, FLOOR + 11.5) for k in range(5)]),
            WAGO, group="head")
    add("cap", cyl((-6.0, 46.0, FLOOR), (-6.0, 46.0, FLOOR + 13.0), 4.0, 20), (0.10, 0.20, 0.55), group="head")
    for i, y0 in enumerate((14.0, 46.0)):
        add(f"wilk{i + 1}", merge(box(-62, -30, y0, y0 + 26, FLOOR, FLOOR + 9),
                                  cyl((-30, y0 + 13, FLOOR + 4.5), (-24, y0 + 13, FLOOR + 4.5), 3.1, 10),
                                  cyl((-62, y0 + 6, FLOOR + 4.5), (-65, y0 + 6, FLOOR + 4.5), 3.1, 10),
                                  cyl((-62, y0 + 20, FLOOR + 4.5), (-65, y0 + 20, FLOOR + 4.5), 3.1, 10)), PCB_BLUE, group="head")
    add("att2", merge(cyl((-24, 59, FLOOR + 4.5), (-6, 59, FLOOR + 4.5), 4.2, 12)), GOLD, group="head", spec=0.6)
    zs = Z_SHELF
    ports = sp_ports()
    sp = [box(-30, 30, -50, 50, zs, zs + 1.6), box(2.7, 7.8, -10.3, 9.3, zs + 1.6, zs + 4.1), box(18, 26, 33, 39, zs + 1.6, zs + 7)]
    for (p, d) in ports + [((30.0, 0.0), (1.0, 0.0))]:
        p0 = np.array([p[0], p[1], zs + 0.8])
        dd = np.array([d[0], d[1], 0.0])
        sp.append(cyl(p0 - 2 * dd, p0 + 9.5 * dd, 3.1, 10))
    add("sp16t", merge(*sp[:3]), PCB, group="head")
    add("sp16t_sma", merge(*sp[3:]), GOLD, group="head", spec=0.6)
    add("sp_pins", merge(*[cyl(sp_pin(r, s) - [0, 0, 6.0], sp_pin(r, s), 0.32, 4) for r in range(1, 9) for s in "LR"]), GOLD, group="head")
    ad = []
    for (p, d) in ports:
        dd = np.array([d[0], d[1], 0.0])
        p0 = np.array([p[0], p[1], zs + 0.8]) + 9.5 * dd
        ad.append(cyl(p0, p0 + 8 * dd, 4.0, 6))
        ad.append(cyl(p0 + 8 * dd, p0 + 13 * dd, 2.2, 10))
    add("adapters", merge(*ad), GOLD, group="head", spec=0.6)
    add("att1", cyl((39.5, 0.0, zs + 0.8), (60.0, 0.0, zs + 0.8), 4.2, 12), GOLD, group="head", spec=0.6)
    # SMA jumpers (J1..J4)
    J = [[(-65, 20, FLOOR + 4.5), (-66, 0, -128), (-30, -30, -128), (-10, -36, zc - 4)],
         [(-65, 34, FLOOR + 4.5), (-66, 45, -128), (-66, 52, FLOOR + 4.5)],
         [(60, 0, zs + 0.8), (64, 20, -118), (40, 70, -118), (-40, 74, -125), (-65, 66, FLOOR + 4.5)],
         [(-24, 59, FLOOR + 4.5), (-30, 60, -128), (-36, -10, -128), (-20, -44, zc - 4)]]
    for i, pts in enumerate(J):
        add_wire(f"jumper{i + 1}", pts, WIRE["cable"], r=1.3, group="head")
    # ---------- antenna cables: film -> spine -> bundle -> box -> SP16T port
    YS, G = -11.25, (-45.0, -11.25)
    node = {}
    for k in range(16):
        x, y = ring_pt(k, 54.0)
        node[k] = float(np.clip(5.625 * round((x - abs(y - YS)) / 5.625), -45.0, 45.0))
    pmap = [3, 2, 1, 0, 7, 8, 10, 12, 14, 15, 13, 11, 9, 6, 4, 5]     # element -> port index (drawing only)
    for k in range(17):
        if k == 16:
            pts = [(1.0, -5.0, 0.0), (0.0, YS, 0.0)]
        else:
            x, y = ring_pt(k, 53.0)
            pts = [(x, y, 0.0), (node[k], YS, 0.0)]
        pts += [(G[0], YS, 0.0), (G[0], YS, -1.0), (G[0], YS, Z_TOP - 4.0), (G[0] + 0.5 * (k % 4), YS, -104.0)]
        if k < 16:
            (p, d) = ports[pmap[k]]
            tip = np.array([p[0], p[1], zs + 0.8]) + 22.5 * np.array([d[0], d[1], 0.0])
            pts += [(tip[0] * 0.6 + G[0] * 0.4, tip[1] * 0.6 + YS * 0.4, -104.0), tuple(tip)]
        else:
            pts += [(-62, 0, -110), (-63, 40, FLOOR + 4.5)]
        add_wire(f"ufl_{k:02d}", pts, WIRE["cable"], r=0.55, group="head")
    # ---------- dupont wires (Pico -> SP16T, TMC, ADXL), power wires, USB
    dup = []
    cols = ["yellow", "green", "blue", "purple", "white", "grey", "brown", "orange", "red", "black"]
    pico_sp = [(4, 1), (5, 2), (6, 3), (7, 4), (9, 5), (10, 6), (11, 7), (12, 8), (14, 9)]
    for i, (pn, ct) in enumerate(pico_sp):
        a = pico_pin(pn)
        b = sp_pin(*SP_CTRL[ct])
        dup.append((f"dup_sp{ct}", a, b, cols[i % 8]))
    dup.append(("dup_spgnd", pico_pin(3), sp_pin(8, "L"), "black"))
    for nm, a, b, c in dup:
        mid = (a + b) / 2.0
        pts = [a, a + [0, 0, 6], [a[0] - 4, a[1], -104.0], [mid[0], mid[1], -96.0], b + [0, 0, 4], b]
        add_wire(nm, pts, WIRE[c], r=0.65, group="head")
        park = [a, a + [0, 0, 6], a + [-6, -3, 22], a + [-8, -4, 30]]
        add_wire(nm + "_park", park, WIRE[c], r=0.65, group="head")
    tmc_w = [("dup_en", 15, "EN", "yellow"), ("dup_step", 16, "STEP", "green"), ("dup_dir", 17, "DIR", "blue"),
             ("dup_tgnd", 18, "GND_L", "black")]
    for nm, pn, tp, c in tmc_w:
        a, b = pico_pin(pn), tmc_pin(tp)
        add_wire(nm, [a, a + [0, 0, 5], [(a[0] + b[0]) / 2 - 3, (a[1] + b[1]) / 2, a[2] + 12], b + [0, 0, 5], b], WIRE[c], 0.65, "head")
    a, b1, b2 = pico_pin(36), tmc_pin("VDD"), adxl_pin("VCC")
    y = a + [-4, 0, 16]
    add_wire("dup_3v3", [a, a + [0, 0, 6], y, b1 + [0, 0, 6], b1], WIRE["orange"], 0.65, "head")
    add_wire("dup_3v3b", [y, y + [-14, 2, 0], b2 + [0, 0, 8], b2], WIRE["orange"], 0.65, "head")
    for nm, pn, ap, c in [("dup_sda", 19, "SDA", "green"), ("dup_scl", 20, "SCL", "yellow"), ("dup_agnd", 38, "GND", "black")]:
        a, b = pico_pin(pn), adxl_pin(ap)
        add_wire(nm, [a, a + [0, 0, 6], [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, a[2] + 12], b + [0, 0, 6], b], WIRE[c], 0.65, "head")
    add_wire("usb_pico", [(48.5, 4.0, Z_PICO - 1.5), (48.5, -6.0, Z_PICO - 1.5), (46, -14, FLOOR + 14), (42.5, -50, zc + 9)],
             WIRE["black"], 1.6, "head")
    # power
    def wp(w, k):
        return wago_port(w, k)
    add_wire("buck_in", [(-24, 6, FLOOR + 4), (-28, 14, FLOOR + 6), wp("A", 1) + [-6, 0, -2]], WIRE["red"], 0.9, "head")
    add_wire("buck_in_g", [(-24, 9, FLOOR + 4), (-12, 16, FLOOR + 6), wp("B", 1) + [-6, 0, -2]], WIRE["black"], 0.9, "head")
    add_wire("buck_out", [(-2, 6, FLOOR + 4), (4, 16, FLOOR + 6), wp("C", 0) + [-6, 0, -2]], WIRE["red"], 0.9, "head")
    add_wire("buck_out_g", [(-2, 9, FLOOR + 4), (19, 14, FLOOR + 6), wp("D", 0) + [-6, 0, -2]], WIRE["black"], 0.9, "head")
    for nm, w, k, b, c in [("pw_vm", "A", 2, tmc_pin("VM"), "red"), ("pw_vmg", "B", 2, tmc_pin("GND_M"), "black")]:
        a = wp(w, k)
        add_wire(nm, [a, a + [0, 0, 10], [10, 50, -118], b + [0, 0, 8], b], WIRE[c], 0.7, "head")
    for nm, w, k, pn, c in [("pw_c2", "C", 1, 2, "red"), ("pw_c4", "C", 2, 4, "red"), ("pw_d6", "D", 1, 6, "black"),
                            ("pw_d14", "D", 2, 14, "black")]:
        a, b = wp(w, k), hdr40_pin(pn)
        add_wire(nm, [a, a + [0, 0, 8], [b[0] + 2 * k, -10, -116], b + [0, 0, 8], b], WIRE[c], 0.7, "head")
    add_wire("pw_cap", [(-8, 46, FLOOR + 13), wp("A", 3)], WIRE["grey"], 0.4, "head")
    add_wire("pw_capg", [(-4, 46, FLOOR + 13), wp("B", 3)], WIRE["grey"], 0.4, "head")
    xh = np.array([22.0, 36.0, zs + 4.0])
    add_wire("pw_xh", [wp("C", 3), wp("C", 3) + [0, 0, 10], [30, 44, -114], xh + [0, 2, 6], xh], WIRE["red"], 0.7, "head")
    add_wire("pw_xhg", [wp("D", 3), wp("D", 3) + [0, 0, 10], [32, 46, -114], xh + [2, 2, 6], xh + [2, 0, 0]], WIRE["black"], 0.7, "head")
    add_wire("pw_xh_park", [wp("C", 3), wp("C", 3) + [0, 0, 10], [30, 44, -104]], WIRE["red"], 0.7, "head")
    add_wire("pw_xhg_park", [wp("D", 3), wp("D", 3) + [0, 0, 10], [32, 46, -104]], WIRE["black"], 0.7, "head")
    # wires that cross the axis: inside the box (head) and in the base (static)
    out_end = np.array([0.0, -94.0, ZP - 11.0])
    def cross(a, dx):
        return [a, a + [0, 0, 8], [a[0], a[1] - 10, -115.5], [dx, -60, -115.5], [dx, -78, ZP - 1.0 + 0.3 * dx],
                [dx * 0.4, -91, ZP - 4.0], out_end + [dx * 0.3, 0, 0]]
    add_wire("x9v", cross(wp("A", 0), -2.0), WIRE["red"], 0.9, "head")
    add_wire("xgnd", cross(wp("B", 0), 2.0), WIRE["black"], 0.9, "head")
    mcol = ["black", "green", "red", "blue"]
    for i, pn in enumerate(["2B", "2A", "1A", "1B"]):
        a = tmc_pin(pn)
        add_wire(f"xmot{i}", [a, a + [0, 0, 6], [48, 50, -116], [-1.5 + i, -40, -116.5], [-1.5 + i, -78, ZP + 1.5 - i],
                              [-1 + 0.6 * i, -91, ZP - 4.0], out_end + [-1 + 0.6 * i, 0, 0.5]], WIRE[mcol[i]], 0.55, "head")
    add("motor_plug_tmc", box(40.3, 49.0, 59.0, 63.0, tmc_pin("2B")[2] - 2, tmc_pin("2B")[2] + 7), WHITE_PLA, group="head")
    # base run: tower groove -> tunnel -> drop -> front strip
    base_run = [out_end, (0, -87.5, -150), (0, -87.5, -159), (50, -87.5, -159), (50, -87.5, -186), (50, -82, -196)]
    add_wire("b9v", base_run + [(50, -72, -199), (52, -66, -199.5)], WIRE["red"], 0.9)
    add_wire("bgnd", [p + np.array([1.5, 0, 0]) for p in map(np.array, base_run)] + [(53, -72, -199), (55, -66, -199.5)], WIRE["black"], 0.9)
    for i in range(4):
        dx = -1.5 + i
        add_wire(f"bmot{i}", [np.array(p, float) + [dx, 0, 0] for p in base_run] + [(52 + dx, 0, -197), (52 + dx, 100, -196),
                                                                                      (30, 108 + i, -194), (22, 108 + i, -193)],
                 WIRE[mcol[i]], 0.55)
    # ---------- base: PD trigger, power bank, motor + gearbox, feet
    zf = P.PL_Z0 + P.PL_FLOOR
    add("pdtrig", merge(box(44, 60, -82, -60, zf, zf + 1.6), box(46, 58, -66, -61, zf + 1.6, zf + 8)), PCB_RED)
    add("dip", box(48, 56, -78, -72, zf + 1.6, zf + 4.5), (0.15, 0.30, 0.75))
    add("usbc_cable", tube([(44, -80, zf + 4), (36, -86, zf + 6), (0, -88, zf + 10), (-30, -86, zf + 13), (-20, -89, zf + 14)], 1.6),
        WIRE["black"])
    add("bank", box(-37.5, 37.5, -90, 60, zf + 0.5, zf + 27.0), (0.20, 0.22, 0.26), spec=0.4)
    add("bank_label", box(-20, 20, -40, 10, zf + 27.0, zf + 27.2), (0.35, 0.65, 0.95))
    gb = merge(box(-20, 20, P.GB_PLANE, P.GB_PLANE + 30.8, P.GB_BOT, P.GB_TOP),
               box(-12, 12, P.GB_PLANE - 2.6, P.GB_PLANE, ZP - 23.0, ZP + 6.0))
    add("gearbox", gb, (0.70, 0.72, 0.75), group="drive", spec=0.5)
    add("dshaft", merge(cyl((0, P.GB_PLANE - 2.6, ZP), (0, 79.5, ZP), 4.0, 20)), STEEL, group="drive", spec=0.7)
    add("motor", merge(box(-21, 21, 88.9, 130.9, P.GB_BOT - 48.0, P.GB_BOT - 8.0),
                       box(-21, 21, 88.9, 130.9, P.GB_BOT - 8.0, P.GB_BOT), box(21, 27, 104, 118, -198, -190)), DARK, group="drive")
    add("motor_caps", merge(box(-21.2, 21.2, 88.7, 131.1, P.GB_BOT - 48.0, P.GB_BOT - 44.0),
                            box(-21.2, 21.2, 88.7, 131.1, P.GB_BOT - 8.0, P.GB_BOT - 4.0)), STEEL, group="drive", spec=0.5)
    add("feet", merge(*[cyl((x, y, P.PL_Z0 - 1.0), (x, y, P.PL_Z0 + 1.0), 6.0, 20) for (x, y) in FEET]), BLACK)
    # ---------- fasteners
    def scr(name, head, axis, length, kind, color, group="head"):
        it = add(name, screw_pd(head, axis, length, kind), color, group=group, spec=0.5)
        it.axis, it.length, it.head = unit(axis), length, np.asarray(head, float)
        return it

    def nut(name, c, axis, color, af=5.5, group="head"):
        it = add(name, hexnut(c, axis, af), color, group=group, spec=0.5)
        it.axis, it.length, it.head = unit(axis), 0.0, np.asarray(c, float)
        return it

    for i, (x, y) in enumerate(CORNERS):
        scr(f"N1_{i}", (x, y, Z_TOP), (0, 0, -1), 14.0, "pan", NYLON)
        nut(f"N1n_{i}", (x, y, Z_RIM - 7.8), (0, 0, 1), NYLON)
    for i, (x, y) in enumerate(SLEEVE_PTS + BUTTRESS_PTS):
        scr(f"N2_{i}", (x, y, Z_TOP + 2.5), (0, 0, -1), 10.0, "pan", NYLON)
        nut(f"N2n_{i}", (x, y, Z_RIM - 1.2), (0, 0, 1), NYLON)
    scr("N3", (-8.0, 0.0, Z_TOP - 5.5), (1, 0, 0), 16.0, "pan", NYLON)
    nut("N3n", (9.2, 0.0, Z_TOP - 5.5), (1, 0, 0), NYLON)
    for i, (x, z) in enumerate(HUB_PTS):
        scr(f"S1_{i}", (x, BY + 4.0, z), (0, -1, 0), 12.0, "pan", STEEL)
        nut(f"S1n_{i}", (x, 74.5, z), (0, 1, 0), STEEL)
    scr("S2", (0.0, 85.6, ZP + 13.5), (0, 0, -1), 10.0, "pan", STEEL)
    for i, (x, z) in enumerate(TRUN_PTS):
        scr(f"S3_{i}", (x, -BY - 3.0, z), (0, 1, 0), 10.0, "flat", STEEL)
        nut(f"S3n_{i}", (x, -BY + 3.7, z), (0, 1, 0), STEEL)
    for i, (x, y) in enumerate(CARRIER_PTS):
        scr(f"S4_{i}", (x, y, Z_CARRIER + 1.6), (0, 0, -1), 6.0, "pan", STEEL).actor.SetScale(1.0)
    for i, (x, y) in enumerate(PICO_PTS):
        scr(f"S5_{i}", (x, y, Z_PICO + 1.0), (0, 0, -1), 5.0, "pan", STEEL)
    for i, (x, y) in enumerate(P.BOSS_PTS):
        scr(f"S6_{i}", (x, y, P.DECK_Z1), (0, 0, -1), 12.0, "flat", STEEL, group="")
        nut(f"S6n_{i}", (x, y, P.DECK_Z0 - 7.65), (0, 0, 1), STEEL, group="")
    for i, (x, z) in enumerate(COVER_PTS):
        scr(f"S7_{i}", (x, P.COVER_Y1, z), (0, -1, 0), 12.0, "pan", STEEL, group="")
    for i, x in enumerate((-16.0, 16.0)):
        scr(f"S8_{i}", (x, YC_L, ZP + 6.0), (0, 0, -1), 20.0, "pan", STEEL, group="")
        nut(f"S8n_{i}", (x, YC_L, ZP - 8.15), (0, 0, 1), STEEL, group="")
    # ---------- wall
    add("wall", box(P.WALL_X - 12.0, P.WALL_X, -160, 210, -420, 60), (0.90, 0.91, 0.93), group="wallplane", spec=0.0)
    yc = (P.PL_Y0 + P.PL_Y1) / 2.0
    for i, z in enumerate((-140.0, -250.0)):
        add(f"cup_{i}", merge(cyl((P.WALL_X - 0.5, yc, z), (P.WALL_X + 9.0, yc, z), 32.0, 36),
                              cyl((P.WALL_X + 9.0, yc, z), (P.WALL_X + 20.0, yc, z), 6.0, 16)), (0.55, 0.75, 0.95), opacity=0.85,
            group="wall")
    add_wire("lanyard", [(P.WALL_X + 7, yc, -118), (P.WALL_X + 9, yc, -95), (P.WALL_X + 2, yc, -40), (P.WALL_X + 1, yc + 10, 0)],
             WIRE["blue"], 1.2, "wall")
    # temporary power bank + PD trigger for the first power test (C16), outside the box
    add("test_bank", box(90, 165, -150, 0, FLOOR - 30, FLOOR - 3), (0.20, 0.22, 0.26))
    add("test_pd", box(95, 115, -175, -158, FLOOR - 3, FLOOR - 1.4), PCB_RED)
    add_wire("test_9v", [(105, -175, FLOOR - 2), (60, -150, -150), (5, -100, -132), out_end], WIRE["red"], 0.9)


# ---------------------------------------------------------------- step script
def cam(focal, az, el, dist):
    f = np.asarray(focal, float)
    a, e = math.radians(az), math.radians(el)
    return (f + dist * np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)]), f)


V_ALL = cam((0, 25, -105), 32, 22, 820)
V_RING = cam((0, 0, 0), 28, 38, 380)
V_BOX = cam((0, -5, -132), 205, 68, 440)
V_BOXLOW = cam((5, 5, -128), 205, 60, 330)
V_CARRIER = cam((6, -46, -132), 125, 62, 270)
V_PICO = cam((42, 40, -132), 205, 64, 250)
V_WAGO = cam((-5, 30, -136), 200, 66, 260)
V_SP = cam((0, 0, -114), 205, 58, 340)
V_HUB = cam((0, 84, -117), 75, 18, 230)
V_TRUN = cam((0, -86, -117), -75, 15, 230)
V_LID = cam((0, -10, -60), 30, 28, 470)
V_TOPC = cam((-20, -10, -20), 20, 55, 330)
V_BASE = cam((0, 25, -190), 35, 42, 560)
V_BASEL = cam((15, -80, -170), -40, 30, 300)
V_BASER = cam((0, 110, -150), 50, 26, 460)
V_FRONT = cam((0, 25, -110), 8, 14, 760)
V_WALL = cam((-40, 25, -170), 38, 12, 980)
V_PRINT = cam((210, 150, 0), 30, 55, 900)


def S(code, title, text, dur, view, parts=None, ops=(), labels=(), note=None):
    return dict(code=code, title=title, text=text, dur=dur, view=view, parts=parts or [], ops=list(ops), labels=list(labels), note=note)


def screws(prefix, n, t0, t1):
    """Ops for screws prefix_0..prefix_{n-1}, one after another between t0 and t1 (seconds), with their nuts first."""
    ops = []
    dt = (t1 - t0) / max(1, n)
    for i in range(n):
        ops.append(("screw", f"{prefix}_{i}", t0 + i * dt, t0 + (i + 0.9) * dt))
    return ops


def nuts(prefix, n, t0, t1, off=(0, 0, 0)):
    ops = []
    dt = (t1 - t0) / max(1, n)
    for i in range(n):
        ops.append(("fly", f"{prefix}_{i}", off, t0 + i * dt, t0 + (i + 0.9) * dt))
    return ops


def grow(names, t0, t1):
    dt = (t1 - t0) / max(1, len(names))
    return [("grow", n, t0 + i * dt, t0 + (i + 0.95) * dt) for i, n in enumerate(names)]


FINS = [f"H10_fin_{k:02d}" for k in range(17)]
FILMS = [f"film_{k:02d}" for k in range(17)]
UFL = [f"ufl_{k:02d}" for k in range(17)]
SPW = [f"dup_sp{c}" for c in range(1, 10)] + ["dup_spgnd"]
BOX_IN = ["carrier", "hdr40_pins", "cm4", "heatsink", "ax210", "pigtails", "pico", "pico_pins", "tmc", "tmc_pins", "adxl",
          "adxl_pins", "adxl_arrow", "buck", "wago_A", "wago_B", "wago_C", "wago_D", "cap", "wilk1", "wilk2", "att2"]
XWIRES = ["x9v", "xgnd", "xmot0", "xmot1", "xmot2", "xmot3"]
BWIRES = ["b9v", "bgnd", "bmot0", "bmot1", "bmot2", "bmot3"]

STEPS = [
    S("0", "DeteCam 수신기 조립", "나사 하나, 선 하나씩 순서대로 보여 줍니다. 설계값으로 그린 그림이며 실물 사진이 아닙니다.", 4.0, V_FRONT,
      ops=[("show_final", None)]),
    S("A1", "A. 출력 부품 19가지 (PLA)", "Bambu Studio에서 판 1~7로 출력합니다. 서포트 없이 출력되도록 눕혀 둔 STL입니다(hw/cad/stl).", 7.0, V_PRINT,
      ops=[("hide_all", None), ("print_bed", None)]),
    S("B1", "B1. 고리 판에 날개 17장 꽂기", "날개 16장은 바깥 홈에, 1장은 가운데 홈에 꽂습니다. 윗끝의 작은 홈(필름 모서리 표시)이 바깥을 보게.", 7.0, V_RING,
      parts=["H9_ring_plate"], ops=[("fly", "H9_ring_plate", (0, 0, 60), 0.0, 1.0)] + [("fly", f, (0, 0, 70), 1.2 + 0.3 * i, 1.2 + 0.3 * i + 0.6) for i, f in enumerate(FINS)]),
    S("B2", "B2. 높이를 맞추고 붙이기", "높이 도구(H14)로 날개 윗끝을 판 위 21 mm에 맞춘 뒤, 판 아래에서 홈마다 순간접착제 한 방울.", 5.0, V_RING,
      ops=[("fly", "H14_fin_gauge", (0, 40, 30), 0.3, 1.5)], labels=[("21 mm", (0, 62, 20))]),
    S("B3", "B3. 필름 안테나 17장 붙이기", "긴 변을 세로로, 날개의 매끈한 면에. 케이블이 고리 안쪽을 보게. 30초씩 눌러 줍니다.", 7.0, V_RING,
      ops=[("hide", ["H14_fin_gauge"])] + [("fly", f, (0, 0, 25), 0.3 + 0.35 * i, 0.3 + 0.35 * i + 0.5) for i, f in enumerate(FILMS)]),
    S("C1", "C1. 상자 통에 너트 넣기", "네 모서리 홈에 나일론 너트 4개, 오른쪽 둥근 패드 안쪽에 M3 너트 4개(허브용). 핀셋으로 밀어 넣습니다.", 7.0, V_BOX,
      parts=["H1_box_tray"], ops=[("hide", ["H9_ring_plate"] + FINS + FILMS), ("fly", "H1_box_tray", (0, 0, -60), 0.0, 1.0)]
      + nuts("N1n", 4, 1.2, 3.8, (0, 0, 30)) + nuts("S1n", 4, 4.0, 6.6, (0, 0, 30)),
      labels=[("N1 너트(나일론)", (60, 71.25, Z_RIM - 7.8)), ("S1 너트", (17, 74.5, ZP + 12))]),
    S("C2", "C2. CM4 보드 준비", "뒷면 M.2에 어댑터와 AX210, 앞면에 CM4와 방열판. AX210 두 단자에 MHF4–SMA 피그테일.", 7.0, V_CARRIER,
      ops=[("fly", "carrier", (0, 0, 90), 0.0, 1.0), ("fly", "hdr40_pins", (0, 0, 90), 0.0, 1.0), ("fly", "ax210", (-60, 0, 0), 1.2, 2.4),
           ("fly", "cm4", (0, 0, 40), 2.6, 3.8), ("fly", "heatsink", (0, 0, 40), 4.0, 5.0), ("grow", "pigtails", 5.2, 6.6)]),
    S("C3", "C3. CM4 보드 고정 · S4 (M2.5 × 6 태핑) 4개", "바닥 왼쪽 뒤 기둥 4개 위에 올리고 4개를 조입니다. USB 단자가 상자 앞(방 쪽).", 6.0, V_CARRIER,
      ops=screws("S4", 4, 0.5, 5.5)),
    S("C4", "C4. Pico H 고정 · S5 (M2 × 5 태핑) 4개 · USB", "핀이 위로 오게 거꾸로 올려 4개를 조이고, 마이크로 USB로 CM4 보드에 잇습니다.", 6.5, V_PICO,
      ops=[("fly", "pico", (0, 0, 50), 0.0, 1.0), ("fly", "pico_pins", (0, 0, 50), 0.0, 1.0)] + screws("S5", 4, 1.2, 4.6) + [("grow", "usb_pico", 4.8, 6.2)],
      labels=[("Pico 1번 핀", pico_pin(1)), ("40번 핀", pico_pin(40))]),
    S("C5", "C5. TMC2209 · ADXL345 붙이기", "드라이버는 방열판이 아래 구멍으로(핀 위), 센서는 x 화살표가 방 쪽을 보게. 양면 폼테이프.", 6.0, V_PICO,
      ops=[("fly", "tmc", (0, 0, 40), 0.3, 1.5), ("fly", "tmc_pins", (0, 0, 40), 0.3, 1.5), ("fly", "adxl", (0, 0, 40), 2.5, 3.7),
           ("fly", "adxl_pins", (0, 0, 40), 2.5, 3.7), ("fly", "adxl_arrow", (0, 0, 40), 2.5, 3.7)],
      labels=[("TMC2209", (50, 66, -120)), ("ADXL345", (15, 57, -128)), ("x → 방 쪽", (23, 56, -138))]),
    S("C6", "C6. 벅 · 와고 A~D · 콘덴서 · 분배기 2개", "SP16T 선반이 올라갈 자리 아래 바닥에 양면 폼테이프로 붙입니다. 와고에 A~D 이름표.", 6.5, V_WAGO,
      ops=[("fly", "buck", (0, 0, 40), 0.2, 1.2)] + [("fly", f"wago_{w}", (0, 0, 40), 1.4 + 0.5 * i, 2.2 + 0.5 * i) for i, w in enumerate("ABCD")]
      + [("fly", "cap", (0, 0, 40), 3.6, 4.4), ("fly", "wilk1", (0, 0, 40), 4.6, 5.4), ("fly", "wilk2", (0, 0, 40), 5.0, 5.8),
         ("fly", "att2", (0, 0, 40), 5.4, 6.2)],
      labels=[("A 9V", wago_port("A", 4) + [0, 4, 6]), ("B GND", wago_port("B", 4) + [0, 4, 6]), ("C 5V", wago_port("C", 4) + [0, 4, 6]),
              ("D GND", wago_port("D", 4) + [0, 4, 6]), ("벅 9→5 V", (-13, 10, -128))]),
    S("C7", "C7. 신호선 ① Pico → SP16T 제어선 10가닥", "GP2~GP10(핀 4·5·6·7·9·10·11·12·14) → CTRL1~9, 핀 3 → GND. SP16T가 아직 없으니 끝을 위로 빼 둡니다.", 7.0, V_PICO,
      ops=grow([n + "_park" for n in SPW], 0.3, 6.5), labels=[("GP2 → CTRL1", pico_pin(4)), ("GND → GND", pico_pin(3))]),
    S("C8", "C8. 신호선 ② TMC2209 · ADXL345", "핀 15 EN · 16 STEP · 17 DIR · 18 GND → TMC, 핀 36 3.3 V(Y 갈래) → TMC VDD · ADXL VCC, 19 SDA · 20 SCL · 38 GND → ADXL.", 8.0, V_PICO,
      ops=grow(["dup_en", "dup_step", "dup_dir", "dup_tgnd", "dup_3v3", "dup_3v3b", "dup_sda", "dup_scl", "dup_agnd"], 0.3, 7.5),
      labels=[("EN", tmc_pin("EN")), ("STEP", tmc_pin("STEP")), ("DIR", tmc_pin("DIR")), ("3.3 V(주황)", pico_pin(36)),
              ("SDA", adxl_pin("SDA")), ("SCL", adxl_pin("SCL"))]),
    S("C9", "C9. 전원선 (와고 A · B · C · D)", "벅 입·출력, TMC VM · GND, CM4 40핀 2·4번(5 V) · 6·14번(GND), 100 µF 콘덴서. 가는 선은 접어서 꽂습니다.", 8.5, V_WAGO,
      ops=grow(["buck_in", "buck_in_g", "buck_out", "buck_out_g", "pw_vm", "pw_vmg", "pw_c2", "pw_c4", "pw_d6", "pw_d14", "pw_cap",
                "pw_capg", "pw_xh_park", "pw_xhg_park"], 0.3, 8.0),
      labels=[("40핀 2·4번 = 5 V", hdr40_pin(4)), ("6·14번 = GND", hdr40_pin(14)), ("1·17번(3.3 V)에는 꽂지 않음", hdr40_pin(1))]),
    S("C10", "C10. SP16T 선반 얹기 · 제어선 꽂기", "SP16T를 선반 네 모서리 턱에 끼우고(8포트 긴 변이 벽 쪽) 선반을 양옆 턱 위에. 빼 둔 선을 헤더와 XH 단자에.", 8.0, V_SP,
      ops=[("fly", "H3_sp16t_bridge", (0, 0, 70), 0.2, 1.4), ("fly", "sp16t", (0, 0, 70), 1.6, 2.8), ("fly", "sp_pins", (0, 0, 70), 1.6, 2.8),
           ("fly", "sp16t_sma", (0, 0, 70), 1.6, 2.8)] + [("swap", n + "_park", n, 3.2 + 0.35 * i) for i, n in enumerate(SPW)]
      + [("swap", "pw_xh_park", "pw_xh", 6.9), ("swap", "pw_xhg_park", "pw_xhg", 7.1)],
      labels=[("CTRL 헤더", sp_pin(1, "L") + [0, 0, 4]), ("XH 5 V", (24, 36, -108))]),
    S("C11", "C11. 어댑터 16개 · 6 dB 감쇠기", "포트 16개에 SMA–U.FL 어댑터, 공통 포트에 6 dB 감쇠기를 손으로 돌려 끼웁니다.", 5.0, V_SP,
      ops=[("fly", "adapters", (0, 0, 40), 0.3, 2.3), ("fly", "att1", (30, 0, 0), 2.6, 3.8)]),
    S("C12", "C12. 무선 점퍼 4개", "조립 순서 3절 그림대로: 분배기 1 → AX210 체인 A, 분배기 1 → 분배기 2, 감쇠기 → 분배기 2, 분배기 2 → 감쇠기 → 체인 B.", 7.0, V_BOX,
      ops=grow(["jumper1", "jumper2", "jumper3", "jumper4"], 0.3, 6.5)),
    S("C13", "C13. 오른쪽 허브 · S1 (M3 × 12) 4개 · 고정나사 S2", "허브를 오른쪽 바깥 패드에 대고 4개를 안쪽 너트에 조입니다. 위 구멍의 S2는 2~3바퀴만.", 7.0, V_HUB,
      ops=[("fly", "H4_hub_right", (0, 50, 0), 0.2, 1.4)] + screws("S1", 4, 1.6, 5.2) + [("screw_part", "S2", 5.4, 6.6, 0.6)],
      labels=[("D 구멍(감속기 축)", (0, 88.5, ZP)), ("S2", (0, 85.6, ZP + 14))]),
    S("C14", "C14. 왼쪽 속 빈 축 · S3 (M3 × 10 접시머리) 3개", "넓은 판을 왼쪽 바깥에 대고 바깥에서 3개를 넣어 안쪽 너트에 조입니다.", 6.0, V_TRUN,
      ops=[("fly", "H5_trunnion_left", (0, -50, 0), 0.2, 1.4)] + nuts("S3n", 3, 1.5, 2.4, (0, 30, 0)) + screws("S3", 3, 2.5, 5.6)),
    S("C15", "C15. 축을 건너는 선 6가닥", "와고 A · B의 9 V(빨강) · GND(검정), 모터 선 4가닥을 속 빈 축 구멍으로 밖으로 뺍니다. 모터 커넥터는 길게 세워 통과.", 7.0, V_BOX,
      ops=grow(XWIRES, 0.3, 6.5), labels=[("속 빈 축", (0, -86, ZP + 12))]),
    S("C16", "C16. 뚜껑 닫기 전 첫 전원 시험", "PD 트리거에 잠시 잇고: 와고 A–B 8.5~9.5 V, C–D 5.0~5.3 V, I2C 0x53, Vref 1.0 V. 끈 뒤 모터 선을 TMC에 꽂습니다.", 8.0, V_ALL,
      ops=[("show", ["test_bank", "test_pd"]), ("grow", "test_9v", 0.3, 1.5), ("fly", "motor_plug_tmc", (0, 0, 25), 6.2, 7.4)],
      note=["✓ 9 V (A–B)", "✓ 5 V (C–D)", "✓ CM4 · Pico 켜짐", "✓ ADXL345 0x53", "✓ Vref 1.0 V", "→ 끄고 모터 선 꽂기"]),
    S("D1", "D1. 뚜껑에 기둥 · 가로 핀 N3 (M3 × 16 나일론)", "기둥 아래 돌기를 뚜껑 가운데 구멍에 꽂고, 뚜껑 아래 가로 구멍에 나일론 나사 + 너트.", 6.0, V_LID,
      ops=[("hide", ["test_bank", "test_pd", "test_9v"]), ("lift", ["H2_box_lid", "H7_column", "N3", "N3n"], 70.0),
           ("fly", "H2_box_lid", (0, 0, 120), 0.2, 1.4), ("fly", "H7_column", (0, 0, 120), 1.6, 2.8), ("screw", "N3", 3.2, 4.6),
           ("fly", "N3n", (12, 0, 0), 4.7, 5.6)]),
    S("D2", "D2. 기둥 덧대 · N2 (M3 × 10 나일론) 3개", "덧대를 기둥 위로 씌워 내리고, 위에서 3개를 넣어 뚜껑 아래 너트에 조입니다. 덧대는 고리 면 50 mm 아래까지만.", 6.0, V_LID,
      ops=[("lift", ["H11_column_sleeve"] + [f"N2_{i}" for i in range(3)] + [f"N2n_{i}" for i in range(3)], 70.0),
           ("fly", "H11_column_sleeve", (0, 0, 90), 0.2, 1.4)] + nuts("N2n", 3, 1.5, 2.4, (0, 0, -20)) + screws("N2", 3, 2.5, 5.6)),
    S("D3", "D3. 레일 2개 · 레일 받침 · N2 2개", "레일을 뚜껑 뒤 홈에 꽂고 받침을 씌워 2개로 조입니다. 레일 아래 끝은 순간접착제 한 방울.", 6.0, V_LID,
      ops=[("lift", ["H8_rail_1", "H8_rail_2", "H12_rail_buttress", "N2_3", "N2_4", "N2n_3", "N2n_4"], 70.0),
           ("fly", "H8_rail_1", (0, 0, 120), 0.2, 1.2), ("fly", "H8_rail_2", (0, 0, 120), 0.6, 1.6),
           ("fly", "H12_rail_buttress", (0, 0, 80), 1.9, 2.9), ("fly", "N2n_3", (0, 0, -20), 3.0, 3.4), ("fly", "N2n_4", (0, 0, -20), 3.2, 3.6),
           ("screw", "N2_3", 3.7, 4.7), ("screw", "N2_4", 4.8, 5.8)]),
    S("D4", "D4. 고리 판 얹기", "가운데 구멍을 기둥 윗 돌기에, 레일 윗끝을 판의 레일 홈에 맞춰 내려 얹습니다.", 5.0, V_LID,
      ops=[("lift", ["H9_ring_plate"] + FINS + FILMS, 70.0), ("show_fly", ["H9_ring_plate"] + FINS + FILMS, (0, 0, 80), 0.3, 2.5)]),
    S("D5", "D5. 안테나 케이블 17가닥", "판 위에서 벽 쪽으로 비스듬히(약 45°) 등뼈에 모아, 판 뒤 묶음 구멍 → 두 레일 사이 → 상자 안 → SP16T 포트(4절 표).", 9.0, V_TOPC,
      ops=[("lift", UFL, 70.0)] + grow(UFL, 0.3, 8.5)),
    S("D6", "D6. 뚜껑 닫기 · N1 (M3 × 14 나일론) 4개", "선이 끼지 않았는지 보고 뚜껑을 덮어 네 모서리를 조입니다. 머리 완성.", 7.0, V_ALL,
      ops=[("unlift", None, 0.3, 2.3)] + screws("N1", 4, 2.6, 6.6)),
    S("E1", "E1. 받침 통 · 고무발 4개 · 너트 6개", "바닥 둥근 자리에 고무발, 안쪽 기둥 6개의 옆 홈에 M3 너트.", 7.0, V_BASE,
      ops=[("away_head", 260.0), ("fly", "B1_plinth_tray", (0, 0, -60), 0.2, 1.2), ("fly", "feet", (0, 0, -20), 1.4, 2.2)]
      + nuts("S6n", 6, 2.6, 6.6, (0, 0, 25))),
    S("E2", "E2. PD 트리거(9 V) · USB-C 케이블", "딥 스위치를 9 V로. 앞쪽 칸 왼쪽 끝에 붙이고 USB-C 케이블 끝을 배터리 칸 쪽으로 넘겨 둡니다.", 6.0, V_BASEL,
      ops=[("fly", "pdtrig", (0, 0, 40), 0.3, 1.3), ("fly", "dip", (0, 0, 40), 0.3, 1.3), ("fly", "usbc_cable", (0, 0, 30), 2.0, 3.2)],
      labels=[("PD 트리거 9 V", (52, -70, -196)), ("딥 스위치", (52, -75, -197))]),
    S("E3", "E3. 덮개판 고정 · S6 (M3 × 12 접시머리) 6개", "왼쪽 탑 축받이 앞 홈에 너트 2개를 먼저 넣고, 덮개판을 얹어 6개를 조입니다.", 8.0, V_BASE,
      ops=[("fly", "S8n_0", (0, 25, 0), 0.2, 0.8), ("fly", "S8n_1", (0, 25, 0), 0.5, 1.1), ("fly", "B2_deck_towers", (0, 0, 80), 1.3, 2.5)]
      + screws("S6", 6, 2.8, 7.6)),
    S("F1", "F1. 머리 내려놓기", "속 빈 축을 왼쪽 탑의 반달 축받이에, 허브를 오른쪽 탑 벽 앞에. 오른쪽은 아직 받침이 없으니 손으로 받칩니다.", 6.0, V_ALL,
      ops=[("head_down", None, 0.3, 4.5)]),
    S("F2", "F2. 선 6가닥을 탑 홈에 눌러 넣기", "탑 바깥쪽 홈(위 → 아래) → 아래 가로 홈 → 덮개판 내림 구멍 → 받침 앞 칸. 홈 위는 전기 테이프로.", 7.0, V_BASEL,
      ops=[("opacity", "B2_deck_towers", 0.35), ("opacity", "B1_plinth_tray", 0.55)] + grow(BWIRES[:2], 0.3, 3.0) + grow(BWIRES[2:], 3.0, 6.5),
      labels=[("탑 홈", (0, -91, -140)), ("내림 구멍", (50, -91, -172))]),
    S("F3", "F3. 9 V 선 → PD 트리거 나사 단자", "빨강을 +, 검정을 −에 끼우고 작은 일자 드라이버로 조입니다.", 4.5, V_BASEL,
      labels=[("+ 빨강", (52, -66, -198)), ("− 검정", (55, -66, -198))]),
    S("F4", "F4. 축받이 뚜껑 · S8 (M3 × 20) 2개 · 축 고리", "뚜껑을 덮고 위에서 2개를 조인 뒤, 축 끝 홈에 축 고리(H6)를 눌러 끼웁니다.", 7.0, V_TRUN,
      ops=[("opacity", "B2_deck_towers", 1.0), ("opacity", "B1_plinth_tray", 1.0), ("fly", "B5_tower_cap", (0, 0, 50), 0.2, 1.4)]
      + screws("S8", 2, 1.6, 4.6) + [("fly", "H6_trunnion_clip", (0, -30, 0), 5.0, 6.4)]),
    S("F5", "F5. 모터 선 → 모터", "받침 앞 칸의 모터 선을 모터 칸 쪽으로 보내 모터 옆 커넥터에 꽂습니다(감속기를 넣기 전에).", 5.0, V_BASER,
      ops=[("drive_set", 140.0), ("show", ["motor", "motor_caps", "gearbox", "dshaft"]), ("opacity", "B2_deck_towers", 0.35),
           ("opacity", "B1_plinth_tray", 0.55)], labels=[("모터 커넥터", (24, 250, -194))]),
    S("F6", "F6. 감속기 · 모터 밀어 넣기", "열린 쪽에서 밀어 넣으며 머리를 천천히 앞뒤로 돌려 D축 평면을 허브 구멍에 맞춥니다.", 7.0, V_BASER,
      ops=[("slide_drive", 140.0, 0.3, 6.0)]),
    S("F7", "F7. 허브 고정나사 S2 조이기", "상자와 탑 벽 사이 틈으로 드라이버를 넣어 조입니다. 세게 조이지 않습니다.", 4.5, V_HUB,
      ops=[("opacity", "B2_deck_towers", 1.0), ("opacity", "B1_plinth_tray", 1.0), ("screw_part", "S2", 0.3, 3.5, 1.0, 0.6)]),
    S("F8", "F8. 탑 덮개 · S7 (M3 × 12) 4개", "덮개를 대고 4개를 조입니다(구멍에 스스로 나사산을 냅니다).", 6.0, V_BASER,
      ops=[("fly", "B3_tower_cover", (0, 60, 0), 0.2, 1.4)] + screws("S7", 4, 1.6, 5.6)),
    S("F9", "F9. 보조배터리 넣고 문 닫기", "문을 열고 배터리를 밀어 넣어 USB-C 케이블을 꽂은 뒤 문을 끼워 닫습니다.", 6.0, V_BASEL,
      ops=[("opacity", "B2_deck_towers", 0.35), ("opacity", "B1_plinth_tray", 0.55), ("fly", "bank", (0, -200, 0), 0.3, 2.5),
           ("fly", "bank_label", (0, -200, 0), 0.3, 2.5), ("fly", "B4_battery_door", (0, -60, 0), 3.0, 4.2),
           ("opacity_at", "B2_deck_towers", 1.0, 5.0), ("opacity_at", "B1_plinth_tray", 1.0, 5.0)]),
    S("G1", "G1. 켜고 끄덕여 보기  −20° · 0° · +20°", "날개 · 상자가 받침에 닿지 않는지(계산 3 mm 넘게 떨어짐), 선이 당겨지지 않는지, U.FL이 빠지지 않는지 봅니다.", 9.0, V_FRONT,
      ops=[("nod", None)]),
    S("G2", "완성  가로 23.8 × 세로 14 × 높이 22.7 cm, 약 2.2 kg(추정)", "보조배터리와 모터가 받침 안에 들어가 밖으로 나오는 것이 없습니다. 기본은 선반 · 세면대 위에 그대로 세웁니다.", 7.0, V_ALL,
      ops=[("orbit", 70.0)]),
    S("H1", "H. 벽에 걸 때: 벽 거치대 + 흡착판 2개 + 안전끈", "매끈한 타일 · 유리는 레버식 흡착판 2개, 페인트 · 벽지는 떼어지는 접착 띠, 자기 시설은 키홀에 나사. 축은 벽에서 120 mm.", 8.0, V_WALL,
      ops=[("show", ["wall"]), ("fly", "W1_wall_bracket", (0, 0, 80), 0.3, 1.8), ("fly", "cup_0", (-40, 0, 0), 2.0, 3.0),
           ("fly", "cup_1", (-40, 0, 0), 2.3, 3.3), ("whole_fly", (0, 0, 120), 3.6, 6.0), ("grow", "lanyard", 6.0, 7.6)]),
    S("H2", "큰 흡착판 하나는 권하지 않습니다", "약 2.2 kg이 벽에서 10 cm 떨어져 있어 위쪽이 떨어지려는 힘을 받습니다. 한 점이면 돌아가거나 바람이 빠질 때 통째로 떨어집니다.", 6.0, V_WALL,
      ops=[]),
]


# ---------------------------------------------------------------- runtime state
class State:
    def __init__(self):
        self.visible = set()
        self.opacity = {}
        self.lift = {}            # name -> z offset (exploded lid group)
        self.head_dz = 0.0        # head away from the base (E steps)
        self.whole = np.zeros(3)  # whole receiver offset (H1)
        self.nod = 0.0
        self.swapped = set()
        self.drive_dy = 0.0
        self.persist = {}         # name -> offset kept after a step (screws left half way)


def head_names():
    return [n for n, it in SCENE.items() if it.group == "head"]


def place(it, st, extra=(0, 0, 0), spin=0.0):
    """Set the user transform of an item: nod (head), lifts, offsets."""
    t = vtk.vtkTransform()
    t.PostMultiply()
    if spin and getattr(it, "axis", None) is not None:
        t.Translate(*(-it.head))
        t.RotateWXYZ(spin, *it.axis)
        t.Translate(*it.head)
    off = np.array(extra, float) + np.array([0, 0, st.lift.get(it.name, 0.0)])
    if it.group == "drive":
        off = off + np.array([0, st.drive_dy, 0])
    t.Translate(*off)
    if it.group == "head":
        t.Translate(0, 0, -ZP)
        t.RotateY(st.nod)
        t.Translate(0, 0, ZP)
        t.Translate(0, 0, st.head_dz)
    if it.group != "wallplane" and it.group != "wall":
        t.Translate(*st.whole)
    it.actor.SetUserTransform(t)


def apply(st, t, step, frames_t):
    """Draw the state of `step` at time t (s).  Ops that started before t are applied."""
    extras = {}
    spins = {}
    for op in step["ops"]:
        k = op[0]
        if k == "fly":
            _, n, off, t0, t1 = op
            if t >= t0:
                st.visible.add(n)
            f = ease((t - t0) / max(1e-6, t1 - t0))
            extras[n] = (1 - f) * np.array(off, float)
            if t < t0:
                SCENE[n].actor.VisibilityOff()
        elif k == "show_fly":
            _, ns, off, t0, t1 = op
            f = ease((t - t0) / max(1e-6, t1 - t0))
            for n in ns:
                st.visible.add(n)
                extras[n] = (1 - f) * np.array(off, float)
        elif k == "screw" or k == "screw_part":
            n, t0, t1 = op[1], op[2], op[3]
            frac_end = op[4] if len(op) > 4 else 1.0
            frac_start = op[5] if len(op) > 5 else 0.0
            it = SCENE[n]
            if t >= t0:
                st.visible.add(n)
            f = frac_start + (frac_end - frac_start) * ease((t - t0) / max(1e-6, t1 - t0))
            dist = (it.length + 18.0) * (1.0 - f)
            extras[n] = -it.axis * dist
            spins[n] = -720.0 * f if f > 0.4 else 0.0
            if k == "screw_part":
                st.persist[n] = extras[n] if frac_end < 1.0 else np.zeros(3)
        elif k == "grow":
            _, n, t0, t1 = op
            if t >= t0:
                st.visible.add(n)
                it = SCENE[n]
                if it.wire is None:
                    continue
                f = min(1.0, (t - t0) / max(1e-6, t1 - t0))
                it.mapper.SetInputData(tube(partial(it.wire, f), it.wire_r) if f < 1.0 else it.pd)
        elif k == "show":
            for n in op[1]:
                st.visible.add(n)
        elif k == "drive_set":
            st.drive_dy = op[1]
        elif k == "hide_all":
            st.visible.clear()
        elif k == "hide":
            for n in op[1]:
                st.visible.discard(n)
        elif k == "swap":
            _, a, b, t0 = op
            if t >= t0:
                st.visible.discard(a)
                st.visible.add(b)
        elif k == "opacity":
            SCENE[op[1]].actor.GetProperty().SetOpacity(op[2])
        elif k == "opacity_at":
            if t >= op[3]:
                SCENE[op[1]].actor.GetProperty().SetOpacity(op[2])
        elif k == "lift":
            for n in op[1]:
                st.lift[n] = op[2]
        elif k == "unlift":
            _, _, t0, t1 = op
            f = ease((t - t0) / max(1e-6, t1 - t0))
            for n in list(st.lift):
                st.lift[n] = 70.0 * (1 - f)
            if f >= 1.0:
                st.lift = {}
        elif k == "away_head":
            st.head_dz = op[1]
        elif k == "head_down":
            _, _, t0, t1 = op
            st.head_dz = 260.0 * (1 - ease((t - t0) / max(1e-6, t1 - t0)))
        elif k == "slide_drive":
            _, d0, t0, t1 = op
            st.drive_dy = d0 * (1 - ease((t - t0) / max(1e-6, t1 - t0)))
            st.nod = 4.0 * math.sin(2 * math.pi * min(1.0, max(0.0, (t - t0) / (t1 - t0))) * 2) * (1 if t < t1 else 0)
        elif k == "nod":
            T = step["dur"]
            u = t / T
            st.nod = 20.0 * math.sin(2 * math.pi * u) if u < 1.0 else 0.0
        elif k == "orbit":
            pass
        elif k == "whole_fly":
            _, off, t0, t1 = op
            st.whole = (1 - ease((t - t0) / max(1e-6, t1 - t0))) * np.array(off, float)
        elif k == "show_final":
            for n, it in SCENE.items():
                if not n.endswith("_park") and not n.startswith("test_") and it.group not in ("wall", "wallplane") \
                        and n not in ("H14_fin_gauge",):
                    st.visible.add(n)
        elif k == "print_bed":
            pass
    for n, it in SCENE.items():
        if n in st.visible:
            it.actor.VisibilityOn()
            place(it, st, extras.get(n, st.persist.get(n, (0, 0, 0))), spins.get(n, 0.0))
        else:
            it.actor.VisibilityOff()


# ---------------------------------------------------------------- rendering
RW = None
FONTS = {}


def font(size, bold=False):
    key = (size, bold)
    if key not in FONTS:
        FONTS[key] = ImageFont.truetype(FONT_B if bold else FONT_R, size, index=1)
    return FONTS[key]


def project(p):
    c = vtk.vtkCoordinate()
    c.SetCoordinateSystemToWorld()
    c.SetValue(*map(float, p))
    x, y = c.GetComputedDoubleDisplayValue(REN)
    return x, H - y


def wrap(text, fnt, width, draw):
    words, lines, cur = text.split(" "), [], ""
    for w_ in words:
        trial = (cur + " " + w_).strip()
        if draw.textlength(trial, font=fnt) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = w_
    lines.append(cur)
    return lines


def overlay(img, step, t, idx, n_steps, head_pos=None):
    d = ImageDraw.Draw(img, "RGBA")
    # title
    ft, fb = font(30, True), font(22)
    tw = d.textlength(step["title"], font=ft)
    d.rounded_rectangle((24, 20, 24 + tw + 36, 74), 12, fill=(255, 255, 255, 228), outline=(40, 70, 120, 255), width=2)
    d.text((42, 26), step["title"], font=ft, fill=(20, 30, 50, 255))
    # instruction
    lines = wrap(step["text"], fb, W - 120, d)
    hh = 18 + 32 * len(lines)
    d.rounded_rectangle((24, H - 40 - hh, W - 24, H - 34), 12, fill=(255, 255, 255, 232), outline=(200, 205, 215, 255), width=1)
    for i, ln in enumerate(lines):
        d.text((44, H - 40 - hh + 8 + 32 * i), ln, font=fb, fill=(25, 30, 40, 255))
    # progress
    f = (idx + min(1.0, t / step["dur"])) / n_steps
    d.rectangle((24, H - 22, W - 24, H - 16), fill=(210, 215, 225, 255))
    d.rectangle((24, H - 22, 24 + (W - 48) * f, H - 16), fill=(45, 110, 200, 255))
    # labels
    fl = font(18, True)
    for i, (txt, p) in enumerate(step["labels"]):
        if t < 0.6:
            continue
        q = np.array(p, float)
        if head_pos is not None:
            q = head_pos(q)
        x, y = project(q)
        lx, ly = x + 40 + 0 * i, y - 46 - 26 * (i % 3)
        lw = d.textlength(txt, font=fl)
        d.line((x, y, lx, ly + 12), fill=(30, 60, 120, 230), width=2)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(30, 60, 120, 255))
        d.rounded_rectangle((lx - 6, ly, lx + lw + 8, ly + 26), 6, fill=(255, 250, 225, 238), outline=(30, 60, 120, 255), width=1)
        d.text((lx + 1, ly + 1), txt, font=fl, fill=(20, 25, 35, 255))
    if step.get("note"):
        fn = font(22, True)
        k = int(min(len(step["note"]), max(0, (t - 1.5) / 0.75)))
        y0 = 100
        for j, s in enumerate(step["note"][:k]):
            d.rounded_rectangle((W - 330, y0 + 40 * j, W - 30, y0 + 40 * j + 34), 8, fill=(255, 255, 255, 235),
                                outline=(40, 140, 80, 255), width=2)
            d.text((W - 316, y0 + 40 * j + 3), s, font=fn, fill=(20, 90, 40, 255))
    return img


def set_cam(view, prev_view, t, step, extra_az=0.0):
    f = ease(t / 1.2) if prev_view is not None else 1.0
    pos0, foc0 = prev_view if prev_view is not None else view
    pos1, foc1 = view
    pos = (1 - f) * np.asarray(pos0) + f * np.asarray(pos1)
    foc = (1 - f) * np.asarray(foc0) + f * np.asarray(foc1)
    if extra_az:
        v = pos - foc
        a = math.radians(extra_az)
        v = np.array([v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a), v[2]])
        pos = foc + v
    c = REN.GetActiveCamera()
    c.SetFocalPoint(*foc)
    c.SetPosition(*pos)
    c.SetViewUp(0, 0, 1)
    c.SetViewAngle(30)
    REN.ResetCameraClippingRange()


def grab():
    RW.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(RW)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    im = w2i.GetOutput()
    from vtk.util.numpy_support import vtk_to_numpy
    a = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(H, W, -1)[::-1, :, :3]
    return Image.fromarray(np.ascontiguousarray(a))


PRINT_ACTORS = []


def print_bed(on):
    """A1: the STL files laid out on two print beds."""
    if not PRINT_ACTORS and on:
        files = ["B1_plinth_tray", "B2_deck_towers", "H1_box_tray", "H2_box_lid", "H3_sp16t_bridge", "H9_ring_plate", "H10_fins_x17",
                 "W1_wall_bracket", "H7_column", "H11_column_sleeve", "H12_rail_buttress", "B3_tower_cover", "B4_battery_door",
                 "H4_hub_right", "H5_trunnion_left", "B5_tower_cap", "H6_trunnion_clip", "H8_rails_x2", "H14_fin_gauge"]
        x, y, rowh, col = 0.0, 0.0, 0.0, 0
        for f in files:
            r = vtk.vtkSTLReader()
            r.SetFileName(os.path.join(HERE, "..", "cad", "stl", f + ".stl"))
            r.Update()
            pd = r.GetOutput()
            b = pd.GetBounds()
            w_, h_ = b[1] - b[0], b[3] - b[2]
            if x + w_ > 720:
                x, y, rowh = 0.0, y + rowh + 30, 0.0
            m = vtk.vtkPolyDataMapper()
            m.SetInputData(pd)
            a = vtk.vtkActor()
            a.SetMapper(m)
            col_ = BASE_PLA if f[0] == "B" else (WALL_PLA if f[0] == "W" else WHITE_PLA)
            a.GetProperty().SetColor(*col_)
            a.GetProperty().SetSpecular(0.25)
            a.SetPosition(x - b[0], y - b[2], -b[4])
            REN.AddActor(a)
            PRINT_ACTORS.append((a, f.split("_")[0], (x + w_ / 2, y + h_ / 2, min(40.0, b[5] - b[4]))))
            x += w_ + 30
            rowh = max(rowh, h_)
    for a, _, _ in PRINT_ACTORS:
        a.SetVisibility(1 if on else 0)


def render(which):
    global REN, RW
    os.makedirs(OUT, exist_ok=True)
    REN = vtk.vtkRenderer()
    REN.SetBackground(0.97, 0.98, 1.0)
    REN.SetBackground2(0.83, 0.87, 0.93)
    REN.GradientBackgroundOn()
    RW = vtk.vtkRenderWindow()
    RW.SetOffScreenRendering(1)
    RW.AddRenderer(REN)
    RW.SetSize(W, H)
    RW.SetMultiSamples(8)
    light = vtk.vtkLight()
    light.SetLightTypeToHeadlight()
    light.SetIntensity(0.55)
    REN.AddLight(light)
    REN.LightFollowCameraOn()
    REN.AutomaticLightCreationOn()
    build()
    st = State()
    prev = None
    files = []
    n = len(STEPS)
    for idx, step in enumerate(STEPS):
        last_view = prev
        run = which == "all" or which == "test" or which == step["code"]
        nf = int(round(step["dur"] * FPS))
        frames = [nf - 1] if which == "test" else range(nf)
        # steps not rendered still have to change the state
        if not run:
            for k in (step["dur"] + 1.0,):
                apply(st, k, step, None)
            prev = step["view"]
            continue
        out = os.path.join(OUT, f"s{idx:02d}_{step['code']}.mp4")
        if which == "all" and os.path.exists(out):          # already rendered (restart after a stop)
            apply(st, step["dur"] + 1.0, step, None)
            st.nod = 0.0
            prev = step["view"]
            files.append(out)
            continue
        print_bed(any(op[0] == "print_bed" for op in step["ops"]))
        if any(op[0] == "print_bed" for op in step["ops"]) and PRINT_ACTORS:
            cs = np.array([c for _, _, c in PRINT_ACTORS])
            ctr = (cs.min(0) + cs.max(0)) / 2.0
            ext = float(np.max(cs.max(0) - cs.min(0)))
            step["view"] = cam((ctr[0], ctr[1] - 0.10 * ext, 0.0), -90, 62, 1.45 * ext + 300)
        out = os.path.join(OUT, f"s{idx:02d}_{step['code']}.mp4")
        proc = None
        if which != "test":
            tmp = out[:-4] + ".tmp.mp4"
            proc = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                     "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                                     "-pix_fmt", "yuv420p", tmp], stdin=subprocess.PIPE)
        for fi in frames:
            t = fi / FPS
            apply(st, t, step, None)
            orbit = next((op[1] for op in step["ops"] if op[0] == "orbit"), 0.0)
            set_cam(step["view"], last_view, t, step, extra_az=orbit * ease(t / step["dur"]))
            img = grab()
            hp = None
            if st.nod or st.head_dz:
                def hp(q, st=st):
                    q = np.array(q, float) - [0, 0, ZP]
                    a = math.radians(st.nod)
                    q = np.array([q[0] * math.cos(a) + q[2] * math.sin(a), q[1], -q[0] * math.sin(a) + q[2] * math.cos(a)])
                    return q + [0, 0, ZP + st.head_dz]
            if any(op[0] == "print_bed" for op in step["ops"]):
                d = ImageDraw.Draw(img, "RGBA")
                for a, lab, c in PRINT_ACTORS:
                    x, y = project(c)
                    tw = d.textlength(lab, font=font(20, True))
                    d.rounded_rectangle((x - tw / 2 - 6, y - 30, x + tw / 2 + 6, y - 4), 6, fill=(255, 255, 255, 230))
                    d.text((x - tw / 2, y - 29), lab, font=font(20, True), fill=(20, 30, 50, 255))
            img = overlay(img, step, t, idx, n, hp)
            if which == "test":
                img.save(os.path.join(OUT, f"test_{idx:02d}_{step['code']}.png"))
            else:
                proc.stdin.write(img.tobytes())
        if proc is not None:
            proc.stdin.close()
            if proc.wait() == 0:
                os.replace(out[:-4] + ".tmp.mp4", out)
                files.append(out)
            else:
                raise SystemExit(f"ffmpeg failed on {step['code']}")
        apply(st, step["dur"] + 1.0, step, None)
        st.nod = 0.0
        prev = step["view"]
        print(f"{step['code']:4s} {nf:4d} frames", flush=True)
    if which == "all":
        lst = os.path.join(OUT, "list.txt")
        with open(lst, "w") as f:
            for p in files:
                f.write(f"file '{os.path.basename(p)}'\n")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
                        os.path.join(OUT, "assembly.mp4")], check=True)
        print("wrote", os.path.join(OUT, "assembly.mp4"))


if __name__ == "__main__":
    render(sys.argv[1] if len(sys.argv) > 1 else "test")
