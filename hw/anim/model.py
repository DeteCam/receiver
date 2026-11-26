"""Geometry of the DeteCam receiver (design values, not measured).  mm, origin = ring centre, +x = room side, z up.
Each part is a function that adds triangles to a Mesh; PARTS lists them with the assembly step they belong to."""
import numpy as np
from r3d import Mesh, box, cyl

PCB = (0.10, 0.45, 0.27)
PCB2 = (0.08, 0.30, 0.50)
GOLD = (0.86, 0.69, 0.26)
BLACK = (0.16, 0.16, 0.18)
WHITE = (0.93, 0.93, 0.91)
SHELL = (0.80, 0.83, 0.87)
FILM = (0.95, 0.62, 0.12)
ALU = (0.76, 0.78, 0.81)
STEEL = (0.45, 0.47, 0.50)
BLUE = (0.20, 0.36, 0.62)
RED = (0.80, 0.20, 0.16)
CABLE = (0.22, 0.22, 0.25)
BUNDLE = (0.85, 0.30, 0.10)

Z_TOP = -88.9                 # box top below the ring plane
LAYOUT = "k2i"                # k2i: bundle inside the back rim of the ring; k2n: bundle just outside the ring
DX = 0.0 if LAYOUT == "k2i" else -11.25     # the box (and everything in it, and the base) sits this far behind the ring axis
BOX = (-67.5 + DX, 67.5 + DX, -78.75, 78.75, 56.25)       # x0, x1, y0, y1, height
Z_BOT = Z_TOP - BOX[4]
PIVOT = (DX, 0.0, Z_TOP - BOX[4] / 2.0)          # nod axis (along y) through the middle of the box
G = (-45.0, -11.25) if LAYOUT == "k2i" else (-67.5, -11.25)     # where the cable bundle goes down
PLUG_BACK = 7.5 if LAYOUT == "k2i" else 5.0
SP_DX = (G[0] - PLUG_BACK) - (-52.5 + DX)  # SP16T inside the box: its 8-port plug line sits this far behind the bundle
PLATE_R = 70.0 if LAYOUT == "k2i" else 66.0
YS = -11.25                   # spine of the in-plane wiring


def _sp_shifted(fn):
    """SP16T, its adapters and the attenuator on its common port: moved inside the box as well."""
    def wrap(*a, **k):
        m = fn(*a, **k)
        m.V = [v + np.array([DX + SP_DX, 0.0, 0.0]) for v in m.V]
        return m
    wrap.__name__ = fn.__name__
    wrap.__doc__ = fn.__doc__
    return wrap


def _shifted(fn):
    """Parts drawn in box coordinates (box centred on x = 0) are moved with the box."""
    def wrap(*a, **k):
        m = fn(*a, **k)
        m.V = [v + np.array([DX, 0.0, 0.0]) for v in m.V]
        return m
    wrap.__name__ = fn.__name__
    wrap.__doc__ = fn.__doc__
    return wrap


def ring_xy(k, r=60.0):
    a = 2.0 * np.pi * k / 16
    return r * np.cos(a), r * np.sin(a)


# ---------------------------------------------------------------- parts (each returns a Mesh)
def box_tray():
    m = Mesh()
    x0, x1, y0, y1, h = BOX
    t = 2.5
    box(m, (x0, y0, Z_BOT), (x1, y1, Z_BOT + t), SHELL)
    box(m, (x0, y0, Z_BOT), (x0 + t, y1, Z_TOP - 3), SHELL)          # back wall
    box(m, (x0, y1 - t, Z_BOT), (x1, y1, Z_TOP - 3), SHELL)          # far side wall
    return m


def box_front_walls():
    m = Mesh()
    x0, x1, y0, y1, h = BOX
    t = 2.5
    box(m, (x1 - t, y0, Z_BOT), (x1, y1, Z_TOP - 3), SHELL)
    box(m, (x0, y0, Z_BOT), (x1, y0 + t, Z_TOP - 3), SHELL)
    return m


def box_lid():
    m = Mesh()
    x0, x1, y0, y1, h = BOX
    box(m, (x0, y0, Z_TOP - 3), (x1, y1, Z_TOP), SHELL)
    return m


@_shifted
def carrier():
    """CM4 IO base (85 x 56) on the lower level."""
    m = Mesh()
    z = Z_BOT + 8
    box(m, (-22, -74, z), (63, -18, z + 1.6), PCB)
    box(m, (50, -70, z + 1.6), (63, -55, z + 14), STEEL)            # USB
    box(m, (50, -50, z + 1.6), (63, -35, z + 8), STEEL)             # HDMI
    box(m, (-20, -72, z + 1.6), (-8, -62, z + 6), BLACK)            # power / connector
    return m


@_shifted
def cm4():
    m = Mesh()
    z = Z_BOT + 8 + 1.6 + 3
    box(m, (-10, -66, z), (45, -26, z + 1.5), PCB)
    box(m, (2, -58, z + 1.5), (22, -40, z + 4), BLACK)               # SoC
    return m


@_shifted
def heatsink():
    m = Mesh()
    z = Z_BOT + 8 + 1.6 + 3 + 4
    box(m, (-10, -66, z), (45, -26, z + 2), ALU)
    for i in range(7):
        x = -8 + i * 8.2
        box(m, (x, -66, z + 2), (x + 3, -26, z + 9), ALU, edges=False)
    return m


@_shifted
def ax210():
    """AX210 (22 x 30) on the M.2 M-key -> A-key adapter, beside the carrier (drawn on top for visibility)."""
    m = Mesh()
    z = Z_BOT + 8
    box(m, (-62, -70, z), (-28, -30, z + 1.6), PCB2)                # adapter
    box(m, (-58, -64, z + 3), (-36, -34, z + 4.2), PCB)             # card
    box(m, (-56, -60, z + 4.2), (-40, -40, z + 6), ALU)             # shield can
    return m


@_sp_shifted
def sp16t():
    m = Mesh()
    z = Z_TOP - 22
    box(m, (-30, -50, z), (30, 50, z + 1.6), PCB)
    for cx, cy in ((-10, -25), (-10, 25), (15, 0)):                  # 1->8, 1->8, 1->2
        box(m, (cx - 3, cy - 3, z + 1.6), (cx + 3, cy + 3, z + 2.6), BLACK, edges=False)
    box(m, (0, -8, z + 1.6), (10, 8, z + 9), BLACK)                  # 2 x 8 header
    box(m, (18, 30, z + 1.6), (28, 40, z + 7), WHITE)                # XH 5 V
    for (p, d) in sp16t_ports() + [((30, 0), (1, 0))]:
        p0 = np.array([p[0], p[1], z + 0.8])
        cyl(m, p0 - 2 * np.array([d[0], d[1], 0]), p0 + 9.5 * np.array([d[0], d[1], 0]), 3.1, GOLD, n=10)
    return m


def sp16t_ports():
    """(position on the board edge, outward direction) of the 16 switched ports."""
    out = []
    for y in (-42, -30, -18, -6, 6, 18, 30, 42):
        out.append(((-30, y), (-1, 0)))
    for x in (-22, -9, 4, 17):
        out.append(((x, -50), (0, -1)))
        out.append(((x, 50), (0, 1)))
    return out


@_sp_shifted
def adapter(i):
    m = Mesh()
    z = Z_TOP - 22 + 0.8
    p, d = sp16t_ports()[i]
    d3 = np.array([d[0], d[1], 0.0])
    p0 = np.array([p[0], p[1], z]) + 9.5 * d3
    cyl(m, p0, p0 + 8 * d3, 4.0, GOLD, n=6)                          # nut
    cyl(m, p0 + 8 * d3, p0 + 13 * d3, 2.2, GOLD, n=10)               # U.FL post
    return m


def plug_point(i):
    z = Z_TOP - 22 + 0.8
    p, d = sp16t_ports()[i]
    return np.array([p[0] + 22.5 * d[0] + DX + SP_DX, p[1] + 22.5 * d[1], z])


@_sp_shifted
def attenuator1():
    m = Mesh()
    z = Z_TOP - 22 + 0.8
    cyl(m, (39.5, 0, z), (62, 0, z), 4.2, GOLD, n=12)
    cyl(m, (45, 0, z), (56, 0, z), 4.8, STEEL, n=12)
    return m


@_shifted
def wilkinsons():
    m = Mesh()
    z = Z_BOT + 6
    for y0 in (18, 48):
        box(m, (-60, y0, z), (-28, y0 + 24, z + 9), BLUE)
        cyl(m, (-28, y0 + 12, z + 4.5), (-18, y0 + 12, z + 4.5), 3.1, GOLD, n=10)
        cyl(m, (-60, y0 + 5, z + 4.5), (-64, y0 + 5, z + 4.5), 3.1, GOLD, n=10)
        cyl(m, (-60, y0 + 19, z + 4.5), (-64, y0 + 19, z + 4.5), 3.1, GOLD, n=10)
    return m


@_shifted
def attenuator2():
    m = Mesh()
    z = Z_BOT + 10.5
    cyl(m, (-18, 30, z), (4, 30, z), 4.2, GOLD, n=12)
    cyl(m, (-12, 30, z), (-2, 30, z), 4.8, STEEL, n=12)
    return m


@_shifted
def pico_buck():
    m = Mesh()
    z = Z_BOT + 6
    box(m, (10, 20, z), (61, 41, z + 1.6), PCB)                      # Pico 51 x 21
    box(m, (53, 26, z + 1.6), (61, 35, z + 4.5), STEEL, edges=False)
    box(m, (16, 26, z + 1.6), (24, 34, z + 2.6), BLACK, edges=False)
    box(m, (10, 50, z), (32, 67, z + 1.6), (0.12, 0.25, 0.60))       # buck 22 x 17
    box(m, (14, 53, z + 1.6), (26, 64, z + 8), BLACK, edges=False)
    box(m, (40, 50, z), (62, 62, z + 1.6), RED)                      # PD trigger
    return m


def column():
    m = Mesh()
    cyl(m, (0, 0, Z_TOP), (0, 0, -4), 9.0, WHITE, n=16, edges=True)
    return m


def cable_post():
    m = Mesh()
    for dy in (-6.0, 4.5):         # two rails, the bundle runs between them
        box(m, (G[0] - 5, G[1] + dy, Z_TOP), (G[0] + 5, G[1] + dy + 1.5, -4), WHITE)
    return m


def plate():
    m = Mesh()
    cyl(m, (0, 0, -4), (0, 0, -1), PLATE_R, WHITE, n=48, edges=True)
    return m


def fin(k):
    """k = 0..15 ring fins (radial), k = 16 centre fin (in the y-z plane)."""
    m = Mesh()
    if k == 16:
        box(m, (-0.9, -7, -20), (0.9, 7, 20), WHITE)
        return m
    a = 2.0 * np.pi * k / 16
    c, s = np.cos(a), np.sin(a)
    r0, r1, t = 52.0, PLATE_R - 3.0 if LAYOUT == "k2i" else 66.0, 0.9
    v = []
    for z in (-20.0, 20.0):
        for (r, tt) in ((r0, -t), (r1, -t), (r1, t), (r0, t)):
            v.append((r * c - tt * s, r * s + tt * c, z))
    m.add(v, [(0, 1, 2), (0, 2, 3), (4, 6, 5), (4, 7, 6), (0, 4, 5), (0, 5, 1), (1, 5, 6), (1, 6, 2), (2, 6, 7), (2, 7, 3), (3, 7, 4), (3, 4, 0)],
          WHITE, [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)])
    return m


def film(k):
    """Film antenna 9 x 35 mm on its fin, long side vertical (drawn from z = -17.5 to 17.5; the lower part sits in
    a slot of the plate)."""
    m = Mesh()
    for t in (1.3, -1.3):          # drawn on both faces of the fin so that it shows from every side
        if k == 16:
            v = [(t, -4.5, -17.5), (t, 4.5, -17.5), (t, 4.5, 17.5), (t, -4.5, 17.5)]
        else:
            a = 2.0 * np.pi * k / 16
            c, s = np.cos(a), np.sin(a)
            v = [((55.5) * c - t * s, 55.5 * s + t * c, -17.5), (64.5 * c - t * s, 64.5 * s + t * c, -17.5),
                 (64.5 * c - t * s, 64.5 * s + t * c, 17.5), (55.5 * c - t * s, 55.5 * s + t * c, 17.5)]
        m.add(v, [(0, 1, 2), (0, 2, 3)], FILM, [(0, 1), (1, 2), (2, 3), (3, 0)])
    return m


def _seg_dist0(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    t = float(np.clip(-np.dot(p, q - p) / np.dot(q - p, q - p), 0.0, 1.0))
    return float(np.linalg.norm(p + t * (q - p)))


def _spine_nodes():
    """Slanted ribs (45 deg towards the back), same rule as sim/mysim/rod_heads.py."""
    n = {}
    for k in range(16):
        x, y = ring_xy(k, 54.0)
        n[k] = float(np.clip(5.625 * round((x - abs(y - YS)) / 5.625), -45.0, 45.0))
        while _seg_dist0((x, y), (n[k], YS)) < 10.0 and n[k] > -45.0:
            n[k] -= 5.625
    return n


SPINE_NODE = _spine_nodes()


def rib(k):
    """Cable of ring element k inside the ring plane: film -> spine (polyline, z = 0)."""
    x, y = ring_xy(k, 54.0)
    return np.array([[x, y, 0.0], [SPINE_NODE[k], YS, 0.0]])


def spine():
    return np.array([[max(SPINE_NODE.values()), YS, 0.0], [G[0], YS, 0.0]])


def ref_cable():
    return np.array([[1.3, -5.0, 0.0], [0.0, YS, 0.0]])


def trunk():
    return np.array([[G[0], YS, 0.0], [G[0], YS, Z_TOP - 6.0]])


@_shifted
def base():
    m = Mesh()
    box(m, (-95, -95, -186), (95, 138, -180), (0.55, 0.57, 0.60))
    for y in (-87, 83):
        box(m, (-14, y, -180), (14, y + 4, PIVOT[2] + 12), WHITE)
    cyl(m, (0, -90, PIVOT[2]), (0, -79, PIVOT[2]), 4.0, STEEL, n=12)
    cyl(m, (0, 79, PIVOT[2]), (0, 92, PIVOT[2]), 4.0, STEEL, n=12)
    return m


@_shifted
def motor():
    m = Mesh()
    zp = PIVOT[2]
    box(m, (-20, 92, zp - 20), (20, 132, zp + 20), (0.62, 0.64, 0.67))           # worm gearbox
    box(m, (20, 91, zp - 21), (62, 133, zp + 21), (0.20, 0.20, 0.22))            # NEMA17
    box(m, (-60, 100, -180), (-25, 125, -178), PCB)                              # TMC2209 carrier (schematic)
    return m


@_shifted
def power_bank():
    m = Mesh()
    box(m, (-75, -190, -186), (75, -120, -161), BLUE)          # beside the base (not counted in the size)
    return m


def dims_marks():
    return None
