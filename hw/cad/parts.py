"""DeteCam receiver, printed parts (design values, not measured).  CadQuery 2.8.
World frame = hw/anim/model.py: mm, origin = ring centre, +x = room side (front), -x = wall side, z up,
the head nods about the y axis through PIVOT = (0, 0, -117.025).
Every part is built in the world frame at nod 0 deg; PRINT gives the turn that puts it flat on the bed.
Head geometry that the simulations use (ring, plate, fins, column r 9, rails, box 135 x 157.5 x 56.25 at
z -145.15..-88.9, pivot) is copied from model.py and must not change: check.py compares it.
Run:  python3 hw/cad/parts.py   (writes hw/cad/stl/*.stl)"""
import math
import os
import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- fixed head geometry (from model.py, do not change)
Z_TOP, BOX_H = -88.9, 56.25
Z_BOT = Z_TOP - BOX_H                       # -145.15
ZP = Z_TOP - BOX_H / 2.0                    # -117.025 pivot height
BX, BY = 67.5, 78.75                        # box half sizes
PLATE_R, PLATE_Z0, PLATE_Z1 = 70.0, -4.0, -1.0
FIN_R0, FIN_R1, FIN_T, FIN_Z = 52.0, 67.0, 1.8, 20.0
COL_R = 9.0
G = (-45.0, -11.25)                         # cable bundle goes down here
RAILS = ((-17.25, -15.75), (-6.75, -5.25))  # y ranges of the two rails, x = -50..-40

# ---------------------------------------------------------------- new exterior (plan sim/plan_20261127_exterior.md)
WALL_T, LID_T = 2.5, 3.0
Z_RIM = Z_TOP - LID_T                       # -91.9 top of the tray walls
FLOOR = Z_BOT + WALL_T                      # -142.65 inside floor of the box
# right drive: 5840 worm gearbox (seller drawing WL5840: 58 x 40 x 30.8 body, 35.8 over the bosses, D8 x 15 output
# 20.2 mm from the far end) over a NEMA17 (42 x 42 x 48); output shaft on the pivot axis
GB_PLANE = 94.5                             # y of the gearbox mounting plane (pads / shaft boss) = face of the wall
GB_TOP, GB_BOT = ZP + 20.2, ZP - 37.8       # -96.825, -154.825
MOTOR_BOT = GB_BOT - 48.0                   # -202.825
HUB_FACE = 88.5                             # outer face of the right hub (1 mm from the tower wall at 89.5)
TOWER_R_Y0, TOWER_R_Y1 = 89.5, 141.0        # right tower, wall side .. open side (closed by the cover)
COVER_Y1 = 143.5
TOWER_L_Y0, TOWER_L_Y1 = -91.0, -83.0       # left tower (8 mm), bearing for the hollow trunnion
TRUN_END = -93.5
# base
PL_X = 65.0
PL_Y0, PL_Y1 = -94.0, COVER_Y1
PL_Z0, DECK_Z0, DECK_Z1 = -206.0, -172.5, -170.0
PL_WALL, PL_FLOOR = 2.4, 2.5
BAY_X = 39.75                               # battery bay half width (inside)
BAY_Y1 = 82.0                               # bay ends at the motor cavity wall
MOTOR_CAV = (-23.0, 23.0, 84.5, TOWER_R_Y1)  # x0, x1, y0, y1 of the motor cavity under the gearbox
# wall bracket: wall plane 120 mm behind the pivot (the fins clear the wall by 10 mm at -20 deg)
WALL_X = -120.0
BOSS_PTS = ((-55.0, -82.0), (55.0, -82.0), (-55.0, 30.0), (55.0, 30.0), (-55.0, 132.0), (55.0, 132.0))
PEG_PTS = ((-50.0, 60.0), (50.0, 60.0))


# ---------------------------------------------------------------- helpers (all shapes in world coordinates)
def B(x0, x1, y0, y1, z0, z1, rz=0.0, rbot=0.0, rtop=0.0):
    w = cq.Workplane("XY").box(x1 - x0, y1 - y0, z1 - z0)
    if rz:
        w = w.edges("|Z").fillet(rz)
    if rbot:
        w = w.faces("<Z").edges().fillet(rbot)
    if rtop:
        w = w.faces(">Z").edges().fillet(rtop)
    return w.val().translate(cq.Vector((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))


def C(r, p0, p1):
    a = cq.Vector(*p0)
    d = cq.Vector(*p1) - a
    return cq.Solid.makeCylinder(r, d.Length, a, d.normalized())


def K(r0, r1, p0, p1):
    a = cq.Vector(*p0)
    d = cq.Vector(*p1) - a
    return cq.Solid.makeCone(r0, r1, d.Length, a, d.normalized())


def XZ(pts, y0, y1):
    """Prism of a polygon given in (x, z), from y0 to y1."""
    return cq.Workplane("XZ").polyline(pts).close().extrude(y1 - y0).val().translate(cq.Vector(0, y1, 0))


def YZ(pts, x0, x1):
    """Prism of a polygon given in (y, z), from x0 to x1."""
    return cq.Workplane("YZ").polyline(pts).close().extrude(x1 - x0).val().translate(cq.Vector(x0, 0, 0))


def U(a, *bs):
    return a.fuse(*bs).clean() if bs else a


def D(a, *bs):
    return a.cut(*bs) if bs else a


def hexnut(x, y, z, axis, depth, af=5.7):
    """Hexagonal pocket for an M3 nut (across flats af), axis 'x' | 'y' | 'z', starting at the point, depth along +axis."""
    r = af / math.sqrt(3.0)
    pts = [(r * math.cos(math.radians(30 + 60 * k)), r * math.sin(math.radians(30 + 60 * k))) for k in range(6)]
    if axis == "z":
        return cq.Workplane("XY").polyline(pts).close().extrude(depth).val().translate(cq.Vector(x, y, z))
    if axis == "y":
        s = cq.Workplane("XZ").polyline(pts).close().extrude(-depth).val()
        return s.translate(cq.Vector(x, y, z))
    s = cq.Workplane("YZ").polyline(pts).close().extrude(depth).val()
    return s.translate(cq.Vector(x, y, z))


def ring_xy(k, r):
    a = 2.0 * math.pi * k / 16
    return r * math.cos(a), r * math.sin(a)


def text(s, size, depth, at, plane="YZ", rot=0.0):
    """Raised text on a face: plane 'YZ' (front face, reading along +y) or 'XY'."""
    w = cq.Workplane(plane).text(s, size, depth, kind="bold", halign="center", valign="center")
    v = w.val()
    if rot:
        v = v.rotate(cq.Vector(0, 0, 0), cq.Vector(1, 0, 0), rot)
    return v.translate(cq.Vector(*at))


# ================================================================ head (nods)
def box_tray():
    """H1: the electronics box without its lid. Same outside as the simulated box."""
    outer = B(-BX, BX, -BY, BY, Z_BOT, Z_RIM, rz=6.0, rbot=3.0)
    inner = B(-BX + WALL_T, BX - WALL_T, -BY + WALL_T, BY - WALL_T, FLOOR, Z_RIM + 1.0, rz=3.5)
    s = D(outer, inner)
    add, cut = [], []
    # lid screw bosses in the four corners (M3 nylon screw from the lid, nut slid in from the side)
    for sx in (-1, 1):
        for sy in (-1, 1):
            cx, cy = sx * 60.0, sy * 71.25
            add.append(C(5.5, (cx, cy, Z_RIM - 15.0), (cx, cy, Z_RIM)))
            add.append(K(0.5, 5.5, (cx, cy, Z_RIM - 20.5), (cx, cy, Z_RIM - 15.0)))
            cut.append(C(1.7, (cx, cy, Z_RIM - 13.0), (cx, cy, Z_RIM + 0.1)))
            cut.append(B(cx - 6.5 if sx > 0 else cx - 2.85, cx + 2.85 if sx > 0 else cx + 6.5,
                         cy - 2.85, cy + 2.85, Z_RIM - 9.2, Z_RIM - 6.4))     # nut slot opens toward the inside
    # right wall: pad around the pivot, 4 x M3 for the D-shaft hub (nuts inside), on a 34 mm circle
    add.append(C(21.0, (0, BY - WALL_T - 3.0, ZP), (0, BY - WALL_T + 0.1, ZP)))
    for a in (45, 135, 225, 315):
        x, z = 17.0 * math.cos(math.radians(a)), ZP + 17.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, BY - 6.0, z), (x, BY + 0.1, z)))
        cut.append(hexnut(x, BY - WALL_T - 3.0 - 0.01, z, "y", 2.6))
    # left wall: 3 x M3 for the trunnion flange (countersunk from outside, nuts inside) and the wire hole
    cut.append(C(6.25, (0, -BY - 0.1, ZP), (0, -BY + WALL_T + 0.1, ZP)))
    for a in (90, 210, 330):
        x, z = 15.0 * math.cos(math.radians(a)), ZP + 15.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, -BY - 0.1, z), (x, -BY + WALL_T + 0.1, z)))
    # ledges for the SP16T bridge on both side walls (45 deg below, so they print without support)
    yl = -BY + WALL_T                                    # left wall inside face
    yr = BY - WALL_T                                     # right wall inside face (the pad stands 3 mm out of it)
    # (shelf 4 mm lower than first drawn: a dupont plug (2.5 + 14 mm) on the SP16T header must clear the lid)
    add.append(B(-33.0, 33.0, yl - 0.01, yl + 3.0, -118.9, -116.9))
    add.append(YZ([(yl - 0.01, -118.9), (yl + 3.0, -118.9), (yl - 0.01, -121.9)], -33.0, 33.0))
    add.append(B(-33.0, 33.0, yr - 6.0, yr + 0.01, -118.9, -116.9))
    add.append(YZ([(yr + 0.01, -118.9), (yr - 6.0, -118.9), (yr + 0.01, -124.9)], -33.0, 33.0))
    cut.append(B(-9.5, 9.5, yl - 0.1, yl + 3.1, -125.0, -116.8))           # keep the wire hole free
    # carrier (CM4-IO-BASE-A) standoffs, Pi 4 pattern 58 x 49, board 9 mm above the floor, M2.5 self-tapping
    for x in (-32.5, 25.5):
        for y in (-70.5, -21.5):
            add.append(C(3.0, (x, y, FLOOR - 0.1), (x, y, FLOOR + 9.0)))
            cut.append(C(1.05, (x, y, FLOOR + 2.0), (x, y, FLOOR + 9.1)))
    # Pico H (upside down, pins up) standoffs: holes 11.4 x 47 mm, M2 self-tapping
    for x in (42.8, 54.2):
        for y in (7.0, 54.0):
            add.append(C(2.4, (x, y, FLOOR - 0.1), (x, y, FLOOR + 4.0)))
            cut.append(C(0.8, (x, y, FLOOR + 1.0), (x, y, FLOOR + 4.1)))
    # TMC2209 cradle (module upside down, heatsink in the pocket, pins up)
    cx0, cx1, cy0, cy1 = 39.45, 60.55, 57.95, 74.05
    add.append(B(cx0 - 1.6, cx1 + 1.6, cy0 - 1.6, cy1 + 1.6, FLOOR - 0.1, FLOOR + 11.0))
    cut.append(B(cx0, cx1, cy0, cy1, FLOOR + 10.0, FLOOR + 11.1))                 # PCB seat
    cut.append(B(cx0 + 3.0, cx1 - 3.0, cy0 + 1.5, cy1 - 1.5, FLOOR + 0.8, FLOOR + 10.1))   # heatsink pocket
    cut.append(B(cx0 + 5.0, cx1 - 5.0, cy0 - 2.0, cy1 + 2.0, FLOOR + 2.0, FLOOR + 7.0))    # air slots
    # ADXL345 (GY-291) seat: a flat pad with a rim, taped (VHB) with its x axis along the box x
    add.append(B(4.0, 26.0, 48.0, 66.0, FLOOR - 0.1, FLOOR + 1.5))
    cut.append(B(5.0, 25.0, 49.0, 65.0, FLOOR + 0.8, FLOOR + 1.6))
    # vents: bottom under the CM4, back wall low
    for k in range(6):
        y = -62.0 + 6.0 * k
        cut.append(B(-24.0, 12.0, y - 1.25, y + 1.25, Z_BOT - 1, FLOOR + 0.1))
    for k in range(8):
        y = -52.5 + 15.0 * k
        cut.append(B(-BX - 1, -BX + WALL_T + 0.1, y - 1.5, y + 1.5, -140.0, -126.0))
    s = U(s, *add)
    s = D(s, *cut)
    return s


def box_lid():
    """H2: lid; carries the column, the two rails and the bundle hole."""
    s = B(-BX, BX, -BY, BY, Z_RIM, Z_TOP, rz=6.0, rtop=1.5)
    cut = [C(6.15, (0, 0, Z_RIM - 1), (0, 0, Z_TOP + 1))]                       # column tenon
    for y0, y1 in RAILS:
        cut.append(B(-50.15, -39.85, y0 - 0.15, y1 + 0.15, Z_RIM - 1, Z_TOP + 1))
    cut.append(B(-48.5, -41.5, -14.5, -8.0, Z_RIM - 1, Z_TOP + 1))              # cable bundle
    for sx in (-1, 1):
        for sy in (-1, 1):
            cut.append(C(1.7, (sx * 60.0, sy * 71.25, Z_RIM - 1), (sx * 60.0, sy * 71.25, Z_TOP + 1)))
    for a in (90, 210, 330):                                                     # column sleeve, 38 mm circle
        x, y = 19.0 * math.cos(math.radians(a)), 19.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, y, Z_RIM - 1), (x, y, Z_TOP + 1)))
    for y in (-24.0, 1.5):                                                       # rail buttress
        cut.append(C(1.7, (-45.0, y, Z_RIM - 1), (-45.0, y, Z_TOP + 1)))
    return D(s, *cut)


def sp16t_bridge():
    """H3: shelf for the SP16T Core Board (60 x 100) at z -114.9, resting on the wall ledges."""
    z0, z1 = -116.9, -114.9
    s = B(-33.0, 33.0, -BY + WALL_T + 0.3, BY - WALL_T - 3.3, z0, z1, rz=2.0)
    s = D(s, B(-24.0, 24.0, -42.0, 42.0, z0 - 1, z1 + 1, rz=4.0))                # air window
    s = D(s, B(-9.5, 9.5, -BY + WALL_T, -BY + WALL_T + 5.0, z0 - 1, z1 + 1))    # wires at the trunnion
    add, cut = [], []
    def span(a, b):
        return (min(a, b), max(a, b))
    for sx in (-1, 1):                                   # corner clips for the board (clear of the SMA ports)
        for sy in (-1, 1):
            xa, xb = span(sx * 30.3, sx * 32.3)
            ya, yb = span(sy * 46.5, sy * 52.3)
            add.append(B(xa, xb, ya, yb, z1 - 0.01, z1 + 4.2))
            xa, xb = span(sx * 26.0, sx * 32.3)
            ya, yb = span(sy * 50.3, sy * 52.3)
            add.append(B(xa, xb, ya, yb, z1 - 0.01, z1 + 4.2))
    for sx in (-1, 1):                                                           # optional M3 holes (board corners)
        for sy in (-1, 1):
            cut.append(C(1.7, (sx * 26.5, sy * 46.5, z0 - 1), (sx * 26.5, sy * 46.5, z1 + 1)))
    s = U(s, *add)
    return D(s, *cut)


def hub_right():
    """H4: D-shaft hub on the right wall; 4 x M3 x 12 into the nuts of the wall pad; M3 set screw on the flat."""
    s = U(C(20.0, (0, BY, ZP), (0, BY + 4.0, ZP)), C(13.0, (0, BY + 4.0, ZP), (0, HUB_FACE, ZP)))
    cut = []
    for a in (45, 135, 225, 315):
        x, z = 17.0 * math.cos(math.radians(a)), ZP + 17.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, BY - 0.1, z), (x, BY + 4.1, z)))
    # D hole: 8.2 mm round with the flat 3.1 mm above the axis (shaft D8 with 7 mm across the flat), 9.3 mm deep
    dh = D(C(4.1, (0, HUB_FACE + 0.1, ZP), (0, HUB_FACE - 9.3, ZP)),
           B(-5, 5, HUB_FACE - 9.4, HUB_FACE + 0.2, ZP + 3.1, ZP + 5))
    cut.append(dh)
    ys = (BY + 4.0 + HUB_FACE) / 2.0
    cut.append(C(1.3, (0, ys, ZP + 2.5), (0, ys, ZP + 13.1)))                    # M3 x 10 set screw, self-tapping (2.6 mm)
    return D(s, *cut)


def trunnion_left():
    """H5: hollow trunnion on the left wall: the 6 wires (9 V, GND, motor x 4) pass inside it."""
    s = U(C(19.0, (0, -BY, ZP), (0, -BY - 3.0, ZP)), C(9.0, (0, -BY - 3.0, ZP), (0, TRUN_END, ZP)))
    cut = [C(6.25, (0, -BY + 0.1, ZP), (0, TRUN_END - 0.1, ZP))]
    for a in (90, 210, 330):                                                     # M3 countersunk, nuts inside the box
        x, z = 15.0 * math.cos(math.radians(a)), ZP + 15.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, -BY + 0.1, z), (x, -BY - 3.1, z)))
        cut.append(K(1.7, 3.2, (x, -BY - 1.5, z), (x, -BY - 3.05, z)))
    # wire slot at the bottom, +-35 deg, inside the bearing (wires leave downward into the tower channel)
    r_out = 9.5
    pts = [(0.0, 0.0), (r_out * math.sin(math.radians(-35)), -r_out * math.cos(math.radians(-35))),
           (0.0, -r_out * 1.1), (r_out * math.sin(math.radians(35)), -r_out * math.cos(math.radians(35)))]
    pts = [(p[0], ZP + p[1]) for p in pts]
    cut.append(XZ(pts, -90.5, -84.5))
    # groove for the clip
    cut.append(D(C(9.5, (0, -91.3, ZP), (0, -93.0, ZP)), C(8.0, (0, -91.4, ZP), (0, -92.9, ZP))))
    return D(s, *cut)


def trunnion_clip():
    """H6: C clip that keeps the trunnion in the left tower."""
    s = D(C(13.0, (0, -91.35, ZP), (0, -92.95, ZP)), C(8.1, (0, -91.0, ZP), (0, -93.3, ZP)))
    return D(s, B(-6.75, 6.75, -94, -90, ZP - 14, ZP))                            # opening (snaps over the 16 mm groove)


def column():
    """H7: column r 9 (unchanged), tenons into the plate (r 8) and the lid (r 6), slot for the centre fin."""
    s = U(C(COL_R, (0, 0, Z_TOP), (0, 0, PLATE_Z0)),
          C(8.0, (0, 0, PLATE_Z0), (0, 0, PLATE_Z1)),
          C(6.0, (0, 0, Z_TOP - 8.0), (0, 0, Z_TOP)))
    cut = [B(-1.05, 1.05, -7.15, 7.15, -FIN_Z - 0.3, PLATE_Z1 + 0.1),            # lower half of the centre fin
           C(1.6, (-7, 0, Z_TOP - 5.5), (7, 0, Z_TOP - 5.5))]                    # cross pin below the lid
    return D(s, *cut)


def rails():
    """H8: two cable rails 10 x 1.5 (unchanged), through the lid and the plate."""
    return [B(-50.0, -40.0, y0, y1, Z_RIM, PLATE_Z1) for y0, y1 in RAILS]


def ring_plate():
    """H9: ring plate r 70, z -4..-1 (unchanged), slots for 16 fins + centre fin, rails, bundle."""
    s = C(PLATE_R, (0, 0, PLATE_Z0), (0, 0, PLATE_Z1))
    cut = [C(8.15, (0, 0, PLATE_Z0 - 1), (0, 0, PLATE_Z1 + 1)),
           B(-1.05, 1.05, -7.15, 7.15, PLATE_Z0 - 1, PLATE_Z1 + 1),
           B(-48.5, -41.5, -14.5, -8.0, PLATE_Z0 - 1, PLATE_Z1 + 1)]
    for y0, y1 in RAILS:
        cut.append(B(-50.15, -39.85, y0 - 0.15, y1 + 0.15, PLATE_Z0 - 1, PLATE_Z1 + 1))
    for k in range(16):
        slot = B(FIN_R0 - 0.15, FIN_R1 + 0.15, -1.05, 1.05, PLATE_Z0 - 1, PLATE_Z1 + 1)
        cut.append(slot.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 22.5 * k))
    return D(s, *cut)


def fin(k):
    """H10: fin k (0..15 radial, 16 = centre). Notches mark the film corners (film 9 x 35, centred on z = 0)."""
    if k == 16:
        s = B(-FIN_T / 2, FIN_T / 2, -7.0, 7.0, -FIN_Z, FIN_Z)
        cut = [B(-2, 2, y - 0.4, y + 0.4, FIN_Z - 0.8, FIN_Z + 1) for y in (-4.5, 4.5)]
        return D(s, *cut)
    s = B(FIN_R0, FIN_R1, -FIN_T / 2, FIN_T / 2, -FIN_Z, FIN_Z)
    cut = [B(r - 0.4, r + 0.4, -2, 2, FIN_Z - 0.8, FIN_Z + 1) for r in (55.5, 64.5)]
    cut += [B(FIN_R1 - 0.8, FIN_R1 + 1, -2, 2, z - 0.4, z + 0.4) for z in (-17.5, 17.5)]
    s = D(s, *cut)
    return s.rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), 22.5 * k)


def column_sleeve():
    """H11: removable stiffener over the foot of the column, z -88.9..-50 only (keeps out of the ring zone)."""
    s = U(K(18.0, 10.8, (0, 0, Z_TOP), (0, 0, -50.0)), C(23.0, (0, 0, Z_TOP), (0, 0, Z_TOP + 2.5)))
    cut = [C(COL_R + 0.2, (0, 0, Z_TOP - 1), (0, 0, -49))]
    for a in (90, 210, 330):
        x, y = 19.0 * math.cos(math.radians(a)), 19.0 * math.sin(math.radians(a))
        cut.append(C(1.7, (x, y, Z_TOP - 1), (x, y, Z_TOP + 3.5)))
        cut.append(C(3.2, (x, y, Z_TOP + 2.5), (x, y, Z_TOP + 12)))               # room for the screw heads
    return D(s, *cut)


def rail_buttress():
    """H12: removable stiffener at the foot of the rails, z -88.9..-50; the bundle stays free between the rails."""
    side = lambda y0, y1: XZ([(-54.0, Z_TOP), (-36.0, Z_TOP), (-40.0, -50.0), (-50.0, -50.0)], y0, y1)  # noqa: E731
    s = U(side(-20.25, -17.4), side(-5.1, -2.25),
          XZ([(-54.0, Z_TOP), (-50.2, Z_TOP), (-50.2, -50.0), (-51.0, -50.0)], -20.25, -2.25),
          B(-56.0, -34.0, -28.0, 5.5, Z_TOP, Z_TOP + 2.5, rz=3.0))
    cut = [B(-48.5, -41.5, -14.5, -8.0, Z_TOP - 1, Z_TOP + 4),
           B(-50.2, -39.8, -17.4, -5.1, Z_TOP - 1, -49.0)]                          # rails + bundle space
    for y in (-24.0, 1.5):
        cut.append(C(1.7, (-45.0, y, Z_TOP - 1), (-45.0, y, Z_TOP + 3.5)))
    return D(s, *cut)


def fin_gauge():
    """H14: height gauge, puts the top of each fin 21 mm above the plate (film centre = cable exit height)."""
    s = B(0, 12, 0, 10, 0, FIN_Z - PLATE_Z1)
    return D(s, B(5.0, 7.1, -1, 11, 3, FIN_Z - PLATE_Z1 + 1))


# ================================================================ base (does not nod)
def plinth_tray():
    """B1: base tray; battery bay under the head, front strip for wires, motor cavity at the right end."""
    s = B(-PL_X, PL_X, PL_Y0, PL_Y1, PL_Z0, DECK_Z0, rz=10.0, rbot=2.0)
    s = D(s, B(-PL_X + PL_WALL, PL_X - PL_WALL, PL_Y0 + PL_WALL, PL_Y1 - PL_WALL, PL_Z0 + PL_FLOOR, DECK_Z0 + 1, rz=7.6))
    zf = PL_Z0 + PL_FLOOR
    add, cut = [], []
    for sx in (-1, 1):                                                           # bay side walls
        x0 = BAY_X if sx > 0 else -BAY_X - 1.6
        add.append(B(x0, x0 + 1.6, PL_Y0 + PL_WALL, BAY_Y1, zf - 0.1, DECK_Z0))
        for y0, y1 in ((PL_Y0 + PL_WALL, -80.0), (72.0, BAY_Y1)):                 # cable notches at both ends
            cut.append(B(x0 - 0.1, x0 + 1.7, y0, y1, zf + 6.0, zf + 21.0))
    add.append(B(-BAY_X - 1.6, BAY_X + 1.6, BAY_Y1, BAY_Y1 + 2.0, zf - 0.1, DECK_Z0))    # bay end wall
    for (x, y) in BOSS_PTS:                                                      # deck screw bosses, nut 6 mm below
        add.append(C(4.5, (x, y, zf - 0.1), (x, y, DECK_Z0)))
        cut.append(C(1.7, (x, y, DECK_Z0 - 12.0), (x, y, DECK_Z0 + 0.1)))
        cut.append(B(x - 2.85 if x < 0 else x - 6.0, x + 6.0 if x < 0 else x + 2.85, y - 2.85, y + 2.85,
                     DECK_Z0 - 9.0, DECK_Z0 - 6.3))
    for (x, y) in PEG_PTS:                                                       # wall-bracket peg holes
        add.append(C(7.0, (x, y, zf - 0.1), (x, y, zf + 7.0)))
        cut.append(C(4.2, (x, y, PL_Z0 - 0.1), (x, y, PL_Z0 + 6.0)))
    for (x, y) in ((-50.0, -80.0), (50.0, -80.0), (-50.0, 130.0), (50.0, 130.0)):   # rubber feet seats
        cut.append(C(6.2, (x, y, PL_Z0 - 0.1), (x, y, PL_Z0 + 1.0)))
    cut.append(B(-38.2, 38.2, PL_Y0 - 1, PL_Y0 + PL_WALL + 0.1, zf, DECK_Z0 - 2.0))      # battery door opening
    cut.append(B(-25.2, 25.2, TOWER_R_Y1 - 0.1, PL_Y1 + 1, zf, DECK_Z0 + 0.1))   # end wall opening, closed by B3
    s = U(s, *add)
    # front face: name
    try:
        add2 = [text("DeteCam", 9.0, 0.6, (PL_X, 24.75, (PL_Z0 + DECK_Z0) / 2.0), plane="YZ")]
        s = U(s, *add2)
    except Exception:
        pass
    return D(s, *cut)


def deck_towers():
    """B2: deck (lid of the tray) with the right drive tower and the left trunnion tower."""
    deck = B(-PL_X, PL_X, PL_Y0, PL_Y1, DECK_Z0, DECK_Z1, rz=10.0, rtop=1.2)
    # right tower: gearbox pocket open to +y (closed by B3), motor cavity below, wall with the shaft hole
    zt, rr = GB_TOP + 2.8, 9.0                         # top of the tower, radius of its rounded top edges
    tr = U(B(-25.0, 25.0, TOWER_R_Y0, TOWER_R_Y1, DECK_Z1 - 0.1, zt - rr),
           B(-25.0 + rr, 25.0 - rr, TOWER_R_Y0, TOWER_R_Y1, zt - rr - 0.01, zt),
           C(rr, (-25.0 + rr, TOWER_R_Y0, zt - rr), (-25.0 + rr, TOWER_R_Y1, zt - rr)),
           C(rr, (25.0 - rr, TOWER_R_Y0, zt - rr), (25.0 - rr, TOWER_R_Y1, zt - rr)),
           B(-25.0, 25.0, 82.0, TOWER_R_Y0 + 1.0, DECK_Z1 - 0.1, GB_BOT - 0.3))       # lower part to the head side
    # left tower: block with a round top around the pivot, bearing for the trunnion, front fairing with the wire tunnel
    tl = U(B(-22.0, 22.0, TOWER_L_Y0, TOWER_L_Y1, DECK_Z1 - 0.1, ZP),          # lower half; the cap B5 closes the bearing
           XZ([(22.0, DECK_Z1 - 0.1), (60.0, DECK_Z1 - 0.1), (60.0, -160.0), (50.0, -148.0), (22.0, -140.0)],
              TOWER_L_Y0, TOWER_L_Y1))
    s = U(deck, tr, tl)
    cut = []
    # gearbox pocket (58 x 40 profile + 0.3), relief for the bosses on its mounting face, shaft hole
    cut.append(B(-20.25, 20.25, GB_PLANE, TOWER_R_Y1 + 1, GB_BOT - 0.3, GB_TOP + 0.8))   # roof is a 40 mm bridge: 0.8 mm for sag
    cut.append(B(-12.5, 12.5, GB_PLANE - 2.7, GB_PLANE + 0.1, ZP - 23.9, ZP + 6.3))
    cut.append(C(5.0, (0, TOWER_R_Y0 - 1, ZP), (0, GB_PLANE + 0.1, ZP)))           # output shaft passes the wall
    # motor cavity (through the deck into the tray)
    cut.append(B(MOTOR_CAV[0], MOTOR_CAV[1], MOTOR_CAV[2], TOWER_R_Y1 + 1, DECK_Z0 - 1, GB_BOT - 0.3 + 0.01))
    # cover screws into the tower walls (nuts from the inside of the pocket walls)
    for (x, z) in ((-22.6, -101.0), (22.6, -101.0), (-22.6, -150.0), (22.6, -150.0)):   # M3 self-tapping
        cut.append(C(1.4, (x, TOWER_R_Y1 - 10.0, z), (x, TOWER_R_Y1 + 0.1, z)))
    # left tower: lower half of the bearing, cap screws (nuts slid in from the front), wire channel into the front strip
    cut.append(C(9.2, (0, TOWER_L_Y0 - 1, ZP), (0, TOWER_L_Y1 + 1, ZP)))
    for x in (-16.0, 16.0):
        cut.append(C(1.7, (x, (TOWER_L_Y0 + TOWER_L_Y1) / 2, ZP - 14.0), (x, (TOWER_L_Y0 + TOWER_L_Y1) / 2, ZP + 0.1)))
        cut.append(B(x - 2.85, x + 2.85, (TOWER_L_Y0 + TOWER_L_Y1) / 2 - 2.85, TOWER_L_Y1 + 0.1, ZP - 9.5, ZP - 6.8))
    # the wire groove is open on the outer face (y = -91): wires and the motor plug are laid in, not threaded
    cut.append(B(-4.5, 4.5, TOWER_L_Y0 - 0.1, -84.0, -162.0, ZP - 5.0))
    cut.append(B(-4.5, 55.0, TOWER_L_Y0 - 0.1, -84.0, -162.0, -156.0))
    cut.append(B(45.0, 55.0, TOWER_L_Y0 - 0.1, -84.0, DECK_Z0 - 1, -156.0))
    cut.append(B(-25.2, 25.2, TOWER_R_Y1 - 0.01, PL_Y1 + 1, DECK_Z0 - 1, DECK_Z1 + 1))   # room for the cover B3
    # deck screws (countersunk)
    for (x, y) in BOSS_PTS:
        cut.append(C(1.7, (x, y, DECK_Z0 - 1), (x, y, DECK_Z1 + 1)))
        cut.append(K(1.7, 3.3, (x, y, DECK_Z1 - 1.6), (x, y, DECK_Z1 + 0.01)))
    return D(s, *cut)


def tower_cap():
    """B5: upper half of the left tower (pillow-block cap); 2 x M3 x 20 down into nuts in the lower half."""
    s = D(U(B(-22.0, 22.0, TOWER_L_Y0, TOWER_L_Y1, ZP, ZP + 8.0), C(22.0, (0, TOWER_L_Y0, ZP), (0, TOWER_L_Y1, ZP))),
          B(-30, 30, TOWER_L_Y0 - 1, TOWER_L_Y1 + 1, ZP - 30, ZP))
    cut = [C(9.2, (0, TOWER_L_Y0 - 1, ZP), (0, TOWER_L_Y1 + 1, ZP))]
    for x in (-16.0, 16.0):
        yc = (TOWER_L_Y0 + TOWER_L_Y1) / 2
        cut.append(C(1.7, (x, yc, ZP - 0.1), (x, yc, ZP + 22.0)))
        cut.append(C(3.0, (x, yc, ZP + 6.0), (x, yc, ZP + 22.0)))                  # screw head counterbore
    return D(s, *cut)


def tower_cover():
    """B3: cover on the open side of the right tower and the motor opening of the tray."""
    s = B(-25.0, 25.0, TOWER_R_Y1, COVER_Y1, PL_Z0 + PL_FLOOR + 0.2, GB_TOP + 2.8, rz=0.0)
    try:
        s = cq.Workplane(obj=s).edges("|Y").fillet(4.0).val()
    except Exception:
        pass
    cut = []
    for (x, z) in ((-22.6, -101.0), (22.6, -101.0), (-22.6, -150.0), (22.6, -150.0)):
        cut.append(C(1.7, (x, TOWER_R_Y1 - 0.1, z), (x, COVER_Y1 + 0.1, z)))
    for k in range(5):                                                           # vents
        z = -112.0 - 9.0 * k
        cut.append(B(-12.0, 12.0, TOWER_R_Y1 - 0.1, COVER_Y1 + 0.1, z - 1.5, z + 1.5))
    return D(s, *cut)


def battery_door():
    """B4: door at the left end of the tray (friction fit + two bumps, finger notch)."""
    s = U(B(-40.5, 40.5, PL_Y0 - 1.6, PL_Y0, PL_Z0 + 0.5, DECK_Z0 - 0.5, rz=0.0),
          B(-37.9, 37.9, PL_Y0, PL_Y0 + PL_WALL, PL_Z0 + PL_FLOOR + 0.2, DECK_Z0 - 2.3))
    add = [C(0.8, (x, PL_Y0 + PL_WALL - 1.2, DECK_Z0 - 2.3), (x, PL_Y0 + PL_WALL - 1.2, DECK_Z0 - 1.6)) for x in (-25, 25)]
    s = U(s, *add)
    return D(s, B(-12.0, 12.0, PL_Y0 - 2.0, PL_Y0 - 0.8, DECK_Z0 - 9.0, DECK_Z0 - 0.4))


# ================================================================ wall
def wall_bracket():
    """W1: wall bracket. Back on the wall (x = -120), shelf under the base, pegs into the base, holes for two lever
    suction cups (1/4"-20 or M6 stud), two keyholes for screws / a hook, flat back for removable adhesive strips."""
    yc = (PL_Y0 + PL_Y1) / 2.0
    y0, y1 = yc - 55.0, yc + 55.0
    z_shelf = PL_Z0 - 1.0                                   # base stands on its rubber feet (1 mm out)
    plate = B(WALL_X, WALL_X + 6.0, y0, y1, -262.0, -112.0)
    shelf = B(WALL_X, 70.0, y0, y1, z_shelf - 6.0, z_shelf)
    lip = XZ([(62.0, z_shelf - 0.01), (70.0, z_shelf - 0.01), (70.0, z_shelf + 8.0)], y0, y1)   # 45 deg: prints standing
    gus = [XZ([(WALL_X + 5.0, -262.0), (WALL_X + 5.0, z_shelf - 5.0), (10.0, z_shelf - 5.0)], y, y + 5.0)
           for y in (y0 + 6.0, y1 - 11.0)]
    pegs = [C(3.9, (x, y, z_shelf - 0.1), (x, y, z_shelf + 5.0)) for (x, y) in PEG_PTS]
    s = U(plate, shelf, lip, *gus, *pegs)
    cut = []
    for y in (yc - 30.0, yc + 30.0):                                             # keyholes
        cut.append(C(4.5, (WALL_X - 1, y, -128.0), (WALL_X + 7, y, -128.0)))
        cut.append(B(WALL_X - 1, WALL_X + 7, y - 2.25, y + 2.25, -128.0, -120.0))
        cut.append(C(2.25, (WALL_X - 1, y, -120.0), (WALL_X + 7, y, -120.0)))
    for z in (-140.0, -250.0):                                                   # suction-cup studs + nut seat
        cut.append(C(3.4, (WALL_X - 1, yc, z), (WALL_X + 7, yc, z)))
        cut.append(hexnut(WALL_X + 6.0 - 4.0, yc, z, "x", 4.1, af=11.4))
    cut.append(C(2.5, (WALL_X - 1, yc, -118.0 - 0.0), (WALL_X + 7, yc, -118.0)))  # lanyard
    return D(s, *cut)


# ================================================================ parts table
PARTS = {
    # name: (builder, group, print turn (axis, deg) applied before dropping onto the bed, count)
    "H1_box_tray": (box_tray, "head", None, 1),
    "H2_box_lid": (box_lid, "head", ("x", 180), 1),
    "H3_sp16t_bridge": (sp16t_bridge, "head", None, 1),
    "H4_hub_right": (hub_right, "head", ("x", 90), 1),
    "H5_trunnion_left": (trunnion_left, "head", ("x", -90), 1),
    "H6_trunnion_clip": (trunnion_clip, "head", ("x", 90), 1),
    "H7_column": (column, "head", ("x", 180), 1),
    "H9_ring_plate": (ring_plate, "head", None, 1),
    "H11_column_sleeve": (column_sleeve, "head", None, 1),
    "H12_rail_buttress": (rail_buttress, "head", None, 1),
    "H14_fin_gauge": (fin_gauge, "tool", None, 1),
    "B1_plinth_tray": (plinth_tray, "base", None, 1),
    "B2_deck_towers": (deck_towers, "base", None, 1),
    "B3_tower_cover": (tower_cover, "base", ("x", -90), 1),
    "B5_tower_cap": (tower_cap, "base", ("x", 90), 1),
    "B4_battery_door": (battery_door, "base", ("x", 90), 1),
    "W1_wall_bracket": (wall_bracket, "wall", ("y", -90), 1),
}


def flat_for_print(shape, turn):
    if turn:
        ax = {"x": cq.Vector(1, 0, 0), "y": cq.Vector(0, 1, 0), "z": cq.Vector(0, 0, 1)}[turn[0]]
        shape = shape.rotate(cq.Vector(0, 0, 0), ax, turn[1])
    bb = shape.BoundingBox()
    return shape.translate(cq.Vector(-(bb.xmin + bb.xmax) / 2, -(bb.ymin + bb.ymax) / 2, -bb.zmin))


def all_world():
    """Every printed part in the world frame at nod 0: {name: shape} (fins and rails expanded)."""
    out = {n: f() for n, (f, g, t, c) in PARTS.items()}
    for i, r in enumerate(rails()):
        out[f"H8_rail_{i + 1}"] = r
    for k in range(17):
        out[f"H10_fin_{k:02d}"] = fin(k)
    return out


def export(out_dir=os.path.join(HERE, "stl")):
    os.makedirs(out_dir, exist_ok=True)
    info = {}
    for name, (f, g, turn, cnt) in PARTS.items():
        s = flat_for_print(f(), turn)
        p = os.path.join(out_dir, name + ".stl")
        cq.exporters.export(cq.Workplane(obj=s), p, tolerance=0.02, angularTolerance=0.15)
        bb = s.BoundingBox()
        info[name] = (round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1), round(s.Volume() / 1000.0, 1))
    # rails: both on one plate (flat), fins: 17 on one plate (flat, film side on the bed)
    r1, r2 = rails()
    rr = [flat_for_print(r, ("x", 90)) for r in (r1, r2)]
    rr = cq.Compound.makeCompound([rr[0].translate(cq.Vector(0, -8, 0)), rr[1].translate(cq.Vector(0, 8, 0))])
    cq.exporters.export(cq.Workplane(obj=rr), os.path.join(out_dir, "H8_rails_x2.stl"), tolerance=0.02, angularTolerance=0.15)
    info["H8_rails_x2"] = tuple(round(v, 1) for v in (rr.BoundingBox().xlen, rr.BoundingBox().ylen, rr.BoundingBox().zlen)) + (round(rr.Volume() / 1000, 1),)
    fins = []
    for k in range(17):
        s = fin(k).rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), -22.5 * k if k < 16 else 0.0)
        s = flat_for_print(s, ("x", 90) if k < 16 else ("y", 90))
        row, col = divmod(k, 6)
        fins.append(s.translate(cq.Vector(-55 + 22.0 * col, -50 + 48.0 * row, 0)))
    fc = cq.Compound.makeCompound(fins)
    cq.exporters.export(cq.Workplane(obj=fc), os.path.join(out_dir, "H10_fins_x17.stl"), tolerance=0.02, angularTolerance=0.15)
    info["H10_fins_x17"] = tuple(round(v, 1) for v in (fc.BoundingBox().xlen, fc.BoundingBox().ylen, fc.BoundingBox().zlen)) + (round(fc.Volume() / 1000, 1),)
    return info


if __name__ == "__main__":
    for k, v in export().items():
        print(f"{k:22s} {v[0]:6.1f} x {v[1]:6.1f} x {v[2]:6.1f} mm   {v[3]:7.1f} cm3")
