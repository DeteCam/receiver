"""Metal of the receiver base next to the K2i head (plan sim/plan_20261127_exterior.md).  [simulation]
The head (ring, cables, real-size box) nods about the pivot P = (0, 0, -117.025 mm) while the base does not.
ring_pynec.simulate_ring turns every wire about the origin by the nod angle, so a part of the base is given to it
already turned back about the pivot:  q_in = P + R(-t) (q - P).  The code then turns it by +t; the result differs from
the world position only by one translation of the whole structure, which leaves every pattern ratio unchanged.
  E0   old drawing (hw/anim/model.py before 11-27): worm gearbox + NEMA17 beside the box at pivot height,
       power bank beside the base plate.  Blocks only.
  E1   new exterior: worm gearbox + NEMA17 (motor axis vertical, under the gearbox) in the right tower, power bank in
       the plinth under the head.  Blocks only.
  E1c  E1 + the wires of the base: 6-core cable from the box side through the left trunnion down to the plinth and to
       the power bank, and the 4 motor wires along the front of the plinth to the motor (all bonded: worst case).
Blocks are wire-grid boxes (mesh 11.25 mm in both bands; at 5 GHz this is coarser than lambda/10, noted in the plan).
Same NEC model as rod_heads.py otherwise.  Run:
  python3 mysim/ext_heads.py mysim/nec_ext.npy d120_16bk2ie1c,d120_16bk2ie1c_ty+20,d120_16bk2ie1c_ty-20 2g4"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rod_heads as RH  # noqa: E402
import ring_pynec as R  # noqa: E402

ZP = -117.025                       # pivot height (middle of the 56.25 mm box whose top is 88.9 mm below the ring)
MESH_EXT = 11.25

# static metal, world coordinates of the head frame at 0 deg (mm): (x0, x1, y0, y1, z0, z1)
BLOCKS = {
    "e0": {"motor": (-20.0, 62.0, 91.0, 133.0, -138.0, -96.0),         # gearbox + NEMA17 of the old drawing
           "bank": (-75.0, 75.0, -190.0, -120.0, -186.0, -161.0)},      # power bank beside the base plate
    "e1": {"motor": (-21.0, 21.0, 92.0, 133.5, -202.8, -96.8),          # 5840 gearbox (58 x 40 x 35.8) over NEMA17 (42 x 42 x 48)
           "bank": (-37.5, 37.5, -92.0, 58.0, -203.0, -176.0)},         # 150 x 75 x 27 power bank in the plinth
}
# POST-HOC (after the 5 GHz results of K2iE1c / K2iE1 / K2iE0): one block at a time
BLOCKS["e1m"] = {"motor": BLOCKS["e1"]["motor"]}
BLOCKS["e1b"] = {"bank": BLOCKS["e1"]["bank"]}
# POST-HOC remedy of plan section 2 D: power bank 10 mm lower, gearbox + motor 20 mm further out along the axis
BLOCKS["e1r"] = {"motor": (-21.0, 21.0, 112.0, 153.5, -202.8, -96.8), "bank": (-37.5, 37.5, -92.0, 58.0, -213.0, -186.0)}
TILT = 0.0          # nod angle of the pattern being built (deg), set by build()


def _static(q):
    """World point of the base -> input point that simulate_ring's nod rotation brings back to the same place."""
    t = np.radians(TILT)
    ct, st = np.cos(t), np.sin(t)
    x, y, z = float(q[0]), float(q[1]), float(q[2]) - ZP
    return (ct * x - st * z, y, st * x + ct * z + ZP)


def _axis(a, b, mesh):
    n = max(1, int(round((b - a) / mesh)))
    return np.linspace(a, b, n + 1)


def block_nodes(blk, mesh=MESH_EXT):
    x0, x1, y0, y1, z0, z1 = blk
    return _axis(x0, x1, mesh), _axis(y0, y1, mesh), _axis(z0, z1, mesh)


def block_wires(blk, mesh=MESH_EXT):
    """Closed wire-grid box, node to node (every crossing is a junction); world coordinates."""
    xs, ys, zs = block_nodes(blk, mesh)
    rad = mesh / (2.0 * np.pi)
    w = []
    for z in (zs[0], zs[-1]):                       # bottom and top faces
        for y in ys:
            for i in range(len(xs) - 1):
                w.append(((xs[i], y, z), (xs[i + 1], y, z), rad))
        for x in xs:
            for j in range(len(ys) - 1):
                w.append(((x, ys[j], z), (x, ys[j + 1], z), rad))
    rim = [(x, ys[0]) for x in xs] + [(x, ys[-1]) for x in xs] + [(xs[0], y) for y in ys[1:-1]] + [(xs[-1], y) for y in ys[1:-1]]
    for (x, y) in rim:                              # vertical edges of the side faces
        for k in range(len(zs) - 1):
            w.append(((x, y, zs[k]), (x, y, zs[k + 1]), rad))
    for z in zs[1:-1]:                              # horizontal rings of the side faces
        for i in range(len(xs) - 1):
            w.append(((xs[i], ys[0], z), (xs[i + 1], ys[0], z), rad))
            w.append(((xs[i], ys[-1], z), (xs[i + 1], ys[-1], z), rad))
        for j in range(len(ys) - 1):
            w.append(((xs[0], ys[j], z), (xs[0], ys[j + 1], z), rad))
            w.append(((xs[-1], ys[j], z), (xs[-1], ys[j + 1], z), rad))
    return w


def nearest_top_node(blk, x, y, mesh=MESH_EXT):
    xs, ys, zs = block_nodes(blk, mesh)
    return (float(xs[np.argmin(np.abs(xs - x))]), float(ys[np.argmin(np.abs(ys - y))]), float(zs[-1]))


def nearest_side_node(blk, side, y, z, mesh=MESH_EXT):
    """Node on the x = x0 ('back') or x = x1 ('front') face of a block."""
    xs, ys, zs = block_nodes(blk, mesh)
    x = float(xs[-1] if side == "front" else xs[0])
    return (x, float(ys[np.argmin(np.abs(ys - y))]), float(zs[np.argmin(np.abs(zs - z))]))


def box_side_node():
    """Grid node on the left side of the head box nearest to the trunnion (x = 0, z = pivot height), head frame."""
    lay = RH.LAY
    if lay.get("box"):                               # bk2i: 135 x 157.5 x 56.25 box, 11.25 mm grid
        bx0, bx1, bhy, bh = lay["box"]
        z_top, z_bot, hy, mesh = lay["z_top"], lay["z_top"] - bh, bhy, (11.25 if RH.SEG > 4.0 else 5.625)
    else:                                            # default 90 x 90 x 35 box of the small-box model
        mesh = 11.25 if RH.SEG > 4.0 else 5.625
        z_top, z_bot, hy = lay["z_top"], lay["z_top"] - 35.0, 45.0
    zs = _axis(z_bot, z_top, mesh)
    return (0.0, -hy, float(zs[np.argmin(np.abs(zs - ZP))]))


def cable_paths(ext, side="front"):
    """E1c wires: list of (polyline in world coordinates, radius, head_end) - head_end: first point is on the box.
    side = 'front': the motor wires run along the room side of the plinth (x = +50, the drawing);
    side = 'back' (POST-HOC remedy): along the wall side (x = -50) to the back face of the motor block."""
    bank, motor = BLOCKS[ext]["bank"], BLOCKS[ext]["motor"]
    xs, ys, zs = block_nodes(bank)
    p_bank = (float(xs[np.argmin(np.abs(xs))]), float(ys[0]), float(zs[-1]))    # top edge of the end face (-y)
    c = (0.0, p_bank[1] - 6.0, p_bank[2])            # bottom of the left tower, inside the plinth, at bank-top height
    # the last piece runs along the bank's top-face grid line and ends on its edge node (no wire passes over a wire)
    w1 = [box_side_node(), (0.0, -92.0, ZP), (0.0, c[1], ZP), c, (p_bank[0], c[1], c[2]), p_bank]
    p_mot = nearest_side_node(motor, side, 100.0, -190.0)
    xw = 50.0 if side == "front" else -50.0
    w2 = [c, (0.0, c[1], p_mot[2]), (xw, c[1], p_mot[2]), (xw, p_mot[1], p_mot[2]), p_mot]   # leaves c straight down
    return [(w1, 1.5, True), (w2, 1.0, False)]


def ext_wires(ext, cables):
    out = []
    for name, blk in BLOCKS[ext].items():
        for p, q, rad in block_wires(blk):
            out.append((1, _static(p), _static(q), rad))
    if cables:
        for pts, rad, head_end in cable_paths(ext, "back" if cables == "wall" else "front"):
            for i in range(len(pts) - 1):
                p, q = pts[i], pts[i + 1]
                if np.linalg.norm(np.subtract(p, q)) < 1e-9:
                    continue
                pp = p if (head_end and i == 0) else _static(p)          # the first point of W1 is on the box
                qq = _static(q)
                seg = max(RH.SEG, 2.5 * rad)                              # NEC thin-wire rule: segment >= 2.5 radii
                n = max(1, int(np.floor(float(np.linalg.norm(np.subtract(qq, pp))) / seg)))
                out.append((n, tuple(map(float, pp)), tuple(map(float, qq)), rad))
    return out


_rod_layout = RH._with_layout


def _with_layout(z_top_mm, half_w=45.0, height=35.0, mesh_mm=11.25, sides=True, rad_mm=None):
    w = _rod_layout(z_top_mm, half_w=half_w, height=height, mesh_mm=mesh_mm, sides=sides, rad_mm=rad_mm)
    if RH.LAY is not None and RH.LAY.get("ext"):
        w = list(w) + ext_wires(RH.LAY["ext"], RH.LAY.get("ext_cables"))
    return w


R._box_wires = _with_layout

_BK2I = dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0, box=(-67.5, 67.5, 78.75, 56.25))
_K2I = dict(kind="back", z_top=-88.9, ref_dz=0.0, thin_fix=True, chevron=1.0)
NEW = {
    "bk2ie0": dict(_BK2I, ext="e0"),
    "bk2ie1": dict(_BK2I, ext="e1"),
    "bk2ie1c": dict(_BK2I, ext="e1", ext_cables=True),
    "k2ie1c": dict(_K2I, ext="e1", ext_cables=True),      # 5 GHz check on the small-box model used at 5 GHz (QK2is)
    # POST-HOC (after the 5 GHz K2iE1c result): which part of the base does it - blocks only, old drawing's blocks
    "k2ie1": dict(_K2I, ext="e1"),
    "k2ie0": dict(_K2I, ext="e0"),
    # POST-HOC: motor block only, bank block only, and the motor wires moved to the wall side of the plinth
    "k2ie1m": dict(_K2I, ext="e1m"),
    "k2ie1b": dict(_K2I, ext="e1b"),
    "k2ie1cw": dict(_K2I, ext="e1", ext_cables="wall"),
    "k2ie1cr": dict(_K2I, ext="e1r", ext_cables=True),
}
RH.LAYOUTS.update(NEW)
for _name in NEW:
    for _t in (0, 20, -20):
        RH.SPECS[f"d120_16{_name}" + ("" if _t == 0 else f"_ty{_t:+d}")] = (_name, float(_t))


def build(keys, out_path, band="2g4"):
    global TILT
    out = {}
    for key in keys:
        TILT = RH.SPECS[key][1]
        out = RH.build([key], out_path, band)
    TILT = 0.0
    return out


if __name__ == "__main__":
    out = sys.argv[1]
    keys = sys.argv[2].split(",")
    band = sys.argv[3] if len(sys.argv) > 3 else "2g4"
    build(keys, out, band)
    print("EXT_NEC_DONE", keys, band, flush=True)
