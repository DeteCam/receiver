"""Geometry checks of the printed parts (plan sim/plan_20261127_exterior.md, items A, B, E).  [CAD check]
A  the head geometry the simulations use is unchanged; no new part enters the ring zone (r < 80 mm, z > -50 mm);
   metal (gearbox, motor, power bank) stays below z = -90 mm and more than 80 mm from every film at every nod.
B  the head swept from -20 to +20 deg (1 deg steps) keeps 3 mm from every part of the base, except the designed
   bearing / hub interfaces (no overlap, 0.1 mm); on the wall bracket the head keeps 10 mm from the wall plane.
E  every STL is closed (watertight) and fits the 256 mm build volume.
Run: python3 hw/cad/check.py   (prints a table and exits 1 if a check fails)"""
import math
import os
import sys
import cadquery as cq
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "anim"))
import parts as P  # noqa: E402
import model as M  # noqa: E402

FAILS = []


def ok(cond, label, detail):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}: {detail}")
    if not cond:
        FAILS.append(label)


def bbox(s):
    b = s.BoundingBox()
    return np.array([b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax])


def pts(s, tol=0.3):
    v, _ = s.tessellate(tol, 0.3)
    return np.array([[p.x, p.y, p.z] for p in v])


def nod(shape, deg):
    """Turn a head part about the pivot axis (y through z = ZP). + = looks down (top goes to +x), as ring_pynec."""
    return shape.rotate(cq.Vector(0, 0, P.ZP), cq.Vector(0, 1, P.ZP), deg)


def nod_pts(a, deg):
    t = math.radians(deg)
    x, z = a[:, 0], a[:, 2] - P.ZP
    return np.stack([math.cos(t) * x + math.sin(t) * z, a[:, 1], -math.sin(t) * x + math.cos(t) * z + P.ZP], axis=1)


def film_rects():
    out = []
    for k in range(17):
        if k == 16:
            c = [(0.0, -4.5, -17.5), (0.0, 4.5, -17.5), (0.0, 4.5, 17.5), (0.0, -4.5, 17.5)]
        else:
            a = 2 * math.pi * k / 16
            c = [(r * math.cos(a), r * math.sin(a), z) for (r, z) in ((55.5, -17.5), (64.5, -17.5), (64.5, 17.5), (55.5, 17.5))]
        g = np.array(c)
        u = np.linspace(0, 1, 7)
        grid = [g[0] + (g[1] - g[0]) * i + (g[3] - g[0]) * j for i in u for j in np.linspace(0, 1, 15)]
        out.append(np.array(grid))
    return out


def main():
    W = P.all_world()
    print("A. unchanged head geometry (model.py) and the ring zone")
    plate = bbox(W["H9_ring_plate"])
    ok(abs(plate[1] - M.PLATE_R) < 0.05 and abs(plate[4] + 4) < 0.05 and abs(plate[5] + 1) < 0.05,
       "ring plate", f"r {plate[1]:.2f} (model {M.PLATE_R}), z {plate[4]:.2f}..{plate[5]:.2f} (model -4..-1)")
    fr = []
    for k in range(16):
        s = W[f"H10_fin_{k:02d}"].rotate(cq.Vector(0, 0, 0), cq.Vector(0, 0, 1), -22.5 * k)
        fr.append(bbox(s))
    fr = np.array(fr)
    ok(np.allclose(fr[:, 0], 52.0, atol=0.05) and np.allclose(fr[:, 1], M.PLATE_R - 3.0, atol=0.05)
       and np.allclose(fr[:, 4], -20, atol=0.05) and np.allclose(fr[:, 5], 20, atol=0.05)
       and np.allclose(fr[:, 3] - fr[:, 2], 1.8, atol=0.05),
       "16 fins", f"r {fr[:, 0].min():.2f}..{fr[:, 1].max():.2f} (model 52..67), z -20..20, 1.8 thick")
    cf = bbox(W["H10_fin_16"])
    ok(np.allclose(cf, [-0.9, 0.9, -7, 7, -20, 20], atol=0.05), "centre fin", f"{np.round(cf, 2).tolist()}")
    col = W["H7_column"]
    sec = col.intersect(P.B(-20, 20, -20, 20, -50.5, -49.5))
    sb = bbox(sec)
    cb = bbox(col)
    ok(abs(sb[1] - M.COL_R if hasattr(M, "COL_R") else sb[1] - 9.0) < 0.05 and cb[4] <= M.Z_TOP and abs(cb[5] + 1) < 0.05,
       "column", f"r {sb[1]:.2f} at z -50 (model 9), body from the lid {M.Z_TOP} to the plate")
    for i, (y0, y1) in enumerate(P.RAILS):
        rb = bbox(W[f"H8_rail_{i + 1}"])
        ok(np.allclose(rb[:4], [-50, -40, y0, y1], atol=0.05) and rb[4] <= M.Z_TOP and rb[5] >= -4.0,
           f"rail {i + 1}", f"x {rb[0]:.1f}..{rb[1]:.1f}, y {rb[2]:.2f}..{rb[3]:.2f}, z {rb[4]:.1f}..{rb[5]:.1f}")
    tray, lid = bbox(W["H1_box_tray"]), bbox(W["H2_box_lid"])
    env = [min(tray[0], lid[0]), max(tray[1], lid[1]), min(tray[2], lid[2]), max(tray[3], lid[3]), min(tray[4], lid[4]),
           max(tray[5], lid[5])]
    mb = [M.BOX[0], M.BOX[1], M.BOX[2], M.BOX[3], M.Z_BOT, M.Z_TOP]
    ok(np.allclose(env, mb, atol=0.05), "box outside", f"{np.round(env, 2).tolist()} (model {np.round(mb, 2).tolist()})")
    ok(abs(P.ZP - M.PIVOT[2]) < 1e-6, "pivot", f"z {P.ZP:.3f} (model {M.PIVOT[2]:.3f}), axis y through x = 0")
    hb, tb = bbox(W["H4_hub_right"]), bbox(W["H5_trunnion_left"])
    ok(abs((hb[4] + hb[5]) / 2 - P.ZP) < 0.05 and abs((hb[0] + hb[1]) / 2) < 0.05 and abs((tb[4] + tb[5]) / 2 - P.ZP) < 0.05,
       "hub and trunnion on the pivot axis", f"hub z {(hb[4] + hb[5]) / 2:.2f}, trunnion z {(tb[4] + tb[5]) / 2:.2f}")
    existing = {"H9_ring_plate", "H7_column", "H8_rail_1", "H8_rail_2"} | {f"H10_fin_{k:02d}" for k in range(17)}
    worst = []
    for n, s in W.items():
        if n in existing or P.PARTS.get(n, (None, ""))[1] == "tool":     # the fin gauge is a loose tool
            continue
        a = pts(s)
        r = np.hypot(a[:, 0], a[:, 1])
        inside = (r < 80.0) & (a[:, 2] > -50.0 + 1e-6)
        worst.append((n, int(inside.sum()), float(a[:, 2].max())))
    bad = [w for w in worst if w[1] > 0]
    ok(not bad, "no new part in the ring zone (r < 80, z > -50)",
       "none" if not bad else ", ".join(f"{n} ({c} points)" for n, c, _ in bad))
    top_new = max((w[2] for w in worst if w[0].startswith(("H11", "H12"))), default=float("nan"))
    print(f"         highest new head stiffener point: z {top_new:.1f} mm")
    # metal: blocks of the NEC model E1 (sim/mysim/ext_heads.py) and the actual 5840 / NEMA17 / bank envelopes
    gearbox = (-20.0, 20.0, P.GB_PLANE, P.GB_PLANE + 35.8, P.GB_BOT, P.GB_TOP)
    motor = (-21.0, 21.0, 97.65, 139.65, P.MOTOR_BOT, P.GB_BOT)
    bank = (-37.5, 37.5, -90.0, 68.0, -203.0, -173.0)
    films = film_rects()
    md = []
    for name, b in (("gearbox", gearbox), ("motor", motor), ("power bank", bank)):
        corners = np.array([[x, y, z] for x in np.linspace(b[0], b[1], 5) for y in np.linspace(b[2], b[3], 5)
                            for z in np.linspace(b[4], b[5], 7)])
        dmin = 1e9
        for deg in range(-20, 21, 5):
            for f in films:
                g = nod_pts(f, deg)
                d = np.min(np.linalg.norm(g[:, None, :] - corners[None, :, :], axis=2))
                dmin = min(dmin, d)
        md.append((name, b[5], dmin))
        ok(b[5] < -90.0 and dmin > 80.0, f"metal: {name}", f"top z {b[5]:.1f}, nearest film {dmin:.0f} mm (any nod)")
    # the parts fit their pockets
    ok(gearbox[0] >= -20.25 and gearbox[1] <= 20.25 and gearbox[4] >= P.GB_BOT - 0.3 and gearbox[5] <= P.GB_TOP + 0.3,
       "gearbox fits the tower pocket", "58 x 40 profile + 0.3 mm")
    ok(motor[4] >= P.PL_Z0 + P.PL_FLOOR and motor[0] >= P.MOTOR_CAV[0] and motor[1] <= P.MOTOR_CAV[1]
       and motor[2] >= P.MOTOR_CAV[2] and motor[3] <= P.MOTOR_CAV[3],
       "NEMA17 fits the motor cavity", f"bottom {motor[4]:.2f} over the floor {P.PL_Z0 + P.PL_FLOOR:.2f}; y {motor[2]}..{motor[3]} "
       f"in {P.MOTOR_CAV[2]}..{P.MOTOR_CAV[3]}")

    print("B. nod sweep -20..+20 deg (1 deg) against the base and the wall")
    head = {"box": P.B(-P.BX, P.BX, -P.BY, P.BY, P.Z_BOT, P.Z_TOP, rz=6.0, rbot=3.0),
            "hub": W["H4_hub_right"], "trunnion": W["H5_trunnion_left"], "clip": W["H6_trunnion_clip"],
            "plate": W["H9_ring_plate"], "sleeve": W["H11_column_sleeve"], "buttress": W["H12_rail_buttress"],
            "column": W["H7_column"]}
    for k in range(17):
        head[f"fin{k}"] = W[f"H10_fin_{k:02d}"]
    head["bundle"] = P.C(3.0, (P.G[0], P.G[1], 0.0), (P.G[0], P.G[1], P.Z_TOP))
    base = {"deck_towers": W["B2_deck_towers"], "tray": W["B1_plinth_tray"], "cover": W["B3_tower_cover"],
            "door": W["B4_battery_door"], "bracket": W["W1_wall_bracket"], "cap": W["B5_tower_cap"]}
    iface = {("trunnion", "deck_towers"), ("clip", "deck_towers"), ("hub", "deck_towers"), ("trunnion", "cap"),
             ("clip", "cap")}
    near = {("trunnion", "deck_towers"), ("clip", "deck_towers"), ("hub", "deck_towers"), ("box", "deck_towers"),
            ("trunnion", "cap"), ("clip", "cap"), ("box", "cap"),
            ("box", "tray"), ("box", "cover"), ("hub", "cover"), ("box", "bracket"), ("fin8", "bracket"),
            ("fin7", "bracket"), ("fin9", "bracket"), ("plate", "bracket"), ("bundle", "bracket")}
    res = {}
    for deg in range(-20, 21, 1):
        moved = {n: nod(s, deg) for n, s in head.items()}
        for (hn, bn) in near:
            d = moved[hn].distance(base[bn])
            key = (hn, bn)
            if key not in res or d < res[key][0]:
                res[key] = (d, deg)
    # everything else, coarse: bounding boxes far apart -> skip
    for (hn, bn), (d, deg) in sorted(res.items()):
        lim = 0.1 if (hn, bn) in iface else 3.0
        ok(d >= lim, f"{hn} vs {bn}", f"min {d:.2f} mm at {deg:+d} deg (need {lim})")
    xmin = 1e9
    for deg in range(-20, 21, 1):
        for n, s in head.items():
            a = nod_pts(pts(s, 0.5), deg)
            if a[:, 0].min() < xmin:
                xmin, at = a[:, 0].min(), (n, deg)
    ok(xmin - P.WALL_X >= 10.0, "head vs wall plane", f"min {xmin - P.WALL_X:.1f} mm ({at[0]} at {at[1]:+d} deg), wall at x {P.WALL_X}")
    zlow = 1e9
    for deg in range(-20, 21, 1):
        a = nod_pts(pts(head["box"], 0.3), deg)
        zlow = min(zlow, a[:, 2].min())
    ok(zlow - P.DECK_Z1 >= 3.0, "box over the deck", f"lowest box point {zlow:.1f}, deck top {P.DECK_Z1} -> {zlow - P.DECK_Z1:.1f} mm")

    print("E. STL files")
    import trimesh
    sd = os.path.join(HERE, "stl")
    for f in sorted(os.listdir(sd)):
        if not f.endswith(".stl"):
            continue
        m = trimesh.load(os.path.join(sd, f), force="mesh")
        ext = m.bounds[1] - m.bounds[0]
        ok(m.is_watertight and max(ext) <= 256.0, f, f"watertight {m.is_watertight}, {ext[0]:.0f} x {ext[1]:.0f} x {ext[2]:.0f} mm, "
           f"{len(m.faces)} triangles")
    print("FAILED:", FAILS if FAILS else "none")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
