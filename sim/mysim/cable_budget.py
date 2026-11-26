"""Cable length budget of the in-plane wirings K1 / K2 (plans sim/plan_20261126_risk3.md section 2 and
sim/plan_20261126_rod.md section 7.9).  [geometry, not measured]
Element cable = run inside the ring plane (film -> rib -> spine -> bundle) + bundle down to the box top + run inside
the box to the U.FL plug point of its SP16T port (straight distance + 30 mm for the drop to the board, the plug and
slack).  SP16T: PCB 60 x 100 mm, 8 ports on one long edge, 4 on each short edge (seller photos); the 8-port edge
faces the back, its plug points 5 to 7.5 mm behind the bundle.
  k1   ribs square to the spine, bundle inside the back rim (45 mm behind the axis)
  k2i  slanted ribs (45 deg), bundle inside the back rim (the wiring chosen on 2026-11-27)
  k2   slanted ribs, bundle outside the ring, 78.75 mm behind the axis
  k2n  slanted ribs, bundle outside the ring, 67.5 mm behind the axis
Run: python3 mysim/cable_budget.py [layout, default k2i] [box top in mm below the ring plane, default 88.9]"""
import sys
import numpy as np

YS = -11.25
SQUARE = {0: 50.0, 1: 45.0, 15: 45.0, 2: 33.75, 14: 33.75, 3: 22.5, 13: 22.5, 4: 11.25, 12: 11.25,
          5: -11.25, 11: -11.25, 6: -33.75, 10: -33.75, 7: -45.0, 8: -45.0, 9: -45.0}
# layout: (slanted ribs, bundle x, plug line of the 8-port edge behind the bundle [mm])
LAYOUTS = {"k1": (False, -45.0, 7.5), "k2i": (True, -45.0, 7.5), "k2": (True, -78.75, 7.5), "k2n": (True, -67.5, 5.0)}


def elem_xy(k):
    a = 2.0 * np.pi * k / 16
    return 54.0 * np.cos(a), 54.0 * np.sin(a)


def seg_dist0(p, q):
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    t = float(np.clip(-np.dot(p, q - p) / np.dot(q - p, q - p), 0.0, 1.0))
    return float(np.linalg.norm(p + t * (q - p)))


def nodes(slanted):
    """Spine node of every rib (same rule as rod_heads.py: 45 deg, 5.625 mm steps, 10 mm clear of the centre)."""
    if not slanted:
        return dict(SQUARE)
    n = {}
    for k in range(16):
        x, y = elem_xy(k)
        n[k] = float(np.clip(5.625 * round((x - abs(y - YS)) / 5.625), -45.0, 45.0))
        while seg_dist0((x, y), (n[k], YS)) < 10.0 and n[k] > -45.0:
            n[k] -= 5.625
    return n


def in_plane(k, node, gx):
    x, y = elem_xy(k)
    return 6.0 + float(np.hypot(x - node[k], y - YS)) + abs(node[k] - gx)     # 6 mm from the film to the rib


def ports(gx, back=7.5, slack=30.0):
    """U.FL plug points (board edge + 9.5 mm jack + 13 mm adapter) and the run from the bundle entry (gx, YS)."""
    xp = gx - back                     # plug line of the 8-port edge
    xb = xp + 22.5                     # back edge of the board
    pts = [(xp, y) for y in (-42, -30, -18, -6, 6, 18, 30, 42)]
    for x in (8, 21, 34, 47):
        pts += [(xb + x, -72.5), (xb + x, 72.5)]
    return [float(np.hypot(px - gx, py - YS)) + slack for px, py in pts]


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "k2i"
    z_top = float(sys.argv[2]) if len(sys.argv) > 2 else 88.9
    slanted, gx, back = LAYOUTS[name]
    node = nodes(slanted)
    elem = np.array([in_plane(k, node, gx) + z_top for k in range(16)])
    port = np.array(ports(gx, back))
    eo = np.argsort(-elem)             # longest cables to the nearest ports minimises the longest total
    po = np.argsort(port)
    tot = elem[eo] + port[po]
    print(f"# layout {name}: bundle {abs(gx):.2f} mm behind the axis, box top {z_top:.1f} mm below the ring plane; mm")
    print("# element  spine node  in-plane+down  port  in-box  total")
    for e, p, t in zip(eo, po, tot):
        print(f"  {e:2d}      {node[e]:7.2f}      {elem[e]:6.1f}      {p:2d}   {port[p]:6.1f}  {t:6.1f}")
    print(f"longest {tot.max():.1f} mm, shortest {tot.min():.1f} mm; margin to 300 mm: {300 - tot.max():+.1f} mm,"
          f" to 250 mm: {250 - tot.max():+.1f} mm")
    ref = 6.0 + abs(YS) + abs(0.0 - gx) + z_top + 60.0
    print(f"reference antenna cable about {ref:.0f} mm (centre -> spine -> bundle -> box, + 60 mm to the divider)")


if __name__ == "__main__":
    main()
