"""Four room designs that no earlier DeteCam simulation has used ("Fresh" hold-out).

Written on 2026-11-25 BEFORE looking at any result of the review experiments (AGENTS.md 5.1.3 / 5.1.6).
They are used once, to check whether a configuration chosen on the development rooms still wins in rooms
that neither the original report nor this review ever tuned on.  Same materials / people / camera-hiding
conventions as receiver/sim/v1_ring/gen_scenes.py (its SceneBuilder is reused unchanged).

  classroom_c  8.0 x 8.5 x 2.8 m, 42 seated (facing the BACK wall blackboard) + 2 proctors, lockers on x=0
  lodging_c    3.6 x 7.2 x 2.4 m long studio, wardrobe mirror door, fridge, TV, metal window frame, 2 people
  office_a     8.0 x 6.0 x 2.7 m open office (space type not in any earlier scene): 8 cubicle desks with
               1.2 m metal partitions, monitors, metal cabinets, metal whiteboard, 6 seated people
  restroom_c   3.2 x 7.5 x 2.6 m narrow restroom, 4 stalls (metal front, wood dividers), mirror on short wall
"""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
import gen_scenes as G  # noqa: E402

FRESH_SCENES = ["classroom_c", "lodging_c", "office_a", "restroom_c"]


def _seated_yplus(b, sx, sy, lean):
    w = 0.21
    b.box((sx - w, sy - 0.11 + lean, 0.45), (sx + w, sy + 0.11 + lean, 1.08), "body", "torso")
    b.box((sx - 0.09, sy - 0.01 + lean, 1.10), (sx + 0.09, sy + 0.19 + lean, 1.32), "body", "head")


def build_fresh(scene_id, fc, rng, mesh_dir):
    b = G.SceneBuilder(fc, mesh_dir)
    refl, cams, types = [], [], []

    if scene_id == "classroom_c":
        L, W, H = 8.0, 8.5, 2.8
        b.shell(L, W, H, wall="concrete")
        b.box((2.2, W - 0.03, 0.9), (5.8, W, 2.1), "metal", "blackboard")
        refl.append((1, W - 0.03, [2.2, W - 0.03, 0.9], [5.8, W, 2.1], 0.90))
        b.box((L - 0.02, 1.0, 0.9), (L, 7.0, 2.2), "glass", "window")
        b.box((0.0, 0.4, 0.0), (0.03, 1.3, 2.05), "metal", "door")
        b.box((3.4, W - 1.4, 0.0), (4.6, W - 0.8, 1.05), "wood", "podium")
        b.box((0.02, 3.0, 0.0), (0.45, 6.0, 1.8), "metal", "lockers")
        refl.append((0, 0.45, [0.02, 3.0, 0.0], [0.45, 6.0, 1.8], 0.85))
        b.box((6.2, W - 0.08, 1.1), (7.5, W - 0.02, 1.85), "metal", "tv")
        b.add_person_standing(rng.uniform(1.5, 6.5), rng.uniform(7.2, 7.6))
        b.add_person_standing(rng.uniform(7.2, 7.6), rng.uniform(1.5, 6.0))
        seats = []
        for ry in np.linspace(1.3, 6.4, 7):
            for cx in np.linspace(1.3, 6.9, 6):
                sx, sy = cx + rng.normal(0, 0.04), ry + rng.normal(0, 0.04)
                lean = rng.uniform(-0.02, 0.09)
                b.add_desk_metal_legs(sx - 0.30, sx + 0.30, sy + 0.17, sy + 0.62)
                _seated_yplus(b, sx, sy, lean)
                seats.append((sx, sy, lean))
        for idx in rng.choice(len(seats), 40, replace=False):
            sx, sy, lean = seats[idx]
            typ = rng.choice(["chest", "chest", "glasses", "pencil_case", "desk"])
            p = dict(chest=[sx + rng.uniform(-0.05, 0.05), sy + 0.15 + lean, 0.98],
                     glasses=[sx, sy + 0.22 + lean, 1.24],
                     pencil_case=[sx + rng.uniform(-0.18, 0.18), sy + 0.40, 0.77],
                     desk=[sx + rng.uniform(-0.22, 0.22), sy + 0.50, 0.76])[typ]
            cams.append(p)
            types.append(typ)
        d, z = 0.45, 2.1

    elif scene_id == "lodging_c":
        L, W, H = 3.6, 7.2, 2.4
        b.shell(L, W, H, wall="plasterboard")
        b.box((0.2, 3.6, 0.0), (1.8, 5.7, 0.5), "wood", "bed")
        b.box((L - 0.62, 0.6, 0.0), (L - 0.02, 2.4, 2.1), "wood", "wardrobe")
        b.box((L - 0.64, 0.7, 0.3), (L - 0.62, 1.5, 1.9), "metal", "mirror")
        refl.append((0, L - 0.64, [L - 0.64, 0.7, 0.3], [L - 0.62, 1.5, 1.9], 0.92))
        b.box((0.02, 1.0, 0.0), (0.60, 2.4, 0.75), "wood", "desk")
        b.box((L - 0.08, 4.0, 1.0), (L - 0.03, 5.2, 1.7), "metal", "tv")
        b.box((0.05, 0.05, 0.0), (0.60, 0.60, 1.10), "metal", "fridge")
        b.box((0.8, W - 0.03, 0.9), (2.8, W, 2.0), "glass", "window")
        b.box((0.75, W - 0.05, 0.85), (2.85, W - 0.02, 0.92), "metal", "winframe")
        b.box((0.9, W - 0.25, 2.05), (1.8, W - 0.03, 2.33), "metal", "aircon")
        b.add_person_standing(rng.uniform(2.0, 2.9), rng.uniform(2.7, 3.4))
        b.add_person_standing(rng.uniform(2.1, 3.0), rng.uniform(5.9, 6.6))
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.35:
                wsel = int(rng.integers(4))
                z0 = rng.uniform(0.3, 2.1)
                cand = [[0.06, rng.uniform(0.2, W - 0.2), z0], [L - 0.06, rng.uniform(0.2, W - 0.2), z0],
                        [rng.uniform(0.2, L - 0.2), 0.06, z0], [rng.uniform(0.2, L - 0.2), W - 0.06, z0]]
                p, typ = np.array(cand[wsel]), "wall_frame"
            elif u < 0.55:
                p, typ = np.array([rng.uniform(0.4, L - 0.4), rng.uniform(0.4, W - 0.4), H - 0.06]), "smoke_alarm"
            elif u < 0.78:
                p, typ = np.array([rng.uniform(0.08, 0.55), rng.uniform(1.05, 2.35), rng.uniform(0.80, 1.5)]), "desk_charger"
            else:
                p, typ = G._sample_free(rng, [0.2, 0.2, 0.5], [L - 0.2, W - 0.2, 2.2], b.boxes, (L, W, H)), "furniture"
            if not G._inside_any(p, b.boxes):
                cams.append(p.tolist())
                types.append(typ)
        d, z = 0.35, 2.05

    elif scene_id == "office_a":
        L, W, H = 8.0, 6.0, 2.7
        b.shell(L, W, H, wall="plasterboard")
        b.box((2.4, W - 0.03, 0.9), (4.8, W, 2.1), "metal", "whiteboard")
        refl.append((1, W - 0.03, [2.4, W - 0.03, 0.9], [4.8, W, 2.1], 0.90))
        b.box((0.0, 1.0, 0.9), (0.02, 5.0, 2.2), "glass", "window")
        b.box((L - 0.03, 0.4, 0.0), (L, 1.3, 2.05), "metal", "door")
        b.box((L - 0.50, 2.2, 0.0), (L - 0.03, 5.2, 1.40), "metal", "cabinets")
        refl.append((0, L - 0.50, [L - 0.50, 2.2, 0.0], [L - 0.03, 5.2, 1.40], 0.85))
        desks = []
        for row, y0 in enumerate((1.3, 3.5)):
            for k in range(4):
                x0 = 1.0 + k * 1.5
                b.add_desk_metal_legs(x0, x0 + 1.3, y0, y0 + 0.7)
                b.box((x0 + 0.40, y0 + 0.55, 0.83), (x0 + 0.90, y0 + 0.60, 1.18), "metal", "monitor")
                b.box((x0 - 0.04, y0 - 0.05, 0.0), (x0 - 0.01, y0 + 0.75, 1.20), "metal", "partition")
                b.box((x0, y0 + 0.72, 0.0), (x0 + 1.3, y0 + 0.75, 1.20), "metal", "partition")
                desks.append((x0, y0))
        for i in rng.choice(8, 6, replace=False):
            x0, y0 = desks[i]
            b.add_person_seated(x0 + 0.65 + rng.normal(0, 0.05), y0 - 0.25, lean=rng.uniform(-0.02, 0.08), facing_axis="y-")
        # note: seated people face +y here physically (towards their monitor); the box model is symmetric enough
        while len(cams) < 40:
            u = rng.uniform()
            x0, y0 = desks[int(rng.integers(8))]
            if u < 0.30:
                p, typ = np.array([x0 + rng.uniform(0.42, 0.88), y0 + 0.575, 1.22]), "monitor_top"
            elif u < 0.50:
                p, typ = np.array([x0 + rng.uniform(0.1, 1.2), y0 + 0.735, 1.24]), "partition_top"
            elif u < 0.65:
                p, typ = np.array([L - rng.uniform(0.1, 0.45), rng.uniform(2.3, 5.1), 1.45]), "cabinet_top"
            elif u < 0.82:
                p, typ = np.array([rng.uniform(0.3, L - 0.3), rng.uniform(0.3, W - 0.3), H - 0.06]), "ceiling"
            else:
                wsel = int(rng.integers(4))
                z0 = rng.uniform(1.6, 2.3)
                cand = [[0.06, rng.uniform(0.3, W - 0.3), z0], [L - 0.06, rng.uniform(0.3, 2.0), z0],
                        [rng.uniform(0.3, L - 0.3), 0.06, z0], [rng.uniform(0.3, L - 0.3), W - 0.06, z0]]
                p, typ = np.array(cand[wsel]), "wall_clock"
            if not G._inside_any(p, b.boxes):
                cams.append(p.tolist())
                types.append(typ)
        d, z = 0.40, 2.2

    elif scene_id == "restroom_c":
        L, W, H = 3.2, 7.5, 2.6
        b.shell(L, W, H, wall="tile", floor="tile")
        for sy in (1.7, 2.8, 3.9, 5.0, 6.1):
            b.box((L - 1.5, sy - 0.02, 0.15), (L, sy + 0.02, 2.0), "wood", "stall")
        b.box((L - 1.53, 1.7, 0.15), (L - 1.47, 6.1, 2.0), "metal", "stall_front")
        b.box((L - 0.10, 1.75, 0.0), (L - 0.02, 6.05, 0.9), "metal", "pipes")
        b.box((0.4, 0.0, 0.0), (2.6, 0.55, 0.82), "tile", "sinks")
        b.box((0.4, 0.0, 0.95), (2.6, 0.02, 2.10), "metal", "mirror")
        refl.append((1, 0.02, [0.4, 0.0, 0.95], [2.6, 0.02, 2.10], 0.92))
        b.box((0.02, 3.0, 0.9), (0.20, 3.4, 1.3), "metal", "hand_dryer")
        for _ in range(2):
            b.add_person_standing(rng.uniform(0.5, 1.3), rng.uniform(1.2, 6.5))
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.35:
                p, typ = np.array([rng.uniform(0.2, L - 0.2), rng.uniform(0.2, W - 0.2), H - 0.06]), "ceiling_vent"
            elif u < 0.65:
                p, typ = np.array([rng.uniform(L - 1.4, L - 0.2), rng.uniform(1.9, 5.9), rng.choice([0.75, 2.07])]), "stall_tp"
            else:
                p, typ = G._sample_free(rng, [0.15, 0.15, 0.5], [L - 0.2, W - 0.2, 2.2], b.boxes, (L, W, H)), "wall_fixture"
            if not G._inside_any(p, b.boxes):
                cams.append(p.tolist())
                types.append(typ)
        d, z = 0.35, 2.15
    else:
        raise ValueError(scene_id)

    nodes = [[d, d, z], [L - d, d, z], [d, W - d, z], [L - d, W - d, z]]
    boxes = list(b.boxes)
    return b.finalize(), (L, W, H), np.array(nodes, dtype=np.float64), np.array(cams, dtype=np.float64), types, refl, boxes


if __name__ == "__main__":
    import argparse
    import gen_ext
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="base")
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--root", default="/opt/detecam/sim/data_fresh")
    args = ap.parse_args()
    for tag in args.tags.split(","):
        for sc in FRESH_SCENES:
            for bd in args.bands.split(","):
                for s in [int(x) for x in args.seeds.split(",")]:
                    od = os.path.join(args.root, tag)
                    if os.path.exists(os.path.join(od, f"{sc}_{bd}_s{s}.npy")):
                        continue
                    gen_ext.trace(sc, bd, s, od, tag, builder=build_fresh)
