"""Sionna RT 2.2.0 3D scene generator for DeteCam V1 switch-ring verification (plan_20261125.md).
STRICT SEPARATION (AGENTS.md 5.1.1, receiver AGENTS.md #3):
  This module ONLY builds physical 3D rooms (6 mandatory spaces x 2 variants + worst_metal + ladder)
  and exports raw ray-traced multipath data (a_V, a_H, tau, theta, phi) plus user-observable
  nominal room dimensions and 1-2 major visible reflector faces (blackboard/mirror/locker wall).
  It NEVER imports or calls any estimator.

Usage:
  python3 sim/v1_ring/gen_scenes.py --scenes all --bands 2g4,5g --seeds 0,1,2
  python3 sim/v1_ring/gen_scenes.py --scenes classroom_a --bands 2g4 --seeds 0
"""
import argparse
import os
import tempfile
import time
import numpy as np
import mitsuba as mi
import sionna.rt as rt

KEEP_PATHS = 80

DEV_SCENES = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
EVAL_SCENES = [
    "lecture_a", "lecture_b",
    "bathhouse_a", "bathhouse_b",
    "classroom_b", "lodging_b", "restroom_b", "locker_b",
    "worst_metal",
]
LADDER_SCENES = ["classroom_a_l0", "classroom_a_l1", "classroom_b_l0", "classroom_b_l1"]


def _box_vf(lo, hi, v_offset):
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    V = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
    ], dtype=np.float64)
    F = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32) + v_offset
    return V, F


def _write_ply(path, V, F):
    with open(path, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(V)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n"
        )
        for v in V:
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
        for t in F:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


class SceneBuilder:
    """Groups boxes by radio material into merged PLY meshes for fast Embree/Mitsuba BVH ray tracing."""
    def __init__(self, fc, mesh_dir):
        self.fc = float(fc)
        self.mesh_dir = mesh_dir
        self.scene = rt.load_scene()
        self.scene.frequency = self.fc
        self.by_mat = {}
        self.boxes = []  # (lo, hi, mat, tag)
        self.mats = {}

    def _get_mat(self, key):
        if key in self.mats:
            return self.mats[key]
        if key == "body":
            eps, sig = (52.7, 1.74) if self.fc < 4e9 else (49.3, 4.5)
            m = rt.RadioMaterial(
                name="m_body", thickness=0.20,
                relative_permittivity=eps, conductivity=sig, scattering_coefficient=0.35,
            )
        elif key == "water":
            eps, sig = (78.0, 1.5) if self.fc < 4e9 else (73.0, 5.2)
            m = rt.RadioMaterial(
                name="m_water", thickness=0.40,
                relative_permittivity=eps, conductivity=sig, scattering_coefficient=0.05,
            )
        elif key == "tile":
            m = rt.ITURadioMaterial(
                name="m_tile", itu_type="concrete", thickness=0.20, scattering_coefficient=0.10,
            )
        else:
            sc = dict(concrete=0.25, plasterboard=0.25, wood=0.30, glass=0.05, metal=0.10,
                      ceiling_board=0.30, floorboard=0.25, chipboard=0.30)
            th = dict(concrete=0.20, plasterboard=0.025, wood=0.03, glass=0.006, metal=0.002,
                      ceiling_board=0.02, floorboard=0.03, chipboard=0.02)
            m = rt.ITURadioMaterial(
                name="m_" + key, itu_type=key, thickness=th[key], scattering_coefficient=sc[key],
            )
        self.mats[key] = m
        return m

    def box(self, lo, hi, mat, tag="o"):
        lo_a = np.asarray(lo, dtype=np.float64)
        hi_a = np.asarray(hi, dtype=np.float64)
        if np.any(hi_a <= lo_a + 1e-4):
            return
        self.boxes.append((lo_a, hi_a, mat, tag))
        self.by_mat.setdefault(mat, []).append((lo_a, hi_a))

    def shell(self, L, W, H, wall="concrete", floor="concrete", ceil="ceiling_board", t=0.15):
        self.box((-t, -t, -t), (L + t, W + t, 0.0), floor, "floor")
        self.box((-t, -t, H), (L + t, W + t, H + t), ceil, "ceil")
        self.box((-t, -t, 0.0), (0.0, W + t, H), wall, "wx0")
        self.box((L, -t, 0.0), (L + t, W + t, H), wall, "wxl")
        self.box((0.0, -t, 0.0), (L, 0.0, H), wall, "wy0")
        self.box((0.0, W, 0.0), (L, W + t, H), wall, "wyw")

    def add_person_standing(self, x, y, z0=0.0):
        # Cylinder-like torso + legs + head (ITU/Gabriel human dielectric)
        self.box((x - 0.16, y - 0.10, z0), (x + 0.16, y + 0.10, z0 + 0.85), "body", "legs")
        self.box((x - 0.22, y - 0.12, z0 + 0.86), (x + 0.22, y + 0.12, z0 + 1.48), "body", "torso")
        self.box((x - 0.09, y - 0.09, z0 + 1.50), (x + 0.09, y + 0.09, z0 + 1.72), "body", "head")

    def add_person_seated(self, sx, sy, z0=0.0, lean=0.0, facing_axis="y-"):
        w = 0.21
        if facing_axis == "y-":
            self.box((sx - w, sy - 0.11 - lean, z0 + 0.45), (sx + w, sy + 0.11 - lean, z0 + 1.08), "body", "torso")
            self.box((sx - 0.09, sy - 0.19 - lean, z0 + 1.10), (sx + 0.09, sy + 0.01 - lean, z0 + 1.32), "body", "head")
        else:  # facing x-
            self.box((sx - 0.11 - lean, sy - w, z0 + 0.45), (sx + 0.11 - lean, sy + w, z0 + 1.08), "body", "torso")
            self.box((sx - 0.19 - lean, sy - 0.09, z0 + 1.10), (sx + 0.01 - lean, sy + 0.09, z0 + 1.32), "body", "head")

    def add_desk_metal_legs(self, x0, x1, y0, y1, z0=0.0, h=0.73):
        self.box((x0, y0, z0 + h - 0.03), (x1, y1, z0 + h), "wood", "desk")
        # 2 metal side leg frames (compact geometry, authentic metal scattering)
        self.box((x0 + 0.02, y0 + 0.03, z0), (x0 + 0.05, y1 - 0.03, z0 + h - 0.035), "metal", "deskleg")
        self.box((x1 - 0.05, y0 + 0.03, z0), (x1 - 0.02, y1 - 0.03, z0 + h - 0.035), "metal", "deskleg")

    def finalize(self):
        objs = []
        for mat, box_list in self.by_mat.items():
            Vs, Fs = [], []
            off = 0
            for lo, hi in box_list:
                V, F = _box_vf(lo, hi, off)
                Vs.append(V)
                Fs.append(F)
                off += 8
            V_all = np.vstack(Vs)
            F_all = np.vstack(Fs)
            ply_path = os.path.join(self.mesh_dir, f"grp_{mat}.ply")
            _write_ply(ply_path, V_all, F_all)
            objs.append(rt.SceneObject(fname=ply_path, name=f"obj_{mat}", radio_material=self._get_mat(mat)))
        self.scene.edit(add=objs)
        return self.scene


def _inside_any(p, boxes, margin=0.025):
    for lo, hi, mat, tag in boxes:
        if tag in ("floor", "ceil", "wx0", "wxl", "wy0", "wyw"):
            continue
        if np.all(p >= lo - margin) and np.all(p <= hi + margin):
            return True
    return False


def _sample_free(rng, lo_b, hi_b, boxes, dims, max_tries=200):
    L, W, H = dims
    for _ in range(max_tries):
        p = np.array([rng.uniform(lo_b[0], hi_b[0]), rng.uniform(lo_b[1], hi_b[1]), rng.uniform(lo_b[2], hi_b[2])])
        if 0.03 <= p[0] <= L - 0.03 and 0.03 <= p[1] <= W - 0.03 and 0.04 <= p[2] <= H - 0.03:
            if not _inside_any(p, boxes):
                return p
    return np.clip(0.5 * (np.asarray(lo_b) + np.asarray(hi_b)), [0.05, 0.05, 0.05], [L - 0.05, W - 0.05, H - 0.05])


def build_scene(scene_id, fc, rng, mesh_dir):
    """Build one of the 13 pre-registered scenes (or ladder variants).
    Returns: (sionna_scene, dims, nodes, cams, cam_types, user_reflectors)
    where user_reflectors = list of (axis, val, lo_xyz, hi_xyz, nominal_rho) for 1-2 major user-visible
    metal reflectors (blackboard / large wall mirror / locker bank face).
    """
    b = SceneBuilder(fc, mesh_dir)
    user_reflectors = []
    cams, cam_types = [], []

    if scene_id in ("classroom_a", "classroom_a_l0", "classroom_a_l1"):
        L, W, H = 9.0, 7.5, 3.0
        b.shell(L, W, H, wall="concrete")
        is_empty = scene_id in ("classroom_a_l0", "classroom_a_l1")
        if not is_empty:
            # Metal blackboard 3600x1200 mm on front wall y=0
            b.box((2.7, 0.0, 0.9), (6.3, 0.03, 2.1), "metal", "blackboard")
            user_reflectors.append((1, 0.03, [2.7, 0.0, 0.9], [6.3, 0.03, 2.1], 0.90))
            b.box((0.0, 1.2, 0.9), (0.02, 6.8, 2.2), "glass", "window")
            b.box((L - 0.03, 0.5, 0.0), (L, 1.4, 2.05), "metal", "door")
            b.box((3.8, 0.75, 0.0), (5.0, 1.35, 1.05), "wood", "podium")
            # TV / e-board & steel lockers (lockers on back/side; TV unmodeled clutter)
            b.box((6.6, 0.02, 1.1), (7.9, 0.08, 1.85), "metal", "tv")
            b.box((L - 0.45, 2.8, 0.0), (L - 0.02, 5.2, 1.8), "metal", "lockers")
            user_reflectors.append((0, L - 0.45, [L - 0.45, 2.8, 0.0], [L - 0.02, 5.2, 1.8], 0.85))
            # 2 moving proctors
            for px, py in ((rng.uniform(1.0, 3.0), rng.uniform(1.5, 2.0)), (rng.uniform(6.0, 8.2), rng.uniform(2.5, 6.5))):
                b.add_person_standing(px, py)
        cols = np.linspace(1.2, 7.6, 8)
        rows = np.linspace(2.3, 6.3, 5)
        for ry in rows:
            for cx in cols:
                sx, sy = cx + rng.normal(0, 0.04), ry + rng.normal(0, 0.04)
                lean = rng.uniform(-0.02, 0.09)
                if not is_empty:
                    b.add_desk_metal_legs(sx - 0.30, sx + 0.30, sy - 0.62, sy - 0.17)
                    b.add_person_seated(sx, sy, lean=lean, facing_axis="y-")
                typ = rng.choice(["chest", "chest", "glasses", "pencil_case", "desk"])
                p = dict(
                    chest=[sx + rng.uniform(-0.05, 0.05), sy - 0.15 - lean, 0.98],
                    glasses=[sx, sy - 0.22 - lean, 1.24],
                    pencil_case=[sx + rng.uniform(-0.18, 0.18), sy - 0.40, 0.77],
                    desk=[sx + rng.uniform(-0.22, 0.22), sy - 0.50, 0.76],
                )[typ]
                cams.append(p)
                cam_types.append(typ)
        nodes = [[0.45, 0.45, 2.2], [8.55, 0.45, 2.2], [0.45, 7.05, 2.2], [8.55, 7.05, 2.2]]

    elif scene_id in ("classroom_b", "classroom_b_l0", "classroom_b_l1"):
        L, W, H = 9.5, 7.0, 3.0
        b.shell(L, W, H, wall="concrete")
        is_empty = scene_id in ("classroom_b_l0", "classroom_b_l1")
        if not is_empty:
            # Variant B: blackboard on side wall x=0, students face x=0
            b.box((0.0, 1.7, 0.9), (0.03, 5.3, 2.1), "metal", "blackboard")
            user_reflectors.append((0, 0.03, [0.0, 1.7, 0.9], [0.03, 5.3, 2.1], 0.90))
            b.box((1.5, W - 0.02, 0.9), (8.0, W, 2.2), "glass", "window")
            b.box((0.75, 2.9, 0.0), (1.35, 4.1, 1.05), "wood", "podium")
            b.box((L - 0.45, 1.0, 0.0), (L - 0.02, 5.8, 1.8), "metal", "lockers")
            user_reflectors.append((0, L - 0.45, [L - 0.45, 1.0, 0.0], [L - 0.02, 5.8, 1.8], 0.85))
            b.box((0.02, 5.6, 1.1), (0.08, 6.6, 1.8), "metal", "tv")
            for px, py in ((rng.uniform(1.6, 2.2), rng.uniform(0.8, 2.5)), (rng.uniform(3.5, 8.0), rng.uniform(5.6, 6.4))):
                b.add_person_standing(px, py)
        xs = np.linspace(2.4, 8.2, 5)
        ys = np.linspace(0.9, 5.8, 8)
        for rx in xs:
            for cy in ys:
                sx, sy = rx + rng.normal(0, 0.04), cy + rng.normal(0, 0.04)
                lean = rng.uniform(-0.02, 0.09)
                if not is_empty:
                    b.add_desk_metal_legs(sx - 0.62, sx - 0.17, sy - 0.28, sy + 0.28)
                    b.add_person_seated(sx, sy, lean=lean, facing_axis="x-")
                typ = rng.choice(["chest", "chest", "glasses", "pencil_case", "desk"])
                p = dict(
                    chest=[sx - 0.15 - lean, sy + rng.uniform(-0.05, 0.05), 0.98],
                    glasses=[sx - 0.22 - lean, sy, 1.24],
                    pencil_case=[sx - 0.40, sy + rng.uniform(-0.18, 0.18), 0.77],
                    desk=[sx - 0.50, sy + rng.uniform(-0.20, 0.20), 0.76],
                )[typ]
                cams.append(p)
                cam_types.append(typ)
        nodes = [[0.45, 0.45, 2.2], [9.05, 0.45, 2.2], [0.45, 6.55, 2.2], [9.05, 6.55, 2.2]]

    elif scene_id in ("lecture_a", "lecture_b"):
        tiered = (scene_id == "lecture_b")
        L, W, H = (14.0, 10.5, 3.8) if tiered else (15.0, 10.0, 3.5)
        b.shell(L, W, H, wall="concrete")
        if not tiered:
            b.box((5.7, 0.0, 1.0), (9.3, 0.03, 2.2), "metal", "blackboard")
            user_reflectors.append((1, 0.03, [5.7, 0.0, 1.0], [9.3, 0.03, 2.2], 0.90))
            b.box((L - 0.03, 0.6, 0.0), (L, 1.8, 2.2), "metal", "door")
            b.box((6.5, 1.0, 0.0), (8.2, 1.8, 1.1), "wood", "podium")
            b.box((3.0, 0.03, 2.5), (12.0, 0.15, 3.3), "metal", "ac_screen")
            b.add_person_standing(7.3, 1.95)
            b.add_person_standing(rng.uniform(1.0, 2.0), rng.uniform(4.0, 8.0))
            cols = np.linspace(1.4, 13.6, 12)
            rows = np.linspace(2.8, 9.0, 7)  # 84 seated
            seats = []
            for ri, ry in enumerate(rows):
                for ci, cx in enumerate(cols):
                    sx, sy = cx + rng.normal(0, 0.04), ry + rng.normal(0, 0.04)
                    lean = rng.uniform(-0.02, 0.08)
                    b.add_desk_metal_legs(sx - 0.32, sx + 0.32, sy - 0.56, sy - 0.18, z0=0.0)
                    b.add_person_seated(sx, sy, z0=0.0, lean=lean, facing_axis="y-")
                    seats.append((sx, sy, 0.0, lean))
            chosen = rng.choice(len(seats), 40, replace=False)
            for idx in chosen:
                sx, sy, z0, lean = seats[idx]
                typ = rng.choice(["chest", "glasses", "pencil_case", "desk"])
                p = dict(
                    chest=[sx, sy - 0.15 - lean, z0 + 0.98],
                    glasses=[sx, sy - 0.22 - lean, z0 + 1.24],
                    pencil_case=[sx + rng.uniform(-0.15, 0.15), sy - 0.38, z0 + 0.77],
                    desk=[sx + rng.uniform(-0.20, 0.20), sy - 0.48, z0 + 0.76],
                )[typ]
                cams.append(p)
                cam_types.append(typ)
            nodes = [[0.6, 0.6, 2.5], [14.4, 0.6, 2.5], [0.6, 9.4, 2.5], [14.4, 9.4, 2.5]]
        else:
            # Tiered lecture hall, blackboard on side wall x=0
            b.box((0.0, 3.45, 1.0), (0.03, 7.05, 2.2), "metal", "blackboard")
            user_reflectors.append((0, 0.03, [0.0, 3.45, 1.0], [0.03, 7.05, 2.2], 0.90))
            b.box((1.0, 4.4, 0.0), (1.8, 6.0, 1.1), "wood", "podium")
            b.box((0.03, 2.0, 2.8), (0.20, 8.5, 3.5), "metal", "ac_screen")
            b.box((L - 0.03, 0.6, 0.0), (L, 1.8, 2.8), "metal", "door")
            # 4 stepped floor risers
            for step in range(4):
                b.box((5.0 + step * 2.1, 0.0, 0.0), (L, W, (step + 1) * 0.20), "concrete", "riser")
            b.add_person_standing(2.0, 5.2)
            b.add_person_standing(rng.uniform(4.0, 6.0), 0.7)
            xs = np.linspace(3.0, 12.6, 7)
            ys = np.linspace(1.1, 9.4, 12)
            seats = []
            for rx in xs:
                z0 = 0.20 * max(0, min(4, int((rx - 5.0) / 2.1) + 1)) if rx >= 5.0 else 0.0
                for cy in ys:
                    sx, sy = rx + rng.normal(0, 0.04), cy + rng.normal(0, 0.04)
                    lean = rng.uniform(-0.02, 0.08)
                    b.add_desk_metal_legs(sx - 0.56, sx - 0.18, sy - 0.30, sy + 0.30, z0=z0)
                    b.add_person_seated(sx, sy, z0=z0, lean=lean, facing_axis="x-")
                    seats.append((sx, sy, z0, lean))
            chosen = rng.choice(len(seats), 40, replace=False)
            for idx in chosen:
                sx, sy, z0, lean = seats[idx]
                typ = rng.choice(["chest", "glasses", "pencil_case", "desk"])
                p = dict(
                    chest=[sx - 0.15 - lean, sy, z0 + 0.98],
                    glasses=[sx - 0.22 - lean, sy, z0 + 1.24],
                    pencil_case=[sx - 0.38, sy + rng.uniform(-0.15, 0.15), z0 + 0.77],
                    desk=[sx - 0.48, sy + rng.uniform(-0.20, 0.20), z0 + 0.76],
                )[typ]
                cams.append(p)
                cam_types.append(typ)
            nodes = [[0.6, 0.6, 2.8], [13.4, 0.6, 2.8], [0.6, 9.9, 2.8], [13.4, 9.9, 2.8]]

    elif scene_id in ("lodging_a", "lodging_b"):
        var_b = (scene_id == "lodging_b")
        L, W, H = (5.0, 5.5, 2.6) if var_b else (4.5, 6.0, 2.5)
        b.shell(L, W, H, wall="plasterboard")
        # Bathroom partition inside lodging room
        b.box((0.0, W - 1.85, 0.0), (2.2, W - 1.75, H), "plasterboard", "bath_part")
        b.box((2.15, W - 1.85, 0.0), (2.25, W - 0.85, H), "plasterboard", "bath_part2")
        # Bed, wardrobe, desk, TV, fridge, metal window frame
        b.box((0.3, 1.7, 0.0), (2.3, 3.5, 0.55), "wood", "bed")
        b.box((L - 0.70, 2.1, 0.0), (L - 0.02, 3.8, 2.1), "wood", "wardrobe")
        b.box((1.2, 0.02, 0.0), (3.1, 0.50, 0.80), "wood", "desk")
        b.box((1.5, 0.04, 0.95), (2.8, 0.09, 1.70), "metal", "tv")
        b.box((L - 0.65, 0.05, 0.0), (L - 0.05, 0.65, 0.90), "metal", "fridge")
        b.box((1.2, W - 0.03, 0.9), (3.4, W, 2.1), "glass", "window")
        b.box((1.15, W - 0.05, 0.85), (3.45, W - 0.02, 0.92), "metal", "winframe")
        if not var_b:
            # Mirrors facing interior: vanity mirror on x=0 + bathroom mirror
            b.box((0.0, 0.4, 0.8), (0.02, 1.55, 2.0), "metal", "mirror")
            user_reflectors.append((0, 0.02, [0.0, 0.4, 0.8], [0.02, 1.55, 2.0], 0.92))
            b.box((0.2, W - 0.03, 1.0), (1.8, W - 0.01, 2.0), "metal", "bath_mirror")
            b.add_person_standing(rng.uniform(2.6, 3.5), rng.uniform(1.0, 1.8))
            b.add_person_standing(rng.uniform(2.8, 3.6), rng.uniform(3.8, 4.8))
        else:
            # Variant B: large mirror on y=0 side wall + bathroom mirror on x=0
            b.box((3.2, 0.0, 0.5), (4.3, 0.02, 2.1), "metal", "mirror")
            user_reflectors.append((1, 0.02, [3.2, 0.0, 0.5], [4.3, 0.02, 2.1], 0.92))
            b.box((0.0, W - 1.6, 1.0), (0.02, W - 0.2, 2.0), "metal", "bath_mirror")
            b.add_person_standing(rng.uniform(2.7, 3.7), rng.uniform(1.1, 2.0))
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.35:  # wall frame / charger / outlet
                w = int(rng.integers(4))
                z = rng.uniform(0.3, 2.1)
                cands = [[0.06, rng.uniform(0.2, W - 0.2), z], [L - 0.06, rng.uniform(0.2, W - 0.2), z],
                         [rng.uniform(0.2, L - 0.2), 0.06, z], [rng.uniform(0.2, L - 0.2), W - 0.06, z]]
                p, typ = np.array(cands[w]), "wall_frame"
            elif u < 0.55:  # smoke detector / ceiling
                p, typ = np.array([rng.uniform(0.4, L - 0.4), rng.uniform(0.4, W - 0.4), H - 0.06]), "smoke_alarm"
            elif u < 0.78:  # beside TV / desk / charger
                p, typ = np.array([rng.uniform(1.1, 3.2), rng.uniform(0.12, 0.52), rng.uniform(0.84, 1.65)]), "tv_charger"
            else:  # wardrobe / bathroom shelf
                p, typ = _sample_free(rng, [0.2, 0.2, 0.6], [L - 0.2, W - 0.2, 2.2], b.boxes, (L, W, H)), "furniture"
            if not _inside_any(p, b.boxes):
                cams.append(p.tolist())
                cam_types.append(typ)
        nodes = [[0.35, 0.35, 2.15], [L - 0.35, 0.35, 2.15], [0.35, W - 0.35, 2.15], [L - 0.35, W - 0.35, 2.15]]

    elif scene_id in ("restroom_a", "restroom_b"):
        var_b = (scene_id == "restroom_b")
        L, W, H = (4.5, 5.5, 2.5) if var_b else (4.0, 6.0, 2.5)
        b.shell(L, W, H, wall="tile", floor="tile")
        pmat = "wood" if var_b else "metal"
        # 3 toilet stalls along x = L-1.6..L
        for sy in (1.4, 2.8, 4.2):
            if sy < W - 0.6:
                b.box((L - 1.6, sy - 0.03, 0.15), (L, sy + 0.03, 2.05), pmat, "stall")
        b.box((L - 1.63, 0.3, 0.15), (L - 1.57, W - 0.4, 2.05), pmat, "stall_front")
        # Metal pipes & sink counter
        b.box((L - 0.12, 0.2, 0.0), (L - 0.02, W - 0.2, 0.9), "metal", "pipes")
        if not var_b:
            b.box((0.0, 0.8, 0.0), (0.60, 4.6, 0.82), "tile", "sinks")
            b.box((0.0, 0.8, 0.95), (0.02, 4.6, 2.10), "metal", "mirror")
            user_reflectors.append((0, 0.02, [0.0, 0.8, 0.95], [0.02, 4.6, 2.10], 0.92))
            for _ in range(2):
                b.add_person_standing(rng.uniform(0.9, 2.1), rng.uniform(1.0, 4.5))
        else:
            b.box((0.6, 0.0, 0.0), (2.6, 0.60, 0.82), "tile", "sinks")
            b.box((0.6, 0.0, 0.95), (2.6, 0.02, 2.10), "metal", "mirror")
            user_reflectors.append((1, 0.02, [0.6, 0.0, 0.95], [2.6, 0.02, 2.10], 0.92))
            for _ in range(3):
                b.add_person_standing(rng.uniform(0.7, 2.4), rng.uniform(1.0, 4.5))
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.35:  # ceiling corner / vent
                p, typ = np.array([rng.uniform(0.2, L - 0.2), rng.uniform(0.2, W - 0.2), H - 0.06]), "ceiling_vent"
            elif u < 0.65:  # stall partition top or TP holder
                p, typ = np.array([rng.uniform(L - 1.5, L - 0.2), rng.uniform(0.4, W - 0.5), rng.choice([0.75, 2.12])]), "stall_tp"
            else:  # near sinks/mirror/wall
                p, typ = _sample_free(rng, [0.15, 0.15, 0.5], [L - 0.2, W - 0.2, 2.2], b.boxes, (L, W, H)), "wall_fixture"
            if not _inside_any(p, b.boxes):
                cams.append(p.tolist())
                cam_types.append(typ)
        nodes = [[0.35, 0.35, 2.15], [L - 0.35, 0.35, 2.15], [0.35, W - 0.35, 2.15], [L - 0.35, W - 0.35, 2.15]]

    elif scene_id in ("bathhouse_a", "bathhouse_b"):
        var_b = (scene_id == "bathhouse_b")
        L, W, H = (9.5, 8.5, 3.0) if var_b else (10.0, 8.0, 3.0)
        b.shell(L, W, H, wall="tile", floor="tile")
        # Water bath pool + metal faucets + sauna door + shower mirror row
        if not var_b:
            b.box((2.5, 2.2, 0.0), (7.5, 5.8, 0.45), "water", "bath_water")
            b.box((1.0, 0.0, 0.9), (9.0, 0.03, 2.1), "metal", "shower_mirrors")
            user_reflectors.append((1, 0.03, [1.0, 0.0, 0.9], [9.0, 0.03, 2.1], 0.92))
            for fx in np.linspace(1.5, 8.5, 7):
                b.box((fx - 0.06, 0.03, 0.65), (fx + 0.06, 0.18, 0.85), "metal", "faucet")
            b.box((L - 0.04, 3.0, 0.0), (L, 4.4, 2.1), "wood", "sauna_door")
            n_ppl = 7
        else:
            b.box((1.2, 1.5, 0.0), (5.8, 5.2, 0.45), "water", "bath_water")
            b.box((1.0, W - 0.03, 0.9), (8.5, W, 2.1), "metal", "shower_mirrors")
            user_reflectors.append((1, W - 0.03, [1.0, W - 0.03, 0.9], [8.5, W, 2.1], 0.92))
            for fx in np.linspace(1.5, 8.0, 7):
                b.box((fx - 0.06, W - 0.18, 0.65), (fx + 0.06, W - 0.03, 0.85), "metal", "faucet")
            b.box((0.0, 5.5, 0.0), (0.04, 6.8, 2.1), "wood", "sauna_door")
            n_ppl = 8
        for _ in range(n_ppl):
            p = _sample_free(rng, [0.8, 0.8, 0.0], [L - 0.8, W - 0.8, 0.1], b.boxes, (L, W, H))
            b.add_person_standing(p[0], p[1])
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.40:
                p, typ = np.array([rng.uniform(0.3, L - 0.3), rng.uniform(0.3, W - 0.3), H - 0.07]), "ceiling_vent"
            elif u < 0.70:
                my = (W - 0.10) if var_b else 0.10
                p, typ = np.array([rng.uniform(0.8, L - 0.8), my, rng.uniform(0.9, 2.3)]), "mirror_shelf"
            else:
                p, typ = _sample_free(rng, [0.2, 0.2, 0.6], [L - 0.2, W - 0.2, 2.5], b.boxes, (L, W, H)), "wall"
            if not _inside_any(p, b.boxes):
                cams.append(p.tolist())
                cam_types.append(typ)
        nodes = [[0.5, 0.5, 2.4], [L - 0.5, 0.5, 2.4], [0.5, W - 0.5, 2.4], [L - 0.5, W - 0.5, 2.4]]

    elif scene_id in ("locker_a", "locker_b"):
        var_b = (scene_id == "locker_b")
        L, W, H = (6.5, 4.8, 2.7) if var_b else (6.0, 5.0, 2.7)
        b.shell(L, W, H, wall="concrete")
        if not var_b:
            # Metal locker rows along both long walls + full-length mirror on x=0
            b.box((0.8, 0.02, 0.0), (5.2, 0.52, 1.90), "metal", "lockers_s")
            b.box((0.8, W - 0.52, 0.0), (5.2, W - 0.02, 1.90), "metal", "lockers_n")
            user_reflectors.append((1, 0.52, [0.8, 0.02, 0.0], [5.2, 0.52, 1.90], 0.90))
            b.box((0.0, 1.6, 0.2), (0.02, 3.4, 2.0), "metal", "full_mirror")
            user_reflectors.append((0, 0.02, [0.0, 1.6, 0.2], [0.02, 3.4, 2.0], 0.92))
            b.box((1.6, 2.25, 0.0), (4.4, 2.75, 0.45), "wood", "bench")
            b.box((L - 0.45, 2.1, 0.0), (L - 0.05, 2.9, 1.2), "metal", "fan_dryer")
            n_ppl = 6
        else:
            # Central island metal locker bank + side mirror
            b.box((2.0, 2.0, 0.0), (4.8, 2.8, 1.85), "metal", "island_lockers")
            b.box((0.8, W - 0.50, 0.0), (5.5, W - 0.02, 1.85), "metal", "lockers_n")
            user_reflectors.append((1, W - 0.50, [0.8, W - 0.50, 0.0], [5.5, W - 0.02, 1.85], 0.90))
            b.box((L - 0.02, 1.2, 0.2), (L, 3.2, 2.0), "metal", "full_mirror")
            user_reflectors.append((0, L - 0.02, [L - 0.02, 1.2, 0.2], [L, 3.2, 2.0], 0.92))
            b.box((1.5, 0.65, 0.0), (4.5, 1.15, 0.45), "wood", "bench")
            b.box((0.05, 1.8, 0.0), (0.45, 2.6, 1.2), "metal", "fan_dryer")
            n_ppl = 5
        for _ in range(n_ppl):
            p = _sample_free(rng, [0.7, 0.7, 0.0], [L - 0.7, W - 0.7, 0.1], b.boxes, (L, W, H))
            b.add_person_standing(p[0], p[1])
        while len(cams) < 40:
            u = rng.uniform()
            if u < 0.35:  # top of lockers
                p, typ = np.array([rng.uniform(0.9, L - 0.9), rng.choice([0.30, W - 0.30]), 1.98]), "locker_top"
            elif u < 0.65:  # ceiling corner / vent
                p, typ = np.array([rng.uniform(0.25, L - 0.25), rng.uniform(0.25, W - 0.25), H - 0.06]), "ceiling_vent"
            else:  # bench edge / wall / mirror frame
                p, typ = _sample_free(rng, [0.15, 0.15, 0.4], [L - 0.15, W - 0.15, 2.2], b.boxes, (L, W, H)), "bench_wall"
            if not _inside_any(p, b.boxes):
                cams.append(p.tolist())
                cam_types.append(typ)
        nodes = [[0.40, 0.40, 2.25], [L - 0.40, 0.40, 2.25], [0.40, W - 0.40, 2.25], [L - 0.40, W - 0.40, 2.25]]

    elif scene_id == "worst_metal":
        # AGENTS.md 5.1.9: metal-walled room + same-wall cameras + human-occluded cameras
        L, W, H = 5.0, 4.5, 2.6
        b.shell(L, W, H, wall="metal", floor="concrete", ceil="metal")
        # 4 people placed specifically to occlude LOS from corners
        ppl = [(1.4, 1.4), (3.6, 1.4), (1.4, 3.1), (3.6, 3.1)]
        for px, py in ppl:
            b.add_person_standing(px, py)
        b.box((2.1, 2.0, 0.0), (2.9, 2.5, 1.5), "metal", "metal_rack")
        # 12 cameras on SAME walls as receivers (within 0.12 m of y=0 or x=0)
        for _ in range(12):
            if rng.uniform() < 0.5:
                p = np.array([rng.uniform(0.7, L - 0.7), 0.08, rng.uniform(0.5, 2.2)])
            else:
                p = np.array([0.08, rng.uniform(0.7, W - 0.7), rng.uniform(0.5, 2.2)])
            cams.append(p.tolist())
            cam_types.append("same_wall")
        # 14 cameras directly behind people / metal rack (occluded)
        while len(cams) < 26:
            px, py = ppl[int(rng.integers(4))]
            p = np.array([px + rng.uniform(-0.35, 0.35), py + rng.uniform(-0.35, 0.35), rng.uniform(0.7, 1.45)])
            if not _inside_any(p, b.boxes) and 0.1 <= p[0] <= L - 0.1 and 0.1 <= p[1] <= W - 0.1:
                cams.append(p.tolist())
                cam_types.append("occluded")
        # 14 general interior cameras
        while len(cams) < 40:
            p = _sample_free(rng, [0.2, 0.2, 0.3], [L - 0.2, W - 0.2, 2.4], b.boxes, (L, W, H))
            cams.append(p.tolist())
            cam_types.append("general")
        nodes = [[0.35, 0.35, 2.1], [4.65, 0.35, 2.1], [0.35, 4.15, 2.1], [4.65, 4.15, 2.1]]
    else:
        raise ValueError(f"Unknown scene_id: {scene_id}")

    return b.finalize(), (L, W, H), np.array(nodes, dtype=np.float64), np.array(cams, dtype=np.float64), cam_types, user_reflectors


def trace_scene(scene_id, band, seed, out_dir):
    fc = 2.437e9 if band == "2g4" else 5.220e9
    rng = np.random.default_rng(2000 + seed * 101 + abs(hash(scene_id)) % 97)
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="detecam_v1_mesh_") as mdir:
        scene, dims, nodes, cams, cam_types, user_reflectors = build_scene(scene_id, fc, rng, mdir)
        scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="dipole", polarization="V")
        scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH")
        for i, p in enumerate(cams):
            ori = rng.uniform(0, np.pi, 3)
            scene.add(rt.Receiver(f"c{i}", position=mi.Point3f(*map(float, p)), orientation=mi.Point3f(*map(float, ori))))
        for j, p in enumerate(nodes):
            scene.add(rt.Transmitter(f"s{j}", position=mi.Point3f(*map(float, p))))

        ps = rt.PathSolver()
        if scene_id.endswith("_l0"):
            # L0 ladder: empty room, 1-bounce specular only, no diffraction/scattering
            paths = ps(
                scene, synthetic_array=False, max_depth=1, samples_per_src=100000,
                refraction=False, diffraction=False, edge_diffraction=False, diffuse_reflection=False,
            )
            rt_cfg = dict(max_depth=1, refraction=False, diffraction=False, diffuse_reflection=False, samples_per_src=100000)
        else:
            # Full realistic RT: >=4 bounces, refraction, edge diffraction, diffuse scattering
            paths = ps(
                scene, synthetic_array=False, max_depth=4, samples_per_src=100000,
                refraction=True, diffraction=True, edge_diffraction=True, diffuse_reflection=True,
                max_num_paths_per_src=300000,
            )
            rt_cfg = dict(max_depth=4, refraction=True, diffraction=True, edge_diffraction=True, diffuse_reflection=True, samples_per_src=100000)

        a, tau = paths.cir(out_type="numpy", normalize_delays=False)
        # Sionna RT 2.2 shape: a (num_rx, rx_ant, num_tx, tx_ant=2[VH], num_paths, num_ts=1)
        a = a[:, 0, :, :, :, 0]
        tau = tau[:, 0, :, 0, :]
        th = paths.theta_t.numpy()[:, 0, :, 0, :]
        ph = paths.phi_t.numpy()[:, 0, :, 0, :]

    pw = (np.abs(a) ** 2).sum(axis=2)
    k = min(KEEP_PATHS, pw.shape[-1])
    idx = np.argsort(-pw, axis=-1)[..., :k]
    take = lambda arr: np.take_along_axis(arr, idx, axis=-1)
    a_keep = np.stack([take(a[:, :, 0]), take(a[:, :, 1])], axis=2).astype(np.complex64)
    tau_keep = take(tau).astype(np.float64)
    th_keep = take(th).astype(np.float32)
    ph_keep = take(ph).astype(np.float32)
    kept_frac = float((np.take_along_axis(pw, idx, axis=-1).sum(-1) / (pw.sum(-1) + 1e-30)).mean())

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_id}_{band}_s{seed}.npy")
    np.save(
        out_path,
        dict(
            scene_id=scene_id,
            band=band,
            fc=fc,
            seed=seed,
            dims=np.array(dims, dtype=np.float64),
            nodes=nodes,
            cams=cams,
            cam_types=cam_types,
            user_reflectors=user_reflectors,
            rt_cfg=rt_cfg,
            paths=dict(a=a_keep, tau=tau_keep, th=th_keep, ph=ph_keep, kept_frac=kept_frac),
        ),
        allow_pickle=True,
    )
    dt = time.time() - t0
    print(f"[RT] {scene_id:15s} {band:3s} s{seed} | paths={pw.shape[-1]:5d}->keep {k:2d} ({kept_frac*100:5.1f}% pwr) | {dt:5.1f}s", flush=True)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", default="all", help="Comma-separated scene IDs, or 'all', 'dev', 'eval', 'ladder'")
    ap.add_argument("--bands", default="2g4,5g")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    args = ap.parse_args()

    if args.scenes == "all":
        scenes = DEV_SCENES + EVAL_SCENES + LADDER_SCENES
    elif args.scenes == "dev":
        scenes = DEV_SCENES
    elif args.scenes == "eval":
        scenes = EVAL_SCENES
    elif args.scenes == "ladder":
        scenes = LADDER_SCENES
    else:
        scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]

    bands = [b.strip() for b in args.bands.split(",") if b.strip()]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    for sc in scenes:
        for b in bands:
            for s in seeds:
                out_f = os.path.join(args.out_dir, f"{sc}_{b}_s{s}.npy")
                if os.path.exists(out_f):
                    print(f"[RT] skip existing {os.path.basename(out_f)}", flush=True)
                    continue
                trace_scene(sc, b, s, args.out_dir)


if __name__ == "__main__":
    main()
