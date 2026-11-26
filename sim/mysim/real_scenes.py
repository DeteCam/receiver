"""More realistic rooms (plan sim/plan_20261126_real.md).  Written on 2026-11-26 BEFORE any result on these rooms.

What is new compared with gen_scenes.py / fresh_scenes.py (same SceneBuilder, materials and people model):
  * rooms that are not plain boxes: bathroom boxes with glass / tile partitions inside lodging rooms, an L-shaped
    pension room, a glass-walled seminar room with a corridor behind the glass, stalls with metal posts
  * unmapped metal the estimator is not told about: ceiling light fixtures, air-conditioner cassettes, radiators,
    window mullions, kitchen appliances, hand dryers, vending machine
  * concealed cameras: part of the cameras sit inside a small plastic-like case or behind a thin glass sheet
  * receiver spots a person can really use (no stand): wall hooks at about 2 m (W), tops of furniture (F), table or
    counter height (T), inside the bathroom (B).  Spots and the single / pair / four combinations are fixed here.
  * exam rooms with spaced seating (CSAT-like 24 desks, fixed long desks with one-seat gaps, glass-walled seminar room)

Set A (used for the improvement loop) and set B (opened only for the final check) are fixed below.
build_real(...) -> builder (call .finalize() for the Sionna scene), dims, spots, cams, cam_types, user_reflectors,
                   boxes, conceal
"""
import os
import sys
import zlib
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
import gen_scenes as G  # noqa: E402

SET_A = ["motel_a", "restroom_d", "fitting_a", "pension_a", "exam_s"]
SET_B = ["goshiwon_a", "studio_b", "restroom_e", "sauna_locker", "lecture_c", "seminar_g"]
SET_C = ["hotel_c", "guesthouse_d", "restroom_f", "pool_locker", "fitting_b"]     # second held-out set (plan section 9)
REAL_SCENES = SET_A + SET_B + SET_C
EXAM = ("exam_s", "lecture_c", "seminar_g")
KIND = {"motel_a": "lodging", "pension_a": "lodging", "goshiwon_a": "lodging", "studio_b": "lodging",
        "restroom_d": "restroom", "restroom_e": "restroom", "fitting_a": "locker", "sauna_locker": "locker",
        "exam_s": "exam", "lecture_c": "exam", "seminar_g": "exam",
        "hotel_c": "lodging", "guesthouse_d": "lodging", "restroom_f": "restroom", "pool_locker": "locker",
        "fitting_b": "locker"}
DIMS = {"motel_a": (4.2, 6.6, 2.4), "restroom_d": (5.6, 7.2, 2.6), "fitting_a": (7.0, 5.5, 2.6),
        "pension_a": (6.5, 5.5, 2.4), "exam_s": (9.0, 7.5, 2.7), "goshiwon_a": (2.3, 3.6, 2.3),
        "studio_b": (5.4, 4.6, 2.4), "restroom_e": (2.6, 4.4, 2.5), "sauna_locker": (11.0, 8.0, 2.8),
        "lecture_c": (12.0, 9.0, 3.2), "seminar_g": (7.2, 6.0, 2.7),
        "hotel_c": (3.4, 7.4, 2.5), "guesthouse_d": (4.4, 5.2, 2.5), "restroom_f": (4.6, 6.4, 2.6),
        "pool_locker": (8.4, 6.2, 2.7), "fitting_b": (5.0, 4.0, 2.6)}

# receiver spots: (label, x, y, z, yaw_deg of the head's +x, type).  z = centre of the ring.
# furniture tops: top + 0.12 m; wall hooks: 0.10 m off the wall.
SPOTS = {
    "motel_a": [("W_entr", 3.6, 0.10, 2.0, 90, "W"), ("W_win", 3.7, 6.50, 2.0, -90, "W"),
                ("W_bedwall", 4.10, 4.05, 2.0, 180, "W"), ("F_wardrobe", 3.9, 0.9, 2.22, 180, "F"),
                ("T_tvcab", 0.25, 4.05, 0.72, 0, "T"), ("T_table", 1.3, 5.6, 0.62, 0, "T"),
                ("T_bedside", 3.92, 2.92, 0.62, 180, "T"), ("B_bath", 0.85, 2.15, 2.0, -90, "B")],
    "restroom_d": [("W_entr", 3.5, 0.10, 2.0, 90, "W"), ("W_urinal", 0.10, 3.6, 2.0, 0, "W"),
                   ("F_stalltop", 4.07, 4.0, 2.14, 180, "F"), ("W_mirror", 4.9, 7.10, 2.2, -90, "W"),
                   ("T_sink", 2.6, 6.9, 0.97, -90, "T"), ("W_x0far", 0.10, 5.9, 2.0, 0, "W")],
    "fitting_a": [("F_island", 2.9, 2.7, 1.97, 0, "F"), ("W_xl", 6.90, 3.8, 2.0, 180, "W"),
                  ("W_x0", 0.10, 4.0, 2.0, 0, "W"), ("F_lock_y0", 2.8, 0.26, 1.97, 90, "F"),
                  ("F_lock_yw", 2.4, 5.24, 1.97, -90, "F"), ("T_vanity", 5.4, 5.25, 0.92, -90, "T"),
                  ("T_bench", 2.9, 1.4, 0.57, 0, "T")],
    "pension_a": [("W_x0", 0.10, 4.4, 2.0, 0, "W"), ("W_block", 4.20, 4.5, 2.0, 180, "W"),
                  ("W_xl", 6.40, 2.2, 2.0, 180, "W"), ("F_fridge", 0.4, 0.4, 1.92, 45, "F"),
                  ("F_wardrobe", 5.7, 0.3, 2.12, 90, "F"), ("T_table", 2.2, 2.2, 0.88, 0, "T"),
                  ("T_tvstand", 2.3, 5.25, 0.62, -90, "T")],
    "exam_s": [("W_frontL", 0.6, 0.10, 2.0, 90, "W"), ("W_frontR", 8.4, 0.10, 2.0, 90, "W"),
               ("W_backL", 0.6, 7.40, 2.0, -90, "W"), ("W_backR", 8.4, 7.40, 2.0, -90, "W"),
               ("T_podium", 4.5, 0.95, 1.17, 90, "T"), ("T_teacher", 1.2, 0.8, 0.87, 90, "T"),
               ("F_lockers", 4.5, 7.27, 1.32, -90, "F"), ("T_sill", 0.30, 3.75, 1.02, 0, "T")],
    "goshiwon_a": [("W_door", 0.6, 0.10, 1.9, 90, "W"), ("W_xl", 2.20, 0.5, 1.95, 180, "W"),
                   ("W_x0", 0.10, 1.0, 1.9, 0, "W"), ("T_desk", 1.85, 2.8, 0.86, 180, "T"),
                   ("T_fridge", 2.0, 1.85, 0.97, 180, "T"), ("T_bed", 0.5, 2.5, 0.57, 0, "T")],
    "studio_b": [("W_entr", 1.5, 0.10, 2.0, 90, "W"), ("W_x0", 0.10, 3.6, 2.0, 0, "W"),
                 ("W_xl", 5.30, 4.4, 2.0, 180, "W"), ("F_fridge", 0.33, 2.38, 1.82, 0, "F"),
                 ("T_desk", 4.6, 4.3, 0.88, -90, "T"), ("T_bed", 4.3, 3.0, 0.62, 180, "T"),
                 ("B_bath", 4.5, 1.0, 1.4, 180, "B")],
    "restroom_e": [("W_door", 2.1, 0.10, 2.0, 90, "W"), ("W_x0", 0.10, 2.1, 2.0, 0, "W"),
                   ("F_cabinet", 2.34, 2.25, 1.92, 180, "F"), ("F_stalltop", 0.65, 2.60, 2.14, 90, "F"),
                   ("T_sink", 0.25, 1.0, 0.97, 0, "T")],
    "sauna_locker": [("F_island1", 2.95, 4.0, 1.92, 0, "F"), ("F_island2", 5.65, 4.0, 1.92, 0, "F"),
                     ("W_xl", 10.90, 6.5, 2.1, 180, "W"), ("W_y0", 4.5, 0.10, 2.1, 90, "W"),
                     ("F_lock_x0", 0.26, 4.0, 1.92, 0, "F"), ("T_vanity", 8.8, 7.72, 0.92, -90, "T"),
                     ("T_platform", 8.5, 3.5, 0.52, 0, "T"), ("T_counter", 1.5, 7.5, 1.12, -90, "T")],
    "lecture_c": [("W_frontL", 0.10, 0.6, 2.0, 0, "W"), ("W_frontR", 0.10, 8.4, 2.0, 0, "W"),
                  ("W_backL", 11.90, 0.6, 2.6, 180, "W"), ("W_backR", 11.90, 8.4, 2.6, 180, "W"),
                  ("T_lectern", 1.3, 6.9, 1.22, 0, "T"), ("W_side", 6.0, 0.10, 2.3, 90, "W")],
    "seminar_g": [("W_frontL", 0.10, 0.5, 2.0, 0, "W"), ("W_frontR", 0.10, 5.5, 2.0, 0, "W"),
                  ("W_backR", 7.10, 5.4, 2.0, 180, "W"), ("W_backL", 7.10, 0.6, 2.0, 180, "W"),
                  ("T_teacher", 0.9, 3.0, 0.86, 0, "T"), ("T_tvcart", 0.8, 5.25, 1.82, 0, "F")],
    # second held-out set: eight spots per room.  type S = on a hook inside a stall / booth
    "hotel_c": [("W_entr", 2.0, 0.10, 2.0, 90, "W"), ("W_win", 0.3, 7.30, 2.0, -90, "W"),
                ("W_x0", 0.10, 6.3, 2.0, 0, "W"), ("F_wardrobe", 3.1, 1.2, 2.22, 180, "F"),
                ("T_desk", 3.1, 3.5, 0.88, 180, "T"), ("T_night", 0.28, 4.2, 0.67, 0, "T"),
                ("W_xl", 3.30, 6.4, 2.0, 180, "W"), ("B_bath", 0.9, 2.25, 2.0, -90, "B")],
    "guesthouse_d": [("W_door", 1.2, 0.10, 2.0, 90, "W"), ("W_win", 3.7, 5.10, 2.0, -90, "W"),
                     ("W_xl", 4.30, 3.5, 2.0, 180, "W"), ("F_locker", 4.14, 1.6, 1.92, 180, "F"),
                     ("T_table", 2.8, 4.1, 0.88, 0, "T"), ("T_shoe", 3.1, 0.19, 1.02, 90, "T"),
                     ("F_fridge", 4.09, 4.8, 1.32, 180, "F"), ("W_y0b", 3.9, 0.10, 2.1, 90, "W")],
    "restroom_f": [("W_entr", 2.4, 0.10, 2.0, 90, "W"), ("W_far", 1.6, 6.30, 2.0, -90, "W"),
                   ("W_x0", 0.10, 1.0, 2.0, 0, "W"), ("F_stalltop", 3.10, 3.5, 2.17, 180, "F"),
                   ("T_sink", 0.3, 3.4, 0.97, 0, "T"), ("W_x0far", 0.10, 5.2, 2.0, 0, "W"),
                   ("S_stall1", 3.25, 2.07, 1.7, 0, "S"), ("S_stall3", 3.25, 3.97, 1.7, 0, "S")],
    "pool_locker": [("F_island", 3.45, 3.1, 1.92, 0, "F"), ("W_y0", 5.8, 0.10, 2.1, 90, "W"),
                    ("W_xl", 8.30, 4.9, 2.1, 180, "W"), ("F_lock_x0", 0.26, 3.1, 1.92, 0, "F"),
                    ("F_lock_n", 6.3, 5.94, 1.92, -90, "F"), ("T_vanity", 8.1, 2.1, 0.92, 180, "T"),
                    ("T_bench", 1.7, 3.0, 0.58, 0, "T"), ("W_y0b", 2.6, 0.10, 2.1, 90, "W")],
    "fitting_b": [("W_y0", 3.6, 0.10, 2.0, 90, "W"), ("W_x0", 0.10, 2.3, 2.1, 0, "W"),
                  ("F_header", 2.6, 2.71, 2.22, -90, "F"), ("W_xl", 4.90, 2.3, 2.1, 180, "W"),
                  ("T_bench", 2.5, 1.1, 0.58, 90, "T"), ("T_counter", 0.65, 0.28, 1.07, 90, "T"),
                  ("S_booth1", 1.4, 2.85, 1.8, 90, "S"), ("S_booth3", 3.8, 2.85, 1.8, 90, "S")],
}
# pre-registered combinations (indices into SPOTS[scene]); "low" = table-height only (what happens if nobody
# bothers to put the unit high)
COMBO = {
    "motel_a": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4, 5, 6)),
    "restroom_d": dict(single=0, pair=(0, 3), four=(0, 1, 2, 3), low=(4,)),
    "fitting_a": dict(single=0, pair=(1, 2), four=(0, 1, 2, 3), low=(5, 6)),
    "pension_a": dict(single=0, pair=(0, 2), four=(0, 1, 2, 3), low=(5, 6)),
    "exam_s": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4, 5, 7)),
    "goshiwon_a": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(3, 4, 5)),
    "studio_b": dict(single=0, pair=(0, 2), four=(0, 1, 2, 3), low=(4, 5)),
    "restroom_e": dict(single=0, pair=(0, 2), four=(0, 1, 2, 3), low=(4,)),
    "sauna_locker": dict(single=0, pair=(0, 2), four=(0, 1, 2, 3), low=(5, 6, 7)),
    "lecture_c": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4,)),
    "seminar_g": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4,)),
    "hotel_c": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4, 5)),
    "guesthouse_d": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4, 5)),
    "restroom_f": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4,)),
    "pool_locker": dict(single=0, pair=(0, 2), four=(0, 1, 2, 3), low=(5, 6)),
    "fitting_b": dict(single=0, pair=(0, 1), four=(0, 1, 2, 3), low=(4, 5)),
}


def crc(*a):
    return zlib.crc32(repr(a).encode()) & 0xFFFFFFFF


def shell_mixed(b, L, W, H, x0="concrete", xl="concrete", y0="concrete", yw="concrete", floor="concrete",
                ceil="ceiling_board", t=0.15):
    b.box((-t, -t, -t), (L + t, W + t, 0.0), floor, "floor")
    b.box((-t, -t, H), (L + t, W + t, H + t), ceil, "ceil")
    b.box((-t, -t, 0.0), (0.0, W + t, H), x0, "wx0")
    b.box((L, -t, 0.0), (L + t, W + t, H), xl, "wxl")
    b.box((0.0, -t, 0.0), (L, 0.0, H), y0, "wy0")
    b.box((0.0, W, 0.0), (L, W + t, H), yw, "wyw")


def lights(b, L, W, H, nx, ny, sx=1.2, sy=0.3):
    """Unmapped metal ceiling light fixtures."""
    for i in range(nx):
        for j in range(ny):
            cx, cy = L * (i + 0.5) / nx, W * (j + 0.5) / ny
            b.box((cx - sx / 2, cy - sy / 2, H - 0.07), (cx + sx / 2, cy + sy / 2, H - 0.01), "metal", "light")


class _People:
    """People draws always consume the random stream; boxes are added only when on (keeps cameras identical)."""
    def __init__(self, b, on):
        self.b, self.on = b, on

    def standing(self, x, y, z0=0.0):
        if self.on:
            self.b.add_person_standing(x, y, z0)

    def seated(self, sx, sy, z0=0.0, lean=0.0, facing="y-"):
        if self.on:
            self.b.add_person_seated(sx, sy, z0=z0, lean=lean, facing_axis=facing)


SPOT_CLEAR = 0.35   # no camera is drawn closer than this to a receiver spot (the unit would be touching the object,
                    # and the plane-wave antenna model does not hold at that distance)


def _sample(rng, b, dims, gens, n=40, spots=()):
    """gens: list of (weight, type, fn(rng) -> xyz).  Rejects points inside objects, outside the room, or within
    SPOT_CLEAR of a receiver spot."""
    L, W, H = dims
    sp = np.array([s[1:4] for s in spots], dtype=np.float64).reshape(-1, 3)
    w = np.array([g[0] for g in gens], dtype=np.float64)
    w /= w.sum()
    cams, types = [], []
    tries = 0
    solid = [bx for bx in b.boxes if bx[2] != "body"]     # people never decide where a camera is (same cameras in every variant)
    while len(cams) < n and tries < 20000:
        tries += 1
        k = int(rng.choice(len(gens), p=w))
        p = np.asarray(gens[k][2](rng), dtype=np.float64)
        if not (0.03 <= p[0] <= L - 0.03 and 0.03 <= p[1] <= W - 0.03 and 0.04 <= p[2] <= H - 0.03):
            continue
        if G._inside_any(p, solid):
            continue
        if len(sp) and np.min(np.linalg.norm(sp - p[None], axis=1)) < SPOT_CLEAR:
            continue
        cams.append(p.tolist())
        types.append(gens[k][1])
    assert len(cams) == n, (len(cams), n)
    return cams, types


def _conceal(rng, b, dims, cams, p_case=0.35, p_glass=0.10):
    """Hide part of the cameras: 'case' = inside a 10 cm plastic-like (chipboard) box, 'glass' = a thin glass sheet
    3 cm in front of the lens side that faces the room centre.  Returns the list of concealment labels."""
    L, W, H = dims
    out = []
    ctr = np.array([L / 2, W / 2, H / 2])
    for p in cams:
        u = rng.uniform()
        p = np.asarray(p)
        if u < p_case:
            h = 0.05
            b.box(p - h, p + h, "chipboard", "camcase")
            out.append("case")
        elif u < p_case + p_glass:
            d = ctr - p
            ax = int(np.argmax(np.abs(d[:2])))
            s = np.sign(d[ax]) if d[ax] != 0 else 1.0
            lo, hi = p - 0.12, p + 0.12
            lo[ax] = p[ax] + s * 0.03 - 0.003
            hi[ax] = p[ax] + s * 0.03 + 0.003
            b.box(lo, hi, "glass", "camglass")
            out.append("glass")
        else:
            out.append("open")
    return out


def _wallpt(rng, L, W, zlo, zhi, off=0.06, walls=(0, 1, 2, 3), xr=None, yr=None):
    k = int(rng.choice(list(walls)))
    z = rng.uniform(zlo, zhi)
    xr = xr or (0.2, L - 0.2)
    yr = yr or (0.2, W - 0.2)
    return [[off, rng.uniform(*yr), z], [L - off, rng.uniform(*yr), z], [rng.uniform(*xr), off, z],
            [rng.uniform(*xr), W - off, z]][k]


def build_real(scene_id, fc, rng, mesh_dir, variant="base"):
    """variant: base (people as designed) | nop (nobody) | op1 (nobody but one operator standing in the room)."""
    b = G.SceneBuilder(fc, mesh_dir)
    ppl = _People(b, variant == "base")
    refl = []
    L, W, H = DIMS[scene_id]
    U = lambda a, c: rng.uniform(a, c)  # noqa: E731

    if scene_id == "motel_a":
        shell_mixed(b, L, W, H, y0="plasterboard")
        # glass-walled bathroom in the entrance corner
        b.box((1.69, 0.0, 0.0), (1.71, 2.3, 2.2), "glass", "bath_glass")
        b.box((0.0, 2.28, 0.0), (1.71, 2.33, H), "plasterboard", "bath_wall")
        b.box((0.0, 0.5, 1.0), (0.02, 1.5, 1.9), "metal", "bath_mirror")
        b.box((0.02, 0.5, 0.0), (0.5, 1.5, 0.85), "tile", "sinks")
        b.box((0.2, 1.7, 0.0), (0.7, 2.2, 0.45), "tile", "toilet")
        b.box((1.55, 0.1, 1.0), (1.62, 0.2, 2.1), "metal", "shower")
        b.box((2.2, 0.0, 0.0), (3.1, 0.04, 2.05), "metal", "door")
        b.box((L - 0.6, 0.2, 0.0), (L - 0.02, 1.6, 2.1), "wood", "wardrobe")
        b.box((L - 0.62, 0.3, 0.3), (L - 0.60, 1.1, 1.9), "metal", "mirror")
        refl.append((0, L - 0.62, [L - 0.62, 0.3, 0.3], [L - 0.60, 1.1, 1.9], 0.92))
        b.box((L - 2.15, 3.2, 0.0), (L - 0.15, 4.9, 0.55), "wood", "bed")
        b.box((L - 0.5, 2.7, 0.0), (L - 0.05, 3.15, 0.5), "wood", "bedside")
        b.box((L - 0.5, 4.95, 0.0), (L - 0.05, 5.4, 0.5), "wood", "bedside")
        b.box((0.03, 3.4, 1.0), (0.09, 4.7, 1.75), "metal", "tv")
        b.box((0.02, 3.2, 0.0), (0.45, 4.9, 0.6), "wood", "tvcab")
        b.box((0.05, 5.0, 0.0), (0.55, 5.5, 0.85), "metal", "fridge")
        b.box((0.02, 2.45, 0.72), (0.5, 3.1, 0.76), "wood", "vanity")
        b.box((0.0, 2.5, 0.9), (0.02, 3.05, 1.7), "metal", "vanity_mirror")
        b.box((0.9, W - 0.03, 0.9), (3.3, W, 2.1), "glass", "window")
        b.box((0.85, W - 0.05, 0.85), (3.35, W - 0.02, 0.92), "metal", "winframe")
        b.box((2.6, W - 0.25, 2.05), (3.5, W - 0.03, 2.33), "metal", "aircon")
        b.box((1.0, 5.3, 0.45), (1.6, 5.9, 0.5), "wood", "table")
        b.box((0.1, 2.6, 0.77), (0.22, 2.72, 0.95), "metal", "kettle")
        ppl.standing(U(2.0, 3.0), U(1.0, 2.0))
        ppl.standing(U(1.0, 1.8), U(3.6, 4.6))
        gens = [
            (15, "smoke_alarm", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.05]),
            (12, "tv_box", lambda r: [r.uniform(0.10, 0.42), r.uniform(3.25, 4.85), r.uniform(0.62, 0.95)]),
            (15, "charger", lambda r: [L - r.uniform(0.07, 0.45), r.choice([2.92, 5.18]) + r.uniform(-0.2, 0.2), r.uniform(0.52, 0.75)]),
            (12, "wall_clock", lambda r: _wallpt(r, L, W, 1.7, 2.1, yr=(2.5, W - 0.3))),
            (8, "aircon", lambda r: [r.uniform(2.5, 3.6), W - r.uniform(0.27, 0.34), r.uniform(1.95, 2.3)]),
            (10, "table_top", lambda r: [r.uniform(1.05, 1.55), r.uniform(5.35, 5.85), r.uniform(0.52, 0.65)]),
            (8, "mirror_edge", lambda r: [0.05, r.choice([2.45, 3.1]) + r.uniform(-0.03, 0.03), r.uniform(0.9, 1.7)]),
            (10, "bath_vent", lambda r: [r.uniform(0.2, 1.5), r.uniform(0.2, 2.1), r.uniform(2.0, 2.35)]),
            (5, "bath_shelf", lambda r: [r.uniform(0.06, 0.45), r.uniform(0.3, 2.2), r.uniform(1.2, 1.6)]),
            (5, "wardrobe_top", lambda r: [L - r.uniform(0.1, 0.5), r.uniform(0.3, 1.5), r.uniform(2.12, 2.2)]),
        ]

    elif scene_id == "restroom_d":
        shell_mixed(b, L, W, H, x0="tile", xl="tile", y0="tile", yw="tile", floor="tile")
        b.box((0.2, 0.0, 0.0), (1.1, 0.04, 2.05), "metal", "door")
        b.box((0.0, 1.5, 0.0), (1.6, 1.6, H), "tile", "baffle")
        ys = [1.2 + 0.95 * k for k in range(7)]
        for yk in ys:
            b.box((L - 1.5, yk - 0.012, 0.15), (L, yk + 0.012, 2.0), "chipboard", "stall")
            b.box((L - 1.55, yk - 0.02, 0.0), (L - 1.49, yk + 0.02, 2.05), "metal", "stallpost")
        b.box((L - 1.53, ys[0], 0.15), (L - 1.50, ys[-1], 2.0), "chipboard", "stall_front")
        for yk in ys[:-1]:
            b.box((L - 0.7, yk + 0.25, 0.0), (L - 0.15, yk + 0.7, 0.42), "tile", "toilet")
        b.box((L - 0.10, 1.25, 0.0), (L - 0.02, 6.85, 0.9), "metal", "pipes")
        for k in range(4):
            b.box((0.0, 2.4 + 0.8 * k, 0.4), (0.35, 2.8 + 0.8 * k, 1.1), "tile", "urinal")
        for k in range(5):
            b.box((0.0, 2.3 + 0.8 * k, 0.5), (0.45, 2.32 + 0.8 * k, 1.4), "metal", "urinal_div")
        b.box((1.2, W - 0.55, 0.0), (4.0, W, 0.85), "tile", "sinks")
        b.box((1.2, W - 0.02, 1.0), (4.0, W, 2.0), "metal", "mirror")
        refl.append((1, W - 0.02, [1.2, W - 0.02, 1.0], [4.0, W, 2.0], 0.92))
        b.box((0.02, 6.4, 1.0), (0.2, 6.75, 1.35), "metal", "hand_dryer")
        b.box((4.1, 6.7, 0.0), (4.45, 7.05, 0.6), "metal", "bin")
        b.box((2.0, 3.0, H - 0.05), (2.4, 3.4, H - 0.01), "metal", "vent")
        b.box((2.0, 5.2, H - 0.05), (2.4, 5.6, H - 0.01), "metal", "vent")
        for _ in range(3):
            ppl.standing(U(1.8, 3.6), U(1.9, 6.2))
        st = lambda r: ys[int(r.integers(6))]  # noqa: E731
        gens = [
            (20, "stall_tp", lambda r: [L - r.uniform(0.9, 1.4), st(r) + r.choice([0.06, 0.89]), r.uniform(0.6, 0.8)]),
            (10, "stall_hook", lambda r: [L - 1.44, st(r) + r.uniform(0.3, 0.65), r.uniform(1.6, 1.8)]),
            (12, "stall_vent", lambda r: [L - r.uniform(0.2, 1.3), r.uniform(1.3, 6.8), H - 0.06]),
            (8, "stall_floor", lambda r: [L - r.uniform(0.12, 0.2), st(r) + r.uniform(0.75, 0.9), r.uniform(0.15, 0.35)]),
            (15, "ceiling_vent", lambda r: [r.uniform(0.3, L - 1.7), r.uniform(0.3, W - 0.3), H - 0.06]),
            (10, "mirror_frame", lambda r: [r.choice([1.12, 4.08]) + r.uniform(-0.03, 0.03), W - 0.06, r.uniform(1.0, 2.05)]),
            (5, "dryer", lambda r: [0.24, r.uniform(6.4, 6.75), r.uniform(1.0, 1.4)]),
            (10, "urinal_top", lambda r: [0.06, r.uniform(2.4, 5.6), r.uniform(1.45, 1.8)]),
            (5, "bin", lambda r: [r.uniform(4.12, 4.43), r.uniform(6.72, 7.03), r.uniform(0.62, 0.7)]),
            (5, "sink_top", lambda r: [r.uniform(1.3, 3.9), W - r.uniform(0.1, 0.5), r.uniform(0.87, 0.95)]),
        ]

    elif scene_id == "fitting_a":
        shell_mixed(b, L, W, H)
        b.box((0.6, 0.02, 0.0), (5.0, 0.5, 1.85), "metal", "lockers_s")
        refl.append((1, 0.5, [0.6, 0.02, 0.0], [5.0, 0.5, 1.85], 0.85))
        b.box((0.6, W - 0.5, 0.0), (4.2, W - 0.02, 1.85), "metal", "lockers_n")
        b.box((1.5, 2.3, 0.0), (4.3, 3.1, 1.85), "metal", "island_lockers")
        b.box((0.0, 1.2, 0.2), (0.02, 2.6, 2.0), "metal", "full_mirror")
        refl.append((0, 0.02, [0.0, 1.2, 0.2], [0.02, 2.6, 2.0], 0.92))
        b.box((4.6, W - 0.02, 0.9), (6.2, W, 1.9), "metal", "mirror")
        b.box((4.5, W - 0.5, 0.0), (6.3, W, 0.8), "wood", "vanity")
        b.box((4.7, W - 0.4, 0.82), (4.85, W - 0.25, 1.0), "metal", "dryer")
        b.box((5.9, W - 0.4, 0.82), (6.05, W - 0.25, 1.0), "metal", "dryer")
        b.box((5.2, 0.0, 0.0), (5.22, 2.2, 2.0), "glass", "shower_glass")
        b.box((5.2, 2.18, 0.0), (L, 2.22, 2.0), "glass", "shower_glass")
        b.box((L - 0.1, 0.5, 1.0), (L - 0.02, 0.6, 2.1), "metal", "shower")
        b.box((L - 0.1, 1.5, 1.0), (L - 0.02, 1.6, 2.1), "metal", "shower")
        b.box((2.0, 1.2, 0.0), (3.8, 1.6, 0.45), "wood", "bench")
        b.box((2.0, 3.9, 0.0), (3.8, 4.3, 0.45), "wood", "bench")
        b.box((6.3, 3.0, 0.0), (6.7, 3.4, 0.1), "metal", "scale")
        b.box((6.5, 4.4, 0.0), (6.9, 4.8, 1.3), "metal", "fan_dryer")
        lights(b, L, W, H, 3, 2)
        for _ in range(5):
            ppl.standing(U(0.6, 6.4), U(0.8, 4.9))
        rows = [((0.7, 4.9), (0.08, 0.44)), ((0.7, 4.1), (W - 0.44, W - 0.08)), ((1.6, 4.2), (2.36, 3.04))]
        def lock_top(r):
            (xa, xb), (ya, yb) = rows[int(r.integers(3))]
            return [r.uniform(xa, xb), r.uniform(ya, yb), r.uniform(1.87, 1.95)]
        gens = [
            (25, "locker_top", lock_top),
            (15, "ceiling_vent", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.09]),
            (12, "wall_clock", lambda r: _wallpt(r, L, W, 1.95, 2.3)),
            (10, "vanity", lambda r: [r.uniform(4.9, 5.85), W - r.uniform(0.1, 0.45), r.uniform(0.82, 1.0)]),
            (8, "bench_side", lambda r: [r.uniform(1.85, 3.95), r.choice([1.15, 1.65, 3.85, 4.35]), r.uniform(0.1, 0.4)]),
            (8, "mirror_frame", lambda r: [0.05, r.choice([1.15, 2.65]) + r.uniform(-0.03, 0.03), r.uniform(0.3, 2.0)]),
            (10, "shower_shelf", lambda r: [r.uniform(5.4, 6.8), r.uniform(0.15, 2.0), r.uniform(1.2, 1.9)]),
            (5, "fan", lambda r: [r.uniform(6.45, 6.95), r.uniform(4.35, 4.85), r.uniform(1.32, 1.4)]),
            (7, "locker_gap", lambda r: [r.choice([0.55, 5.05]), r.uniform(0.08, 0.45), r.uniform(0.5, 1.5)]),
        ]

    elif scene_id == "pension_a":
        shell_mixed(b, L, W, H)
        b.box((4.3, 3.5, 0.0), (L, W, H), "concrete", "block")          # L-shaped room: this corner is not part of it
        b.box((0.05, 0.05, 0.0), (0.75, 0.75, 1.8), "metal", "fridge")
        b.box((0.8, 0.02, 0.0), (3.0, 0.6, 0.86), "wood", "counter")
        b.box((1.4, 0.05, 0.86), (2.2, 0.55, 0.89), "metal", "sink")
        b.box((2.3, 0.02, 1.7), (2.9, 0.5, 1.9), "metal", "hood")
        b.box((0.8, 0.02, 1.5), (2.3, 0.35, 2.2), "wood", "cabinet_up")
        b.box((3.0, 0.02, 0.0), (3.6, 0.5, 0.9), "wood", "shelf")
        b.box((3.05, 0.05, 0.9), (3.5, 0.45, 1.2), "metal", "microwave")
        b.box((1.6, 1.8, 0.72), (2.8, 2.6, 0.76), "wood", "table")
        for tx, ty in ((1.65, 1.85), (2.71, 1.85), (1.65, 2.51), (2.71, 2.51)):
            b.box((tx, ty, 0.0), (tx + 0.04, ty + 0.04, 0.72), "metal", "tableleg")
        b.box((0.1, 3.6, 0.0), (0.9, 5.3, 0.75), "wood", "sofa")
        b.box((1.6, W - 0.08, 0.9), (3.0, W - 0.02, 1.7), "metal", "tv")
        b.box((1.5, W - 0.45, 0.0), (3.1, W - 0.02, 0.5), "wood", "tvstand")
        b.box((0.0, 1.2, 0.0), (0.02, 3.4, 2.1), "glass", "window")
        for yy in (1.2, 2.3, 3.4):
            b.box((0.0, yy - 0.03, 0.0), (0.05, yy + 0.03, 2.1), "metal", "winframe")
        b.box((4.6, 1.2, 0.0), (6.4, 3.2, 0.5), "wood", "bed")
        b.box((5.0, 0.02, 0.0), (6.48, 0.6, 2.0), "wood", "wardrobe")
        b.box((5.2, 0.6, 0.3), (6.0, 0.62, 1.9), "metal", "mirror")
        refl.append((1, 0.62, [5.2, 0.6, 0.3], [6.0, 0.62, 1.9], 0.92))
        b.box((4.8, 3.25, 2.0), (5.7, 3.48, 2.3), "metal", "aircon")
        ppl.standing(U(1.0, 3.8), U(1.0, 1.6))
        ppl.standing(U(1.2, 3.8), U(3.0, 4.6))
        gens = [
            (15, "smoke_alarm", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.05]),
            (12, "wall_clock", lambda r: _wallpt(r, L, W, 1.7, 2.1, xr=(0.3, 4.1))),
            (12, "tv_stand", lambda r: [r.uniform(1.55, 3.05), W - r.uniform(0.1, 0.4), r.uniform(0.52, 0.85)]),
            (12, "kitchen_shelf", lambda r: [r.uniform(0.85, 3.55), r.uniform(0.08, 0.5), r.uniform(0.91, 1.45)]),
            (8, "plant_floor", lambda r: [r.uniform(0.2, 4.0), r.uniform(0.9, 5.2), r.uniform(0.25, 0.5)]),
            (6, "aircon", lambda r: [r.uniform(4.75, 5.75), r.uniform(3.15, 3.22), r.uniform(1.95, 2.3)]),
            (10, "charger", lambda r: [r.uniform(4.35, 4.55), r.uniform(1.0, 3.3), r.uniform(0.3, 0.6)]),
            (8, "wardrobe_top", lambda r: [r.uniform(5.1, 6.4), r.uniform(0.1, 0.5), r.uniform(2.02, 2.1)]),
            (7, "sofa_side", lambda r: [r.uniform(0.95, 1.1), r.uniform(3.6, 5.3), r.uniform(0.3, 0.8)]),
            (10, "fridge_top", lambda r: [r.uniform(0.1, 0.7), r.uniform(0.1, 0.7), r.uniform(1.82, 1.9)]),
        ]

    elif scene_id == "exam_s":
        shell_mixed(b, L, W, H)
        b.box((2.7, 0.0, 0.9), (6.3, 0.03, 2.1), "metal", "blackboard")
        refl.append((1, 0.03, [2.7, 0.0, 0.9], [6.3, 0.03, 2.1], 0.90))
        b.box((6.7, 0.02, 1.2), (7.9, 0.08, 1.9), "metal", "tv")
        b.box((0.6, 0.5, 0.0), (1.8, 1.1, 0.75), "wood", "teacher_desk")
        b.box((3.9, 0.7, 0.0), (5.1, 1.2, 1.05), "wood", "podium")
        b.box((0.0, 1.0, 0.9), (0.02, 6.9, 2.3), "glass", "window")
        for yy in (1.0, 2.5, 4.0, 5.5, 6.9):
            b.box((0.0, yy - 0.025, 0.9), (0.04, yy + 0.025, 2.3), "metal", "winframe")
        b.box((0.05, 1.2, 0.1), (0.2, 6.6, 0.7), "metal", "radiator")
        b.box((L - 0.04, 0.4, 0.0), (L, 1.3, 2.1), "metal", "door")
        b.box((L - 0.04, 6.2, 0.0), (L, 7.1, 2.1), "metal", "door")
        b.box((L - 0.02, 1.6, 1.8), (L, 5.9, 2.4), "glass", "window")
        b.box((1.0, W - 0.45, 0.0), (8.0, W - 0.02, 1.2), "metal", "lockers")
        refl.append((1, W - 0.45, [1.0, W - 0.45, 0.0], [8.0, W - 0.02, 1.2], 0.85))
        lights(b, L, W, H, 2, 4)
        b.box((4.0, 3.3, H - 0.25), (4.9, 4.2, H - 0.02), "metal", "ac_cassette")
        ppl.standing(U(1.0, 8.0), U(1.45, 1.6))
        ppl.standing(U(8.3, 8.6), U(2.0, 5.8))
        seats = []
        for ry in np.linspace(2.3, 6.4, 6):
            for cx in (1.5, 3.5, 5.5, 7.5):
                sx, sy = cx + rng.normal(0, 0.04), ry + rng.normal(0, 0.04)
                lean = U(-0.02, 0.09)
                b.add_desk_metal_legs(sx - 0.325, sx + 0.325, sy - 0.62, sy - 0.17)
                ppl.seated(sx, sy, lean=lean, facing="y-")
                seats.append((sx, sy, 0.0, lean))
        cams, types = _exam_cams(rng, seats, "y-")
        gens = None

    elif scene_id == "goshiwon_a":
        shell_mixed(b, L, W, H, x0="plasterboard", xl="plasterboard", y0="plasterboard", yw="concrete")
        b.box((0.02, 1.5, 0.0), (0.95, 3.55, 0.45), "wood", "bed")
        b.box((1.4, 2.2, 0.70), (2.28, 3.4, 0.74), "wood", "desk_top")
        b.box((1.42, 2.22, 0.0), (1.46, 2.26, 0.70), "metal", "deskleg")
        b.box((1.42, 3.34, 0.0), (1.46, 3.38, 0.70), "metal", "deskleg")
        b.box((1.6, 2.2, 1.30), (2.28, 3.4, 1.33), "wood", "shelf")
        b.box((1.6, 2.2, 1.70), (2.28, 3.4, 1.73), "wood", "shelf")
        b.box((1.75, 1.6, 0.0), (2.25, 2.1, 0.85), "metal", "fridge")
        b.box((2.22, 0.9, 1.3), (2.28, 1.5, 1.7), "metal", "tv")
        b.box((0.05, 0.2, 1.8), (1.0, 0.24, 1.84), "metal", "rail")
        b.box((0.5, W - 0.02, 1.0), (1.7, W, 1.9), "glass", "window")
        b.box((0.45, W - 0.05, 0.95), (1.75, W - 0.02, 1.0), "metal", "winframe")
        b.box((1.3, 0.0, 0.0), (2.1, 0.04, 2.0), "wood", "door")
        b.box((1.0, 2.5, 0.42), (1.4, 2.9, 0.45), "wood", "chair")
        ppl.standing(U(1.1, 1.3), U(0.8, 1.3))
        gens = [
            (25, "desk_shelf", lambda r: [r.uniform(1.65, 2.2), r.uniform(2.3, 3.3), r.choice([r.uniform(0.76, 1.0), r.uniform(1.35, 1.65)])]),
            (15, "smoke_alarm", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.05]),
            (15, "charger", lambda r: _wallpt(r, L, W, 0.25, 0.5, yr=(0.3, 1.4))),
            (15, "wall_hook", lambda r: _wallpt(r, L, W, 1.6, 2.0)),
            (10, "tv", lambda r: [2.16, r.uniform(0.85, 1.55), r.uniform(1.2, 1.8)]),
            (10, "fridge_top", lambda r: [r.uniform(1.8, 2.2), r.uniform(1.65, 2.05), r.uniform(0.87, 0.95)]),
            (10, "rail", lambda r: [r.uniform(0.1, 0.95), r.uniform(0.27, 0.35), r.uniform(1.75, 1.9)]),
        ]

    elif scene_id == "studio_b":
        shell_mixed(b, L, W, H)
        # bathroom box in the corner x in [3.7, 5.4], y in [0, 1.9]: tile walls, wooden door
        b.box((3.65, 0.0, 0.0), (3.75, 0.9, H), "tile", "bath_wall")
        b.box((3.67, 0.9, 0.0), (3.71, 1.7, 2.05), "wood", "bath_door")
        b.box((3.65, 0.9, 2.05), (3.75, 1.7, H), "tile", "bath_wall")
        b.box((3.65, 1.7, 0.0), (3.75, 1.95, H), "tile", "bath_wall")
        b.box((3.75, 1.85, 0.0), (L, 1.95, H), "tile", "bath_wall")
        b.box((L - 0.02, 0.4, 1.0), (L, 1.2, 1.8), "metal", "bath_mirror")
        b.box((L - 0.5, 0.4, 0.0), (L - 0.02, 1.2, 0.85), "tile", "sinks")
        b.box((3.9, 0.1, 0.0), (4.4, 0.6, 0.45), "tile", "toilet")
        b.box((2.4, 0.0, 0.0), (3.3, 0.04, 2.05), "metal", "door")
        b.box((0.02, 0.3, 0.0), (0.6, 2.0, 0.86), "wood", "counter")
        b.box((0.05, 0.9, 0.86), (0.55, 1.5, 0.89), "metal", "sink")
        b.box((0.02, 2.05, 0.0), (0.65, 2.7, 1.7), "metal", "fridge")
        b.box((0.02, 0.3, 1.5), (0.35, 2.0, 2.2), "wood", "cabinet_up")
        b.box((0.6, W - 0.6, 0.0), (3.0, W - 0.02, 2.3), "wood", "wardrobe")
        b.box((0.7, W - 0.62, 0.1), (2.9, W - 0.60, 2.2), "metal", "mirror")
        refl.append((1, W - 0.62, [0.7, W - 0.62, 0.1], [2.9, W - 0.60, 2.2], 0.92))
        b.box((3.3, 2.3, 0.0), (5.3, 3.8, 0.5), "wood", "bed")
        b.box((3.4, W - 0.6, 0.72), (4.9, W - 0.02, 0.76), "wood", "desk_top")
        b.box((3.9, W - 0.2, 0.85), (4.5, W - 0.15, 1.25), "metal", "monitor")
        b.box((L - 0.02, 2.3, 0.9), (L, 4.2, 2.1), "glass", "window")
        b.box((L - 0.05, 2.25, 0.85), (L - 0.02, 4.25, 0.92), "metal", "winframe")
        b.box((L - 0.25, 2.6, 2.05), (L - 0.02, 3.5, 2.33), "metal", "aircon")
        ppl.standing(U(1.2, 3.0), U(1.0, 3.4))
        gens = [
            (15, "smoke_alarm", lambda r: [r.uniform(0.3, 3.5), r.uniform(0.3, W - 0.8), H - 0.05]),
            (10, "wall_clock", lambda r: _wallpt(r, L, W, 1.7, 2.1, walls=(0, 2), xr=(0.8, 2.3), yr=(2.8, 3.9))),
            (15, "charger", lambda r: [L - r.uniform(0.06, 0.12), r.uniform(2.3, 4.2), r.uniform(0.55, 0.8)]),
            (12, "desk", lambda r: [r.uniform(3.5, 4.8), W - r.uniform(0.25, 0.55), r.uniform(0.78, 1.0)]),
            (8, "mirror_edge", lambda r: [r.choice([0.66, 2.94]) + r.uniform(-0.02, 0.02), W - 0.66, r.uniform(0.5, 2.1)]),
            (10, "kitchen_shelf", lambda r: [r.uniform(0.08, 0.5), r.uniform(0.35, 1.95), r.uniform(0.91, 1.45)]),
            (8, "fridge_top", lambda r: [r.uniform(0.08, 0.6), r.uniform(2.1, 2.65), r.uniform(1.72, 1.85)]),
            (7, "aircon", lambda r: [L - r.uniform(0.27, 0.34), r.uniform(2.55, 3.55), r.uniform(1.95, 2.3)]),
            (10, "bath_vent", lambda r: [r.uniform(3.9, 5.2), r.uniform(0.2, 1.7), r.uniform(2.1, 2.35)]),
            (5, "bath_shelf", lambda r: [r.uniform(3.8, 5.3), r.uniform(1.72, 1.8), r.uniform(1.2, 1.6)]),
        ]

    elif scene_id == "restroom_e":
        shell_mixed(b, L, W, H, x0="tile", xl="tile", y0="tile", yw="tile", floor="tile")
        b.box((0.9, 0.0, 0.0), (1.7, 0.04, 2.05), "metal", "door")
        b.box((1.29, 2.6, 0.15), (1.31, W, 2.0), "metal", "stall")
        b.box((0.0, 2.58, 0.15), (L, 2.62, 2.0), "metal", "stall_front")
        b.box((0.4, 3.7, 0.0), (0.9, 4.3, 0.42), "tile", "toilet")
        b.box((1.7, 3.7, 0.0), (2.2, 4.3, 0.42), "tile", "toilet")
        b.box((L - 0.35, 1.4, 0.4), (L, 1.8, 1.1), "tile", "urinal")
        b.box((0.0, 0.6, 0.0), (0.5, 1.4, 0.85), "tile", "sinks")
        b.box((0.0, 0.6, 1.0), (0.02, 1.4, 1.8), "metal", "mirror")
        refl.append((0, 0.02, [0.0, 0.6, 1.0], [0.02, 1.4, 1.8], 0.92))
        b.box((L - 0.4, 0.2, 1.6), (L - 0.02, 0.6, 2.2), "metal", "heater")
        b.box((0.8, W - 0.02, 1.6), (1.8, W, 2.1), "glass", "window")
        b.box((L - 0.5, 2.0, 0.0), (L - 0.02, 2.5, 1.8), "metal", "cabinets")
        b.box((1.1, 1.4, H - 0.05), (1.5, 1.8, H - 0.01), "metal", "vent")
        ppl.standing(U(0.9, 1.6), U(0.9, 2.2))
        gens = [
            (20, "stall_tp", lambda r: [r.choice([0.06, 1.24, 1.36, L - 0.06]), r.uniform(3.0, 3.9), r.uniform(0.6, 0.8)]),
            (10, "stall_hook", lambda r: [r.uniform(0.2, 2.4), 2.68, r.uniform(1.6, 1.8)]),
            (10, "stall_floor", lambda r: [r.choice([r.uniform(0.2, 0.35), r.uniform(2.25, 2.4)]), r.uniform(4.2, 4.33), r.uniform(0.15, 0.35)]),
            (20, "ceiling_vent", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.07]),
            (8, "heater", lambda r: [L - r.uniform(0.44, 0.5), r.uniform(0.2, 0.6), r.uniform(1.6, 2.2)]),
            (10, "mirror_frame", lambda r: [0.05, r.choice([0.55, 1.45]) + r.uniform(-0.03, 0.03), r.uniform(1.0, 1.8)]),
            (10, "cabinet_top", lambda r: [L - r.uniform(0.08, 0.45), r.uniform(2.05, 2.45), r.uniform(1.82, 1.9)]),
            (12, "window_sill", lambda r: [r.uniform(0.85, 1.75), W - 0.06, r.uniform(1.5, 1.58)]),
        ]

    elif scene_id == "sauna_locker":
        shell_mixed(b, L, W, H)
        b.box((0.02, 1.0, 0.0), (0.5, 7.0, 1.8), "metal", "lockers")
        refl.append((0, 0.5, [0.02, 1.0, 0.0], [0.5, 7.0, 1.8], 0.85))
        b.box((2.5, 1.5, 0.0), (3.4, 6.5, 1.8), "metal", "island_lockers")
        b.box((5.2, 1.5, 0.0), (6.1, 6.5, 1.8), "metal", "island_lockers")
        b.box((7.0, 0.02, 0.0), (10.5, 0.5, 1.8), "metal", "lockers_s")
        b.box((7.3, 2.5, 0.0), (9.8, 4.5, 0.4), "wood", "platform")
        b.box((7.0, W - 0.55, 0.0), (10.6, W, 0.8), "wood", "vanity")
        b.box((7.0, W - 0.02, 0.9), (10.6, W, 1.9), "metal", "mirror")
        refl.append((1, W - 0.02, [7.0, W - 0.02, 0.9], [10.6, W, 1.9], 0.92))
        b.box((10.3, 5.0, 0.0), (10.95, 5.8, 1.85), "metal", "vending")
        b.box((6.6, 7.0, 0.0), (6.95, 7.4, 0.1), "metal", "scale")
        b.box((0.8, 7.2, 0.0), (2.2, 7.8, 1.0), "wood", "counter")
        b.box((L - 0.08, 2.8, 1.6), (L - 0.02, 4.2, 2.3), "metal", "tv")
        b.box((6.5, 0.6, 0.0), (6.9, 1.0, 1.3), "metal", "fan_dryer")
        b.box((4.0, 7.3, 0.0), (4.4, 7.7, 1.3), "metal", "fan_dryer")
        lights(b, L, W, H, 4, 3)
        for _ in range(8):
            ppl.standing(U(0.8, 10.2), U(0.8, 7.0))
        rows = [((0.08, 0.44), (1.1, 6.9)), ((2.56, 3.34), (1.6, 6.4)), ((5.26, 6.04), (1.6, 6.4)), ((7.1, 10.4), (0.08, 0.44))]
        def lock_top(r):
            (xa, xb), (ya, yb) = rows[int(r.integers(4))]
            return [r.uniform(xa, xb), r.uniform(ya, yb), r.uniform(1.82, 1.9)]
        gens = [
            (25, "locker_top", lock_top),
            (15, "ceiling_vent", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.09]),
            (10, "wall_clock", lambda r: _wallpt(r, L, W, 1.95, 2.4)),
            (10, "vanity", lambda r: [r.uniform(7.1, 10.5), W - r.uniform(0.1, 0.5), r.uniform(0.82, 1.0)]),
            (5, "vending_side", lambda r: [10.24, r.uniform(5.0, 5.8), r.uniform(0.5, 1.8)]),
            (5, "platform_edge", lambda r: [r.choice([7.24, 9.86]), r.uniform(2.5, 4.5), r.uniform(0.1, 0.35)]),
            (5, "tv", lambda r: [L - 0.14, r.uniform(2.75, 4.25), r.uniform(1.5, 2.35)]),
            (5, "fan", lambda r: [r.uniform(6.45, 6.95), r.uniform(0.55, 1.05), r.uniform(1.32, 1.4)]),
            (5, "counter", lambda r: [r.uniform(0.9, 2.1), r.uniform(7.25, 7.75), r.uniform(1.02, 1.15)]),
            (15, "locker_gap", lambda r: [r.choice([2.56, 3.34, 5.26, 6.04]), r.choice([1.44, 6.56]), r.uniform(0.5, 1.7)]),
        ]

    elif scene_id == "lecture_c":
        shell_mixed(b, L, W, H)
        b.box((0.0, 2.1, 0.9), (0.03, 6.9, 2.1), "metal", "whiteboard")
        refl.append((0, 0.03, [0.0, 2.1, 0.9], [0.03, 6.9, 2.1], 0.90))
        b.box((3.5, 4.3, H - 0.5), (3.9, 4.7, H - 0.2), "metal", "projector")
        b.box((1.0, 6.5, 0.0), (1.6, 7.3, 1.1), "wood", "podium")
        b.box((2.0, W - 0.02, 1.0), (11.0, W, 2.4), "glass", "window")
        for xx in np.arange(2.0, 11.01, 1.5):
            b.box((xx - 0.025, W - 0.04, 1.0), (xx + 0.025, W, 2.4), "metal", "winframe")
        b.box((0.6, 0.0, 0.0), (1.6, 0.04, 2.1), "metal", "door")
        b.box((10.2, 0.0, 0.0), (11.2, 0.04, 2.9), "metal", "door")
        lights(b, L, W, H, 4, 3)
        row_x = [3.0, 4.2, 5.4, 6.6, 7.8, 9.0, 10.2]
        for k in range(2, 7):                     # gentle tiers from the third row on
            b.box((row_x[k] - 0.9, 0.0, 0.0), (L, W, 0.15 * (k - 1)), "concrete", "riser")
        ppl.standing(U(1.2, 2.2), U(2.0, 5.5))
        ppl.standing(U(4.0, 9.5), U(4.35, 4.65))
        seat_y = [1.2, 1.8, 2.4, 3.0, 3.6, 5.1, 5.7, 6.3, 6.9, 7.5]
        seats = []
        for k, rx in enumerate(row_x):
            z0 = 0.15 * (k - 1) if k >= 2 else 0.0
            for j, cy in enumerate(seat_y):
                sx, sy = rx + rng.normal(0, 0.02), cy + rng.normal(0, 0.03)
                lean = U(-0.02, 0.08)
                b.add_desk_metal_legs(rx - 0.62, rx - 0.22, cy - 0.30, cy + 0.30, z0=z0)   # one segment of the long desk
                if j % 2 == 0:                    # one-seat gaps
                    ppl.seated(sx, sy, z0=z0, lean=lean, facing="x-")
                    seats.append((sx, sy, z0, lean))
        cams, types = _exam_cams(rng, seats, "x-")
        gens = None

    elif scene_id == "seminar_g":
        shell_mixed(b, L, W, H, y0="glass")
        # corridor behind the glass wall
        b.box((-0.15, -1.95, -0.15), (L + 0.15, -0.15, 0.0), "concrete", "corr_floor")
        b.box((-0.15, -1.95, H), (L + 0.15, -0.15, H + 0.15), "ceiling_board", "corr_ceil")
        b.box((-0.15, -2.10, 0.0), (L + 0.15, -1.95, H), "concrete", "corr_wall")
        for xx in np.arange(0.0, L + 0.01, 1.2):
            b.box((xx - 0.025, 0.0, 0.0), (xx + 0.025, 0.05, H), "metal", "mullion")
        b.box((0.0, 1.2, 0.9), (0.03, 4.8, 2.1), "metal", "whiteboard")
        refl.append((0, 0.03, [0.0, 1.2, 0.9], [0.03, 4.8, 2.1], 0.90))
        b.box((0.6, 2.4, 0.71), (1.2, 3.6, 0.74), "wood", "teacher_desk")
        b.box((0.5, 4.9, 0.0), (1.1, 5.6, 1.7), "metal", "tvcart")
        b.box((L - 0.02, 1.5, 1.0), (L, 4.5, 2.2), "glass", "window")
        lights(b, L, W, H, 2, 3)
        ppl.standing(U(1.4, 1.7), U(0.8, 2.0))
        seats = []
        for rx in (2.2, 3.5, 4.8, 6.1):
            for cy in (0.9, 1.95, 3.0, 4.05, 5.1):
                sx, sy = rx + rng.normal(0, 0.04), cy + rng.normal(0, 0.04)
                lean = U(-0.02, 0.09)
                b.add_desk_metal_legs(sx - 0.62, sx - 0.17, sy - 0.30, sy + 0.30)
                ppl.seated(sx, sy, lean=lean, facing="x-")
                seats.append((sx, sy, 0.0, lean))
        cams, types = _exam_cams(rng, seats, "x-")
        gens = None

    elif scene_id == "hotel_c":
        shell_mixed(b, L, W, H, x0="plasterboard", xl="plasterboard")
        # bathroom box x in [0, 1.8], y in [0, 2.4]: tile walls, wooden door on the corridor side
        b.box((1.75, 0.0, 0.0), (1.85, 0.5, H), "tile", "bath_wall")
        b.box((1.78, 0.5, 0.0), (1.82, 1.3, 2.05), "wood", "bath_door")
        b.box((1.75, 0.5, 2.05), (1.85, 1.3, H), "tile", "bath_wall")
        b.box((1.75, 1.3, 0.0), (1.85, 2.45, H), "tile", "bath_wall")
        b.box((0.0, 2.35, 0.0), (1.75, 2.45, H), "tile", "bath_wall")
        b.box((0.02, 0.5, 1.0), (0.04, 1.4, 1.9), "metal", "bath_mirror")
        b.box((0.02, 0.5, 0.0), (0.5, 1.4, 0.85), "tile", "sinks")
        b.box((0.6, 1.8, 0.0), (1.1, 2.3, 0.42), "tile", "toilet")
        b.box((1.2, 0.9, H - 0.05), (1.5, 1.2, H - 0.01), "metal", "bath_vent")
        b.box((2.2, 0.0, 0.0), (3.1, 0.04, 2.05), "metal", "door")
        b.box((2.85, 0.5, 0.0), (3.38, 1.9, 2.1), "wood", "wardrobe")
        b.box((2.83, 0.55, 0.1), (2.85, 1.85, 2.0), "metal", "mirror")
        refl.append((0, 2.83, [2.83, 0.55, 0.1], [2.85, 1.85, 2.0], 0.92))
        b.box((0.02, 2.9, 0.0), (2.0, 3.9, 0.5), "wood", "bed")
        b.box((0.02, 4.5, 0.0), (2.0, 5.5, 0.5), "wood", "bed")
        b.box((0.02, 2.8, 0.5), (0.08, 5.6, 1.1), "wood", "headboard")
        b.box((0.08, 4.0, 0.0), (0.5, 4.4, 0.55), "wood", "nightstand")
        b.box((2.85, 3.2, 0.72), (3.38, 5.6, 0.76), "wood", "desk_top")
        b.box((3.32, 3.9, 1.1), (3.38, 4.9, 1.7), "metal", "tv")
        b.box((2.9, 5.1, 0.0), (3.35, 5.55, 0.6), "metal", "minibar")
        b.box((0.6, W - 0.02, 0.9), (2.8, W, 2.1), "glass", "window")
        b.box((0.55, W - 0.05, 0.84), (2.85, W - 0.02, 0.9), "metal", "winframe")
        b.box((1.3, 4.2, H - 0.08), (2.1, 5.0, H - 0.01), "metal", "aircon")
        b.box((2.5, 6.3, 0.0), (3.1, 6.9, 0.45), "wood", "armchair")
        ppl.standing(U(2.2, 2.7), U(2.8, 6.0))
        gens = [
            (14, "smoke_alarm", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(2.8, W - 0.3), H - 0.05]),
            (10, "tv", lambda r: [3.27, r.uniform(3.85, 4.95), r.choice([r.uniform(1.02, 1.08), r.uniform(1.72, 1.78)])]),
            (12, "desk", lambda r: [r.uniform(2.9, 3.3), r.uniform(3.3, 5.0), r.uniform(0.78, 1.0)]),
            (8, "nightstand", lambda r: [r.uniform(0.12, 0.45), r.uniform(4.05, 4.35), r.uniform(0.57, 0.72)]),
            (10, "wall_art", lambda r: [0.06, r.uniform(2.9, 5.5), r.uniform(1.3, 1.8)]),
            (8, "aircon", lambda r: [r.choice([1.26, 2.14]), r.uniform(4.2, 5.0), H - r.uniform(0.04, 0.1)]),
            (8, "wardrobe_top", lambda r: [r.uniform(2.9, 3.3), r.uniform(0.6, 1.8), r.uniform(2.12, 2.2)]),
            (10, "charger", lambda r: _wallpt(r, L, W, 0.25, 0.5, walls=(0, 1), yr=(2.7, 7.0))),
            (8, "window_sill", lambda r: [r.uniform(0.7, 2.7), W - 0.08, r.uniform(0.92, 1.0)]),
            (8, "bath_vent", lambda r: [r.uniform(0.3, 1.5), r.uniform(0.3, 2.1), H - 0.07]),
            (4, "bath_shelf", lambda r: [0.08, r.uniform(0.5, 1.4), r.uniform(1.92, 2.0)]),
        ]

    elif scene_id == "guesthouse_d":
        shell_mixed(b, L, W, H, x0="plasterboard")
        for y0b in (0.8, 3.0):                      # two bunk beds along the x0 wall: wooden decks, metal posts
            b.box((0.02, y0b, 0.35), (1.0, y0b + 2.0, 0.45), "wood", "bunk_low")
            b.box((0.02, y0b, 1.45), (1.0, y0b + 2.0, 1.55), "wood", "bunk_up")
            for px in (0.02, 0.96):
                for py in (y0b, y0b + 1.96):
                    b.box((px, py, 0.0), (px + 0.04, py + 0.04, 1.9), "metal", "bunk_post")
            b.box((0.98, y0b, 1.75), (1.0, y0b + 2.0, 1.79), "metal", "bunk_rail")
        b.box((3.9, 0.6, 0.0), (4.38, 2.6, 1.8), "metal", "lockers")
        refl.append((0, 3.9, [3.9, 0.6, 0.0], [4.38, 2.6, 1.8], 0.85))
        b.box((2.2, 3.6, 0.72), (3.4, 4.6, 0.76), "wood", "table_top")
        for px in (2.22, 3.34):
            for py in (3.62, 4.54):
                b.box((px, py, 0.0), (px + 0.04, py + 0.04, 0.72), "metal", "deskleg")
        b.box((3.8, 4.5, 0.0), (4.38, 5.1, 1.2), "metal", "fridge")
        b.box((1.6, W - 0.02, 1.0), (3.4, W, 2.0), "glass", "window")
        b.box((1.55, W - 0.05, 0.94), (3.45, W - 0.02, 1.0), "metal", "winframe")
        b.box((2.0, W - 0.25, 2.15), (3.0, W - 0.02, 2.42), "metal", "aircon")
        b.box((1.6, 0.0, 0.0), (2.5, 0.04, 2.05), "wood", "door")
        b.box((2.7, 0.02, 0.0), (3.5, 0.35, 0.9), "metal", "shoe_rack")
        b.box((2.0, 2.2, H - 0.07), (3.2, 2.5, H - 0.01), "metal", "light")
        for _ in range(3):
            ppl.standing(U(1.4, 3.4), U(0.8, 3.2))
        gens = [
            (16, "bunk_frame", lambda r: [r.choice([0.08, 0.94]), r.choice([0.8, 3.0]) + r.choice([0.1, 1.9]), r.uniform(1.6, 1.9)]),
            (10, "bunk_under", lambda r: [r.uniform(0.1, 0.9), r.choice([0.8, 3.0]) + r.uniform(0.2, 1.8), r.uniform(1.36, 1.42)]),
            (12, "locker_top", lambda r: [r.uniform(3.95, 4.33), r.uniform(0.7, 2.5), r.uniform(1.82, 1.9)]),
            (8, "locker_gap", lambda r: [3.86, r.choice([0.56, 2.64]), r.uniform(0.5, 1.7)]),
            (12, "smoke_alarm", lambda r: [r.uniform(1.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.05]),
            (8, "wall_clock", lambda r: _wallpt(r, L, W, 1.8, 2.2, walls=(1, 2), xr=(0.3, 1.4), yr=(2.8, 4.3))),
            (10, "power_strip", lambda r: [r.uniform(2.3, 3.3), r.uniform(3.7, 4.5), r.uniform(0.78, 0.9)]),
            (8, "fridge_top", lambda r: [r.uniform(3.85, 4.3), r.uniform(4.55, 5.05), r.uniform(1.22, 1.32)]),
            (8, "aircon", lambda r: [r.uniform(2.0, 3.0), W - r.uniform(0.27, 0.33), r.uniform(2.1, 2.4)]),
            (8, "shoe_rack", lambda r: [r.uniform(2.75, 3.45), r.uniform(0.38, 0.42), r.uniform(0.3, 0.85)]),
        ]

    elif scene_id == "restroom_f":
        shell_mixed(b, L, W, H, x0="tile", xl="tile", y0="tile", yw="tile", floor="tile")
        b.box((0.3, 0.0, 0.0), (1.2, 0.04, 2.05), "metal", "door")
        ys = [1.6 + 0.95 * k for k in range(5)]     # four stalls along the xl wall: laminate panels on aluminium posts
        for yk in ys:
            b.box((3.1, yk - 0.012, 0.15), (L, yk + 0.012, 2.0), "chipboard", "stall")
            b.box((3.07, yk - 0.02, 0.0), (3.13, yk + 0.02, 2.05), "metal", "stallpost")
        b.box((3.09, ys[0], 0.15), (3.115, ys[-1], 2.0), "chipboard", "stall_front")
        b.box((3.07, ys[0], 2.0), (3.13, ys[-1], 2.05), "metal", "stall_rail")
        for yk in ys[:-1]:
            b.box((L - 0.7, yk + 0.25, 0.0), (L - 0.15, yk + 0.7, 0.42), "tile", "toilet")
        b.box((0.0, 2.0, 0.0), (0.55, 4.8, 0.85), "tile", "sinks")
        b.box((0.0, 2.0, 1.0), (0.02, 4.8, 2.0), "metal", "mirror")
        refl.append((0, 0.02, [0.0, 2.0, 1.0], [0.02, 4.8, 2.0], 0.92))
        b.box((1.6, 0.02, 1.0), (1.95, 0.2, 1.35), "metal", "hand_dryer")
        b.box((0.05, 1.3, 0.0), (0.4, 1.65, 0.6), "metal", "bin")
        b.box((0.0, 5.6, 0.0), (1.2, 5.68, H), "plasterboard", "closet_wall")
        b.box((1.12, 5.68, 0.0), (1.2, W, H), "plasterboard", "closet_wall")
        b.box((2.0, W - 0.02, 1.7), (3.0, W, 2.2), "glass", "window")
        b.box((1.6, 2.6, H - 0.05), (2.0, 3.0, H - 0.01), "metal", "vent")
        b.box((1.6, 4.6, H - 0.05), (2.0, 5.0, H - 0.01), "metal", "vent")
        for _ in range(2):
            ppl.standing(U(1.2, 2.6), U(1.0, 5.2))
        st = lambda r: ys[int(r.integers(4))]  # noqa: E731
        gens = [
            (20, "stall_tp", lambda r: [L - r.uniform(0.9, 1.3), st(r) + r.choice([0.06, 0.89]), r.uniform(0.6, 0.8)]),
            (10, "stall_hook", lambda r: [3.17, st(r) + r.uniform(0.3, 0.65), r.uniform(1.6, 1.8)]),
            (12, "stall_vent", lambda r: [L - r.uniform(0.2, 1.3), r.uniform(1.7, 5.3), H - 0.06]),
            (8, "stall_floor", lambda r: [L - r.uniform(0.12, 0.2), st(r) + r.uniform(0.75, 0.9), r.uniform(0.15, 0.35)]),
            (15, "ceiling_vent", lambda r: [r.uniform(0.3, 2.9), r.uniform(0.3, W - 0.3), H - 0.06]),
            (10, "mirror_frame", lambda r: [0.06, r.choice([1.94, 4.86]) + r.uniform(-0.03, 0.03), r.uniform(1.0, 2.05)]),
            (5, "dryer", lambda r: [r.uniform(1.6, 1.95), 0.24, r.uniform(1.0, 1.4)]),
            (10, "sink_top", lambda r: [r.uniform(0.1, 0.5), r.uniform(2.1, 4.7), r.uniform(0.87, 0.95)]),
            (5, "bin", lambda r: [r.uniform(0.07, 0.38), r.uniform(1.32, 1.63), r.uniform(0.62, 0.7)]),
            (5, "window_sill", lambda r: [r.uniform(2.05, 2.95), W - 0.06, r.uniform(1.6, 1.68)]),
        ]

    elif scene_id == "pool_locker":
        shell_mixed(b, L, W, H, floor="tile")
        b.box((0.02, 0.8, 0.0), (0.5, 5.4, 1.8), "metal", "lockers")
        refl.append((0, 0.5, [0.02, 0.8, 0.0], [0.5, 5.4, 1.8], 0.85))
        b.box((3.0, 1.2, 0.0), (3.9, 5.0, 1.8), "metal", "island_lockers")
        b.box((4.6, W - 0.5, 0.0), (8.0, W - 0.02, 1.8), "metal", "lockers_n")
        b.box((1.4, 1.5, 0.42), (2.0, 4.5, 0.46), "wood", "bench")
        b.box((5.2, 2.0, 0.42), (5.8, 4.4, 0.46), "wood", "bench")
        b.box((L - 0.55, 0.8, 0.0), (L, 3.4, 0.8), "wood", "vanity")
        b.box((L - 0.02, 0.8, 0.9), (L, 3.4, 1.9), "metal", "mirror")
        refl.append((0, L - 0.02, [L - 0.02, 0.8, 0.9], [L, 3.4, 1.9], 0.92))
        b.box((7.2, 3.8, 0.0), (L, 3.9, H), "tile", "shower_baffle")
        b.box((6.6, 0.1, 0.0), (7.1, 0.6, 0.9), "metal", "spin_dryer")
        b.box((4.6, 0.05, 0.0), (5.0, 0.45, 1.2), "metal", "dispenser")
        b.box((1.0, 0.0, 0.0), (1.9, 0.04, 2.05), "metal", "door")
        lights(b, L, W, H, 3, 2)
        for _ in range(6):
            ppl.standing(U(0.8, 7.8), U(0.6, 5.4))
        rows = [((0.08, 0.44), (0.9, 5.3)), ((3.06, 3.84), (1.3, 4.9)), ((4.7, 7.9), (W - 0.44, W - 0.08))]
        def lock_top(r):
            (xa, xb), (ya, yb) = rows[int(r.integers(3))]
            return [r.uniform(xa, xb), r.uniform(ya, yb), r.uniform(1.82, 1.9)]
        gens = [
            (25, "locker_top", lock_top),
            (15, "ceiling_vent", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, W - 0.3), H - 0.09]),
            (10, "wall_clock", lambda r: _wallpt(r, L, W, 1.95, 2.3)),
            (10, "vanity", lambda r: [L - r.uniform(0.1, 0.5), r.uniform(0.9, 3.3), r.uniform(0.82, 1.0)]),
            (6, "dispenser_top", lambda r: [r.uniform(4.62, 4.98), r.uniform(0.08, 0.42), r.uniform(1.22, 1.3)]),
            (6, "dryer_top", lambda r: [r.uniform(6.65, 7.05), r.uniform(0.15, 0.55), r.uniform(0.92, 1.0)]),
            (8, "bench_under", lambda r: [r.choice([r.uniform(1.45, 1.95), r.uniform(5.25, 5.75)]), r.uniform(2.1, 4.3), r.uniform(0.3, 0.39)]),
            (15, "locker_gap", lambda r: [r.choice([2.96, 3.94]), r.choice([1.14, 5.06]), r.uniform(0.5, 1.7)]),
            (5, "baffle_top", lambda r: [r.uniform(7.3, 8.3), r.choice([3.74, 3.96]), r.uniform(1.9, 2.3)]),
        ]

    elif scene_id == "fitting_b":
        shell_mixed(b, L, W, H, x0="plasterboard", xl="plasterboard", yw="plasterboard")
        xs = [0.8, 2.0, 3.2, 4.4]                   # three booths along the yw wall, wooden partitions
        for xk in xs:
            b.box((xk - 0.015, 2.7, 0.1), (xk + 0.015, W, 2.1), "chipboard", "booth_wall")
        b.box((0.8, 2.69, 2.05), (4.4, 2.73, 2.1), "wood", "booth_header")
        b.box((0.82, 2.70, 0.1), (1.98, 2.73, 2.05), "chipboard", "booth_door")    # booths 1 and 3 closed, booth 2 open
        b.box((3.22, 2.70, 0.1), (4.38, 2.73, 2.05), "chipboard", "booth_door")
        for k in range(3):
            xc = xs[k] + 0.6
            b.box((xc - 0.35, W - 0.02, 0.3), (xc + 0.35, W, 1.9), "metal", "booth_mirror")
            refl.append((1, W - 0.02, [xc - 0.35, W - 0.02, 0.3], [xc + 0.35, W, 1.9], 0.92))
            b.box((xs[k] + 0.08, 3.5, 0.40), (xs[k] + 0.43, 3.85, 0.44), "wood", "stool")
        b.box((1.5, 0.9, 0.42), (3.5, 1.3, 0.46), "wood", "bench")
        b.box((4.3, 0.4, 1.60), (4.9, 1.9, 1.63), "metal", "rack_bar")
        for py in (0.4, 1.86):
            b.box((4.58, py, 0.0), (4.62, py + 0.04, 1.6), "metal", "rack_post")
        b.box((0.0, 1.0, 0.2), (0.02, 1.8, 2.0), "metal", "mirror")
        refl.append((0, 0.02, [0.0, 1.0, 0.2], [0.02, 1.8, 2.0], 0.92))
        b.box((0.1, 0.05, 0.0), (1.2, 0.5, 0.95), "wood", "counter")
        b.box((2.0, 0.0, 0.0), (2.9, 0.04, 2.05), "wood", "door")
        lights(b, L, W, H, 2, 2, sx=0.6, sy=0.6)
        b.box((2.4, 3.2, H - 0.05), (2.8, 3.6, H - 0.01), "metal", "vent")
        for _ in range(2):
            ppl.standing(U(1.0, 4.0), U(0.6, 2.4))
        bk = lambda r: xs[int(r.integers(3))]  # noqa: E731
        gens = [
            (16, "booth_hook", lambda r: [bk(r) + r.choice([0.05, 1.15]), r.uniform(3.0, 3.8), r.uniform(1.5, 1.9)]),
            (10, "booth_mirror_edge", lambda r: [bk(r) + 0.6 + r.choice([-0.39, 0.39]), W - 0.05, r.uniform(0.4, 1.9)]),
            (8, "booth_stool", lambda r: [bk(r) + r.uniform(0.1, 0.4), r.uniform(3.52, 3.83), r.uniform(0.28, 0.37)]),
            (10, "booth_ceiling", lambda r: [r.uniform(0.9, 4.3), r.uniform(2.9, 3.8), H - 0.06]),
            (12, "ceiling", lambda r: [r.uniform(0.3, L - 0.3), r.uniform(0.3, 2.5), H - 0.09]),
            (8, "rack", lambda r: [r.uniform(4.35, 4.85), r.uniform(0.5, 1.8), r.uniform(1.65, 1.75)]),
            (8, "counter", lambda r: [r.uniform(0.2, 1.1), r.uniform(0.1, 0.45), r.uniform(0.97, 1.1)]),
            (8, "wall_clock", lambda r: _wallpt(r, L, W, 1.9, 2.3, walls=(1, 2), xr=(3.1, 4.6), yr=(0.3, 2.5))),
            (10, "outlet", lambda r: _wallpt(r, L, W, 0.25, 0.45, walls=(0, 1), yr=(0.3, 2.5))),
            (10, "header", lambda r: [r.uniform(0.9, 4.3), 2.66, r.uniform(1.95, 2.1)]),
        ]
    else:
        raise ValueError(scene_id)

    if gens is not None:
        cams, types = _sample(rng, b, (L, W, H), gens, spots=SPOTS[scene_id])
        conceal = _conceal(rng, b, (L, W, H), cams)
    else:
        conceal = _exam_conceal(rng, b, cams, types)
    if variant == "op1":
        # the operator: one standing person, at least 1.2 m from every receiver spot and 0.5 m from every camera
        r2 = np.random.default_rng(crc("op1", scene_id, int(rng.integers(1 << 30))))
        sp = np.array([s[1:3] for s in SPOTS[scene_id]])
        # body test heights: the second held-out set checks every 10 cm so that nobody stands through a bunk deck or a
        # bench; the earlier rooms keep the three heights they were traced with
        zs = tuple(np.round(np.arange(0.1, 1.75, 0.1), 2)) if scene_id in SET_C else (0.3, 1.0, 1.6)
        for _ in range(400):
            p = np.array([r2.uniform(0.5, L - 0.5), r2.uniform(0.5, W - 0.5)])
            if np.min(np.linalg.norm(sp - p, axis=1)) < 1.2:
                continue
            if np.min(np.linalg.norm(np.asarray(cams)[:, :2] - p, axis=1)) < 0.5:
                continue
            if any(G._inside_any(np.array([p[0] + dx, p[1] + dy, z]), b.boxes, margin=0.0)
                   for dx in (-0.22, 0.22) for dy in (-0.12, 0.12) for z in zs):
                continue
            b.add_person_standing(p[0], p[1])
            break
        else:
            if scene_id in SET_C:
                # small rooms with eight spots: no place 1.2 m from every spot -> allow 0.8 m (second held-out set only,
                # so the earlier rooms stay exactly as they were traced)
                for _ in range(2000):
                    p = np.array([r2.uniform(0.4, L - 0.4), r2.uniform(0.4, W - 0.4)])
                    if np.min(np.linalg.norm(sp - p, axis=1)) < 0.8:
                        continue
                    if np.min(np.linalg.norm(np.asarray(cams)[:, :2] - p, axis=1)) < 0.4:
                        continue
                    if any(G._inside_any(np.array([p[0] + dx, p[1] + dy, z]), b.boxes, margin=0.0)
                           for dx in (-0.22, 0.22) for dy in (-0.12, 0.12) for z in zs):
                        continue
                    b.add_person_standing(p[0], p[1])
                    break
    else:
        rng.integers(1 << 30)     # keep the stream position identical for every variant
    boxes = list(b.boxes)
    spots = SPOTS[scene_id]
    return b, (L, W, H), spots, np.array(cams, dtype=np.float64), types, refl, boxes, conceal


def _exam_cams(rng, seats, facing, n=40):
    """Cheating cameras on candidates: chest button, glasses, pencil case, on the desk.  Seats drawn with replacement
    (each camera is evaluated on its own)."""
    cams, types = [], []
    for _ in range(n):
        sx, sy, z0, lean = seats[int(rng.integers(len(seats)))]
        typ = rng.choice(["chest", "chest", "glasses", "pencil_case", "desk"])
        if facing == "y-":
            p = dict(chest=[sx + rng.uniform(-0.05, 0.05), sy - 0.15 - lean, z0 + 0.98],
                     glasses=[sx, sy - 0.22 - lean, z0 + 1.24],
                     pencil_case=[sx + rng.uniform(-0.18, 0.18), sy - 0.40, z0 + 0.77],
                     desk=[sx + rng.uniform(-0.22, 0.22), sy - 0.50, z0 + 0.76])[typ]
        else:
            p = dict(chest=[sx - 0.15 - lean, sy + rng.uniform(-0.05, 0.05), z0 + 0.98],
                     glasses=[sx - 0.22 - lean, sy, z0 + 1.24],
                     pencil_case=[sx - 0.40, sy + rng.uniform(-0.18, 0.18), z0 + 0.77],
                     desk=[sx - 0.50, sy + rng.uniform(-0.20, 0.20), z0 + 0.76])[typ]
        cams.append(p)
        types.append(str(typ))
    return cams, types


def _exam_conceal(rng, b, cams, types):
    """Pencil-case cameras sit inside a small case; the others are open (a button or glasses camera looks out)."""
    out = []
    for p, t in zip(cams, types):
        if t == "pencil_case" and rng.uniform() < 0.7:
            p = np.asarray(p)
            b.box(p - np.array([0.09, 0.04, 0.025]), p + np.array([0.09, 0.04, 0.025]), "chipboard", "camcase")
            out.append("case")
        else:
            out.append("open")
    return out
