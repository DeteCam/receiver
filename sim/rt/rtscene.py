"""Sionna RT scene builders (classroom / lodging room) for independent verification of DeteCam localisation."""
import os, numpy as np
import mitsuba as mi, drjit as dr
import sionna.rt as rt

MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meshes")
os.makedirs(MESH_DIR, exist_ok=True)

def _write_ply(path, V, F):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex %d\nproperty float x\nproperty float y\nproperty float z\nelement face %d\nproperty list uchar int vertex_indices\nend_header\n" % (len(V), len(F)))
        for v in V: f.write("%.5f %.5f %.5f\n" % tuple(v))
        for t in F: f.write("3 %d %d %d\n" % tuple(t))

def box_mesh(name, lo, hi):
    x0, y0, z0 = lo; x1, y1, z1 = hi
    V = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]], float)
    F = [[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],[1,2,6],[1,6,5],[2,3,7],[2,7,6],[3,0,4],[3,4,7]]
    p = os.path.join(MESH_DIR, name + ".ply"); _write_ply(p, V, F); return p

class Builder:
    def __init__(self, fc):
        self.scene = rt.load_scene(); self.scene.frequency = fc; self.objs = []; self.mats = {}; self.n = 0; self.fc = fc; self.boxes = []
    def mat(self, key):
        if key in self.mats: return self.mats[key]
        sc = dict(concrete=0.25, plasterboard=0.25, wood=0.30, glass=0.05, metal=0.10, ceiling_board=0.30, floorboard=0.25, chipboard=0.30)
        if key == "body":
            eps, sig = (52.7, 1.74) if self.fc < 4e9 else (49.3, 4.5)
            m = rt.RadioMaterial(name="body", thickness=0.20, relative_permittivity=eps, conductivity=sig, scattering_coefficient=float(os.environ.get("BODY_S", 0.35)))
        else:
            th = dict(concrete=0.20, plasterboard=0.025, wood=0.03, glass=0.006, metal=0.002, ceiling_board=0.02, floorboard=0.03, chipboard=0.02)[key]
            m = rt.ITURadioMaterial(name="m_" + key, itu_type=key, thickness=th, scattering_coefficient=sc[key])
        self.mats[key] = m; return m
    def box(self, lo, hi, mat, tag="o"):
        self.n += 1; name = f"{tag}{self.n}"; self.boxes.append((np.array(lo, float), np.array(hi, float), mat, tag))
        self.objs.append(rt.SceneObject(fname=box_mesh(name, lo, hi), name=name, radio_material=self.mat(mat)))
    def done(self):
        self.scene.edit(add=self.objs); return self.scene

def shell(b, L, W, H, wall="concrete", t=0.15):
    b.box((-t, -t, -t), (L + t, W + t, 0.0), "concrete", "floor")
    b.box((-t, -t, H), (L + t, W + t, H + t), "ceiling_board", "ceil")
    b.box((-t, -t, 0), (0, W + t, H), wall, "wx0"); b.box((L, -t, 0), (L + t, W + t, H), wall, "wxl")
    b.box((0, -t, 0), (L, 0, H), wall, "wy0"); b.box((0, W, 0), (L, W + t, H), wall, "wyw")

def classroom(fc, rng, people=None, L=8.5, W=7.5, H=2.8, furn=None):
    """40-seat classroom following the repository's 'classroom_40' dimensions. Blackboard wall at y=0."""
    people = (os.environ.get("PEOPLE", "1") != "0") if people is None else people
    furn = (os.environ.get("FURN", "1") != "0") if furn is None else furn
    b = Builder(fc); shell(b, L, W, H)
    if furn:
        b.box((2.45, 0.0, 0.9), (6.05, 0.03, 2.1), "metal", "board")                       # steel blackboard
        b.box((0.0, 1.2, 0.9), (0.02, 6.8, 2.2), "glass", "win")                           # window strip on wall x=0
        b.box((L - 0.03, 0.4, 0.0), (L, 1.3, 2.0), "wood", "door"); b.box((L - 0.03, 6.2, 0.0), (L, 7.1, 2.0), "wood", "door")
        b.box((3.7, 0.7, 0.0), (4.8, 1.3, 1.05), "wood", "podium")
        b.box((L - 0.45, 3.0, 0.0), (L - 0.02, 4.4, 1.8), "metal", "cabinet")              # steel cabinet (not in the nominal map)
    cols = np.linspace(1.3, 7.2, 8); rows = np.linspace(2.2, 6.4, 5); seats = []; leans = []
    for ry in rows:
        for cx in cols:
            sx, sy = cx + rng.normal(0, 0.04), ry + rng.normal(0, 0.04)
            seats.append((sx, sy))
            if furn: b.box((sx - 0.30, sy - 0.62, 0.71), (sx + 0.30, sy - 0.17, 0.74), "wood", "desk")
            lean = rng.uniform(-0.03, 0.10); w = rng.uniform(0.19, 0.24); leans.append(lean)
            if people:
                b.box((sx - w, sy - 0.11 - lean, 0.45), (sx + w, sy + 0.11 - lean, 1.08), "body", "torso")
                b.box((sx - 0.09, sy - 0.20 - lean, 1.10), (sx + 0.09, sy - 0.0 - lean, 1.32), "body", "head")
    if people:                                                                          # walking proctor
        px, py = rng.choice([0.6, 4.25, 7.9]), rng.uniform(1.6, 6.8)
        b.box((px - 0.22, py - 0.12, 0.0), (px + 0.22, py + 0.12, 1.50), "body", "proctor"); b.box((px - 0.09, py - 0.1, 1.52), (px + 0.09, py + 0.1, 1.75), "body", "phead")
    return b.done(), dict(dims=(L, W, H), seats=np.array(seats), leans=np.array(leans), boxes=b.boxes)

MOTEL_FURN = lambda L, W: [((0.3, 1.8, 0.0), (2.3, 3.6, 0.55), "wood", "bed"), ((3.8, 2.0, 0.0), (4.5, 3.7, 2.1), "wood", "wardrobe"),
                           ((1.2, 0.0, 0.0), (3.2, 0.45, 0.85), "wood", "tvconsole"), ((0.0, 0.8, 0.7), (0.02, 2.4, 2.1), "metal", "mirror"),
                           ((1.5, 0.02, 0.95), (2.9, 0.06, 1.75), "metal", "tv"),            # TV panel (not in nominal map)
                           ((0.05, 0.05, 0.0), (0.6, 0.6, 0.85), "metal", "fridge"),         # mini fridge (not in nominal map)
                           ((1.2, W - 0.02, 0.9), (3.0, W, 2.2), "glass", "window"), ((2.45, 2.0, 0.0), (2.9, 2.45, 0.5), "wood", "nightstand")]

def motel(fc, rng, person=True, L=4.5, W=3.8, H=2.6, furn=None):
    """Lodging room following the repository's 'motel_std' layout, plus items a 5-second AR scan would not capture."""
    furn = (os.environ.get("FURN", "1") != "0") if furn is None else furn
    b = Builder(fc); shell(b, L, W, H, wall=os.environ.get("WALL", "plasterboard"))          # WALL=concrete | metal: stronger wall reflections
    if furn:
        for lo, hi, mat, tag in MOTEL_FURN(L, W): b.box(lo, hi, mat, tag)
    if person:
        px, py = rng.uniform(2.6, 3.6), rng.uniform(0.8, 1.7)
        b.box((px - 0.22, py - 0.12, 0.0), (px + 0.22, py + 0.12, 1.50), "body", "user"); b.box((px - 0.09, py - 0.1, 1.52), (px + 0.09, py + 0.1, 1.75), "body", "uhead")
    return b.done(), dict(dims=(L, W, H), boxes=b.boxes)
