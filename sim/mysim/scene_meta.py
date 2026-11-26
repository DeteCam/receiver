"""Scene metadata without ray tracing: re-run the room builders with a recording builder (same RNG seeds as the
ray-traced data) and keep what an AR app could register: desks (seat layout) and furniture boxes.

  python3 mysim/scene_meta.py            -> results/scene_meta.pkl  {(scene, seed): dict(dims, desks, boxes, cams)}

desks: list of (x0, x1, y0, y1, z_top) of every desk; boxes: (lo, hi, mat, tag) of every object (walls/floor/ceiling,
furniture, people).  cams is kept only to check that the builder reproduced the ray-traced scene exactly.
"""
import os
import sys
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
import gen_scenes as G  # noqa: E402
import fresh_scenes as F  # noqa: E402


class RecordingBuilder(G.SceneBuilder):
    def __init__(self, fc, mesh_dir):  # no Sionna scene
        self.fc = float(fc)
        self.mesh_dir = mesh_dir
        self.by_mat, self.boxes, self.mats, self.desks = {}, [], {}, []

    def add_desk_metal_legs(self, x0, x1, y0, y1, z0=0.0, h=0.73):
        self.desks.append((float(x0), float(x1), float(y0), float(y1), float(z0 + h)))
        super().add_desk_metal_legs(x0, x1, y0, y1, z0=z0, h=h)

    def finalize(self):
        return self


def meta_for(scene_id, seed):
    rng = np.random.default_rng(2000 + seed * 101 + abs(hash(scene_id)) % 97)
    orig = G.SceneBuilder
    G.SceneBuilder = RecordingBuilder
    try:
        if scene_id in F.FRESH_SCENES:
            b, dims, nodes, cams, types, refl, boxes = F.build_fresh(scene_id, 2.437e9, rng, "/tmp")
        else:
            b, dims, nodes, cams, types, refl = G.build_scene(scene_id, 2.437e9, rng, "/tmp")
    finally:
        G.SceneBuilder = orig
    return dict(dims=np.array(dims, dtype=np.float64), desks=list(b.desks), boxes=list(b.boxes),
                cams=np.array(cams, dtype=np.float64), cam_types=list(types))


if __name__ == "__main__":
    scenes = G.DEV_SCENES + G.EVAL_SCENES + F.FRESH_SCENES
    out = {}
    for sc in scenes:
        for s in (0, 1, 2):
            m = meta_for(sc, s)
            # check against the ray-traced file (base data or fresh data)
            fp = (f"/opt/detecam/sim/data_fresh/base/{sc}_2g4_s{s}.npy" if sc in F.FRESH_SCENES
                  else f"/opt/detecam/sim/data/{sc}_2g4_s{s}.npy")
            if os.path.exists(fp):
                D = np.load(fp, allow_pickle=True).item()
                ok = np.allclose(D["cams"], m["cams"])
            else:
                ok = None
            out[(sc, s)] = m
            print(sc, s, "desks", len(m["desks"]), "boxes", len(m["boxes"]), "cams match", ok, flush=True)
    with open("/opt/detecam/sim/results/scene_meta.pkl", "wb") as f:
        pickle.dump(out, f)
