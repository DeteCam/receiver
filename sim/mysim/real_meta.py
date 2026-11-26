"""Scene metadata and a geometry check for the realistic rooms (no ray tracing).
  python3 mysim/real_meta.py check     -> prints a geometry report (objects, cameras, spots)
  python3 mysim/real_meta.py           -> adds the realistic rooms to results/scene_meta.pkl
Seeds use zlib.crc32 (not hash()), so the rooms do not depend on PYTHONHASHSEED."""
import os
import sys
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
import gen_scenes as G  # noqa: E402
import real_scenes as R  # noqa: E402


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


def scene_rng(scene_id, seed):
    return np.random.default_rng(7000 + seed * 101 + R.crc("real", scene_id) % 9973)


def meta_for(scene_id, seed, variant="base"):
    orig = G.SceneBuilder
    G.SceneBuilder = RecordingBuilder
    try:
        b, dims, spots, cams, types, refl, boxes, conceal = R.build_real(scene_id, 2.437e9, scene_rng(scene_id, seed), "/tmp", variant)
    finally:
        G.SceneBuilder = orig
    return dict(dims=np.array(dims, dtype=np.float64), desks=list(b.desks), boxes=boxes, cams=cams, cam_types=list(types),
                conceal=list(conceal), spots=spots, refl=refl)


def check():
    shell = ("floor", "ceil", "wx0", "wxl", "wy0", "wyw")
    for sc in R.REAL_SCENES:
        for s in (0, 1, 2):
            m = meta_for(sc, s, "base")
            mn = meta_for(sc, s, "nop")
            mo = meta_for(sc, s, "op1")
            assert np.allclose(m["cams"], mn["cams"]) and np.allclose(m["cams"], mo["cams"]), (sc, s)
            assert m["conceal"] == mn["conceal"] == mo["conceal"]
            L, W, H = m["dims"]
            bad = []
            for lab, x, y, z, yaw, typ in m["spots"]:
                p = np.array([x, y, z])
                for lo, hi, mat, tag in mn["boxes"]:
                    if tag in shell:
                        continue
                    if np.all(p >= lo - 0.07) and np.all(p <= hi + 0.07) and not (abs(p[2] - hi[2]) < 0.16 and p[2] > hi[2]):
                        bad.append((lab, tag))
                if not (0.05 <= x <= L - 0.05 and 0.05 <= y <= W - 0.05 and 0.3 <= z <= H - 0.12):
                    bad.append((lab, "outside"))
            types, cnt = np.unique(m["cam_types"], return_counts=True)
            cc = {k: m["conceal"].count(k) for k in ("open", "case", "glass")}
            nb = dict(base=len(m["boxes"]), nop=len(mn["boxes"]), op1=len(mo["boxes"]))
            if s == 0:
                print(f"{sc:13s} dims {L}x{W}x{H} spots {len(m['spots'])} desks {len(m['desks'])} boxes {nb}")
                print("   cams:", dict(zip(types.tolist(), cnt.tolist())))
            print(f"   s{s} conceal {cc} zrange {m['cams'][:,2].min():.2f}-{m['cams'][:,2].max():.2f} bad spots {bad}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "check":
        check()
        sys.exit(0)
    fp = "/opt/detecam/sim/results/scene_meta.pkl"
    out = pickle.load(open(fp, "rb")) if os.path.exists(fp) else {}
    for sc in R.REAL_SCENES:
        for s in (0, 1, 2):
            m = meta_for(sc, s, "base")
            out[(sc, s)] = dict(dims=m["dims"], desks=m["desks"], boxes=m["boxes"], cams=m["cams"], cam_types=m["cam_types"])
    with open(fp + ".tmp", "wb") as f:
        pickle.dump(out, f)
    os.replace(fp + ".tmp", fp)      # atomic: running evaluations never read a half-written file
    print("real scenes added:", len(R.REAL_SCENES) * 3)
