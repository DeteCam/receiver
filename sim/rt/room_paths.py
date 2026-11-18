"""숙소 방 장면에서 패널을 두는 곳 -> 숨긴 카메라 점의 전파 길을 구해 npz 로 담는다 (교실의 rt_paths.py 와 같은 꼴).
    python room_paths.py 이름 [--seed 1] [--per 16] [--depth 3] [--door-open]
"""
import os, time, argparse
import numpy as np
import mitsuba as mi
import sionna.rt as rt
import scene_room as R
from rt_paths import solve_spot, FC


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--per", type=int, default=16)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--samples", type=int, default=10 ** 6); ap.add_argument("--door-open", action="store_true"); ap.add_argument("--free", action="store_true"); ap.add_argument("--out", default="paths")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed)
    scene = R.build(door_open=a.door_open); scene.frequency = FC
    arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH"); scene.tx_array = arr; scene.rx_array = arr
    cams, cat = R.hide_points(rng, a.per, a.free)
    for i, c in enumerate(cams): scene.add(rt.Receiver("c%03d" % i, position=mi.Point3f(*[float(v) for v in c])))
    solver = rt.PathSolver(); res = []; t0 = time.time(); none = np.array([-1])
    for k, (nm, pos, aim) in enumerate(R.PLACES):
        o, _ = solve_spot(scene, solver, pos, cams, a.depth, True, a.samples, none); res.append(o)
        print("%2d/%d %-22s 길 수 가운데값 %d, 직접파 있는 점 %d/%d  (%.0f초)" % (k + 1, len(R.PLACES), nm, np.median(o["ok"].sum(1)), int(((o["nint"] == 0) & o["ok"]).any(1).sum()), len(cams), time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True); P = max(o["A"].shape[1] for o in res)
    def pad(x):
        return np.pad(x, [(0, 0), (0, P - x.shape[1])] + [(0, 0)] * (x.ndim - 2))
    np.savez_compressed(os.path.join(a.out, a.name + ".npz"), spots=np.array([p for _, p, _ in R.PLACES]), aims=np.array([q for _, _, q in R.PLACES]), names=np.array([n for n, _, _ in R.PLACES]),
                        cams=cams, cat=cat, cats=np.array(R.all_cats(a.free)), fc=FC, meta=str(vars(a)), **{k: np.stack([pad(o[k]) for o in res]) for k in res[0]})
    print("저장", a.name, "%.0f초" % (time.time() - t0))


if __name__ == "__main__":
    main()
