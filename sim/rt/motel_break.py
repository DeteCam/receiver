"""Lodging room: direction error by spot, by how steeply the camera is seen, and with close cameras set aside.  Usage: motel_break.py TAG [BAND]"""
import sys, numpy as np
tag = sys.argv[1]; band = sys.argv[2] if len(sys.argv) > 2 else "2g4"
d = np.load(f"res_pe2_{tag}.npy", allow_pickle=True).item(); rows = [r for r in d["rows"] if not r["beacon"]]
cams = [np.load(f"data/xm_{band}_{i}.npy", allow_pickle=True).item()["cams"] for i in range(4)]
nodes = [np.array(n["pos"]) for n in np.load(f"data/xm_{band}_0.npy", allow_pickle=True).item()["nodes"]]
err = np.array([r["err"] for r in rows]); n = np.array([r["n"] for r in rows]); sp = np.array([r["spread"] for r in rows])
v = np.array([cams[r["f"]][r["r"]] - nodes[r["n"]] for r in rows]); hd = np.hypot(v[:, 0], v[:, 1]); el = np.degrees(np.arctan2(v[:, 2], hd))
st = lambda e: f"n={len(e):3d}  {e.mean():5.1f} / {np.median(e):5.1f} / {np.percentile(e, 90):5.1f}" if len(e) else "n=0"
print(f"== {tag}: mean / median / P90 of the direction error (deg)")
print("all                                   ", st(err))
far = hd >= 1.0; print(f"camera >= 1 m away horizontally ({100 * far.mean():.0f}%)   ", st(err[far]))
k = sp <= np.percentile(sp[far], 80); print("  ... and the most confident 80%       ", st(err[far & k]))
print("camera closer than 1 m horizontally   ", st(err[~far]), " cross-range miss median %.2f m" % np.median(hd[~far] * np.abs(np.sin(np.radians(np.minimum(err[~far], 90))))))
for s in range(4): m = n == s; print(f"spot {s} {np.round(nodes[s], 2).tolist()}: all {st(err[m])} | >= 1 m away {st(err[m & far])}")
