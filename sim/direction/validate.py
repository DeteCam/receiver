"""모형을 손으로 잰 값과 맞춰 본다. 손 측정(2026-11-11): 판 200 x 200 을 와이파이 모듈에서 10cm 에 놓고 각도마다 3초씩, 방 안 1m.
측정값은 data/scan_*/ 의 pcap 에서 각도별 가운데값(dB, AGC 보정)을 뽑은 것이다 (hand_scans.json)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import os, json
import numpy as np
from model import field
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "hand_scans.json"), encoding="utf-8"))


def model_curve(dphis, dist=1000.0):
    return 20 * np.log10(np.abs(field(dphis, 0.0, R=100.0, w=200.0, z0=-100.0, z1=100.0, dist=dist, step=5.0)))


def fit(rows, base=None):
    ang = np.array([float(r["angle"]) for r in rows]); lv = np.array([r["lvl"] for r in rows])
    grid = np.arange(-180.0, 180.0, 1.0); mc = model_curve(grid)
    best = None
    for psi in np.arange(0, 360, 1.0):
        m = np.interp(((ang - psi + 180) % 360) - 180, grid, mc)
        off = np.median(lv - m) if base is None else base
        e = float(np.sqrt(np.mean((lv - m - off) ** 2)))
        if best is None or e < best[0]:
            best = (e, psi, off, m)
    return best, ang, lv


if __name__ == "__main__":
    grid = np.arange(-90, 91, 15.0); mc = model_curve(grid)
    print("모형 (판 200 x 200, 10cm, 1m): 가운데 %.1f dB, 가장 깊은 곳 %.1f dB" % (model_curve(np.array([0.0]))[0], model_curve(np.arange(-90, 91, 1.0)).min()))
    print("   각도별:", " ".join("%+d:%.1f" % (a, v) for a, v in zip(grid, mc)))
    out = {}
    for key, v in D.items():
        rows = [r for r in v["rows"] if r["angle"] != "base"]
        base = [r["lvl"] for r in v["rows"] if r["angle"] == "base"]
        (e, psi, off, m), ang, lv = fit(rows)
        ref = base[0] if base else float(np.median(lv))
        print("\n%s  (%s)" % (key, v["meta"].get("note", "")))
        print("   가장 잘 맞는 방향 %d도%s, 모형과의 차이(RMS) %.1f dB" % (psi, " (실제 %s도)" % v["meta"]["truth"] if "truth" in v["meta"] else "", e))
        print("   잰 값에서 가장 깊은 곳: 기준보다 %.1f dB (모형 %.1f dB)" % (lv.min() - ref, m.min()))
        for a, l, mm in sorted(zip(ang, lv, m)):
            print("      %5.0f도  잰 값 %+6.1f   모형 %+6.1f" % (a, l - ref, mm + off - ref))
        out[key] = dict(psi=float(psi), rms=e, depth_meas=float(lv.min() - ref), depth_model=float(m.min()),
                        ang=[float(x) for x in ang], meas=[float(x - ref) for x in lv], model=[float(x + off - ref) for x in m])
    json.dump(out, open(os.path.join(HERE, "validate.json"), "w"), ensure_ascii=False, indent=1)
