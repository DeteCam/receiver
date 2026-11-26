"""Section 6 of plan_20261126_plate.md: ring (2 wall spots + nod, H4, 2.4 GHz) vs freely placed plate (F2/F3/F5/F9).
Usage: python3 mysim/analyze_plate_pos.py [seeds]"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as W
from analyze_bearing import boot

R = "/opt/detecam/sim/results/"


def summ(e):
    return (f"0.5 m 안 {100 * (e <= 0.5).mean():.0f}% · 1 m 안 {100 * (e <= 1.0).mean():.0f}% · 평균 {e.mean():.2f} · "
            f"중앙 {np.median(e):.2f} · 90% {np.percentile(e, 90):.2f} · 95% {np.percentile(e, 95):.2f} m")


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    ring = pickle.load(open(R + "exp_final_anchor_outside.pkl", "rb"))
    plate = {}
    for f in glob.glob(R + "exp_plate_pos_s*.pkl"):
        plate.update(pickle.load(open(f, "rb")))
    keys = sorted(k for k in plate if k[1] in seeds and (k[0], "2g4", k[1]) in ring)
    if not keys:
        print("no data"); return
    room = np.concatenate([[k[0]] * 40 for k in keys]); typ = np.array([W.TYPE[r] for r in room])
    er = np.concatenate([ring[(k[0], "2g4", k[1])][1]["H4"]["pos_err"]["nod_moppA_equal"] for k in keys])
    print(f"## 시험실 밖, 2.4 GHz, 시드 {sorted(set(k[1] for k in keys))}, 방 {len(set(room))}개, 카메라 {len(er)}")
    print("   고리 벽 2곳(m0·m2)+끄덕이기: " + summ(er))
    print("      종류별 0.5 m 안: " + " · ".join(f"{t} {100 * (er[typ == t] <= 0.5).mean():.0f}%" for t in dict.fromkeys(typ)))
    for pn, desc in (("W2", "벽 2곳 m0·m2 (고리와 같은 자리)"), ("F2", "방 안 2곳"), ("F3", "방 안 3곳"), ("F5", "방 안 5곳"), ("F9", "방 안 5곳 + 벽 4곳")):
        ep = np.concatenate([plate[k][1]["pos"][pn] for k in keys])
        nsp = {"W2": 2, "F2": 2, "F3": 3, "F5": 5, "F9": 9}[pn]
        print(f"   돌판 {pn} {desc}, 재는 시간 약 {nsp * 110 / 60:.0f}분: " + summ(ep))
        print("      종류별 0.5 m 안: " + " · ".join(f"{t} {100 * (ep[typ == t] <= 0.5).mean():.0f}%" for t in dict.fromkeys(typ)))
        d, lo, hi = boot(100 * ((er <= 0.5).astype(float) - (ep <= 0.5).astype(float)), room)
        verdict = "고리가 낫다" if lo > 0 else ("돌판이 낫다" if hi < 0 else "비슷함")
        print(f"      0.5 m 안 짝 차이 고리 − 돌판: {d:+.1f}%p [{lo:+.1f}, {hi:+.1f}] → {verdict}")
        dm, lom, him = boot(er - ep, room)
        print(f"      위치 오차 짝 차이 고리 − 돌판: {dm:+.2f} m [{lom:+.2f}, {him:+.2f}]")
    mf = R + "exp_ringmulti.pkl"
    if os.path.exists(mf):
        rm = pickle.load(open(mf, "rb"))
        if all((k[0], "2g4", k[1]) in rm for k in keys):
            for gn, nsp in (("nod_W4", 4), ("nod_W8", 8)):
                eg = np.concatenate([rm[(k[0], "2g4", k[1])][1]["H4"]["pos_err"][gn + "_equal"] for k in keys])
                print(f"   (7절, 사후 추가) 고리 벽 {nsp}곳+끄덕이기: " + summ(eg))
                print("      종류별 0.5 m 안: " + " · ".join(f"{t} {100 * (eg[typ == t] <= 0.5).mean():.0f}%" for t in dict.fromkeys(typ)))
                for pn in ("F3", "F5", "F9"):
                    ep = np.concatenate([plate[k][1]["pos"][pn] for k in keys])
                    d, lo, hi = boot(100 * ((eg <= 0.5).astype(float) - (ep <= 0.5).astype(float)), room)
                    v = "고리가 낫다" if lo > 0 else ("돌판이 낫다" if hi < 0 else "비슷함")
                    print(f"      대 돌판 {pn}: 0.5 m 안 짝 차이 {d:+.1f}%p [{lo:+.1f}, {hi:+.1f}] → {v}")
    # plate bearing at each spot (one pose per spot)
    print("   돌판 자리별 방위각 오차 평균/중앙 (도): " + " · ".join(
        f"{lab} {np.concatenate([plate[k][1]['az'][lab] for k in keys]).mean():.1f}/{np.median(np.concatenate([plate[k][1]['az'][lab] for k in keys])):.1f}"
        for lab in ("m0", "m2", "m1", "m3", "f0", "f1", "f2", "f3", "f4")))


if __name__ == "__main__":
    main()
