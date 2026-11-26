"""Ring vs legacy plate, single-receiver azimuth error (plan sim/plan_20261126_plate.md).
Usage: python3 mysim/analyze_bearing.py [seeds]   (default 0,1,2)"""
import sys, os, glob, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_wall as W

R = "/opt/detecam/sim/results/"
NODE = {"c0": 0, "c2": 2, "m0": 4, "m2": 6}


def load(f):
    return pickle.load(open(f, "rb"))


def stats(e):
    return dict(n=len(e), mean=e.mean(), med=np.median(e), p90=np.percentile(e, 90), p95=np.percentile(e, 95),
                over15=100 * (e > 15).mean())


def fmt(d):
    return (f"n={d['n']} 평균 {d['mean']:.1f}° 중앙 {d['med']:.1f}° 90% {d['p90']:.1f}° 95% {d['p95']:.1f}° "
            f"15° 넘음 {d['over15']:.0f}%")


def boot(dv, room, n=2000, seed=0):
    rr = sorted(set(room)); idx = {r: np.where(room == r)[0] for r in rr}; rng = np.random.default_rng(seed)
    bs = [dv[np.concatenate([idx[p] for p in rng.choice(rr, len(rr))])].mean() for _ in range(n)]
    return dv.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def collect(seeds, labels, band="2g4"):
    ring = load(R + "exp_bearing_ring.pkl")
    plate = {}
    for f in glob.glob(R + "exp_bearing_plate*.pkl"):
        plate.update(load(f))
    rows = []
    for (sc, b, s), (meta, out) in sorted(ring.items()):
        if b != band or s not in seeds:
            continue
        o = out["H4"]
        pk = [k for k in plate if k[0] == sc and k[1] == band and k[2] == s and set(labels) <= set(k[3])]
        for lab in labels:
            rn = o["extra"]["sg_" + lab][:, 0]
            r1 = o["one"][:, NODE[lab]]
            if pk:
                pm, po = plate[pk[0]]; j = list(pm["labels"]).index(lab)
                pg, ps = po["az_grid"][:, j], po["az_sph"][:, j]
            else:
                pg = ps = np.full(len(rn), np.nan)
            for c in range(len(rn)):
                rows.append((sc, s, lab, c, rn[c], r1[c], pg[c], ps[c]))
    return rows


def main():
    seeds = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [0, 1, 2]
    for labels, name in ((["m0", "m2"], "맞은편 벽 가운데 m0·m2"), (["c0", "c2"], "대각 모서리 c0·c2")):
        rows = collect(seeds, labels)
        if not rows:
            continue
        sc = np.array([r[0] for r in rows]); sd = np.array([r[1] for r in rows])
        A = np.array([r[4:] for r in rows], float)       # ring nod, ring 1 look, plate grid, plate sphere
        ok = np.isfinite(A[:, 2])
        print(f"\n## 2.4 GHz {name} 시드 {seeds} (돌판 있는 카메라 {ok.sum()}/{len(ok)})")
        for k, nm in ((0, "고리 끄덕이기 3자세"), (1, "고리 1자세"), (2, "돌판 tilt18 (방 격자)"), (3, "돌판 tilt18 (6 m 구면, 레거시 읽기)")):
            e = A[ok, k] if k >= 2 else A[ok, k]
            if len(e):
                print(f"   {nm}: " + fmt(stats(e)))
        if ok.sum() == 0:
            continue
        typ = np.array([W.TYPE[x] for x in sc])
        d, lo, hi = boot(A[ok, 0] - A[ok, 2], sc[ok])
        pm = A[ok, 2].mean()
        better = (d <= -0.10 * pm) and (hi < 0)
        print(f"   짝 차이 고리(끄덕이기) − 돌판: {d:+.1f}° [{lo:+.1f}, {hi:+.1f}] (돌판 평균의 {100 * d / pm:+.0f}%) → "
              + ("고리가 낫다" if better else "낮거나 비슷함 → 돌판 전환 조건"))
        d1, lo1, hi1 = boot(A[ok, 1] - A[ok, 2], sc[ok])
        print(f"   참고 고리 1자세 − 돌판: {d1:+.1f}° [{lo1:+.1f}, {hi1:+.1f}]")
        print("   종류별 평균 (고리 끄덕이기 / 돌판): " + " · ".join(
            f"{t} {A[ok & (typ == t), 0].mean():.1f}/{A[ok & (typ == t), 2].mean():.1f}" for t in dict.fromkeys(typ) if (ok & (typ == t)).any()))
        print("   시드별 평균 (고리 / 돌판): " + " · ".join(
            f"s{s} {A[ok & (sd == s), 0].mean():.1f}/{A[ok & (sd == s), 2].mean():.1f}" for s in seeds if (ok & (sd == s)).any()))
        rm = sorted(((A[ok & (sc == r), 0].mean(), A[ok & (sc == r), 2].mean(), r) for r in set(sc[ok])), reverse=True)[:3]
        print("   고리가 가장 나쁜 방: " + ", ".join(f"{r} {a:.1f}° (돌판 {b:.1f}°)" for a, b, r in rm))
        rp = sorted(((A[ok & (sc == r), 2].mean(), A[ok & (sc == r), 0].mean(), r) for r in set(sc[ok])), reverse=True)[:3]
        print("   돌판이 가장 나쁜 방: " + ", ".join(f"{r} {a:.1f}° (고리 {b:.1f}°)" for a, b, r in rp))
    ring = load(R + "exp_bearing_ring.pkl")
    for labels, name in ((["m0", "m2"], "m0·m2"), (["c0", "c2"], "c0·c2")):
        e = [ring[k][1]["H4"]["extra"]["sg_" + l][:, 0] for k in ring if k[1] == "5g" and k[2] in seeds for l in labels]
        if e:
            print(f"\n## 참고: 고리 5 GHz {name} 끄덕이기: " + fmt(stats(np.concatenate(e))))


if __name__ == "__main__":
    main()
