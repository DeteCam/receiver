"""교실에서 재는 과정을 한 번 돌려, 곳마다의 신호 곡선과 자리 확률을 demo_run.json 에 적는다 (동작 시뮬레이션 영상의 재료).

    python demo_run.py [--seed 1003] [--kind 책상]

rt_eval.trial 과 같은 과정이다. 다만 통계를 내는 대신 한 번의 진행을 그대로 남긴다.
"""
import os, json, argparse
import numpy as np
import rt_eval as R
from rt_eval import K

HERE = os.path.dirname(os.path.abspath(__file__))


def run(seed, kind, name="kr40_h24_s1", npos=6, plan="tilt18", start=16):
    R._init(160.0); D = R._load(name); pat, fus, tm = R._G["pat"], R._G["fus"], R._G["tm"]
    rng = np.random.default_rng(seed); frames = R.PLANS[plan]; seats_xy = D["seats"]; empty = set(D["empty"].tolist()) if "empty" in D else set()
    while True:
        seat = int(rng.integers(len(seats_xy)))
        if seat not in empty:
            break
    c = seat * 4 + int(rng.choice(R.KIND[kind])); u = rng.normal(size=3); u /= np.linalg.norm(u)
    seats = [dict(id=str(k), x=float(x), y=float(y) + 0.225) for k, (x, y) in enumerate(seats_xy)]
    spots = [dict(name="곳 %d" % i, x=float(p[0]), y=float(p[1]), z=float(p[2])) for i, p in enumerate(D["spots"])]
    E = K.ExamSession(seats, spots, (0.0, -1.0)); logs = []; tlogs = []; steps = []
    for k in range(npos):
        idx = start if k == 0 else E.next_spot(R.post_seat(E.pts, logs))
        if idx is None:
            break
        w = R.RTWorld(rng, D, idx, c, frames, pat, u)
        both = [w.scan(y, t) for y, t in frames]; curves = [b[0] for b in both]; pcurves = [b[1] for b in both]; obs = fus.read(curves)
        pose = (w.base + rng.normal(0, R.POSE["pos"], 3) * np.array([1, 1, 0.5]), w.head + rng.normal(0, R.POSE["yaw"]))
        E.add(pose, obs, frames, None, spot=idx); Xd = K.to_dev(E.pts.X, pose)
        tlogs.append(K.score_points(Xd, obs, frames, K.SIG_SCALE * (K.SIG_MANY if len(frames) >= K.MANY_N else 1.0)))
        a, b = tm.score(Xd, [curves, pcurves], frames); logs.append(b[1])
        p = R.post_seat(E.pts, logs); pt = R.post_seat(E.pts, tlogs); o = np.argsort(-p)
        lv = [np.where(np.isfinite(pc[1]), pc[1], np.nan).tolist() for pc in pcurves[:3]]
        steps.append(dict(spot=int(idx), xy=[float(D["spots"][idx][0]), float(D["spots"][idx][1])], post=[round(float(v), 4) for v in p], post_trough=[round(float(v), 4) for v in pt],
                          top3=[int(i) for i in o[:3]], curves=[[None if (v != v) else round(float(v), 2) for v in row] for row in lv], frames=[[float(a_), float(b_)] for a_, b_ in frames[:3]]))
    return dict(seed=seed, kind=kind, data=name, seat=seat, cam=[float(v) for v in D["cams"][c]], seats=[[float(x), float(y)] for x, y in seats_xy], empty=sorted(int(e) for e in empty),
                spots=[[float(p[0]), float(p[1]), float(p[2])] for p in D["spots"]], room=[R.S.W, R.S.D, R.S.CEIL] if hasattr(R, "S") else [7.5, 9.0, 2.7], scans=len(frames), steps=steps)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1003); ap.add_argument("--kind", default="책상"); ap.add_argument("--try", dest="tries", type=int, default=1)
    a = ap.parse_args()
    for s in range(a.seed, a.seed + a.tries):
        r = run(s, a.kind); ok = r["steps"][-1]["top3"][0] == r["seat"]
        print("seed", s, "자리", r["seat"], "곳", [st["spot"] for st in r["steps"]], "1순위", [st["top3"][0] for st in r["steps"]], "확률", [round(st["post"][r["seat"]], 2) for st in r["steps"]], "맞음" if ok else "틀림", flush=True)
        if a.tries == 1:
            json.dump(r, open(os.path.join(HERE, "demo_run.json"), "w"), ensure_ascii=False)
