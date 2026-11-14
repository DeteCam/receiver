"""여러 방향에 카메라를 두고 재는 과정을 흉내 내어, 방향을 얼마나 잘 찾는지 본다.
    python3 montecarlo.py [횟수]        결과: result.json, result_raw.npz

흉내에 넣은 것과 넣지 않은 것은 모형 파일(firmware/core/world.py)의 머리말에 적혀 있다. 수신기의 가짜 모드도 같은 모형을 쓴다.
"""
import sys, os, json, time
import numpy as np
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                               # 계산 코드(core)를 찾는다
from world import *                                        # noqa: F401,F403  (World, plates, cameras, ENV, TILT3, REV_S, BIN)
from world import World, plates, cameras, ENV, TILT3, REV_S, BIN      # noqa: F401
from model import Plate                                    # noqa: F401
from estimator import Solver, unit, azel, frame, to_scan   # noqa: F401
from fusion import Fusion, next_scan

HERE = os.path.dirname(os.path.abspath(__file__))
STRATS = {
    "세워서 1번": dict(frames=[(0.0, 0.0)], adapt=False),
    "기울여 3번": dict(frames=[(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], adapt=False),
    "기울여 3번 (+필요하면 1번 더)": dict(frames=[(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], adapt=True),
    "기울여 4번": dict(frames=[(-135.0, 60.0), (-45.0, 60.0), (45.0, 60.0), (135.0, 60.0)], adapt=False),
    "세워서 1번 + 기울여 3번": dict(frames=[(0.0, 0.0), (-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], adapt=False),
    "기울여 6번": dict(frames=[(-150.0, TILT3), (-90.0, TILT3), (-30.0, TILT3), (30.0, TILT3), (90.0, TILT3), (150.0, TILT3)], adapt=False),
}


def run(n=1500, shape="W", env="보통", seed=7, strategies=None, estimator="fusion", quiet=False):
    cams = cameras(n, seed)
    fus = Fusion(); sol = Solver(plates()[1]) if estimator == "template" else None
    out = {}
    for name, st in STRATS.items():
        if strategies and name not in strategies:
            continue
        t0 = time.time(); rows = []
        for i, c in enumerate(cams):
            w = World(np.random.default_rng(seed * 100003 + i), c, shape, env)
            frames = list(st["frames"]); curves = [w.scan(y, t) for y, t in frames]
            if estimator == "template":
                est, _, _ = sol.solve(curves, frames); nseen = len(frames); nscan = len(frames)
            else:
                est, obs = fus.solve(curves, frames)
                nseen = sum(o["seen"] for o in obs)
                if st["adapt"] and nseen == 1:             # 한 스캔에서만 보였으면 그 방향을 가로질러 한 번 더 잰다
                    y, t = next_scan(obs, frames, frames[0][1])
                    frames.append((y, t)); curves.append(w.scan(y, t))
                    est, obs = fus.solve(curves, frames); nseen = sum(o["seen"] for o in obs)
                nscan = len(frames)
            az, el = azel(c)
            if est is None:
                rows.append((float(el), 180.0, 180.0, 90.0, 0, nscan)); continue
            if len(st["frames"]) == 1:                     # 한 번만 재면 높이를 알 수 없다 -> 수평이라고 본다
                eaz, _ = azel(est); est = unit(float(eaz), 0.0)
            eaz, eel = azel(est)
            ang = float(np.degrees(np.arccos(np.clip(est @ c, -1, 1))))
            rows.append((float(el), ang, float((eaz - az + 180) % 360 - 180), float(eel - el), nseen, nscan))
        out[name] = np.array(rows)
        if not quiet:
            R = out[name]
            print("[%s·%s] %-24s 가운데값 %5.1f도, 90%% %5.1f도, 30도 넘음 %4.1f%%, 평균 스캔 %.2f번 (%.0f초)" % (
                shape, env, name, np.median(R[:, 1]), np.percentile(R[:, 1], 90), np.mean(R[:, 1] > 30) * 100, R[:, 5].mean(), time.time() - t0))
    return cams, out


BANDS = [(-80, -50, "바닥 쪽 (-80~-50도)"), (-50, -20, "낮은 곳 (-50~-20도)"), (-20, 20, "수평 근처 (-20~20도)"), (20, 50, "높은 곳 (20~50도)"), (50, 90, "천장 쪽 (50~90도)")]


def summarize(out):
    S = {}
    for name, R in out.items():
        d = {}
        for lo, hi, label in BANDS + [(-80, 90, "전체")]:
            m = (R[:, 0] >= lo) & (R[:, 0] < hi)
            if m.sum() < 5:
                continue
            horiz = np.abs(R[m, 2]) * np.cos(np.radians(R[m, 0]))
            d[label] = dict(n=int(m.sum()), med=round(float(np.median(R[m, 1])), 1), p90=round(float(np.percentile(R[m, 1], 90)), 1),
                            over30=round(float(np.mean(R[m, 1] > 30) * 100), 1), az_med=round(float(np.median(horiz)), 1), az_mean=round(float(np.mean(np.minimum(horiz, 180))), 1),
                            el_med=round(float(np.median(np.abs(R[m, 3]))), 1), scans=round(float(R[m, 5].mean()), 2))
        S[name] = d
    return S


def run_only(name, n=1500):
    """전략 하나만 돌려서 이미 있는 result.json 에 끼워 넣는다.  python3 montecarlo.py --only "전략 이름" """
    path = os.path.join(HERE, "result.json"); R = json.load(open(path, encoding="utf-8"))
    z = np.load(os.path.join(HERE, "result_raw.npz"), allow_pickle=False)
    raw = {str(k): z["r%d" % i] for i, k in enumerate(z["keys"])}
    for shape in ("W", "V"):
        for env in ("보통", "나쁨"):
            cams, out = run(n, shape, env, strategies=(name,))
            R["summary"]["%s·%s" % (shape, env)].update(summarize(out))
            raw["%s|%s|%s" % (shape, env, name)] = out[name]
    json.dump(R, open(path, "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(HERE, "result_raw.npz"), **{("r%d" % i): v for i, v in enumerate(raw.values())}, keys=np.array(list(raw.keys())))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--only":
        run_only(sys.argv[2]); sys.exit(0)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    res = {}; raw = {}
    for shape in ("W", "V"):
        for env in ("보통", "나쁨"):
            cams, out = run(n, shape, env)
            res["%s·%s" % (shape, env)] = summarize(out)
            for k, R in out.items():
                raw["%s|%s|%s" % (shape, env, k)] = R
    # 참고: 곡선 모양을 정확히 알 때(보정 스캔으로 표를 만든 뒤)의 결과. 곡선 전체를 모형 표에 맞춘다 (estimator.py)
    nt = min(n, 500)
    for env in ("보통", "나쁨"):
        cams, out = run(nt, "W", env, strategies=("기울여 3번", "기울여 4번"), estimator="template")
        res["W·%s·곡선 전체 맞춤" % env] = summarize(out)
        for k, R in out.items():
            raw["W|%s|%s|template" % (env, k)] = R
    json.dump(dict(n=n, n_template=nt, summary=res), open(os.path.join(HERE, "result.json"), "w"), ensure_ascii=False, indent=1)
    np.savez_compressed(os.path.join(HERE, "result_raw.npz"), **{("r%d" % i): v for i, v in enumerate(raw.values())}, keys=np.array(list(raw.keys())))
    for key, S in res.items():
        print("\n===== 골 모양·환경: %s" % key)
        for name, d in S.items():
            print("  -- %s" % name)
            for label, v in d.items():
                print("     %-20s n=%4d  방향 오차 가운데값 %5.1f도, 90%% %5.1f도, 30도 넘음 %5.1f%% | 좌우 %4.1f도(평균 %4.1f), 높이 %4.1f도" % (
                    label, v["n"], v["med"], v["p90"], v["over30"], v["az_med"], v["az_mean"], v["el_med"]))
