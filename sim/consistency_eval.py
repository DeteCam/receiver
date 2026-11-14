"""'스캔끼리 맞는지' 점검(fusion.consistency)이 크게 틀린 결과를 얼마나 걸러 내는지 본다.
    python3 consistency_eval.py [횟수]        결과: consistency.json
재는 방법은 '기울여 3번 (+필요하면 1번 더)'. 점검 통과 = 골이 두 스캔 이상에서 보였고, 찾은 방향이 모든 스캔과 맞는다."""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json
import numpy as np
import montecarlo as mc
from fusion import Fusion, next_scan, consistency

HERE = os.path.dirname(os.path.abspath(__file__))


def run(n=1500, seed=7):
    fus = Fusion(); out = {}
    for shape in ("W", "V"):
        for env in ("보통", "나쁨"):
            cams = mc.cameras(n, seed); st = mc.STRATS["기울여 3번 (+필요하면 1번 더)"]
            rows = []
            for i, c in enumerate(cams):
                w = mc.World(np.random.default_rng(seed * 100003 + i), c, shape, env)
                frames = list(st["frames"]); curves = [w.scan(y, t) for y, t in frames]
                est, obs = fus.solve(curves, frames); nseen = sum(o["seen"] for o in obs)
                if nseen == 1:
                    y, t = next_scan(obs, frames, frames[0][1]); frames.append((y, t)); curves.append(w.scan(y, t))
                    est, obs = fus.solve(curves, frames); nseen = sum(o["seen"] for o in obs)
                if est is None:
                    rows.append((180.0, 0)); continue
                ok, _ = consistency(est, obs, frames)
                rows.append((float(np.degrees(np.arccos(np.clip(est @ c, -1, 1)))), int(ok and nseen >= 2)))
            R = np.array(rows); bad = R[:, 0] > 30; flag = R[:, 1] == 0
            d = dict(n=n, fail=float(bad.mean() * 100), flagged=float(flag.mean() * 100), fail_when_pass=float(bad[~flag].mean() * 100), med_when_pass=float(np.median(R[~flag, 0])),
                     fail_when_flagged=float(bad[flag].mean() * 100), caught=float((bad & flag).sum() / max(bad.sum(), 1) * 100))
            out["%s·%s" % (shape, env)] = {k: round(v, 1) for k, v in d.items()}
            print("%s·%s: 크게 틀림 %.1f%% | 점검에 걸림 %.1f%% | 통과한 것 중 크게 틀림 %.1f%% (가운데값 %.1f도) | 걸린 것 중 크게 틀림 %.1f%% | 크게 틀린 것 중 걸러 낸 비율 %.0f%%" % (
                shape, env, d["fail"], d["flagged"], d["fail_when_pass"], d["med_when_pass"], d["fail_when_flagged"], d["caught"]))
    return out


if __name__ == "__main__":
    out = run(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
    json.dump(out, open(os.path.join(HERE, "consistency.json"), "w"), ensure_ascii=False, indent=1)
