"""시험 중에 감독관이 수신기 1대를 들고 교실 둘레(네 모서리, 옆벽·뒷벽)로만 옮겨 가며 잴 때 (자리 사이로는 들어가지 않는다).   python3 perimeter.py [횟수]   결과: perimeter_result.json

교실·사람·카메라는 kr_classroom.py (한국 교실 규격). 견주는 것:
  - corners : 네 모서리를 차례로 (앞 왼쪽 -> 앞 오른쪽 -> 뒤 왼쪽 -> 뒤 오른쪽)
  - adaptive: 앞 왼쪽 모서리에서 한 번 잰 뒤, 그때까지 가장 그럴듯한 자리에 가장 가까운 둘레 자리로 옮긴다
              (이미 잰 자리들과 그 자리를 보는 방향이 40° 넘게 달라야 한다). 둘레 자리: 네 모서리 + 옆벽 4곳씩 + 뒷벽 3곳
  - 재는 횟수: tilt3 (3번) / tilt12 (숙임 40°·70° x 받침 6방향 = 12번)
  - 수신기 높이: 2.4m (조명 스탠드 같은 받침대 위) / 2.0m
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths
import os, sys, json, time
import numpy as np
import kr_classroom as KR, classroom as C, locate3d as L3, locate as K
from fusion import Fusion
L3.PLANS["tilt12"] = [(y, p) for p in (40.0, 70.0) for y in (-150, -90, -30, 30, 90, 150)]
KR.setup(2.7)
def perimeter(h):
    W, D = KR.W, KR.D; pts = [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)]
    pts += [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(x, 8.6) for x in (2.0, 3.75, 5.5)]
    return [(x, y, h) for x, y in pts]
def run(n, mode, plan, h, where, seed=5, npos=4):
    fus = Fusion(); frames = L3.PLANS[plan]; grid = C.SeatGrid(); res = {k: [] for k in range(1, npos + 1)}
    PER = perimeter(h)
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(C.SEATS))); cam, body = C.place(rng, seat, where)
        P = np.array([(*(body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25)), C.BODY_H + rng.uniform(-0.1, 0.1), C.BODY_R + rng.uniform(-0.03, 0.03)) for k, (sx, sy) in enumerate(C.SEATS)])
        ms = []; w0 = None; used = []
        def at(b):
            nonlocal w0
            w = C.ClassWorld(rng, P, cam=cam, base=np.array(b, float))
            if w0 is None: w0 = w
            else: w.absorb = w0.absorb
            ms.append(L3.measure(w, fus, rng, frames)); used.append(b)
        for k in range(npos):
            if mode == "corners":
                at([(0.5, 0.5, h), (7.0, 0.5, h), (0.5, 8.6, h), (7.0, 8.6, h)][k])
            else:
                if k == 0:
                    at((0.5, 0.5, h))
                else:
                    p = np.bincount(grid.owner, K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE), minlength=len(C.SEATS)); b = C.SEATS[int(np.argmax(p))]
                    bt = [np.degrees(np.arctan2(b[1] - u[1], b[0] - u[0])) for u in used]
                    cand = sorted([q for q in PER if q not in used], key=lambda q: np.hypot(q[0] - b[0], q[1] - b[1]))
                    pick = None
                    for q in cand:
                        a = np.degrees(np.arctan2(b[1] - q[1], b[0] - q[0]))
                        if all(abs((a - t + 180) % 360 - 180) >= 40 for t in bt): pick = q; break
                    at(pick or cand[0])
            p = np.bincount(grid.owner, K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE), minlength=len(C.SEATS)); order = np.argsort(-p); best = int(order[0])
            (r0, c0), (r1, c1) = C.rc(seat), C.rc(best)
            res[k + 1].append((best == seat, abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1, seat in order[:3]))
    return {k: tuple(round(100 * np.mean([x[j] for x in v]), 0) for j in range(3)) for k, v in res.items()}
HERE = os.path.dirname(os.path.abspath(__file__))


def main(n=80):
    out = {}
    for mode, plan, h in (("corners", "tilt3", 2.4), ("corners", "tilt12", 2.4), ("adaptive", "tilt3", 2.4), ("adaptive", "tilt12", 2.4), ("adaptive", "tilt12", 2.0)):
        for wh in ("책상 위", "가슴"):
            t0 = time.time(); r = run(n, mode, plan, h, wh); out["%s | %s | %.1f | %s" % (mode, plan, h, wh)] = {str(k): dict(exact=v[0], near=v[1], top3=v[2]) for k, v in r.items()}
            print("%-9s %-6s %.1fm %-4s " % (mode, plan, h, wh) + " | ".join("%d곳 %g/%g/%g" % (k, *v) for k, v in r.items()) + " (%.0f초)" % (time.time() - t0), flush=True)
    json.dump(dict(n=n, results=out), open(os.path.join(HERE, "perimeter_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
