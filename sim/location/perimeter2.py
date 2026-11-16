"""교실 둘레로만 옮겨 가며 잴 때, 무엇을 바꾸면 더 좋아지나.   python3 perimeter2.py [횟수]   결과: perimeter2_result.json

기준 (perimeter.py 의 가장 좋은 것): 앞 왼쪽 모서리에서 시작, 의심 자리에 가장 가까운 둘레 자리로 옮김, 한 곳에 12번 재기, 받침 높이 2.4m, 4곳.
하나씩 바꿔 본다:
  - 곳 수: 4 -> 6
  - 한 곳에서 재는 횟수: 12 -> 18 (숙임 30·54.7·78° x 받침 6방향)
  - 받침 높이: 2.4 -> 2.2m
  - 다음 자리 고르기: '가장 가까운 곳' -> '후보 자리들이 가장 벌어져 보이는 곳' (지금 확률이 높은 자리들끼리 그 자리에서 본 각도 차이가 클수록,
    가까울수록 좋다. 그 자리에서 재면 후보끼리 가장 잘 갈린다)
  - 받은 세기도 쓰기: 각 둘레 자리에서 카메라 ACK 의 세기를 같이 재서, 자리끼리의 세기 차이로 거리 비를 더한다
  - 시작 자리: 앞 왼쪽 모서리 -> 뒤 가운데 (교실 전체를 비슷한 거리로 본다)
  - 교탁 자리도 후보로: 칠판과 첫 줄 사이 (감독관이 늘 서는 곳) 3곳을 둘레 자리에 더한다
교실은 kr_classroom.py (한국 교실 규격), 사람 40명, 카메라는 책상 위 / 가슴.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths
import os, sys, json, time
import numpy as np
import kr_classroom as KR, classroom as C, locate3d as L3, locate as K
from fusion import Fusion

HERE = os.path.dirname(os.path.abspath(__file__))
L3.PLANS["tilt12"] = [(y, p) for p in (40.0, 70.0) for y in (-150, -90, -30, 30, 90, 150)]
L3.PLANS["tilt18"] = [(y, p) for p in (30.0, 54.7, 78.0) for y in (-150, -90, -30, 30, 90, 150)]


def perimeter(h, front=False):
    pts = [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)]
    pts += [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(x, 8.6) for x in (2.0, 3.75, 5.5)]
    if front:                                                                     # 교탁 자리 (칠판과 첫 줄 사이, 감독관이 늘 서는 곳)
        pts += [(x, 1.3) for x in (1.5, 3.75, 6.0)]
    return [(x, y, h) for x, y in pts]


def seat_post(grid, ms, rssi):
    if rssi and len(ms) >= 2:
        post = L3.posterior(grid, ms, None, [m["rssi"] for m in ms], K.Q_BAD, K.SIG_SCALE)
    else:
        post = K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE)
    p = np.bincount(grid.owner, post, minlength=len(C.SEATS)); return p / p.sum()


def choose(p, used, PER, how):
    cand = [q for q in PER if q not in used]
    if how == "near":
        b = C.SEATS[int(np.argmax(p))]
        bt = [np.degrees(np.arctan2(b[1] - u[1], b[0] - u[0])) for u in used]
        cand = sorted(cand, key=lambda q: np.hypot(q[0] - b[0], q[1] - b[1]))
        for q in cand:
            a = np.degrees(np.arctan2(b[1] - q[1], b[0] - q[0]))
            if all(abs((a - t + 180) % 360 - 180) >= 40 for t in bt):
                return q
        return cand[0]
    # spread: 확률 높은 자리들(합 90% 까지)이 그 자리에서 얼마나 벌어져 보이나. 각도 차이를 그 자리의 예상 오차(가까울수록 작다)로 나눠 본다
    o = np.argsort(-p); k = int(np.searchsorted(np.cumsum(p[o]), 0.9) + 1); top = o[:max(k, 2)]; w = p[top]
    best, bs = None, -1.0
    for q in cand:
        v = C.SEATS[top] - np.array(q[:2]); az = np.arctan2(v[:, 1], v[:, 0]); d = np.hypot(v[:, 0], v[:, 1])
        sig = np.radians(6.0 + 2.5 * d)                                                   # 거리가 멀수록 방향 오차가 크다 (교실 계산에서 어림)
        da = np.abs((az[:, None] - az[None, :] + np.pi) % (2 * np.pi) - np.pi) / ((sig[:, None] + sig[None, :]) / 2)
        sc = float(np.sum(w[:, None] * w[None, :] * np.minimum(da, 3.0)))
        if sc > bs:
            best, bs = q, sc
    return best


def run(n=80, where="책상 위", npos=4, plan="tilt12", h=2.4, how="near", rssi=False, start=(0.5, 0.5), seed=5, front=False):
    KR.setup(2.7)
    fus = Fusion(); frames = L3.PLANS[plan]; grid = C.SeatGrid(); PER = perimeter(h, front); res = {k: [] for k in range(1, npos + 1)}
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(C.SEATS))); cam, body = C.place(rng, seat, where)
        P = np.array([(*(body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25)), C.BODY_H + rng.uniform(-0.1, 0.1), C.BODY_R + rng.uniform(-0.03, 0.03))
                      for k, (sx, sy) in enumerate(C.SEATS)])
        ms = []; used = []; w0 = None
        for k in range(npos):
            q = (start[0], start[1], h) if k == 0 else choose(seat_post(grid, ms, rssi), used, PER, how)
            w = C.ClassWorld(rng, P, cam=cam, base=np.array(q, float))
            if w0 is None:
                w0 = w
            else:
                w.absorb = w0.absorb
            ms.append(L3.measure(w, fus, rng, frames)); used.append(q)
            p = seat_post(grid, ms, rssi); o = np.argsort(-p); best = int(o[0]); (r0, c0), (r1, c1) = C.rc(seat), C.rc(best)
            res[k + 1].append((best == seat, abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1, seat in o[:3]))
    return {str(k): dict(exact=round(100 * np.mean([x[0] for x in v]), 1), near=round(100 * np.mean([x[1] for x in v]), 1), top3=round(100 * np.mean([x[2] for x in v]), 1)) for k, v in res.items()}


VARIANTS = [("기준: 가까운 곳, 12번, 2.4m", {}),
            ("곳 수 6", dict(npos=6)),
            ("한 곳 18번", dict(plan="tilt18")),
            ("높이 2.2m", dict(h=2.2)),
            ("다음 자리: 후보가 가장 벌어져 보이는 곳", dict(how="spread")),
            ("받은 세기도 쓰기", dict(rssi=True)),
            ("시작: 뒤 가운데", dict(start=(3.75, 8.6))),
            ("벌어져 보이는 곳 + 18번 + 6곳", dict(how="spread", plan="tilt18", npos=6)),
            ("벌어져 보이는 곳 + 18번 + 6곳 + 세기", dict(how="spread", plan="tilt18", npos=6, rssi=True)),
            ("교탁 자리도 후보", dict(front=True)),
            ("교탁 자리도 + 벌어져 보이는 곳 + 18번 + 6곳", dict(front=True, how="spread", plan="tilt18", npos=6))]


def main(n=80, only=None):
    path = os.path.join(HERE, "perimeter2_result.json")
    out = json.load(open(path, encoding="utf-8"))["results"] if only else {}
    for lab, kw in VARIANTS:
        if only and lab not in only:
            continue
        for wh in ("책상 위", "가슴"):
            t0 = time.time(); r = run(n, wh, **kw); out["%s | %s" % (lab, wh)] = r
            ks = sorted(r, key=int)
            print("%-36s %-4s " % (lab, wh) + " | ".join("%s곳 %g/%g/%g" % (k, r[k]["exact"], r[k]["near"], r[k]["top3"]) for k in ks) + " (%.0f초)" % (time.time() - t0), flush=True)
    json.dump(dict(n=n, results=out), open(path, "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
