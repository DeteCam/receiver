"""장치의 시험장 모드(firmware/core/locate.py 의 ExamSession)를 교실 흉내(반사, 앉은 사람 40명)에 그대로 넣어 본다.   python3 exam_mode_check.py [횟수]
결과: exam_mode_check_result.json

perimeter2.py 의 가장 좋은 방법(교탁 자리도 후보, 후보가 가장 벌어져 보이는 곳, 한 곳 18번, 6곳)을 장치 코드가 그대로 해내는지 본다.
장치 코드는 '좁혀짐'(4곳 넘게 재고 후보 3자리 합 60% 넘음)이면 멈춘다. 그래서 몇 곳에서 멈췄는지도 센다.
멈추지 않고 6곳까지 재는 것도 같이 본다 (멈추는 규칙이 너무 이른지).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths
import os, sys, json, time
import numpy as np
import kr_classroom as KR, classroom as C, locate3d as L3
import locate as K                                   # 장치 코드 (firmware/core)
from fusion import Fusion

HERE = os.path.dirname(os.path.abspath(__file__))
L3.PLANS["tilt18"] = [(y, p) for i, p in enumerate((30.0, 54.7356, 78.0)) for y in ((-150.0, -90.0, -30.0, 30.0, 90.0, 150.0) if i % 2 == 0 else (150.0, 90.0, 30.0, -30.0, -90.0, -150.0))]


def setup():
    KR.setup(2.7)
    seats = [dict(id=str(k), x=float(x), y=float(y)) for k, (x, y) in enumerate(C.SEATS)]
    pts = [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)] + [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + \
          [(x, 8.6) for x in (2.0, 3.75, 5.5)] + [(x, 1.3) for x in (1.5, 3.75, 6.0)]
    spots = [dict(name="곳 %d" % (i + 1), x=x, y=y, z=2.4) for i, (x, y) in enumerate(pts)]
    return seats, spots


def run(n=100, where="책상 위", stop=True, seed=5):
    seats, spots = setup(); fus = Fusion(); frames = L3.PLANS[K.EXAM_PLAN]; rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(C.SEATS))); cam, body = C.place(rng, seat, where)
        P = np.array([(*(body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25)), C.BODY_H + rng.uniform(-0.1, 0.1), C.BODY_R + rng.uniform(-0.03, 0.03))
                      for k, (sx, sy) in enumerate(C.SEATS)])
        E = K.ExamSession(seats, spots, (0.0, -1.0)); w0 = None
        for k in range(K.EXAM_MAX_POS):
            r = E.result(); nxt = r["next"]
            if nxt is None and (stop or E.next_spot() is None):
                break
            idx = nxt["index"] if nxt is not None else E.next_spot()
            q = spots[idx]
            w = C.ClassWorld(rng, P, cam=cam, base=np.array([q["x"], q["y"], q["z"]]))
            if w0 is None:
                w0 = w
            else:
                w.absorb = w0.absorb
            m = L3.measure(w, fus, rng, frames)
            E.add(m["pose"], m["obs"], m["frames"], None, spot=idx)
        r = E.result(); ids = [t["id"] for t in r["top"]]; best = int(ids[0])
        (r0, c0), (r1, c1) = C.rc(seat), C.rc(best)
        rows.append(dict(n=r["positions"], verdict=r["verdict"], exact=best == seat, near=abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1, top3=str(seat) in ids))
    return dict(exact=round(100 * np.mean([x["exact"] for x in rows]), 1), near=round(100 * np.mean([x["near"] for x in rows]), 1),
                top3=round(100 * np.mean([x["top3"] for x in rows]), 1), positions=round(float(np.mean([x["n"] for x in rows])), 2),
                verdicts=dict((v, sum(1 for x in rows if x["verdict"] == v)) for v in set(x["verdict"] for x in rows)))


def main(n=100):
    out = {}
    for stop in (True, False):
        for wh in ("책상 위", "가슴"):
            t0 = time.time(); s = run(n, wh, stop); key = "%s | %s" % ("좁혀지면 멈춤" if stop else "6곳까지", wh); out[key] = s
            print("%-14s %-4s 정확 %4.1f%% 옆자리까지 %4.1f%% 3후보 %4.1f%% | 평균 %.1f곳, 판정 %s (%.0f초)" % (
                "좁혀지면 멈춤" if stop else "6곳까지", wh, s["exact"], s["near"], s["top3"], s["positions"], s["verdicts"], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=out), open(os.path.join(HERE, "exam_mode_check_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
