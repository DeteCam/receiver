#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""시험장 모드(core/locate.py 의 ExamSession) 검사.   python3 tests/test_exam.py
1) 다음 자리 고르기: 두 후보 자리가 한 줄로 겹쳐 보이는 곳보다 옆에서 갈라 보이는 곳을 고르는가
2) 잘못된 입력은 거절하고, 이미 잰 곳·목록 밖의 곳은 권하지 않는가
3) 흉내 낸 측정(골 각에 흔들림 8도, 15% 는 통째로 틀린 골)으로 한국 교실(7.5 x 9m, 40자리)을 둘레로만 6곳 재면
   자리를 옆자리까지 맞히는가. 반사·사람 가림은 sim/location/perimeter2.py 가 본다. 여기서는 장치 코드가 그 계산을 그대로 하는지 본다.
파이(파이썬 3.7, numpy 1.16)에서도 돈다."""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../core",):
    p = os.path.normpath(os.path.join(HERE, d))
    if p not in sys.path:
        sys.path.insert(0, p)
import locate as K
from estimator import frame, to_scan
from fusion import antenna_pos, p_seen

FRAMES18 = [(y, p) for i, p in enumerate((30.0, 54.7356, 78.0)) for y in ((-150.0, -90.0, -30.0, 30.0, 90.0, 150.0) if i % 2 == 0 else (150.0, 90.0, 30.0, -30.0, -90.0, -150.0))]


def classroom():
    xs = 0.625 + np.arange(6) * 1.1; ys = 2.0 + np.arange(7) * 0.95
    seats = [dict(id="%d-%d" % (j + 1, i + 1), x=float(x), y=float(y)) for j, y in enumerate(ys) for i, x in enumerate(xs) if not (j == 6 and i in (0, 5))]
    pts = [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)] + [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + \
          [(x, 8.6) for x in (2.0, 3.75, 5.5)] + [(x, 1.3) for x in (1.5, 3.75, 6.0)]
    spots = [dict(name="곳 %d" % (i + 1), x=x, y=y, z=2.4) for i, (x, y) in enumerate(pts)]
    return seats, spots


def fake_obs(cam, base, head, frames, rng, sig=8.0, bad=0.15):
    """수신기가 base(받침 바닥), head 로 놓였을 때 카메라 cam 의 골 각 (흔들림 sig 도, bad 는 아무 각)"""
    v = K.to_dev(np.asarray(cam, dtype=float)[None, :], (base, head))[0]
    out = []
    for yaw, tilt in frames:
        u = v - antenna_pos(yaw, tilt); u = u / np.linalg.norm(u)
        psi, beta = to_scan(u[None, :], frame(yaw, tilt)); psi = float(psi[0]); beta = float(beta[0])
        if rng.random_sample() > p_seen(np.array([beta]))[0]:
            out.append(dict(seen=False, psi=0.0, depth=0.0)); continue
        psi = rng.uniform(-180, 180) if rng.random_sample() < bad else psi + rng.normal(0, sig)
        out.append(dict(seen=True, psi=float(psi), depth=10.0))
    return out


def main():
    # 1) 다음 자리 고르기
    xy = np.array([[0.0, 0.0], [0.0, 1.0]]); p = np.array([0.5, 0.5])
    in_line, side = K.spread_value(p, xy, (0.0, 5.0)), K.spread_value(p, xy, (3.0, 0.5))
    assert side > 3 * in_line, (side, in_line)
    # 2) 입력 검사, 권하는 곳
    seats, spots = classroom()
    for bad in ((seats, []), ([], spots)):
        try:
            K.ExamSession(*bad); raise AssertionError("잘못된 입력을 받았습니다")
        except ValueError:
            pass
    E = K.ExamSession(seats, spots, (0.0, -1.0)); r = E.result()
    assert r["verdict"] == "none" and r["next"]["index"] == 0 and len(r["top"]) == 3 and len(E.pts.X) == 9 * len(seats), r
    # 3) 흉내 낸 측정으로 6곳
    rng = np.random.RandomState(3); n = 60; near = top3 = exact = 0; verdicts = {}      # numpy 1.16 (파이) 에는 default_rng 가 없다
    for t in range(n):
        s = int(rng.randint(len(seats))); c = seats[s]
        cam = (c["x"] + rng.uniform(-0.3, 0.3), c["y"] - rng.uniform(0.05, 0.3), rng.uniform(0.75, 0.9)) if t % 2 == 0 else (c["x"], c["y"] + 0.01, 1.17)
        E = K.ExamSession(seats, spots, (0.0, -1.0))
        for k in range(K.EXAM_MAX_POS):
            nxt = E.result()["next"]
            if nxt is None:
                break
            i = nxt["index"]; assert i not in E.used and 0 <= i < len(spots)
            head = float(rng.uniform(0, 360)); base = np.array([spots[i]["x"], spots[i]["y"], spots[i]["z"]])
            E.add((base, head), fake_obs(cam, base, head, FRAMES18, rng), FRAMES18, spot=i)
        r = E.result(); verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1
        ids = [q["id"] for q in r["top"]]; rr, cc = [int(v) for v in c["id"].split("-")]; br, bc = [int(v) for v in ids[0].split("-")]
        exact += ids[0] == c["id"]; top3 += c["id"] in ids; near += abs(rr - br) <= 1 and abs(cc - bc) <= 1
    print("흉내 낸 측정 %d번 (6곳까지): 정확히 %.0f%%, 옆자리까지 %.0f%%, 3후보 안 %.0f%% | 판정 %s" % (n, 100.0 * exact / n, 100.0 * near / n, 100.0 * top3 / n, verdicts))
    assert near >= 0.85 * n and top3 >= 0.7 * n, (near, top3)
    print("시험장 모드 검사 통과")


if __name__ == "__main__":
    main()
