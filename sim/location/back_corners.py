"""수신기를 칠판 반대쪽 뒤 두 모서리에만 둘 때, 40명 교실에서 카메라 자리를 좁히는 방법들을 견준다.   python3 back_corners.py [횟수]   결과: back_corners_result.json

교실·자리·사람·카메라는 classroom.py 와 같다. 바꿔 보는 것:
  ① 뒤 두 모서리, 높이 1.9m / 2.6m (사물함 위, 벽에 단 선반)
  ② ① + 여섯 번 재기 (기울여 6번)
  ③ ① + 두 모서리에서 받은 세기의 비 (가까운 쪽이 세다)
  ④ 감독관이 시험 중에 늘 하는 순회: 통로 다섯 줄을 한 번 걷는 동안, 주머니의 작은 수신기(ESP32 하나)가 카메라의 세기만 잰다.
     카메라를 ESP32 가 찔러(Null Data) 돌아오는 ACK 의 세기를 25cm 마다 잰다. 판도 모터도 없다.
  ⑤ ① + ④
세기 모형: 방 반사(거울상)·사람 가림은 방향 계산과 같은 것을 쓰고, 20MHz 안 8개 주파수의 평균 세기를 쓴다.
  더한 것: 카메라 안테나가 쪽마다 다르게 내보내는 것 (±3dB 두 겹), 감독관 몸이 뒤쪽을 가리는 것 (6dB), 재는 흔들림 1.5dB.
  자리 고르기: 자리마다 '거리의 제곱에 반비례(지수 2.2)'로 예상한 세기와 잰 세기를 견준다 (전체 세기의 크기는 모르는 값으로 맞춘다, 흔들림 4dB).
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
from fusion import Fusion
import locate as K
import locate3d as L3
from locate3d import measure, PLANS
import classroom as C
from classroom import ClassWorld, SEATS, SeatGrid, place, rc, BODY_H, BODY_R
from room import C0, DF

HERE = os.path.dirname(os.path.abspath(__file__))
BACK = {"뒤 두 모서리 · 1.9m": [(0.6, 6.6, 1.9), (8.4, 6.6, 1.9)], "뒤 두 모서리 · 2.6m": [(0.6, 6.6, 2.6), (8.4, 6.6, 2.6)]}
AISLES = (0.4, 2.5, 4.5, 6.5, 8.6)                     # 짝 책상 (1·2칸, 3·4칸 ...) 사이와 벽 쪽 통로
PATROL_Z = 1.0                                        # 주머니·손 높이
N_EXP, SIG_PATROL, SIG_RATIO = 2.2, 4.0, 5.0
NOISE, SHADOW, GAIN_X, STEP = 1.5, 6.0, 1.0, 0.25      # 재는 흔들림(dB), 감독관 몸 가림(dB), 카메라 안테나 쪽마다 차이의 배수, 재는 간격(m)


def rssi(w, ant, gfun, rng, behind=None):
    """ant 자리에서 받은 카메라 세기 (dB, 기준은 아무것이나)"""
    amp, d, r, _ = w.paths(np.asarray(ant, float))
    ph = np.exp(-2j * np.pi * DF[:, None] * (r - r[0])[None, :] / C0)
    p = np.mean(np.abs((ph * amp[None, :]).sum(1)) ** 2) / r[0] ** 2
    v = np.asarray(ant, float) - w.cam; az = np.degrees(np.arctan2(v[1], v[0]))
    db = 10 * np.log10(max(p, 1e-12)) + GAIN_X * gfun(az) + rng.normal(0, NOISE)
    if behind is not None and np.dot(-v[:2], behind) < 0:      # 카메라가 감독관의 등 뒤쪽이면
        db -= SHADOW
    return db


def cam_gain(rng):
    a1, a2 = rng.uniform(1.5, 3.0), rng.uniform(1.0, 2.5); p1, p2 = rng.uniform(0, 2 * np.pi, 2)
    return lambda az: a1 * np.cos(np.radians(az) - p1) + a2 * np.cos(2 * np.radians(az) - p2)


def patrol_points(P):
    pts, heads = [], []
    for i, x in enumerate(AISLES):
        ys = np.arange(1.6, 7.01, STEP)
        if i % 2:
            ys = ys[::-1]                                          # 갈 지(之)자로 걷는다
        for y in ys:
            p = np.array([x, y, PATROL_Z])
            if P is not None and np.min(np.hypot(P[:, 0] - x, P[:, 1] - y) - P[:, 3]) < 0.15:
                continue
            pts.append(p); heads.append(np.array([0.0, 1.0 if i % 2 == 0 else -1.0]))
    return pts, heads


def ll_rssi(grid, ants, vals, sig, n=N_EXP):
    """칸마다의 로그 그럴듯함: 잰 세기 - 예상(거리) 의 차이를, 모르는 전체 크기를 맞춘 뒤의 제곱합으로"""
    A = np.array(ants); y = np.array(vals)
    d = np.linalg.norm(grid.X[:, None, :] - A[None, :, :], axis=2)
    pred = -10 * n * np.log10(np.maximum(d, 0.3))
    res = y[None, :] - pred; res = res - res.mean(1, keepdims=True)
    return -0.5 * np.sum((res / sig) ** 2, 1)


def seat_prob(grid, ll):
    l = np.exp(ll - ll.max()); p = np.bincount(grid.owner, l, minlength=len(SEATS)); return p / p.sum()


def score(row, tag, seat, p):
    order = np.argsort(-p); best = int(order[0]); (r0, c0), (r1, c1) = rc(seat), rc(best)
    row["exact_" + tag] = best == seat; row["near_" + tag] = abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1; row["top3_" + tag] = seat in order[:3]


def run(n=150, rx="뒤 두 모서리 · 2.6m", where="책상 위", people=True, seed=5):
    fus = Fusion(); grid = SeatGrid(); rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(SEATS))); cam, body = place(rng, seat, where)
        P = None
        if people:
            P = np.array([(*(body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25)), BODY_H + rng.uniform(-0.1, 0.1), BODY_R + rng.uniform(-0.03, 0.03))
                          for k, (sx, sy) in enumerate(SEATS)])
        g = cam_gain(rng); row = dict(seat=seat)
        ws = []
        for base in BACK[rx]:
            w = ClassWorld(rng, P, cam=cam, base=np.array(base, float))
            if ws:
                w.absorb = ws[0].absorb
            ws.append(w)
        m3 = [measure(w, fus, rng, PLANS["tilt3"]) for w in ws]
        m6 = [measure(w, fus, rng, PLANS["tilt6"]) for w in ws]
        row["dir"] = [m["err"] for m in m3]
        ll_dir = np.log(K.posterior(grid, m3, None, K.Q_BAD, K.SIG_SCALE) + 1e-300)
        ll_dir6 = np.log(K.posterior(grid, m6, None, K.Q_BAD, K.SIG_SCALE) + 1e-300)
        # 두 모서리의 세기 비
        refs = [m["ref"] for m in m3]; rs = [rssi(ws[0], r_, g, rng) for r_ in refs]
        ll_ratio = ll_rssi(grid, refs, rs, SIG_RATIO)
        # 순회
        pts, heads = patrol_points(P)
        vals = [rssi(ws[0], p_, g, rng, behind=h) for p_, h in zip(pts, heads)]
        ll_pat = ll_rssi(grid, pts, vals, SIG_PATROL)
        j = int(np.argmax(vals)); row["patrol_max_d"] = float(np.linalg.norm(pts[j][:2] - cam[:2]))
        for tag, ll in (("dir", ll_dir), ("dir6", ll_dir6), ("dir+ratio", ll_dir + ll_ratio), ("patrol", ll_pat), ("dir+patrol", ll_dir + ll_pat)):
            score(row, tag, seat, seat_prob(grid, ll))
        rows.append(row)
    return rows


TAGS = ("dir", "dir6", "dir+ratio", "patrol", "dir+patrol")


def summ(rows):
    out = {t: dict(exact=round(100 * np.mean([r["exact_" + t] for r in rows]), 1), near=round(100 * np.mean([r["near_" + t] for r in rows]), 1),
                   top3=round(100 * np.mean([r["top3_" + t] for r in rows]), 1)) for t in TAGS}
    out["dir_med"] = [round(float(np.median([r["dir"][j] for r in rows])), 1) for j in range(2)]
    out["patrol_max_d_med"] = round(float(np.median([r["patrol_max_d"] for r in rows])), 2)
    out["n"] = len(rows)
    return out


def main(n=150):
    res = {}
    for rx in BACK:
        for wh in ("책상 위", "가슴"):
            key = "%s | %s" % (rx, wh); t0 = time.time(); s = summ(run(n, rx, wh)); res[key] = s
            print("%-26s " % key + " | ".join("%s %4.1f/%4.1f/%4.1f" % (t, s[t]["exact"], s[t]["near"], s[t]["top3"]) for t in TAGS) +
                  " | 방향 %s, 순회 최대 세기 자리와 카메라 %.2fm (%.0f초)" % (s["dir_med"], s["patrol_max_d_med"], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=res), open(os.path.join(HERE, "back_corners_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 150)
