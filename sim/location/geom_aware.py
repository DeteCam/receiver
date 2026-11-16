"""칠판·사물함·벽에서 오는 반사를 '방 모양을 알고' 걸러 내면 교실에서 자리를 더 잘 맞히나.   python3 geom_aware.py [횟수]   결과: geom_aware_result.json

지금 장치의 방법: 판 각마다의 세기 곡선에서 '골'을 읽어 방향을 낸다. 반사파가 골을 메우거나 가짜 골을 만들면 틀린다.
방 모양을 아는 방법 (여기서 해 보는 것):
  - 방의 크기·벽 자리는 폰 AR 이 안다 (설치할 때 재 둔다). 칠판·사물함 같은 큰 반사체도 벽에 붙어 있다.
  - 후보 점(자리마다 책상 위·가슴 여러 점)마다, 그 점에서 수신기로 오는 길을 '직접파 + 벽·바닥·천장 6면에 한 번 반사된 파' 7개로 그린다
    (거울상: 반사파는 벽 너머 거울에 비친 카메라에서 오는 것과 같다).
  - 길마다 판이 어떻게 가리는지(판 모형)는 알므로, 판 각마다의 세기 곡선을 '길 7개 세기의 합 + 바닥값'으로 맞춘다 (길마다 세기는 모르는 값, 0 이상).
  - 잘 맞는 후보 점일수록 그럴듯하다. 반사파를 '틀린 방향의 골'이 아니라 '그 점에서 생기는 게 당연한 반사'로 설명하는 셈이다.
일부러 어긋나게 둔 것 (계산과 세상이 똑같으면 너무 좋게 나온다):
  - 세상은 반사를 3번까지 (63길), 계산은 1번까지 (7길). 세상에는 사람 가림, 가구 흩어짐, 카메라 안테나 세기 차이, 서브캐리어 섞임이 있고 계산은 모른다.
  - 방 크기는 ±10cm 틀리게 안다. 수신기 자리·향한 쪽도 AR 오차만큼 틀린다.
교실과 사람은 kr_classroom.py (한국 교실 규격) 그대로.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))      # sim/ 의 paths.py 를 찾는다
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
from scipy.optimize import nnls
import room
from room import ANT_R, ANT_H, AXIS_H, DISTS, plates
from estimator import frame, to_scan
from fusion import Fusion, consistency, origin
import locate as K
from locate3d import PLANS, POSE
import classroom as C
import kr_classroom as KR

HERE = os.path.dirname(os.path.abspath(__file__))
SIG_DB = 2.0
HYB_T = 10.0       # 합칠 때 방 모양 방법의 로그 그럴듯함을 이만큼 눌러 준다 (판 각 72 x 스캔 3 개가 서로 독립이 아니어서)
LAM = 1.0          # 반사파 세기에 거는 벌점 (직접파보다 세게 나오기 어렵다는 앞선 믿음). 0 이면 없음


def measure2(w, fus, rng, frames):
    curves = [w.scan(y, t) for y, t in frames]
    obs = fus.read(curves)
    head = w.head + rng.normal(0, POSE["yaw"])
    base = w.base + np.array([rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"] / 2)])
    return dict(pose=(base, head), obs=obs, frames=frames, curves=curves, ref=base + origin(frames))


def images1(X, L):
    """점들 X (N,3) 의 직접 + 한 번 반사 거울상 (N, 7, 3)"""
    out = [X]
    for ax in range(3):
        for wall in (0.0, L[ax]):
            Y = X.copy(); Y[:, ax] = 2 * wall - X[:, ax]; out.append(Y)
    return np.stack(out, 1)


def ant_pos(base, head, yaw, tilt):
    t = np.radians(tilt); x = -ANT_R * np.cos(t) + ANT_H * np.sin(t); z = AXIS_H + ANT_R * np.sin(t) + ANT_H * np.cos(t)
    a = np.radians(yaw + head)
    return base + np.array([x * np.cos(a), x * np.sin(a), z])


def geo_ll(grid, meas, L):
    """후보 점마다의 로그 그럴듯함 (방 모양을 아는 방법)"""
    base, head = meas["pose"]; PL = plates(); ld = np.log(np.array(DISTS))
    IM = images1(grid.X, L)                                                  # (N, 7, 3)
    ll = np.zeros(len(grid.X))
    a = np.radians(-head); c, s = np.cos(a), np.sin(a)
    for (yaw, tilt), (phi, lv) in zip(meas["frames"], meas["curves"]):
        ok = np.isfinite(lv)
        if ok.sum() < 10:
            continue
        ant = ant_pos(base, head, yaw, tilt); fr = frame(yaw, tilt)
        V = IM - ant; R = np.linalg.norm(V, axis=2); Dd = V / R[..., None]
        Dv = np.stack([c * Dd[..., 0] - s * Dd[..., 1], s * Dd[..., 0] + c * Dd[..., 1], Dd[..., 2]], -1)          # 장치 좌표
        psi, beta = to_scan(Dv, fr)                                                                             # (N, 7)
        y = 10 ** (lv[ok] / 10.0); ph = phi[ok]
        for i in range(len(grid.X)):
            A = np.empty((ok.sum(), 15))
            for k in range(7):
                P = PL[int(np.argmin(np.abs(ld - np.log(R[i, k] * 1000.0))))]
                A[:, k] = np.abs(P.look(beta[i, k], ph - psi[i, k])) ** 2
                xb = ((ph - psi[i, k]) % 360.0) - 180.0                                   # 판이 반대쪽에 있을 때 등판에서 되튀는 몫 (장치마다 세기가 달라 모르는 값)
                A[:, 7 + k] = (np.exp(-(xb / 45.0) ** 2) * np.cos(np.radians(beta[i, k])) ** 2) ** 2
            A[:, 14] = 1.0
            if LAM > 0:                                                                     # 반사 길(1~6번)과 그 등판 몫에 벌점 줄을 더한다
                sc = np.sqrt(LAM) * np.median(y)
                Rg = np.zeros((12, 15)); Rg[np.arange(6), np.arange(1, 7)] = sc / max(np.median(y), 1e-9) * np.median(A[:, 0]) * 0 + sc; Rg[6 + np.arange(6), 8 + np.arange(6)] = sc
                x, _ = nnls(np.vstack([A, Rg]), np.r_[y, np.zeros(12)])
            else:
                x, _ = nnls(A, y)
            m = np.maximum(A @ x, 1e-6)
            ll[i] += -0.5 * np.sum(((10 * np.log10(m) - lv[ok]) / SIG_DB) ** 2)
    return ll


def seat_p(grid, ll):
    l = np.exp(ll - ll.max()); p = np.bincount(grid.owner, l, minlength=len(C.SEATS)); return p / p.sum()


def score(row, tag, seat, p):
    order = np.argsort(-p); best = int(order[0]); (r0, c0), (r1, c1) = C.rc(seat), C.rc(best)
    row["exact_" + tag] = best == seat; row["near_" + tag] = abs(r0 - r1) <= 1 and abs(c0 - c1) <= 1; row["top3_" + tag] = seat in order[:3]


def run(n, rxs, where, seed=5, room_err=0.10):
    KR.setup(2.7)
    fus = Fusion(); frames = PLANS["tilt3"]; grid = C.SeatGrid(); rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 7919 + i)
        seat = int(rng.integers(len(C.SEATS))); cam, body = C.place(rng, seat, where)
        P = np.array([(*(body if k == seat else (sx + rng.uniform(-0.1, 0.1), sy + 0.25)), C.BODY_H + rng.uniform(-0.1, 0.1), C.BODY_R + rng.uniform(-0.03, 0.03))
                      for k, (sx, sy) in enumerate(C.SEATS)])
        ms = []; w0 = None
        for b in rxs:
            w = C.ClassWorld(rng, P, cam=cam, base=np.array(b, float))
            if w0 is None:
                w0 = w
            else:
                w.absorb = w0.absorb
            ms.append(measure2(w, fus, rng, frames))
        Lk = np.array(room.ROOMS["교실"]["L"]) + rng.uniform(-room_err, room_err, 3)       # 폰 AR 이 잰 방 크기 (오차)
        row = dict(seat=seat)
        p0 = np.bincount(grid.owner, K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE), minlength=len(C.SEATS)); p0 = p0 / p0.sum()
        score(row, "now", seat, p0)
        llg = sum(geo_ll(grid, m, Lk) for m in ms)
        score(row, "geo", seat, seat_p(grid, llg))
        post_now = K.posterior(grid, ms, None, K.Q_BAD, K.SIG_SCALE)
        score(row, "both", seat, seat_p(grid, llg / HYB_T + np.log(post_now + 1e-300)))
        rows.append(row)
    return rows


def summ(rows):
    return {t: dict(exact=round(100 * np.mean([r["exact_" + t] for r in rows]), 1), near=round(100 * np.mean([r["near_" + t] for r in rows]), 1),
                    top3=round(100 * np.mean([r["top3_" + t] for r in rows]), 1)) for t in ("now", "geo", "both")}


CASES = {"앞 모서리 하나 (복도 쪽)": [(7.0, 0.5, 2.4)], "뒤 두 모서리": [(0.5, 8.6, 2.4), (7.0, 8.6, 2.4)],
         "앞 두 모서리": [(0.5, 0.5, 2.4), (7.0, 0.5, 2.4)], "칠판 위 가운데": [(3.75, 0.25, 2.4)], "천장 가운데 (2.5m)": [(3.75, 4.8, 2.5)]}


def main(n=80):
    res = {}
    for name, rxs in CASES.items():
        for wh in ("책상 위", "가슴"):
            t0 = time.time(); s = summ(run(n, rxs, wh)); res["%s | %s" % (name, wh)] = s
            print("%-24s %-4s 지금 %4.1f/%4.1f/%4.1f | 방 모양 %4.1f/%4.1f/%4.1f | 둘 다 %4.1f/%4.1f/%4.1f (%.0f초)" % (
                name, wh, *[s[t][k] for t in ("now", "geo", "both") for k in ("exact", "near", "top3")], time.time() - t0), flush=True)
    json.dump(dict(n=n, results=res), open(os.path.join(HERE, "geom_aware_result.json"), "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 80)
