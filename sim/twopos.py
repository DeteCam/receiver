"""수신기를 두 자리에 놓고 재서 카메라의 '위치'를 구하면 얼마나 맞는지 본다.  python3 twopos.py [횟수]   결과: twopos_result.json

한 자리에서 나오는 것은 방향(선)이다. 위치를 구하는 방법 두 가지를 견준다.
  선 두 개   : 두 자리에서 나온 선이 가장 가깝게 지나는 곳의 가운데 점. 벽을 몰라도 된다.
  선 + 벽면  : 선이 벽이나 천장에 닿는 점 (카메라는 대개 벽·천장에 붙어 있다). 폰의 AR 이 벽면을 알아야 한다. 한 자리만 재도 된다.
방 반사와 수신기 모형은 room.py 그대로다. 여기에 '수신기가 어디에 어느 쪽을 보고 놓였는지'를 폰이 재는 오차를 더한다:
  자리 2cm, 받침이 향한 쪽 1.5도, 바닥 기울기 0.5도, 벽면 위치 3cm  (폰이 수신기의 마커를 읽어 자리를 잴 때의 어림값)
"""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import RoomWorld
from fusion import Fusion, consistency
import montecarlo as mc

HERE = os.path.dirname(os.path.abspath(__file__))
FRAMES = [(-120.0, mc.TILT3), (0.0, mc.TILT3), (120.0, mc.TILT3)]
POSE = dict(pos=0.02, yaw=1.5, level=0.5, wall=0.03)


def rot_z(v, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def tilt_small(v, rng, sig_deg):
    """바닥이 조금 기울어 있을 때: 아무 수평축 둘레로 작은 각만큼 돌린다"""
    a = rng.uniform(0, 2 * np.pi); k = np.array([np.cos(a), np.sin(a), 0.0]); t = np.radians(rng.normal(0, sig_deg))
    return v * np.cos(t) + np.cross(k, v) * np.sin(t) + k * (k @ v) * (1 - np.cos(t))


def closest(o1, d1, o2, d2):
    """두 선이 가장 가까운 곳: (가운데 점, 두 선 사이 틈, 선 위의 거리 t1, t2, 두 선이 이루는 각)"""
    w = o1 - o2; b = d1 @ d2; den = 1 - b * b
    if den < 1e-6:
        return None, np.inf, np.nan, np.nan, 0.0
    t1 = (b * (d2 @ w) - (d1 @ w)) / den; t2 = ((d2 @ w) - b * (d1 @ w)) / den
    p1 = o1 + t1 * d1; p2 = o2 + t2 * d2
    return (p1 + p2) / 2, float(np.linalg.norm(p1 - p2)), float(t1), float(t2), float(np.degrees(np.arccos(np.clip(b, -1, 1))))


def wall_hit(o, d, L, rng, sig):
    """선이 방(상자)의 면에 닿는 점. 면 위치에는 AR 이 재는 오차를 더한다."""
    best = None
    for ax in range(3):
        if abs(d[ax]) < 1e-9:
            continue
        plane = (L[ax] if d[ax] > 0 else 0.0) + rng.normal(0, sig)
        t = (plane - o[ax]) / d[ax]
        if t > 0 and (best is None or t < best):
            best = t
    return o + best * d


def measure(w, fus, rng):
    curves = [w.scan(y, t) for y, t in FRAMES]
    est, obs = fus.solve(curves, FRAMES)
    if est is None:
        return None
    ok, _ = consistency(est, obs, FRAMES)
    d = rot_z(est, w.head + rng.normal(0, POSE["yaw"]))                   # 수신기 좌표 -> 방 좌표 (받침이 향한 쪽을 폰이 잰 값으로)
    d = tilt_small(d, rng, POSE["level"])
    o = w.ref + np.array([rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"] / 2)])
    err = float(np.degrees(np.arccos(np.clip(est @ w.c, -1, 1))))
    return dict(o=o, d=d / np.linalg.norm(d), ok=bool(ok), err=err)


def run(n=400, rm="보통 방", shape="W", sep=(0.6, 1.2), seed=21, person=None):
    fus = Fusion(); rows = []
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        wA = RoomWorld(rng, rm, shape)
        Lx, Ly, Lz = wA.L; pB = None
        for _ in range(200):
            a = rng.uniform(0, 2 * np.pi); s = rng.uniform(*sep)
            q = wA.base + np.array([s * np.cos(a), s * np.sin(a), 0.0])
            if 0.3 <= q[0] <= Lx - 0.3 and 0.3 <= q[1] <= Ly - 0.3 and np.linalg.norm(wA.cam - q - (wA.ref - wA.base)) >= 0.45:
                pB = q; break
        if pB is None:
            continue
        wB = RoomWorld(rng, rm, shape, cam=wA.cam, base=pB)
        wB.absorb = wA.absorb                                             # 같은 방이다
        if person:
            wA.person = room.person_spot(wA, rng, person); wB.person = room.person_spot(wB, rng, person)
        A = measure(wA, fus, rng); B = measure(wB, fus, rng)
        cam = wA.cam; L = wA.L; row = dict(dist=float(np.linalg.norm(cam - wA.ref)), sep=float(np.linalg.norm(pB - wA.base)))
        row["errA"] = A["err"] if A else 180.0; row["errB"] = B["err"] if B else 180.0
        if A:
            hA = wall_hit(A["o"], A["d"], L, rng, POSE["wall"]); row["hitA"] = float(np.linalg.norm(hA - cam)); row["okA"] = A["ok"]
        if A and B:
            p, gap, t1, t2, ang = closest(A["o"], A["d"], B["o"], B["d"])
            hB = wall_hit(B["o"], B["d"], L, rng, POSE["wall"]); dh = float(np.linalg.norm(hA - hB))
            row.update(tri=float(np.linalg.norm(p - cam)) if p is not None else np.nan, gap=gap, ang=ang, t1=t1, t2=t2,
                       hit2=float(np.linalg.norm((hA + hB) / 2 - cam)), dh=dh, okB=B["ok"])
            inside = p is not None and all(-0.3 <= p[k] <= L[k] + 0.3 for k in range(3))
            # 믿을 만한 결과인지: 두 선이 만나듯 지나가고(틈 25cm 안), 앞쪽에서 만나고, 방 안이고, 두 스캔 묶음이 저마다 앞뒤가 맞을 때
            row["good_tri"] = bool(A["ok"] and B["ok"] and gap <= 0.25 and t1 > 0.2 and t2 > 0.2 and inside and ang >= 8.0)
            row["good_hit2"] = bool(A["ok"] and B["ok"] and dh <= 0.5)
        rows.append(row)
    return rows


def pct(x, q):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    return round(float(np.percentile(x, q)), 2) if len(x) else None


def summ(rows):
    n = len(rows)
    both = [r for r in rows if "tri" in r]
    gt = [r for r in both if r["good_tri"]]; gh = [r for r in both if r["good_hit2"]]
    one = [r for r in rows if "hitA" in r]; one_ok = [r for r in one if r["okA"]]
    f = lambda rs, k, lim: round(100.0 * np.mean([r[k] <= lim for r in rs]), 1) if rs else None
    return dict(n=n, both=round(100.0 * len(both) / max(n, 1), 1),
                dir_med=pct([r["errA"] for r in rows], 50),
                tri=dict(med=pct([r["tri"] for r in both], 50), p90=pct([r["tri"] for r in both], 90), in30=f(both, "tri", 0.3), in50=f(both, "tri", 0.5)),
                tri_good=dict(share=round(100.0 * len(gt) / max(n, 1), 1), med=pct([r["tri"] for r in gt], 50), p90=pct([r["tri"] for r in gt], 90), in30=f(gt, "tri", 0.3), in50=f(gt, "tri", 0.5),
                              miss=round(100.0 * np.mean([r["tri"] > 0.5 for r in gt]) * len(gt) / max(n, 1), 1) if gt else None),
                hit1=dict(med=pct([r["hitA"] for r in one], 50), p90=pct([r["hitA"] for r in one], 90), in30=f(one, "hitA", 0.3), in50=f(one, "hitA", 0.5)),
                hit1_ok=dict(share=round(100.0 * len(one_ok) / max(n, 1), 1), med=pct([r["hitA"] for r in one_ok], 50), in30=f(one_ok, "hitA", 0.3), in50=f(one_ok, "hitA", 0.5)),
                hit2_good=dict(share=round(100.0 * len(gh) / max(n, 1), 1), med=pct([r["hit2"] for r in gh], 50), p90=pct([r["hit2"] for r in gh], 90), in30=f(gh, "hit2", 0.3), in50=f(gh, "hit2", 0.5),
                               miss=round(100.0 * np.mean([r["hit2"] > 0.5 for r in gh]) * len(gh) / max(n, 1), 1) if gh else None),
                dist=pct([r["dist"] for r in rows], 50), sep=pct([r["sep"] for r in rows], 50))


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    out = {}
    for rm, sep in (("보통 방", (0.8, 1.5)), ("작은 방", (0.6, 1.2)), ("화장실", (0.4, 1.0))):
        for shape in ("W", "V"):
            t0 = time.time(); S = summ(run(n, rm, shape, sep)); out["%s|%s" % (rm, shape)] = S
            print("%s·%s (카메라까지 %.1fm, 두 자리 사이 %.1fm)  방향 오차 가운데값 %.1f도" % (rm, shape, S["dist"], S["sep"], S["dir_med"]))
            print("   선 두 개          : 가운데값 %.2fm, 90%% %.2fm, 30cm 안 %.0f%%, 50cm 안 %.0f%%" % (S["tri"]["med"], S["tri"]["p90"], S["tri"]["in30"], S["tri"]["in50"]))
            g = S["tri_good"]; print("   선 두 개 (믿을 만) : 전체의 %.0f%%, 가운데값 %.2fm, 50cm 안 %.0f%%, 믿었는데 50cm 넘게 틀림 = 전체의 %.1f%%" % (g["share"], g["med"], g["in50"], g["miss"]))
            h = S["hit1"]; print("   한 자리 + 벽면     : 가운데값 %.2fm, 90%% %.2fm, 30cm 안 %.0f%%, 50cm 안 %.0f%%" % (h["med"], h["p90"], h["in30"], h["in50"]))
            g = S["hit2_good"]; print("   두 자리 + 벽면 (두 점이 50cm 안에서 만날 때): 전체의 %.0f%%, 가운데값 %.2fm, 50cm 안 %.0f%%, 믿었는데 틀림 = 전체의 %.1f%%  (%.0f초)" % (
                g["share"], g["med"], g["in50"], g["miss"], time.time() - t0), flush=True)
    json.dump(dict(n=n, pose=POSE, summary=out), open(os.path.join(HERE, "twopos_result.json"), "w"), ensure_ascii=False, indent=1)
