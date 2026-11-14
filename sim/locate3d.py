"""카메라가 벽이 아니라 방 안 어디에나(화분, 탁자 위 물건 등) 있을 수 있을 때, 여러 자리에서 잰 결과로 '위치'를 구하는 방법들을 견준다.
    python3 locate3d.py [횟수] [방 이름 ...]        결과: locate3d_result.json

한 자리에서 나오는 것은 방향이다. 그런데 방향 '하나'가 아니라 '어느 방향이 얼마나 그럴듯한가'(구 위의 점수)가 나온다.
  ① 선 두 개의 교차       : 두 자리의 가장 좋은 방향 선이 가장 가깝게 지나는 점 (제안서의 처음 생각).
  ② 가능성 부피 (2자리)   : 방 안을 10cm 칸으로 나누고, 칸마다 "여기에 카메라가 있다면 두 자리의 측정이 얼마나 그럴듯한가"를 곱한다.
                            그럴듯함이 큰 칸들의 덩어리가 '추정 구간'이다. 선이 엇갈려도 되고, 한 자리가 틀렸을 때 구간이 커져서 티가 난다.
  ③ ② + 둘째 자리를 골라서: 첫 측정의 선을 보고, 그 선을 옆에서 가장 잘 가르는 자리를 둘째 자리로 권한다 (아무 데나 놓지 않는다).
  ④ ③ + 셋째 자리         : 구간이 아직 크면 한 번 더 잰다.
  ⑤ 표면 정보를 더함       : 카메라는 허공이 아니라 어떤 물건의 겉에 있다. 폰 AR(깊이)이 아는 표면 가까이의 칸만 남긴다.
  ⑥ 세기 비를 더함         : 두 자리에서 받은 세기의 차이로 거리의 비를 어림한다 (참고용).
방 반사, 수신기, 자리 재는 오차는 room.py, twopos.py 와 같다.
"""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import RoomWorld, ROOMS
from estimator import frame, to_scan
from fusion import Fusion, consistency, antenna_pos, origin, p_seen
import montecarlo as mc
from twopos import rot_z, tilt_small, closest, POSE

HERE = os.path.dirname(os.path.abspath(__file__))
TILT3 = mc.TILT3
PLANS = {"tilt3": [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)],
         "tilt6": [(-150.0, TILT3), (-90.0, TILT3), (-30.0, TILT3), (30.0, TILT3), (90.0, TILT3), (150.0, TILT3)]}
from locate import (STEP, Q_BAD, ZONE_CUT, GOOD_LEN, GOOD_CONF, score_points, to_dev, Grid, likelihood, grow, zone, in_zone, zone_gap, long_axis)   # 장치와 같은 계산
import locate as K
SURF_NEAR = 0.08                # 표면에서 이 거리 안의 칸만 남긴다 (m)
RSSI_SIG = 5.0                  # 두 자리 세기 차이의 흔들림 (dB): 반사 + 카메라 안테나가 쪽마다 다르게 내보내는 것


def posterior(grid, measures, prior=None, rssi=None, q=Q_BAD, sig_scale=1.0):
    """칸마다의 확률 (합 = 1). prior: 칸마다의 사전 가중 (표면 가까이만 1 등). rssi: 자리마다 받은 세기(dB)를 주면 세기 비도 곱한다"""
    p = np.ones(len(grid.X)) if prior is None else prior.astype(float).copy()
    if rssi is not None and len(measures) >= 2:
        # 세기 비: 두 자리에서 받은 세기 차이(dB) = -20 log10(dA / dB) + 흔들림
        for i in range(1, len(measures)):
            dA = np.linalg.norm(grid.X - measures[0]["ref"], axis=1); dB = np.linalg.norm(grid.X - measures[i]["ref"], axis=1)
            pred = -20.0 * np.log10(np.maximum(dA, 0.2) / np.maximum(dB, 0.2))
            p *= np.exp(-0.5 * ((rssi[0] - rssi[i] - pred) / RSSI_SIG) ** 2) * 0.97 + 0.03
    return K.posterior(grid, measures, p, q, sig_scale)


# ---------------------------------------------------------------- 수신기를 놓고 재기
def measure(w, fus, rng, frames):
    curves = [w.scan(y, t) for y, t in frames]
    obs = fus.read(curves)
    est = None; ok = False
    if any(o["seen"] for o in obs):
        est, _ = fus.solve(curves, frames)
        ok, _ = consistency(est, obs, frames)
    head = w.head + rng.normal(0, POSE["yaw"])                               # 폰 AR 이 잰 받침의 자리와 향한 쪽 (오차 포함)
    base = w.base + np.array([rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"]), rng.normal(0, POSE["pos"] / 2)])
    ref = base + origin(frames)
    d = None
    if est is not None:
        d = tilt_small(rot_z(est, head), rng, POSE["level"]); d = d / np.linalg.norm(d)
    # 받은 세기 (dB): 직접파의 세기(거리의 제곱에 반비례) + 반사가 보태는 흔들림 + 카메라 안테나가 쪽마다 다른 것
    amp, _, r, _ = w.paths(w.antenna(0.0, TILT3), None)
    p_db = 10 * np.log10(np.sum(np.abs(amp) ** 2)) - 20 * np.log10(r[0]) + rng.normal(0, 3.0)
    return dict(pose=(base, head), obs=obs, frames=frames, ref=ref, d=d, ok=bool(ok), est=est, rssi=float(p_db),
                err=float(np.degrees(np.arccos(np.clip(est @ w.c, -1, 1)))) if est is not None else 180.0)


def spots(L, z, avoid, min_sep=0.6, margin=0.3, step=0.25):
    """수신기를 놓을 수 있는 자리 후보 (바닥이나 같은 높이의 탁자 위라고 본다)"""
    return K.spots(np.zeros(3), np.asarray(L, float), z, avoid, min_sep, margin, step)


def choose_second(mA, L, z, frames, sig_deg=8.0):
    """첫 측정의 선을 보고 둘째 자리를 고른다: 선 위의 점들을 옆에서 봤을 때 '깊이'가 가장 잘 갈리는 자리 (locate.second_cost)"""
    P = spots(L, z, [mA["pose"][0]])
    if len(P) == 0:
        return None
    cost = K.second_cost(mA["ref"], mA["d"], P, np.zeros(3), np.asarray(L, float), frames, sig_deg)
    i = int(np.argmin(cost))
    return P[i] if np.isfinite(cost[i]) else None


def choose_third(ms, centre, axis, L, z, frames):
    """셋째 자리: 지금 구간의 긴 쪽(axis)을 옆에서 보는 자리 (locate.third_score)"""
    P = spots(L, z, [m["pose"][0] for m in ms])
    if len(P) == 0:
        return None
    sc = K.third_score([m["ref"] for m in ms], centre, axis, P, frames)
    i = int(np.argmax(sc))
    return P[i] if np.isfinite(sc[i]) else None


# ---------------------------------------------------------------- 방 안의 물건과 카메라 자리
def furniture(rng, L, n=5):
    """방 안의 물건(상자) n 개: (x0, y0, z0, x1, y1, z1). 절반은 벽에 붙고 절반은 방 안쪽에 놓인다 (탁자, 화분, 서랍장, 의자...)"""
    out = []
    for i in range(n):
        sx, sy = rng.uniform(0.25, 0.9), rng.uniform(0.25, 0.7); h = rng.uniform(0.3, 1.5)
        if rng.random() < 0.5:
            x0 = rng.choice([0.0, L[0] - sx]) if rng.random() < 0.5 else rng.uniform(0, L[0] - sx)
            y0 = rng.uniform(0, L[1] - sy) if x0 in (0.0, L[0] - sx) else rng.choice([0.0, L[1] - sy])
        else:
            x0 = rng.uniform(0.3, max(L[0] - sx - 0.3, 0.31)); y0 = rng.uniform(0.3, max(L[1] - sy - 0.3, 0.31))
        out.append((float(x0), float(y0), 0.0, float(min(x0 + sx, L[0])), float(min(y0 + sy, L[1])), float(h)))
    return out


def on_box(rng, b):
    """상자 겉(윗면이나 옆면) 위의 한 점"""
    x0, y0, z0, x1, y1, z1 = b
    f = rng.integers(5)
    if f == 0:
        return np.array([rng.uniform(x0, x1), rng.uniform(y0, y1), z1])
    z = rng.uniform(max(z0 + 0.05, 0.05), z1)
    return [np.array([x0, rng.uniform(y0, y1), z]), np.array([x1, rng.uniform(y0, y1), z]), np.array([rng.uniform(x0, x1), y0, z]), np.array([rng.uniform(x0, x1), y1, z])][f - 1]


def surface_mask(grid, boxes, near=SURF_NEAR):
    """폰 AR 이 아는 표면(벽, 천장, 물건의 겉) 가까이에 있는 칸. 물건 속은 뺀다."""
    X = grid.X; L = grid.L
    d = np.minimum.reduce([X[:, 0], L[0] - X[:, 0], X[:, 1], L[1] - X[:, 1], L[2] - X[:, 2]])       # 벽 4면과 천장까지
    m = d <= near + grid.step / 2
    for x0, y0, z0, x1, y1, z1 in boxes:
        dx = np.maximum.reduce([x0 - X[:, 0], np.zeros(len(X)), X[:, 0] - x1]); dy = np.maximum.reduce([y0 - X[:, 1], np.zeros(len(X)), X[:, 1] - y1])
        dz = np.maximum.reduce([z0 - X[:, 2], np.zeros(len(X)), X[:, 2] - z1])
        out = np.sqrt(dx * dx + dy * dy + dz * dz)                      # 상자 밖에서 상자까지 거리 (안이면 0)
        inside = (out == 0)
        depth_in = np.minimum.reduce([X[:, 0] - x0, x1 - X[:, 0], X[:, 1] - y0, y1 - X[:, 1], z1 - X[:, 2]])
        m |= (~inside & (out <= near + grid.step / 2)) | (inside & (depth_in <= grid.step))
    return m


def which_surface(p, L, boxes):
    """점 p 가 어느 표면에 있는가 (물건 번호, 또는 벽이면 -1)"""
    best = -1; db = min(p[0], L[0] - p[0], p[1], L[1] - p[1], L[2] - p[2])
    for i, (x0, y0, z0, x1, y1, z1) in enumerate(boxes):
        dx = max(x0 - p[0], 0, p[0] - x1); dy = max(y0 - p[1], 0, p[1] - y1); dz = max(z0 - p[2], 0, p[2] - z1)
        d = (dx * dx + dy * dy + dz * dz) ** 0.5
        if d < db - 1e-9:
            db = d; best = i
    return best


def place_camera(rng, L, boxes, where):
    """where: wall(벽·천장), inside(방 안쪽 물건의 겉), air(허공 아무 데나: 표면 정보를 못 쓰는 가장 나쁜 경우)"""
    if where == "inside":
        inner = [b for b in boxes if b[0] > 0.05 and b[1] > 0.05 and b[3] < L[0] - 0.05 and b[4] < L[1] - 0.05] or boxes
        return on_box(rng, inner[rng.integers(len(inner))])
    if where == "air":
        return np.array([rng.uniform(0.2, L[0] - 0.2), rng.uniform(0.2, L[1] - 0.2), rng.uniform(0.1, min(L[2] - 0.1, 2.0))])
    return None                                                          # wall: RoomWorld 가 고른다


# ---------------------------------------------------------------- 실험
def run(n=200, rm="보통 방", shape="W", where="inside", plan="tilt3", seed=31, max_pos=3):
    fus = Fusion(); frames = PLANS[plan]; rows = []
    R = ROOMS[rm]; L = np.array(R["L"]); grid = Grid(L)
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        boxes = furniture(rng, L)
        cam = place_camera(rng, L, boxes, where)
        wA = None
        for _ in range(50):
            w = RoomWorld(rng, rm, shape, cam=cam)
            in_box = any(b[0] - 0.1 < w.base[0] < b[3] + 0.1 and b[1] - 0.1 < w.base[1] < b[4] + 0.1 for b in boxes)
            if np.linalg.norm(w.cam - w.ref) >= 0.5 and not (in_box and w.base[2] < 0.05):
                wA = w; break
        if wA is None:
            continue
        cam = wA.cam; z = float(wA.base[2])
        smask = surface_mask(grid, boxes)
        row = dict(dist=float(np.linalg.norm(cam - wA.ref)), where=where)
        A = measure(wA, fus, rng, frames); row["errA"] = A["err"]
        def world_at(p):
            w = RoomWorld(rng, rm, shape, cam=cam, base=p); w.absorb = wA.absorb; return w
        # ---- 둘째 자리: 아무 데나(0.6~1.5m) / 골라서
        P = spots(L, z, [wA.base])
        if len(P) == 0:
            continue
        cand = P[(np.linalg.norm(P[:, :2] - wA.base[:2], axis=1) <= 1.5) & (np.linalg.norm(P - cam, axis=1) >= 0.5)]
        pR = (cand if len(cand) else P)[rng.integers(len(cand) if len(cand) else len(P))]
        Br = measure(world_at(pR), fus, rng, frames)
        pC = choose_second(A, L, z, frames) if A["d"] is not None else None
        if pC is None or np.linalg.norm(pC + origin(frames) - cam) < 0.3:
            pC = pR
        Bc = measure(world_at(pC), fus, rng, frames)
        row["sep_rand"] = float(np.linalg.norm(pR - wA.base)); row["sep_chosen"] = float(np.linalg.norm(pC - wA.base))
        # ① 선 두 개
        for tag, B in (("rand", Br), ("chosen", Bc)):
            if A["d"] is not None and B["d"] is not None:
                p, gap, t1, t2, ang = closest(A["ref"], A["d"], B["ref"], B["d"])
                row["tri_" + tag] = float(np.linalg.norm(p - cam)) if p is not None else 9.9
                row["tri_ok_" + tag] = bool(p is not None and A["ok"] and B["ok"] and gap <= 0.25 and t1 > 0.2 and t2 > 0.2 and ang >= 8.0 and all(-0.3 <= p[k] <= L[k] + 0.3 for k in range(3)))
            else:
                row["tri_" + tag] = 9.9; row["tri_ok_" + tag] = False
        # ②③ 가능성 부피
        def evaluate(tag, ms, prior=None, rssi=False):
            post = posterior(grid, ms, prior, [m["rssi"] for m in ms] if rssi else None)
            c, idx, length, vol, conf, second = zone(grid, post)
            row["e_" + tag] = float(np.linalg.norm(c - cam)); row["len_" + tag] = length; row["vol_" + tag] = vol; row["in_" + tag] = in_zone(grid, idx, cam)
            row["conf_" + tag] = conf; row["second_" + tag] = second
            return post, c, idx
        evaluate("1", [A])
        evaluate("2rand", [A, Br]); post2, c2, idx2 = evaluate("2", [A, Bc])
        evaluate("2rssi", [A, Bc], rssi=True)
        evaluate("2surf", [A, Bc], prior=smask)
        # ④ 셋째 자리 (구간의 긴 쪽을 옆에서 보는 자리)
        if max_pos >= 3:
            Xz = grid.X[idx2]; wz = post2[idx2]; mu = (Xz * wz[:, None]).sum(0) / wz.sum()
            cov = ((Xz - mu) * wz[:, None]).T @ (Xz - mu) / wz.sum() + np.eye(3) * 1e-6
            ev, evec = np.linalg.eigh(cov); axis = evec[:, -1]
            p3 = choose_third([A, Bc], c2, axis, L, z, frames)
            if p3 is None or np.linalg.norm(p3 + origin(frames) - cam) < 0.3:
                p3 = pR
            C3 = measure(world_at(p3), fus, rng, frames)
            evaluate("3", [A, Bc, C3]); post3s, c3s, idx3s = evaluate("3surf", [A, Bc, C3], prior=smask)
            row["errC"] = C3["err"]
        # ⑤ 표면 정보를 쓸 때: 맞는 물건(또는 벽)을 짚었는가
        post_s = posterior(grid, [A, Bc], smask); cs = zone(grid, post_s)[0]
        row["obj_ok_2surf"] = bool(which_surface(cs, L, boxes) == which_surface(cam, L, boxes))
        if max_pos >= 3:
            row["obj_ok_3surf"] = bool(which_surface(c3s, L, boxes) == which_surface(cam, L, boxes))
        row["errB"] = Bc["err"]; row["okA"] = A["ok"]; row["okB"] = Bc["ok"]
        rows.append(row)
    return rows


def trials(n=200, rm="보통 방", shape="W", where="inside", plan="tilt3", seed=31, npos=3):
    """run() 과 같은 순서로 방·카메라·측정을 만들어 하나씩 내놓는다 (설정만 바꿔 보는 실험용). npos 자리까지 잰다: 셋째부터는 그때까지의 구간의 긴 쪽을 옆에서 보는 자리."""
    fus = Fusion(); frames = PLANS[plan]
    R = ROOMS[rm]; L = np.array(R["L"]); grid = Grid(L)
    for i in range(n):
        rng = np.random.default_rng(seed * 100003 + i)
        boxes = furniture(rng, L)
        cam = place_camera(rng, L, boxes, where)
        wA = None
        for _ in range(50):
            w = RoomWorld(rng, rm, shape, cam=cam)
            in_box = any(b[0] - 0.1 < w.base[0] < b[3] + 0.1 and b[1] - 0.1 < w.base[1] < b[4] + 0.1 for b in boxes)
            if np.linalg.norm(w.cam - w.ref) >= 0.5 and not (in_box and w.base[2] < 0.05):
                wA = w; break
        if wA is None:
            continue
        cam = wA.cam; z = float(wA.base[2])
        A = measure(wA, fus, rng, frames)
        def world_at(p):
            w = RoomWorld(rng, rm, shape, cam=cam, base=p); w.absorb = wA.absorb; return w
        P = spots(L, z, [wA.base])
        if len(P) == 0:
            continue
        cand = P[(np.linalg.norm(P[:, :2] - wA.base[:2], axis=1) <= 1.5) & (np.linalg.norm(P - cam, axis=1) >= 0.5)]
        pR = (cand if len(cand) else P)[rng.integers(len(cand) if len(cand) else len(P))]
        Br = measure(world_at(pR), fus, rng, frames)
        pC = choose_second(A, L, z, frames) if A["d"] is not None else None
        if pC is None or np.linalg.norm(pC + origin(frames) - cam) < 0.3:
            pC = pR
        Ms = [A, measure(world_at(pC), fus, rng, frames)]
        while len(Ms) < npos:
            post = posterior(grid, Ms); c, idx = zone(grid, post)[:2]
            _, axis = long_axis(grid, post, idx)
            pn = choose_third(Ms, c, axis, L, z, frames)
            if pn is None or np.linalg.norm(pn + origin(frames) - cam) < 0.3:
                pn = pR
            Ms.append(measure(world_at(pn), fus, rng, frames))
        yield dict(grid=grid, L=L, cam=cam, boxes=boxes, A=A, Br=Br, Ms=Ms)


def run_variants(variants, n=120, rm="보통 방", shape="W", where="inside", plan="tilt3", seed=31, npos=3):
    """variants: 이름 -> dict(q, sig, cut, surf). surf: None(안 씀) / 1.0(표면 칸만) / 0~1(표면 아닌 칸에 주는 가중)
    돌려주는 것: 이름 -> {자리 수: 요약}"""
    rows = {k: {m: [] for m in range(2, npos + 1)} for k in variants}
    for T in trials(n, rm, shape, where, plan, seed, npos):
        grid = T["grid"]; cam = T["cam"]; sm = None
        for name, v in variants.items():
            prior = None
            if v.get("surf") is not None:
                sm = surface_mask(grid, T["boxes"]) if sm is None else sm
                prior = np.where(sm, 1.0, 0.0 if v["surf"] >= 1.0 else v["surf"])
            for m in range(2, npos + 1):
                post = posterior(grid, T["Ms"][:m], prior, None, v.get("q", Q_BAD), v.get("sig", 1.0))
                c, idx, length, vol, conf, second = zone(grid, post, v.get("cut", ZONE_CUT))
                rows[name][m].append(dict(e=float(np.linalg.norm(c - cam)), len=length, vol=vol, conf=conf, gap=zone_gap(grid, idx, cam)))
    out = {}
    pc = lambda r, f: round(100.0 * float(np.mean([f(x) for x in r])), 1)
    for name in variants:
        out[name] = {}
        for m in range(2, npos + 1):
            r = rows[name][m]
            if not r:
                continue
            good = [x for x in r if x["len"] <= GOOD_LEN and x["conf"] >= GOOD_CONF]
            out[name][m] = dict(n=len(r), med=round(float(np.median([x["e"] for x in r])), 2), in30=pc(r, lambda x: x["e"] <= 0.3), in50=pc(r, lambda x: x["e"] <= 0.5),
                                zone_len=round(float(np.median([x["len"] for x in r])), 2), zone_vol=round(float(np.median([x["vol"] for x in r])), 0),
                                inside=pc(r, lambda x: x["gap"] <= 0.09), near15=pc(r, lambda x: x["gap"] <= 0.15), near25=pc(r, lambda x: x["gap"] <= 0.25),
                                good_share=round(100.0 * len(good) / len(r), 1), good_in50=pc(good, lambda x: x["e"] <= 0.5) if good else None,
                                good_miss=round(100.0 * sum(x["e"] > 0.5 for x in good) / len(r), 1))
    return out


def med(rows, k):
    v = [r[k] for r in rows if k in r and r[k] is not None]
    return round(float(np.median(v)), 2) if v else None


def share(rows, k, lim=None):
    v = [r[k] for r in rows if k in r]
    if not v:
        return None
    return round(100.0 * float(np.mean([(x <= lim) if lim is not None else bool(x) for x in v])), 1)


def summ(rows):
    out = dict(n=len(rows), dist=med(rows, "dist"), dir_med=med(rows, "errA"), sep_rand=med(rows, "sep_rand"), sep_chosen=med(rows, "sep_chosen"))
    for tag in ("rand", "chosen"):
        ok = [r for r in rows if r.get("tri_ok_" + tag)]
        out["tri_" + tag] = dict(med=med(rows, "tri_" + tag), in30=share(rows, "tri_" + tag, 0.3), in50=share(rows, "tri_" + tag, 0.5),
                                 ok_share=round(100.0 * len(ok) / max(len(rows), 1), 1), ok_med=med(ok, "tri_" + tag), ok_in50=share(ok, "tri_" + tag, 0.5),
                                 ok_miss=round(100.0 * sum(r["tri_" + tag] > 0.5 for r in ok) / max(len(rows), 1), 1))
    for tag in ("1", "2rand", "2", "2rssi", "2surf", "3", "3surf"):
        if not any(("e_" + tag) in r for r in rows):
            continue
        good = [r for r in rows if r.get("len_" + tag, 9) <= GOOD_LEN and r.get("conf_" + tag, 0) >= GOOD_CONF]
        out[tag] = dict(med=med(rows, "e_" + tag), in30=share(rows, "e_" + tag, 0.3), in50=share(rows, "e_" + tag, 0.5), zone_len=med(rows, "len_" + tag), zone_vol=med(rows, "vol_" + tag),
                        inside=share(rows, "in_" + tag), conf=med(rows, "conf_" + tag),
                        good_share=round(100.0 * len(good) / max(len(rows), 1), 1), good_med=med(good, "e_" + tag), good_in50=share(good, "e_" + tag, 0.5), good_inside=share(good, "in_" + tag),
                        good_miss=round(100.0 * sum(r["e_" + tag] > 0.5 for r in good) / max(len(rows), 1), 1))
    out["obj_ok_2surf"] = share(rows, "obj_ok_2surf"); out["obj_ok_3surf"] = share(rows, "obj_ok_3surf")
    return out


def show(key, S):
    print("== %s  (카메라까지 %.1fm, 방향 오차 가운데값 %.1f도, 둘째 자리까지: 아무 데나 %.1fm / 골라서 %.1fm)" % (key, S["dist"], S["dir_med"], S["sep_rand"], S["sep_chosen"]))
    for tag, name in (("rand", "① 선 두 개 (아무 데나)"), ("chosen", "① 선 두 개 (자리 골라서)")):
        t = S["tri_" + tag]
        print("   %-22s 가운데값 %.2fm  30cm 안 %4.1f%%  50cm 안 %4.1f%% | 믿을 만 %4.1f%%: 가운데값 %s m, 50cm 안 %s%%, 믿었는데 틀림 = 전체의 %.1f%%" % (
            name, t["med"], t["in30"], t["in50"], t["ok_share"], t["ok_med"], t["ok_in50"], t["ok_miss"]))
    for tag, name in (("1", "한 자리만 (참고)"), ("2rand", "② 부피 2자리 (아무 데나)"), ("2", "③ 부피 2자리 (골라서)"), ("2rssi", "⑥ ③ + 세기 비"), ("2surf", "⑤ ③ + 표면"), ("3", "④ 부피 3자리"), ("3surf", "④⑤ 3자리 + 표면")):
        if tag not in S:
            continue
        t = S[tag]
        print("   %-22s 가운데값 %.2fm  30cm 안 %4.1f%%  50cm 안 %4.1f%% | 구간 길이 %.2fm 부피 %4.0fL 믿음 %.2f, 참 위치가 구간 안 %4.1f%% | 믿을 만 %4.1f%%: 가운데값 %s m, 50cm 안 %s%%, 구간 안 %s%%, 믿었는데 틀림 = 전체의 %.1f%%" % (
            name, t["med"], t["in30"], t["in50"], t["zone_len"], t["zone_vol"], t["conf"], t["inside"], t["good_share"], t["good_med"], t["good_in50"], t["good_inside"], t["good_miss"]))
    print("   표면을 쓸 때 맞는 물건(벽)을 짚은 비율: 2자리 %s%%, 3자리 %s%%" % (S["obj_ok_2surf"], S["obj_ok_3surf"]), flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rooms = sys.argv[2:] or ["보통 방", "작은 방", "화장실"]
    path = os.path.join(HERE, "locate3d_result.json")
    out = json.load(open(path, encoding="utf-8"))["summary"] if os.path.exists(path) else {}
    for rm in rooms:
        for where in ("inside", "wall", "air"):
            for shape in ("W", "V"):
                t0 = time.time(); S = summ(run(n, rm, shape, where)); key = "%s|%s|%s" % (rm, where, shape); out[key] = S
                show(key, S); print("   (%.0f초)" % (time.time() - t0), flush=True)
                json.dump(dict(n=n, step=STEP, q_bad=Q_BAD, zone_cut=ZONE_CUT, good=dict(length=GOOD_LEN, conf=GOOD_CONF), pose=POSE, summary=out), open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
