"""파이썬 원본(firmware/core/, firmware/pi/rx3.py)의 계산 결과를 뽑아 둔다 -> 이 폴더의 testvec.json.
core3.js 가 같은 값을 내는지는 test_core3.js 가 확인한다.
    python3 make_testvec.py && node test_core3.js        (numpy 만 있으면 된다)"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def _find(cands, name):
    for c in cands:
        d = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(d, name)):
            return d
    sys.exit("%s 를 찾지 못했습니다. 찾아본 곳: %s" % (name, ", ".join(cands)))


# 저장소(console/src)에서는 옆 폴더들을 찾는다: sim (montecarlo), firmware (core, pi, host)
sys.path.insert(0, _find(["../../firmware/core", "../rx2/sim"], "estimator.py"))
sys.path.insert(0, _find(["../../sim", "../rx2/sim"], "montecarlo.py")); sys.path.insert(0, _find(["../../firmware/pi", "../code/pi"], "rx3.py"))
sys.path.insert(0, _find(["../../firmware/host", "../code/host"], "scan3.py"))
import montecarlo as mc
from estimator import unit, azel, frame, to_scan, sphere
from fusion import Fusion, trough, consistency, next_scan, p_seen
import rx3
import scan3

rng = np.random.default_rng(20261113)
out = {}
f = lambda a: [float(x) for x in np.asarray(a, dtype=float).ravel()]
nan = lambda a: [None if not np.isfinite(x) else float(x) for x in a]

# 1) 표 꺼내기
P = mc.plates(); keys = ["1500", "3000", "5000"]; L = []
for k, pl in zip(keys, P):
    b = np.r_[rng.uniform(-95, 95, 60), [-90, 90, 0, 37.0, -36.5]]; d = np.r_[rng.uniform(-400, 400, 60), [-180, 180, 179.5, 0, 359.999]]
    v = pl.look(b, d)
    L.append(dict(key=k, beta=f(b), dphi=f(d), re=f(v.real), im=f(v.imag)))
out["look"] = L

# 2) 좌표
G = sphere(8000)
out["sphere"] = dict(n=int(len(G)), first=f(G[0]), last=f(G[-1]), mid=f(G[4000]))
fr = []
for _ in range(20):
    y, t = rng.uniform(-180, 180), rng.uniform(-5, 90); c = unit(rng.uniform(-180, 180), rng.uniform(-90, 90))
    a, ex, ey = frame(y, t); psi, beta = to_scan(c, (a, ex, ey))
    fr.append(dict(yaw=y, tilt=t, c=f(c), a=f(a), ex=f(ex), ey=f(ey), psi=float(psi), beta=float(beta)))
out["frames"] = fr
out["p_seen"] = dict(beta=f(np.arange(-90, 91, 7.5)), p=f(p_seen(np.arange(-90, 91, 7.5))))

# 3) 잡음 없는 곡선: 방 값을 바깥에서 정해 주고 World.scan 과 같은 식으로 계산
def level_curve(w, yaw, tilt, zero, phi):
    frm = frame(yaw, tilt); E = np.zeros(len(phi), dtype=complex)
    for k, (a, d) in enumerate(w.rays):
        psi, beta = to_scan(d, frm); fv = w.look(beta, phi + zero - psi)
        if k == 0:
            x = ((phi + zero - psi) % 360.0) - 180.0
            fv = fv + w.back * np.exp(-(x / 45.0) ** 2) * np.cos(np.radians(beta)) ** 2
        E += a * fv
    E += w.leak
    return 20 * np.log10(np.maximum(np.abs(E), 1e-4))

lv = []
for i in range(16):
    shape = "WV"[i % 2]; env = ["보통", "나쁨"][(i // 2) % 2]
    c = unit(rng.uniform(-180, 180), rng.uniform(-75, 85))
    w = mc.World(np.random.default_rng(900 + i), c, shape, env)
    yaw, tilt, zero = rng.uniform(-170, 170), rng.uniform(0, 80), rng.normal(0, 2.0)
    phi = np.arange(0.0, 360.0, 2.0) + 1.0
    lv.append(dict(shape=shape, env="normal" if env == "보통" else "bad", dist=keys[P.index(w.P)], yaw=yaw, tilt=tilt, zero=float(zero), phi=f(phi),
                   rays=[dict(re=float(a.real), im=float(a.imag), d=f(d)) for a, d in w.rays], leak=[float(w.leak.real), float(w.leak.imag)], back=float(w.back), v={k: float(x) for k, x in w.v.items()},
                   lv=f(level_curve(w, yaw, tilt, zero, phi))))
out["levels"] = lv

# 4) 골 찾기와 방향 합치기: 몬테카를로와 같은 곡선을 그대로 넘긴다
fus = Fusion(); cases = []
plans = {k: [list(x) for x in v] for k, v in rx3.PLANS.items()}
for i in range(70):
    shape = "WV"[i % 2]; env = ["보통", "나쁨"][(i // 2) % 2]
    c = mc.cameras(1, 5000 + i)[0]
    w = mc.World(np.random.default_rng(7000 + i), c, shape, env)
    name = ["tilt3", "tilt4", "up+tilt3", "up", "tilt6"][(i // 4) % 5]; frames = [tuple(x) for x in plans[name]]
    curves = [w.scan(y, t) for y, t in frames]
    est, obs = fus.solve(curves, frames)
    d = dict(plan=name, frames=[list(x) for x in frames], phi=f(curves[0][0]), lv=[nan(lvv) for _, lvv in curves], truth=f(azel(c)),
             obs=[dict(psi=float(o["psi"]), depth=float(o["depth"]), noise=float(o["noise"]), seen=bool(o["seen"])) for o in obs])
    if est is not None:
        az, el = azel(est); ok, det = consistency(est, obs, frames)
        d.update(est=f(est), az=float(az), el=float(el), spread=fus.spread(obs, frames, est), cons_ok=bool(ok),
                 cons=[dict(ok=bool(x["ok"]), beta=float(x["beta"]), resid=None if x["resid"] is None else float(x["resid"])) for x in det])
        if sum(o["seen"] for o in obs) == 1:
            y, t = next_scan(obs, frames, rx3.TILT3); d["next"] = [float(y), float(t)]
    cases.append(d)
out["cases"] = cases
# 5도 칸(analyze3 이 쓰는 폭)에서도 trough 가 같은지
t5 = []
for i in range(12):
    w = mc.World(np.random.default_rng(8000 + i), mc.cameras(1, 6000 + i)[0], "WV"[i % 2], "보통")
    ph, l2 = w.scan(0.0, rx3.TILT3)
    l5 = np.array([np.nanmean(l2[k:k + 5][::2]) if np.isfinite(l2[k:k + 5][::2]).any() else np.nan for k in range(0, 180, 5)])[:72]
    ph5 = (np.arange(len(l5)) + 0.5) * 5.0
    psi, depth, noise = trough(ph5, l5)
    t5.append(dict(phi=f(ph5), lv=nan(l5), psi=psi, depth=depth, noise=noise))
out["trough5"] = t5

# 5) 모터 시간: rx3 를 --dry 로 돌린 것과 같다
def run(plan_frames):
    rig = rx3.Rig(dry=True); rows = []
    for y, p in plan_frames:
        rec = rig.scan(y, p)
        rows.append(dict(yaw=rec["yaw"], pitch=rec["pitch"], t_end=rig.clock, t0=rec["t"][0], t1=rec["t"][-1], n=len(rec["k"]), k_mid=rec["k"][len(rec["k"]) // 2], t_mid=rec["t"][len(rec["k"]) // 2], spin0=rec["spin0"]))
    return rows

tm = {k: run(v) for k, v in rx3.PLANS.items()}
rig = rx3.Rig(dry=True); rig.home(); tm["home"] = rig.clock
rig = rx3.Rig(dry=True); [rig.scan(y, p) for y, p in rx3.PLANS["tilt3"]]; t0 = rig.clock                     # 한 번 잰 뒤 이어서 한 번 더: 받침이 +120도에 있으니 거꾸로 잰다
again = scan3.order(list(scan3.PLANS["tilt3"]), rig.deg("yaw"), "tilt3"); [rig.scan(y, p) for y, p in again]; tm["tilt3_again"] = rig.clock - t0; tm["tilt3_again_first"] = again[0][0]
rig = rx3.Rig(dry=True); rec = rig.spin_once(1.0); tm["spin"] = dict(k=rec["k"][::8], t=rec["t"][::8], total=rig.clock)
moves = []
for j, a, b in (("yaw", 0, 120), ("yaw", 120, -120), ("yaw", -120, -174), ("yaw", 0, -3), ("pitch", 0, rx3.TILT3), ("pitch", rx3.TILT3, 0), ("pitch", 0, 85), ("pitch", 10, 9.99)):
    rig = rx3.Rig(dry=True); rig.state["pos"][j] = int(round(a * rig.per_deg(j))); t0 = rig.clock; rig.move_to(j, b)
    moves.append(dict(j=j, a=a, b=b, t=rig.clock - t0, pos=rig.state["pos"][j]))
tm["moves"] = moves
out["timing"] = tm

json.dump(out, open(os.path.join(HERE, "testvec.json"), "w"), ensure_ascii=False)
print("testvec.json %.2f MB" % (os.path.getsize(os.path.join(HERE, "testvec.json")) / 1e6))
print("시간:", {k: round(v[-1]["t_end"], 1) for k, v in tm.items() if isinstance(v, list) and v and "t_end" in v[0]}, "원점", round(tm["home"], 1), "이어서 tilt3", round(tm["tilt3_again"], 1))
print("골이 보인 스캔 수:", [sum(o["seen"] for o in c["obs"]) for c in cases])
