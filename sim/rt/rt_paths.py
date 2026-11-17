"""교실 장면에서 수신기 자리(출발점) -> 카메라 후보 점 160개의 전파 길을 Sionna RT 로 구해 npz 로 담는다.
    python rt_paths.py 이름 [--h 2.4] [--depth 3] [--people 40|0|20] [--jitter 0.05] [--pose 0] [--seed 1] [--spots exam|ceil|real|real2|frame|grid] [--spotlist "x,y,z;..."] [--no-diff]
    grid = 패널 자리 찾기 후보 102곳 (place_spots.py), --spotlist 로 몇 곳만 따로
길마다: A[2,2] (카메라 편파 V/H x 수신기 편파 V/H), 지연 tau, 수신기에서의 방향(theta_t, phi_t), 카메라에서의 방향(theta_r, phi_r), 몸에 닿았는가, 직접파인가.
상반성: 수신기를 '송신기'로 두고 한 번 풀면 모든 카메라 점까지의 길이 한꺼번에 나온다."""
import sys, os, time, argparse
import numpy as np
import mitsuba as mi
import sionna.rt as rt
import scene_kr as S
import place_spots as PSP

FC = float(os.environ.get("DETECAM_FC", "2.437e9"))   # 2.4GHz 6번 채널 (5G 3.5 GHz 를 볼 때만 DETECAM_FC 로 바꾼다)
KEEP = 160              # 점마다 센 길부터 이만큼만 담는다

EXAM_SPOTS = [(0.5, 0.5), (7.0, 0.5), (0.5, 8.6), (7.0, 8.6)] + [(0.3, y) for y in (2.5, 4.3, 6.1, 7.6)] + [(7.2, y) for y in (2.5, 4.3, 6.1, 7.6)] + \
             [(x, 8.6) for x in (2.0, 3.75, 5.5)] + [(x, 1.3) for x in (1.5, 3.75, 6.0)]            # exam_mode_check.py 와 같은 17곳
EXTRA = {"ceil": [(3.75, 4.8), (3.75, 3.0), (3.75, 6.5)], "aisle": [(1.175, 4.85), (3.375, 4.85), (5.575, 4.85), (3.375, 2.95), (3.375, 6.75)]}

# 높이까지 정해 둔 곳 (x, y, z). real: 스탠드 없이 실제로 놓을 만한 곳 (칠판 3.6 x 1.2m 가 앞벽 가운데, 윗변 2.1m)
SPOTS3 = {"real": [(1.95, 0.12, 2.20), (5.55, 0.12, 2.20), (3.75, 0.12, 2.20), (3.75, 1.00, 1.20), (0.5, 0.5, 1.5), (7.0, 0.5, 1.5), (1.95, 0.12, 0.95), (5.55, 0.12, 0.95),
                   (0.5, 8.6, 1.5), (7.0, 8.6, 1.5), (3.75, 8.8, 1.8), (1.95, 8.88, 2.2), (5.55, 8.88, 2.2), (0.3, 4.85, 1.5), (7.2, 4.85, 1.5), (0.15, 4.85, 2.2), (7.35, 4.85, 2.2),
                   (0.3, 0.3, 2.2), (7.2, 0.3, 2.2), (3.75, 4.8, 2.6)]}
# frame: 칠판(3.6 x 1.2m) 테두리를 따라 늘어놓은 안테나 자리. 윗변(2.15m) 12개, 그 위(2.50m) 12개, 분필받이(0.95m) 12개
SPOTS3["frame"] = [(1.95 + 3.6 * i / 11.0, 0.06, z) for z in (2.15, 2.50, 0.95) for i in range(12)]
# real2: 스탠드 없이, 손이 닿는 높이에 걸거나 올려 둘 수 있는 곳 (칠판에는 자석, 유리에는 흡착판, 문에는 걸이, 사물함·교탁·창턱에는 올려 둔다)
SPOTS3["real2"] = [(1.95, 0.10, 1.95), (3.75, 0.10, 1.95), (5.55, 0.10, 1.95), (3.75, 1.15, 1.15), (0.45, 0.45, 1.60), (7.05, 0.45, 1.60),
                   (0.08, 2.50, 2.00), (0.08, 4.85, 2.00), (0.08, 7.00, 2.00), (0.14, 4.85, 1.05), (7.42, 0.85, 2.00), (7.42, 4.85, 1.95), (7.42, 8.15, 2.00),
                   (1.50, 8.42, 1.35), (3.75, 8.42, 1.35), (6.00, 8.42, 1.35), (3.75, 8.92, 2.00), (7.42, 2.50, 1.95), (7.42, 7.00, 1.95), (0.08, 1.00, 2.00)]
# lec2: 대학 강의실(9 x 12m, 72석)에서 손이 닿는 곳
SPOTS3["lec2"] = [(0.08, 6.4, 2.0), (8.92, 6.4, 2.0), (0.08, 3.5, 2.0), (8.92, 3.5, 2.0), (0.08, 9.5, 2.0), (8.92, 9.5, 2.0), (4.5, 0.10, 1.95), (2.7, 0.10, 1.95), (6.3, 0.10, 1.95)]
SPOTS3["grid"] = PSP.grid_spots()                   # 3차원 자리 찾기: 창·복도·앞·뒷벽의 높이 셋과 천장 (place_spots.py)
SPOT3_NAMES = {"real": ["칠판 위 왼쪽 끝", "칠판 위 오른쪽 끝", "칠판 위 가운데", "교탁 위", "앞 왼쪽 모서리 1.5m", "앞 오른쪽 모서리 1.5m", "분필받이 왼쪽", "분필받이 오른쪽",
                        "뒤 왼쪽 모서리 1.5m", "뒤 오른쪽 모서리 1.5m", "뒷벽 가운데 1.8m", "뒷벽 왼쪽 2.2m", "뒷벽 오른쪽 2.2m", "왼쪽 벽 가운데 1.5m", "오른쪽 벽 가운데 1.5m",
                        "왼쪽 벽 가운데 2.2m", "오른쪽 벽 가운데 2.2m", "앞 왼쪽 모서리 2.2m", "앞 오른쪽 모서리 2.2m", "천장 가운데 2.6m"]}


def cpx(x):
    return np.array(x[0].numpy()) + 1j * np.array(x[1].numpy()) if isinstance(x, (tuple, list)) else np.array(x.numpy())


def solve_spot(scene, solver, pos, cams, depth, diff, samples, body_ids):
    for n in list(scene.transmitters): scene.remove(n)
    scene.add(rt.Transmitter("dev", position=mi.Point3f(*[float(v) for v in pos])))
    p = solver(scene, max_depth=depth, samples_per_src=samples, max_num_paths_per_src=samples, synthetic_array=False,
               los=True, specular_reflection=True, diffuse_reflection=False, refraction=True, diffraction=diff, edge_diffraction=diff, diffraction_lit_region=diff)
    a = cpx(p.a)[:, :, 0, :, :]                                   # [cam, 2(cam pol), 2(dev pol), P]
    val = np.array(p.valid.numpy())
    val = val.reshape(a.shape[0], -1, a.shape[-1]).any(1) if val.ndim > 2 else val
    def g(x):
        x = np.array(x.numpy())
        return x[:, 0, 0, 0, :] if x.ndim == 5 else x.reshape(a.shape[0], -1)
    tau, tht, pht, thr, phr = g(p.tau), g(p.theta_t), g(p.phi_t), g(p.theta_r), g(p.phi_r)
    inter = np.array(p.interactions.numpy()); obj = np.array(p.objects.numpy())                 # [depth, cam, 2, 1, 2, P]
    inter = inter[:, :, 0, 0, 0, :] if inter.ndim == 6 else inter; obj = obj[:, :, 0, 0, 0, :] if obj.ndim == 6 else obj
    body = np.isin(obj, body_ids).any(0); nint = (inter != 0).sum(0); isdiff = (inter == int(rt.InteractionType.DIFFRACTION)).any(0)
    pw = (np.abs(a) ** 2).sum((1, 2)); pw[~val] = 0
    K = min(KEEP, a.shape[-1]); idx = np.argsort(-pw, axis=1)[:, :K]; r = np.arange(a.shape[0])[:, None]
    out = dict(A=np.moveaxis(a, -1, 1)[r, idx].astype(np.complex64), tau=tau[r, idx].astype(np.float32), tht=tht[r, idx].astype(np.float32), pht=pht[r, idx].astype(np.float32),
               thr=thr[r, idx].astype(np.float32), phr=phr[r, idx].astype(np.float32), body=body[r, idx], nint=nint[r, idx].astype(np.int8), diff=isdiff[r, idx], ok=(pw[r, idx] > 0))
    return out, p


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("name"); ap.add_argument("--h", type=float, nargs="+", default=[2.4]); ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--people", type=int, default=999); ap.add_argument("--jitter", type=float, default=0.05); ap.add_argument("--pose", type=float, default=0.0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spots", default="exam"); ap.add_argument("--no-diff", action="store_true"); ap.add_argument("--samples", type=int, default=10 ** 6)
    ap.add_argument("--ceil", default="concrete"); ap.add_argument("--out", default="paths"); ap.add_argument("--room", default="kr40"); ap.add_argument("--spotlist", default="")
    a = ap.parse_args(); rng = np.random.default_rng(a.seed); S.set_room(a.room); ns = len(S.SEATS)
    empty = () if a.people >= ns else tuple(rng.choice(ns, ns - a.people, replace=False).tolist())
    layout = S.people_layout(rng, a.jitter, a.pose) if a.people > 0 else None
    scene = S.build(people=a.people > 0, empty_seats=empty, ceil_mat=a.ceil, layout=layout)
    scene.frequency = FC
    arr = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="VH"); scene.tx_array = arr; scene.rx_array = arr
    cams = S.cam_points(layout).reshape(-1, 3)
    for i, c in enumerate(cams): scene.add(rt.Receiver("c%03d" % i, position=mi.Point3f(*[float(v) for v in c])))
    body_ids = np.array([o.object_id for o in scene.objects.values() if o.radio_material.name == "m_body"])
    mid = [(S.W / 2, S.D * f) for f in (0.53, 0.33, 0.72)]
    pts = S.exam_spots() if a.spots == "exam" else mid if a.spots == "ceil" else EXTRA.get(a.spots, [])
    spots = SPOTS3[a.spots] if a.spots in SPOTS3 else [(x, y, h) for h in a.h for (x, y) in pts]
    if a.spotlist:                                                          # "x,y,z;x,y,z": 이 곳들만
        spots = [tuple(float(v) for v in t.split(",")) for t in a.spotlist.split(";") if t.strip()]
    solver = rt.PathSolver(); res = []; t0 = time.time()
    for k, pos in enumerate(spots):
        o, _ = solve_spot(scene, solver, pos, cams, a.depth, not a.no_diff, a.samples, body_ids); res.append(o)
        print("%2d/%d %s  길 수 가운데값 %d, 직접파 있는 점 %d  (%.0f초)" % (k + 1, len(spots), pos, np.median(o["ok"].sum(1)), int((o["nint"][:, :] == 0)[o["ok"]].sum()), time.time() - t0), flush=True)
    os.makedirs(a.out, exist_ok=True)
    P = max(o["A"].shape[1] for o in res)
    def pad(x):
        w = [(0, 0), (0, P - x.shape[1])] + [(0, 0)] * (x.ndim - 2); return np.pad(x, w)
    np.savez_compressed(os.path.join(a.out, a.name + ".npz"), spots=np.array(spots), cams=cams, seats=S.SEATS, empty=np.array(empty, int), fc=FC,
                        people=np.array(layout if layout is not None else []), meta=str(vars(a)),
                        **{k: np.stack([pad(o[k]) for o in res]) for k in res[0]})
    print("저장", a.name, "%.0f초" % (time.time() - t0))


if __name__ == "__main__":
    main()
