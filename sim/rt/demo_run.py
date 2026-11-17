"""교실에서 패널 둘이 '사진을 가끔 보내는 카메라'의 자리를 좁혀 가는 과정을 한 번 돌려 demo_run.json 에 적는다 (동작 시뮬레이션 영상의 재료).

    python demo_run.py [--seed 1003] [--kind 책상] [--spots 7 17] [--recv ura4x4p37.5h]

rt_study 의 계산 그대로다. 사람 자세가 다른 길 묶음(kr40_t_s1..s8)을 '사진을 보낸 순간'으로 차례로 쓴다.
"""
import os, json, argparse
import numpy as np
import rt_study as S
from rt_study import K, post_seat, seat_rc

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = dict(mount="hinge", tilt=15.0, yaw_sd=2.0, tilt_sd=1.0, pos_sd=0.05)


def run(seed, kind, spots, recv, sets, period=30.0):
    S._init(); rng = np.random.default_rng(seed); D0 = S._load(sets[0]); seats_xy = D0["seats"]
    seat = int(rng.integers(len(seats_xy))); c = seat * 4 + int(rng.choice(S.KIND[kind])); u = rng.normal(size=3); u /= np.linalg.norm(u)
    seats = [dict(id=str(k), x=float(x), y=float(y) + 0.225) for k, (x, y) in enumerate(seats_xy)]; pts = S.Points(seats); nm = recv[:-1]
    G = S._G["G"]; logs = []; steps = []
    for k, name in enumerate(sets):
        D = S._load(name); ppl = np.load(os.path.join(HERE, "paths", name + ".npz"))["people"]; views = []; now = []
        for s in spots:
            r2 = np.random.default_rng([seed, s, k]); P = S.Paths(D, s, c, u); spot = D["spots"][s].astype(float)
            H, Rw, bore = S.array_snap(r2, P, nm, holo=True, mount=OPT["mount"], tilt=OPT["tilt"])
            sp, Rw, bore = S.pose_err(r2, spot, Rw, bore, OPT["pos_sd"], OPT["yaw_sd"], OPT["tilt_sd"])
            sc = S.array_score(H, Rw, bore, sp, pts.X); bg = S.array_score(H, Rw, bore, sp, sp + 6.0 * G); bg = np.where(G @ bore > 0.1, bg, -1e9); d = G[int(np.argmax(bg))]
            err = S.ang_err(d, P.c_true); now.append(sc); logs.append(sc)
            views.append(dict(spot=int(s), los=bool(P.los), az=float(np.degrees(np.arctan2(d[1], d[0]))), az_true=float(np.degrees(np.arctan2(P.c_true[1], P.c_true[0]))), err=[round(e, 1) for e in err],
                              bore=[float(v) for v in bore]))
        p1 = post_seat(pts, now); pc = post_seat(pts, logs); o = np.argsort(-pc)[:3]
        steps.append(dict(t=round((k + 1) * period, 1), set=name, views=views, p_now=[round(float(v), 4) for v in p1], p_cum=[round(float(v), 4) for v in pc], top=[int(i) for i in o],
                          people=[[round(float(v), 3) for v in row[:4]] for row in ppl]))
        print("%d번째  지금만 %2d (%.0f%%)  모아서 %2d (%.0f%%)  참 %d  직접파 %s  오차 %s" % (k + 1, int(np.argmax(p1)), 100 * p1.max(), o[0], 100 * pc[o[0]], seat,
                                                                              [v["los"] for v in views], [v["err"][2] for v in views]))
    return dict(seed=seed, kind=kind, recv=recv, seat=seat, cam=[float(v) for v in D0["cams"][c]], seats=[[float(x), float(y)] for x, y in seats_xy], spots=[[float(v) for v in D0["spots"][s]] for s in spots],
                spot_names=[S.SPOT_NAMES2[s] for s in spots], room=list(S.ROOM), period=period, steps=steps, rc=[list(seat_rc(xy, seats_xy)) for xy in seats_xy])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=1003); ap.add_argument("--kind", default="책상"); ap.add_argument("--spots", type=int, nargs="+", default=[7, 17])
    ap.add_argument("--recv", default="ura4x4p37.5h"); ap.add_argument("--n", type=int, default=6); ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(); out = run(a.seed, a.kind, a.spots, a.recv, ["kr40_t_s%d" % i for i in range(1, a.n + 1)])
    if not a.dry:
        json.dump(out, open(os.path.join(HERE, "demo_run.json"), "w"), ensure_ascii=False); print("저장 demo_run.json")
