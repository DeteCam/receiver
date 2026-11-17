"""사각지대: 패널 둘이 교실의 어느 자리를 잘 못 맞히나, 거는 곳을 바꾸거나 하나 더 걸면 어떻게 되나.

    ../../../.venv-rt/bin/python rt_study.py measure --recv ura4x4p37.5h --n 4000 --same --sets kr40_t_s1 ... kr40_t_s8 --tag t1b \
        --opt '{"mount":"hinge","tilt":15,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05,"spots":[7,17]}'       # 지금 거는 곳 둘만, 종류마다 카메라 500대
    ../../../.venv-rt/bin/python rt_study.py measure --recv ura4x4p37.5h --n 2400 --same --sets kr40_t_s1 ... kr40_t_s8 --tag t1c \
        --opt '{"mount":"hinge","tilt":15,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05}'                        # 스무 곳 모두, 종류마다 카메라 300대
    ../../../.venv-rt/bin/python blind_study.py                                                       # -> ../pre/blind.json

카메라 한 대를 사람 자세가 다른 여덟 순간(kr40_t_s1..s8)에 잰다. K번 모은 것 = 이어진 K순간의 점수를 곱한 것 (rt_study 의 tdiv).
열: 정확히 / 옆자리까지(앞뒤·옆 한 칸 안) / 후보 셋 안.
"""
import os, sys, json, itertools
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rt_study as S                                    # noqa: E402

OUT = os.path.join(HERE, "..", "pre", "blind.json")
NOW = (7, 17)                                           # 창 유리 가운데 2.0 m + 복도 창턱 앞쪽 1.95 m (지금 거는 곳)
ALT = {"7+17": (7, 17), "6+17": (6, 17), "1+7": (1, 7), "1+7+17": (1, 7, 17)}


def load(tag):
    Z = np.load(os.path.join(HERE, "study", "sc_ura4x4p37.5h_%s.npz" % tag)); Z = {k: Z[k] for k in Z.files}
    pts, seats_xy = S._pts(str(Z["sets"][0])); F = S.Fast(Z, pts, seats_xy); kd = Z["kind"]; kinds = list(dict.fromkeys(kd.tolist()))
    ns = len(Z["sets"]); key = np.array([kinds.index(k) for k in kd]) * 100000 + Z["grp"]
    idx = np.array([[np.where((key == g) & (Z["setk"] == j))[0][0] for j in range(ns)] for g in np.unique(key)
                    if all(((key == g) & (Z["setk"] == j)).any() for j in range(ns))])
    return dict(F=F, idx=idx, ns=ns, kinds=kinds, seat=np.tile(F.seat[idx[:, 0]], ns), kind=np.tile(kd[idx[:, 0]], ns), seats_xy=seats_xy,
                spots=np.load(os.path.join(HERE, "paths", str(Z["sets"][0]) + ".npz"))["spots"].astype(float), ncam=len(idx))


def run(D, combo, K):
    F, idx, ns = D["F"], D["idx"], D["ns"]; Lc = F.L[:, list(combo)].sum(1); R = []; B = []
    for j in range(ns):
        Ls = sum(Lc[idx[:, (j + i) % ns]] for i in range(K)); R.append(F.judge(Ls, F.seat[idx[:, 0]]))
        B.append(np.argmax(np.exp(Ls - Ls.max(1, keepdims=True)) @ F.O, 1))
    return np.concatenate(R).astype(float), np.concatenate(B)


def summary(D, combo, K, maps=False):
    R, B = run(D, combo, K); seat, kind = D["seat"], D["kind"]; n = len(D["seats_xy"])
    pe = np.array([R[seat == s, 0].mean() for s in range(n)]); pn = np.array([R[seat == s, 1].mean() for s in range(n)])
    out = {"kind": {k: (100 * R[kind == k].mean(0)).round(1).tolist() for k in D["kinds"]}, "all": (100 * R.mean(0)).round(1).tolist(),
           "worst5": round(100 * float(np.sort(pe)[:5].mean()), 1), "below40": int((pe < 0.4).sum())}
    if maps:
        out["seat_exact"] = (100 * pe).round(0).astype(int).tolist(); out["seat_near"] = (100 * pn).round(0).astype(int).tolist()
        rc = D["F"].rc; w = R[:, 0] == 0; dr = rc[B, 0] - rc[seat, 0]; dc = rc[B, 1] - rc[seat, 1]
        out["wrong"] = {"앞뒤로만": round(float(((dr != 0) & (dc == 0))[w].mean()), 3), "옆으로만": round(float(((dr == 0) & (dc != 0))[w].mean()), 3),
                        "대각": round(float(((dr != 0) & (dc != 0))[w].mean()), 3), "두 칸 넘게": round(float(((np.abs(dr) > 1) | (np.abs(dc) > 1))[w].mean()), 3)}
    return out


def geometry(D, combo):
    """자리마다: 패널까지의 거리, 패널이 보는 쪽에서 벗어난 각 (경첩식, 15도 숙임), 두 패널에서 본 방향이 만나는 각"""
    xy = D["seats_xy"]; aim = np.array([xy[:, 0].mean(), xy[:, 1].mean() + 0.2, 0.9]); rows = []
    for (x, y) in xy:
        c = np.array([x, y + 0.125, 0.8]); r = {}
        for nm, s in zip("AB", combo):
            sp = D["spots"][s]; v = c - sp; d = np.linalg.norm(v); h = aim - sp; h[2] = 0; h /= np.linalg.norm(h); t = np.radians(15.0)
            ey = np.cos(t) * h + np.sin(t) * np.array([0, 0, -1.0]); r[nm] = [round(float(d), 2), round(float(np.degrees(np.arccos(np.clip(v @ ey / d, -1, 1)))), 1)]
        va = c - D["spots"][combo[0]]; vb = c - D["spots"][combo[1]]; va[2] = vb[2] = 0
        r["cross"] = round(float(np.degrees(np.arccos(np.clip(va @ vb / np.linalg.norm(va) / np.linalg.norm(vb), -1, 1)))), 1); rows.append(r)
    return rows


if __name__ == "__main__":
    out = {"note": "교실 40석, 패널(ura4x4p37.5h) 경첩식 15도 숙임, 패널 자세 오차(방향 2도, 숙임 1도, 자리 5 cm). 열: 정확히 / 옆자리까지 / 후보 셋 (%)"}
    D = load("t1b"); out["seats_xy"] = D["seats_xy"].round(3).tolist(); out["rc"] = D["F"].rc.tolist()
    out["now"] = {"tag": "t1b", "cameras": int(D["ncam"]), "spots": {str(s): D["spots"][s].round(2).tolist() for s in NOW},
                  "K": {str(K): summary(D, NOW, K, maps=K in (1, 4)) for K in (1, 2, 4, 8)}, "geometry": geometry(D, NOW)}
    D = load("t1c"); out["alt"] = {"tag": "t1c", "cameras": int(D["ncam"]), "spots": D["spots"].round(2).tolist(),
                                    "combos": {nm: {str(K): summary(D, c, K, maps=K in (1, 4)) for K in (1, 4, 8)} for nm, c in ALT.items()}}
    pairs = {}
    for c in itertools.combinations(range(len(D["spots"])), 2):
        s = summary(D, c, 1); pairs["%d+%d" % c] = {"all": s["all"], "worst5": s["worst5"], "below40": s["below40"]}
    out["alt"]["pairs_top"] = {"by_exact": sorted(pairs, key=lambda k: -pairs[k]["all"][0])[:8], "by_worst5": sorted(pairs, key=lambda k: -pairs[k]["worst5"])[:8],
                               "values": {k: pairs[k] for k in set(sorted(pairs, key=lambda k: -pairs[k]["all"][0])[:8] + sorted(pairs, key=lambda k: -pairs[k]["worst5"])[:8])}}
    json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("저장", os.path.normpath(OUT))
