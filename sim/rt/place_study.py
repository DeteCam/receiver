"""패널을 어디에 걸지 (3차원 자리 찾기)와, 카메라가 사진을 16번 넘게 보낼 때 (문제마다 한 장씩 찍는다고 보고).

후보 자리는 place_spots.py: 손이 닿는 스무 곳(real2) + 창·복도·앞·뒷벽의 높이 셋과 천장(grid, 102곳) = 122곳.
같은 카메라를 사람 자세가 다른 16순간(kr40_tg_s1..s16)에 재서 점수를 모은다 (rt_study 의 tdiv 와 같은 셈: 순간마다의 로그 점수를 더한다).
두 곳·세 곳의 모든 조합을 견줘 자리마다 맞힌 비율이 가장 고른 배치를 찾고, 고른 배치는 새 카메라로 다시 잰다 (검증).

    python place_study.py merge  --n 16                                  kr40_t_sN + kr40_g_sN -> kr40_tg_sN
    python place_study.py measure --tag g16 --sets kr40_tg_s1 .. s16 --cams 400 --seed0 20000
    python place_study.py search  --tag g16                              두 곳, 세 곳 -> study/place_g16.json
    python place_study.py measure --tag v16 ... --spots 7 17 ... --seed0 40000   (검증용 새 카메라)
    python place_study.py curve   --tag c32 --sets kr40_cv_s1 .. s32 --combos 0,1 2,3,4   K = 1..32

열: 정확히 / 옆자리까지(앞뒤·옆 한 칸 안) / 후보 셋 안. 자리마다의 '정확히'가 사각지대를 본다.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, argparse, itertools                 # noqa: E402
import numpy as np                                          # noqa: E402
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import rt_study as R                                        # noqa: E402
import rt_common as K                                       # noqa: E402
import place_spots as PS                                    # noqa: E402

PATHS = os.path.join(HERE, "paths"); OUT = os.path.join(HERE, "study")
KEYS = ("A", "tau", "tht", "pht", "thr", "phr", "body", "nint", "diff", "ok")


def merge(a):
    """같은 순간(같은 사람 배치)의 두 길 묶음을 곳 축으로 잇는다"""
    for n in range(a.first, a.n + 1):
        out = os.path.join(PATHS, "%s_s%d.npz" % (a.to, n))
        if os.path.exists(out):
            continue
        parts = [np.load(os.path.join(PATHS, "%s_s%d.npz" % (b, n))) for b in a.base]
        assert all(np.allclose(p["cams"], parts[0]["cams"]) for p in parts), "카메라 점이 다르다 (사람 배치가 다름)"
        P = max(p["A"].shape[2] for p in parts)
        def pad(x):
            return np.pad(x, [(0, 0), (0, 0), (0, P - x.shape[2])] + [(0, 0)] * (x.ndim - 3))
        D = {k: np.concatenate([pad(p[k]) for p in parts]) for k in KEYS}
        np.savez_compressed(out, spots=np.concatenate([p["spots"] for p in parts]), cams=parts[0]["cams"], seats=parts[0]["seats"], empty=parts[0]["empty"],
                            fc=parts[0]["fc"], people=parts[0]["people"], meta=str([str(p["meta"]) for p in parts]), **D)
        print("merge", out, D["A"].shape, flush=True)


def subset(a):
    """길 묶음에서 몇 곳만 골라 새 이름으로 (K 곡선용으로 순간을 늘릴 때)"""
    idx = [int(i) for i in a.spots]
    for n in range(a.first, a.n + 1):
        out = os.path.join(PATHS, "%s_s%d.npz" % (a.to, n))
        if os.path.exists(out):
            continue
        d = np.load(os.path.join(PATHS, "%s_s%d.npz" % (a.base[0], n)))
        np.savez_compressed(out, spots=d["spots"][idx], cams=d["cams"], seats=d["seats"], empty=d["empty"], fc=d["fc"], people=d["people"], meta=str(d["meta"]),
                            **{k: d[k][idx] for k in KEYS})
        print("subset", out, flush=True)


def lfun(sc, temp=1.0, q=K.Q_BAD):
    """rt_study.Fast 와 같은 로그 항: 점수 -> 곳마다 후보 점의 로그 확률 (틀린 측정 q 를 섞음)"""
    l = np.exp((sc - sc.max(-1, keepdims=True)) / temp)
    return np.log(l + q / (1.0 - q) * l.mean(-1, keepdims=True))


def _winit():
    R._init()
    def _load2(name):                                           # 순간 하나씩만 메모리에 (곳이 많아 순간마다 수백 MB)
        D = R._G["D"]
        if name not in D:
            while len(D) >= 1:                                  # 순간 차례로 재므로 지금 순간 하나만 둔다 (메모리)
                D.pop(next(iter(D)))
            d = np.load(os.path.join(PATHS, name + ".npz")); D[name] = {k: d[k] for k in d.files if k not in ("meta", "people")}
        return D[name]
    R._load = _load2


def measure(a):
    """카메라마다 순간들의 로그 항을 모은다. A = 모든 순간의 합 [C, S, N], W<k> = k 순간 창의 합, L<j> = 순간 j 하나"""
    import multiprocessing as mp
    sets = a.sets; ns = len(sets); kinds = a.kinds; G = a.cams
    opt = json.loads(a.opt); opt.update(bal=1, bal0=a.seed0)
    D0 = np.load(os.path.join(PATHS, sets[0] + ".npz")); S_all = len(D0["spots"])
    sel = [int(i) for i in a.spots] if a.spots else list(range(S_all)); opt["spots"] = sel
    pts, seats_xy = R._pts(sets[0]); N = len(pts.X)
    cams = [(ki, g) for ki in range(len(kinds)) for g in range(G)]; C = len(cams)
    full = a.full                                               # full: 순간마다 다 남긴다 (곳이 적을 때, K 곡선용)
    A = np.zeros((C, len(sel), N), np.float32)
    Lall = np.zeros((C, ns, len(sel), N), np.float16) if full else None
    W8 = np.zeros((C, len(sel), N), np.float32) if (not full and ns >= 16) else None
    W4 = np.zeros((C, len(sel), N), np.float32) if (not full and ns >= 8) else None
    L1 = np.zeros((C, 2, len(sel), N), np.float16) if not full else None
    seat = np.full(C, -1, int); t0 = time.time()
    with mp.Pool(a.jobs, initializer=_winit) as pool:
        for j, nm in enumerate(sets):
            tasks = [(a.seed0 + g, nm, kinds[ki], a.recv, opt) for (ki, g) in cams]
            for idx, r in enumerate(pool.imap(R._trial, tasks, chunksize=4)):
                L = lfun(r[1][sel].astype(np.float64)).astype(np.float32); A[idx] += L; seat[idx] = r[0]
                if full:
                    Lall[idx, j] = L
                else:
                    if W8 is not None and j < 8:
                        W8[idx] += L
                    if W4 is not None and j < 4:
                        W4[idx] += L
                    if j in (0, ns // 2):
                        L1[idx, 0 if j == 0 else 1] = L
            print("%s 순간 %d/%d  %.0f초" % (a.tag, j + 1, ns, time.time() - t0), flush=True)
    extra = dict(Lall=Lall) if full else dict(L1=L1, **({"W8": W8} if W8 is not None else {}), **({"W4": W4} if W4 is not None else {}))
    np.savez(os.path.join(OUT, "place_%s.npz" % a.tag), A=A, seat=seat, kind=np.array([kinds[ki] for ki, _ in cams]), spots=D0["spots"][sel], sel=np.array(sel),
             sets=np.array(sets), opt=json.dumps(opt), seats_xy=seats_xy, owner=pts.owner, **extra)
    print("저장 place_%s.npz  %.0f초" % (a.tag, time.time() - t0), flush=True)


class Judge:
    """후보 점을 자리끼리 묶어 두고, 로그 합 [C, N] -> 카메라마다 (정확히, 옆자리까지, 후보 셋)"""
    def __init__(self, owner, seats_xy, seat):
        self.order = np.argsort(owner, kind="stable"); cnt = np.bincount(owner); assert (cnt == cnt[0]).all(); self.m = int(cnt[0]); self.ns = len(cnt)
        self.rc = np.array([K.seat_rc(xy, seats_xy) for xy in seats_xy]); self.seat = seat

    def __call__(self, Ls):
        Ls = Ls[:, self.order]; P = np.exp(Ls - Ls.max(1, keepdims=True)).reshape(len(Ls), self.ns, self.m).sum(-1)
        o = np.argpartition(-P, 3, 1)[:, :3]; o = np.take_along_axis(o, np.argsort(-np.take_along_axis(P, o, 1), 1), 1); b = o[:, 0]; s = self.seat
        near = (np.abs(self.rc[b, 0] - self.rc[s, 0]) <= 1) & (np.abs(self.rc[b, 1] - self.rc[s, 1]) <= 1)
        return np.stack([b == s, near, (o == s[:, None]).any(1)], 1), b


def metrics(res, seat, ns_seat=40):
    """카메라마다의 판정 [C, 3] -> 평균과 자리마다 (정확히) 의 통계"""
    ex = res[:, 0].astype(float); per = np.array([ex[seat == k].mean() for k in range(ns_seat)]); nr = np.array([res[seat == k, 1].mean() for k in range(ns_seat)])
    srt = np.sort(per)
    return dict(exact=float(ex.mean()), near=float(res[:, 1].mean()), top3=float(res[:, 2].mean()), min_seat=float(srt[0]), worst5=float(srt[:5].mean()),
                below40=int((per < 0.4).sum()), below60=int((per < 0.6).sum()), below80=int((per < 0.8).sum()), min_near=float(nr.min()), per_seat=per.round(3).tolist())


def _score_many(args):
    combos, key = args; Z = _SH["Z"]; J = _SH["J"]; out = []
    for c in combos:
        Ls = Z[key][:, list(c)].sum(1); res, _ = J(Ls); m = metrics(res, J.seat); m.pop("per_seat"); out.append((c, m))
    return out


_SH = {}


def rank_key(m):
    """사각지대가 적은 차례: 가장 약한 자리 -> 약한 다섯 자리 평균 -> 평균"""
    return (m["min_seat"], m["worst5"], m["exact"])


def search(a):
    import multiprocessing as mp
    Z = np.load(os.path.join(OUT, "place_%s.npz" % a.tag)); Z = {k: Z[k] for k in Z.files}
    spots = Z["spots"].astype(float); sel = Z["sel"].tolist(); S = len(sel); seat = Z["seat"]; kinds = list(dict.fromkeys(Z["kind"].tolist()))
    J = Judge(Z["owner"], Z["seats_xy"], seat); _SH.update(Z=Z, J=J)
    allsp = PS.all_spots(); names = [PS.name(tuple(spots[i]), sel[i]) for i in range(S)]
    out = dict(tag=a.tag, spots=spots.tolist(), sel=sel, names=names, wall=[PS.where(tuple(p)) for p in spots], reach=[bool(PS.reach(tuple(p))) for p in spots])
    t0 = time.time(); ctx = mp.get_context("fork")
    # 한 곳
    singles = [((i,), m) for (i,), m in [(c, m) for c, m in _score_many(([(i,) for i in range(S)], "A"))]]
    out["single"] = [dict(combo=c, **m) for c, m in sorted(singles, key=lambda t: -t[1]["exact"])]
    # 두 곳: 모두
    pairs = list(itertools.combinations(range(S), 2)); chunks = [pairs[i::a.jobs] for i in range(a.jobs)]
    with ctx.Pool(a.jobs) as pool:
        P = [x for part in pool.map(_score_many, [(c, "A") for c in chunks]) for x in part]
    P.sort(key=lambda t: rank_key(t[1]), reverse=True); out["pairs_n"] = len(P)
    out["pairs"] = [dict(combo=c, **m) for c, m in P[:a.top]]
    out["pairs_by_mean"] = [dict(combo=c, **m) for c, m in sorted(P, key=lambda t: -t[1]["exact"])[:a.top]]
    for cat, ok in (("reach", lambda c: all(out["reach"][i] for i in c)), ("noceil", lambda c: all(out["wall"][i] != "천장" for i in c))):
        out["pairs_" + cat] = [dict(combo=c, **m) for c, m in P if ok(c)][:a.top]
    print("두 곳 %d가지 %.0f초" % (len(P), time.time() - t0), flush=True)
    # 세 곳: 모두 (곳이 많으면 두 곳 순위 위쪽에 든 곳들로 좁힌다)
    if a.triples:
        cand = list(range(S))
        if S > a.tri_max:
            sc = {}
            for c, m in P[:400]:
                for i in c:
                    sc[i] = sc.get(i, 0) + 1
            cand = sorted(sc, key=lambda i: -sc[i])[:a.tri_max]
        tri = list(itertools.combinations(sorted(cand), 3)); chunks = [tri[i::a.jobs] for i in range(a.jobs)]
        with ctx.Pool(a.jobs) as pool:
            T = [x for part in pool.map(_score_many, [(c, "A") for c in chunks]) for x in part]
        # 가장 좋은 두 곳 + 아무 곳 하나 (후보 밖의 곳도)
        extra = set()
        for c, m in P[:30]:
            for i in range(S):
                if i not in c:
                    extra.add(tuple(sorted(c + (i,))))
        extra -= set(t[0] for t in T)
        if extra:
            ex = sorted(extra); chunks = [ex[i::a.jobs] for i in range(a.jobs)]
            with ctx.Pool(a.jobs) as pool:
                T += [x for part in pool.map(_score_many, [(c, "A") for c in chunks]) for x in part]
        T.sort(key=lambda t: rank_key(t[1]), reverse=True); out["triples_n"] = len(T); out["tri_cand"] = cand
        out["triples"] = [dict(combo=c, **m) for c, m in T[:a.top]]
        for cat, ok in (("reach", lambda c: all(out["reach"][i] for i in c)), ("noceil", lambda c: all(out["wall"][i] != "천장" for i in c))):
            out["triples_" + cat] = [dict(combo=c, **m) for c, m in T if ok(c)][:a.top]
        print("세 곳 %d가지 %.0f초" % (len(T), time.time() - t0), flush=True)
    # 자리마다 (위쪽 몇 개와 지금 거는 곳)
    def detail(c):
        res, b = J(Z["A"][:, list(c)].sum(1)); m = metrics(res, seat)
        for key in ("W8", "W4"):
            if key in Z:
                r2, _ = J(Z[key][:, list(c)].sum(1)); m[key] = metrics(r2, seat); m[key].pop("per_seat")
        if "L1" in Z:
            r1 = np.concatenate([J(Z["L1"][:, w, list(c)].astype(np.float32).sum(1))[0] for w in range(Z["L1"].shape[1])])
            m["K1"] = metrics(r1, np.tile(seat, Z["L1"].shape[1])); m["K1"].pop("per_seat")
        m["by_kind"] = {k: [float(v) for v in res[Z["kind"] == k].mean(0)] for k in kinds}
        return m
    want = {}
    for key in ("pairs", "pairs_reach", "pairs_noceil", "triples", "triples_reach", "triples_noceil"):
        for r in out.get(key, [])[:a.detail]:
            want[tuple(r["combo"])] = 1
    now = [i for i, s in enumerate(sel) if s in (7, 17)]
    if len(now) == 2:
        want[tuple(now)] = 1; out["now"] = now
    out["detail"] = {",".join(map(str, c)): detail(c) for c in want}
    with open(os.path.join(OUT, "place_%s.json" % a.tag), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print("저장 place_%s.json  %.0f초" % (a.tag, time.time() - t0), flush=True)
    show(out)


def show(out, n=8):
    nm = out["names"]
    def line(r):
        return "%-70s 정확 %3.0f 옆 %3.0f 셋 %3.0f | 최약 %3.0f 약5 %3.0f <40:%d <60:%d <80:%d" % (
            " + ".join(nm[i] for i in r["combo"])[:70], 100 * r["exact"], 100 * r["near"], 100 * r["top3"], 100 * r["min_seat"], 100 * r["worst5"], r["below40"], r["below60"], r["below80"])
    for key in ("pairs", "pairs_noceil", "pairs_reach", "triples", "triples_noceil", "triples_reach"):
        if key in out:
            print("--", key)
            for r in out[key][:n]:
                print(line(r))
    if "now" in out:
        print("-- 지금 거는 곳"); d = out["detail"][",".join(map(str, out["now"]))]; print(line(dict(combo=out["now"], **d)))


def curve(a):
    """K = 1..ns: 이어진 K 순간을 모은다 (순간을 돌려 가며 모든 시작점)"""
    Z = np.load(os.path.join(OUT, "place_%s.npz" % a.tag)); Z = {k: Z[k] for k in Z.files}
    seat = Z["seat"]; L = Z["Lall"].astype(np.float32); C, ns = L.shape[:2]; J = Judge(Z["owner"], Z["seats_xy"], seat); kinds = list(dict.fromkeys(Z["kind"].tolist()))
    sel = Z["sel"].tolist(); out = dict(tag=a.tag, sel=sel, spots=Z["spots"].tolist(), ks=[int(k) for k in a.ks], combos={})
    for cs in a.combos:
        c = [int(i) for i in cs.split(",")]; Lc = L[:, :, c].sum(2); rows = {}
        cum = np.concatenate([np.zeros((C, 1, Lc.shape[-1]), np.float32), np.cumsum(np.concatenate([Lc, Lc], 1), 1)], 1)
        for k in a.ks:
            k = int(k); starts = range(ns) if k < ns else [0]
            res = np.concatenate([J(cum[:, s + k] - cum[:, s])[0] for s in starts]); st = np.tile(seat, len(starts)); kd = np.tile(Z["kind"], len(starts))
            m = metrics(res, st); m["by_kind"] = {kk: [float(v) for v in res[kd == kk].mean(0)] for kk in kinds}; rows[k] = m
            print("%-12s K=%2d  정확 %3.0f 옆 %3.0f 셋 %3.0f | 최약 %3.0f 약5 %3.0f <40:%d <80:%d   %s" % (cs, k, 100 * m["exact"], 100 * m["near"], 100 * m["top3"], 100 * m["min_seat"],
                  100 * m["worst5"], m["below40"], m["below80"], " ".join("%s %.0f/%.0f" % (kk, 100 * v[0], 100 * v[1]) for kk, v in m["by_kind"].items())), flush=True)
        out["combos"][cs] = rows
    with open(os.path.join(OUT, "place_%s_curve.json" % a.tag), "w") as f:
        json.dump(out, f, ensure_ascii=False)


def evaluate(a):
    """정해 둔 조합들을 (검증 카메라로 잰) 묶음에서 다시 본다. --combos 는 곳 번호(kr40_tg 차례)"""
    Z = np.load(os.path.join(OUT, "place_%s.npz" % a.tag)); Z = {k: Z[k] for k in Z.files}
    sel = Z["sel"].tolist(); seat = Z["seat"]; J = Judge(Z["owner"], Z["seats_xy"], seat); kinds = list(dict.fromkeys(Z["kind"].tolist())); out = {}
    for cs in a.combos:
        c = [sel.index(int(i)) for i in cs.split(",")]; res, _ = J(Z["A"][:, c].sum(1)); m = metrics(res, seat)
        m["by_kind"] = {kk: [float(v) for v in res[Z["kind"] == kk].mean(0)] for kk in kinds}; out[cs] = m
        print("%-16s 정확 %3.0f 옆 %3.0f 셋 %3.0f | 최약 %3.0f 약5 %3.0f <40:%d <60:%d <80:%d" % (cs, 100 * m["exact"], 100 * m["near"], 100 * m["top3"], 100 * m["min_seat"],
              100 * m["worst5"], m["below40"], m["below60"], m["below80"]), flush=True)
    with open(os.path.join(OUT, "place_%s_eval.json" % a.tag), "w") as f:
        json.dump(out, f, ensure_ascii=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--tag", default="g16"); ap.add_argument("--sets", nargs="+")
    ap.add_argument("--n", type=int, default=16); ap.add_argument("--first", type=int, default=1); ap.add_argument("--base", nargs="+", default=["kr40_t", "kr40_g"]); ap.add_argument("--to", default="kr40_tg")
    ap.add_argument("--cams", type=int, default=400); ap.add_argument("--seed0", type=int, default=20000); ap.add_argument("--kinds", nargs="+", default=["책상", "가슴", "안경"])
    ap.add_argument("--recv", default="ura4x4p37.5h"); ap.add_argument("--opt", default='{"mount":"auto","tilt":15,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05}')
    ap.add_argument("--spots", nargs="*"); ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2)); ap.add_argument("--full", action="store_true")
    ap.add_argument("--top", type=int, default=40); ap.add_argument("--detail", type=int, default=5); ap.add_argument("--triples", type=int, default=1); ap.add_argument("--tri_max", type=int, default=200)
    ap.add_argument("--combos", nargs="*"); ap.add_argument("--ks", nargs="*", default=["1", "2", "4", "8", "16", "24", "32"])
    a = ap.parse_args()
    dict(merge=merge, subset=subset, measure=measure, search=search, curve=curve, evaluate=evaluate)[a.cmd](a)


if __name__ == "__main__":
    main()
