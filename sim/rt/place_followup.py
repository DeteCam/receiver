"""자리 찾기(run_place.sh) 다음의 일. 이미 한 것은 건너뛴다.

    python place_followup.py all        아래를 차례로

1) 검증: search 가 고른 배치들을 새 카메라(씨앗 40000)로 다시 잰다           -> study/place_v16_eval.json
2) 최종 배치: 검증한 숫자로 두 곳·세 곳을 고른다 (손이 닿는 곳 / 천장 빼고 / 아무 곳)  -> study/place_final.json
3) K 곡선: 고른 곳만 따로 순간 17~32 를 더 계산해 사진 1~32장                    -> study/place_c32_curve.json
4) 대안 수신기 (수신 체인 둘: 기준 안테나 + 스위치 패널, 'r')                     -> study/place_r32_curve.json
5) 나쁜 것이 모두 겹친 패널 (이득 1 dB, 위상 단계 20도, 세기 튀기, 8비트)          -> study/place_x16_curve.json
6) 5G (3.55 GHz) 휴대폰을 3.5 GHz 패널(4x4, 반 파장)과 2채널 수신기로             -> study/place_n16_curve.json
7) 벽 패널을 거는 각도 (경첩 0도·30도 숙이기, 한가운데를 똑바로)                     -> study/place_o0/o30/oa_curve.json
"""
import os, sys, json, subprocess, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "study"); PATHS = os.path.join(HERE, "paths")
PY = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(HERE))), ".venv-rt", "bin", "python")
LOG = os.path.join(OUT, "log_place2.txt"); JOBS = "11"
sys.path.insert(0, HERE)
import place_spots as PS                                            # noqa: E402


def log(*a):
    with open(LOG, "a") as f:
        print(time.strftime("%H:%M:%S"), *a, file=f, flush=True)


def run(args, env=None):
    log("$", " ".join(args[1:])[:300])
    e = dict(os.environ); e.update(env or {})
    with open(LOG, "a") as f:
        r = subprocess.run(args, cwd=HERE, stdout=f, stderr=subprocess.STDOUT, env=e)
    if r.returncode:
        log("실패", r.returncode); sys.exit(r.returncode)


def rank(m):
    return (m["min_seat"], m["worst5"], m["exact"])


def ok_cat(c, cat, wall, reach):
    if cat == "reach":
        return all(reach[i] for i in c)
    if cat == "noceil":
        return all(wall[i] != "천장" for i in c)
    return True


def step_validate():
    G = json.load(open(os.path.join(OUT, "place_g16.json"), encoding="utf-8")); sel = G["sel"]
    cand = {tuple(G["now"]): 1}
    for key, n in (("pairs", 8), ("pairs_reach", 5), ("pairs_noceil", 5), ("pairs_by_mean", 3), ("triples", 6), ("triples_reach", 4), ("triples_noceil", 4)):
        for r in G.get(key, [])[:n]:
            cand[tuple(r["combo"])] = 1
    combos = [tuple(sel[i] for i in c) for c in cand]                 # 곳 번호(kr40_tg 차례)
    spots = sorted({i for c in combos for i in c})
    json.dump(dict(combos=[list(c) for c in combos], spots=spots), open(os.path.join(OUT, "place_cand.json"), "w"))
    if not os.path.exists(os.path.join(OUT, "place_v16.npz")):
        run([PY, "place_study.py", "measure", "--tag", "v16", "--sets"] + ["kr40_tg_s%d" % s for s in range(1, 17)] +
            ["--cams", "400", "--seed0", "40000", "--jobs", JOBS, "--spots"] + [str(i) for i in spots])
    run([PY, "place_study.py", "evaluate", "--tag", "v16", "--combos"] + [",".join(map(str, c)) for c in combos])


def step_final():
    G = json.load(open(os.path.join(OUT, "place_g16.json"), encoding="utf-8")); V = json.load(open(os.path.join(OUT, "place_v16_eval.json"), encoding="utf-8"))
    sp = [tuple(p) for p in PS.all_spots()]; wall = [PS.where(p) for p in sp]; reach = [bool(PS.reach(p)) for p in sp]
    items = [(tuple(int(i) for i in k.split(",")), m) for k, m in V.items()]
    now = tuple(G["sel"][i] for i in G["now"]); fin = dict(now=list(now))
    for n in (2, 3):
        for cat in ("reach", "noceil", "any"):
            c = [t for t in items if len(t[0]) == n and ok_cat(t[0], cat, wall, reach)]
            if c:
                best = max(c, key=lambda t: rank(t[1])); fin["best%d_%s" % (n, cat)] = list(best[0])
    keys = list(fin); F = sorted({i for k in keys for i in fin[k]})
    fin["spots"] = F; fin["coords"] = [list(sp[i]) for i in F]; fin["names"] = [PS.name(sp[i], i) for i in F]
    fin["local"] = {k: [F.index(i) for i in fin[k]] for k in keys}
    json.dump(fin, open(os.path.join(OUT, "place_final.json"), "w"), ensure_ascii=False, indent=1)
    log("최종", json.dumps(fin, ensure_ascii=False))
    return fin


def spotlist(fin):
    return ";".join(",".join("%.3f" % v for v in p) for p in fin["coords"])


def combos_local(fin):
    keys = [k for k in fin["local"]]
    seen = []
    for k in keys:
        c = ",".join(map(str, fin["local"][k]))
        if c not in seen:
            seen.append(c)
    return seen


def step_curve(fin):
    F = fin["spots"]
    run([PY, "place_study.py", "subset", "--base", "kr40_tg", "--to", "kr40_cv", "--n", "16", "--spots"] + [str(i) for i in F])
    for s in range(17, 33):
        if not os.path.exists(os.path.join(PATHS, "kr40_cv_s%d.npz" % s)):
            run([PY, "rt_paths.py", "kr40_cv_s%d" % s, "--spotlist", spotlist(fin), "--seed", str(100 + s), "--jitter", "0.07", "--pose", "1"])
    sets = ["kr40_cv_s%d" % s for s in range(1, 33)]
    if not os.path.exists(os.path.join(OUT, "place_c32.npz")):
        run([PY, "place_study.py", "measure", "--tag", "c32", "--sets"] + sets + ["--cams", "400", "--seed0", "60000", "--jobs", JOBS, "--full"])
    run([PY, "place_study.py", "curve", "--tag", "c32", "--combos"] + combos_local(fin))
    if not os.path.exists(os.path.join(OUT, "place_r32.npz")):
        run([PY, "place_study.py", "measure", "--tag", "r32", "--recv", "ura4x4p37.5r", "--sets"] + sets + ["--cams", "400", "--seed0", "60000", "--jobs", JOBS, "--full"])
    run([PY, "place_study.py", "curve", "--tag", "r32", "--combos"] + combos_local(fin))
    if not os.path.exists(os.path.join(OUT, "place_x16.npz")):
        opt = '{"mount":"auto","tilt":15,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05,"imp":{"gjit":1.0,"sjit":0.5,"pherr":20,"txp":0.1,"txdb":2,"q8":8}}'
        run([PY, "place_study.py", "measure", "--tag", "x16", "--opt", opt, "--sets"] + sets[:16] + ["--cams", "400", "--seed0", "60000", "--jobs", JOBS, "--full"])
    run([PY, "place_study.py", "curve", "--tag", "x16", "--ks", "1", "2", "4", "8", "16", "--combos"] + combos_local(fin))


def step_orient(fin):
    """벽 패널을 어떻게 걸까: 경첩으로 0도·30도 숙이기, 자리 한가운데를 똑바로 (지금 계산은 15도 숙이기). 천장이 든 배치는 빼고 본다"""
    sp = [tuple(p) for p in PS.all_spots()]
    walls = [c for c in combos_local(fin) if all(sp[fin["spots"][i]][2] < 2.6 for i in map(int, c.split(",")))]
    sets = ["kr40_cv_s%d" % s for s in range(1, 17)]
    for tag, opt in (("o0", '{"mount":"hinge","tilt":0,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05}'), ("o30", '{"mount":"hinge","tilt":30,"yaw_sd":2,"tilt_sd":1,"pos_sd":0.05}'),
                     ("oa", '{"mount":"aim","yaw_sd":2,"tilt_sd":1,"pos_sd":0.05}')):
        if not os.path.exists(os.path.join(OUT, "place_%s.npz" % tag)):
            run([PY, "place_study.py", "measure", "--tag", tag, "--opt", opt, "--sets"] + sets + ["--cams", "400", "--seed0", "60000", "--jobs", JOBS, "--full"])
        run([PY, "place_study.py", "curve", "--tag", tag, "--ks", "1", "4", "16", "--combos"] + walls)


def step_5g(fin):
    env = {"DETECAM_FC": "3.55e9"}
    for s in range(1, 17):
        if not os.path.exists(os.path.join(PATHS, "kr40_5g_s%d.npz" % s)):
            run([PY, "rt_paths.py", "kr40_5g_s%d" % s, "--spotlist", spotlist(fin), "--seed", str(100 + s), "--jitter", "0.07", "--pose", "1"], env)
    sets = ["kr40_5g_s%d" % s for s in range(1, 17)]
    if not os.path.exists(os.path.join(OUT, "place_n16.npz")):
        run([PY, "place_study.py", "measure", "--tag", "n16", "--recv", "ura4x4p42.2r", "--sets"] + sets + ["--cams", "400", "--seed0", "60000", "--jobs", JOBS, "--full"], env)
    run([PY, "place_study.py", "curve", "--tag", "n16", "--ks", "1", "2", "4", "8", "16", "--combos"] + combos_local(fin), env)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    log("# 시작", cmd)
    if cmd in ("all", "validate"):
        step_validate()
    fin = step_final() if cmd in ("all", "final", "curve", "5g", "orient") else None
    if cmd in ("all", "curve"):
        step_curve(fin)
    if cmd in ("all", "5g"):
        step_5g(fin)
    if cmd in ("all", "orient"):
        step_orient(fin)
    log("FOLLOWDONE", cmd)


if __name__ == "__main__":
    main()
