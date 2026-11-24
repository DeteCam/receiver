"""Tables for the palm-size receiver report from the res_pe2_*.npy files.  Usage: mk_tables.py [TAG ...]  (no argument = the standard report)."""
import sys, os, numpy as np

def load(tag):
    f = f"res_pe2_{tag}.npy"
    return np.load(f, allow_pickle=True).item() if os.path.exists(f) else None

def q(e): return f"{np.mean(e):.1f}° / {np.median(e):.1f}° / {np.percentile(e, 90):.1f}°"

def single(d, nodes=None, conf=None):
    R = [x for x in d["rows"] if not x["beacon"] and (nodes is None or x["n"] in nodes)]
    e = np.array([x["err"] for x in R])
    if conf:
        sp = np.array([x["spread"] for x in R]); e = e[sp <= np.percentile(sp, conf)]
    return e

def fused(d, sub):
    P = [x for x in d["pos"] if x["sub"] == sub and not x["beacon"]]
    if not P: return None
    fa = np.concatenate([x["fuse_az"] for x in P]); pe = np.array([x["err"] for x in P]); st = np.array([x["seat"] for x in P], float)
    return fa, pe, st

def line(tag, label=None, front=(0, 1), pair="x2 adj"):
    d = load(tag)
    if d is None: return None                                               # that run is not in this folder
    nn = max(x["n"] for x in d["rows"]) + 1; out = [label or tag, q(single(d))]
    if nn == 4:
        out += [q(single(d, front)), f"{single(d, front, 80).mean():.1f}°"]
        fa, pe, st = fused(d, pair); out += [q(fa), f"{100 * np.mean(pe <= 1):.0f}%"]
        out += [f"{100 * st[:, 0].mean():.0f}% / {100 * st[:, 1].mean():.0f}%" if np.isfinite(st).all() else "–"]
        fa, pe, st = fused(d, "x4"); out += [q(fa), f"{100 * np.mean(pe <= 1):.0f}%"]
    else:
        out += ["–", f"{single(d, None, 80).mean():.1f}°"]; fa, pe, st = fused(d, "x2"); out += [q(fa), f"{100 * np.mean(pe <= 1):.0f}%", "–", "–", "–"]
    return "| " + " | ".join(out) + " |"

HEAD = ("| 조건 | 한 자리, 네 모서리 평균 | 한 자리, 좋은 두 자리 | 믿을 만한 80% (평균) | 두 자리 융합 방향 | 두 자리 1 m 이내 | 자리 / 3×3 | 네 자리 융합 방향 | 네 자리 1 m 이내 |\n"
        "|---|---|---|---|---|---|---|---|---|")

def show(x):
    if x: print(x)

def pernode(tag):
    d = load(tag)
    if d is None: return
    for n in sorted(set(x["n"] for x in d["rows"])):
        R = [x for x in d["rows"] if x["n"] == n and not x["beacon"]]; e = np.array([x["err"] for x in R]); lf = np.array([x["lf"] for x in R])
        print(f"  {tag} spot {n}: {q(e)}   direct path >= 20% of power: {100 * np.mean(lf >= 0.2):.0f}%")

if __name__ == "__main__":
    tags = sys.argv[1:]
    if tags:
        print(HEAD)
        for t in tags: show(line(t, front=(0, 1) if load(t) and load(t)["scene"] == "classroom" else (0, 3), pair="x2 adj" if load(t) and load(t)["scene"] == "classroom" else "x2 diag"))
        for t in tags: pernode(t)
        sys.exit()
    print("## classroom 2.4 GHz (mean / median / P90)"); print(HEAD)
    for t, l in (("s_base", "이상적 안테나"), ("s_emfree", "실제 응답, 금속 없음"), ("s_em100", "실제 응답, 금속 10 cm 아래"), ("s_em60", "금속 6 cm 아래"), ("s_emr115", "기준 아래, 금속 11.5 cm"),
                 ("s_emr75", "기준 아래, 금속 7.5 cm"), ("s_emr75f", "같은 구조, 뒤집어 매달기"), ("s_em65", "금속 1.5 cm 아래"), ("s_em0", "기준 같은 높이, 금속 1.5 cm"), ("c_r90em", "실제 응답, 팔 9 cm"), ("xl24em", "실제 응답, 앞쪽 낮은 자리"), ("xe24", "이상적, 반 미터씩 옮긴 네 자리")): show(line(t, l))
    print("\n## classroom 5 GHz"); print(HEAD)
    for t, l in (("f_c5", "이상적 안테나"), ("s5_em100", "실제 응답, 금속 10 cm 아래"), ("s5_bw80", "80 MHz"), ("xl5em", "실제 응답, 앞쪽 낮은 자리")): show(line(t, l))
    print("\n## motel"); print(HEAD)
    for t, l in (("f_m24", "2.4 GHz 이상적"), ("m_em100", "2.4 GHz 실제 응답"), ("m_em100f", "2.4 GHz 실제 응답, 뒤집어 매달기"), ("m_r90em", "2.4 GHz 실제 응답, 팔 9 cm"), ("f_m5", "5 GHz 이상적"), ("m5_em100", "5 GHz 실제 응답"),
                 ("xc24em", "2.4 GHz 실제 응답, 벽에서 떨어진 자리"), ("xd24em", "2.4 GHz 실제 응답, 반 미터씩 옮긴 네 자리(좋은 모서리)"), ("xf24em", "같음(나쁜 모서리)"),
                 ("w_conc", "2.4 GHz 이상적, 콘크리트 벽"), ("w_metal", "2.4 GHz 이상적, 금속 벽")): show(line(t, l, front=(0, 3), pair="x2 diag"))
    print("\n## sensitivity, classroom 2.4 GHz"); print(HEAD)
    for t, l in (("s_base", "기준"), ("s_dim10", "방 크기 오차 10 cm"), ("s_dim25", "방 크기 오차 25 cm"), ("s_pos10", "수신기 위치 오차 10 cm"), ("s_pos25", "수신기 위치 오차 25 cm"),
                 ("s_head3", "수신기 방향 오차 3°"), ("s_jit3", "팔 각도 오차 3°"), ("s_jit8", "팔 각도 오차 8°"), ("s_cal15", "자리별 위상 오차 15°"), ("s_cal30", "자리별 위상 오차 30°"),
                 ("s_n20", "잡음 -20 dB"), ("s_tv50", "반사 절반이 흔들림"), ("s_tv50x4", "반사 절반이 흔들림, 네 바퀴 평균"), ("s_tv100", "반사 전부가 흔들림"), ("s_tv100x4", "반사 전부가 흔들림, 네 바퀴 평균")): show(line(t, l))
    print("\n## calibration, classroom 2.4 GHz (8 beacons left out of the statistics)"); print(HEAD)
    for t, l in (("s_base8", "오차 없음"), ("s_head6", "방향 오차 6°"), ("s_head6cal", "방향 오차 6° + 비콘 보정"), ("s_all", "방향 6° · 위치 20 cm · 방 크기 20 cm"), ("s_allcal", "같은 오차 + 비콘 보정"),
                 ("s_reflcal", "오차 없음 + 반사율 보정"), ("s_allcalr", "같은 오차 + 비콘 · 반사율 보정")): show(line(t, l))
    print("\n## per spot"); [pernode(t) for t in ("s_base", "s_em100", "f_c5", "s5_em100", "f_m24", "m_em100", "f_m5", "m5_em100")]
