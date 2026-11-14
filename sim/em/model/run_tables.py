"""전파 표를 만든다 (판 각 72개 x 방향 37 x 72). 한 번 만든 것은 cache/ 에 남는다.
    python3 run_tables.py [묶음 이름 ...]      묶음: base(지금 구조, 부품을 하나씩 더해 가며), full(요크·받침까지, 받침 각 3개, 판 틀 전의 설계), alt(다른 판 모양·크기),
                                             dev(지금 설계 그대로, 받침 각 3개: 배포하는 표), dev6(지금 설계, 받침 각 6개), geo, shape(판 모양 비교)
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import sys, time
import numpy as np
import device as V

HEAD = ("ant", "conn", "head", "plate")
FULL = ("ant", "conn", "head", "yoke", "base", "plate", "cw")
JOBS = {
    "base": [
        ("ideal_z", ("plate",), dict(feed="z")),
        ("ideal_x", ("plate",), dict(feed="x")),
        ("ant", ("ant", "plate"), {}),
        ("ant_conn", ("ant", "conn", "plate"), {}),
        ("head", HEAD, {}),
        ("head_cw", HEAD + ("cw",), {}),
        ("head_nocable", HEAD, dict(cable=False)),
        ("ant_mono", ("ant", "plate"), dict(feed="mono")),
        ("head_mono", HEAD, dict(feed="mono")),
    ],
    "full": [("full_y%+04d" % y, FULL, dict(yaw=float(y))) for y in (0, 120, -120)] + [("full_nocw_y+000", FULL[:-1], dict(yaw=0.0))]
            + [("full6_y%+04d" % y, FULL, dict(yaw=float(y))) for y in (-150, -90, -30, 30, 90, 150)],
    "alt": [
        ("alt_100x200", HEAD, dict(plate=dict(z1=142.0))),
        ("alt_150x150", HEAD, dict(plate=dict(w=150.0))),
        ("alt_150x200", HEAD, dict(plate=dict(w=150.0, z1=142.0))),
        ("alt_200x200", HEAD, dict(plate=dict(w=200.0, z0=-80.0, z1=120.0, R=100.0))),
        ("alt_R60", HEAD, dict(plate=dict(R=66.0))),
        ("alt_R100", HEAD, dict(plate=dict(R=100.0))),
        ("alt_R120", HEAD, dict(plate=dict(R=120.0))),
        ("alt_arc100", HEAD, dict(plate=dict(kind="arc"))),
        ("alt_arc150", HEAD, dict(plate=dict(kind="arc", w=150.0))),
        ("alt_vee30", HEAD, dict(plate=dict(kind="vee", ang=30.0, w=120.0))),
        ("alt_wings30", HEAD, dict(plate=dict(kind="wings", d=30.0))),
        ("alt_60x150", HEAD, dict(plate=dict(w=60.0))),
    ],
}

# 판 모양·크기 비교: 안테나 모형 4가지(이상적인 전기·자기 쌍극자, 기판+틈, 기판+막대) x 판 13가지
ANTS = {"ez": (("plate",), dict(feed="z")), "mz": (("plate",), dict(feed="mz")), "slot": (("ant", "conn", "plate"), dict(feed="slot")), "mono": (("ant", "conn", "plate"), dict(feed="mono"))}
PLATES = {
    "w100_R80": dict(), "w150_R80": dict(w=150.0), "w200_R80": dict(w=200.0), "w100_R100": dict(R=100.0), "w150_R100": dict(w=150.0, R=100.0), "w200_R100": dict(w=200.0, R=100.0),
    "w150_R120": dict(w=150.0, R=120.0), "w200h200_R100": dict(w=200.0, R=100.0, z0=-100.0, z1=100.0), "vee150_R80": dict(kind="vee", w=150.0, ang=30.0),
    "tilt150_R80": dict(kind="tilt", w=150.0, tau=20.0), "w100h200_R80": dict(z1=142.0), "w60_R80": dict(w=60.0), "w150_R66": dict(w=150.0, R=66.0),
}
JOBS["geo"] = [("geo_%s_%s" % (a, pn), parts, dict(plate=pk, **kw)) for pn, pk in PLATES.items() for a, (parts, kw) in ANTS.items()]

# 판 모양 2차 비교 (2026-11-14 저녁): 틈, 철망(큰 구멍), 슬롯, 접기, 곡면. 폭 100, 높이 150, 팔 80 은 그대로
SHAPES = {
    "split10": dict(kind="split", g=10.0), "split20": dict(kind="split", g=20.0), "split40": dict(kind="split", g=40.0),
    "mesh10": dict(kind="mesh", mesh=10.0, rad=0.5), "mesh20": dict(kind="mesh", mesh=20.0, rad=0.5),
    "slotv60": dict(kind="slotv", L=60.0), "sloth60": dict(kind="sloth", L=60.0),
    "veein20": dict(kind="vee", ang=-20.0), "veeout20": dict(kind="vee", ang=20.0), "veeout40": dict(kind="vee", ang=40.0),
    "arc100": dict(kind="arc"), "flange15": dict(kind="flange", d=15.0), "wings15": dict(kind="wings", d=15.0),
}
JOBS["shape"] = [("geo_%s_%s" % (a, pn), parts, dict(plate=pk, **kw)) for pn, pk in SHAPES.items() for a, (parts, kw) in ANTS.items()]
JOBS["shape_a"] = JOBS["shape"][0::2]; JOBS["shape_b"] = JOBS["shape"][1::2]           # 두 줄로 나눠 돌릴 때

# 지금 설계 그대로 (판 틀로 감싼 알루미늄 판, 동전 9개 균형추, 요크·받침·배터리(두께 22)·전선): 받침 각 3개. 배포하는 무늬 표로 쓴다
JOBS["dev"] = [("dev_y%+04d" % y, FULL, dict(yaw=float(y))) for y in (0, 120, -120)]
JOBS["dev6"] = [("dev6_y%+04d" % y, FULL, dict(yaw=float(y))) for y in (-150, -90, -30, 30, 90, 150)]          # 여섯 자세로 잴 때 (plan_eval.py)

if __name__ == "__main__":
    groups = sys.argv[1:] or ["base", "full", "alt"]
    for g in groups:
        for name, parts, kw in JOBS[g]:
            t0 = time.time()
            try:
                spins, E = V.table(name, parts, **kw)
                V.no_plate(name + "_0", parts, **{k: v for k, v in kw.items() if k not in ("plate", "cw")})
                print("%-18s 끝 (%.0f초)  E %s" % (name, time.time() - t0, E.shape), flush=True)
            except Exception as e:
                print("%-18s 실패: %s" % (name, e), flush=True)
