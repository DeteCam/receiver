"""Hardware-structure comparisons against the Rev.2 default (D = 85 mm, gap 60 mm, dipoles), same cameras.

  gap80 / gap100        antenna-to-box gap 60 -> 80 / 100 mm      (exp_gap.pkl,  eval 9 + fresh 4)
  gap40 / gap20 / gap10 smaller gap (lower total height)          (exp_size.pkl, eval 9)
  gp88_up / gp88_inv    ground-plane monopole ring, 88 mm plate   (exp_size.pkl, eval 9)
Default = R2 in exp_base.pkl (13 rooms) and EXACT_D85_MAP in exp_gap_fresh.pkl (fresh rooms).

Usage: python3 mysim/analyze_hw.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze as A  # noqa: E402

RES = "/opt/detecam/sim/results"
FRESH = ["classroom_c", "lodging_c", "office_a", "restroom_c"]


def merged(var_file, var_cfgs, base_file, base_cfg):
    var = A.load(os.path.join(RES, var_file))
    base = A.load(os.path.join(RES, base_file))
    comb = {}
    for k, (m, o) in var.items():
        if k not in base or base_cfg not in base[k][1]:
            continue
        d = {c: o[c] for c in var_cfgs if c in o}
        d["BASE"] = base[k][1][base_cfg]
        comb[k] = (m, d)
    return comb


def report(comb, cfgs, scenes, title):
    print(f"\n===== {title}")
    for band in ("2g4", "5g"):
        for c in cfgs:
            for metric in ("one", "diag_equal"):
                r = A.paired_boot(comb, c, "BASE", band, scenes, metric, n_boot=2000)
                if r:
                    print(f"  [{band}] {c:8s} {metric:10s} | {A.fmt_cmp(r)}")
            for pn in ("pos_diag_equal",):
                e1, _ = A.collect(comb, c, band, scenes, pn)
                e0, _ = A.collect(comb, "BASE", band, scenes, pn)
                if e1 is not None:
                    print(f"  [{band}] {c:8s} 위치오차 대각: {e1.mean():.2f} m vs 기본 {e0.mean():.2f} m")


def main():
    comb = merged("exp_gap.pkl", ["gap80", "gap100"], "exp_base.pkl", "R2")
    report(comb, ["gap80", "gap100"], A.EVAL_SCENES, "간격 60 -> 80/100 mm, 평가 9방")
    combf = A.load(os.path.join(RES, "exp_gap_fresh.pkl"))
    combf = {k: (m, {"gap80": o["gap80"], "gap100": o["gap100"], "BASE": o["EXACT_D85_MAP"]}) for k, (m, o) in combf.items()}
    report(combf, ["gap80", "gap100"], FRESH, "간격 60 -> 80/100 mm, 새 방 4곳")
    combs = merged("exp_size.pkl", ["gap40", "gap20", "gap10", "gp88_up", "gp88_inv"], "exp_base.pkl", "R2")
    report(combs, ["gap40", "gap20", "gap10", "gp88_up", "gp88_inv"], A.EVAL_SCENES, "더 작은 구조, 평가 9방")


if __name__ == "__main__":
    main()
