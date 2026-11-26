"""AR suspect-zone threshold rules: no calibration vs pooled split conformal (report) vs worst-room rule.

Thresholds are set on the 4 dev rooms only (diagonal pair), then coverage is measured on the 9 eval rooms and the
4 fresh rooms.
  HPD90      : uncalibrated softmax highest-posterior set holding 90 % mass
  pooled     : one conformal quantile over all dev cameras pooled (what the report did)
  worst-room : conformal quantile per dev room, take the largest (every dev room reaches 90 %)
Coverage = share of cameras whose true cell lies inside the zone; area = zone share of the room floor.
Also checks whether the threshold set with the diagonal pair carries over to the board-side pair in classrooms.

Zone statistics are stored only in rev2.pkl (pairs (0,1) and diagonal), fresh.pkl (all six pairs) and
zone_front.pkl (classroom_b / lecture_b, all six pairs; made by exp_zone_front.py).

Usage: python3 mysim/analyze_zone.py
"""
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from my_eval import TAUS  # noqa: E402

RES = "/opt/detecam/sim/results"
DEV = ["classroom_a", "lodging_a", "restroom_a", "locker_a"]
EVAL = ["lecture_a", "lecture_b", "bathhouse_a", "bathhouse_b", "classroom_b", "lodging_b",
        "restroom_b", "locker_b", "worst_metal"]
FRESH = ["classroom_c", "lodging_c", "office_a", "restroom_c"]
CFGS = ("EXACT_D85_MAP", "S5CAL_D85_MAP")   # Rev.2 default, and with S5 calibration
ALPHA = 0.10


def load(name):
    with open(os.path.join(RES, name), "rb") as f:
        return pickle.load(f)


def scores(res, cfg, band, scene, pair):
    s, a = [], []
    for key in sorted(k for k in res if k[0] == scene and k[1] == band):
        o = res[key][1].get(cfg)
        if o is None:
            continue
        s.append(np.asarray(o["zone_s"][pair], dtype=np.float64))
        a.append(np.asarray(o["zone_area"][pair], dtype=np.float64))
    if not s:
        return None, None
    return np.concatenate(s), np.vstack(a)


def conformal_q(s, alpha=ALPHA):
    s = np.sort(np.asarray(s))
    n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))) - 1
    return float(s[min(k, n - 1)])


def area_at(areas, q):
    """areas: (n, len(TAUS)) zone share at each mass level; linear interpolation in the mass level."""
    return np.array([np.interp(q, TAUS, row) for row in areas])


def main():
    base, fresh, front = load("rev2.pkl"), load("fresh.pkl"), load("zone_front.pkl")
    # board-side pair per classroom-type room, and where its zone data lives
    board = {"lecture_a": (base, "adj"), "classroom_b": (front, "p02"),
             "lecture_b": (front, "p02"), "classroom_c": (fresh, "p23")}
    for cfg in CFGS:
        for band in ("2g4", "5g"):
            dev = {sc: scores(base, cfg, band, sc, "diag")[0] for sc in DEV}
            q_pool = conformal_q(np.concatenate(list(dev.values())))
            q_worst = max(conformal_q(v) for v in dev.values())
            print(f"\n== {cfg} [{band}] 합친 규칙 q={q_pool:.3f}, 최악 방 규칙 q={q_worst:.3f} "
                  f"(개발 방별 q: " + ", ".join(f"{sc} {conformal_q(v):.3f}" for sc, v in dev.items()) + ")")
            for name, q in (("보정 없음(HPD90)", 0.90), ("합친 규칙", q_pool), ("최악 방 규칙", q_worst)):
                parts, room_line = [], ""
                for label, rooms, res in (("평가 9방", EVAL, base), ("새 방 4곳", FRESH, fresh)):
                    cov, area, per = [], [], {}
                    for sc in rooms:
                        s, a = scores(res, cfg, band, sc, "diag")
                        per[sc] = float((s <= q).mean())
                        cov.append(s <= q)
                        area.append(area_at(a, q))
                    worst = min(per, key=per.get)
                    parts.append(f"{label} 포함 {np.concatenate(cov).mean()*100:5.1f}% · 가장 낮은 방 {worst} "
                                 f"{per[worst]*100:4.1f}% · 넓이 {np.concatenate(area).mean()*100:4.1f}%")
                    room_line += " ".join(f"{sc}:{v*100:.0f}" for sc, v in per.items()) + "  "
                print(f"  {name:14s} | " + " | ".join(parts))
                print(f"  {'':14s}   방별 포함률 {room_line}")
            for name, q in (("합친 규칙", q_pool), ("최악 방 규칙", q_worst)):
                fc, fa, dc, da = [], [], [], []
                for sc, (res, pr) in board.items():
                    sf, af = scores(res, cfg, band, sc, pr)
                    sd, ad = scores(res, cfg, band, sc, "diag")
                    fc.append(sf <= q)
                    fa.append(area_at(af, q))
                    dc.append(sd <= q)
                    da.append(area_at(ad, q))
                print(f"  {name:14s} | 교실형 4방(교실 B, 강의실 A·B, 교실 C) 칠판 쪽 짝 포함 "
                      f"{np.concatenate(fc).mean()*100:4.1f}% 넓이 {np.concatenate(fa).mean()*100:4.1f}% "
                      f"vs 대각 포함 {np.concatenate(dc).mean()*100:4.1f}% 넓이 {np.concatenate(da).mean()*100:4.1f}%")


if __name__ == "__main__":
    main()
