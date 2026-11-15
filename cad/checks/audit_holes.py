"""나사 구멍 검사: 3D 프린터로 나사산을 뽑지 않으니, 나사가 스스로 나사산을 내는 구멍이 쓸 만한지 본다.
  1) 스스로 나사산을 내는 구멍(audit_screws.py 의 나사 가운데 너트가 없는 것)마다, 나사가 그 부품 안에 있는 구간에서
     구멍 둘레에 플라스틱 살이 얼마나 두껍게 있는지 (구멍 둘레를 8 쪽으로 나눠 가장 얇은 쪽, 0.5mm 간격). 1mm 아래면 돌려 넣을 때 갈라질 수 있다.
  2) 슬라이싱한 0판의 G코드에서 시험 조각의 나사 구멍이 몇 mm 로 찍히는지 (구멍 벽 경로에 원을 맞추고 선 폭의 반을 뺀다).
     실제 출력은 보통 이보다 0.1~0.2mm 작게 나온다: 그래서 시험 조각으로 먼저 돌려 본다 (조립 안내 3절).
결과: 결과 폴더의 audit_holes.json. 몇 분 걸린다."""
import os, sys, json, zipfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "slicing")]
import numpy as np
import paths
import audit_screws as A
import parts2 as P
from cadlib import M, CS

WALLS = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
THIN = 1.0


def frame(p, d):
    d = d / np.linalg.norm(d); x = np.cross([0, 0, 1.0], d)
    x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-9 else x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(d, x), d, p])


def wedge(a0, a1, R, L):
    return M.extrude(CS([[(0, 0)] + [(R * np.cos(a), R * np.sin(a)) for a in np.linspace(a0, a1, 6)]]), L)


def boss_walls():
    rows = []
    for name, p, d, L, through, thread, stage, head, opt in A.S:
        if thread is None:
            continue
        q = A.SPEC[opt.get("m", 3.0)]; p = np.asarray(p, float); d = np.asarray(d, float); d = d / np.linalg.norm(d)
        part = A.PARTS[thread][0]
        ts = []
        for t in np.arange(0.0, L, 0.5):                                       # 나사가 그 부품 안에 있는 곳
            seg = A.cyl(p + d * t, d, 0.5, q["shank"]) - A.cyl(p + d * (t - 0.1), d, 0.7, q["pilot"])
            if (seg ^ part).volume() / (np.pi * (q["shank"] ** 2 - q["pilot"] ** 2) * 0.5) > 0.5:
                ts.append(t)
        if not ts:
            continue
        t0, t1 = min(ts), max(ts) + 0.5; Ls = t1 - t0; T = frame(p + d * t0, d); sec = []
        for k in range(8):
            w_ok = 0.0
            for w in WALLS:
                tube = (M.cylinder(Ls, q["pilot"] + w, q["pilot"] + w, 48) - M.cylinder(Ls + 0.2, q["pilot"], q["pilot"], 48).translate([0, 0, -0.1])) \
                    ^ wedge(k * np.pi / 4, (k + 1) * np.pi / 4, q["pilot"] + w + 1, Ls)
                if (tube.transform(T) ^ part).volume() / tube.volume() >= 0.95:
                    w_ok = w
                else:
                    break
            sec.append(w_ok)
        rows.append(dict(name=name, m=opt.get("m", 3.0), part=thread, pilot=round(2 * q["pilot"], 2), engage=round(Ls, 1), wall_min=min(sec), walls=sec))
        print("%-16s M%-3g %-6s 구멍 %.1f  물림 %.1fmm  둘레 살 가장 얇은 쪽 %.1fmm 이상" % (name, rows[-1]["m"], thread, rows[-1]["pilot"], Ls, min(sec)), flush=True)
    return rows


# 시험 조각(test_coupon)의 나사 구멍: 모델 자리 (x, y)와 설계 지름. parts2.test_coupon() 과 같은 자리
COUPON_HOLES = [("M3 스스로 나사산", 24.0, 25.0, 2.6), ("M3 지나감", 32.0, 25.0, 3.4), ("M2.5 (점 없음)", 46.0, 21.0, 2.1), ("M2.5 (점)", 56.0, 21.0, 2.3),
                ("M2 (점 없음)", 46.0, 37.0, 1.6), ("M2 (점)", 56.0, 37.0, 1.8)]


def gcode_holes():
    import trimesh
    from print_check import parse, Y_OFF
    G = os.path.join(paths.SLICE, "gcode", "0판_시험조각.gcode.3mf")
    if not os.path.exists(G):
        print("0판 G코드가 없다:", G); return []
    S = json.load(open(os.path.join(paths.SLICE, "summary.json"), encoding="utf-8"))
    it = [x for x in S["plates"]["0판_시험조각"]["items"] if x[0] == "90_test_coupon"][0]
    st = trimesh.load(os.path.join(paths.STL, "90_test_coupon.stl")).bounds
    mb = np.array(P.test_coupon().bounding_box())                             # (x0, y0, z0, x1, y1, z1)
    sh = np.array([(mb[0] + mb[3]) / 2 - (st[0, 0] + st[1, 0]) / 2, (mb[1] + mb[4]) / 2 - (st[0, 1] + st[1, 1]) / 2])     # 모델 -> STL
    off = np.array([it[1] - (st[0, 0] + st[1, 0]) / 2, it[2] - Y_OFF - (st[0, 1] + st[1, 1]) / 2])                       # STL -> G코드
    z = zipfile.ZipFile(G); layers, feats = parse(z.read([n for n in z.namelist() if n.endswith(".gcode")][0]).decode())
    out = []
    for lab, hx, hy, dd in COUPON_HOLES:
        gx, gy = np.array([hx, hy]) - sh + off; got = []
        for L in layers:
            if not 2.0 <= L["z"] <= 5.0:                                       # 판의 가운데 높이 (아래 1.5mm 는 작은 구멍이 아직 없다)
                continue
            path, cur = [], None
            for s in L["seg"]:                                                 # 이어진 경로 -> 고리
                if cur is not None and abs(cur[-1][0] - s[0]) < 1e-3 and abs(cur[-1][1] - s[1]) < 1e-3:
                    cur.append((s[2], s[3]))
                else:
                    cur = [(s[0], s[1]), (s[2], s[3])]; path.append((cur, s[4]))
            for pts, w in path:
                p = np.asarray(pts)
                if len(p) < 6 or np.hypot(*(p[0] - p[-1])) > 0.6:
                    continue
                Aa = np.c_[2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))]; c = np.linalg.lstsq(Aa, (p ** 2).sum(1), rcond=None)[0]
                r = np.sqrt(c[2] + c[0] ** 2 + c[1] ** 2)
                if np.hypot(c[0] - gx, c[1] - gy) < 1.0 and r < dd / 2 + 1.5:
                    got.append((r, w, L["z"]))
        if not got:
            out.append(dict(name=lab, design=dd, gcode=None)); print("  %-18s 못 찾음" % lab); continue
        inner = {}
        for r, w, zz in got:                                                   # 층마다 가장 안쪽 고리
            if zz not in inner or r < inner[zz][0]:
                inner[zz] = (r, w)
        dg = [2 * (r - w / 2) for r, w in inner.values()]
        walls_n = round(len(got) / len(inner))
        out.append(dict(name=lab, design=dd, gcode=round(float(np.median(dg)), 2), walls=walls_n))
        print("  %-18s 설계 %.1f -> G코드 %.2f mm (벽 %d줄)" % (lab, dd, np.median(dg), walls_n))
    return out


def main():
    rows = boss_walls(); gh = gcode_holes()
    thin = [r["name"] for r in rows if r["wall_min"] < THIN]
    res = dict(holes=rows, n=len(rows), thin=thin, wall_min=min(r["wall_min"] for r in rows), wall_min_at=[r["name"] for r in rows if r["wall_min"] == min(x["wall_min"] for x in rows)],
               engage_min=min(r["engage"] for r in rows), gcode=gh)
    json.dump(res, open(os.path.join(paths.RES, "audit_holes.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("\n스스로 나사산을 내는 구멍 %d곳: 둘레 살이 %.1fmm 아래인 곳 %d (가장 얇은 곳 %.1fmm: %s), 물리는 길이 %.1fmm 이상"
          % (len(rows), THIN, len(thin), res["wall_min"], ", ".join(res["wall_min_at"]), res["engage_min"]))


if __name__ == "__main__":
    main()
