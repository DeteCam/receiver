"""전선이 겉에서 얼마나 보이는지 센다: 전선을 주황색으로 그려 넣고 사방(8방향 x 높이 2가지)에서 본 그림에서 주황색 화소를 센다.
    xvfb-run -a python3 cable_vis.py [그림 폴더]     ->  결과 폴더의 cable_vis.json
전선 한 가닥의 굵기는 그림에서 약 6화소라서, 화소 수 / 6 / (화소당 mm) 로 '보이는 길이'를 어림한다."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import render, paths
from cadlib import *
from p2 import *
from PIL import Image
from fig_change_view import HP_factory, tube

HP = HP_factory(Y_T, S_T, H_T)
CAB = "#FF00FF"                                 # 다른 데 안 쓰는 색 (세기 쉽게)
R_CAB = 2.2


def cables(yaw, pitch, lan=False):
    """전선의 길 (세계 좌표). 길이는 room_check.py 와 같은 길. 혼자 동작할 때는 3가닥(모터 1·2, 전원), lan=True 면 랜선까지 4가닥"""
    hp = lambda p: HP(p, yaw, pitch)
    rot = lambda p: np.array([np.cos(np.radians(yaw)) * p[0] - np.sin(np.radians(yaw)) * p[1], np.sin(np.radians(yaw)) * p[0] + np.cos(np.radians(yaw)) * p[1], p[2]])
    sx, sy = P.m1_shaft(); ux, uy = M1_BODY_DIR
    m1c = np.array([sx + ux * (M_OFFSET + 17.0), sy + uy * (M_OFFSET + 17.0), DECK_Z1 + 8.0])
    cx_, cz_ = P.m2_cap_xz(); m2c = np.array([cx_, ARM_Y1 + 8.0, cz_])
    x0, x1, z0, z1 = P.M2_SLOT; slot = np.array([(x0 + x1) / 2, CHEEK_Y - 4.0, (z0 + z1) / 2]); R = BASE["S"]["R"]
    out = {
        "모터 1 전선": [rot(m1c), rot(m1c + [0, 0, 16]), hp((DRV_X0 + 17.5, -12, HZ0 - 4)), hp((DRV_X0 + 17.5, -8, DRV_ZC[2] - 10))],
        "모터 2 전선": [rot(m2c), rot((cx_, ARM_Y0 - 3.0, cz_)), rot(slot), rot((-40.0, 30.0, 60.0)), hp((DRV_X0 + 24, -14, HZ0 - 4)), hp((DRV_X0 + 24, -8, DRV_ZC[1] - 10))],
        "전원선": [(0.0, -R + BATT_IN - 8.0, 10.0), (20.0, -R + BATT_IN + 6.0, 24.0), (3.0, 0, 26.0), (3.0, 0, TUBE_Z1 + 6), hp((BX0 - 6.6, -4.2, HZ0 - 6)), hp((BX0 - 6.6, -4.2, BZ1 - 14))],
    }
    if lan:
        out["랜선(받침 안쪽부터)"] = [(-R + 4.0, 0.0, 25.0), (-50.0, 24.0, 22.0), (-4.0, 0, 26.0), (-4.0, 0, TUBE_Z1 + 6), hp((BX0 + 10.25, -8.0, BZ0 - 34.0)), hp((BX0 + 10.25, -8.0, BZ0 - 20.0))]
    return out


def count(A, az, el, size=700, dist=700.0, focus=(0, 0, 110), save=None):
    f = save or os.path.join(paths.TMP, "vis.png")
    render.render(A, f, cam_az=az, cam_el=el, dist=dist, focus=focus, size=(size, size), ground=False, ssao=False)
    a = np.asarray(Image.open(f).convert("RGB")).astype(int)
    m = (a[:, :, 0] > 150) & (a[:, :, 2] > 150) & (a[:, :, 1] < 110)
    return int(m.sum())


def measure(lan, outdir=None, size=700, dist=700.0, fov=28.0):
    mm_per_px = 2 * dist * np.tan(np.radians(fov / 2)) / size
    res = {}
    for pitch in (0.0, TILT3, 85.0):
        A = P.assembly(0.0, pitch, 90.0, lan=lan)
        for k, pts in cables(0.0, pitch, lan).items():
            A["w_" + k] = (tube(pts, R_CAB), "X", CAB, False)
        row = {}
        for el in (15.0, 40.0):
            for az in range(0, 360, 45):
                n = count(A, az, el, size, dist, save=(os.path.join(outdir, "%sp%02d_e%02d_a%03d.png" % ("lan_" if lan else "", pitch, el, az)) if outdir else None))
                row["%d/%d" % (az, el)] = round(n * mm_per_px / (2 * R_CAB / mm_per_px), 0)     # 보이는 길이 mm (어림)
        res["%.0f" % pitch] = row
        print("%s 끄덕임 %4.1f도: 보이는 전선 길이 mm (방향/높이)  " % ("랜선도 꽂았을 때" if lan else "혼자 동작할 때  ", pitch) + "  ".join("%s=%d" % (k, v) for k, v in row.items()), flush=True)
    return res


SIDES = {"앞": (315, 0, 45), "옆": (90, 270), "뒤": (135, 180, 225)}


def mean_of(row, azs=None):
    v = [x for k, x in row.items() if azs is None or int(k.split("/")[0]) in azs]
    return float(np.mean(v))


if __name__ == "__main__":
    outdir = sys.argv[1] if len(sys.argv) > 1 else None
    path = os.path.join(paths.RES, "cable_vis.json")
    old = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    res = dict(before=old.get("before", {}), after=measure(False, outdir), after_lan=measure(True, outdir))     # before = 고치기 전 모델로 재 둔 값 (전선 4가닥)
    summ = {}
    for k in res["after"]:
        b = res["before"].get(k)
        summ[k] = dict(before_mean=mean_of(b) if b else None, after_mean=mean_of(res["after"][k]), after_lan_mean=mean_of(res["after_lan"][k]))
        for name, azs in SIDES.items():
            summ[k][name] = [mean_of(b, azs) if b else None, mean_of(res["after"][k], azs), mean_of(res["after_lan"][k], azs)]
    res["summary"] = summ
    json.dump(res, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    for k, v in summ.items():
        print("끄덕임 %s도: 고치기 전 %s -> 혼자 동작 %.0f mm (랜선을 꽂으면 %.0f mm) | 앞 %s 옆 %s 뒤 %s" % (k, "%.0f" % v["before_mean"] if v["before_mean"] else "-", v["after_mean"], v["after_lan_mean"],
              [round(x) if x is not None else None for x in v["앞"]], [round(x) if x is not None else None for x in v["옆"]], [round(x) if x is not None else None for x in v["뒤"]]))
