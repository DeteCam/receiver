"""USB 동글 두 개(폰과 통신하는 것, 탐지용)의 틈: 서로, 요크와, 받침과. 끄덕임 전 범위에서 잰다.   python3 dongle_gap.py
탐지 동글은 모델(34 x 16 x 8)과 들어가는 가장 큰 크기(45 x 18 x 8) 두 가지로 본다."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import parts as v1
import paths
from cadlib import *
from p2 import *
mb, mc = v1.motor_body_dummy()
YOKE = union([P.yoke(), P.pin(1), P.pin(-1), P.pin_keeper(1), P.pin_keeper(-1), mb.transform(P.motor1_T()), mc.transform(P.motor1_T()), mb.transform(P.motor2_T()), mc.transform(P.motor2_T()), P.pinion1(), P.pinion2(), P.m2_ring()])
BASE = {s: union([P.tub(s), P.cover(s), P.battery_dummy(s), P.usb_a_dummy(s)]) for s in ("S", "L")}
pitches = list(np.arange(PITCH_MIN, PITCH_MAX + 0.01, 2.5)) + [TILT3]
out = {}
def run(tag, d):
    D = P.dongle_dummy(d); wy = (1e9, 0); wb = (1e9, 0)
    for p in pitches:
        H = P.pose_head(D, 0, p)
        v = (H ^ YOKE).volume(); g = 0.0 if v > 0.5 else H.min_gap(YOKE, 60.0)
        if g < wy[0]: wy = (g, p)
        for s in ("S", "L"):
            v = (H ^ BASE[s]).volume(); g = 0.0 if v > 0.5 else H.min_gap(BASE[s], 80.0)
            if g < wb[0]: wb = (g, p)
    out[tag] = dict(yoke=round(wy[0], 1), yoke_at=float(wy[1]), base=round(wb[0], 1), base_at=float(wb[1]))
    print(tag, out[tag])
run("무선 동글 34x16x8", DONGLE)
run("탐지 동글 34x16x8", DONGLE2)
big = dict(DONGLE2, L=45.0, W=18.0)
run("탐지 동글 45x18x8", big)
g1 = P.dongle_dummy(DONGLE).min_gap(P.dongle_dummy(DONGLE2), 30.0); g2 = P.dongle_dummy(DONGLE).min_gap(P.dongle_dummy(big), 30.0)
out["동글 사이"] = dict(small=round(g1, 1), big=round(g2, 1)); print("동글 사이", out["동글 사이"])
# 머리 치마 밖으로 나오는 길이
z_tip = BZ0 - 2.0 - (DONGLE["L"] - DONGLE["plug"]); print("동글 끝 z = %.1f, 머리 치마 아래끝 HZ0 = %.1f -> 밖으로 %.1f mm" % (z_tip, HZ0, HZ0 - z_tip))
out["skirt_out"] = round(HZ0 - z_tip, 1)
# 무선 동글과 CSI 안테나 사이의 거리 (머리 좌표: 안테나 = 원점)
D = P.dongle_dummy(DONGLE); bb = D.bounding_box(); c = [(bb[0] + bb[3]) / 2, (bb[1] + bb[4]) / 2, (bb[2] + bb[5]) / 2]
out["antenna_dist"] = round(float(np.linalg.norm(c)), 0); print("안테나에서 무선 동글 가운데까지 %.0f mm" % out["antenna_dist"])
json.dump(out, open(os.path.join(paths.RES, "dongle_gap.json"), "w"), ensure_ascii=False, indent=1)
