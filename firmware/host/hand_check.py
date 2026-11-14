#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""손으로 잰 스캔(tools/manual_scan.py 가 만든 폴더)에 자동 스캔과 같은 계산을 돌려 본다.

  .venv/bin/python firmware/host/hand_check.py data/scan_20261111_225427 [폴더 ...]

자동 스캔 코드의 두 부분, 곧 패킷에서 세기를 뽑는 계산(analyze3.packet_levels)과 골 가운데를 찾는 계산(sim/fusion.trough)이
실제 데이터에서 제대로 도는지 보는 용도다. 손 스캔은 각도가 듬성듬성해서, 잰 각도 사이를 직선으로 이어 5도 칸으로 만든 뒤 넣는다.
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (".", "../core", "../pi"):                 # 저장소에서는 계산 코드가 core 에, 파이 코드가 pi 에 있다
    _p = os.path.normpath(os.path.join(HERE, _d))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import analyze3 as A
from fusion import trough


def check(outdir):
    meta = json.load(open(os.path.join(outdir, "meta.json"))) if os.path.exists(os.path.join(outdir, "meta.json")) else {}
    rows = list(csv.DictReader(open(os.path.join(outdir, "angles.csv"))))
    t, C, mac = A.packet_levels(os.path.join(outdir, "csi.pcap"), meta.get("mac"))
    span = t[-1] - t[0]
    g = {}
    for r in rows:
        if r["angle"] == "base":
            continue
        m = (t >= float(r["t0"])) & (t <= float(r["t1"]))
        if m.sum():
            g.setdefault(float(r["angle"]), []).append(C[m])
    if len(g) < 5:
        print("%s: 각도가 %d개뿐이라 건너뜀" % (outdir, len(g))); return None
    ang = np.array(sorted(g)); V = np.stack([np.median(np.vstack(g[a]), axis=0) for a in ang])
    db = V - np.median(V, axis=0)
    contrast = np.median(db, axis=0) - db.min(axis=0)
    sel = np.argsort(-contrast)[: max(5, db.shape[1] // 3)]
    curve = np.median(db[:, sel], axis=1)
    grid = np.arange(2.5, 360.0, 5.0)
    y = np.interp(grid, np.r_[ang - 360, ang, ang + 360], np.r_[curve, curve, curve])
    psi, depth, noise = trough(grid, y)
    truth = meta.get("truth")
    print("%s" % outdir)
    print("   패킷 %d개 (초당 %.0f개), 서브캐리어 %d개, MAC %s" % (len(t), len(t) / max(span, 1e-6), C.shape[1], mac))
    print("   잰 각도 %d개: %s" % (len(ang), " ".join("%g" % a for a in ang)))
    print("   골 가운데 %.1f도, 깊이 %.1f dB (가장 낮은 점은 %g도)%s" % (
        psi % 360, depth, ang[int(np.argmin(curve))], "" if truth is None else "   실제 %s도 -> 오차 %.1f도" % (truth, abs((psi - float(truth) + 180) % 360 - 180))))
    return psi % 360, depth


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for d in sys.argv[1:]:
        check(d)
