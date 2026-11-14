#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""test_core.py 가 쓰는 자료(fixture.npz, fixture.json)를 만든다. 노트북에서 (numpy 1.17 이상) 한 번 돌려 두면 된다.
가짜 방에서 세 자리를 잰 패킷과, 그것으로 이 컴퓨터가 계산한 답을 적어 둔다. 파이에서 test_core.py 를 돌리면 같은 답이 나오는지 본다."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in ("../pi", "../core", "../host"):
    p = os.path.normpath(os.path.join(HERE, d))
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import rx3, analyze3 as A, locate as K, identify as I
from estimator import azel, unit
from fusion import origin

CAM = np.array([2.4, 3.1, 0.9]); ROOM = ([0.0, 0.0, 0.0], [3.5, 4.5, 2.4])
POSES = [((1.0, 1.0, 0.0), 0.0), ((2.6, 1.2, 0.4), 90.0), ((0.9, 3.4, 0.0), -30.0)]


def main():
    arrays = {}; J = dict(cam=CAM.tolist(), room=ROOM, poses=[[list(b), h] for b, h in POSES], scans=[])
    for i, pose in enumerate(POSES):
        rig = rx3.Rig(dry=True); rig.state["pos"] = {"yaw": 0, "pitch": 0, "spin": 0}
        recs = [rig.scan(y, p) for y, p in rx3.PLANS["tilt3"]]
        fr = [(r["yaw"], r["pitch"]) for r in recs]
        v = K.to_dev(CAM[None, :], pose)[0] - origin(fr); az, el = azel(v / np.linalg.norm(v))
        t, C = A.fake_packets(recs, (float(az), float(el)), seed=40 + i, nsc=16)
        arrays["t%d" % i] = t.astype(np.float64); arrays["C%d" % i] = C.astype(np.float16)
        J["scans"].append(dict(recs=recs, truth=[float(az), float(el)]))
    np.savez_compressed(os.path.join(HERE, "fixture.npz"), **arrays)
    # 저장한 것을 다시 읽어 답을 계산한다 (반올림된 값으로 계산해야 다른 컴퓨터와 똑같이 견줄 수 있다)
    z = np.load(os.path.join(HERE, "fixture.npz")); S = K.Session(); S.set_room(*ROOM)
    for i, pose in enumerate(POSES):
        sc = J["scans"][i]; res, curves, extra = A.solve(sc["recs"], z["t%d" % i], z["C%d" % i].astype(float), quiet=True)
        obs, usable = extra; frames = [(float(sc["recs"][k]["yaw"]), float(sc["recs"][k]["pitch"])) for k in usable]
        est = [float(v) for v in unit(res["az"], res["el"])]
        o = [dict(seen=bool(q["seen"]), psi=float(q["psi"]), depth=float(q["depth"])) for q in obs]
        S.add(pose, o, frames, est)
        sc["expect"] = dict(az=res["az"], el=res["el"], n_seen=res["n_seen"], psi=[q["psi"] for q in o], depth=[q["depth"] for q in o])
        e = np.degrees(np.arccos(np.clip(np.dot(unit(res["az"], res["el"]), unit(*sc["truth"])), -1, 1)))
        print("자리 %d: 참 방향 (%+.1f, %+.1f), 계산 (%+.1f, %+.1f), 오차 %.1f도, 패킷 %d개" % (i + 1, sc["truth"][0], sc["truth"][1], res["az"], res["el"], e, len(z["t%d" % i])))
    r = S.result()
    J["zone"] = dict(centre=r["centre"], length_m=r["length_m"], box=r["box"], verdict=r["verdict"], confidence=r["confidence"])
    print("구간: 가운데 %s, 길이 %.2f m, 참값과 %.2f m" % (r["centre"], r["length_m"], float(np.linalg.norm(np.array(r["centre"]) - CAM))))
    # 판별: 정해 둔 숫자열
    rs = np.random.RandomState(3); stim = I.plan_stimulus("walk"); n = len(stim); s = np.concatenate([np.zeros(2), stim])[:n]
    J["identify"] = []
    for g in (1.0, 1.5):
        x = rs.poisson(65.0 * I.DT * (1 + (g - 1) * s) * np.exp(rs.normal(0, 0.12, n))).astype(float); d = I.decide(x, stim)
        J["identify"].append(dict(gain=g, counts=[float(v) for v in x], expect=dict((k, d[k]) for k in ("camera", "verdict", "ratio", "p", "up", "cycles"))))
        print("판별 (배율 %.1f): %s, 배율 %.2f, p %.3f" % (g, d["verdict"], d["ratio"], d["p"]))
    json.dump(J, open(os.path.join(HERE, "fixture.json"), "w", encoding="utf-8"), ensure_ascii=False)
    print("저장: fixture.npz (%d KB), fixture.json (%d KB)" % (os.path.getsize(os.path.join(HERE, "fixture.npz")) // 1024, os.path.getsize(os.path.join(HERE, "fixture.json")) // 1024))


if __name__ == "__main__":
    main()
