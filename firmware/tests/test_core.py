#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""계산 코드 검사: 적어 둔 패킷(fixture)으로 방향·구간·판별을 다시 계산해, 만든 컴퓨터의 답과 같은지 본다.
파이(파이썬 3.7, numpy 1.16)에서도 그대로 돈다:   python3 tests/test_core.py      (파이에서는  cd ~/detecam && python3 test_core.py)
장치가 실제로 재는 계산(analyze3 -> locate)이 이 컴퓨터에서 제대로 도는지 보는 검사다."""
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
for d in (".", "../pi", "../core", "../host"):
    p = os.path.normpath(os.path.join(HERE, d))
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)
import analyze3 as A, locate as K, identify as I, detect as D
from estimator import unit


def main():
    J = json.load(open(os.path.join(HERE, "fixture.json"), encoding="utf-8")); z = np.load(os.path.join(HERE, "fixture.npz"))
    S = K.Session(); S.set_room(*J["room"]); worst = 0.0
    for i, sc in enumerate(J["scans"]):
        res, curves, extra = A.solve(sc["recs"], z["t%d" % i], z["C%d" % i].astype(float), quiet=True)
        obs, usable = extra; e = sc["expect"]
        d = float(np.degrees(np.arccos(np.clip(np.dot(unit(res["az"], res["el"]), unit(e["az"], e["el"])), -1, 1)))); worst = max(worst, d)
        assert res["n_seen"] == e["n_seen"] and d < 0.5, ("자리 %d" % (i + 1), res["az"], res["el"], e)
        frames = [(float(sc["recs"][k]["yaw"]), float(sc["recs"][k]["pitch"])) for k in usable]
        S.add((tuple(J["poses"][i][0]), J["poses"][i][1]), [dict(seen=bool(q["seen"]), psi=float(q["psi"]), depth=float(q["depth"])) for q in obs], frames, [float(v) for v in unit(res["az"], res["el"])])
        t = float(np.degrees(np.arccos(np.clip(np.dot(unit(res["az"], res["el"]), unit(*sc["truth"])), -1, 1))))
        print("자리 %d: 방위 %+.1f도, 높이 %+.1f도 (적어 둔 답과 %.3f도 차이, 참 방향과 %.1f도)" % (i + 1, res["az"], res["el"], d, t))
    r = S.result(); e = J["zone"]; dc = float(np.linalg.norm(np.array(r["centre"]) - np.array(e["centre"])))
    assert dc < 0.03 and abs(r["length_m"] - e["length_m"]) < 0.05 and r["verdict"] == e["verdict"], (r["centre"], r["length_m"], e)
    print("구간: 가운데 %s, 길이 %.2f m (적어 둔 답과 %.3f m 차이, 참 위치와 %.2f m)" % (r["centre"], r["length_m"], dc, float(np.linalg.norm(np.array(r["centre"]) - np.array(J["cam"])))))
    stim = I.plan_stimulus("walk")
    for c in J["identify"]:
        d = I.decide(np.array(c["counts"]), stim)
        assert d["verdict"] == c["expect"]["verdict"] and abs(d["ratio"] - c["expect"]["ratio"]) < 0.011 and abs(d["p"] - c["expect"]["p"]) < 0.002, (d, c["expect"])
        print("판별 (배율 %.1f): %s" % (c["gain"], d["verdict"]))
    # 탐지: 적어 둔 모니터 모드 파일을 읽어 같은 표가 나오는지
    M = json.load(open(os.path.join(HERE, "fixture_mon.json"), encoding="utf-8")); fr = D.read_frames(os.path.join(HERE, "fixture_mon.pcap"))
    rows = D.analyse(fr, [tuple(h) for h in M["hops"]]); assert rows == M["rows"], "탐지 표가 적어 둔 것과 다르다"
    tt, nn = D.sent_by(fr, M["cam"]); assert len(tt) == M["cam_frames"] and int(sum(nn)) == M["cam_bytes"]
    print("탐지: 프레임 %d개에서 기기 %d개, 후보 %s" % (len(fr), len(rows), [r["mac"] for r in rows if r["candidate"]]))
    print("계산 검사 통과: 파이썬 %s, numpy %s (방향 차이 최대 %.3f도)" % (sys.version.split()[0], np.__version__, worst))


if __name__ == "__main__":
    main()
