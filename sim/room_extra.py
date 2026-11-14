"""room.py 에 덧붙이는 실험: 어떻게 하면 좁은 방에서 덜 틀리는지 조건을 바꿔 가며 본다.  python3 room_extra.py [횟수]   결과: room_extra.json"""
import paths                                   # 계산 코드(core)를 찾는다
import sys, os, json, time
import numpy as np
import room
from room import RoomWorld, summ
from fusion import Fusion, consistency
import montecarlo as mc

HERE = os.path.dirname(os.path.abspath(__file__))
T = mc.TILT3
PLANS = {"세 번 (tilt3)": [(-120.0, T), (0.0, T), (120.0, T)],
         "네 번 (tilt4)": [(-135.0, 60.0), (-45.0, 60.0), (45.0, 60.0), (135.0, 60.0)],
         "세워서 1 + 세 번": [(-120.0, 0.0), (-120.0, T), (0.0, T), (120.0, T)],
         "여섯 번 (60도씩)": [(-150.0, T), (-90.0, T), (-30.0, T), (30.0, T), (90.0, T), (150.0, T)]}


def run(n, rm="화장실", shape="W", frames=PLANS["세 번 (tilt3)"], seed=11, height=None, wall=None, person=None, keep=None):
    """height = 수신기를 놓는 높이(m), wall = 가장 가까운 벽까지 거리 범위(m), keep = 카메라 거리 범위(m)"""
    fus = Fusion(); rows = []; i = 0
    while len(rows) < n and i < 20 * n:
        rng = np.random.default_rng(seed * 100003 + i); i += 1
        base = None
        if height is not None or wall is not None:
            Lx, Ly, _ = room.ROOMS[rm]["L"]
            for _ in range(300):
                b = np.array([rng.uniform(0.3, Lx - 0.3), rng.uniform(0.3, Ly - 0.3), 0.0 if height is None else height])
                d = min(b[0], Lx - b[0], b[1], Ly - b[1])
                if wall is None or wall[0] <= d <= wall[1]:
                    base = b; break
        w = RoomWorld(rng, rm, shape, base=base)
        if keep is not None and not (keep[0] <= w.dist < keep[1]):
            continue
        if person:
            w.person = room.person_spot(w, rng, person)
        curves = [w.scan(y, t) for y, t in frames]
        est, obs = fus.solve(curves, frames)
        if est is None:
            rows.append((180.0, 0, 0, w.kfac, w.block_db, w.dist, 0.0)); continue
        ok, _ = consistency(est, obs, frames)
        rows.append((float(np.degrees(np.arccos(np.clip(est @ w.c, -1, 1)))), sum(o["seen"] for o in obs), int(ok), w.kfac, w.block_db, w.dist, 0.0))
    return np.array(rows)


def line(tag, R):
    s = summ(R)
    print("%-40s 가운데값 %5.1f도  10도 안 %4.1f%%  30도 넘음 %4.1f%% | 경고 %4.1f%%  경고 없이 틀림 %4.1f%% | K %4.1fdB 거리 %.2fm" % (
        tag, s["med"], s["within10"], s["over30"], s["flagged"] + s["none"], s["bad_unflagged"], s["kfac"], s["dist"]), flush=True)
    return s


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
    out = {}
    for shape in ("W", "V"):
        print("===== 화장실, 골 모양 %s" % shape)
        for name, fr in PLANS.items():
            out["화장실|%s|%s" % (shape, name)] = line("재는 횟수: " + name, run(n, "화장실", shape, fr))
        for h, lab in ((0.0, "바닥"), (0.45, "변기 뚜껑 (45cm)"), (0.85, "세면대 (85cm)"), (1.2, "선반 (1.2m)")):
            out["화장실|%s|높이 %s" % (shape, lab)] = line("놓는 높이: " + lab, run(n, "화장실", shape, height=h))
        for wr, lab in (((0.30, 0.42), "벽에서 30~42cm"), ((0.55, 0.80), "벽에서 55~80cm (가운데)")):
            out["화장실|%s|%s" % (shape, lab)] = line("놓는 자리: " + lab, run(n, "화장실", shape, wall=wr))
        for kr, lab in (((0.45, 1.0), "1m 안쪽"), ((1.0, 1.6), "1~1.6m"), ((1.6, 9.0), "1.6m 넘음")):
            out["화장실|%s|카메라 %s" % (shape, lab)] = line("카메라까지: " + lab, run(n, "화장실", shape, keep=kr))
    # 거울과 문이 없는 화장실, 조금 큰 화장실
    room.ROOMS["화장실 (거울 없음)"] = dict(room.ROOMS["화장실"], patches=[])
    room.ROOMS["큰 화장실 2.0x2.6"] = dict(room.ROOMS["화장실"], L=(2.0, 2.6, 2.3))
    room.ROOMS["타일 대신 석고보드"] = dict(room.ROOMS["화장실"], wall="plaster", floor="concrete", ceil="plaster", patches=[])
    for rm in ("화장실 (거울 없음)", "큰 화장실 2.0x2.6", "타일 대신 석고보드"):
        for shape in ("W", "V"):
            out["%s|%s" % (rm, shape)] = line("%s · %s" % (rm, shape), run(n, rm, shape))
    # 사람이 선에서 조금 비켜 설 때 (보통 방): 가림이 얼마일 때부터 틀리나
    for shape in ("W", "V"):
        R = run(2 * n, "보통 방", shape, person="block")
        for lo, hi in ((0, 3), (3, 6), (6, 10), (10, 15), (15, 99)):
            m = (R[:, 4] >= lo) & (R[:, 4] < hi)
            if m.sum() >= 20:
                out["보통 방|%s|가림 %d~%ddB" % (shape, lo, hi)] = line("보통 방 %s, 직접파가 %d~%ddB 줄었을 때 (n=%d)" % (shape, lo, hi, m.sum()), R[m])
    json.dump(dict(n=n, summary=out), open(os.path.join(HERE, "room_extra.json"), "w"), ensure_ascii=False, indent=1)
