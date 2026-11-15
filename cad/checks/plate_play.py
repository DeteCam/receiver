"""알루미늄 판이 판 틀 속에서 어느 쪽으로 얼마나 움직일 수 있는지 (판을 단단한 한 덩어리로 보고 조금씩 옮기거나 기울여, 틀·누름쇠에 처음 닿는 곳을 찾는다).
좌표는 회전부 좌표: 판 = X 79.5~80.5 (뒷면이 홈 바닥), Y ±50, Z 아래 끝~위 끝(150). +X = 판 앞(바깥), +Z = 위(누름쇠 쪽).
  옆으로(±Y), 길이 방향(±Z): 홈 벽과 홈 끝까지의 틈
  앞으로 들림: 아래 변(위 뒤 모서리를 축으로), 옆 변 하나(반대쪽 뒤 모서리를 축으로), 판 전체
결과: 결과 폴더의 plate_play.json. 1분쯤."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import numpy as np
import paths
import parts2 as P
from p2 import FL_X1, AL_T, BL_Z0

plate = P.al_plate(150.0); FRAME = P.plate_frame(); CLAMP = P.plate_clamp()
ZT, ZB = BL_Z0 + 150.0, BL_Z0


def contact(f, ref, tol=0.02, hi=2.0):
    """f(s) 로 옮긴 판이 ref 에 새로 닿는 가장 작은 s (2mm 까지, 0.005 정밀도). 닿지 않으면 None"""
    if (f(hi) ^ ref).volume() <= tol:
        return None
    lo = 0.0
    for _ in range(12):
        mid = (lo + hi) / 2
        if (f(mid) ^ ref).volume() > tol:
            hi = mid
        else:
            lo = mid
    return round(hi, 2)


def rot_about(m, pivot, axis, ang):
    u = np.asarray(axis, float); u /= np.linalg.norm(u)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]); R = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K
    p = np.asarray(pivot, float); return m.transform(np.column_stack([R, p - R @ p])), R


def lifter(pivot, axis, far):
    """far 점이 앞(+X)으로 s mm 나오게 돌리는 함수"""
    pivot = np.asarray(pivot, float); far = np.asarray(far, float); u = np.asarray(axis, float) / np.linalg.norm(axis)
    Lr = np.linalg.norm(np.cross(u, far - pivot))
    def f(s):
        for sg in (1, -1):
            m, R = rot_about(plate, pivot, axis, sg * np.arcsin(min(1.0, s / Lr)))
            if (R @ (far - pivot) + pivot)[0] > far[0]:
                return m
        raise RuntimeError("돌리는 방향을 못 찾음")
    return f


def main():
    res = {}
    for key, lab, v in (("y", "옆으로", (0, 1, 0)), ("z_dn", "길이 방향 아래 (아래 턱 쪽)", (0, 0, -1)), ("z_up", "길이 방향 위 (누름쇠 쪽)", (0, 0, 1))):
        c = [contact(lambda s, sg=sg: plate.translate(list(sg * s * np.asarray(v, float))), FRAME) for sg in (1, -1)] if key == "y" else [contact(lambda s: plate.translate(list(s * np.asarray(v, float))), FRAME)]
        res[key] = min(x for x in c if x is not None)
        print("%-30s 판 틀에 닿기까지 %.2f mm" % (lab, res[key]))
    cases = (("lift_bottom", "아래 변이 앞으로 들림", (FL_X1, 0, ZT), (0, 1, 0), (FL_X1 + AL_T, 0, ZB)),
             ("lift_side", "옆 변 하나가 앞으로 들림", (FL_X1, -50.0, 0), (0, 0, 1), (FL_X1 + AL_T, 50.0, ZB + 75)))
    for key, lab, pv, ax, far in cases:
        f = lifter(pv, ax, far); res[key] = dict(frame=contact(f, FRAME), clamp=contact(f, CLAMP))
        print("%-30s 판 틀(아래 턱)에 %s, 누름쇠에 %s" % (lab, "%.2f mm" % res[key]["frame"] if res[key]["frame"] is not None else "-", "%.2f mm" % res[key]["clamp"] if res[key]["clamp"] is not None else "-"))
    res["forward"] = dict(clamp=contact(lambda s: plate.translate([s, 0, 0]), CLAMP))
    print("%-30s 누름쇠에 %.2f mm" % ("판 전체가 앞으로", res["forward"]["clamp"]))
    json.dump(res, open(os.path.join(paths.RES, "plate_play.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
