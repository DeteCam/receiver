"""무게와 힘 계산: 회전부 균형, 끄덕임 관절에 걸리는 중력 모멘트, 받침 회전 마찰."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
from p2 import *
RHO = 1.24e-3                      # PLA g/mm3
import paths
OUT = paths.RES + '/'

def props(m):
    mesh = m.to_mesh(); V = np.asarray(mesh.vert_properties)[:, :3].astype(float); T = np.asarray(mesh.tri_verts)
    a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]
    v6 = np.einsum('ij,ij->i', a, np.cross(b, c)); vol = v6.sum() / 6
    return vol, ((a + b + c) / 4 * v6[:, None]).sum(0) / 6 / vol

def report(fill=None, quiet=False):
    fill = fill or {}
    # ---- 회전부 균형 (머리 좌표, 회전축 = Z). 막대는 속을 꽉 채운다.
    out = {}
    for length in (150.0, 200.0):
        items = [("막대+균형추", P.rotor_bar(), 1.0), ("판", P.blade(length), 1.0)]
        mx = 0; mass = 0
        for name, m, f in items:
            v, c = props(m); g = v * RHO * f; mx += g * c[0]; mass += g
        foil_g = 100 * length * 0.08 * 2.0e-3            # 알루미늄 테이프: 두께 0.08mm, 접착제 포함 2.0 g/cm3 로 어림
        mx += foil_g * R_FOIL; mass += foil_g
        screw = 0.6; mx += screw * (COLLAR_X0 + 3.6); mass += screw      # 판 고정 나사 M3x8
        tv, tc = props(P.trim_plate()); tg = tv * RHO
        if length > 150:
            bv, bc = props(P.balance_200()); bg = bv * RHO; mx += bg * bc[0] + 1.0 * (-POD_R1 - 6); mass += bg + 1.0     # 덩어리 + M3x16 나사
        out[length] = dict(mass=mass, imbalance_gmm=mx, trim_g=tg, trim_gmm=tg * abs(tc[0]), foil_g=foil_g)
        if not quiet:
            print("판 %d: 회전부 %.1f g, 불균형 %+.0f g·mm (판 쪽이 +) = %.2f mN·m, 균형 조각 1장 = %.1f g, %.0f g·mm" % (
                length, mass, mx, abs(mx) * 9.81e-3, tg, tg * abs(tc[0])))
    # ---- 끄덕임 관절: 머리 좌표의 점 (y, z) 를 끄덕임 축에서의 거리로
    H = []
    def add(name, g, c):
        H.append((name, g, c[1] - Y_T, c[2] - S_T))
    for name, fn, f in (("머리 몸통", P.head_shell, 1.0), ("머리 뚜껑", P.head_lid, 1.0), ("모터 자리", P.crown, 0.9), ("부채꼴 기어", P.sector_head, 0.8),
                        ("막대+균형추", P.rotor_bar, 1.0), ("판 150", lambda: P.blade(150.0), 1.0)):
        v, c = props(fn()); add(name, v * RHO * f, c)
    add("알루미늄 테이프", out[150.0]["foil_g"], (80, 0, 17))
    add("모터 3", 34.0, (-8, 0, 19.5)); add("파이 3B+", 50.0, (-14, -4, -37.5)); add("드라이버 3장", 24.0, (32, 0, -48))
    add("점퍼선·모터선", 22.0, (20, -12, -45)); add("전원 플러그와 선", 12.0, (-48, -4, -50)); add("나사", 8.0, (0, -10, -20))
    add("버튼 모듈과 선", 5.0, (BTN_X, BTN_PCB_Y, BTN_Z - 8))
    add("무선 동글", 4.0, (BX0 + DONGLE["x"], -PI_T / 2 - DONGLE["up"], BZ0 - 2.0 - (DONGLE["L"] - DONGLE["plug"]) / 2))
    add("탐지 동글", 4.0, (BX0 + DONGLE2["x"], -PI_T / 2 - DONGLE2["up"], BZ0 - 2.0 - (DONGLE2["L"] - DONGLE2["plug"]) / 2))
    def calc(H):
        mass = sum(h[1] for h in H); my = sum(h[1] * h[2] for h in H) / mass; mz = sum(h[1] * h[3] for h in H) / mass
        # 끄덕임 각 p 에서 중력 모멘트: 무게중심 (요크 x, z) = 회전( -(dy), dz ).  mN·m (앞으로 넘어가려는 쪽이 +)
        moment = lambda p: mass * 9.81e-3 * (-my * np.cos(np.radians(p)) + mz * np.sin(np.radians(p)))
        return mass, my, mz, {p: moment(p) for p in (0.0, TILT3, 85.0)}, max(abs(moment(p)) for p in np.arange(PITCH_MIN, PITCH_MAX + 0.1, 1.0))
    mass, my, mz, mom, worst = calc(H)                                     # 혼자 동작할 때의 모습: 무선 동글, 랜선 없음
    if not quiet:
        for h in H:
            print("   %-12s %6.1f g   축에서 앞뒤 %+6.1f, 위 %+6.1f" % h)
    Hl = list(H); H = Hl; add("랜선 플러그와 선", 14.0, (BX0 + 10.25, -8, BZ0 - 15))      # 개발할 때: 랜선도 꽂았다 (축 아래에 달려 오히려 머리를 세우는 쪽으로 돕는다)
    lan = calc(Hl)
    ratio = SEC_T / ZP
    cap = 30.0 * ratio * 0.9                                             # 모터 30 mN·m, 기어 효율 0.9
    if not quiet:
        print("머리 %.0f g, 무게중심: 축에서 위로 %.1f mm, 뒤로 %.1f mm" % (mass, mz, my))
        print("(랜선을 꽂으면 머리 %.0f g, 깊이 숙임 %.0f mN·m, 가장 큰 값 %.0f)" % (lan[0], lan[3][85.0], lan[4]))
        print("중력 모멘트 mN·m: 세움 %.0f, 기울임 %.0f, 깊이 숙임(85도) %.0f, 가장 큰 값 %.0f" % (mom[0.0], mom[TILT3], mom[85.0], worst))
        print("기어 %.2f:1, 모터가 낼 수 있는 힘 %.0f mN·m -> 기울임 %.0f%%, 깊이 숙임 %.0f%%. 전원 없이 버티는 힘 %.0f 이상" % (
            ratio, cap, abs(mom[TILT3]) / cap * 100, abs(mom[85.0]) / cap * 100, 59 * ratio))
    # ---- 받침 회전: 위에 얹힌 무게와 마찰
    yv, yc = props(P.yoke()); yoke_g = yv * RHO * 0.6
    turn_g = mass + yoke_g + 34 * 2 + 10
    res = dict(rotor=out, head_g=mass, com_up=mz, com_back=my, moment=mom, moment_max=worst, ratio=ratio, cap=cap, yoke_g=yoke_g, turn_g=turn_g, lan=dict(head_g=lan[0], moment=lan[3], moment_max=lan[4]))
    for mu, label in ((0.15, "양초를 문질렀을 때"), (0.30, "그냥"), (0.02, "비비탄을 넣었을 때")):
        r = BB_R if mu < 0.05 else (RING_R0 + RING_R1) / 2
        t = turn_g * 9.81e-3 * mu * r
        res["yaw_%s" % mu] = t
        if not quiet:
            print("받침 회전 마찰 (%s, 마찰계수 %.2f): %.0f mN·m = 모터 힘(30 x 4 x 0.9 = 108)의 %.0f%%" % (label, mu, t, t / 108 * 100))
    return res

if __name__ == "__main__":
    r = report()
    json.dump(r, open(OUT + "mass.json", "w"), ensure_ascii=False, indent=1, default=float)
