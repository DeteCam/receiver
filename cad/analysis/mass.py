"""무게와 힘 계산: 회전부 균형, 끄덕임 관절에 걸리는 중력 모멘트, 받침 회전 마찰."""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
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

BOLT_G, NUT_G = 0.70, 0.35          # M3 x 8 둥근머리 볼트, M3 너트 (강철)
CSK_G = 0.40                        # M3 x 6 접시머리


def rotor_items(length=150.0, t=AL_T, n_coin=8):
    """회전부를 이루는 것들: (이름, 무게 g, 무게중심 (x, y, z) 머리 좌표). 회전축 = Z, 판은 +X 쪽."""
    it = []
    v, c = props(P.rotor_bar()); it.append(("회전 막대 (속을 꽉 채워 출력)", v * RHO, c))
    v, c = props(P.al_plate(length, t)); it.append(("알루미늄 판 100×%d×%gT" % (length, t), v * AL_RHO, c))
    v, c = props(P.plate_frame(length)); it.append(("판 틀", v * RHO, c))
    v, c = props(P.plate_clamp(length)); it.append(("판 누름쇠", v * RHO, c))
    for y, z in PLATE_BOLTS:
        it.append(("판 틀 나사 M3 x 6 접시머리·너트", CSK_G + NUT_G, np.array([FL_XH - 2.0, y, z])))
    for y, z in P.clamp_screws(length):
        it.append(("누름쇠 나사 M3 x 6 접시머리·너트", CSK_G + NUT_G, np.array([FL_X1, y, z])))
    it.append(("축 고정 나사·너트", BOLT_G + NUT_G, np.array([0.0, 6.0, M1_ZT + 7.0])))
    v, c = props(P.coin_cap(n_coin)); it.append(("균형추 마개", v * RHO, c))
    for s_ in (1, -1):
        it.append(("마개 나사", BOLT_G, np.array([POD_XC, s_ * CAP_EAR_Y, POD_Z0 - 2.0])))
    it.append(("100원 동전 %d개" % n_coin, COIN_G * n_coin, np.array([POD_XC, 0.0, POD_Z1 - COIN_T * n_coin / 2])))
    k = COIN_SLOTS - n_coin
    if k > 0:
        v, c = props(P.coin_spacer()); it.append(("채움 조각 %d개" % k, v * RHO * k, np.array([POD_XC, 0.0, POD_Z1 - COIN_T * n_coin - SPACER_T * k / 2])))
    return it


def rotor_sum(items):
    m = sum(g for _, g, c in items); mx = sum(g * c[0] for _, g, c in items); my = sum(g * c[1] for _, g, c in items); mz = sum(g * c[2] for _, g, c in items)
    Izz = sum(g * (c[0] ** 2 + c[1] ** 2) for _, g, c in items)                      # 점 질량으로 본 값 (아래에서 판과 막대의 퍼진 몫을 더한다)
    return m, mx, my, mz / m, Izz


def best_coins(length=150.0, t=AL_T):
    """불균형이 가장 작아지는 동전 수"""
    return min(range(0, COIN_SLOTS + 1), key=lambda n: abs(rotor_sum(rotor_items(length, t, n))[1]))


def spin_inertia(length=150.0, t=AL_T, n_coin=8):
    """회전축 둘레의 관성 모멘트 (g·mm2). 판과 막대는 모양에서, 나머지는 점 질량으로."""
    def Iz(m, rho):
        mesh = m.to_mesh(); V = np.asarray(mesh.vert_properties)[:, :3].astype(float); T = np.asarray(mesh.tri_verts)
        a, b, c = V[T[:, 0]], V[T[:, 1]], V[T[:, 2]]; v6 = np.einsum('ij,ij->i', a, np.cross(b, c))
        f = lambda u: (u[:, 0] ** 2 + u[:, 1] ** 2)
        # 사면체(원점, a, b, c)의 x2 + y2 적분: 부피/10 x (꼭짓점 제곱합 + 꼭짓점 쌍의 곱 합)
        s2 = sum(a[:, k] ** 2 + b[:, k] ** 2 + c[:, k] ** 2 + a[:, k] * b[:, k] + a[:, k] * c[:, k] + b[:, k] * c[:, k] for k in (0, 1))
        return float((v6 / 6 * s2 / 10).sum()) * rho
    I = Iz(P.rotor_bar(), RHO) + Iz(P.al_plate(length, t), AL_RHO) + Iz(P.coin_cap(n_coin), RHO) + Iz(P.plate_frame(length), RHO) + Iz(P.plate_clamp(length), RHO)
    I += COIN_G * n_coin * (POD_XC ** 2 + COIN_D ** 2 / 8)
    for name, g, c in rotor_items(length, t, n_coin):
        if "볼트" in name or "나사" in name or "채움" in name:
            I += g * (c[0] ** 2 + c[1] ** 2)
    return I


def report(fill=None, quiet=False):
    fill = fill or {}
    # ---- 회전부 균형 (머리 좌표, 회전축 = Z). 막대는 속을 꽉 채운다. 판 두께·길이마다 동전 수를 고른다.
    out = {}
    for length, t in ((150.0, AL_T),):
        n = best_coins(length, t); items = rotor_items(length, t, n); mass, mx, my, zc, _ = rotor_sum(items)
        m1 = rotor_sum(rotor_items(length, t, min(n + 1, COIN_SLOTS)))[1]; m0 = rotor_sum(rotor_items(length, t, max(n - 1, 0)))[1]
        key = "%d_%g" % (length, t)
        out[key] = dict(length=length, t=t, coins=n, spacers=COIN_SLOTS - n, mass=mass, imbalance_gmm=mx, side_gmm=my, com_z=zc, coin_step_gmm=COIN_G * abs(POD_XC),
                        plus1_gmm=m1, minus1_gmm=m0, inertia_gmm2=spin_inertia(length, t, n), gap=P.coin_gap(n),
                        items=[dict(name=nm, g=round(g, 2), x=round(float(c[0]), 1), z=round(float(c[2]), 1)) for nm, g, c in items])
        if (length, t) in P.COINS and P.COINS[(length, t)] != n and not quiet:
            print("!! parts2.COINS 의 동전 수(%d)가 가장 잘 맞는 수(%d)와 다르다: p2.N_COIN 을 고친다" % (P.COINS[(length, t)], n))
        if not quiet:
            print("판 100x%d x %gT: 동전 %d개 + 채움 %d개 -> 회전부 %.1f g, 남는 불균형 %+.0f g·mm (판 쪽이 +) = %.2f mN·m. 동전 하나 = %.0f g·mm. 관성 %.2e g·mm2" % (
                length, t, n, COIN_SLOTS - n, mass, mx, abs(mx) * 9.81e-3, COIN_G * abs(POD_XC), out[key]["inertia_gmm2"]))
    out["150.0"] = out["150_%g" % AL_T]                               # 기본: 100 x 150 x 1T
    # ---- 끄덕임 관절: 머리 좌표의 점 (y, z) 를 끄덕임 축에서의 거리로
    H = []; H3 = []
    def add(name, g, c):
        H.append((name, g, c[1] - Y_T, c[2] - S_T)); H3.append((name, float(g), [float(c[0]), float(c[1]), float(c[2])]))
    for name, fn, f in (("머리 몸통", P.head_shell, 1.0), ("머리 뚜껑", P.head_lid, 1.0), ("모터 자리", P.crown, 0.9), ("부채꼴 기어", P.sector_head, 0.8),
                        ("마커 틀", P.marker_frame, 1.0), ("ESP32 안테나 누름판", P.ant_clip, 1.0), ("ESP32 막음쇠", P.xiao_cap, 1.0)):
        v, c = props(fn()); add(name, v * RHO * f, c)
    plate_t = (fill or {}).get("plate_t", AL_T); n = out["150_%g" % plate_t]["coins"]
    for name, g, c in rotor_items(150.0, plate_t, n):                # 회전부 (판 100 x 150): 균형이 맞으면 무게중심이 회전축 위에 온다
        add(name, g, c)
    add("모터 3", 34.0, (-8, 0, 19.5)); add("파이 3B+", 50.0, (-14, -4, -37.5)); add("드라이버 3장", 24.0, (32, 0, -48))
    add("점퍼선·모터선", 22.0, (20, -12, -45)); add("전원 플러그와 선", 12.0, (-48, -4, -50)); add("나사", 14.0, (0, -8, -30))
    add("버튼 모듈과 선", 5.0, (BTN_X, BTN_PCB_Y, BTN_Z - 8))
    add("무선 동글", 4.0, (BX0 + DONGLE["x"], -PI_T / 2 - DONGLE["up"], BZ0 - 2.0 - (DONGLE["L"] - DONGLE["plug"]) / 2))
    xb = P._xiao_box(); add("ESP32 (XIAO ESP32-C5, 핀 머리)", 4.0, ((xb[0] + xb[1]) / 2, (xb[2] + xb[3]) / 2, (xb[4] + xb[5]) / 2))
    add("ESP32 안테나 (FPC + 동축선)", 2.0, (0.0, ESP_ANT["yc"], HZ1 - 15.0)); add("ESP32 점퍼선 4가닥", 5.0, (-5.0, -20.0, -30.0))
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
    res = dict(rotor=out, head_g=mass, com_up=mz, com_back=my, moment=mom, moment_max=worst, ratio=ratio, cap=cap, yoke_g=yoke_g, turn_g=turn_g, lan=dict(head_g=lan[0], moment=lan[3], moment_max=lan[4]),
               plate_t=plate_t, head_items=[dict(name=h[0], g=round(float(h[1]), 1), back=round(float(h[2]), 1), up=round(float(h[3]), 1)) for h in H[:-1]],
               head_items3=[dict(name=n_, g=round(g_, 2), c=[round(v, 2) for v in c_]) for n_, g_, c_ in H3[:-1]])
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
