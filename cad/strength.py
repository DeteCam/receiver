"""강도 검사: 출력한 부품이 쓰는 동안, 그리고 손으로 다룰 때 부러지지 않는지 본다.
    python3 strength.py        ->  결과 폴더의 strength.json

방법: 부품마다 가장 약한 단면을 모델에서 잘라 넓이와 단면 2차 모멘트를 구하고, 걸리는 힘으로 응력을 계산해 재료 세기와 견준다.
재료(PLA 매트) 세기는 낮게 잡았다. 층과 층 사이는 층 안쪽의 절반도 안 되게 본다 (매트 PLA 는 층 붙는 힘이 약하다).
여유(안전율) = 재료 세기 / 걸리는 응력. 2 이상이면 됨, 2 아래면 '조심'으로 표시한다.
걸리는 힘은 세 가지로 나눈다: ① 스스로 움직일 때 (무게, 모터가 낼 수 있는 가장 큰 힘, 멈춤턱에 밀어붙일 때)
                              ② 손이 닿을 때 (판을 툭 침 5N, 머리를 손으로 밂 10N, 머리를 잡고 들어 올림)  ③ 조립할 때
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import paths
from cadlib import *
from p2 import *

# ---- 재료 (MPa). 매트 PLA 를 낮게 잡은 값
S_XY, S_Z = 40.0, 15.0            # 당기는 세기: 층 안쪽 방향 / 층 사이
T_XY, T_Z = 25.0, 12.0            # 미는(전단) 세기: 층 안쪽 / 층 사이
S_BEAR = 50.0                     # 눌리는 세기
E_MOD = 2500.0                    # 탄성 계수
G9 = 9.81e-3                      # g -> N

# ---- 힘
T_MOTOR = 50.0                    # 모터가 낼 수 있는 가장 큰 힘 (mN·m). 정격 34 보다 높게 잡았다
T_BACK = 118.0                    # 꺼진 모터를 억지로 돌릴 때 버티는 힘의 위쪽 값 (mN·m, 28BYJ-48 자료의 마찰 토크 600~1200 gf·cm)
F_BUMP, F_HAND = 5.0, 10.0        # 판을 툭 침, 머리를 손으로 밂 (N)
M = json.load(open(os.path.join(paths.RES, "mass.json"), encoding="utf-8"))
W_HEAD = M["head_g"] * G9
W_DEV = 1100.0 * G9               # 장치 전체 (큰 배터리까지 넣은 어림, N)

rows = []


def add(part, case, what, stress, allow, note=""):
    sf = allow / stress if stress > 1e-9 else 999.0
    rows.append(dict(part=part, case=case, what=what, stress=round(float(stress), 2), allow=allow, sf=round(float(min(sf, 999.0)), 1), ok=bool(sf >= 2.0), note=note))
    print("%-16s %-34s %-22s %6.2f MPa / %4.0f  여유 %6.1f배 %s  %s" % (part, case, what, stress, allow, min(sf, 999.0), "" if sf >= 2.0 else "<== 조심", note))


def section(m, axis, at):
    """부품 m 을 axis(0,1,2) = at 인 면으로 자른 단면의 넓이, 도심, 단면 2차 모멘트 (단면 안의 두 축에 대해). 반환: A, (c1, c2), (I1, I2), 꼭짓점 범위"""
    if axis == 0:
        r = m.rotate([0, 90, 0]); zc = -at            # x -> -z
    elif axis == 1:
        r = m.rotate([-90, 0, 0]); zc = -at           # y -> -z
    else:
        r = m; zc = at
    cs = r.slice(float(zc))
    A = 0.0; Sx = 0.0; Sy = 0.0; Ixx = 0.0; Iyy = 0.0; lo = np.array([1e9, 1e9]); hi = -lo
    for pg in cs.to_polygons():
        p = np.asarray(pg, dtype=float); q = np.roll(p, -1, axis=0)
        cr = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
        A += cr.sum() / 2
        Sx += ((p[:, 0] + q[:, 0]) * cr).sum() / 6; Sy += ((p[:, 1] + q[:, 1]) * cr).sum() / 6
        Ixx += ((p[:, 1] ** 2 + p[:, 1] * q[:, 1] + q[:, 1] ** 2) * cr).sum() / 12
        Iyy += ((p[:, 0] ** 2 + p[:, 0] * q[:, 0] + q[:, 0] ** 2) * cr).sum() / 12
        lo = np.minimum(lo, p.min(0)); hi = np.maximum(hi, p.max(0))
    if abs(A) < 1e-9:
        return 0.0, (0, 0), (0, 0), (lo, hi)
    cx_, cy_ = Sx / A, Sy / A
    return abs(A), (cx_, cy_), (abs(Ixx - A * cy_ * cy_), abs(Iyy - A * cx_ * cx_)), (lo, hi)        # I1: 단면의 가로축 둘레, I2: 세로축 둘레


res = {}
zc = (BAR_Z0 + BAR_Z1) / 2

# ================================================================ 1. 판 (얇은 판 0.8mm + 살)
for length in (150.0, 200.0):
    b = P.blade(length)
    for tag, zcut, arm in (("아래쪽", zc - 6.5, zc - 6.5 - BL_Z0), ("위쪽", zc + 6.5, BL_Z0 + length - (zc + 6.5))):
        A, c, I, (lo, hi) = section(b, 2, zcut)
        # 단면은 (x, y). 판 두께 방향 = x. 꺾이는 축 = y 축 둘레 -> I2
        cmax = max(hi[0] - c[0], c[0] - lo[0])
        Mb = F_BUMP * arm
        sig = Mb * cmax / I[1]
        defl = 1.0 * arm ** 3 / (3 * E_MOD * I[1])
        add("판 %d" % length, "판 %s 끝을 5N 으로 밂" % tag, "판 꺾임 (꽂는 자리 옆)", sig, S_XY, "1N 에 끝이 %.1fmm 휜다" % defl)
        res["blade_%d_%s" % (length, tag)] = dict(I=round(I[1], 1), c=round(cmax, 2), arm=round(arm, 1), defl_per_N=round(defl, 2))
    # 제 무게로 처지는 양 (누운 자세: 판이 수평)
    A, c, I, _ = section(b, 2, zc - 6.5)
    w = b.volume() * 1.24e-3 * G9 / length                                         # N/mm
    L = zc - 6.5 - BL_Z0
    res["blade_%d_self_sag" % length] = round(w * L ** 4 / (8 * E_MOD * I[1]), 3)

# 판을 꽂는 자리: 막대 끝 돌기(10 x 6)가 7.3mm 들어가고 나사 1개가 누른다
eng = TENON_X1 - COLLAR_X0
arm_lo = zc - BL_Z0                                                                # 판 아래끝까지
Mt = F_BUMP * arm_lo
# 돌기를 비트는 쪽 (판 아래 모서리를 옆으로 밂): 사각 단면 비틀림
bw, th = TENON_W, BAR_Z1 - BAR_Z0
beta = 0.208 + (0.231 - 0.208) * ((bw / th) - 1.5) / 0.25 if bw / th < 1.75 else 0.239
tau = Mt / (beta * bw * th * th)
add("회전 막대", "판 아래 모서리를 옆으로 5N", "막대 끝 돌기 비틀림", tau, T_Z, "돌기 %g x %g, 층 사이 전단으로 봄" % (bw, th))
# 돌기를 꺾는 쪽 (판 아래끝을 머리 쪽으로 밂): 나사 구멍 자리의 남은 단면
In = (bw - 2.6) * th ** 3 / 12
add("회전 막대", "판 아래끝을 머리 쪽으로 5N", "막대 끝 돌기 꺾임", Mt * (th / 2) / In, S_XY, "나사 구멍을 뺀 단면 %.1f x %g" % (bw - 2.6, th))
# 꽂는 자리 벽: 돌기가 지렛대처럼 눌러서 생기는 힘
Fc = Mt / (eng * 2 / 3)
add("판", "판 아래끝을 머리 쪽으로 5N", "꽂는 자리 벽 눌림", Fc / (bw * eng / 3), S_BEAR, "돌기가 %.1fmm 들어감, 누르는 힘 %.0fN" % (eng, Fc))
wall = 6.0 - (th + 0.3) / 2                                                         # 꽂는 자리 위·아래 벽 두께 (collar 높이 12)
add("판", "판 아래끝을 머리 쪽으로 5N", "꽂는 자리 벽 찢김", Fc / (bw * wall), T_Z, "벽 두께 %.2fmm" % wall)
# 막대 팔 (가운데 근처): 판 끝을 위아래로 밂
A, c, I, (lo, hi) = section(P.rotor_bar(), 0, 14.0)
add("회전 막대", "판을 위에서 5N 으로 누름", "막대 팔 꺾임 (가운데 옆)", F_BUMP * (R_FOIL - 14.0) * ((hi[1] - lo[1]) / 2) / I[0], S_XY, "단면 %.0f x %.0f" % (hi[0] - lo[0], hi[1] - lo[1]))
# 축 구멍 (D 모양): 꺼진 모터를 판으로 억지로 돌릴 때
flat = D_BORE_FLAT; dia = D_BORE_D; Lf = 6.0
lever = (dia / 2 + flat / 2) / 2 * 0.9                                              # 납작한 면에서 힘이 걸리는 팔 길이 어림
Ff = T_BACK / (2 * lever)
add("회전 막대", "꺼진 판을 손으로 돌림", "축 구멍 납작한 면 눌림", Ff / (Lf * 1.0), S_BEAR, "모터가 버티는 힘 %.0f mN·m 까지" % T_BACK)
res["blade_bump_torque_mNm"] = F_BUMP * R_FOIL
res["note_motor3"] = "판 가장자리를 5N 으로 치면 축에 %.0f mN·m 가 걸린다: 모터가 버티는 힘(%.0f)보다 커서 모터가 따라 돈다 (출력물보다 모터 기어가 먼저 받는다)" % (F_BUMP * R_FOIL, T_BACK)

# ================================================================ 2. 끄덕임 핀 (세워서 출력: 층이 축에 직각)
r = PIN_FITS[2]; Apin = np.pi * r * r; Zpin = np.pi * (2 * r) ** 3 / 32
gap = ARM_Y0 - HX                                                                   # 머리 옆면과 팔 사이 (핀이 드러난 길이)
for case, F in (("머리 무게", W_HEAD / 2), ("머리를 잡고 장치를 듦", W_DEV / 2), ("머리를 옆에서 세게 침 50N", 50.0)):
    add("끄덕임 핀", case, "핀 잘림", F / Apin, T_Z)
    add("끄덕임 핀", case, "핀 꺾임", F * gap / Zpin, S_Z, "드러난 길이 %.0fmm" % gap)

# ================================================================ 3. 요크 팔 (높이 100 에서 자른 단면: 두 팔 + 뒷가림 벽)
yk = P.yoke()
for zcut in (95.0, 110.0):
    A, c, I, (lo, hi) = section(yk, 2, zcut)
    h_top = 235.0                                                                   # 머리 꼭대기(판 위끝 근처) 높이
    # 옆으로 밂 (y 방향): x 축 둘레로 꺾임 -> I1 (단면 가로축 = x), 가장 먼 곳 = y 끝
    cy_max = max(hi[1] - c[1], c[1] - lo[1]); cx_max = max(hi[0] - c[0], c[0] - lo[0])
    add("요크 팔", "머리 꼭대기를 옆으로 10N", "팔 꺾임 (높이 %d)" % zcut, F_HAND * (h_top - zcut) * cy_max / I[0], S_Z, "층 사이로 당겨짐. 단면 넓이 %.0f mm2" % A)
    add("요크 팔", "머리 꼭대기를 앞뒤로 10N", "팔 꺾임 (높이 %d)" % zcut, F_HAND * (h_top - zcut) * cx_max / I[1], S_Z)
res["arm_section_110"] = dict(A=round(A, 0), I_side=round(I[0], 0), I_fore=round(I[1], 0))

# ================================================================ 4. 기어 이 (루이스 식)
rp = MOD * ZP / 2.0
Y_PIN, Y_BIG = 0.30, 0.45                                                           # 모양 계수: 10T(전위 +0.45), 큰 기어
for name, face, ratio, Rbig in (("끄덕임", SEC_Y1 - SEC_Y0, SEC_T / float(ZP), SEC_R), ("받침 회전", 5.3, ZW / float(ZP), MOD * ZW / 2)):
    Ft = T_MOTOR / rp
    add("작은 기어 (%s)" % name, "모터가 가장 세게 밀 때", "이 뿌리 꺾임", Ft / (face * MOD * Y_PIN), S_XY, "이에 걸리는 힘 %.1fN, 이 폭 %.1fmm" % (Ft, face))
    add("큰 기어 (%s)" % name, "모터가 가장 세게 밀 때", "이 뿌리 꺾임", Ft / (face * MOD * Y_BIG), S_XY)
h_arm = (HZ1 + 21.0) - S_T                                                           # 끄덕임 축에서 머리 꼭대기(모터 자리 위끝)까지
Th = F_HAND * h_arm
Ft = Th / SEC_R
add("작은 기어 (끄덕임)", "머리 꼭대기를 앞으로 10N", "이 뿌리 꺾임", Ft / ((SEC_Y1 - SEC_Y0) * MOD * Y_PIN), S_XY, "모터 축에 %.0f mN·m (버티는 힘 %d~%d)" % (Ft * rp, 59, T_BACK))
# 부채꼴 기어의 살 (폭 7, 두께 4.5): 이에 걸리는 힘이 살을 꺾는다. 살 4개 중 2개가 받는다고 본다
Ft = T_MOTOR / rp
Zs = (SEC_Y1 - SEC_Y0) * 7.0 ** 2 / 6
add("부채꼴 기어", "모터가 가장 세게 밀 때", "살 꺾임", Ft * (SEC_R - 23.0) / (2 * Zs), S_XY)
# 부채꼴 기어 고정 나사 2개 (반지름 17)
Fs = T_MOTOR * (SEC_T / float(ZP)) / (2 * SEC_MOUNT[0][0])
add("부채꼴 기어", "모터가 가장 세게 밀 때", "나사 구멍 눌림", Fs / (3.0 * (SEC_Y1 - SEC_Y0 - SEC_CB_D)), S_BEAR, "나사 하나에 %.0fN" % Fs)

# ================================================================ 5. 멈춤턱 (원점 잡을 때 모터가 밀어붙인다)
Fl = T_MOTOR * (SEC_T / float(ZP)) / LUG_R
add("머리 몸통", "끄덕임 멈춤에 밀어붙임", "멈춤 돌기 잘림", Fl / (5.0 * 4.6), T_XY, "돌기에 %.0fN" % Fl)
add("요크", "끄덕임 멈춤에 밀어붙임", "멈춤 나사 자리 눌림", Fl / (3.0 * (ARM_Y1 - ARM_Y0)), S_BEAR)
rs = float(np.hypot(*YAW_STOP_XY))
Fy = T_MOTOR * (ZW / float(ZP)) / rs
add("받침 덮개", "받침 회전 멈춤에 밀어붙임", "멈춤턱 잘림", Fy / (4.0 * (2 * rs * np.radians(3.5))), T_Z, "턱에 %.0fN" % Fy)

# ================================================================ 6. 회전판 고리 (요크가 들리지 않게 잡는다)
ro = TUBE_RO - 1.5 + 0.15
arc = 2 * np.pi * TUBE_RO * (1 - 2 * np.arcsin((ro - 0.9) / (ro + 4.0)) / (2 * np.pi))
add("고리", "요크를 잡고 받침째 듦", "고리 잘림", (W_DEV * 0.5) / (arc * CLIP_T), T_Z, "받침+배터리 무게를 %.0fN 으로 봄" % (W_DEV * 0.5))
# 고리가 벌어져 빠지는 힘: 홈 윗벽이 45도라 들어 올리면 고리를 밖으로 민다
Rm = ro + 2.0; Iclip = CLIP_T * 4.0 ** 3 / 12
P_open = (2 * np.pi * (TUBE_RO - ro)) * E_MOD * Iclip / (3 * np.pi * Rm ** 3)        # 틈을 벌리는 힘 어림
F_lift = P_open * 2.5 * (1 + 0.3) / (1 - 0.3)                                        # 마찰 0.3, 45도 벽
res["clip_release_N"] = round(F_lift, 0)
res["tip_over"] = {}

# ================================================================ 7. 넘어짐: 무게중심이 받침 밖으로 나가는지, 얼마나 밀면 넘어가는지
def tip(size, batt_g, length, pitch):
    my, mz = M["com_back"], M["com_up"]; p = np.radians(pitch)
    hx = -my * np.cos(p) + mz * np.sin(p); hz = H_T + my * np.sin(p) + mz * np.cos(p)
    head = M["head_g"] + (16.0 if length > 150 else 0.0)
    yoke = M["yoke_g"] + 34 * 2 + 10
    basep = (60.0 if size == "S" else 95.0) + 45.0 + batt_g                           # 통 + 덮개 + 배터리
    tot = head + yoke + basep
    cgx = (head * hx + yoke * (-12.0)) / tot                                          # 요크는 전선 상자 때문에 뒤로 조금
    cgz = (head * hz + yoke * 70.0 + basep * 14.0) / tot
    R = BASE[size]["R"]
    F_tip = tot * G9 * (R - abs(cgx)) / 235.0                                         # 머리 꼭대기(높이 235)를 넘어가는 쪽으로 미는 힘
    return dict(total_g=round(tot), cg_off=round(cgx, 1), cg_h=round(cgz, 1), margin_mm=round(R - abs(cgx), 1), push_N=round(F_tip, 2))
for size, batt in (("S", 200.0), ("S", 0.0), ("L", 380.0)):
    for length in (150.0, 200.0):
        for pitch in (TILT3, 85.0):
            res["tip_over"]["%s 배터리 %dg 판 %d 끄덕임 %.0f" % (size, batt, length, pitch)] = tip(size, batt, length, pitch)

# ================================================================ 8. 작은 것들
d = 2.5
F_peg = S_Z * (np.pi * d ** 3 / 32) / 3.0
res["pi_peg_break_N"] = round(F_peg, 1)
add("머리 몸통", "파이를 꽂을 때 옆으로 5N", "파이 기둥 못 꺾임 (지름 2.5)", 5.0 * 3.0 / (np.pi * d ** 3 / 32), S_Z, "못 하나가 옆 힘 %.0fN 에서 부러진다: 파이는 똑바로 눌러 꽂는다" % F_peg)
add("머리 뚜껑", "버튼을 5N 으로 누름", "버튼 받침 살 눌림", 5.0 / (1.8 * 6.0), S_BEAR)
add("모터 자리", "판 모터가 가장 세게 돌 때", "모터 귀 기둥 꺾임", (T_MOTOR / (2 * M_EAR)) * 14.0 / (np.pi * 6.0 ** 3 / 32), S_Z)

bad = [r_ for r_ in rows if not r_["ok"]]
print("\n검사 %d개 가운데 여유 2배 아래 %d개" % (len(rows), len(bad)))
for k, v in res["tip_over"].items():
    print("   넘어짐 %-28s 전체 %4dg, 무게중심이 축에서 %5.1fmm, 받침 끝까지 %5.1fmm, 꼭대기를 %.1fN 으로 밀면 넘어감" % (k, v["total_g"], v["cg_off"], v["margin_mm"], v["push_N"]))
print("   판이 제 무게로 처지는 양: 150 = %.2fmm, 200 = %.2fmm.  고리가 벌어져 빠지는 힘 약 %dN.  %s" % (res["blade_150_self_sag"], res["blade_200_self_sag"], res["clip_release_N"], res["note_motor3"]))
res["rows"] = rows; res["n"] = len(rows); res["n_low"] = len(bad)
res["material"] = dict(S_XY=S_XY, S_Z=S_Z, T_XY=T_XY, T_Z=T_Z, S_BEAR=S_BEAR, E=E_MOD)
res["loads"] = dict(T_MOTOR=T_MOTOR, T_BACK=T_BACK, F_BUMP=F_BUMP, F_HAND=F_HAND, W_HEAD=round(W_HEAD, 2), W_DEV=round(W_DEV, 2))
json.dump(res, open(os.path.join(paths.RES, "strength.json"), "w"), ensure_ascii=False, indent=1)
