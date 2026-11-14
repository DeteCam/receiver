"""강도 검사: 출력한 부품이 쓰는 동안, 그리고 손으로 다룰 때 부러지지 않는지 본다.
    python3 strength.py        ->  결과 폴더의 strength.json

방법: 부품마다 가장 약한 단면을 모델에서 잘라 넓이와 단면 2차 모멘트를 구하고, 걸리는 힘으로 응력을 계산해 재료 세기와 견준다.
재료(PLA 매트) 세기는 낮게 잡았다. 층과 층 사이는 층 안쪽의 절반도 안 되게 본다 (매트 PLA 는 층 붙는 힘이 약하다).
여유(안전율) = 재료 세기 / 걸리는 응력. 2 이상이면 됨, 2 아래면 '조심'으로 표시한다.
걸리는 힘은 세 가지로 나눈다: ① 스스로 움직일 때 (무게, 모터가 낼 수 있는 가장 큰 힘, 멈춤턱에 밀어붙일 때)
                              ② 손이 닿을 때 (판을 툭 침 5N, 머리를 손으로 밂 10N, 머리를 잡고 들어 올림)  ③ 조립할 때
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
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

# ================================================================ 1. 회전부: 판 틀(한 장) + 알루미늄 판 1T + 누름쇠 + 판 받이(플랜지) + 판 틀 나사 4개 + 축 고정 나사
AL_YIELD = {"1050-H14 (무른 것)": 95.0, "5052-H32 (단단한 것)": 190.0}                # 알루미늄 판이 휘어서 안 돌아오기 시작하는 응력 (MPa)
E_AL = 69000.0
zb_lo = min(z for y, z in PLATE_BOLTS); zb_hi = max(z for y, z in PLATE_BOLTS); yb = max(y for y, z in PLATE_BOLTS)
FR = P.plate_frame(); CL = P.plate_clamp()
# 판은 틀 홈에 들어가 아래 턱과 누름쇠에 잡힌다: 판 끝을 밀면 판과 틀이 같이 휘며 굽힘 강성(EI)의 비로 힘을 나눠 받는다. 뿌리 = 판 받침(받이) 아래끝·위끝
zcut_lo, zcut_hi = FL_Z0 - 4.0, BAR_Z1 + 4.0
z_bot, z_top = BL_Z0 - FR_END, BL_Z0 + 150.0 + FR_END_TOP
t = AL_T
Ip = PLATE_W * t ** 3 / 12; Zp = PLATE_W * t ** 2 / 6
for tag, zcut, arm in (("아래", zcut_lo, FL_Z0 - z_bot), ("위", zcut_hi, z_top - BAR_Z1)):
    Ab, cb, Ib, (lb, hb) = section(FR, 2, zcut)
    EI = {"판": E_AL * Ip, "틀": E_MOD * Ib[1]}; tot = sum(EI.values())
    Mb = F_BUMP * arm
    sig_p = Mb * EI["판"] / tot / Zp
    sig_b = Mb * EI["틀"] / tot * max(hb[0] - cb[0], cb[0] - lb[0]) / Ib[1]
    defl = arm ** 3 / (3 * tot)
    add("알루미늄 판 %gT" % t, "판 %s 끝(틀)을 5N 으로 밂" % tag, "판 꺾임 (판 받침 %s끝)" % tag, sig_p, AL_YIELD["1050-H14 (무른 것)"],
        "판이 힘의 %.0f%% 를 받는다 (틀이 나머지). 1N 에 끝이 %.1fmm 휜다. 무른 판(1050) 기준" % (EI["판"] / tot * 100, defl))
    add("판 틀", "판 %s 끝(틀)을 5N 으로 밂" % tag, "틀 꺾임 (판 받침 %s끝, 층 안쪽)" % tag, sig_b, S_XY, "틀 단면 %.0f mm2" % Ab)
    res["plate_150_%g_%s" % (t, tag)] = dict(arm=round(arm, 1), stress=round(sig_p, 1), share=round(EI["판"] / tot, 2), defl_per_N=round(defl, 2), frame=round(sig_b, 1))
    if tag == "아래":                                                     # 판을 넣기 전, 빈 틀만 손에 들고 아래끝을 밀 때 (판이 없으면 틀이 혼자 받는다)
        add("판 틀", "판을 넣기 전, 빈 틀 아래끝을 2N 으로 밂", "틀 꺾임 (판 받침 아래끝)", 2.0 * arm * max(hb[0] - cb[0], cb[0] - lb[0]) / Ib[1], S_XY, "빈 틀은 판보다 훨씬 무르다. 판을 넣기 전에는 받침(판 받이 자리)을 잡고 다룬다")
# 누운 자세(끄덕임 85도)에서 판과 틀이 제 무게로 처지는 양: 판 받침 아래끝에서 아래끝까지를 한쪽 고정 보로
Ab, cb, Ib, _ = section(FR, 2, zcut_lo - 20.0)
L = FL_Z0 - z_bot; w = (PLATE_W * t * AL_RHO + Ab * 1.24e-3) * G9
res["plate_%g_self_sag" % t] = round(w * L ** 4 / (8 * (E_AL * Ip + E_MOD * Ib[1])), 3)
# 판이 홈에서 들리는 쪽 (판 끝을 바깥으로 5N 당김): 아래는 45도 턱, 위는 누름쇠가 받는다
lip_w = 2 * FR_LIP_HW
add("판 틀", "판 아래끝을 바깥으로 5N 당김", "아래 턱 뿌리 꺾임 (층 사이)", 6 * F_BUMP * 0.5 / (lip_w * FR_LIP ** 2), S_Z, "턱 길이 %.0fmm, 두께 %.1f. 판 모서리가 턱 뿌리에서 0.5mm 안쪽을 민다" % (lip_w, FR_LIP))
e_cl = (BL_Z0 + 150.0 + FR_TOP_DZ) - (BL_Z0 + 150.0 - CL_LAP / 2)
add("누름쇠", "판 위끝을 바깥으로 5N 당김", "누름쇠 꺾임 (나사 줄과 판 사이)", 6 * F_BUMP * e_cl / (2 * CL_HW * CL_T ** 2), S_XY, "나사 줄에서 판을 누르는 곳까지 %.1fmm" % e_cl)
ceil = FL_X1 + AL_T - CL_SEAT - (FL_XH + NUT_POCKET)                     # 너트 자리 위에 남은 살 두께
per = 6 * 5.5 / np.sqrt(3.0)                                              # 너트 둘레 (맞변 5.5 육각)
add("판 틀", "판 위끝을 바깥으로 5N 당김", "너트 자리 위 살 뚫림 (층 사이로 잘림)", (F_BUMP / 2) / (per * ceil), T_Z, "나사 2개, 너트 위 살 %.1fmm" % ceil)
F_cl = 0.1e3 / (0.2 * 3.0)                                               # 0.1 N·m 로 조였을 때 나사가 당기는 힘 (N): T = 0.2 x d x F
add("판 틀", "누름쇠 나사를 조임 (0.1 N·m)", "너트 자리 위 살 뚫림 (층 사이로 잘림)", F_cl / (per * ceil), T_Z, "나사가 당기는 힘 약 %.0fN. 누름쇠가 판에 닿아 멈출 때까지만 조인다" % F_cl)
add("누름쇠", "누름쇠 나사를 조임 (0.1 N·m)", "접시머리 밑 눌림", F_cl / (np.pi * (3.35 ** 2 - 1.7 ** 2) * 0.7), S_BEAR, "누름쇠 두께 %.1f" % CL_T)
# 판 틀 나사 4개: 판 아래끝(틀)을 머리 쪽으로 5N -> 나사 줄 사이(18mm)가 지렛대. 위 줄 나사 2개가 당겨진다
arm_lo = zb_lo - z_bot
F_bolt = F_BUMP * arm_lo / (zb_hi - zb_lo) / 2
add("판 받이", "판 아래끝(틀)을 머리 쪽으로 5N", "너트 밑 플라스틱 눌림", F_bolt / (np.pi * (2.75 ** 2 - 1.7 ** 2)), S_BEAR, "나사 하나에 %.1fN (너트 자리 밑 두께 %.1f)" % (F_bolt, FL_T - 2.2))
add("판 받이", "판 아래끝(틀)을 머리 쪽으로 5N", "너트가 뽑혀 나오는 쪽 (너트 밑 살 잘림)", F_bolt / (2 * np.pi * 3.0 * (FL_T - 2.2)), T_Z)
add("판 틀", "판 아래끝(틀)을 머리 쪽으로 5N", "접시머리 밑 받침 눌림", F_bolt / (np.pi * (3.35 ** 2 - 1.7 ** 2) * 0.7), S_BEAR, "바닥 두께 %.1f, 접시 자리 경사면" % PAD_T)
# 판 받이가 막대에 붙은 뿌리: 막대 윗면 아래 34mm 높이의 벽(두께 3.6, 폭 54)이 세워진 채 출력된다 = 층 사이로 당겨진다. 세모 살 2개가 같이 받는다
Mroot = F_BUMP * (BAR_Z0 - (BL_Z0 - FR_END))
rb = P.rotor_bar()
A, c, I, (lo, hi) = section(rb ^ box(60.0, FL_X1 + 1, -FL_HW - 1, FL_HW + 1, BAR_Z0 - 2.0, BAR_Z0 + 2.0), 2, BAR_Z0 - 0.5)        # 막대 바로 아래에서 자른 단면 (판 받이 + 세모 살)
cmax = max(hi[0] - c[0], c[0] - lo[0])
add("판 받이", "판 아래끝(틀)을 머리 쪽으로 5N", "판 받이 뿌리 꺾임 (층 사이)", Mroot * cmax / I[1], S_Z, "막대 바로 아래 단면: 넓이 %.0f mm2 (세모 살 2개 포함)" % A)
res["flange_root"] = dict(A=round(A, 1), I=round(I[1], 1), c=round(cmax, 2), M=round(Mroot, 1))
# 판 옆 모서리를 5N 으로 밂 (판을 돌리는 쪽): 볼트 4개가 비틀림을 받고, 막대 팔이 옆으로 꺾인다
Mt = F_BUMP * (PLATE_W / 2)
rbolt = np.hypot(yb, (zb_hi - zb_lo) / 2)
add("판 받이", "판 옆 모서리를 5N 으로 밂", "나사 구멍 눌림", (Mt / (4 * rbolt)) / (3.0 * (FL_T - 2.2)), S_BEAR, "나사 하나에 %.1fN" % (Mt / (4 * rbolt)))
# 막대 팔 (가운데 근처): 판 끝을 위아래로 밂 / 옆으로 밂
A, c, I, (lo, hi) = section(rb, 0, 14.0)
add("회전 막대", "판을 위에서 5N 으로 누름", "막대 팔 꺾임 (가운데 옆)", F_BUMP * (R_FOIL - 14.0) * ((hi[1] - lo[1]) / 2) / I[0], S_XY, "단면 %.0f x %.0f" % (hi[0] - lo[0], hi[1] - lo[1]))
add("회전 막대", "판 옆 모서리를 5N 으로 밂", "막대 팔 옆으로 꺾임", F_BUMP * (R_FOIL - 14.0) * ((hi[0] - lo[0]) / 2) / I[1], S_XY)
# 균형추 쪽 팔: 동전 12개(65g) + 통의 무게가 누운 자세에서 옆으로 걸린다
A2, c2, I2, (lo2, hi2) = section(rb, 0, -14.0)
Wp = (COIN_G * COIN_SLOTS + 8.0) * G9
add("회전 막대", "동전 12개를 넣고 누운 자세", "균형추 쪽 팔 꺾임", Wp * (abs(POD_XC) - 14.0) * ((hi2[0] - lo2[0]) / 2) / I2[1], S_XY, "동전·통 무게 %.2fN" % Wp)
# 마개 나사 2개: 동전 무게를 아래에서 받친다
add("회전 막대", "동전 12개 + 떨어뜨림(3배)", "마개 나사산 뽑힘 (M3, 4.7mm 물림)", 3 * Wp / (2 * np.pi * 3.0 * 4.7 * 0.5), T_XY, "나사 2개")
# 축 구멍 (D 모양): 꺼진 모터를 판으로 억지로 돌릴 때. 축 고정 나사(M3)가 납작한 면을 누른다
flat = D_BORE_FLAT; dia = D_BORE_D; Lf = 6.0
lever = (dia / 2 + flat / 2) / 2 * 0.9                                              # 납작한 면에서 힘이 걸리는 팔 길이 어림
Ff = T_BACK / (2 * lever)
add("회전 막대", "꺼진 판을 손으로 돌림", "축 구멍 납작한 면 눌림", Ff / (Lf * 1.0), S_BEAR, "모터가 버티는 힘 %.0f mN·m 까지" % T_BACK)
A_nut = 0.866 * 5.5 ** 2 - np.pi * 1.7 ** 2                                         # 너트 한 면의 넓이에서 구멍을 뺀 것 (17 mm2)
add("회전 막대", "축 고정 나사를 조임 (0.2 N·m)", "너트 자리 벽 눌림", (200.0 / (0.2 * 3.0)) / A_nut, S_BEAR, "나사가 미는 힘 약 330N. 손끝으로 멈추는 데까지만 조인다 (세게 조이면 너트 자리가 눌린다)")
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

# 넘어짐(무게중심, 기울기, 미는 힘)은 tipover.py 와 tip_mujoco.py 가 따로 본다

# ================================================================ 8. 작은 것들
# 파이 기둥 (M2.5 나사가 4.3mm 물린다), 드라이버 기둥 (M2 나사가 4.3mm 물린다): 나사산이 뽑히는 힘, 기둥이 옆으로 꺾이는 힘
pull = T_Z * np.pi * 2.5 * 4.3 * 0.5                                                 # 나사산의 절반이 힘을 받는다고 본다
res["m25_pullout_N"] = round(pull, 0)
add("머리 몸통", "파이를 앞으로 20N 으로 당김", "M2.5 나사산 뽑힘 (기둥 하나)", (20.0 / 4) / (np.pi * 2.5 * 4.3 * 0.5), T_Z, "기둥 하나가 약 %.0fN 까지 버틴다" % pull)
add("머리 몸통", "드라이버 보드를 앞으로 10N 으로 당김", "M2 나사산 뽑힘 (기둥 하나)", (10.0 / 4) / (np.pi * 2.0 * 4.3 * 0.5), T_Z, "M2 는 굵기가 작아 M2.5 의 0.8배쯤 버틴다")
Zp = np.pi * ((2 * PI_POST_R) ** 4 - (2 * PILOT25) ** 4) / (32 * 2 * PI_POST_R)
add("머리 몸통", "파이를 옆으로 10N 으로 밂", "파이 기둥 꺾임 (지름 6, 높이 4)", (10.0 / 4) * 4.0 / Zp, S_Z)
Zd = np.pi * ((2 * DRV_POST_R) ** 4 - (2 * PILOT25) ** 4) / (32 * 2 * DRV_POST_R)
add("머리 몸통", "드라이버 보드를 옆으로 10N 으로 밂", "드라이버 기둥 꺾임 (지름 5.4, 높이 3.4)", (10.0 / 4) * DRV_POST_H / Zd, S_Z)
add("마커 틀", "틀 가운데를 5N 으로 당김", "틀 꺾임 (나사 사이 103mm, 두께 2)", 5.0 * 103.0 / 4 / (2 * 6.5 * MK_FRAME["t"] ** 2 / 6), S_XY, "옆 띠 2개(폭 6.5)가 받는다")
add("핀 막이", "핀이 5N 으로 밀려 나옴", "막이 턱 꺾임", 5.0 * 3.0 / (9.0 * 1.65 ** 2 / 6), S_XY, "턱 두께 1.65, 폭 9")
add("머리 뚜껑", "버튼을 5N 으로 누름", "버튼 받침 살 눌림", 5.0 / (1.8 * 6.0), S_BEAR)
add("모터 자리", "판 모터가 가장 세게 돌 때", "모터 귀 기둥 꺾임", (T_MOTOR / (2 * M_EAR)) * 14.0 / (np.pi * 6.0 ** 3 / 32), S_Z)

bad = [r_ for r_ in rows if not r_["ok"]]
print("\n검사 %d개 가운데 여유 2배 아래 %d개" % (len(rows), len(bad)))
print("   누운 자세에서 판과 틀이 제 무게로 처지는 양: %gT = %.3fmm.  고리가 벌어져 빠지는 힘 약 %dN.  %s" % (AL_T, res["plate_%g_self_sag" % AL_T], res["clip_release_N"], res["note_motor3"]))
res["rows"] = rows; res["n"] = len(rows); res["n_low"] = len(bad)
res["material"] = dict(S_XY=S_XY, S_Z=S_Z, T_XY=T_XY, T_Z=T_Z, S_BEAR=S_BEAR, E=E_MOD)
res["loads"] = dict(T_MOTOR=T_MOTOR, T_BACK=T_BACK, F_BUMP=F_BUMP, F_HAND=F_HAND, W_HEAD=round(W_HEAD, 2), W_DEV=round(W_DEV, 2))
json.dump(res, open(os.path.join(paths.RES, "strength.json"), "w"), ensure_ascii=False, indent=1)
