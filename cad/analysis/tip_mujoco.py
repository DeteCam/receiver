"""넘어짐 검사 (움직일 때): 물리 엔진 MuJoCo 로 장치를 실제 속도로 움직여 본다.
    python3 tip_mujoco.py        ->  결과 폴더의 tip_mujoco.json

장치를 네 덩어리(받침, 요크, 머리, 회전부)의 강체로 만들고 관절 3개를 실제 모터 속도와 모터가 낼 수 있는 힘(토크 한계)으로 움직인다.
받침은 바닥에 그냥 놓여 있다 (고정하지 않는다): 넘어지거나 미끄러지면 그대로 보인다.
  1) 재기 한 번(세 자세, 판 3바퀴)을 실제 속도로: 받침이 들리거나 미끄러지는가
  2) 세 관절이 한꺼번에 가장 빨리 돌다가 갑자기 멈출 때 (실제로는 한 번에 하나씩만 돈다: 일부러 더 나쁘게)
  3) 바닥을 천천히 기울여 넘어가는 각 (tipover.py 의 계산과 견준다), 마찰이 보통일 때 미끄러지기 시작하는 각
  4) 꼭대기를 옆으로 미는 힘을 천천히 키워 넘어가는 힘
  5) 머리를 숙이다가 판 위끝이 벽에 걸렸을 때: 끄덕임 모터의 힘으로 자기를 넘어뜨리는가
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import mujoco
import tipmodel as TM
import paths
from p2 import *

DEG_S = {"yaw": 360.0 / (4076 * 4.0 * 0.0020), "pitch": 360.0 / (4076 * 11.2 * 0.0030), "spin": 360.0 / (4076 * 0.0015)}      # 관절 속도 (도/초): 11.0, 2.6, 58.9
TQ = {"yaw": 34.3e-3 * 4.0 * 0.9, "pitch": 34.3e-3 * 11.2 * 0.9, "spin": 34.3e-3}                                              # 모터가 낼 수 있는 힘 (N·m)
BACK = {"yaw": 0.06 * 4.0, "pitch": 0.06 * 11.2, "spin": 0.06}       # 밖에서 억지로 돌릴 때 모터 속 기어가 버티는 힘 (N·m): 28BYJ-48 자료의 600~1200 gf·cm 가운데 낮은 값. 관절 마찰로 넣는다


def inertial(items, origin):
    """(무게 g, 세계 좌표 mm) 목록 -> 무게(kg), 무게중심(m, origin 기준), 관성(무게중심 기준, kg·m2). 하나하나는 반지름 15mm 공으로 본다."""
    m = sum(g for g, c in items) * 1e-3; com = sum(g * np.asarray(c) for g, c in items) / sum(g for g, c in items)
    I = np.zeros((3, 3))
    for g, c in items:
        r = (np.asarray(c) - com) * 1e-3; I += g * 1e-3 * ((r @ r) * np.eye(3) - np.outer(r, r)) + 0.4 * g * 1e-3 * 0.015 ** 2 * np.eye(3)
    return m, (com - np.asarray(origin)) * 1e-3, I


def xml_inertial(m, c, I):
    return '<inertial mass="%.6f" pos="%.6f %.6f %.6f" fullinertia="%.4e %.4e %.4e %.4e %.4e %.4e"/>' % (m, c[0], c[1], c[2], I[0, 0], I[1, 1], I[2, 2], I[0, 1], I[0, 2], I[1, 2])


def build(d, mu=0.6, wall_x=None, fixed=False):
    b = inertial([(g, c) for _, g, c in d.base], (0, 0, 0)); y = inertial([(g, c) for _, g, c in d.yoke], (0, 0, 0))
    h = inertial([(g, TM.head_to_world(c, 0, 0)) for _, g, c in d.head], (0, 0, H_T))
    ax = TM.head_to_world([0, 0, 0], 0, 0) - np.array([0, 0, H_T])                         # 판 회전축이 지나는 점 (머리 덩어리 좌표)
    r = inertial([(g, TM.head_to_world(c, 0, 0, 0.0)) for _, g, c in d.rotor], TM.head_to_world([0, 0, 0], 0, 0))
    R = d.R * 1e-3; hh = COV_Z1 / 2 * 1e-3
    fx0, fx1 = FL_XH, FL_X1 + AL_T + CL_T; fz0, fz1 = BL_Z0 - FR_END, BL_Z0 + d.length + FR_END_TOP                   # 판 틀 뒷면 ~ 누름쇠 바깥면
    pc = TM.head_to_world([(fx0 + fx1) / 2, 0, (fz0 + fz1) / 2], 0, 0) - TM.head_to_world([0, 0, 0], 0, 0)          # 판 틀(판을 넣은 채) 가운데 (회전부 좌표)
    fw = PLATE_W / 2 + FR_GAP + FR_WALL                                                                               # 틀 반폭 (옆 테두리 바깥면)
    wall = "" if wall_x is None else '<geom name="wall" type="box" pos="%.4f 0 0.5" size="0.005 1 0.5" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" rgba="0.8 0.8 0.8 0.3"/>' % (wall_x + 0.005)
    return """<mujoco model="detecam">
  <option timestep="0.0005" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="5" noslip_iterations="20"/>
  <default><geom solref="0.003 1" solimp="0.99 0.999 0.0005"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 0.1" friction="%(mu).3f %(mt).5f 0.0001" condim="4" contype="1" conaffinity="1"/>
    %(wall)s
    <body name="base" pos="0 0 %(z0)s">
      %(free)s
      <site name="b0" pos="0 0 0" size="0.002"/>
      <geom name="foot" type="cylinder" size="%(R).4f %(hh).4f" pos="0 0 %(hh).4f" friction="%(mu).3f %(mt).5f 0.0001" condim="4" contype="1" conaffinity="1" mass="0"/>
      %(ib)s
      <body name="yoke" pos="0 0 0">
        <joint name="yaw" type="hinge" axis="0 0 1" damping="0.02" frictionloss="%(fy).4f"/>
        %(iy)s
        <body name="head" pos="0 0 %(ht).4f">
          <joint name="pitch" type="hinge" axis="0 1 0" damping="0.05" frictionloss="%(fp).4f"/>
          %(ih)s
          <body name="rotor" pos="%(ax0).5f %(ax1).5f %(ax2).5f">
            <joint name="spin" type="hinge" axis="0 0 1" damping="0.002" frictionloss="%(fs).4f"/>
            %(ir)s
            <geom name="plate" type="box" pos="%(p0).5f %(p1).5f %(p2).5f" size="%(pw).5f %(pt).5f %(pl).5f" contype="2" conaffinity="2" friction="0.4 0.005 0.0001" mass="0"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  %(sens)s
  <actuator>
    <position name="a_yaw" joint="yaw" kp="60" kv="3" forcerange="-%(ty).4f %(ty).4f"/>
    <position name="a_pitch" joint="pitch" kp="200" kv="8" forcerange="-%(tp).4f %(tp).4f"/>
    <position name="a_spin" joint="spin" kp="20" kv="0.6" forcerange="-%(ts).4f %(ts).4f"/>
  </actuator>
</mujoco>""" % dict(z0="0" if fixed else "0.0002", free="" if fixed else '<freejoint name="free"/>',
                    sens='<sensor><force name="bf" site="b0"/><torque name="bt" site="b0"/></sensor>' if fixed else "", mu=mu, mt=2.0 / 3.0 * mu * R, wall=wall, R=R, hh=hh, ib=xml_inertial(*b), iy=xml_inertial(*y), ih=xml_inertial(*h), ir=xml_inertial(*r), ht=H_T * 1e-3, ax0=ax[0] * 1e-3, ax1=ax[1] * 1e-3, ax2=ax[2] * 1e-3,
                    p0=pc[0] * 1e-3, p1=pc[1] * 1e-3, p2=pc[2] * 1e-3, pt=(fx1 - fx0) / 2 * 1e-3, pw=fw * 1e-3, pl=(fz1 - fz0) / 2 * 1e-3, ty=TQ["yaw"] + BACK["yaw"], tp=TQ["pitch"] + BACK["pitch"], ts=TQ["spin"] + BACK["spin"], fy=BACK["yaw"], fp=BACK["pitch"], fs=BACK["spin"])      # 구동기 한계 = 낼 수 있는 힘 + 제 마찰 (마찰을 이기고 남는 힘이 낼 수 있는 힘이 되게)


class Sim:
    def __init__(self, d, mu=0.6, wall_x=None, fixed=False):
        self.d = d; self.fixed = fixed; self.m = mujoco.MjModel.from_xml_string(build(d, mu, wall_x, fixed)); self.x = mujoco.MjData(self.m); self.log = []
        self.jq = {j: self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in ("yaw", "pitch", "spin")}
        self.bid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base"); self.rid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "rotor")
        self.tgt = dict(yaw=0.0, pitch=0.0, spin=0.0); mujoco.mj_forward(self.m, self.x); self.settle()

    def settle(self, t=0.5):
        for _ in range(int(t / self.m.opt.timestep)):
            self.apply(); mujoco.mj_step(self.m, self.x)
        self.p0 = self.x.xpos[self.bid].copy()

    def apply(self):
        self.x.ctrl[0] = np.radians(self.tgt["yaw"]); self.x.ctrl[1] = np.radians(self.tgt["pitch"]); self.x.ctrl[2] = np.radians(self.tgt["spin"])

    def tilt(self):
        """받침이 기운 각 (도)"""
        zb = self.x.xmat[self.bid].reshape(3, 3)[:, 2]
        return float(np.degrees(np.arccos(np.clip(zb[2], -1, 1))))

    def slide(self):
        return float(np.hypot(*(self.x.xpos[self.bid][:2] - self.p0[:2])) * 1000.0)

    def joint(self, j):
        return float(np.degrees(self.x.qpos[self.jq[j]]))

    def move(self, to, rate=None, hold=0.3, watch=None, force=None):
        """관절 목표를 실제 속도로 옮긴다. to = {관절: 각}. 돌아오는 값: 그동안의 (받침이 가장 많이 기운 각, 가장 많이 미끄러진 거리)"""
        dt = self.m.opt.timestep; start = dict(self.tgt); rate = rate or DEG_S
        T = max(abs(to[j] - start[j]) / rate[j] for j in to) if to else 0.0
        n = int((T + hold) / dt) + 1; mt = 0.0; ms = 0.0
        for k in range(n):
            t = k * dt
            for j in to:
                s = min(1.0, t * rate[j] / max(abs(to[j] - start[j]), 1e-9)); self.tgt[j] = start[j] + (to[j] - start[j]) * s
            self.apply()
            if force is not None:
                force(self, t)
            mujoco.mj_step(self.m, self.x)
            mt = max(mt, self.tilt()); ms = max(ms, self.slide())
            if self.fixed and k % 5 == 0:
                self.log.append(np.array(self.x.sensordata[:6]))
            if watch is not None and watch(self, t):
                break
        return mt, ms

    def wrench(self):
        """받침을 바닥에 붙여 놓았을 때 바닥이 받침을 받치는 힘: (힘이 지나는 점 = 눌림 중심의 x, y (mm), 옆으로 미는 힘 / 누르는 힘, 세로축 둘레로 비트는 힘 N·m) 의 시간 기록"""
        L = np.array(self.log); F = L[:, :3]; Tq = L[:, 3:]
        sgn = -1.0 if np.median(F[:, 2]) > 0 else 1.0                          # 센서의 부호를 '바닥이 받침을 미는 힘'으로 맞춘다 (위로 +)
        F = -sgn * F; Tq = -sgn * Tq
        k = np.ones(20) / 20.0                                               # 0.05초 평균: 받침이 실제로 움직이기에는 너무 짧은 순간 값은 고르게 편다
        F = np.column_stack([np.convolve(F[:, i], k, "same") for i in range(3)]); Tq = np.column_stack([np.convolve(Tq[:, i], k, "same") for i in range(3)])
        F = F[20:-20]; Tq = Tq[20:-20]
        zx = -Tq[:, 1] / F[:, 2] * 1000.0; zy = Tq[:, 0] / F[:, 2] * 1000.0
        return zx, zy, np.hypot(F[:, 0], F[:, 1]) / F[:, 2], Tq[:, 2], F[:, 2]


def scan_test(d):
    s = Sim(d); mt = ms = 0.0; t0 = s.x.time
    for yaw in (-120.0, 0.0, 120.0):
        a, b = s.move(dict(yaw=yaw, pitch=TILT3)); mt = max(mt, a); ms = max(ms, b)
        a, b = s.move(dict(spin=s.tgt["spin"] + 360.0)); mt = max(mt, a); ms = max(ms, b)
    a, b = s.move(dict(yaw=0.0, pitch=0.0)); mt = max(mt, a); ms = max(ms, b)
    err = max(abs(s.joint(j) - s.tgt[j]) for j in ("yaw", "pitch"))
    return dict(tilt_deg=round(mt, 4), slide_mm=round(ms, 3), sim_s=round(s.x.time - t0, 1), joint_err_deg=round(err, 2))


def zmp_test(d):
    """받침을 바닥에 붙여 놓고 재기 한 번 + 갑자기 멈추기를 한다. 바닥이 받치는 힘의 중심(눌림 중심)이 받침 원 안에서 얼마나 움직이는지,
    미끄러지지 않으려면 마찰이 얼마나 필요한지를 본다. 눌림 중심이 받침 끝에 닿으면 넘어가기 시작한다."""
    s = Sim(d, fixed=True); s.log = []
    for yaw in (-120.0, 0.0, 120.0):
        s.move(dict(yaw=yaw, pitch=TILT3)); s.move(dict(spin=s.tgt["spin"] + 360.0))
    s.move(dict(yaw=0.0, pitch=0.0))
    zx, zy, mu_need, tz, fz = s.wrench(); r1 = float(np.max(np.hypot(zx, zy))); mu1 = float(np.max(mu_need)); t1 = float(np.max(np.abs(tz)))
    s.log = []                                                              # 세 관절을 한꺼번에 3배 속도로 돌리다 갑자기 멈추기를 네 번
    fast = {k: v * 3.0 for k, v in DEG_S.items()}
    for k in range(4):
        s.move(dict(yaw=(-170.0, 170.0)[k % 2], pitch=(85.0, -5.0)[k % 2], spin=s.tgt["spin"] + 400.0), rate=fast, hold=0.0)
        for j in ("yaw", "pitch", "spin"):
            s.tgt[j] = s.joint(j)
        s.move({}, hold=0.4)
    zx, zy, mu_need, tz, fz = s.wrench(); r2 = float(np.max(np.hypot(zx, zy)))
    W = sum(d.mass().values()) * 9.81e-3
    return dict(scan=dict(zmp_max_mm=round(r1, 1), margin_mm=round(d.R - r1, 1), mu_need=round(mu1, 4), twist_Nm=round(t1, 4), twist_hold_Nm=round(0.35 * W * d.R * 1e-3 * 2 / 3, 3)),
                abrupt=dict(zmp_max_mm=round(r2, 1), margin_mm=round(d.R - r2, 1), mu_need=round(float(np.max(mu_need)), 4), twist_Nm=round(float(np.max(np.abs(tz))), 4)))


def abrupt_test(d):
    """세 관절을 한꺼번에 가장 빠르게 돌리다가 목표를 그 자리에 딱 멈춘다 (모터가 갑자기 서는 것). 끄덕임 0->85, 받침 -170->170, 판 여러 바퀴"""
    s = Sim(d); s.move(dict(yaw=-170.0))
    fast = {k: v * 3.0 for k, v in DEG_S.items()}                         # 실제보다 3배 빠르게까지
    out = {}
    for name, rate in (("실제 속도", DEG_S), ("3배 속도", fast)):
        s2 = Sim(d); s2.move(dict(yaw=-170.0)); mt = ms = 0.0
        to = dict(yaw=170.0, pitch=85.0, spin=1440.0); dt = s2.m.opt.timestep; start = dict(s2.tgt); t = 0.0; stop_at = [3.0, 8.0, 14.0, 22.0]
        while t < 26.0:
            for j in to:
                s2.tgt[j] = start[j] + np.sign(to[j] - start[j]) * min(abs(to[j] - start[j]), rate[j] * t)
            if any(abs(t - q) < dt / 2 for q in stop_at):                  # 갑자기 서기: 0.3초 동안 목표를 얼린다
                for _ in range(300):
                    s2.apply(); mujoco.mj_step(s2.m, s2.x); mt = max(mt, s2.tilt()); ms = max(ms, s2.slide())
            s2.apply(); mujoco.mj_step(s2.m, s2.x); t += dt
            mt = max(mt, s2.tilt()); ms = max(ms, s2.slide())
        out[name] = dict(tilt_deg=round(mt, 4), slide_mm=round(ms, 3))
    return out


def worst_dir(d, pose):
    m, c = d.com(*pose); return np.arctan2(c[1], c[0])


def slope_test(d, pose, mu, slide_stop=True):
    """바닥을 가장 나쁜 쪽으로 천천히 기울인다 (중력의 방향을 돌린다). 돌아오는 값: 넘어가거나 미끄러지기 시작한 각.
    slide_stop=False 면 미끄러짐은 보지 않고 넘어가는 각만 찾는다 (마찰이 충분하다고 볼 때)."""
    s = Sim(d, mu=mu); s.move(dict(yaw=pose[0], pitch=pose[1], spin=pose[2]), rate={k: v * 4 for k, v in DEG_S.items()}); s.settle(0.3)
    a = worst_dir(d, pose); dt = s.m.opt.timestep; ang = 0.0; kind = None
    while ang < 70.0:
        ang += 1.0 * dt                                                    # 초당 1도
        g = 9.81; th = np.radians(ang); s.m.opt.gravity[:] = [g * np.sin(th) * np.cos(a), g * np.sin(th) * np.sin(a), -g * np.cos(th)]
        s.apply(); mujoco.mj_step(s.m, s.x)
        if s.tilt() > 5.0:
            kind = "넘어감"; break
        if slide_stop and s.slide() > 5.0:
            kind = "미끄러짐"; break
    return dict(angle_deg=round(ang, 1), kind=kind)


def push_test(d, pose, hub=False):
    """가장 나쁜 쪽으로 미는 힘을 천천히 키운다. 마찰은 크게 (미끄러지지 않고 넘어가게).
    hub=False: 판의 가장 높은 귀를 민다 (판이 밀려 돌아갈 수 있다).  hub=True: 같은 높이에서 판 회전축 위의 점을 민다 (판이 돌지 않는다: tipover.py 의 계산과 같은 조건)"""
    s = Sim(d, mu=2.0); s.move(dict(yaw=pose[0], pitch=pose[1], spin=pose[2]), rate={k: v * 4 for k, v in DEG_S.items()}); s.settle(0.3)
    a = worst_dir(d, pose); top = d.top(*pose) * 1e-3; dt = s.m.opt.timestep; F = 0.0
    if hub:
        zc = TM.head_to_world([0, 0, 0], *pose[:2]) * 1e-3; ax = TM.head_to_world([0, 0, 100.0], *pose[:2]) * 1e-3 - zc; ax /= np.linalg.norm(ax)
        top = zc + ax * ((top[2] - zc[2]) / ax[2])
    while F < 12.0:
        F += 0.25 * dt                                                     # 초당 0.25 N
        s.x.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(s.m, s.x, np.array([F * np.cos(a), F * np.sin(a), 0.0]), np.zeros(3), top + (s.x.xpos[s.bid] - s.p0), s.rid, s.x.qfrc_applied)
        s.apply(); mujoco.mj_step(s.m, s.x)
        if s.tilt() > 5.0:
            return dict(force_N=round(F, 2), top_h=round(float(top[2]) * 1000))
    return dict(force_N=None, top_h=round(float(top[2]) * 1000))


def wall_test(d, mu, gap_mm):
    """벽을 수신기 앞에 두고 머리를 85도까지 숙인다. 판 위끝이 벽에 걸리면 끄덕임 모터가 낼 수 있는 힘껏 민다. gap = 받침 가장자리에서 벽까지."""
    wall_x = (BASE[d.size]["R"] + gap_mm) * 1e-3
    s = Sim(d, mu=mu, wall_x=wall_x); s.move(dict(spin=0.0)); s.settle(0.2)
    # 판이 앞(+X)을 보게: 판은 머리 +X 쪽 = 세계 +Y 쪽이므로 판을 -90도 돌려 세계 +X 로 보낸다
    s.move(dict(spin=-90.0), rate={k: v * 3 for k, v in DEG_S.items()})
    wid = mujoco.mj_name2id(s.m, mujoco.mjtObj.mjOBJ_GEOM, "wall"); hit = [False]; s.p0 = s.x.xpos[s.bid].copy()
    def watch(S_, t):
        if not hit[0]:
            hit[0] = any(S_.x.contact[i].geom1 == wid or S_.x.contact[i].geom2 == wid for i in range(S_.x.ncon))
        return S_.tilt() > 25.0
    mt, ms = s.move(dict(pitch=85.0), hold=3.0, watch=watch)
    if not hit[0]:
        result = "벽에 닿지 않음"
    elif mt > 20.0:
        result = "넘어짐"
    elif mt > 1.5:
        result = "받침이 들림 (%.1f도)" % mt
    elif ms > 3.0:
        result = "밀려남 %.0f mm" % ms
    else:
        result = "모터가 멈춤 (끄덕임 %.0f도에서)" % s.joint("pitch")
    return dict(gap_mm=gap_mm, mu=mu, touched=bool(hit[0]), tilt_deg=round(mt, 2), slide_mm=round(ms, 1) if hit[0] else 0.0, reached_pitch=round(s.joint("pitch"), 1), result=result)


if __name__ == "__main__":
    res = {"speed_deg_s": {k: round(v, 2) for k, v in DEG_S.items()}, "torque_Nm": {k: round(v, 4) for k, v in TQ.items()}, "mujoco": mujoco.__version__}
    ST = json.load(open(os.path.join(paths.RES, "tipover.json"), encoding="utf-8"))
    for size, batt in (("S", True), ("S", False), ("L", True)):
        d = TM.Device(size, batt); key = "받침 %s, 배터리 %s" % (size, "있음" if batt else "없음"); r = {}
        skey = [k for k, v in ST.items() if v.get("size") == size and v.get("battery") == batt and v.get("plate_t") == AL_T and v.get("length") == 150.0][0]; pose = tuple(ST[skey]["worst"]["pose"])
        r["scan"] = scan_test(d); print("%s | 재기 한 번 (받침을 그냥 놓고): 받침이 가장 많이 기운 각 %.4f도 (모의 %.0f초, 관절 오차 %.2f도)" % (key, r["scan"]["tilt_deg"], r["scan"]["sim_s"], r["scan"]["joint_err_deg"]), flush=True)
        r["zmp"] = zmp_test(d); z = r["zmp"]
        print("%s | 받침을 붙여 놓고 본 눌림 중심: 재는 동안 가운데에서 가장 멀리 %.1f mm (받침 끝까지 %.1f mm 남음), 필요한 마찰 %.4f, 비트는 힘 %.4f N·m (바닥 마찰 0.35 가 버티는 힘 %.3f)" % (
            key, z["scan"]["zmp_max_mm"], z["scan"]["margin_mm"], z["scan"]["mu_need"], z["scan"]["twist_Nm"], z["scan"]["twist_hold_Nm"]), flush=True)
        r["abrupt"] = abrupt_test(d)
        print("%s | 세 관절을 한꺼번에 돌리다 갑자기 멈춤 (받침을 그냥 놓고): 받침이 기운 각 실제 속도 %.3f도, 3배 속도 %.3f도. 눌림 중심(0.05초 평균)은 가운데에서 %.1f mm 까지" % (
            key, r["abrupt"]["실제 속도"]["tilt_deg"], r["abrupt"]["3배 속도"]["tilt_deg"], z["abrupt"]["zmp_max_mm"]), flush=True)
        r["slope_tip"] = slope_test(d, pose, mu=2.0, slide_stop=False); r["slope_slide"] = slope_test(d, pose, mu=0.35)
        print("%s | 바닥 기울이기: %s %.1f도 (계산 %.1f도). 마찰이 보통(0.35)이면 %s %.1f도" % (key, r["slope_tip"]["kind"], r["slope_tip"]["angle_deg"], ST[skey]["worst"]["tip_deg"], r["slope_slide"]["kind"], r["slope_slide"]["angle_deg"]), flush=True)
        r["push"] = push_test(d, pose); r["push_axis"] = push_test(d, pose, hub=True)
        print("%s | 꼭대기(높이 %d mm) 밀기: 회전축 위를 밀면 %s N 에서 넘어감 (계산 %.2f N). 판 귀를 밀면 %s N (판이 밀려 돌면서)" % (key, r["push"]["top_h"], r["push_axis"]["force_N"], ST[skey]["worst"]["push_top_N"], r["push"]["force_N"]), flush=True)
        r["wall"] = []
        for mu in (0.35, 0.8):
            for gap in (40.0, 80.0, ST["reach"]["150"]["max_mm"] - BASE[size]["R"] + 30.0):     # 마지막 = 판 틀이 닿는 가장 먼 곳 + 3cm
                w = wall_test(d, mu, gap); r["wall"].append(w)
                print("%s | 벽이 받침에서 %d mm, 바닥 마찰 %.2f: %s (받침 기운 각 %.1f도, 미끄러짐 %.0f mm, 끄덕임 %.0f도까지)" % (key, gap, mu, w["result"], w["tilt_deg"], w["slide_mm"], w["reached_pitch"]), flush=True)
        res[key] = r
    json.dump(res, open(os.path.join(paths.RES, "tip_mujoco.json"), "w"), ensure_ascii=False, indent=1)
