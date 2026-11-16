#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeteCam 수신기 v2 모터 제어 (라즈베리파이에서 실행, 파이썬 3.7 이상, 추가 설치 없음)

관절 3개
  yaw   받침 회전 (J1)  기어 4 : 1     범위 -174 ~ +174도   드라이버 1  GPIO 12, 16, 20, 21
  pitch 끄덕임   (J2)  기어 11.2 : 1  범위 -5 ~ 85도       드라이버 2  GPIO 17, 27, 22, 23
  spin  판 회전  (J3)  모터 축 그대로  계속 돈다            드라이버 3  GPIO 6, 13, 19, 26 (DiffLoc 과 같은 핀)

자주 쓰는 명령
  python3 rx3.py test                      모터를 하나씩 조금 움직여 도는 방향을 확인한다 (처음 한 번)
  python3 rx3.py jog spin 5                관절 하나를 5도 움직인다 (판 0도 표시를 맞출 때)
  python3 rx3.py zero spin                 지금 판 위치를 0도로 정한다 (막대의 표시와 모터 자리의 홈을 맞춘 뒤)
  python3 rx3.py home                      원점 잡기: 끄덕임과 받침을 - 쪽 멈춤턱까지 밀었다가 0도로 돌아온다 (약 1분 반)
  python3 rx3.py home --quick              기억한 위치를 믿고 짧게 (약 40초). 지난번 동작이 중간에 끊겼으면 알아서 길게 한다
  python3 rx3.py goto --yaw 0 --pitch 54.74
  python3 rx3.py scan --yaw -120 --pitch 54.74      자세를 잡고 판을 한 바퀴. 시간표를 JSON 한 줄로 출력 (노트북의 scan3.py 가 읽는다)
  python3 rx3.py plan tilt3                세 번 재기를 한 번에 (JSON 여러 줄). tilt6 = 좁은 방용 여섯 번 재기, tilt18 = 시험장 (교실 둘레) 열여덟 번
  python3 rx3.py park                      세운 자세(0, 0)로 돌아가고 전류를 끈다
  python3 rx3.py off                       세운 자세로 돌아간 뒤 파이를 안전하게 끈다 (머리 앞 전원 버튼을 2초 누른 것과 같다)
  python3 rx3.py status
모든 명령에 --dry 를 붙이면 GPIO 를 건드리지 않고 걸리는 시간만 계산한다 (노트북에서 미리 볼 때).

각도 약속 (추정 코드 core/estimator.py 와 같다)
  yaw   위에서 볼 때 반시계가 +, 받침 덮개의 앞 표시가 0도
  pitch 머리가 앞으로 숙는 쪽이 +, 똑바로 선 것이 0도
  spin  머리 위에서 볼 때 반시계가 +, 판이 머리 오른쪽(부채꼴 기어 쪽)에 있을 때가 0도
모터가 반대로 돌면 전선을 바꾸지 말고 아래 DIR 의 부호만 바꾼다.
"""
from __future__ import print_function
import argparse, json, os, signal, sys, time

# ----------------------------------------------------------------- 설정
PINS = {"yaw": (12, 16, 20, 21), "pitch": (17, 27, 22, 23), "spin": (6, 13, 19, 26)}     # IN1..IN4 (BCM 번호)
DIR = {"yaw": +1, "pitch": +1, "spin": +1}          # [확인] test 명령으로 보고, 반대로 돌면 -1
HALF_STEPS_PER_REV = 4076                            # [확인] 28BYJ-48 모터 축 한 바퀴 (감속 63.68:1). 열 바퀴 돌려 표시가 어긋나면 4096 으로
RATIO = {"yaw": 4.0, "pitch": 11.2, "spin": 1.0}     # 모터 축이 몇 바퀴 돌아야 관절이 한 바퀴 도는가
LIMIT = {"yaw": (-174.0, 174.0), "pitch": (-5.0, 85.0)}     # 명령으로 갈 수 있는 범위. 끄덕임 85도 위(눕힘)는 판과 받침 사이가 2~3mm 라 막아 둔다
STOP = {"yaw": (-175.4, 175.4), "pitch": (-5.3, 86.9)}      # 멈춤턱에 실제로 닿는 각 (모델에서 잰 값). 원점은 - 쪽 턱에서 잡는다
# 반 스텝 하나에 걸리는 시간 (초). 느릴수록 힘이 세다: 이 모터의 힘 34mN·m 는 3.5바퀴/분(반 스텝 4.2ms)에서 잰 값이고,
# 1.5ms(약 10바퀴/분)에서는 어림으로 그 55~75% 까지 떨어진다 (사양서의 한계 속도에서 이은 값, 재 본 것은 아니다).
# 그래서 힘이 드는 끄덕임과 받침은 늦추고, 가벼운 판만 DiffLoc 과 같은 1.5ms 로 둔다. 모터가 놓치면(드르륵) 숫자를 더 키운다.
STEP_S = {"yaw": 0.0020, "pitch": 0.0030, "spin": 0.0015}
FULL_STEP = {"yaw": False, "pitch": True, "spin": False}      # 끄덕임은 힘이 센 2상 구동 (한 번에 반 스텝 2칸)
RAMP_STEPS = 80                                      # 처음과 끝 80칸을 천천히 (출발할 때 탈조 방지)
# 관절마다의 (천천히 가는 칸 수, 처음 속도를 몇 배 늦추나). 판은 알루미늄 판과 균형추가 무거워서(관성이 커서) 더 길게, 더 느리게 출발한다
RAMP = {"yaw": (RAMP_STEPS, 2.0), "pitch": (RAMP_STEPS, 2.0), "spin": (240, 4.0)}
DRY_SLEEP = float(os.environ.get("DETECAM_DRY_SLEEP", "0") or 0)       # 검사용: --dry 에서도 (걸리는 시간 x 이 값) 만큼 실제로 기다린다
YAW_BACKLASH_DEG = 3.0                               # 받침은 늘 + 방향으로 마무리해서 기어 틈(약 2도)을 한쪽으로 몬다
TILT3 = 54.7356                                      # 이 각으로 기울이고 받침을 120도씩 돌리면 세 회전축이 서로 직각
LOG_EVERY = 16                                       # 판을 돌릴 때 반 스텝 16칸(약 1.4도)마다 시각을 적는다
STATE_FILE = os.environ.get("DETECAM_STATE") or os.path.expanduser("~/.detecam_rx3.json")      # 검사할 때는 DETECAM_STATE 로 다른 파일을 쓴다
PLANS = {
    "tilt3": [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)],
    "up+tilt3": [(-120.0, 0.0), (-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)],
    "tilt4": [(-135.0, 60.0), (-45.0, 60.0), (45.0, 60.0), (135.0, 60.0)],
    "tilt6": [(-150.0, TILT3), (-90.0, TILT3), (-30.0, TILT3), (30.0, TILT3), (90.0, TILT3), (150.0, TILT3)],     # 화장실처럼 좁고 반사가 센 방: 받침을 60도씩 여섯 번
    "up": [(0.0, 0.0)],
    # 시험장 (교실 둘레에서 잴 때): 숙임 30·54.7·78도 x 받침 6방향 = 18번. 숙임마다 받침을 돌리는 쪽을 바꿔 덜 돈다 (sim/location/perimeter2.py)
    "tilt18": [(y, p) for i, p in enumerate((30.0, TILT3, 78.0)) for y in ((-150.0, -90.0, -30.0, 30.0, 90.0, 150.0) if i % 2 == 0 else (150.0, 90.0, 30.0, -30.0, -90.0, -150.0))],
}
HALF_SEQ = ((1, 0, 0, 0), (1, 1, 0, 0), (0, 1, 0, 0), (0, 1, 1, 0), (0, 0, 1, 0), (0, 0, 1, 1), (0, 0, 0, 1), (1, 0, 0, 1))


def log(*a):
    print(*a, file=sys.stderr); sys.stderr.flush()


# ----------------------------------------------------------------- GPIO (없으면 흉내만)
class FakeGPIO(object):
    BCM = OUT = 0
    def setmode(self, *a): pass
    def setwarnings(self, *a): pass
    def setup(self, *a): pass
    def output(self, *a): pass
    def cleanup(self, *a): pass


class Rig(object):
    def __init__(self, dry=False):
        self.dry = dry
        self.clock = 0.0                                 # --dry 일 때 쓰는 가짜 시계
        if dry:
            self.gpio = FakeGPIO()
        else:
            try:
                import RPi.GPIO as GPIO
            except Exception as e:                       # 노트북에서 --dry 없이 돌렸을 때
                sys.exit("RPi.GPIO 를 쓸 수 없습니다 (%s). 파이에서 실행하거나 --dry 를 붙이세요." % e)
            self.gpio = GPIO
        g = self.gpio
        g.setwarnings(False); g.setmode(g.BCM)
        for pins in PINS.values():
            for p in pins:
                g.setup(p, g.OUT); g.output(p, 0)
        self.state = self.load()

    # ---- 저장해 두는 상태: 관절마다 반 스텝 수(pos)와 코일 위상(phase)
    def load(self):
        st = {"pos": {"yaw": 0, "pitch": 0, "spin": 0}, "phase": {"yaw": 0, "pitch": 1, "spin": 0}, "homed": False, "dirty": False}
        if os.path.exists(STATE_FILE) and not (self.dry and not os.environ.get("DETECAM_STATE")):
            try:
                d = json.load(open(STATE_FILE))
                for k in ("pos", "phase"):
                    st[k].update({j: int(v) for j, v in d.get(k, {}).items() if j in st[k]})
                st["homed"] = bool(d.get("homed", False))
                st["dirty"] = bool(d.get("dirty", False))       # 지난번 동작이 끝나기 전에 끊겼다 = 적힌 위치를 믿을 수 없다
            except Exception as e:
                log("상태 파일을 읽지 못해 0에서 시작합니다:", e)
        return st

    def save(self):
        if self.dry and not os.environ.get("DETECAM_STATE"):      # --dry 는 보통 아무것도 남기지 않는다. 검사용 상태 파일을 따로 준 때만 적는다
            return
        tmp = STATE_FILE + ".tmp"
        json.dump(self.state, open(tmp, "w")); os.rename(tmp, STATE_FILE)

    # ---- 단위 바꾸기
    def per_deg(self, j):
        return HALF_STEPS_PER_REV * RATIO[j] / 360.0

    def deg(self, j):
        return self.state["pos"][j] / self.per_deg(j)

    def now(self):
        return self.clock if self.dry else time.time()

    def wait(self, s):
        if self.dry:
            self.clock += s
            if DRY_SLEEP > 0:
                time.sleep(s * DRY_SLEEP)
        else:
            time.sleep(s)

    def coils(self, j, phase):
        pins = PINS[j]; seq = HALF_SEQ[phase % 8]
        for k in range(4):
            self.gpio.output(pins[k], seq[k])

    def release(self, j=None):
        for name in ([j] if j else PINS):
            for p in PINS[name]:
                self.gpio.output(p, 0)

    # ---- 반 스텝 n 칸 움직이기 (n 의 부호 = 관절 각이 커지는 쪽이 +)
    def step(self, j, n, ramp=True, stamp=None):
        """stamp 를 주면 LOG_EVERY 칸마다 (지금까지 간 칸 수, 시각) 을 적는다."""
        if n == 0:
            return
        unit = 2 if FULL_STEP[j] else 1
        count = abs(int(n)) // unit
        sgn = 1 if n > 0 else -1
        d = sgn * DIR[j] * unit
        base = STEP_S[j] * unit
        ph = self.state["phase"][j]
        if FULL_STEP[j] and ph % 2 == 0:
            ph += 1                                      # 2상 구동은 홀수 위상만 쓴다
        if j != "spin":
            self.state["dirty"] = True; self.save()      # 움직이는 동안 전원이 끊기면 다음 원점 잡기를 길게 하도록 표시해 둔다
        self.coils(j, ph); self.wait(0.02)               # 코일을 켜고 잠깐 자리 잡기
        if stamp is not None:
            stamp.append((0, self.now()))
        for i in range(count):
            ph = (ph + d) % 8
            self.coils(j, ph)
            dt = base
            if ramp:
                edge = min(i, count - 1 - i); rn, slow = RAMP[j]
                if edge < rn:
                    dt = base * (slow - (slow - 1.0) * edge / float(rn))      # 양 끝은 slow 배 느리게 시작해서 차츰 빨라진다
            self.wait(dt)
            if stamp is not None and ((i + 1) * unit) % LOG_EVERY == 0:
                stamp.append(((i + 1) * unit, self.now()))
        if stamp is not None and stamp[-1][0] != count * unit:
            stamp.append((count * unit, self.now()))
        self.state["phase"][j] = ph
        self.state["pos"][j] += sgn * count * unit
        self.wait(0.05)
        self.release(j)
        if j != "spin":
            self.state["dirty"] = False
        self.save()

    def move_to(self, j, target_deg, check=True):
        if check and j in LIMIT:
            lo, hi = LIMIT[j]
            if not (lo - 1e-6 <= target_deg <= hi + 1e-6):
                sys.exit("%s %.1f도는 범위(%.0f ~ %.0f) 밖입니다." % (j, target_deg, lo, hi))
        n = int(round(target_deg * self.per_deg(j))) - self.state["pos"][j]
        if j == "yaw" and n < 0 and YAW_BACKLASH_DEG > 0:              # - 방향으로 갈 때는 조금 지나쳤다가 + 로 돌아온다
            over = int(round(YAW_BACKLASH_DEG * self.per_deg(j)))
            lo_steps = int(round(LIMIT["yaw"][0] * self.per_deg(j)))
            over = max(0, min(over, self.state["pos"][j] + n - lo_steps))
            self.step(j, n - over); self.step(j, over)
        else:
            self.step(j, n)

    def goto(self, yaw=None, pitch=None):
        if yaw is not None:
            self.move_to("yaw", yaw)
        if pitch is not None:
            self.move_to("pitch", pitch)

    # ---- 원점 잡기: 멈춤턱에 닿을 만큼 넉넉히 민다 (턱에 닿으면 모터가 '드르륵' 하며 헛돈다. 28BYJ-48 은 그래도 괜찮다)
    def home(self, quick=False):
        if quick and (self.state.get("dirty") or not self.state["homed"]):
            log("기억한 위치를 믿을 수 없어 길게 원점을 잡습니다.")
            quick = False
        for j, margin in (("pitch", 8.0), ("yaw", 10.0)):
            lo, hi = STOP[j]
            known = self.state["homed"] and quick
            travel = (self.deg(j) - lo if known else hi - lo) + margin
            log("%s 원점: - 쪽으로 %.0f도 민다 (약 %.0f초)" % (j, travel, travel * self.per_deg(j) * STEP_S[j]))
            self.step(j, -int(round(travel * self.per_deg(j))))
            self.state["pos"][j] = int(round(lo * self.per_deg(j)))
            self.save()
            self.move_to(j, 0.0, check=False)
        self.state["homed"] = True; self.save()
        log("원점 잡기 끝. 받침 0도, 끄덕임 0도.")

    # ---- 판 한 바퀴 (재는 동작)
    def spin_once(self, turns=1.0):
        n = int(round(HALF_STEPS_PER_REV * turns))
        start = self.deg("spin")
        stamp = []
        self.step("spin", n, ramp=True, stamp=stamp)     # 판과 균형추가 무거워서 천천히 출발한다. 각도는 적어 둔 시각으로 계산하므로 속도가 변해도 된다
        return {"spin0": round(start % 360.0, 3), "steps": n, "per_rev": HALF_STEPS_PER_REV,
                "k": [s for s, _ in stamp], "t": [round(t, 6) for _, t in stamp]}

    def scan(self, yaw, pitch, settle=0.4):
        self.goto(yaw, pitch)
        self.wait(settle)
        rec = self.spin_once(1.0)
        rec.update({"kind": "scan", "yaw": round(self.deg("yaw"), 3), "pitch": round(self.deg("pitch"), 3), "dry": self.dry})
        return rec


# ----------------------------------------------------------------- 명령
def main():
    ap = argparse.ArgumentParser(description="DeteCam 수신기 v2 모터 제어")
    ap.add_argument("cmd", choices=["test", "jog", "zero", "home", "goto", "spin", "scan", "plan", "park", "off", "status", "release"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("--yaw", type=float); ap.add_argument("--pitch", type=float)
    ap.add_argument("--turns", type=float, default=1.0)
    ap.add_argument("--quick", action="store_true", help="home: 기억한 위치를 믿고 짧게 민다")
    ap.add_argument("--dry", action="store_true", help="GPIO 를 건드리지 않고 시간만 계산")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, lambda *x: sys.exit(143))      # '멈춤'으로 죽일 때도 아래의 finally 가 돌아 코일을 끈다
    rig = Rig(dry=a.dry)
    t0 = rig.now()
    try:
        if a.cmd == "status":
            print(json.dumps({"yaw": round(rig.deg("yaw"), 2), "pitch": round(rig.deg("pitch"), 2), "spin": round(rig.deg("spin") % 360, 2), "homed": rig.state["homed"], "dirty": rig.state["dirty"]}))
        elif a.cmd == "test":
            for j, d in (("spin", 20.0), ("yaw", 10.0), ("pitch", 5.0)):
                log("%s 를 +%.0f도 갔다가 돌아옵니다. 설명서의 + 방향과 같은지 보세요." % (j, d))
                rig.step(j, int(round(d * rig.per_deg(j)))); rig.wait(0.8); rig.step(j, -int(round(d * rig.per_deg(j)))); rig.wait(0.8)
        elif a.cmd == "jog":
            j, d = a.args[0], float(a.args[1])
            if j in LIMIT:                                   # 받침과 끄덕임은 범위 안에서만 (판은 얼마든지 돈다)
                rig.move_to(j, rig.deg(j) + d)
            else:
                rig.step(j, int(round(d * rig.per_deg(j))))
            print(json.dumps({"kind": "jog", "yaw": round(rig.deg("yaw"), 3), "pitch": round(rig.deg("pitch"), 3), "spin": round(rig.deg("spin") % 360, 3)}))
        elif a.cmd == "zero":
            j = a.args[0] if a.args else "spin"
            rig.state["pos"][j] = 0; rig.save(); log("%s 의 지금 위치를 0도로 정했습니다." % j)
        elif a.cmd == "home":
            rig.home(quick=a.quick)
        elif a.cmd == "goto":
            rig.goto(a.yaw, a.pitch)
            print(json.dumps({"kind": "goto", "yaw": round(rig.deg("yaw"), 3), "pitch": round(rig.deg("pitch"), 3)}))
        elif a.cmd == "spin":
            print(json.dumps(rig.spin_once(a.turns)))
        elif a.cmd == "scan":
            print(json.dumps(rig.scan(rig.deg("yaw") if a.yaw is None else a.yaw, rig.deg("pitch") if a.pitch is None else a.pitch)))
        elif a.cmd == "plan":
            name = a.args[0] if a.args else "tilt3"
            frames = PLANS[name]
            if abs(rig.deg("yaw") - frames[-1][0]) < abs(rig.deg("yaw") - frames[0][0]) and name != "up+tilt3":
                frames = frames[::-1]                    # 가까운 쪽 끝에서 시작한다 (받침을 덜 돌린다)
            for yaw, pitch in frames:
                print(json.dumps(rig.scan(yaw, pitch))); sys.stdout.flush()
        elif a.cmd == "park":
            rig.goto(0.0, 0.0); rig.release()
        elif a.cmd == "off":
            rig.goto(0.0, 0.0); rig.release()
            log("세운 자세로 돌아왔습니다. 파이를 끕니다. 초록 불이 10번 깜빡이고 꺼지면 끝난 것입니다.")
            if not a.dry:
                os.system("sudo poweroff")
        elif a.cmd == "release":
            rig.release()
    finally:
        rig.release()                                    # 코일을 모두 끈다. 핀은 출력(0)인 채로 둔다 (입력으로 되돌리면 드라이버 입력이 떠서 코일이 살짝 켜질 수 있다)
    if a.dry:
        log("(--dry) 걸리는 시간 약 %.1f초" % (rig.now() - t0))


if __name__ == "__main__":
    main()
