#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DeteCam 수신기 혼자 동작 (라즈베리파이에서 실행. 파이썬 3.7 이상, numpy 만 있으면 된다)

노트북 없이 수신기 혼자 찾고, 재고, 계산하고, 결과를 폰에 보여 준다. 네 단계:
  1 탐지   둘레의 Wi-Fi 기기를 훑어 영상을 내보내는 것처럼 보이는 기기를 추린다 (detect.py. 탐지 동글이 없으면 CSI 수로 대신한다)
  2 판별   사람이 멈췄다 움직였다 할 때 그 기기의 전송량이 따라 변하는지 본다 (identify.py). 안 변하면 '의심 기기'로 두고 다음으로 간다
  3 위치   판을 돌려 한 자리의 방향을 재고(analyze3.py), 여러 자리를 합쳐 추정 구간을 낸다 (locate.py)
  4 표시   폰이 수신기 등판의 마커를 보고 수신기가 놓인 곳을 알아낸 뒤(/api/marker), 추정 구간(/api/zone)을 방 안에 그린다
  - 모터: rx3.py 를 불러 움직인다 (모터 박자가 흔들리지 않게 따로 띄운다)
  - CSI : 내장 Wi-Fi 로 받는다 (csi.py). 폰과의 통신은 USB 무선 동글이나 랜선으로 한다
  - 화면: 폰이나 노트북의 브라우저에서 http://<수신기 주소>/  (AR 앱도 같은 주소의 /api/… 를 쓴다)
  - 버튼: 머리 앞 버튼을 짧게 누르면 한 번 잰다 (2초 누르면 꺼지는 것은 파이 설정이 한다)
  - 불  : 초록 불이 천천히 깜빡 = 준비, 빠르게 깜빡 = 재는 중, 켜짐 = 끝, 짧게 번쩍 = 문제

실행
  python3 node.py                 파이에서 (보통은 systemd 가 켠다: install.sh)
  python3 node.py --dry --port 8080       장치 없이 노트북에서 흐름만 본다 (가짜 방, 가짜 카메라). --fast 를 붙이면 기다리는 시간을 20배 줄인다

주소 목록 (모두 JSON)
  GET  /api/status                지금 상태, 설정, 잰 자리들
  GET  /api/zone                  추정 구간 (자리마다 놓인 곳을 알려 준 뒤)
  POST /api/measure  {plan, pose, spot} 한 자리 재기. pose = {x, y, z, heading} (m, 도) 는 나중에 /api/pose 로 줘도 된다. spot = 시험장 모드에서 놓은 곳의 번호
  POST /api/pose     {index, pose}
  POST /api/room     {lo:[x,y,z], hi:[x,y,z]}      찾을 범위 (방의 두 구석). 안 주면 잰 자리 둘레 4m
  POST /api/reset                 잰 자리들을 지우고 새로 시작
  POST /api/exam  {seats, spots, front}            시험장 모드로 바꾼다 (빈 몸이면 끈다). seats = [{id, x, y}] 자리 점, spots = [{name, x, y, z}] 수신기를 놓을 수 있는 곳
                                                   (모서리·벽·교탁 자리, z = 받침대 위 높이), front = 학생이 보는 쪽 [x, y]. 탐지·점검은 몰카와 똑같고, 위치 찾기만 다르다:
                                                   답이 '자리 번호 후보 3개'이고, 다음 자리는 spots 안에서만 고르고, 한 곳에서 18번 잰다
  GET  /api/exam                  시험장 모드의 결과: 자리 후보 3개와 확률, 다음에 놓을 곳 (next.index 를 /api/measure 의 spot 으로 준다)
  POST /api/detect   {channels}                    탐지: 둘레 기기 찾기 (약 30초). 결과는 /api/status 의 survey
  POST /api/watch    {minutes, walk, channels}     점검: 기준선을 잡고 '멈춤'을 누를 때까지 (또는 minutes 분) 감시하며 기기마다 의심 점수와 이유를 낸다 (core/score.py).
                                                   의심 기기가 둘 넘게 애매하면 그때만 '걸으며 확인'을 한다 (walk=false 면 안 한다). 결과는 /api/status 의 watch
  POST /api/mine     {mac, on}                     점검 중에 그 기기를 '내 기기'로 표시 (점수를 매기지 않는다)
  POST /api/identify {mode}                        판별: mode = walk(걸으며, 약 55초) · light(불을 껐다 켜며) · leave(방을 나갔다 들어오며, 약 1분 25초)
                                                   하는 동안 /api/status 의 status.prompt 에 '지금 할 일'과 남은 시간이 나온다. 결과는 ident
  GET  /api/marker                 등판 마커 귀의 좌표 (받침 좌표, m). 지금 관절 각도로 계산한다. 폰이 수신기의 자리를 알아낼 때 쓴다
  POST /api/present  {yaw}                         등판이 위를 보게 머리를 숙인다 (마커를 폰에 보여 주는 자세)
  POST /api/config   {ch, mac, plan}
  POST /api/home {quick} · /api/park · /api/off · /api/time {epoch}
  POST /api/stop                    하던 일을 바로 멈춘다 (모터 전류도 끈다). 바쁠 때에도 받는 유일한 명령
  POST /api/goto     {yaw, pitch}   받침·끄덕임을 그 각도로 (범위: 받침 -174~174, 끄덕임 -5~85)
  POST /api/jog      {joint, deg}   관절 하나를 지금 자리에서 deg 만큼. joint = yaw · pitch · spin
  POST /api/spin     {turns}        판만 돌려 본다 (재지는 않는다)
  POST /api/zero     {joint}        지금 판 자리를 0도로 정한다 (joint = spin 만 된다)
  POST /api/test · /api/release · /api/reboot      모터를 하나씩 조금 움직여 보기 · 모터 전류 끄기 · 다시 켜기
  GET  /api/health                  전압 경고, 온도, 남은 저장 공간, 관절 각도
  GET  /api/export                  지금까지의 결과 전부 (자리, 자리마다의 세기 곡선, 추정 구간, 기기 목록, 판별 결과)를 JSON 한 덩어리로
  GET  /manifest.json · /icon-192.png · /icon-512.png   폰의 '홈 화면에 추가'용
폰 앱(브라우저 화면이든 따로 만든 앱이든)은 위 주소만으로 수신기의 모든 동작을 무선으로 시킬 수 있다.
"""
from __future__ import print_function
import argparse, json, os, shutil, signal, subprocess, sys, tempfile, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def _find(cands, must):
    for c in cands:
        p = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(p, must)):
            return p
    sys.exit("%s 를 찾지 못했습니다 (%s)" % (must, ", ".join(cands)))


for _d in (_find([".", "../core", "../host"], "analyze3.py"), _find([".", "../core"], "locate.py")):
    if _d not in sys.path:
        sys.path.insert(0, _d)
import csi as CSI                      # noqa: E402
import detect as D                     # noqa: E402
import analyze3 as A                   # noqa: E402
import locate as K                     # noqa: E402
import identify as I                   # noqa: E402
import score as SC                     # noqa: E402  (의심 점수: 증거 -> 점수와 이유)
import rx3 as RX                       # noqa: E402  (설정값과 상태 파일의 자리만 쓴다)
from estimator import unit, azel      # noqa: E402
from fusion import origin             # noqa: E402

VERSION = "2026-11-14"
RX3 = os.path.join(HERE, "rx3.py")
TILT3 = 54.7356
PLANS = {"tilt3": 3, "tilt6": 6, "tilt18": 18}      # 한 자리에서 판을 돌리는 횟수 (tilt18 = 시험장 모드)
HOME = os.path.expanduser("~/detecam")
CONF = os.path.join(HOME, "config.json")
DATA = os.path.join(HOME, "data")
SESSION = os.path.join(HOME, "session.json")         # 잰 자리와 방 크기: 다시 켜도 남는다
MANIFEST = dict(name="DeteCam 수신기", short_name="DeteCam", start_url="/", display="standalone", background_color="#0f1218", theme_color="#1b2a57", lang="ko",
                icons=[dict(src="/icon-192.png", sizes="192x192", type="image/png"), dict(src="/icon-512.png", sizes="512x512", type="image/png", purpose="any maskable")])
LED = "/sys/class/leds/led0"
MARKER = os.path.join(HERE, "marker.json")           # 등판 마커의 자리 (cad/marker.py 가 만든다)
BTN_PIN = 3                                          # 전원 버튼 (GPIO3, 5번 핀)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a); sys.stdout.flush()


class Busy(Exception):
    pass


class Stopped(Exception):
    pass


class Node(object):
    def __init__(self, dry=False, fake_cam=(2.4, 3.1, 0.9), fast=False):
        self.dry = dry; self.fake_cam = np.array(fake_cam, dtype=float)
        self.speed = 20.0 if (dry and fast) else 1.0     # 가짜 모드에서 기다리는 시간을 줄일 때
        self.lock = threading.Lock(); self.worker = None
        self.st = dict(state="idle", task="", step="", done=0, total=0, message="준비됐습니다.", since=time.time(), prompt=None)
        self.conf = dict(ch=1, mac="", plan="tilt3")
        if not dry and os.path.exists(CONF):
            try:
                self.conf.update(json.load(open(CONF)))
            except Exception as e:
                log("설정 파일을 읽지 못했습니다:", e)
        self.cap = CSI.Capture(dry=dry)
        self.mon = D.Monitor(dry=dry)                 # 탐지 동글 (없어도 된다)
        self.positions = []                           # 잰 자리들
        self.room = None; self.zone_cache = None
        self.exam = None                              # 시험장 모드: dict(seats, spots, front). None 이면 몰카 찾기 (방 안 어디든)
        self.survey = []; self.detect = dict(method=None, time=None); self.ident = {}; self.ident_last = None
        self.tracker = None; self.watch = None              # 점검(감시): 기기마다 의심 점수 (core/score.py)
        self.homed_this_boot = False; self.dry_clock = 0.0; self.joints = dict(yaw=0.0, pitch=0.0)
        self.marker_geo = json.load(open(MARKER, encoding="utf-8")) if os.path.exists(MARKER) else None
        self.proc = None; self.stopping = False          # 지금 도는 모터 명령, '멈춤'이 눌렸는가
        self.state_file = RX.STATE_FILE                  # rx3.py 가 관절 위치를 적어 두는 파일
        if dry:
            self.room = dict(lo=[0.0, 0.0, 0.0], hi=[3.5, 4.5, 2.4])
            self.state_file = os.path.join(tempfile.gettempdir(), "detecam_dry_%d.json" % os.getpid())     # 가짜 모드도 명령 사이에 위치를 기억한다 (실물과 같은 흐름)
            if os.path.exists(self.state_file):
                os.remove(self.state_file)
        else:
            self.load_session()
        self.led("ready")

    # ---- 상태
    def set(self, **kw):
        with self.lock:
            self.st.update(kw); self.st["since"] = time.time()

    def led(self, mode):
        """초록 불(ACT): ready 천천히, busy 빠르게, done 켜짐, error 짧게 번쩍"""
        if self.dry or not os.path.isdir(LED):
            return
        cmd = {"ready": "echo timer > {0}/trigger; echo 100 > {0}/delay_on; echo 1900 > {0}/delay_off",
               "busy": "echo timer > {0}/trigger; echo 120 > {0}/delay_on; echo 120 > {0}/delay_off",
               "done": "echo none > {0}/trigger; echo 1 > {0}/brightness",
               "error": "echo timer > {0}/trigger; echo 60 > {0}/delay_on; echo 400 > {0}/delay_off"}[mode].format(LED)
        subprocess.call(["sudo", "sh", "-c", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def start(self, name, fn, *a, **kw):
        with self.lock:
            if self.st["state"] == "busy":
                raise Busy("지금 '%s' 을(를) 하는 중입니다." % self.st["task"])
            def run():                                     # 상태는 뒷정리까지 다 끝난 뒤에 바꾼다 (앱이 '끝남'을 보자마자 다음 일을 시켜도 되게)
                self.led("busy")
                try:
                    msg = fn(*a, **kw)
                    if self.stopping:
                        raise Stopped()
                    self.led("done"); self.set(state="idle", step="", prompt=None, message=msg or "끝났습니다.")
                except Exception as e:                     # 무슨 일이 있어도 캡처와 모터 전류는 끈다
                    stopped = self.stopping or isinstance(e, Stopped)
                    if not stopped:
                        log("오류:", repr(e))
                    try:
                        self.stopping = False; self.proc = None
                        self.cap.stop(); self.mon.stop(); self.rx3(["release"], quiet=True); self.led("ready" if stopped else "error")
                    finally:
                        if stopped:                        # 움직이다 끊겼으니 적어 둔 각도를 믿을 수 없다: 다음에 원점부터 다시 잡는다
                            self.homed_this_boot = False
                            self.set(state="idle", step="", prompt=None, message="멈췄습니다. 모터 전류를 껐습니다. 다음에 움직일 때 원점부터 다시 잡습니다.")
                        else:
                            self.set(state="error", step="", prompt=None, message=str(e))
            self.stopping = False
            self.st.update(state="busy", task=name, step="시작", done=0, total=0, message="", since=time.time(), prompt=None)
            self.worker = threading.Thread(target=run); self.worker.daemon = True; self.worker.start()

    # ---- 모터 (rx3.py 를 따로 띄운다)
    def rx3(self, args, on_rec=None, quiet=False, timeout=400):
        if self.stopping and args[0] != "release":
            raise Stopped()
        cmd = [sys.executable, RX3] + list(args) + (["--dry"] if self.dry else [])
        env = dict(os.environ, DETECAM_STATE=self.state_file) if self.dry else None
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, env=env)
        if args[0] != "release":
            self.proc = p
        recs = []; t0 = time.time()
        for line in p.stdout:
            line = line.strip()
            if line.startswith("{"):
                r = json.loads(line); recs.append(r)
                if "yaw" in r and "pitch" in r:
                    self.joints = dict(yaw=float(r["yaw"]), pitch=float(r["pitch"]))
                if on_rec:
                    on_rec(r)
            if time.time() - t0 > timeout:
                p.kill(); raise RuntimeError("모터 동작이 너무 오래 걸려 멈췄습니다.")
        err = p.stderr.read(); rc = p.wait()
        if self.proc is p:
            self.proc = None
        if self.stopping and args[0] != "release":
            raise Stopped()
        if rc != 0 and not quiet:
            raise RuntimeError("모터 명령이 실패했습니다 (rx3.py %s): %s" % (" ".join(args), err.strip()[-300:]))
        return recs

    # ---- 한 자리 재기
    def measure(self, plan=None, pose=None, spot=None):
        plan = plan or (K.EXAM_PLAN if self.exam else self.conf["plan"])
        if plan not in PLANS:
            raise ValueError("모르는 계획입니다: %s" % plan)
        ch, mac = int(self.conf["ch"]), (self.conf["mac"] or "").strip().lower() or None
        self.need_home()
        stamp = time.strftime("%Y%m%d_%H%M%S"); outdir = os.path.join(DATA, "pos_" + stamp + ("_dry" if self.dry else ""))
        if not self.dry:
            os.makedirs(outdir, exist_ok=True)
        pcap = os.path.join(outdir if not self.dry else "/tmp", "csi.pcap")
        n = PLANS[plan]; recs = []
        self.set(step="재는 중", done=0, total=n)
        self.cap.start(ch, mac, pcap)
        def on_rec(r):
            if self.dry:                               # 가짜 시계는 스캔마다 0 에서 시작하므로 겹치지 않게 민다
                r["t"] = [t + self.dry_clock for t in r["t"]]; self._dry_last = r["t"][-1]
            recs.append(r); self.set(done=len(recs), step="재는 중 (%d / %d)" % (len(recs), n))
        try:
            self.rx3(["plan", plan], on_rec)
            if self.dry:
                self.dry_clock = getattr(self, "_dry_last", self.dry_clock) + 5.0
            t, C, mac_used = self.levels(recs, mac, pose)
            if len(t) < 40:
                raise RuntimeError("CSI 패킷이 거의 없습니다 (%d개). 카메라가 영상을 보내는 중인지, 채널(%d)과 주소가 맞는지 확인하세요." % (len(t), ch))
            self.set(step="계산 중")
            res, curves, extra = A.solve(recs, t, C, quiet=True)
            nxt = A.suggest_extra(res, extra, TILT3)
            if nxt:                                    # 한 스캔에서만 골이 보였으면 한 번 더
                self.set(step="한 번 더 재는 중 (받침 %+.0f도)" % nxt[0], total=n + 1)
                self.rx3(["scan", "--yaw", "%.3f" % nxt[0], "--pitch", "%.3f" % nxt[1]], on_rec)
                if self.dry:
                    self.dry_clock = self._dry_last + 5.0
                t, C, mac_used = self.levels(recs, mac, pose)
                res, curves, extra = A.solve(recs, t, C, quiet=True)
        finally:
            self.cap.stop()
        obs, usable = extra if extra else ([], [])
        frames = [(float(recs[i]["yaw"]), float(recs[i]["pitch"])) for i in usable]
        est = None if res["az"] is None else [float(v) for v in unit(res["az"], res["el"])]
        pos = dict(time=stamp, plan=plan, ch=ch, mac=mac_used, packets=int(len(t)), az=res["az"], el=res["el"], n_seen=res["n_seen"], consistent=bool(res.get("consistent", False)),
                   spread=res.get("spread_deg"), scans=res["scans"], frames=frames, est=est,
                   obs=[dict(seen=bool(o["seen"]), psi=float(o["psi"]), depth=float(o["depth"])) for o in obs], pose=self.clean_pose(pose),
                   spot=None if spot is None else int(spot),
                   # 판 각마다의 세기 곡선 (dB, 2도 칸, 못 잰 칸은 null). 실물 곡선을 모형과 견주거나 보정할 때 쓴다: GET /api/export 로 나온다
                   curves=[dict(yaw=float(r["yaw"]), pitch=float(r["pitch"]), bin_deg=float(ph[1] - ph[0]) if len(ph) > 1 else None,
                                lv=[None if not np.isfinite(v) else round(float(v), 2) for v in lv]) for r, (ph, lv) in zip(recs, curves)])
        self.positions.append(pos); self.save_session()
        if not self.dry:
            json.dump(pos, open(os.path.join(outdir, "result.json"), "w"), ensure_ascii=False, indent=1)
            with open(os.path.join(outdir, "motion.jsonl"), "w") as f:
                for r in recs:
                    f.write(json.dumps(r) + "\n")
        if res["az"] is None:
            return "자리 %d: 골이 보이지 않았습니다. 수신기를 옮겨서 다시 재 보세요." % len(self.positions)
        return "자리 %d: 방위 %+.0f도, 높이 %+.0f도%s" % (len(self.positions), res["az"], res["el"], "" if pos["consistent"] else " (스캔끼리 딱 맞지는 않음)")

    def levels(self, recs, mac, pose):
        if not self.dry:
            return CSI.packet_levels(self.cap.packets(), mac)
        p = self.clean_pose(pose) or dict(x=1.0, y=1.0, z=0.0, heading=0.0)       # 가짜 방: 수신기 자리에서 본 가짜 카메라의 방향
        fr = [(r["yaw"], r["pitch"]) for r in recs]
        v = K.to_dev(self.fake_cam[None, :], ((p["x"], p["y"], p["z"]), p["heading"]))[0] - origin(fr)
        az, el = azel(v / np.linalg.norm(v))
        t, C = A.fake_packets(recs, (float(az), float(el)), seed=len(self.positions) * 7 + len(recs))
        return t, C, mac or "fa:ke:00:00:00:01"

    @staticmethod
    def clean_pose(pose):
        if not pose:
            return None
        return dict(x=float(pose["x"]), y=float(pose["y"]), z=float(pose.get("z", 0.0)), heading=float(pose.get("heading", 0.0)))

    # ---- 여러 자리 합치기
    def zone(self):
        if self.zone_cache is not None:
            return self.zone_cache
        S = K.Session()
        if self.room:
            S.set_room(self.room["lo"], self.room["hi"])
        used = []
        for i, p in enumerate(self.positions):
            if p["pose"] and p["obs"]:
                q = p["pose"]; S.add(((q["x"], q["y"], q["z"]), q["heading"]), p["obs"], p["frames"], p["est"]); used.append(i + 1)
        r = S.result(); r["used"] = used; r["waiting_pose"] = [i + 1 for i, p in enumerate(self.positions) if not p["pose"]]
        self.zone_cache = r
        return r

    # ---- 시험장 모드: 자리 후보
    def set_exam(self, q):
        if not q or not q.get("seats"):
            self.exam = None
        else:
            K.ExamSession(q["seats"], q.get("spots") or [], q.get("front") or (0.0, -1.0))          # 잘못된 값이면 여기서 오류
            self.exam = dict(seats=list(q["seats"]), spots=list(q.get("spots") or []), front=list(q.get("front") or (0.0, -1.0)))
        self.save_session()

    def exam_result(self):
        if not self.exam:
            return dict(mode="room", message="시험장 모드가 아닙니다. POST /api/exam 으로 자리와 놓을 곳을 주세요.")
        E = K.ExamSession(self.exam["seats"], self.exam["spots"], self.exam["front"]); used = []
        for i, p in enumerate(self.positions):
            if p.get("pose") and p.get("obs"):
                q = p["pose"]; E.add(((q["x"], q["y"], q["z"]), q["heading"]), p["obs"], p["frames"], p["est"], p.get("spot")); used.append(i + 1)
        r = E.result(); r["used_positions"] = used; r["waiting_pose"] = [i + 1 for i, p in enumerate(self.positions) if not p.get("pose")]
        return r

    # ---- 1 탐지: 둘레 기기 찾기
    def do_detect(self, channels=None, dwell=1.5):
        """탐지 동글이 있으면 그것으로 (프레임 수와 길이), 없으면 내장 Wi-Fi 의 CSI 수로 (초당 몇 개인지만)"""
        if self.dry:
            for k in range(4):
                self.set(step="채널 훑는 중", done=k, total=4); time.sleep(1.0 / self.speed)
                if self.stopping:
                    raise Stopped()
            rows, method = D.fake_rows(), "monitor"
        elif self.mon.refresh():
            rows = D.scan(self.mon, tuple(channels) if channels else None, progress=lambda s, a, b: self.set(step=s, done=a, total=b)); method = "monitor"
        else:
            rows = []; method = "csi"; chans = tuple(channels or D.CH24)
            for k, ch in enumerate(chans):
                self.set(step="채널 %d 듣는 중 (탐지 동글이 없어 내장 Wi-Fi 로)" % ch, done=k, total=len(chans))
                path = "/tmp/detecam_survey.pcap"
                self.cap.start(int(ch), None, path); time.sleep(dwell); self.cap.stop()
                if self.stopping:
                    raise Stopped()
                for r in CSI.mac_table(CSI.read_csi(path))[:8]:
                    m0 = int(r["mac"][:2], 16)
                    if m0 & 1:
                        continue
                    weak = r["rssi"] < D.WEAK_RSSI; rnd = bool(m0 & 2)
                    note = ["길이는 모름"] + (["신호가 약함: 다른 방일 수 있다"] if weak else []) + (["임의 주소 (폰·노트북에 많다)"] if rnd else [])
                    rows.append(dict(mac=r["mac"], role="?", bssid="", ssid="", ch=int(ch), n=r["n"], per_s=r["per_s"], kbps=None, mean_len=None, rssi=r["rssi"],
                                     candidate=bool(r["data_per_s"] >= D.MIN_RATE), weak=weak, random=rnd, heard="csi", note=" · ".join(note), kinds=r["kinds"]))
            best = {}
            for r in rows:                                  # 이웃 채널에서도 들린 기기는 가장 많이 들린 채널 것만 남긴다
                if r["mac"] not in best or r["per_s"] > best[r["mac"]]["per_s"]:
                    best[r["mac"]] = r
            rows = sorted(best.values(), key=lambda r: (not r["candidate"], -r["per_s"]))
        self.survey = rows; self.detect = dict(method=method, time=time.time(), iface=self.mon.iface if method == "monitor" else self.cap.iface)
        n = len([r for r in rows if r["candidate"] and r["role"] != "ap"])
        return "기기 %d개를 찾았습니다. 카메라 후보 %d개%s" % (len(rows), n, "" if method == "monitor" else " (탐지 동글이 없어 초당 개수만 보았습니다)")

    # ---- 2 판별: 움직임에 전송량이 반응하는가
    def do_identify(self, mode="walk", macs=None, ch=None):
        """macs 를 주면 (점검 중 확인 요청) 같은 채널의 여러 기기를 한 번 걸어서 함께 판별한다"""
        if mode not in I.PLANS:
            raise ValueError("모르는 방법입니다: %s" % mode)
        P = I.PLANS[mode]; stim = I.plan_stimulus(mode); n = len(stim)
        ch = int(ch if ch is not None else self.conf["ch"])
        macs = [m.strip().lower() for m in (macs or [self.conf["mac"] or ""]) if m and m.strip()]
        if not macs:
            raise RuntimeError("먼저 확인할 기기를 고르세요 ('기기 찾기' 표에서 줄을 누른다).")
        mac = macs[0]
        use_mon = (not self.dry) and self.mon.refresh()
        if self.dry:
            pass
        elif use_mon:
            D.listen(self.mon, ch)
        else:
            self.cap.start(ch, mac if len(macs) == 1 else None, "/tmp/detecam_ident.pcap")
        t0 = time.time(); ph = I.phases(stim)
        try:
            for k, (a, b, mv) in enumerate(ph):
                end = t0 + b * I.DT / self.speed
                self.led("busy" if mv else "done")
                while True:
                    left = end - time.time()
                    if self.stopping:
                        raise Stopped()
                    if left <= 0:
                        break
                    nxt = P["say"][ph[k + 1][2]] if k + 1 < len(ph) else "끝"
                    self.set(step=P["say"][mv], done=int((time.time() - t0) * self.speed / I.DT), total=n,
                             prompt=dict(move=bool(mv), text=P["say"][mv], left=round(left * self.speed, 1), next=nxt, mode=mode))
                    time.sleep(min(0.2, left))
        finally:
            self.cap.stop(); self.mon.stop(); self.set(prompt=None, step="계산 중")
        frames = D.read_frames(D.PCAP) if (use_mon and not self.dry) else None
        pkts = self.cap.packets() if (not use_mon and not self.dry) else None
        msgs = []
        for mac in macs:
            pk = None; unit = "packets"
            if self.dry:
                x = self.fake_counts(stim, mac)
            elif use_mon:
                tt, nn = D.sent_by(frames, mac); x = I.bins(tt, t0, n, nn); pk = I.bins(tt, t0, n); unit = "bytes"
            else:
                tt = [p[0] for p in pkts if p[1] == mac]; x = I.bins(tt, t0, n)
            r = I.decide(x, stim, pkts=pk, guard=P["guard"])
            r.update(mac=mac, ch=ch, mode=mode, name=P["name"], source="monitor" if use_mon else "csi", unit=unit, time=time.strftime("%Y%m%d_%H%M%S"), dt=I.DT,
                     series=[float(v) for v in x], stim=[int(v) for v in stim])
            self.ident[mac] = r; self.ident_last = mac; msgs.append(r["message"] if len(macs) == 1 else "%s: %s" % (mac, r["message"]))
            if self.tracker is not None:
                self.tracker.walked(mac, r["verdict"])
        return " / ".join(msgs)

    # ---- 점검: 버튼 하나로 감시하며 기기마다 의심 점수를 매긴다 (core/score.py, ai 저장소의 docs/의심 점수.md)
    def do_watch(self, channels=None, minutes=None, walk=True, cycle=None):
        """처음 한 바퀴가 기준선. 그 뒤 '멈춤'을 누를 때까지 (또는 minutes 분) 채널을 돌며 듣고 점수를 고친다.
        의심 기기가 둘 이상 애매하면 그때만 '걸으며 확인'을 한다 (walk=False 면 하지 않는다: 시험장처럼 사람이 걸을 수 없는 곳).
        의심 기기와 그 허브가 있는 채널은 바퀴마다 자세히 듣는다 (몇 초마다 몰아서 보내는 카메라를 놓치지 않으려고)."""
        T = self.tracker = SC.Tracker(); t_end = time.time() + 60.0 * float(minutes) if minutes else None
        extra = []; asked = set(); seen5 = None            # seen5: AP 가 들린 5GHz 채널 (듀얼밴드 동글일 때만 쓴다)
        logf = None
        if not self.dry:                                   # 바퀴마다의 표를 남긴다 (실험 S1~S5 를 host/watch_report.py 로 다시 계산한다)
            try:
                os.makedirs(DATA, exist_ok=True); logf = open(os.path.join(DATA, "watch_%s.jsonl" % time.strftime("%Y%m%d_%H%M%S")), "a")
            except Exception as e:
                log("점검 기록을 열지 못했습니다:", e)

        def check(*a):
            if self.stopping:
                raise Stopped()

        def publish(note=""):
            r = T.result(); r.update(time=time.time(), started=T.t0, note=note, levels=SC.LEVEL_NAME)
            self.watch = r
            return r
        try:
            while True:
                k = T.cycles
                self.set(step="기준선 만드는 중 (둘레 기기를 적어 둔다)" if k == 0 else "감시 %d바퀴째" % (k + 1), done=k, total=0)
                if self.dry:
                    rows = self.fake_watch_rows(k)
                    for _ in range(int(3 * 4)):
                        time.sleep(0.25 / self.speed); check()
                elif self.mon.refresh():
                    chans = tuple(channels) if channels else D.plan(self.mon.channels(), seen5, k)
                    rows = D.scan(self.mon, chans, quick=0.5, focus=cycle or 6.0, n_focus=2, extra=extra,
                                  progress=lambda s, a, b: (check(), self.set(step="감시 %d바퀴째 · %s" % (k + 1, s))))
                else:
                    raise RuntimeError("점검에는 탐지 동글이 필요합니다 (기기마다 길이와 주소 짝을 봐야 한다). 동글을 꽂고 다시 하세요.")
                now = time.time(); T.update(rows, now); self.survey = rows
                seen5 = (seen5 or set()) | D.ap_channels5(rows)
                if logf:
                    logf.write(json.dumps(dict(t=now, rows=rows, walk=dict(T.walk), mine=sorted(T.my)), ensure_ascii=False) + "\n"); logf.flush()
                r = publish()
                extra = sorted(set(d["ch"] for d in r["devices"] if d["level"] in ("camera", "suspect") and d["ch"]))[:3]
                if walk and r["need_confirm"]:
                    todo = [m for m in r["confirm"] if m not in asked]
                    if todo:
                        dev = dict((d["mac"], d) for d in r["devices"]); ch = dev[todo[0]]["ch"]
                        same = [m for m in todo if dev[m]["ch"] == ch]; asked.update(same)
                        publish("의심 기기 %d개를 가리려고 '걸으며 확인'을 합니다." % len(same))
                        self.do_identify("walk", macs=same, ch=ch)
                        publish()
                if t_end and time.time() >= t_end:
                    break
        except Stopped:
            self.stopping = False
        finally:
            if logf:
                logf.close()
        r = publish()
        top = [d for d in r["devices"] if d["level"] in ("camera", "suspect")]
        if not top:
            return "점검을 마쳤습니다 (%d바퀴). 의심할 기기가 없습니다." % r["cycles"]
        return "점검을 마쳤습니다 (%d바퀴). %s" % (r["cycles"], ", ".join("%s%s %s(%d점)" % (d["mac"], " (Wi-Fi 를 만드는 기기)" if d["role"] == "ap" else "", SC.LEVEL_NAME[d["level"]], d["score"]) for d in top[:3]))

    def fake_watch_rows(self, k):
        """가짜 모드의 점검: 처음엔 공유기뿐, 세 바퀴째부터 핫스팟과 몇 초마다 올리는 카메라가 나타난다"""
        hall = dict(mac="fa:ke:00:00:00:a6", role="ap", bssid="fa:ke:00:00:00:a6", ssid="room-507", ch=6, n=0, per_s=0.0, mean_len=0, rssi=-60.0, candidate=False, beacon_only=True, series=[])
        rows = [hall]
        if k >= 2:
            rows.append(dict(mac="fe:ke:00:00:00:b2", role="ap", bssid="fe:ke:00:00:00:b2", ssid="", ch=11, n=0, per_s=0.0, mean_len=0, rssi=-54.0, candidate=False, beacon_only=True, series=[]))
            rows.append(dict(mac="fa:ke:00:00:00:c1", role="sta", bssid="fe:ke:00:00:00:b2", ssid="", ch=11, n=24, per_s=4.0, mean_len=880, rssi=-55.0, candidate=False, beacon_only=False,
                             series=[10 if i % 6 == 0 else 0 for i in range(12)]))
        return rows

    def fake_counts(self, stim, mac):
        """가짜 모드: 주소가 01 로 끝나는 기기만 움직임에 반응하는 카메라인 척한다"""
        rs = np.random.RandomState(len(self.ident) * 13 + 5); n = len(stim); lag = int(1.0 / I.DT)
        s = np.concatenate([np.zeros(lag), stim])[:n]; g = 1.6 if mac.endswith(":01") else 1.0
        return rs.poisson(65.0 * I.DT * (1 + (g - 1) * s) * np.exp(rs.normal(0, 0.12, n))).astype(float)

    # ---- 4 표시: 등판 마커의 자리 (폰이 수신기의 자리를 알아내는 데 쓴다)
    def joints_now(self):
        """지금 관절 각도. 실물에서는 rx3.py 가 적어 둔 상태 파일에서, 가짜 모드에서는 마지막 명령에서"""
        if os.path.exists(self.state_file):
            try:
                d = json.load(open(self.state_file))["pos"]
                return dict((j, d[j] / (RX.HALF_STEPS_PER_REV * RX.RATIO[j] / 360.0)) for j in ("yaw", "pitch"))
            except Exception as e:
                log("관절 상태를 읽지 못했습니다:", e)
        return dict(self.joints)

    def marker(self):
        g = self.marker_geo
        if g is None:
            raise RuntimeError("marker.json 이 없습니다 (cad/marker.py 가 만든 파일을 node.py 옆에 둔다).")
        j = self.joints_now(); k = g["kinematics"]; t, a = np.radians(j["pitch"]), np.radians(j["yaw"])
        def base(p):                                    # 머리 좌표(mm) -> 받침 좌표(m). cad/parts2.py 의 pose_head 와 같은 계산
            x, y, z = p
            f = -(y - k["Y_T"]) * np.cos(t) + (z - k["S_T"]) * np.sin(t); up = k["H_T"] + (y - k["Y_T"]) * np.sin(t) + (z - k["S_T"]) * np.cos(t)
            return [round(float(v) / 1000.0, 5) for v in (f * np.cos(a) - x * np.sin(a), f * np.sin(a) + x * np.cos(a), up)]
        ms = [dict(id=int(i), corners=[base(p) for p in g["corners_head_mm"][str(i)]]) for i in g["ids"]]
        allc = np.array([c for m in ms for c in m["corners"]])
        with self.lock:
            busy = self.st["state"] == "busy"
        return dict(dictionary=g["dictionary"], size_m=g["size_mm"] / 1000.0, yaw=round(j["yaw"], 3), pitch=round(j["pitch"], 3), moving=busy, markers=ms,
                    centre=[round(float(v), 5) for v in allc.mean(0)], normal=[round(float(v), 5) for v in (-np.cos(t) * np.cos(a), -np.cos(t) * np.sin(a), np.sin(t))],
                    corner_order=g["corner_order"],
                    frame="받침 좌표 (m): 원점 = 받침 바닥 한가운데, +x = 받침 앞 표시(0도) 쪽, +z = 위. /api/measure 와 /api/pose 의 pose 가 가리키는 점·방향과 같다")

    def present(self, yaw=None):
        j = self.joints_now(); y = j["yaw"] if yaw is None else float(yaw)
        self.rx3(["goto", "--yaw", "%.3f" % y, "--pitch", "%.4f" % TILT3]); self.joints = dict(yaw=y, pitch=TILT3)
        return "등판이 위를 봅니다. 폰으로 마커를 비추세요."

    # ---- 멈춤: 바쁠 때에도 받는다
    def stop(self):
        with self.lock:
            if self.st["state"] != "busy":
                return "하고 있는 일이 없습니다."
            self.stopping = True; p = self.proc
        if p is not None and p.poll() is None:
            try:
                p.send_signal(signal.SIGTERM)             # rx3.py 가 코일을 끄고 끝난다
            except Exception as e:
                log("모터 명령을 멈추지 못했습니다:", e)
        return "멈추는 중입니다."

    # ---- 손으로 움직이기 (폰에서)
    def goto(self, yaw=None, pitch=None):
        if yaw is None and pitch is None:
            raise ValueError("yaw 나 pitch 를 주세요.")
        self.need_home()
        a = ["goto"] + (["--yaw", "%.3f" % float(yaw)] if yaw is not None else []) + (["--pitch", "%.3f" % float(pitch)] if pitch is not None else [])
        self.rx3(a)
        j = self.joints_now(); return "받침 %+.1f도, 끄덕임 %.1f도" % (j["yaw"], j["pitch"])

    def jog(self, joint, deg):
        if joint not in ("yaw", "pitch", "spin"):
            raise ValueError("모르는 관절입니다: %s (yaw, pitch, spin 가운데 하나)" % joint)
        deg = float(deg)
        if abs(deg) > 360.0:
            raise ValueError("한 번에 360도까지만 움직입니다.")
        if joint != "spin":
            self.need_home()
        self.rx3(["jog", joint, "%.3f" % deg])
        return "%s 을(를) %+.1f도 움직였습니다." % ({"yaw": "받침", "pitch": "끄덕임", "spin": "판"}[joint], deg)

    def need_home(self):
        if not self.homed_this_boot:
            self.set(step="원점 잡기 (켜고 처음 한 번, 40초~1분 반)")
            self.rx3(["home", "--quick"]); self.homed_this_boot = True

    def spin(self, turns=1.0):
        turns = float(turns)
        if not 0.0 < abs(turns) <= 5.0:
            raise ValueError("판은 한 번에 5바퀴까지만 돌립니다.")
        self.rx3(["spin", "--turns", "%.3f" % turns]); return "판을 %.2g바퀴 돌렸습니다." % turns

    def health(self):
        """전압 경고(throttled), 온도, 저장 공간, 관절 각도. 가짜 모드에서는 꾸민 값."""
        out = dict(dry=self.dry, uptime_s=None, temp_c=None, throttled=None, undervoltage_now=None, undervoltage_seen=None, disk_free_mb=None, homed=bool(self.homed_this_boot))
        j = self.joints_now(); out["yaw"] = round(j["yaw"], 2); out["pitch"] = round(j["pitch"], 2); out["spin"] = self.spin_now()
        try:
            out["disk_free_mb"] = int(shutil.disk_usage(HOME if os.path.isdir(HOME) else "/").free / 1e6)
            out["uptime_s"] = int(float(open("/proc/uptime").read().split()[0]))
        except Exception:
            pass
        if self.dry:
            out.update(temp_c=41.0, throttled="0x0", undervoltage_now=False, undervoltage_seen=False)
            return out
        try:
            out["temp_c"] = round(int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000.0, 1)
            t = subprocess.check_output(["vcgencmd", "get_throttled"], universal_newlines=True).strip().split("=")[-1]
            v = int(t, 16); out.update(throttled=t, undervoltage_now=bool(v & 0x1), undervoltage_seen=bool(v & 0x10000))
        except Exception as e:
            out["note"] = "전압 경고를 읽지 못했습니다: %s" % e
        return out

    def spin_now(self):
        if os.path.exists(self.state_file):
            try:
                d = json.load(open(self.state_file))["pos"]
                return round((d["spin"] / (RX.HALF_STEPS_PER_REV / 360.0)) % 360.0, 2)
            except Exception:
                pass
        return None

    def export(self):
        return dict(version=VERSION, time=time.strftime("%Y%m%d_%H%M%S"), config=self.conf, room=self.room, positions=self.positions, zone=self.zone(),
                    exam=self.exam, exam_result=self.exam_result() if self.exam else None, survey=self.survey, detect=self.detect, ident=self.ident, watch=self.watch)

    # ---- 잰 자리와 방 크기는 파일에 남긴다 (다시 켜도 이어서 한다)
    def save_session(self):
        self.zone_cache = None
        if self.dry:
            return
        try:
            os.makedirs(HOME, exist_ok=True); tmp = SESSION + ".tmp"
            json.dump(dict(positions=self.positions, room=self.room, exam=self.exam), open(tmp, "w"), ensure_ascii=False); os.replace(tmp, SESSION)
        except Exception as e:
            log("자리 목록을 저장하지 못했습니다:", e)

    def load_session(self):
        if os.path.exists(SESSION):
            try:
                d = json.load(open(SESSION)); self.positions = list(d.get("positions") or []); self.room = d.get("room"); self.exam = d.get("exam")
                log("지난번에 잰 자리 %d개를 불러왔습니다." % len(self.positions))
            except Exception as e:
                log("자리 목록을 읽지 못했습니다:", e)

    def hint(self):
        """방이 좁으면 '여섯 번 재기'를 권한다 (시뮬레이션: 변기 칸에서 30도 넘게 틀리는 일이 반쯤으로 준다). 시험장 모드면 받침대 높이를 알린다"""
        if self.exam:
            return "시험장 모드: 받침대 위 2.4m 높이에서 재세요 (2.2m 로 낮추면 옆자리까지 맞히는 비율이 9할에서 7할 대로 떨어집니다). 한 곳에서 18번 잽니다 (약 2분)."
        if self.room:
            if min(self.room["hi"][i] - self.room["lo"][i] for i in range(2)) < 2.0:
                return "좁은 곳입니다 (가로나 세로가 2m 아래). 재는 방법을 '여섯 번'으로 하면 크게 틀리는 일이 반쯤으로 줍니다."
        return ""

    def save_conf(self):
        if not self.dry:
            os.makedirs(HOME, exist_ok=True); json.dump(self.conf, open(CONF, "w"), indent=1)

    def status(self):
        with self.lock:
            st = dict(self.st)
        keys = ("time", "plan", "ch", "mac", "packets", "az", "el", "n_seen", "consistent", "spread", "scans", "pose")
        last = self.ident.get(self.ident_last) if self.ident_last else None
        j = self.joints_now()
        return dict(version=VERSION, numpy=np.__version__, python=sys.version.split()[0], dry=self.dry, status=st, config=self.conf, room=self.room, mode="exam" if self.exam else "room", hint=self.hint(), survey=self.survey, detect=self.detect, now=time.time(),
                    joints=dict(yaw=round(j["yaw"], 1), pitch=round(j["pitch"], 1), spin=self.spin_now(), homed=bool(self.homed_this_boot)),
                    ident=dict((m, dict(verdict=r["verdict"], ratio=r["ratio"], mode=r["mode"])) for m, r in self.ident.items()), ident_last=last, watch=self.watch,
                    plans=dict((k, dict(name=p["name"], seconds=len(I.plan_stimulus(k)) * I.DT, say=list(p["say"]))) for k, p in I.PLANS.items()),
                    csi_iface=self.cap.iface, mon_iface=self.mon.iface, mon_driver=self.mon.driver,
                    link_ifaces=[] if self.dry else [d["name"] for d in D.usb_wifi() if d["name"] != self.mon.iface],
                    positions=[dict((k, p.get(k)) for k in keys) for p in self.positions])


# ----------------------------------------------------------------- 버튼 (짧게 누르면 한 번 잰다)
def button_loop(node):
    try:
        import RPi.GPIO as GPIO
    except Exception as e:
        log("버튼을 쓸 수 없습니다 (RPi.GPIO):", e); return
    GPIO.setwarnings(False); GPIO.setmode(GPIO.BCM); GPIO.setup(BTN_PIN, GPIO.IN)
    down = None
    while True:
        lvl = GPIO.input(BTN_PIN)
        if lvl == 0 and down is None:
            down = time.time()
        elif lvl == 1 and down is not None:
            held = time.time() - down; down = None
            if 0.05 <= held <= 1.2:
                try:
                    node.start("재기", node.measure); log("버튼: 재기 시작")
                except Busy:
                    pass
        time.sleep(0.03)


# ----------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    node = None

    def log_message(self, *a):
        pass

    def send(self, obj, code=200, ctype="application/json; charset=utf-8"):
        body = obj if isinstance(obj, bytes) else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        n = self.node; path = self.path.split("?")[0]
        if path in ("/", "/index.html"):
            return self.send(open(os.path.join(HERE, "web.html"), "rb").read(), ctype="text/html; charset=utf-8")
        if path == "/api/status":
            return self.send(n.status())
        if path == "/api/zone":
            return self.send(n.zone())
        if path == "/api/exam":
            return self.send(n.exam_result())
        if path == "/api/marker":
            try:
                return self.send(n.marker())
            except Exception as e:
                return self.send(dict(error=str(e)), 404)
        if path == "/api/health":
            return self.send(n.health())
        if path == "/api/export":
            return self.send(n.export())
        if path == "/manifest.json":
            return self.send(json.dumps(MANIFEST, ensure_ascii=False).encode("utf-8"), ctype="application/manifest+json; charset=utf-8")
        if path in ("/icon-192.png", "/icon-512.png"):
            f = os.path.join(HERE, path[1:])
            if os.path.exists(f):
                return self.send(open(f, "rb").read(), ctype="image/png")
        self.send(dict(error="없는 주소입니다."), 404)

    def do_POST(self):
        n = self.node; path = self.path.split("?")[0]
        try:
            size = int(self.headers.get("Content-Length") or 0)
            q = json.loads(self.rfile.read(size).decode("utf-8")) if size else {}
            if path == "/api/stop":
                return self.send(dict(ok=True, message=n.stop()))
            if path == "/api/measure":
                n.start("재기", n.measure, q.get("plan"), q.get("pose"), q.get("spot"))
            elif path == "/api/goto":
                if q.get("yaw") is None and q.get("pitch") is None:
                    raise ValueError("yaw 나 pitch 를 주세요.")
                for k, (lo, hi) in RX.LIMIT.items():
                    if q.get(k) is not None and not (lo <= float(q[k]) <= hi):
                        raise ValueError("%s %.1f도는 범위(%.0f ~ %.0f) 밖입니다." % ({"yaw": "받침", "pitch": "끄덕임"}[k], float(q[k]), lo, hi))
                n.start("움직이기", n.goto, q.get("yaw"), q.get("pitch"))
            elif path == "/api/jog":
                j = q.get("joint"); d = float(q.get("deg", 0))
                if j in RX.LIMIT:
                    cur = n.joints_now()[j]; lo, hi = RX.LIMIT[j]
                    if not (lo <= cur + d <= hi):
                        raise ValueError("%s 이(가) 범위(%.0f ~ %.0f) 밖으로 나갑니다 (지금 %.1f도)." % ({"yaw": "받침", "pitch": "끄덕임"}[j], lo, hi, cur))
                n.start("움직이기", n.jog, j, d)
            elif path == "/api/spin":
                n.start("판 돌리기", n.spin, q.get("turns", 1.0))
            elif path == "/api/zero":
                if (q.get("joint") or "spin") != "spin":
                    raise ValueError("0도로 정할 수 있는 것은 판(spin)뿐입니다. 받침과 끄덕임은 '원점 다시 잡기'로 맞춥니다.")
                n.start("판 0도 정하기", lambda: n.rx3(["zero", "spin"]) and None)
            elif path == "/api/test":
                n.start("모터 시험", lambda: n.rx3(["test"]) and None)
            elif path == "/api/release":
                n.start("모터 전류 끄기", lambda: n.rx3(["release"]) and None)
            elif path == "/api/reboot":
                def reboot():
                    n.rx3(["park"]); n.joints = dict(yaw=0.0, pitch=0.0)
                    if not n.dry:
                        subprocess.call(["sudo", "reboot"])
                    return "다시 켜는 중입니다. 1분쯤 뒤에 화면을 새로 고치세요."
                n.start("다시 켜기", reboot)
            elif path in ("/api/detect", "/api/survey"):
                n.start("기기 찾기", n.do_detect, q.get("channels"))
            elif path == "/api/watch":
                n.start("점검", n.do_watch, q.get("channels"), q.get("minutes"), q.get("walk", True))
            elif path == "/api/mine":
                if n.tracker is None:
                    raise RuntimeError("점검을 시작한 뒤에 표시할 수 있습니다.")
                n.tracker.mine((q.get("mac") or "").lower(), bool(q.get("on", True)))
                r = n.tracker.result(); r.update(time=time.time(), started=n.tracker.t0, note="", levels=SC.LEVEL_NAME); n.watch = r
                return self.send(dict(ok=True, message="내 기기 표시를 바꿨습니다."))
            elif path == "/api/identify":
                mode = q.get("mode") or "walk"
                if mode not in I.PLANS:
                    raise ValueError("모르는 방법입니다: %s (walk, light, leave 가운데 하나)" % mode)
                n.start("판별", n.do_identify, mode)
            elif path == "/api/present":
                n.start("마커 보여 주기", n.present, q.get("yaw"))
            elif path == "/api/home":
                def home():
                    n.rx3(["home"] + (["--quick"] if q.get("quick") else [])); n.homed_this_boot = True; n.joints = dict(yaw=0.0, pitch=0.0)
                    return "원점을 잡았습니다."
                n.start("원점 잡기", home)
            elif path == "/api/park":
                def park():
                    n.rx3(["park"]); n.joints = dict(yaw=0.0, pitch=0.0)
                n.start("세우기", park)
            elif path == "/api/off":
                n.start("끄기", lambda: n.rx3(["off"]) and None)
            elif path == "/api/config":
                for k in ("ch", "mac", "plan"):
                    if k in q:
                        n.conf[k] = int(q[k]) if k == "ch" else str(q[k]).strip()
                n.save_conf()
            elif path == "/api/pose":
                i = int(q["index"]) - 1
                n.positions[i]["pose"] = n.clean_pose(q.get("pose")); n.save_session()
            elif path == "/api/room":
                n.room = dict(lo=[float(v) for v in q["lo"]], hi=[float(v) for v in q["hi"]]) if q.get("lo") else None; n.save_session()
            elif path == "/api/exam":
                n.set_exam(q)
            elif path == "/api/reset":
                n.positions = []; n.save_session()
            elif path == "/api/delete":
                del n.positions[int(q["index"]) - 1]; n.save_session()
            elif path == "/api/time":
                if not n.dry and abs(float(q["epoch"]) - time.time()) > 60:          # 파이에는 시계 전지가 없다: 폰의 시각으로 맞춘다
                    subprocess.call(["sudo", "date", "-s", "@%d" % int(float(q["epoch"]))], stdout=subprocess.DEVNULL)
            else:
                return self.send(dict(error="없는 주소입니다."), 404)
            self.send(dict(ok=True))
        except Busy as e:
            self.send(dict(error=str(e)), 409)
        except Exception as e:
            self.send(dict(error="%s: %s" % (type(e).__name__, e)), 400)


def main():
    ap = argparse.ArgumentParser(description="DeteCam 수신기 혼자 동작")
    ap.add_argument("--dry", action="store_true", help="장치 없이 흐름만 (가짜 방, 가짜 카메라)")
    ap.add_argument("--fast", action="store_true", help="--dry 와 함께: 기다리는 시간을 20배 줄인다 (검사용)")
    ap.add_argument("--port", type=int, default=80)
    ap.add_argument("--fake-cam", nargs=3, type=float, default=(2.4, 3.1, 0.9), metavar=("X", "Y", "Z"), help="--dry 일 때 가짜 카메라가 있는 곳 (m)")
    a = ap.parse_args()
    node = Node(dry=a.dry, fake_cam=a.fake_cam, fast=a.fast); Handler.node = node
    if not a.dry:
        os.makedirs(DATA, exist_ok=True)
        t = threading.Thread(target=button_loop, args=(node,)); t.daemon = True; t.start()
    try:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    except (PermissionError, OSError) as e:
        log("포트 %d 를 열지 못해 8080 을 씁니다 (%s)" % (a.port, e)); a.port = 8080
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    log("DeteCam 수신기 준비. 브라우저에서 http://<이 장치의 주소>:%d/  (CSI 는 %s, 탐지 동글은 %s%s)" % (a.port, node.cap.iface, node.mon.iface or "없음", ", 가짜 모드" if a.dry else ""))
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.stop(); node.mon.stop()
        if a.dry and os.path.exists(node.state_file):
            os.remove(node.state_file)


if __name__ == "__main__":
    main()
