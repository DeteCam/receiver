# -*- coding: utf-8 -*-
"""의심 점수: 둘레 기기마다 증거를 쌓아 점수와 이유를 낸다 (docs/의심 점수.md 4절).
버튼 하나('점검 시작')로 숙소의 몰카와 시험장의 부정행위 카메라를 같은 흐름으로 찾는다. 숙소인지 시험장인지 고르지 않는다:
환경이 다르면 쌓이는 증거가 달라 결과가 달라진다. 점수는 손으로 정한 시작값이고 실물 실험(S1~S5)으로 고친다.
수신기(파이)와 시험이 같은 파일을 쓴다. 파이썬 3.7, 표준 라이브러리만.

쓰는 법
  T = Tracker()                          # 점검 하나
  T.update(rows, now)                    # detect.analyse() 가 낸 표를 감시 한 바퀴마다 넣는다. 첫 바퀴가 기준선이 된다
  T.walked(mac, verdict)                 # 판별(걸으며 확인) 결과: 'camera' / 'suspect' / 'quiet'
  T.mine(mac, True)                      # 사용자가 '내 기기'로 표시
  r = T.result()                         # 기기마다 점수·등급·이유, 확인 요청이 필요한지, 울린 기록(alarms)
  기록을 다시 돌리기: host/watch_report.py 가 점검 기록(jsonl)을 이 파일로 다시 계산한다 (규칙을 고친 뒤 같은 실험 자료로 견줄 때)

증거 (점수는 시작값)
  새로 나타남        기준선(첫 바퀴) 뒤에 처음 들린 기기                                   +2
  올려 보내는 양      초당 30개 넘게, 평균 300바이트 넘게 (detect.py 의 '카메라 후보')       +2
  송출 패턴          꾸준히 +1 / 일정 주기로 몰아서 +2 / 가끔 터짐 +1                      (0.5초 칸의 개수로 가린다)
  걷기에 반응        판별 결과가 'camera'                                                +3
  수상한 곳에 붙음    붙은 Wi-Fi 가 개인 핫스팟처럼 보이거나(임의 주소) 이름을 숨겼거나 새로 생김  +1
  카메라 같은 이름    붙은 Wi-Fi (또는 자기가 만든 Wi-Fi) 의 이름이 카메라 앱 계열              +1
  가까움 / 멂        세기 -60dBm 넘게 +1, -75dBm 아래 -1
  내 기기            점수를 매기지 않는다
등급: 5 이상 '카메라 가능성 높음', 3~4 '의심 기기', 2 이하 '무시'.
허브: 의심 기기가 붙은 Wi-Fi(핫스팟·공유기)는 그 기기의 등급까지 올려 같이 보여 준다. 허브는 비컨을 늘 보내서 카메라가 조용할 때도 보인다.
확인 요청: 걷기를 안 한 '의심 기기'(3~4점, 올려 보내는 기기)가 둘 이상이면 '걸으며 확인'을 하자고 한다. 시험장처럼 기기가 거의 없으면 요청하지 않는다.
"""
import re

DT = 0.5
LV_CAMERA, LV_SUSPECT = 5, 3
NEAR, FAR = -60.0, -75.0
# 카메라 앱·핫스팟 카메라의 Wi-Fi 이름 (docs/비콘 검토.md 4절의 설명서 예시). 대소문자는 가리지 않는다
CAM_NAMES = [r"^DDD-", r"^A9_", r"HDWIFICAM", r"HIDVCAM", r"^IPC[-_]", r"^CAM[-_]", r"CAMERA", r"^MINI[-_ ]?CAM", r"^IPCAM", r"^SPY", r"^HD[-_]?CAM", r"^V380", r"^LOOKCAM", r"^YI[-_]"]
# 개인 핫스팟에 흔한 이름 (폰이 만든 Wi-Fi)
HOTSPOT_NAMES = [r"IPHONE", r"GALAXY", r"ANDROIDAP", r"ANDROID_?HOTSPOT", r"^SM-", r"HOTSPOT", r"핫스팟", r"\bAP\b"]
PATTERN_NAME = {"steady": "꾸준히 보냄", "periodic": "일정 주기로 몰아서 보냄", "burst": "가끔 터지듯 보냄", "quiet": "거의 안 보냄", "none": "데이터 없음 (비컨만)"}


def _match(names, s):
    s = (s or "").upper()
    return any(re.search(p, s) for p in names)


def is_random(mac):
    try:
        return bool(int(mac.split(":")[0], 16) & 2)
    except Exception:
        return False


def pattern(windows):
    """windows = 들은 구간마다의 0.5초 칸 패킷 수 목록들. (꾸준함 'steady' / 주기 'periodic' / 터짐 'burst' / 'quiet', 주기 초)
    꾸준함 = 패킷이 있는 칸이 70% 넘음. 주기 = 몰아서 오는 덩어리의 간격이 고름 (흔들림 35% 아래, 간격이 3개 이상). 나머지는 터짐."""
    cells = [c for w in windows for c in w]
    if not cells:
        return "none", None
    rate = sum(cells) / (len(cells) * DT)
    if rate < 0.5:
        return "quiet", None
    duty = sum(1 for c in cells if c > 0) / float(len(cells))
    if duty >= 0.7:
        return "steady", None
    gaps = []
    for w in windows:                                     # 구간 안에서 덩어리가 시작하는 칸 사이의 간격
        starts = [i for i, c in enumerate(w) if c > 0 and (i == 0 or w[i - 1] == 0)]
        gaps += [(b - a) * DT for a, b in zip(starts, starts[1:])]
    if len(gaps) >= 3:
        m = sum(gaps) / len(gaps)
        sd = (sum((g - m) ** 2 for g in gaps) / len(gaps)) ** 0.5
        if m > 0 and sd / m < 0.35:
            return "periodic", round(m, 1)
    return "burst", None


class Tracker(object):
    def __init__(self):
        self.dev = {}; self.base = None; self.aps_base = set(); self.cycles = 0; self.t0 = None
        self.walk = {}; self.my = set(); self.events = []; self.alarms = []; self.alarmed = {}

    # ---- 감시 한 바퀴의 표를 넣는다
    def update(self, rows, now):
        if self.t0 is None:
            self.t0 = now
        self.last_t = now
        first = self.base is None
        for r in rows:
            mac = r["mac"]; d = self.dev.get(mac)
            if d is None:
                d = self.dev[mac] = dict(mac=mac, role=r.get("role"), first=now, cycles=0, windows=[], per_s=0.0, mean_len=0, rssi=None, candidate=False,
                                         bssid=r.get("bssid") or "", ssid=r.get("ssid") or "", ch=r.get("ch"), beacon_only=bool(r.get("beacon_only")))
                if not first:
                    self.events.append(dict(t=now, mac=mac, what="새로 나타남"))
            d["last"] = now; d["cycles"] += 1
            for k in ("role", "ch"):
                if r.get(k) is not None:
                    d[k] = r[k]
            if r.get("bssid"):
                d["bssid"] = r["bssid"]
            if r.get("ssid"):
                d["ssid"] = r["ssid"]
            if not r.get("beacon_only"):
                d["beacon_only"] = False
            if r.get("per_s") is not None and r.get("n"):
                d["per_s"] = max(d["per_s"], float(r["per_s"])); d["mean_len"] = max(d["mean_len"] or 0, int(r.get("mean_len") or 0))
            if r.get("rssi") is not None:
                d["rssi"] = r["rssi"] if d["rssi"] is None else round(0.7 * d["rssi"] + 0.3 * r["rssi"], 1)
            d["candidate"] = d["candidate"] or bool(r.get("candidate"))
            if r.get("series"):
                d["windows"] = (d["windows"] + [list(r["series"])])[-12:]
        if first:
            self.base = set(self.dev)
            self.aps_base = set(m for m, d in self.dev.items() if d["role"] == "ap")
        self.cycles += 1
        self._alarms(now)

    def _alarms(self, now):
        """기기가 처음 '의심 기기'나 '카메라 가능성 높음'에 오른 때를 적는다 (시험장에서 '한 시간에 잘못 울리는 횟수'를 셀 때 쓴다)"""
        for r in self.result(_alarms=False)["devices"]:
            lv = r["level"]
            if lv in ("camera", "suspect") and _rank(lv) > _rank(self.alarmed.get(r["mac"], "ignore")):
                self.alarmed[r["mac"]] = lv
                self.alarms.append(dict(t=now, mac=r["mac"], level=lv, score=r["score"], role=r["role"], reasons=r["reasons"]))

    def walked(self, mac, verdict):
        self.walk[mac] = verdict
        if self.t0 is not None:
            self._alarms(self.last_t)

    def mine(self, mac, on=True):
        (self.my.add if on else self.my.discard)(mac)

    # ---- 증거와 점수
    def evidence(self, d):
        ev = []
        if self.base is not None and d["mac"] not in self.base:
            ev.append(("new", 2, "점검을 시작한 뒤 새로 나타남"))
        if d["candidate"] and d["role"] != "ap":
            ev.append(("heavy", 2, "올려 보내는 양이 많음 (초당 %.0f개, 평균 %d바이트)" % (d["per_s"], d["mean_len"])))
        if d["role"] != "ap":
            p, per = pattern(d["windows"])
            pts = {"steady": 1, "periodic": 2, "burst": 1}.get(p, 0)
            if pts:
                ev.append(("pattern", pts, PATTERN_NAME[p] + ("" if per is None else " (약 %.1f초마다)" % per)))
        v = self.walk.get(d["mac"])
        if v == "camera":
            ev.append(("walk", 3, "걷기에 전송량이 따라 변함 (판별)"))
        hub = self.dev.get(d["bssid"]) if d["role"] != "ap" else None
        hub_ssid = (hub or {}).get("ssid") or d.get("ssid") or ""
        if d["role"] != "ap" and d["bssid"]:
            why = []
            if is_random(d["bssid"]) or _match(HOTSPOT_NAMES, hub_ssid):
                why.append("개인 핫스팟처럼 보임")
            if hub is not None and hub.get("ssid") == "" and hub.get("role") == "ap":
                why.append("이름을 숨긴 Wi-Fi")
            if self.base is not None and d["bssid"] not in self.aps_base and d["bssid"] in self.dev:
                why.append("새로 생긴 Wi-Fi")
            if why:
                ev.append(("hub", 1, "공유기가 아닌 곳에 붙음: " + ", ".join(why)))
        if d["role"] == "ap":
            why = []
            if is_random(d["mac"]) or _match(HOTSPOT_NAMES, d["ssid"]):
                why.append("개인 핫스팟처럼 보임")
            if d["ssid"] == "":
                why.append("이름을 숨김")
            if why:
                ev.append(("hub", 1, "Wi-Fi 를 만드는 기기: " + ", ".join(why)))
        if _match(CAM_NAMES, hub_ssid if d["role"] != "ap" else d["ssid"]):
            ev.append(("name", 1, "카메라 앱 계열의 Wi-Fi 이름 (%s)" % (hub_ssid if d["role"] != "ap" else d["ssid"])))
        if d["rssi"] is not None:
            if d["rssi"] >= NEAR:
                ev.append(("near", 1, "가까움 (세기 %.0fdBm)" % d["rssi"]))
            elif d["rssi"] < FAR:
                ev.append(("far", -1, "멂: 다른 방일 수 있음 (세기 %.0fdBm)" % d["rssi"]))
        return ev

    def result(self, _alarms=True):
        out = {}
        for mac, d in self.dev.items():
            if mac in self.my:
                out[mac] = self._row(d, [], mine=True); continue
            out[mac] = self._row(d, self.evidence(d))
        for mac, r in out.items():                          # 허브: 붙은 기기 가운데 가장 높은 등급까지 올린다
            if r["role"] == "ap" or r["mine"]:
                continue
            h = out.get(r["bssid"])
            if h is None or h["mine"] or r["level"] == "ignore":
                continue
            h["clients"].append(mac)
            if _rank(r["level"]) > _rank(h["level"]):
                h["level"] = r["level"]
            h["reasons"].append("의심 기기 %s 가 붙은 Wi-Fi: 카메라가 조용할 때도 비컨으로 보인다" % mac)
        rows = sorted(out.values(), key=lambda r: (-_rank(r["level"]), -r["score"], r["mac"]))
        amb = [r["mac"] for r in rows if r["level"] == "suspect" and r["role"] != "ap" and r["mac"] not in self.walk and r["per_s"] >= 5]
        return dict(cycles=self.cycles, baseline=len(self.base or ()), devices=rows, events=self.events[-30:], alarms=list(self.alarms) if _alarms else [],
                    need_confirm=len(amb) >= 2, confirm=amb[:4],
                    top=[r["mac"] for r in rows if r["level"] in ("camera", "suspect")][:5])

    def _row(self, d, ev, mine=False):
        s = sum(p for _, p, _ in ev)
        lv = "mine" if mine else ("camera" if s >= LV_CAMERA else ("suspect" if s >= LV_SUSPECT else "ignore"))
        p, per = pattern(d["windows"]) if d["role"] != "ap" else ("none", None)
        return dict(mac=d["mac"], role=d["role"], bssid=d["bssid"], ssid=d["ssid"], ch=d["ch"], score=s, level=lv, mine=mine, reasons=[t for _, _, t in ev],
                    keys=[k for k, _, _ in ev], pattern=p, period=per, per_s=round(d["per_s"], 1), mean_len=d["mean_len"], rssi=d["rssi"],
                    new=bool(self.base is not None and d["mac"] not in self.base), walked=self.walk.get(d["mac"]), first=d["first"], last=d.get("last"),
                    beacon_only=d["beacon_only"], clients=[])


def _rank(lv):
    return {"camera": 3, "suspect": 2, "ignore": 1, "mine": 0}.get(lv, 0)


LEVEL_NAME = {"camera": "카메라 가능성 높음", "suspect": "의심 기기", "ignore": "무시", "mine": "내 기기"}
