#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""탐지: 네트워크를 모르는 채로 둘레의 Wi-Fi 기기를 훑어, 영상을 내보내는 것처럼 보이는 기기를 추린다.
모니터 모드가 되는 USB 무선 동글(탐지 동글)로 듣는다. 파이(파이썬 3.7 이상, 추가 설치 없음)와 노트북 둘 다에서 쓴다.

  python3 detect.py 파일.pcap            tcpdump 로 받아 둔 파일(radiotap)을 분석해 표로 보여 준다
  python3 detect.py --live               파이에서: 채널을 돌며 듣고 표로 보여 준다 (node.py 의 '기기 찾기'와 같은 일)
  python3 detect.py --check              탐지 동글이 잡혔는지만 본다

왜 동글이 하나 더 필요한가
  내장 Wi-Fi 는 CSI 를 받는 펌웨어(nexmon_csi)로 바뀌어 있어, 프레임마다 CSI 와 보낸 기기의 주소만 주고 길이는 주지 않는다.
  '영상을 내보내는 기기'를 가리려면 프레임이 몇 개인지와 함께 얼마나 긴지를 알아야 한다.
  폰과 통신하는 동글(WNA3100M, 드라이버 rtl8192cu)은 남의 프레임을 엿듣는 데 못 쓴다. 그래서 모니터 모드가 되는 동글을 하나 더 꽂는다.
  이 동글은 듣기만 하고 아무것도 보내지 않으므로 CSI 를 재는 데 방해가 되지 않는다.
동글이 없으면
  내장 Wi-Fi 의 CSI 수로 '초당 몇 개 보내는지'만 본다 (node.py 가 알아서 그렇게 한다). 길이를 모르므로 덜 정확하다.

가리는 기준 (DiffLoc+ 논문의 기준을 따랐다. 실물로 맞춰 봐야 하는 숫자다)
  기기가 보낸 데이터 프레임이 5초에 150개(초당 30개) 이상이고 평균 길이가 300바이트 이상이면 '카메라 후보'.
  작은 프레임만 많이 보내는 기기(영상을 받는 쪽이 보내는 응답)와 가끔만 보내는 기기는 빠진다.
  영상 통화나 화면 공유처럼 카메라가 아닌데 많이 내보내는 기기는 여기서 못 가린다: 그것은 다음 단계(판별)가 한다.
한계
  2.4GHz 전용 동글(RT5370, MT7601U)은 5GHz 를 못 듣는다. 듀얼밴드 동글(MT7610U, MT7612U)이면 5GHz 도 듣는다:
  처음 한 바퀴는 5GHz 를 모두 훑고, 그 뒤로는 AP 가 들린 5GHz 채널만 돈다 (카메라는 어딘가의 AP 에 붙어야 하므로). FULL5 바퀴마다 다시 모두 훑는다.
  안테나가 하나인 동글은 두 개로 한꺼번에 보내는(2 스트림) 프레임을 못 읽는다. 몰래카메라는 거의 안테나 하나다.
"""
from __future__ import print_function
import os, struct, subprocess, sys, time

CH24 = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
CH5 = (36, 40, 44, 48, 52, 56, 60, 64, 100, 104, 108, 112, 116, 120, 124, 128, 132, 136, 140, 144, 149, 153, 157, 161, 165)   # 한국 5GHz 20MHz 채널 (동글이 되는 것만 쓴다)
FULL5 = 5                      # 이 바퀴마다 5GHz 를 모두 다시 훑는다 (새로 켠 5GHz 핫스팟을 찾으려고)
MIN_RATE = 30.0                # 초당 데이터 프레임 수 (5초에 150개)
MIN_LEN = 300.0                # 평균 길이 (바이트, 헤더와 암호 덧붙임을 뺀 것)
WEAK_RSSI = -75.0              # 이보다 약하면 '다른 방일 수 있음'
SNAP = 200                     # 프레임 앞부분만 받는다 (길이는 pcap 이 따로 적어 준다)
SETTLE = 0.12                  # 채널을 바꾼 뒤 자리 잡는 시간 (초)
MON_OK = ("rt2800usb", "mt7601u", "ath9k_htc", "mt76x0u", "mt76x2u")        # 모니터 모드가 된다고 알려진 드라이버 (파이 커널 5.4 에 모듈로 들어 있다)
MON_BAD = ("rtl8192cu", "rtl8xxxu", "r8188eu", "8192cu", "r8712u", "brcmfmac")
LINK_NAME, MON_NAME = "wlink0", "wmon0"       # link_setup.sh 가 붙이는 이름
PCAP = "/tmp/detecam_mon.pcap"


def mac_str(b):
    return ":".join("%02x" % c for c in bytearray(b))


# ----------------------------------------------------------------- 받아 둔 파일 읽기 (radiotap + 802.11)
def read_frames(path):
    """[(시각, 종류, 세부 종류, 플래그, 주소1, 주소2, 주소3, 길이, 세기, 주파수, 이름, 채널)]
    종류 0 = 관리(비컨 등), 1 = 제어, 2 = 데이터. 길이 = 프레임 전체(끝의 검사 4바이트는 뺀다). 이름·채널은 비컨에서만.
    tcpdump 가 아직 쓰는 중인 파일도 읽는다 (끝의 덜 쓰인 것은 버린다)."""
    try:
        d = open(path, "rb").read()
    except IOError:
        return []
    if len(d) < 24:
        return []
    magic = struct.unpack("<I", d[:4])[0]
    endian = "<" if magic in (0xa1b2c3d4, 0xa1b23c4d) else ">"
    nano = magic in (0xa1b23c4d, 0x4d3cb2a1)
    if struct.unpack(endian + "I", d[20:24])[0] != 127:
        raise RuntimeError("이 파일은 radiotap 형식이 아닙니다 (모니터 모드로 받은 것이 아니다).")
    out = []; i = 24; n_all = len(d); div = 1e9 if nano else 1e6
    while i + 16 <= n_all:
        ts, frac, cl, ol = struct.unpack(endian + "IIII", d[i:i + 16])
        if i + 16 + cl > n_all:
            break
        p = bytearray(d[i + 16:i + 16 + cl]); i += 16 + cl
        if cl < 8 or p[0] != 0:
            continue
        rlen = p[2] | (p[3] << 8)
        if rlen > cl:
            continue
        present = p[4] | (p[5] << 8) | (p[6] << 16) | (p[7] << 24); off = 8; w = present
        while w & 0x80000000 and off + 4 <= rlen:              # 표시 칸이 여러 개일 수 있다
            w = p[off] | (p[off + 1] << 8) | (p[off + 2] << 16) | (p[off + 3] << 24); off += 4
        flags = 0; rssi = None; freq = None
        try:
            if present & 1:
                off = ((off + 7) & ~7) + 8                     # 시각 (8바이트, 8의 배수 자리에서 시작)
            if present & 2:
                flags = p[off]; off += 1
            if present & 4:
                off += 1                                       # 속도
            if present & 8:
                off = (off + 1) & ~1; freq = p[off] | (p[off + 1] << 8); off += 4
            if present & 16:
                off += 2
            if present & 32:
                rssi = p[off] - 256 if p[off] > 127 else p[off]
        except IndexError:
            continue
        if flags & 0x40:                                       # 검사 값이 틀린 프레임
            continue
        f = p[rlen:]
        if len(f) < 10 or (f[0] & 3):
            continue
        typ = (f[0] >> 2) & 3; sub = f[0] >> 4
        if typ == 1 or len(f) < 24:                            # 제어 프레임(응답 등)은 쓰지 않는다
            continue
        n = ol - rlen - (4 if flags & 0x10 else 0)
        ssid = None; dsch = None
        if typ == 0 and sub in (8, 5):                         # 비컨, 탐색 응답: 이름과 채널을 읽는다
            j = 36
            while j + 2 <= len(f):
                eid, el = f[j], f[j + 1]
                if j + 2 + el > len(f):
                    break
                if eid == 0 and ssid is None:
                    try:
                        ssid = bytes(f[j + 2:j + 2 + el]).decode("utf-8", "replace").replace("\x00", "")
                    except Exception:
                        ssid = ""
                elif eid == 3 and el == 1:
                    dsch = f[j + 2]; break
                j += 2 + el
        out.append((ts + frac / div, typ, sub, f[1], bytes(f[4:10]), bytes(f[10:16]), bytes(f[16:22]), n, rssi, freq, ssid, dsch))
    return out


def body_len(n, sub, fl):
    """데이터 프레임에서 헤더와 암호 덧붙임을 뺀 길이"""
    h = 24 + (6 if (fl & 3) == 3 else 0) + (2 if sub & 8 else 0) + (4 if (fl & 0x80) and (sub & 8) else 0) + (16 if fl & 0x40 else 0)
    return max(n - h, 0)


def freq_ch(f):
    if f is None:
        return None
    if f == 2484:
        return 14
    return (f - 2407) // 5 if f < 3000 else (f - 5000) // 5


def ap_table(frames):
    """비컨에서: {공유기 주소: (이름, 채널)}"""
    aps = {}
    for fr in frames:
        if fr[1] == 0 and fr[10] is not None:
            old = aps.get(fr[6])
            aps[fr[6]] = (fr[10] or (old[0] if old else ""), fr[11] if fr[11] is not None else (old[1] if old else freq_ch(fr[9])))
    return aps


def tally(frames, t0, t1):
    """t0 ~ t1 사이에 '보낸 기기'별로: {(주소, 방향): [프레임 수, 길이 합, 세기 합, 세기 개수, 상대 공유기]}. 방향 'sta' = 기기가 공유기로, 'ap' = 공유기가 기기로"""
    st = {}
    for t, typ, sub, fl, a1, a2, a3, n, rssi, freq, _, _ in frames:
        if typ != 2 or (sub & 4) or t < t0 or t >= t1:
            continue
        ds = fl & 3
        if ds == 1:
            key, bssid = (a2, "sta"), a1
        elif ds == 2:
            key, bssid = (a2, "ap"), a2
        else:
            key, bssid = (a2, "sta"), a3
        s = st.get(key)
        if s is None:
            s = st[key] = [0, 0, 0.0, 0, bssid]
        s[0] += 1; s[1] += body_len(n, sub, fl)
        if rssi is not None:
            s[2] += rssi; s[3] += 1
    return st


def busiest(frames, hops, n_focus=3):
    """훑은 결과에서 데이터가 많은 채널을 고른다. 이웃 채널로 새어 들린 것은 비컨에 적힌 진짜 채널로 센다."""
    aps = ap_table(frames); cnt = {}
    for t0, t1, ch, kind in hops:
        for (mac, role), s in tally(frames, t0, t1).items():
            true = aps.get(s[4], (None, None))[1] or ch
            cnt[true] = cnt.get(true, 0) + s[0]
    order = sorted([c for c in cnt if cnt[c] > 0], key=lambda c: -cnt[c])
    return order[:n_focus], cnt


def series_of(frames, mac, t0, t1, dt=0.5):
    """한 기기가 t0 ~ t1 사이에 보낸 데이터 프레임 수를 dt 초 칸으로 (송출 패턴을 가리는 데 쓴다: core/score.py)"""
    n = max(1, int((t1 - t0) / dt)); out = [0] * n
    for fr in frames:
        if fr[1] == 2 and not (fr[2] & 4) and fr[5] == mac and t0 <= fr[0] < t1:
            k = int((fr[0] - t0) / dt)
            if k < n:
                out[k] += 1
    return out


def analyse(frames, hops=None, beacons=True):
    """기기 표를 만든다. hops = [(시작, 끝, 채널, 'quick' 또는 'focus')]. 안 주면 파일 전체를 한 구간으로 본다.
    beacons=True 면 데이터는 없고 비컨만 들린 Wi-Fi(공유기, 핫스팟, 카메라가 만든 Wi-Fi)도 줄로 넣는다 (beacon_only=True, 후보는 아님).
    줄마다 series = 가장 잘 들은 구간의 0.5초 칸 패킷 수."""
    if not frames:
        return []
    if not hops:
        hops = [(frames[0][0], frames[-1][0] + 1e-3, freq_ch(frames[0][9]), "focus")]
    aps = ap_table(frames); best = {}
    for t0, t1, ch, kind in hops:
        dur = max(t1 - t0, 1e-3)
        for key, s in tally(frames, t0, t1).items():
            rank = (kind == "focus", s[0] / dur)
            if key not in best or rank > best[key][0]:
                best[key] = (rank, s, dur, ch, kind, t0, t1)
    rows = []; seen = set()
    for (mac, role), (_, s, dur, ch, kind, t0, t1) in best.items():
        n, tot, rs, rn, bssid = s
        ssid, apch = aps.get(bssid, ("", None))
        per_s = n / dur; mean = tot / float(max(n, 1)); rssi = rs / rn if rn else None
        m = bytearray(mac)
        if m[0] & 1:                                           # 여럿에게 보내는 주소는 기기가 아니다
            continue
        cand = per_s >= MIN_RATE and mean >= MIN_LEN
        note = []
        if kind != "focus":
            note.append("짧게만 들음")
        if cand and role == "ap":
            note.append("공유기도 이렇게 보인다. 카메라가 직접 Wi-Fi 를 만드는 종류일 수도 있다")
        if rssi is not None and rssi < WEAK_RSSI:
            note.append("신호가 약함: 다른 방일 수 있다")
        if m[0] & 2:
            note.append("임의 주소 (폰·노트북에 많다)")
        seen.add(mac)
        rows.append(dict(mac=mac_str(mac), role=role, bssid=mac_str(bssid), ssid=ssid, ch=int(apch or ch or 0), n=int(n), per_s=round(per_s, 1), kbps=round(tot * 8 / dur / 1000.0, 1),
                         mean_len=int(round(mean)), rssi=None if rssi is None else round(rssi, 1), candidate=bool(cand), weak=bool(rssi is not None and rssi < WEAK_RSSI),
                         random=bool(m[0] & 2), heard=kind, note=" · ".join(note), beacon_only=False, series=series_of(frames, mac, t0, t1)))
    if beacons:
        bs = {}
        for fr in frames:
            if fr[1] == 0 and fr[2] == 8 and fr[6] not in seen and not (bytearray(fr[6])[0] & 1):
                b = bs.setdefault(fr[6], [0, 0.0, 0])
                b[0] += 1
                if fr[8] is not None:
                    b[1] += fr[8]; b[2] += 1
        for bssid, (k, rsum, rn) in bs.items():
            ssid, apch = aps.get(bssid, ("", None)); rssi = round(rsum / rn, 1) if rn else None; m = bytearray(bssid)
            note = ["비컨만 들림"] + (["이름을 숨김"] if ssid == "" else []) + (["임의 주소 (폰 핫스팟에 많다)"] if m[0] & 2 else [])
            rows.append(dict(mac=mac_str(bssid), role="ap", bssid=mac_str(bssid), ssid=ssid, ch=int(apch or 0), n=0, per_s=0.0, kbps=0.0, mean_len=0, rssi=rssi,
                             candidate=False, weak=bool(rssi is not None and rssi < WEAK_RSSI), random=bool(m[0] & 2), heard="beacon", note=" · ".join(note),
                             beacon_only=True, series=[]))
    rows.sort(key=lambda r: (not (r["candidate"] and r["role"] == "sta"), not r["candidate"], r["beacon_only"], -r["kbps"]))
    return rows


# ----------------------------------------------------------------- 동글 찾기와 듣기 (파이에서)
def usb_wifi():
    """USB 로 붙은 무선 장치: [{name, driver}]"""
    base = "/sys/class/net"; out = []
    try:
        for name in sorted(os.listdir(base)):
            dev = os.path.join(base, name, "device")
            if not (os.path.isdir(os.path.join(base, name, "wireless")) or os.path.exists(os.path.join(base, name, "phy80211"))):
                continue
            sub = os.path.join(dev, "subsystem"); drv = os.path.join(dev, "driver")
            if os.path.islink(sub) and os.path.basename(os.readlink(sub)) == "usb":
                out.append(dict(name=name, driver=os.path.basename(os.readlink(drv)) if os.path.islink(drv) else "?"))
    except OSError:
        pass
    return out


def has_ip(name):
    try:
        return bool(subprocess.check_output(["ip", "-4", "-o", "addr", "show", "dev", name], stderr=subprocess.DEVNULL).strip())
    except Exception:
        return False


def find_monitor():
    """탐지 동글: 폰과 통신하는 동글(wlink0 이거나 주소를 받은 것)이 아니고, 모니터 모드가 안 되는 드라이버가 아닌 USB 무선 장치"""
    c = [d for d in usb_wifi() if d["name"] != LINK_NAME and d["driver"] not in MON_BAD and not has_ip(d["name"])]
    c.sort(key=lambda d: (d["name"] != MON_NAME, d["driver"] not in MON_OK))
    return c[0] if c else None


class Monitor(object):
    def __init__(self, dry=False):
        self.dry = dry; self.proc = None; self.iface = None; self.driver = None
        self.refresh()

    def refresh(self):
        d = None if self.dry else find_monitor()
        self.iface = d["name"] if d else None; self.driver = d["driver"] if d else None
        return self.iface is not None

    def available(self):
        return self.iface is not None

    def prepare(self):
        ifc = self.iface; q = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.call(["sudo", "ip", "link", "set", ifc, "down"], **q)
        rc = subprocess.call(["sudo", "iw", "dev", ifc, "set", "type", "monitor"], **q)
        rc2 = subprocess.call(["sudo", "ip", "link", "set", ifc, "up"], **q)
        if rc != 0 or rc2 != 0:
            raise RuntimeError("탐지 동글(%s, 드라이버 %s)을 모니터 모드로 바꾸지 못했습니다. 다른 프로그램이 쓰는 중인지 보세요:  sudo ./link_setup.sh check" % (ifc, self.driver))

    def channels(self):
        """이 동글이 들을 수 있는 채널 (iw phy 의 주파수 목록에서 꺼진 것을 뺀다). 못 읽으면 2.4GHz"""
        if getattr(self, "_chans", None) is not None:
            return self._chans
        ch = ()
        try:
            out = subprocess.check_output(["iw", "dev", self.iface, "info"], universal_newlines=True)
            phy = [l.split()[-1] for l in out.splitlines() if l.strip().startswith("wiphy")][0]
            ch = parse_phy_channels(subprocess.check_output(["iw", "phy", "phy" + phy, "info"], universal_newlines=True))
        except Exception:
            pass
        self._chans = ch or CH24
        return self._chans

    def channel(self, ch):
        return subprocess.call(["sudo", "iw", "dev", self.iface, "set", "channel", str(int(ch))], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

    def start(self, path=PCAP):
        self.stop(); self.path = path
        subprocess.call("sudo pkill -x tcpdump >/dev/null 2>&1; sudo rm -f %s" % path, shell=True)
        self.proc = subprocess.Popen(["sudo", "tcpdump", "-i", self.iface, "-s", str(SNAP), "-U", "-w", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.8)

    def stop(self):
        if self.proc is not None:
            subprocess.call("sudo pkill -x tcpdump >/dev/null 2>&1", shell=True)
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
            self.proc = None


def parse_phy_channels(text):
    """'iw phy … info' 의 '* 5180 MHz [36] (20.0 dBm)' 줄에서 쓸 수 있는 채널 번호. disabled 는 뺀다 (no IR 은 듣기만 하므로 괜찮다)"""
    out = []
    for l in text.splitlines():
        l = l.strip()
        if l.startswith("*") and "MHz [" in l and "disabled" not in l:
            try:
                c = int(l.split("[", 1)[1].split("]", 1)[0])
            except ValueError:
                continue
            if (c in CH24 or c in CH5) and c not in out:
                out.append(c)
    return tuple(out)


def plan(supported, seen5=None, cycle=0):
    """이번 바퀴에 훑을 채널: 2.4GHz 는 모두, 5GHz 는 처음·FULL5 바퀴마다 모두, 그 밖에는 AP 가 들린 채널만"""
    c24 = [c for c in supported if c <= 14]; c5 = [c for c in supported if c > 14]
    if not c5:
        return tuple(c24)
    if seen5 is None or cycle % FULL5 == 0:
        return tuple(c24 + c5)
    return tuple(c24 + [c for c in c5 if c in seen5])


def ap_channels5(rows):
    """5GHz 에서 AP(또는 AP 에 붙은 기기)가 들린 채널"""
    return set(int(r["ch"]) for r in rows if r.get("ch") and int(r["ch"]) > 14)


def scan(mon, channels=CH24, quick=0.6, focus=5.0, n_focus=3, path=PCAP, progress=None, extra=()):
    """채널을 한 번 훑고(채널마다 quick 초), 데이터가 많은 채널 n_focus 개를 focus 초씩 자세히 듣는다. 13채널이면 약 30초.
    channels=None 이면 동글이 들을 수 있는 채널 모두 (듀얼밴드면 5GHz 까지).
    extra = 데이터가 적어도 꼭 자세히 들을 채널 (감시할 때: 의심 기기와 그 허브가 있는 채널. 몇 초마다 몰아서 보내는 카메라를 놓치지 않으려고)"""
    say = progress or (lambda *a: None)
    channels = tuple(channels or mon.channels())
    mon.prepare(); mon.start(path); hops = []
    try:
        for k, ch in enumerate(channels):
            say("채널 %d 훑는 중" % ch, k, len(channels) + n_focus)
            if not mon.channel(ch):
                continue
            t0 = time.time() + SETTLE; time.sleep(SETTLE + quick); hops.append((t0, time.time(), int(ch), "quick"))
        if not hops:
            raise RuntimeError("탐지 동글의 채널을 바꾸지 못했습니다 (%s)." % mon.iface)
        best, _ = busiest(read_frames(path), hops, n_focus)
        best = list(best) + [c for c in extra if c and c not in best]
        for k, ch in enumerate(best):
            say("채널 %d 자세히 듣는 중 (%.0f초)" % (ch, focus), len(channels) + k, len(channels) + len(best))
            if mon.channel(ch):
                t0 = time.time() + SETTLE; time.sleep(SETTLE + focus); hops.append((t0, time.time(), int(ch), "focus"))
    finally:
        mon.stop()
    return analyse(read_frames(path), hops)


def listen(mon, ch, path=PCAP):
    """한 채널에 머물며 듣기 시작한다 (판별 단계에서 쓴다). 끝낼 때는 mon.stop()"""
    mon.prepare()
    if not mon.channel(ch):
        raise RuntimeError("탐지 동글을 채널 %d 로 맞추지 못했습니다." % ch)
    mon.start(path)


def sent_by(frames, mac):
    """한 기기가 보낸 데이터 프레임의 (시각들, 길이들)"""
    m = bytes(bytearray(int(x, 16) for x in mac.split(":")))
    t = []; n = []
    for fr in frames:
        if fr[1] == 2 and not (fr[2] & 4) and fr[5] == m:
            t.append(fr[0]); n.append(body_len(fr[7], fr[2], fr[3]))
    return t, n


def fake_rows():
    """장치 없이 흐름을 볼 때 쓰는 가짜 표"""
    r = [dict(mac="fa:ke:00:00:00:01", role="sta", bssid="fa:ke:00:00:00:a6", ssid="room-507", ch=6, n=318, per_s=63.6, kbps=371.2, mean_len=729, rssi=-47.0, candidate=True, weak=False, random=False, heard="focus", note=""),
         dict(mac="fa:ke:00:00:00:a6", role="ap", bssid="fa:ke:00:00:00:a6", ssid="room-507", ch=6, n=702, per_s=140.4, kbps=1412.0, mean_len=1257, rssi=-52.0, candidate=True, weak=False, random=False, heard="focus",
              note="공유기도 이렇게 보인다. 카메라가 직접 Wi-Fi 를 만드는 종류일 수도 있다"),
         dict(mac="fa:ke:00:00:00:02", role="sta", bssid="fa:ke:00:00:00:a6", ssid="room-507", ch=6, n=455, per_s=91.0, kbps=43.7, mean_len=60, rssi=-55.0, candidate=False, weak=False, random=False, heard="focus", note=""),
         dict(mac="fe:ke:00:00:00:03", role="sta", bssid="fa:ke:00:00:00:b1", ssid="guest", ch=11, n=61, per_s=12.2, kbps=14.8, mean_len=152, rssi=-63.0, candidate=False, weak=False, random=True, heard="focus", note="임의 주소 (폰·노트북에 많다)"),
         dict(mac="fa:ke:00:00:00:04", role="sta", bssid="fa:ke:00:00:00:c3", ssid="room-508", ch=1, n=205, per_s=41.0, kbps=290.3, mean_len=885, rssi=-81.0, candidate=True, weak=True, random=False, heard="focus", note="신호가 약함: 다른 방일 수 있다")]
    return r


def show(rows):
    print("%-17s %-4s %2s %7s %8s %6s %5s  %s" % ("주소", "방향", "채널", "초당", "kbit/s", "평균", "세기", "비고"))
    for r in rows:
        print("%-17s %-4s %4d %7.1f %8.1f %6d %5s  %s%s%s" % (r["mac"], "올림" if r["role"] == "sta" else "내림", r["ch"], r["per_s"], r["kbps"], r["mean_len"],
                                                          "-" if r["rssi"] is None else "%.0f" % r["rssi"], "★ 후보  " if r["candidate"] else "", ("[" + r["ssid"] + "] ") if r["ssid"] else "", r["note"]))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--check":
        print("USB 무선 장치:", usb_wifi() or "없음"); d = find_monitor()
        if d:
            c = Monitor().channels(); print("들을 수 있는 채널: %s (%s)" % (" ".join(map(str, c)), "2.4GHz 와 5GHz" if any(x > 14 for x in c) else "2.4GHz 만"))
        print("탐지 동글:", ("%s (드라이버 %s%s)" % (d["name"], d["driver"], "" if d["driver"] in MON_OK else ", 확인 안 된 드라이버")) if d else "없음 (모니터 모드가 되는 동글을 가운데 USB 자리에 꽂는다)")
    elif sys.argv[1] == "--live":
        m = Monitor()
        if not m.available():
            sys.exit("탐지 동글을 찾지 못했습니다.")
        print("들을 수 있는 채널:", m.channels())
        show(scan(m, None, progress=lambda s, a, b: print("  ", s)))
    else:
        show(analyse(read_frames(sys.argv[1])))
