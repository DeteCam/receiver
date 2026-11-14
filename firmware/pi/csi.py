#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nexmon CSI 다루기: 캡처 켜고 끄기, pcap 읽기, 패킷마다의 세기 구하기.
파이(수신기 혼자 동작)와 노트북(개발) 둘 다에서 쓴다. 파이썬 3.7 이상, numpy 만 있으면 된다.

  python3 csi.py 파일.pcap            MAC 별 패킷 수를 보여 준다
패킷 형식은 tools/csi_view.py 와 같다 (bcm43455c0, 20MHz: 0x1111, RSSI, 프레임 종류, MAC, 순번, …, 18바이트 뒤부터 int16 실수·허수 64쌍).
"""
from __future__ import print_function
import os, struct, subprocess, sys, time
import numpy as np

USED = np.r_[4:32, 33:61]             # 20MHz: 가드·DC 를 뺀 서브캐리어 (fftshift 한 뒤의 번호)
PORT = 5500
FC_NAMES = {0x80: "beacon", 0x88: "QoSdata", 0x08: "data", 0x50: "probeResp", 0x40: "probeReq", 0xd0: "action", 0x48: "null", 0xc8: "QoSnull", 0x94: "BlockAck", 0x84: "BlockAckReq"}


# ----------------------------------------------------------------- pcap 읽기
def read_csi(path):
    """[(시각, MAC, RSSI, 프레임 종류, 순번, CSI 복소수 배열), ...]. tcpdump 가 아직 쓰는 중인 파일도 읽는다 (끝의 덜 쓰인 패킷은 버린다)."""
    d = open(path, "rb").read()
    if len(d) < 24:
        return []
    magic = struct.unpack("<I", d[:4])[0]
    endian = "<" if magic in (0xa1b2c3d4, 0xa1b23c4d) else ">"
    nano = magic in (0xa1b23c4d, 0x4d3cb2a1)
    linktype = struct.unpack(endian + "I", d[20:24])[0]
    l2 = {1: 14, 113: 16, 276: 20}.get(linktype, 14)
    out = []
    i = 24; n_all = len(d)
    while i + 16 <= n_all:
        ts, frac, cl, _ = struct.unpack(endian + "IIII", d[i:i + 16])
        if i + 16 + cl > n_all:
            break
        p = d[i + 16:i + 16 + cl]; i += 16 + cl
        if len(p) < l2 + 20:
            continue
        ihl = (p[l2] & 0x0f) * 4
        u = p[l2 + ihl + 8:]
        if len(u) < 18 + 4 or u[0:2] != b"\x11\x11":
            continue
        rssi = struct.unpack("b", u[2:3])[0]
        fc = u[3]
        mac = ":".join("%02x" % b for b in bytearray(u[4:10]))
        seq = struct.unpack("<H", u[10:12])[0] >> 4
        n = (len(u) - 18) // 4
        iq = np.frombuffer(u[18:18 + n * 4], dtype="<i2").astype(np.float32)
        out.append((ts + frac / (1e9 if nano else 1e6), mac, rssi, fc, seq, iq[0::2] + 1j * iq[1::2]))
    return out


def mac_table(pk):
    """MAC 별 통계: [{mac, n, per_s, data_per_s, rssi, kinds}], 많이 보낸 순서"""
    if not pk:
        return []
    dur = max(pk[-1][0] - pk[0][0], 1e-6); st = {}
    for t, mac, rssi, fc, _, _ in pk:
        s = st.setdefault(mac, [0, 0.0, {}]); s[0] += 1; s[1] += rssi; s[2][fc] = s[2].get(fc, 0) + 1
    rows = []
    for m, (n, r, kinds) in st.items():
        top = sorted(kinds.items(), key=lambda kv: -kv[1])[:3]
        nd = kinds.get(0x88, 0) + kinds.get(0x08, 0)                    # 데이터 프레임만 (비컨·응답은 뺀다)
        rows.append(dict(mac=m, n=n, per_s=round(n / dur, 1), data_per_s=round(nd / dur, 1), rssi=round(r / n, 1), kinds=", ".join("%s:%d" % (FC_NAMES.get(k, hex(k)), v) for k, v in top)))
    return sorted(rows, key=lambda r: -r["n"])


def packet_levels(pk, mac=None):
    """(시각 배열, 세기 행렬[패킷, 서브캐리어] dB, MAC). 패킷마다 CSI 전력으로 나누고 RSSI 를 더해 자동 이득 조절을 되돌린다 (tools/manual_scan.py 와 같은 계산).
    패킷이 없으면 (빈 배열, 빈 행렬, mac)"""
    if mac:
        mac = mac.lower(); pk = [p for p in pk if p[1] == mac]
    elif pk:
        mac = mac_table(pk)[0]["mac"]; pk = [p for p in pk if p[1] == mac]
    pk = [p for p in pk if len(p[5]) == 64]
    if not pk:
        return np.zeros(0), np.zeros((0, len(USED))), mac
    t = np.array([p[0] for p in pk]); rs = np.array([p[2] for p in pk], dtype=float)
    A = np.array([np.abs(np.fft.fftshift(p[5])) for p in pk])[:, USED]
    A = A[:, np.median(A, 0) <= 5 * np.median(A)]             # 진폭이 비정상적으로 큰 서브캐리어는 뺀다
    C = 20 * np.log10(np.maximum(A, 1)) - 10 * np.log10(np.maximum(np.mean(A ** 2, 1), 1e-9))[:, None] + rs[:, None]
    return t, C, mac


# ----------------------------------------------------------------- 캡처 (파이에서)
def csi_iface():
    """CSI 를 받는 내장 Wi-Fi 의 이름. 드라이버가 brcmfmac 인 것을 찾는다 (USB 동글을 꽂으면 wlan0, wlan1 순서가 바뀔 수 있어서 이름을 믿지 않는다)."""
    base = "/sys/class/net"
    try:
        for name in sorted(os.listdir(base)):
            drv = os.path.join(base, name, "device", "driver")
            if os.path.islink(drv) and os.path.basename(os.readlink(drv)).startswith("brcmfmac"):
                return name
    except OSError:
        pass
    return "wlan0"


def link_ifaces():
    """폰과 통신하는 쪽 후보: USB 로 붙은 무선 장치들의 이름"""
    base = "/sys/class/net"; out = []
    try:
        for name in sorted(os.listdir(base)):
            dev = os.path.join(base, name, "device")
            if os.path.isdir(os.path.join(base, name, "wireless")) or os.path.exists(os.path.join(base, name, "phy80211")):
                sub = os.path.join(dev, "subsystem")
                if os.path.islink(sub) and os.path.basename(os.readlink(sub)) == "usb":
                    out.append(name)
    except OSError:
        pass
    return out


class Capture(object):
    """파이에서 CSI 캡처를 켜고 끈다. 패킷은 path (pcap) 에 쌓인다."""
    def __init__(self, dry=False):
        self.dry = dry; self.proc = None; self.path = None; self.iface = "wlan0" if dry else csi_iface()

    def start(self, ch, mac, path):
        self.stop(); self.path = path
        if self.dry:
            open(path, "wb").close(); return
        ifc = self.iface
        subprocess.call("sudo pkill -x tcpdump >/dev/null 2>&1; sudo ip link set %s up" % ifc, shell=True)
        args = ["makecsiparams", "-c", "%d/20" % int(ch), "-C", "1", "-N", "1"] + (["-m", mac] if mac else [])
        p = subprocess.check_output(args).decode().strip()
        rc = subprocess.call(["sudo", "nexutil", "-I" + ifc, "-s500", "-b", "-l34", "-v" + p], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if rc != 0:
            raise RuntimeError("nexutil 이 실패했습니다 (내장 Wi-Fi %s). nexmon_csi 펌웨어가 올라가 있는지 확인하세요." % ifc)
        self.proc = subprocess.Popen(["sudo", "tcpdump", "-i", ifc, "dst", "port", str(PORT), "-U", "-w", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)

    def stop(self):
        if self.proc is not None:
            subprocess.call("sudo pkill -x tcpdump >/dev/null 2>&1", shell=True)
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
            self.proc = None

    def packets(self):
        return read_csi(self.path) if self.path and os.path.exists(self.path) else []


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    pk = read_csi(sys.argv[1])
    print("패킷 %d개" % len(pk))
    for r in mac_table(pk)[:20]:
        print("%s  %6d개  초당 %6.1f  RSSI %5.0f  %s" % (r["mac"], r["n"], r["per_s"], r["rssi"], r["kinds"]))
