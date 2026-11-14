#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect.py 검사: scapy 로 진짜와 같은 모양의 프레임(radiotap + 802.11)을 만들어 파일로 쓰고, 읽은 값이 scapy 가 읽은 값과 같은지,
'카메라 후보'가 맞게 골라지는지 본다. 실물 동글 없이 할 수 있는 데까지의 검사다 (동글과 드라이버는 실물에서 확인해야 한다).
    python3 tests/test_detect.py          (scapy 필요: pip install scapy)
    python3 tests/test_detect.py --save   작은 파일(fixture_mon.pcap)과 답(fixture_mon.json)을 적어 둔다: scapy 가 없는 파이에서는 test_core.py 가 이것으로 검사한다"""
import json, os, random, struct, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import detect as D
try:
    from scapy.all import RadioTap, Dot11, Dot11FCS, Dot11QoS, Dot11Beacon, Dot11Elt, Raw, conf
    conf.verb = 0
except Exception as e:
    print("scapy 가 없어 건너뜁니다:", e); sys.exit(0)

AP1, AP2 = "a4:2b:b0:00:00:a6", "a4:2b:b0:00:00:b1"
CAM, LAP, PHONE, FAR = "c0:49:ef:11:22:33", "3c:22:fb:44:55:66", "da:a1:19:77:88:99", "c0:49:ef:aa:bb:cc"
FREQ = lambda ch: 2407 + 5 * ch
rnd = random.Random(7)


def rt(ch, rssi, bad=False, ext=False):
    """radiotap 머리. ext=True 면 표시 칸이 두 개인 모양 (안테나별 세기를 따로 적는 드라이버)"""
    flags = 0x10 | (0x40 if bad else 0)
    if not ext:
        return bytes(RadioTap(present="TSFT+Flags+Rate+Channel+dBm_AntSignal+Antenna", mac_timestamp=123456, Flags=flags, Rate=13, ChannelFrequency=FREQ(ch), ChannelFlags=0x00c0, dBm_AntSignal=rssi, Antenna=0))
    # 손으로 만든다: 표시 칸 2개 (첫 칸: Flags, Rate, Channel, 세기, 다음 칸 있음 / 둘째 칸: 세기, 안테나)
    p1 = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 5) | (1 << 29) | (1 << 31); p2 = (1 << 5) | (1 << 11)
    body = struct.pack("<BB", flags, 13) + struct.pack("<HH", FREQ(ch), 0x00c0) + struct.pack("<b", rssi) + struct.pack("<bB", rssi - 2, 1)
    return struct.pack("<BBHII", 0, 0, 12 + len(body), p1, p2) + body


def data(src, dst, bssid, to_ds, size, null=False, qos=True, prot=True):
    fc = (1 if to_ds else 2) | (0x40 if prot and not null else 0)
    a1, a2, a3 = (bssid, src, dst) if to_ds else (dst, bssid, src)
    sub = (12 if qos else 4) if null else (8 if qos else 0)
    f = Dot11FCS(type=2, subtype=sub, FCfield=fc, addr1=a1, addr2=a2, addr3=a3)
    if qos:
        f = f / Dot11QoS()
    if not null:
        f = f / Raw(bytes(bytearray(rnd.getrandbits(8) for _ in range(size + (16 if prot else 0)))))
    return bytes(f)


def beacon(bssid, ssid, ch):
    return bytes(Dot11FCS(type=0, subtype=8, addr1="ff:ff:ff:ff:ff:ff", addr2=bssid, addr3=bssid) / Dot11Beacon(cap=0x1411) / Dot11Elt(ID=0, info=ssid.encode()) /
                 Dot11Elt(ID=1, info=b"\x82\x84\x8b\x96\x0c\x12\x18\x24") / Dot11Elt(ID=3, info=bytes([ch])))


def traffic(t0, t1, tuned):
    """tuned 채널에 맞춰 듣는 동안 들리는 프레임: [(시각, 바이트)]"""
    out = []
    def add(rate, make, rssi, ext=False):
        t = t0 + rnd.random() / rate
        while t < t1:
            out.append((t, rt(tuned, rssi + rnd.randint(-2, 2), ext=ext) + make())); t += rnd.expovariate(rate)
    if tuned == 6:
        add(10, lambda: beacon(AP1, "room-507", 6), -50)
        add(60, lambda: data(CAM, "00:11:22:33:44:55", AP1, True, rnd.choice((1380, 1380, 1380, 620, 240))), -47, ext=True)       # 카메라: 큰 프레임을 초당 60개
        add(5, lambda: data(CAM, AP1, AP1, True, 0, null=True), -47)                                                             # 절전 알림 (세지 않아야 한다)
        add(90, lambda: data(LAP, "00:11:22:33:44:55", AP1, True, 60), -55)                                                      # 영상을 받는 노트북: 작은 응답만 많이
        add(140, lambda: data("00:11:22:33:44:55", LAP, AP1, False, 1400), -52)                                                  # 공유기가 노트북에 내려보냄
        out.append((t0 + 0.01, rt(tuned, -47, bad=True) + data(CAM, AP1, AP1, True, 1380)))                                      # 깨진 프레임 (세지 않아야 한다)
    if tuned in (5, 7):                                                                                                          # 이웃 채널로 조금 샌다
        add(12, lambda: data(CAM, "00:11:22:33:44:55", AP1, True, 1380), -60)
        add(2, lambda: beacon(AP1, "room-507", 6), -62)
    if tuned == 11:
        add(10, lambda: beacon(AP2, "guest", 11), -60)
        add(12, lambda: data(PHONE, "00:11:22:33:44:55", AP2, True, 150, qos=False), -63)
    if tuned == 1:
        add(41, lambda: data(FAR, "00:11:22:33:44:55", "a4:2b:b0:00:00:c3", True, 900), -81)                                     # 옆방 카메라 (약한 신호)
    return out


def write_pcap(path, recs, snap=D.SNAP):
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, snap, 127))
        for t, b in sorted(recs, key=lambda r: r[0]):
            f.write(struct.pack("<IIII", int(t), int((t - int(t)) * 1e6), min(len(b), snap), len(b)) + b[:snap])


def main(save=False):
    t = 1794600000.0; hops = []; recs = []; focus = 2.0 if save else 5.0
    for ch in ((1, 5, 6, 7, 11) if save else D.CH24):        # 한 번 훑기
        hops.append((t, t + 0.6, ch, "quick")); recs += traffic(t, t + 0.6, ch); t += 0.72
    path = os.path.join(HERE, "fixture_mon.pcap") if save else os.path.join(tempfile.mkdtemp(), "mon.pcap"); write_pcap(path, recs)
    best, cnt = D.busiest(D.read_frames(path), hops, 3)
    assert best[0] == 6 and set(best) == {6, 1, 11}, (best, cnt)          # 이웃 채널(5, 7)로 샌 것은 6번으로 세어져야 한다
    for ch in best:                                          # 자세히 듣기
        hops.append((t, t + focus, ch, "focus")); recs += traffic(t, t + focus, ch); t += focus + 0.12
    write_pcap(path, recs)
    fr = D.read_frames(path)

    # 1) 읽은 값이 scapy 가 읽은 값과 같은가 (데이터·비컨 400개를 뽑아서)
    full = sorted(recs, key=lambda r: r[0])
    by_t = {}
    for r in fr:
        by_t[int(round(r[0] * 1e6))] = r
    n_cmp = 0
    for tt, b in rnd.sample(full, 400):
        s = RadioTap(b); key = int(tt) * 1000000 + int((tt - int(tt)) * 1e6)
        bad = bool(int(s.Flags) & 0x40) if s.Flags is not None else False
        if bad:
            assert key not in by_t, "깨진 프레임이 걸러지지 않았다"; continue
        d = s.getlayer(Dot11)
        r = by_t[key]
        assert r[1] == d.type and r[2] == d.subtype and r[3] == int(d.FCfield), (r[:4], d.type, d.subtype, int(d.FCfield))
        assert (D.mac_str(r[4]), D.mac_str(r[5]), D.mac_str(r[6])) == (d.addr1, d.addr2, d.addr3)
        assert r[8] == s.dBm_AntSignal and r[9] == s.ChannelFrequency, (r[8], s.dBm_AntSignal, r[9], s.ChannelFrequency)
        assert r[7] == len(b) - len(bytes(s)[:s.len]) - 4, (r[7], len(b), s.len)
        if d.type == 0:
            assert r[10] == d[Dot11Elt].info.decode() and r[11] == D.freq_ch(s.ChannelFrequency) or r[11] == 6, (r[10], r[11])
        n_cmp += 1
    # 2) 표
    rows = D.analyse(fr, hops); D.show(rows)
    R = dict(((r["mac"], r["role"]), r) for r in rows)
    cam = R[(CAM, "sta")]
    assert rows[0]["mac"] == CAM and cam["candidate"] and cam["ch"] == 6 and cam["ssid"] == "room-507" and cam["heard"] == "focus", cam
    lo, hi = (800, 1200) if save else (930, 1100)                                                       # 평균 길이: (1380 x 3 + 620 + 240) / 5 = 1000. 짧게 들으면 더 흔들린다
    assert 40 < cam["per_s"] < 80 and lo < cam["mean_len"] < hi and -50 < cam["rssi"] < -44, cam
    lap = R[(LAP, "sta")]; assert not lap["candidate"] and lap["mean_len"] == 60 and 60 < lap["per_s"] < 120, lap
    ap = R[(AP1, "ap")]; assert ap["candidate"] and ap["mean_len"] == 1400 and "공유기" in ap["note"], ap
    ph = R[(PHONE, "sta")]; assert not ph["candidate"] and ph["random"] and ph["ch"] == 11 and ph["mean_len"] == 150, ph
    far = R[(FAR, "sta")]; assert far["candidate"] and far["weak"] and far["ch"] == 1, far
    assert [r["mac"] for r in rows if r["candidate"] and r["role"] == "sta"] == [CAM, FAR]
    # 3) 판별에 쓰는 '한 기기가 보낸 것' 뽑기
    tt, nn = D.sent_by(fr, CAM); f6 = [h for h in hops if h[3] == "focus" and h[2] == 6][0]
    k = [i for i, x in enumerate(tt) if f6[0] <= x < f6[1]]
    assert abs(len(k) / focus - cam["per_s"]) < 0.5 and abs(sum(nn[i] for i in k) / float(len(k)) - cam["mean_len"]) < 1.0
    if save:
        json.dump(dict(hops=hops, rows=rows, cam=CAM, cam_frames=len(tt), cam_bytes=int(sum(nn))), open(os.path.join(HERE, "fixture_mon.json"), "w", encoding="utf-8"), ensure_ascii=False)
        print("저장: fixture_mon.pcap (%d KB), fixture_mon.json" % (os.path.getsize(path) // 1024))
    print("detect 검사 통과: 프레임 %d개를 읽음, scapy 와 견준 것 %d개, 후보 = %s" % (len(fr), n_cmp, [r["mac"] for r in rows if r["candidate"]]))


if __name__ == "__main__":
    main(save="--save" in sys.argv)
