#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""S4 응답 유도 실험 읽기: 주소 필터 없이 받은 nexmon CSI 기록에서 '응답 프레임(ACK)의 CSI'를 세고, 누가 보낸 것인지 가리고,
판을 손으로 돌린 기록이 있으면 응답 프레임만으로도 골이 같은 곳에 생기는지 본다 (docs/의심 점수.md 10절 S4).

  python3 ack_check.py 폴더 --cam 카메라주소 --peer 상대주소 [--truth 각도]
  python3 ack_check.py csi.pcap --cam 카메라주소 --peer 상대주소
  python3 ack_check.py --selftest

  폴더    tools/manual_scan.py 가 만든 폴더 (csi.pcap, angles.csv, meta.json). angles.csv 가 있으면 골도 찾는다
  --cam   카메라 주소          --peer  카메라가 붙은 쪽 (폰 핫스팟이나 공유기) 주소

ACK 에는 보낸 쪽 주소가 없다 (받는 쪽 주소만 있다). 그래서 nexmon 이 적는 '주소' 칸이 ACK 에서는 엉뚱한 값이다.
누가 보낸 ACK 인지를 두 가지로 가리고, 둘이 얼마나 맞는지도 낸다.
  ① 순서: ACK 는 데이터를 받은 쪽이 곧바로(16µs 뒤) 보낸다. 바로 앞의 CSI 가 상대가 보낸 데이터면 카메라의 ACK, 카메라가 보낸 데이터면 상대의 ACK.
  ② 모양: 같은 자리에서 보낸 프레임은 서브캐리어별 세기 모양이 닮는다. 카메라 데이터의 모양과 상대 데이터의 모양 가운데 더 닮은 쪽.
ACK 는 옛 OFDM(서브캐리어 52개)으로 오므로, 견줄 때는 모든 프레임에서 가운데 52개만 쓴다."""
import argparse, csv, json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in ("../pi", "../core", "."):
    _p = os.path.normpath(os.path.join(HERE, _d))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import csi as CSI                                           # noqa: E402
from fusion import trough                                   # noqa: E402

ACK, CTS, BA, BAR = 0xd4, 0xc4, 0x94, 0x84
NAMES = dict(CSI.FC_NAMES); NAMES.update({ACK: "ACK", CTS: "CTS", 0xb4: "RTS"})
DATA = (0x88, 0x08, 0x48, 0xc8)
LEG = np.r_[6:32, 33:59]                                    # fftshift 뒤 -26..-1, 1..26: 옛 OFDM 과 HT 가 함께 쓰는 서브캐리어
GAP = 0.004                                                 # 앞 프레임과 이만큼 안에 온 ACK 만 순서로 가린다 (초)


def levels(pk):
    """패킷마다 52개 서브캐리어의 세기(dB), 자동 이득 조절을 되돌린 값 (pi/csi.py 의 packet_levels 와 같은 계산)"""
    A = np.array([np.abs(np.fft.fftshift(p[5]))[LEG] for p in pk])
    return 20 * np.log10(np.maximum(A, 1)) - 10 * np.log10(np.maximum(np.mean(A ** 2, 1), 1e-9))[:, None] + np.array([p[2] for p in pk], float)[:, None]


def corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-9))


def attribute(pk, cam, peer):
    """응답 프레임(ACK·CTS)마다 (순서로 본 보낸 쪽, 모양으로 본 보낸 쪽). 'cam' / 'peer' / None"""
    pk = [p for p in pk if len(p[5]) == 64]
    L = levels(pk) if pk else np.zeros((0, len(LEG)))
    shape = L - L.mean(1, keepdims=True)                    # 전체 세기는 빼고 모양만
    ref = {}
    for who, m in (("cam", cam), ("peer", peer)):
        idx = [i for i, p in enumerate(pk) if p[1] == m and p[3] in DATA]
        ref[who] = np.median(shape[idx], 0) if len(idx) >= 5 else None
    out = []
    for i, p in enumerate(pk):
        if p[3] not in (ACK, CTS):
            continue
        by_order = None
        if i > 0 and p[0] - pk[i - 1][0] <= GAP and pk[i - 1][3] in DATA:
            by_order = {peer: "cam", cam: "peer"}.get(pk[i - 1][1])
        by_shape = None
        if ref["cam"] is not None and ref["peer"] is not None:
            by_shape = "cam" if corr(shape[i], ref["cam"]) >= corr(shape[i], ref["peer"]) else "peer"
        out.append(dict(i=i, t=p[0], rssi=p[2], order=by_order, shape=by_shape))
    return pk, L, out


def report(pk, cam, peer):
    dur = max(pk[-1][0] - pk[0][0], 1e-6)
    print("기록 %.1f초, CSI %d개 (초당 %.0f개)" % (dur, len(pk), len(pk) / dur))
    kinds = {}
    for p in pk:
        k = p[1] if p[3] not in (ACK, CTS) else "(주소 없음)"
        kinds.setdefault((p[3], k), []).append(p[2])
    print("  프레임 종류별 (많은 순):")
    for (fc, m), r in sorted(kinds.items(), key=lambda kv: -len(kv[1]))[:12]:
        who = {cam: "  <- 카메라", peer: "  <- 상대"}.get(m, "")
        print("    %-10s %-19s %6d개  초당 %6.1f  세기 %5.0fdBm%s" % (NAMES.get(fc, hex(fc)), m, len(r), len(r) / dur, np.mean(r), who))
    pk64, L, acks = attribute(pk, cam, peer)
    n = len(acks)
    if not n:
        print("  응답 프레임(ACK·CTS)의 CSI 가 하나도 없음 -> 이 칩·펌웨어로는 응답 프레임 CSI 가 안 나오는 것으로 본다 (다른 조건도 바꿔 보고 결론)")
        return pk64, L, acks
    oc = sum(1 for a in acks if a["order"] == "cam"); op = sum(1 for a in acks if a["order"] == "peer")
    sc = sum(1 for a in acks if a["shape"] == "cam")
    both = [a for a in acks if a["order"] and a["shape"]]
    agree = sum(1 for a in both if a["order"] == a["shape"])
    print("  응답 프레임 CSI %d개 (초당 %.1f개)" % (n, n / dur))
    print("    순서로 가림: 카메라가 보냄 %d, 상대가 보냄 %d, 모름 %d" % (oc, op, n - oc - op))
    print("    모양으로 가림: 카메라가 보냄 %d, 상대가 보냄 %d" % (sc, n - sc) if acks[0]["shape"] else "    모양으로 가림: 카메라나 상대의 데이터 CSI 가 5개 미만이라 못 함")
    if both:
        print("    두 방법이 맞은 비율 %.0f%% (%d개 중) -> 높으면 '주소 없는 ACK 도 누구 것인지 가릴 수 있다'" % (100.0 * agree / len(both), len(both)))
    for who in ("cam", "peer"):
        r = [a["rssi"] for a in acks if (a["order"] or a["shape"]) == who]
        if r:
            print("    %s 의 응답 세기 가운데 %.0fdBm (%d개)" % ("카메라" if who == "cam" else "상대", np.median(r), len(r)))
    return pk64, L, acks


def curve(t, C, rows):
    """hand_check.py 와 같은 계산: 각도마다 가운데값 -> 골이 뚜렷한 서브캐리어 -> 5도 칸 -> 골 가운데"""
    g = {}
    for r in rows:
        if r["angle"] == "base":
            continue
        m = (t >= float(r["t0"])) & (t <= float(r["t1"]))
        if m.sum():
            g.setdefault(float(r["angle"]), []).append(C[m])
    if len(g) < 5:
        return None
    ang = np.array(sorted(g)); V = np.stack([np.median(np.vstack(g[a]), axis=0) for a in ang])
    db = V - np.median(V, axis=0)
    contrast = np.median(db, axis=0) - db.min(axis=0)
    sel = np.argsort(-contrast)[: max(5, db.shape[1] // 3)]
    cv = np.median(db[:, sel], axis=1)
    grid = np.arange(2.5, 360.0, 5.0)
    y = np.interp(grid, np.r_[ang - 360, ang, ang + 360], np.r_[cv, cv, cv])
    psi, depth, _ = trough(grid, y)
    per_angle = np.mean([sum(len(x) for x in g[a]) for a in ang])
    return psi % 360, depth, per_angle


def scan(pk64, L, acks, cam, rows, truth):
    t = np.array([p[0] for p in pk64])
    sets = [("카메라 데이터", [i for i, p in enumerate(pk64) if p[1] == cam and p[3] in DATA]),
            ("카메라 응답 (순서로 가림)", [a["i"] for a in acks if a["order"] == "cam"]),
            ("카메라 응답 (모양으로 가림)", [a["i"] for a in acks if a["shape"] == "cam"])]
    print("  판을 돌린 기록으로 골 찾기 (5도 칸, 52개 서브캐리어):")
    got = {}
    for name, idx in sets:
        if len(idx) < 20:
            print("    %-24s CSI %d개 -> 너무 적어 건너뜀" % (name, len(idx))); continue
        r = curve(t[idx], L[idx], rows)
        if r is None:
            print("    %-24s 잰 각도가 5개 미만" % name); continue
        psi, depth, per = r; got[name] = psi
        err = "" if truth is None else ", 실제 %g도 -> 오차 %.1f도" % (truth, abs((psi - truth + 180) % 360 - 180))
        print("    %-24s 골 가운데 %5.1f도, 깊이 %4.1fdB, 각도마다 CSI %.0f개%s" % (name, psi, depth, per, err))
    if "카메라 데이터" in got:
        for k, v in got.items():
            if k != "카메라 데이터":
                print("    %s 와 카메라 데이터의 차이 %.1f도" % (k, abs((v - got["카메라 데이터"] + 180) % 360 - 180)))


def run(src, cam, peer, truth=None):
    folder = src if os.path.isdir(src) else None
    pcap = os.path.join(folder, "csi.pcap") if folder else src
    meta = json.load(open(os.path.join(folder, "meta.json"))) if folder and os.path.exists(os.path.join(folder, "meta.json")) else {}
    pk = CSI.read_csi(pcap)
    if not pk:
        sys.exit("CSI 가 없습니다: " + pcap)
    print(pcap)
    pk64, L, acks = report(pk, cam.lower(), peer.lower())
    ang = os.path.join(folder, "angles.csv") if folder else None
    if ang and os.path.exists(ang):
        truth = truth if truth is not None else meta.get("truth")
        scan(pk64, L, acks, cam.lower(), list(csv.DictReader(open(ang))), None if truth is None else float(truth))


# ----------------------------------------------------------------- 자체 검사: 가짜 기록을 만들어 돌린다
def _write_pcap(path, recs):
    """nexmon 이 UDP 5500 으로 보내는 꼴 그대로 (이더넷 + IP + UDP + 머리 18바이트 + CSI 64개)"""
    with open(path, "wb") as f:
        f.write(struct.pack("<IHHiIII", 0xa1b2c3d4, 2, 4, 0, 0, 65535, 1))
        for t, mac, rssi, fc, seq, H in recs:
            iq = np.empty(128, "<i2"); iq[0::2] = np.clip(np.round(H.real), -32767, 32767); iq[1::2] = np.clip(np.round(H.imag), -32767, 32767)
            u = b"\x11\x11" + struct.pack("bB", int(rssi), fc) + bytes(int(x, 16) for x in mac.split(":")) + struct.pack("<HHHH", seq << 4, 0, 0x1006, 0x4345) + iq.tobytes()
            p = b"\x00" * 12 + b"\x08\x00" + b"\x45" + b"\x00" * 19 + b"\x00" * 8 + u
            sec = int(t); f.write(struct.pack("<IIII", sec, int(round((t - sec) * 1e6)), len(p), len(p)) + p)


def selftest(tmp):
    rnd = np.random.default_rng(5); cam, peer = "c6:26:aa:01:02:03", "6a:11:22:33:44:55"; truth = 140.0
    base_cam = 300 * (1 + 0.6 * np.sin(np.arange(64) / 3.0)); base_peer = 300 * (1 + 0.6 * np.cos(np.arange(64) / 5.0 + 1))
    sc = np.fft.fftshift(np.arange(64) - 32)
    def dip(a):                                             # 판이 카메라 쪽을 가리면 일부 서브캐리어에 골
        d = np.exp(-0.5 * (((a - truth + 180) % 360 - 180) / 12.0) ** 2)
        return 1 - 0.8 * d * (0.5 + 0.5 * np.cos(sc / 4.0))
    recs = []; rows = [dict(angle="base", t0=0, t1=0)]; t = 100.0
    for a in np.arange(0, 360, 10.0):
        t0 = t
        while t < t0 + 3.0:
            g = rnd.normal(1, 0.03, 64)
            if rnd.random() < 0.5:                           # 카메라가 올림 -> 상대가 ACK (주소 칸은 엉뚱한 값)
                recs.append((t, cam, -45, 0x88, 1, base_cam * dip(a) * g)); recs.append((t + 0.0002, "%02x:00:00:00:00:%02x" % (rnd.integers(256), rnd.integers(256)), -70, ACK, 0, base_peer * rnd.normal(1, 0.03, 64)))
            else:                                            # 상대가 보냄 (핑 등) -> 카메라가 ACK
                recs.append((t, peer, -70, 0x88, 1, base_peer * g)); recs.append((t + 0.0002, "%02x:11:00:00:00:%02x" % (rnd.integers(256), rnd.integers(256)), -45, ACK, 0, base_cam * dip(a) * rnd.normal(1, 0.03, 64)))
            t += 0.03
        rows.append(dict(angle=a, t0=t0, t1=t)); t += 0.5
    os.makedirs(tmp, exist_ok=True)
    _write_pcap(os.path.join(tmp, "csi.pcap"), recs)
    with open(os.path.join(tmp, "angles.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["angle", "t0", "t1"]); w.writeheader(); w.writerows(rows)
    json.dump(dict(truth=truth, mac=cam), open(os.path.join(tmp, "meta.json"), "w"))
    run(tmp, cam, peer)
    pk64, L, acks = attribute(CSI.read_csi(os.path.join(tmp, "csi.pcap")), cam, peer)
    good = sum(1 for a in acks if a["order"] == a["shape"]) / float(len(acks))
    assert good > 0.95, good
    t = np.array([p[0] for p in pk64]); idx = [a["i"] for a in acks if a["order"] == "cam"]
    psi = curve(t[idx], L[idx], rows)[0]
    assert abs((psi - truth + 180) % 360 - 180) < 10, psi
    print("ack_check 자체 검사 통과")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?"); ap.add_argument("--cam"); ap.add_argument("--peer"); ap.add_argument("--truth", type=float)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        import tempfile
        selftest(os.path.join(tempfile.mkdtemp(), "ack"))
    elif a.src and a.cam and a.peer:
        run(a.src, a.cam, a.peer, a.truth)
    else:
        ap.print_help()
