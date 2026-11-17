"""가진 장비(라즈베리파이 + nexmon)로 이미 받아 둔 CSI 로, '세기만 재서 위상 되찾기'가 버틸 만큼 카메라 신호의 진폭이 고른지 본다. 돈이 들지 않는 확인이다.

    python amp_stability.py 파일.pcap [--mac aa:bb:cc:dd:ee:ff] [--fc 0x88 0x08] [--gap 0.1]

무엇을 보나
  1. 이어진 두 패킷 사이에 진폭(서브캐리어 평균)이 얼마나 달라지나 (dB): 고른 정도(가운데 덩어리의 표준편차)와 1dB 넘게 뛰는 비율
  2. 진폭의 모양(서브캐리어마다의 비)이 얼마나 달라지나 (dB)
  3. 이어진 패킷 넷의 진폭을 그대로 써서 네 단계 재기를 흉내 내면 위상이 몇 도 틀리나
     (가) 그대로  (나) 진폭이 1dB 넘게 뛴 묶음은 버리고 다시 잰다고 할 때

주의: 파이(nexmon)의 CSI 는 수신 이득이 바뀌면 진폭이 통째로 뛴다. ESP32 는 패킷마다 이득 값을 알려 주므로 그만큼은 되돌릴 수 있다. 그래서 (가)는 실제보다 나쁜 쪽의 값이다.
"""
import sys, struct, argparse, collections
import numpy as np


def read_csi(path):
    """nexmon CSI pcap (bcm43455c0) -> (시각, MAC, RSSI, 프레임 종류, CSI[64]) 의 목록"""
    d = open(path, "rb").read(); magic = struct.unpack("<I", d[:4])[0]
    en = "<" if magic in (0xa1b2c3d4, 0xa1b23c4d) else ">"; nano = magic in (0xa1b23c4d, 0x4d3cb2a1)
    l2 = {1: 14, 113: 16, 276: 20}.get(struct.unpack(en + "I", d[20:24])[0], 14); out = []; i = 24
    while i + 16 <= len(d):
        ts, fr, cl, _ = struct.unpack(en + "IIII", d[i:i + 16]); p = d[i + 16:i + 16 + cl]; i += 16 + cl
        if len(p) < l2 + 20: continue
        u = p[l2 + (p[l2] & 0x0f) * 4 + 8:]
        if len(u) < 18 + 256 or u[0:2] != b"\x11\x11": continue
        iq = np.frombuffer(u[18:18 + 256], dtype="<i2").astype(np.float64)
        out.append((ts + fr / (1e9 if nano else 1e6), ":".join("%02x" % b for b in u[4:10]), struct.unpack("b", u[2:3])[0], u[3], iq[0::2] + 1j * iq[1::2]))
    return out


def rstd(x):
    """가운데 덩어리의 표준편차 (튀는 값에 끌리지 않게 사분위 폭으로)"""
    q1, q3 = np.percentile(x, [25, 75]); return 0.7413 * (q3 - q1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pcap"); ap.add_argument("--mac"); ap.add_argument("--fc", nargs="*", default=["0x88", "0x08"]); ap.add_argument("--gap", type=float, default=0.1)
    a = ap.parse_args(); rows = read_csi(a.pcap); fcs = {int(f, 16) for f in a.fc}
    cnt = collections.Counter(r[1] for r in rows if r[3] in fcs)
    if not cnt:
        print("맞는 프레임이 없다. 받은 것 %d개" % len(rows)); return
    mac = a.mac or cnt.most_common(1)[0][0]; R = [r for r in rows if r[1] == mac and r[3] in fcs]
    if len(R) < 30:
        print("%s  맞는 프레임이 너무 적다 (%d개)" % (a.pcap.split("/")[-1], len(R))); return
    t = np.array([r[0] for r in R]); H = np.abs(np.array([r[4] for r in R])); rssi = np.array([r[2] for r in R])
    H = H[:, np.r_[1:29, 36:64]]; med = np.median(H, 0); ok = (med > 0.2 * np.median(med)) & (med < 5 * np.median(med))      # 20MHz 의 쓰는 서브캐리어 56개 가운데 값이 이상한 것은 뺀다
    H = H[:, ok]; A = H.mean(1); keep = A > 0; t, H, A, rssi = t[keep], H[keep], A[keep], rssi[keep]
    dur = t[-1] - t[0]; dA = 20 * np.log10(A[1:] / A[:-1]); near = np.diff(t) < a.gap; dA = dA[near]
    print("%s  패킷 %d개, %.0f초, 초당 %.1f개, 쓰는 서브캐리어 %d개, RSSI %d~%d dBm" % (a.pcap.split("/")[-1], len(t), dur, len(t) / max(dur, 1e-9), int(ok.sum()), rssi.min(), rssi.max()))
    if len(dA) < 30:
        print("  이어진 패킷이 너무 적다 (%d쌍)" % len(dA)); return
    print("  1. 이어진 두 패킷의 진폭 차: 고른 정도 %.2f dB, 1dB 넘게 뛴 비율 %.1f%%, 3dB 넘게 %.1f%%  (%d쌍)" % (rstd(dA) / np.sqrt(2), 100 * (np.abs(dA) > 1).mean(), 100 * (np.abs(dA) > 3).mean(), len(dA)))
    sh = H / A[:, None]; ds = 20 * np.log10(np.maximum(sh[1:], 1e-6) / np.maximum(sh[:-1], 1e-6))[near]
    print("  2. 진폭 모양의 차 (서브캐리어마다): 고른 정도 %.2f dB" % (rstd(ds.ravel()) / np.sqrt(2)))
    rng = np.random.default_rng(1); idx = [i for i in range(len(t) - 3) if t[i + 3] - t[i] < 2 * a.gap]
    if len(idx) < 30:
        print("  3. 이어진 패킷 넷이 너무 적다 (%d묶음)" % len(idx)); return
    steps = np.exp(1j * np.pi / 2 * np.arange(4)); errs = {1.0: ([], []), 0.5: ([], [])}
    for i in idx:
        g = H[i:i + 4] / H[i:i + 4].mean(0, keepdims=True)                                  # [4, 서브캐리어] 그 넷 사이의 진폭 비 (전체 이득과 모양을 함께)
        jump = 20 * np.log10(A[i:i + 4].max() / A[i:i + 4].min()) > 1.0
        for rho, (e_all, e_ok) in errs.items():
            th = rng.uniform(0, 2 * np.pi); I = (g ** 2) * np.abs(1 + rho * steps[:, None] * np.exp(1j * th)) ** 2
            z = ((I[0] - I[2]) - 1j * (I[1] - I[3])).mean(); e = np.degrees(np.abs(np.angle(z * np.exp(-1j * th))))      # 서브캐리어 평균 뒤의 위상
            e_all.append(e)
            if not jump: e_ok.append(e)
    for rho, (e_all, e_ok) in errs.items():
        e_all, e_ok = np.array(e_all), np.array(e_ok)
        print("  3. 네 단계 흉내 (안테나 신호가 기준의 %.1f배): (가) 그대로 위상 오차 가운데값 %.1f도, 90%%가 %.1f도 안  (나) 뛴 묶음을 버리면 %.1f도, %.1f도 (버린 비율 %.0f%%)"
              % (rho, np.median(e_all), np.percentile(e_all, 90), np.median(e_ok), np.percentile(e_ok, 90), 100 * (1 - len(e_ok) / len(e_all))))


if __name__ == "__main__":
    main()
