#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""자동 스캔: 파이의 모터를 움직이며 CSI 를 받고, 끝나면 카메라 방향을 계산한다. 노트북에서 실행.

  .venv/bin/python receiver-firmware/host/scan3.py --ch 11 --mac c6:26:06:68:85:83
  .venv/bin/python receiver-firmware/host/scan3.py --ch 11 --mac .. --plan up+tilt3 --truth 35 -40
  .venv/bin/python receiver-firmware/host/scan3.py --dry --truth 35 -40       장치 없이 흐름만 (가짜 데이터)

계획(--plan)
  tilt3      54.7도 기울여 받침 -120, 0, +120 도에서 세 번. 한 스캔에서만 골이 보이면 한 번 더 잰다 (기본)
             받침이 +120도 쪽에 있으면 거꾸로(+120, 0, -120) 잰다. 그래서 이어서 잴 때는 약 44초, 처음(0도에서)은 약 76초
  up+tilt3   세워서 한 번(논문 방식) + 기울여 세 번. 논문 방식과 한 번에 비교할 때
  tilt4      60도 기울여 네 번
  tilt6      54.7도 기울여 받침을 60도씩 돌려 여섯 번 (약 1분 45초). 화장실처럼 좁고 타일·거울이 많은 방에서 쓴다:
             반사 때문에 크게 틀리는 일이 절반으로 줄고, 틀렸을 때는 거의 다 경고가 뜬다 (sim/room_extra.py)
  up         세워서 한 번만 (논문 방식, 높이는 못 잰다)

준비: 파이에 rx3.py 를 복사해 둔다   scp receiver-firmware/pi/rx3.py pi@raspberrypi.local:~/detecam/rx3.py
      처음 한 번   ssh pi@raspberrypi.local "python3 ~/detecam/rx3.py home"
환경 변수: PI (기본 pi@raspberrypi.local), RX3 (기본 "python3 ~/detecam/rx3.py")
시각은 패킷도 모터도 파이의 시계로 찍히므로 노트북과 시계를 맞출 필요가 없다.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (".", "../core", "../pi"):                 # 저장소에서는 계산 코드가 core 에, 파이 코드가 pi 에 있다
    _p = os.path.normpath(os.path.join(HERE, _d))
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
import analyze3 as A

PI = os.environ.get("PI", "pi@raspberrypi.local")
RX3 = os.environ.get("RX3", "python3 ~/detecam/rx3.py")
TILT3 = 54.7356
PLANS = {"tilt3": [(-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)], "up+tilt3": [(-120.0, 0.0), (-120.0, TILT3), (0.0, TILT3), (120.0, TILT3)],
         "tilt4": [(-135.0, 60.0), (-45.0, 60.0), (45.0, 60.0), (135.0, 60.0)], "up": [(0.0, 0.0)],
         "tilt6": [(-150.0, TILT3), (-90.0, TILT3), (-30.0, TILT3), (30.0, TILT3), (90.0, TILT3), (150.0, TILT3)]}


def pi_run(args, timeout=180):
    """파이에서 rx3.py 를 실행하고 JSON 줄들을 돌려준다."""
    r = subprocess.run(["ssh", "-o", "BatchMode=yes", PI, RX3 + " " + args], capture_output=True, text=True, timeout=timeout)
    if r.stderr.strip():
        print("   [파이] " + r.stderr.strip().replace("\n", "\n   [파이] "))
    if r.returncode != 0:
        sys.exit("파이에서 명령이 실패했습니다: rx3.py %s" % args)
    return [json.loads(l) for l in r.stdout.splitlines() if l.strip().startswith("{")]


def order(frames, yaw, plan):
    """받침이 지금 있는 자리에서 가까운 끝부터 잰다 (받침을 덜 돌린다: 이어서 잴 때 65초 -> 44초). rx3.py 의 plan 명령과 같은 규칙."""
    if plan != "up+tilt3" and abs(yaw - frames[-1][0]) < abs(yaw - frames[0][0]):
        return frames[::-1]
    return frames


def start_capture(ch, mac, path):
    mopt = "-m %s" % mac if mac else ""
    cmd = ("sudo pkill -x tcpdump >/dev/null 2>&1; sudo ip link set wlan0 up; "
           "P=$(makecsiparams -c %d/20 -C 1 -N 1 %s) && sudo nexutil -Iwlan0 -s500 -b -l34 -v$P >/dev/null 2>&1 && "
           "exec sudo tcpdump -i wlan0 dst port 5500 -U -w - 2>/dev/null" % (ch, mopt))
    f = open(path, "wb")
    return subprocess.Popen(["ssh", "-o", "ServerAliveInterval=5", PI, cmd], stdout=f, stderr=subprocess.DEVNULL), f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ch", type=int, default=1); ap.add_argument("--mac")
    ap.add_argument("--plan", default="tilt3", choices=sorted(PLANS))
    ap.add_argument("--no-extra", action="store_true", help="한 스캔에서만 보여도 더 재지 않는다")
    ap.add_argument("--truth", nargs=2, type=float, metavar=("방위", "높이"), help="실제 방위와 높이 (도). 오차 계산용. 예: --truth 35 -40")
    ap.add_argument("--park", action="store_true", help="끝나면 세운 자세로 돌아간다")
    ap.add_argument("--off", action="store_true", help="끝나면 세운 자세로 돌아간 뒤 파이를 안전하게 끈다")
    ap.add_argument("--dry", action="store_true", help="장치 없이 가짜 데이터로 흐름만 본다")
    ap.add_argument("--no-show", action="store_true")
    a = ap.parse_args()
    truth = tuple(a.truth) if a.truth else None
    frames = list(PLANS[a.plan])
    if a.dry:
        sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "pi")))
        import rx3
        rig = rx3.Rig(dry=True); yaw0 = rig.deg("yaw")
    else:
        st = pi_run("status")[0]
        if not st.get("homed"):
            sys.exit("원점을 아직 잡지 않았습니다. 먼저:  ssh %s \"%s home\"" % (PI, RX3))
        yaw0 = float(st.get("yaw", 0.0))
    first = frames[0]; frames = order(frames, yaw0, a.plan)
    if frames[0] != first:
        print("받침이 %+.0f도에 있어서 가까운 쪽(%+.0f도)부터 잽니다." % (yaw0, frames[0][0]))
    outdir = time.strftime("data/scan3_%Y%m%d_%H%M%S") + ("_dry" if a.dry else "")
    os.makedirs(outdir, exist_ok=True)
    json.dump(dict(vars(a), truth=truth, frames=frames), open(os.path.join(outdir, "meta.json"), "w"), ensure_ascii=False, indent=1)
    recs = []; proc = pcap = None; t = C = None

    if a.dry:
        cam = truth or (35.0, -42.0)
        do_scan = lambda y, p: rig.scan(y, p)
        get = lambda: A.fake_packets(recs, cam, seed=5)
        print("(--dry) 가짜 카메라: 방위 %+.0f도, 높이 %+.0f도" % cam)
    else:
        proc, pcap = start_capture(a.ch, a.mac, os.path.join(outdir, "csi.pcap")); time.sleep(2.5)
        do_scan = lambda y, p: pi_run("scan --yaw %.3f --pitch %.3f" % (y, p))[0]
        def get():
            pcap.flush(); time.sleep(0.8)
            tt, CC, _ = A.packet_levels(os.path.join(outdir, "csi.pcap"), a.mac)
            return tt, CC
    try:
        for i, (y, p) in enumerate(frames):
            print("스캔 %d/%d: 받침 %+.0f도, 끄덕임 %.1f도 …" % (i + 1, len(frames), y, p), flush=True)
            recs.append(do_scan(y, p))
        t, C = get()
        res, curves, extra = A.solve(recs, t, C, truth=truth, quiet=True)
        nxt = None if (a.no_extra or a.plan == "up") else A.suggest_extra(res, extra, TILT3)
        if nxt:
            print("한 스캔에서만 골이 보였습니다 -> 받침 %+.0f도에서 한 번 더 잽니다." % nxt[0], flush=True)
            recs.append(do_scan(nxt[0], nxt[1]))
            t, C = get()
    finally:
        with open(os.path.join(outdir, "motion.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        if proc:
            proc.terminate(); time.sleep(0.5); pcap.close()
        if a.park and not a.dry and not a.off:
            pi_run("park")
    if t is None:
        sys.exit("스캔을 마치지 못했습니다. 지금까지의 시간표는 %s/motion.jsonl 에 있습니다." % outdir)
    if not a.dry:
        t, C, _ = A.packet_levels(os.path.join(outdir, "csi.pcap"), a.mac)
    res, curves, _ = A.solve(recs, t, C, truth=truth)
    json.dump(res, open(os.path.join(outdir, "result.json"), "w"), ensure_ascii=False, indent=1)
    A.plot(res, curves, os.path.join(outdir, "result.png"), show=not (a.no_show or a.dry), title=os.path.basename(outdir))
    print("저장:", outdir)
    if a.off and not a.dry:
        print("파이를 끕니다 (세운 자세로 돌아간 뒤). 초록 불이 꺼지면 보조배터리 버튼을 눌러 전원을 끊어도 됩니다.")
        subprocess.run(["ssh", "-o", "BatchMode=yes", PI, RX3 + " off"], timeout=120)


if __name__ == "__main__":
    main()
