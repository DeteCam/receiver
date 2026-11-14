#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""폰에서 시키는 동작(손으로 움직이기, 멈춤, 상태 보기, 내려받기)을 가짜 모드로 불러 본다.
    python3 tests/test_control.py [파이썬 실행 파일]
가짜 모드라 모터는 돌지 않는다. 다만 모터 명령이 실제 시간이 걸리게 해 놓고(DETECAM_DRY_SLEEP), 도는 도중에 '멈춤'이 먹는지를 본다."""
import json, os, subprocess, sys, time
from urllib.request import urlopen, Request
HERE = os.path.dirname(os.path.abspath(__file__)); PI = os.path.join(HERE, "..", "pi")
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
PORT = 8098; URL = "http://127.0.0.1:%d" % PORT


def get(path, raw=False):
    r = urlopen(URL + path, timeout=10); b = r.read()
    return (b, r.headers.get("Content-Type")) if raw else json.loads(b.decode("utf-8"))


def post(path, body=None):
    try:
        r = urlopen(Request(URL + path, data=json.dumps(body or {}).encode("utf-8"), headers={"Content-Type": "application/json"}), timeout=10)
        return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return json.loads(e.read().decode("utf-8")) if hasattr(e, "read") else dict(error=str(e))


def wait(limit=60, ok=("idle",)):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = get("/api/status")
        if s["status"]["state"] != "busy":
            assert s["status"]["state"] in ok, s["status"]
            return s
        time.sleep(0.03)
    raise RuntimeError("끝나지 않았습니다: %s" % get("/api/status")["status"])


def main():
    env = dict(os.environ, DETECAM_DRY_SLEEP="0.05")                 # 모터 명령이 실제 걸리는 시간의 1/20 만큼 걸린다
    p = subprocess.Popen([PY, os.path.join(PI, "node.py"), "--dry", "--fast", "--port", str(PORT)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(100):
            try:
                st = get("/api/status"); break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("node.py 가 뜨지 않았습니다.")
        assert st["joints"] == dict(yaw=0.0, pitch=0.0, spin=None, homed=False), st["joints"]

        # 화면과 '홈 화면에 추가'용 파일
        html = get("/", raw=True)[0].decode("utf-8")
        for word in ("/api/stop", "/api/jog", "/api/goto", "/api/export", "manifest.json", "멈춤"):
            assert word in html, word
        man = get("/manifest.json"); assert man["display"] == "standalone" and len(man["icons"]) == 2, man
        for ic in man["icons"]:
            b, ct = get(ic["src"], raw=True); assert ct == "image/png" and b[:8] == b"\x89PNG\r\n\x1a\n" and len(b) > 2000, (ic, ct, len(b))

        # 상태 보기
        h = get("/api/health"); assert h["dry"] and h["throttled"] == "0x0" and h["undervoltage_now"] is False and h["homed"] is False, h

        # 손으로 움직이기: 처음 움직일 때 원점을 잡는다
        assert post("/api/goto", {"yaw": 40, "pitch": 30}).get("ok"); s = wait()
        assert abs(s["joints"]["yaw"] - 40) < 0.2 and abs(s["joints"]["pitch"] - 30) < 0.2 and s["joints"]["homed"], s["joints"]
        assert post("/api/jog", {"joint": "yaw", "deg": -15}).get("ok"); s = wait(); assert abs(s["joints"]["yaw"] - 25) < 0.2, s["joints"]
        assert post("/api/jog", {"joint": "pitch", "deg": 10}).get("ok"); s = wait(); assert abs(s["joints"]["pitch"] - 40) < 0.2, s["joints"]
        assert post("/api/jog", {"joint": "spin", "deg": 90}).get("ok"); wait()
        assert post("/api/spin", {"turns": 0.5}).get("ok"); wait()
        assert post("/api/zero", {"joint": "spin"}).get("ok"); wait()
        assert post("/api/release").get("ok"); wait()

        # 범위 밖, 틀린 값은 움직이기 전에 거절한다
        for path, body, word in (("/api/goto", {"pitch": 100}, "범위"), ("/api/goto", {"yaw": -200}, "범위"), ("/api/goto", {}, "yaw"), ("/api/jog", {"joint": "pitch", "deg": 60}, "범위"),
                                 ("/api/jog", {"joint": "arm", "deg": 5}, "모르는"), ("/api/spin", {"turns": 9}, "5바퀴"), ("/api/zero", {"joint": "yaw"}, "판"), ("/api/jog", {"joint": "spin", "deg": 999}, "360")):
            r = post(path, body)
            if r.get("ok"):                                           # 일을 시작한 뒤에 걸러지는 것 (spin, jog 의 값 검사)
                s = wait(ok=("idle", "error")); assert s["status"]["state"] == "error" and word in s["status"]["message"], (path, body, s["status"])
            else:
                assert word in r.get("error", ""), (path, body, r)
        s = get("/api/status"); assert abs(s["joints"]["pitch"] - 40) < 0.2 and abs(s["joints"]["yaw"] - 25) < 0.2, s["joints"]      # 거절된 명령으로는 움직이지 않았다

        # 멈춤: 모터가 도는 도중에
        assert post("/api/stop").get("message") == "하고 있는 일이 없습니다."
        assert post("/api/goto", {"yaw": -170, "pitch": 0}).get("ok")
        time.sleep(0.4); assert get("/api/status")["status"]["state"] == "busy"
        assert post("/api/goto", {"yaw": 0}).get("error", "").startswith("지금"), "바쁠 때 다른 명령은 거절해야 한다"
        t0 = time.time(); assert post("/api/stop").get("ok"); s = wait(limit=5); dt = time.time() - t0
        assert "멈췄습니다" in s["status"]["message"] and s["joints"]["homed"] is False and dt < 2.0, (s["status"], dt)
        # 멈춘 뒤 다시 시키면 원점부터 잡고 간다
        assert post("/api/goto", {"yaw": 10, "pitch": 0}).get("ok"); s = wait(); assert s["joints"]["homed"] and abs(s["joints"]["yaw"] - 10) < 0.2, s["joints"]

        # 멈춤: 판별 도중에 (모터 없이 기다리는 일)
        post("/api/config", {"ch": 6, "mac": "fa:ke:00:00:00:01"})
        assert post("/api/identify", {"mode": "leave"}).get("ok"); time.sleep(0.3)
        assert get("/api/status")["status"]["prompt"] is not None
        t0 = time.time(); assert post("/api/stop").get("ok"); s = wait(limit=5); assert "멈췄습니다" in s["status"]["message"] and s["status"]["prompt"] is None and time.time() - t0 < 1.5, s["status"]
        assert s["ident_last"] is None, "멈춘 판별은 결과를 남기지 않는다"

        # 멈춤: 재는 도중에. 잰 자리는 늘지 않는다
        st = get("/api/status"); old_numpy = tuple(int(v) for v in st["numpy"].split(".")[:2]) < (1, 17)
        if old_numpy:                                            # 파이의 numpy 1.16: 가짜 방을 만드는 코드만 못 돈다 (test_node_dry.py 와 같은 사정)
            print("재는 도중 멈춤, 끝까지 재기: numpy %s 라 가짜 방을 못 만들어 건너뜀 (노트북의 파이썬에서 본다)" % st["numpy"])
        else:
            assert post("/api/measure", {"plan": "tilt3", "pose": {"x": 1.0, "y": 1.0, "z": 0.0, "heading": 0.0}}).get("ok"); time.sleep(0.8)
            assert post("/api/stop").get("ok"); s = wait(limit=6); assert "멈췄습니다" in s["status"]["message"] and len(s["positions"]) == 0, (s["status"], len(s["positions"]))

            # 끝까지 한 번 재고, 결과를 통째로 내려받는다
            assert post("/api/measure", {"plan": "tilt3", "pose": {"x": 1.0, "y": 1.0, "z": 0.0, "heading": 0.0}}).get("ok"); s = wait(limit=120)
            assert len(s["positions"]) == 1, s["status"]
            ex = get("/api/export"); assert len(ex["positions"]) == 1 and "obs" in ex["positions"][0] and "zone" in ex and ex["config"]["mac"] == "fa:ke:00:00:00:01", list(ex)
            cv = ex["positions"][0]["curves"]; assert len(cv) >= 3 and len(cv[0]["lv"]) == 72 and cv[0]["bin_deg"] == 5.0 and abs(cv[0]["pitch"] - 54.74) < 0.1, (len(cv), len(cv[0]["lv"]))      # 세기 곡선도 같이 나온다
        ex = get("/api/export"); assert "positions" in ex and "zone" in ex and ex["config"]["mac"] == "fa:ke:00:00:00:01", list(ex)
        assert post("/api/park").get("ok"); s = wait(); assert s["joints"]["yaw"] == 0.0 and s["joints"]["pitch"] == 0.0
        print("폰 조종 검사 통과: 손으로 움직이기, 범위 밖 거절 8가지, 멈춤 %s, 상태, 내려받기, 홈 화면 아이콘 (파이썬 %s)" % ("2가지(모터·판별)" if old_numpy else "3가지(모터·판별·재기)", st.get("python", "?")))
    finally:
        p.terminate()
        try:
            out = p.communicate(timeout=5)[0].decode("utf-8", "replace")
        except Exception:
            p.kill(); out = ""
        if sys.exc_info()[0] is not None:
            print(out[-3000:])


if __name__ == "__main__":
    main()
