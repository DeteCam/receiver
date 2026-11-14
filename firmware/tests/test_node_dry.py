#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""수신기 프로그램(node.py)을 가짜 모드로 띄워 네 단계를 차례로 불러 본다: 탐지 -> 판별 -> 위치 재기(두 자리) -> 구간 -> 마커.
    python3 tests/test_node_dry.py [파이썬 실행 파일]        (파이와 같은 파이썬 3.7 로 해 보려면 그 경로를 준다)
장치가 없어도 되는 검사다. 모터·CSI·동글이 실제로 도는지는 실물에서 확인해야 한다."""
import json, os, subprocess, sys, time
try:
    from urllib.request import urlopen, Request
except ImportError:
    sys.exit("파이썬 3 로 실행하세요.")
HERE = os.path.dirname(os.path.abspath(__file__)); PI = os.path.join(HERE, "..", "pi")
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
PORT = 8097; URL = "http://127.0.0.1:%d" % PORT


def get(path):
    return json.loads(urlopen(URL + path, timeout=10).read().decode("utf-8"))


def post(path, body=None):
    try:
        r = urlopen(Request(URL + path, data=json.dumps(body or {}).encode("utf-8"), headers={"Content-Type": "application/json"}), timeout=10)
        return json.loads(r.read().decode("utf-8"))
    except Exception as e:
        return json.loads(e.read().decode("utf-8")) if hasattr(e, "read") else dict(error=str(e))


def wait(what, limit=240):
    t0 = time.time(); seen = []
    while time.time() - t0 < limit:
        s = get("/api/status")["status"]
        if s.get("prompt"):
            seen.append(s["prompt"])
        if s["state"] != "busy":
            assert s["state"] == "idle", (what, s)
            return s, seen
        time.sleep(0.05)
    raise RuntimeError("%s 이(가) 끝나지 않았습니다." % what)


def main():
    p = subprocess.Popen([PY, os.path.join(PI, "node.py"), "--dry", "--fast", "--port", str(PORT), "--fake-cam", "2.4", "3.1", "0.9"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        for _ in range(100):
            try:
                st = get("/api/status"); break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError("node.py 가 뜨지 않았습니다.")
        assert st["dry"] and st["survey"] == [] and set(st["plans"]) == {"walk", "light", "leave"}, st
        assert abs(st["plans"]["walk"]["seconds"] - 55.0) < 1e-6 and abs(st["plans"]["leave"]["seconds"] - 83.0) < 1e-6, st["plans"]
        html = urlopen(URL + "/", timeout=10).read().decode("utf-8"); assert "판별" in html and "/api/detect" in html

        # 1 탐지
        assert post("/api/identify", {"mode": "walk"}).get("ok"); s, _ = None, None
        s = get("/api/status")
        for _ in range(100):
            s = get("/api/status")
            if s["status"]["state"] != "busy":
                break
            time.sleep(0.05)
        assert s["status"]["state"] == "error" and "고르세요" in s["status"]["message"], s["status"]          # 기기를 고르기 전에는 판별을 못 한다
        assert post("/api/detect").get("ok"); assert "error" in post("/api/measure")                              # 하는 중에는 다른 일을 받지 않는다
        s, _ = wait("탐지"); rows = get("/api/status")["survey"]
        assert rows[0]["candidate"] and rows[0]["role"] == "sta" and rows[0]["mac"].endswith(":01"), rows[0]
        print("탐지:", s["message"])

        # 2 판별: 카메라인 기기와 아닌 기기
        out = {}
        for mac, mode in ((rows[0]["mac"], "walk"), (rows[2]["mac"], "walk"), (rows[0]["mac"], "leave")):
            assert post("/api/config", {"ch": rows[0]["ch"], "mac": mac}).get("ok")
            assert post("/api/identify", {"mode": mode}).get("ok")
            s, seen = wait("판별"); st = get("/api/status"); r = st["ident_last"]
            assert r["mac"] == mac and r["mode"] == mode and len(r["series"]) == len(r["stim"]) and r["source"] == "csi", r
            texts = [q["text"] for q in seen]; assert any(q["move"] for q in seen) and any(not q["move"] for q in seen), "움직임/멈춤 안내가 모두 나와야 한다"
            assert set(texts) == set(st["plans"][mode]["say"]), (set(texts), st["plans"][mode]["say"])
            out[(mac, mode)] = r["verdict"]; print("판별 (%s, %s): %s | %s" % (mac, mode, r["verdict"], s["message"]))
        assert out[(rows[0]["mac"], "walk")] == "camera" and out[(rows[2]["mac"], "walk")] == "unclear" and out[(rows[0]["mac"], "leave")] == "camera", out
        assert get("/api/status")["ident"][rows[2]["mac"]]["verdict"] == "unclear"
        assert "모르는 방법" in post("/api/identify", {"mode": "dance"}).get("error", "") and get("/api/status")["status"]["state"] == "idle"

        # 3 위치 재기: 두 자리, 그리고 장치가 권한 자리에서 한 번 더
        assert post("/api/config", {"mac": rows[0]["mac"]}).get("ok")
        assert post("/api/room", {"lo": [0, 0, 0], "hi": [0.9, 1.5, 2.3]}).get("ok") and "여섯 번" in get("/api/status")["hint"]      # 좁은 칸이면 '여섯 번'을 권한다
        assert post("/api/room", {"lo": [0, 0, 0], "hi": [3.5, 4.5, 2.4]}).get("ok") and get("/api/status")["hint"] == ""
        old_numpy = tuple(int(v) for v in st["numpy"].split(".")[:2]) < (1, 17)
        if old_numpy:                                            # 파이의 numpy 1.16: 가짜 방을 만드는 코드만 못 돈다 (실제로 재는 계산은 test_core.py 가 본다)
            assert post("/api/measure", {"plan": "tilt3", "pose": {"x": 1.0, "y": 1.0}}).get("ok")
            for _ in range(600):
                s = get("/api/status")["status"]
                if s["state"] != "busy":
                    break
                time.sleep(0.05)
            assert s["state"] == "error" and "numpy 1.17" in s["message"], s
            print("재기: numpy %s 라 가짜 방을 만들지 못한다고 알려 줌 (정상)" % st["numpy"])
            assert post("/api/present", {"yaw": 120}).get("ok"); wait("마커 보여 주기")
        cam = (2.4, 3.1, 0.9)
        def zone(tag):
            z = get("/api/zone"); c = z["centre"]; err = sum((a - b) ** 2 for a, b in zip(c, cam)) ** 0.5
            inside = all(z["box"]["lo"][i] - 0.15 <= cam[i] <= z["box"]["hi"][i] + 0.15 for i in range(3))
            print("구간 (%s): 가운데 (%.2f, %.2f, %.2f), 참값과 %.2f m, 길이 %.2f m, 참값이 구간 안 %s | %s" % (tag, c[0], c[1], c[2], err, z["length_m"], inside, z["message"]))
            return z, err, inside
        if not old_numpy:
            for pose in ({"x": 1.0, "y": 1.0, "z": 0.0, "heading": 0.0}, {"x": 2.6, "y": 1.2, "z": 0.4, "heading": 90.0}):
                assert post("/api/measure", {"plan": "tilt3", "pose": pose}).get("ok")
                s, _ = wait("재기"); print("재기:", s["message"])
            z2, e2, in2 = zone("두 자리"); assert z2["positions"] == 2 and in2 and z2["next"], z2
            nx = z2["next"][0]
            assert post("/api/measure", {"plan": "tilt3", "pose": {"x": nx["x"], "y": nx["y"], "z": 0.0, "heading": 0.0}}).get("ok"); wait("재기")
            z3, e3, in3 = zone("세 자리"); assert z3["positions"] == 3 and in3 and z3["length_m"] < z2["length_m"] and e3 < 0.5, (e3, z3["length_m"])

        # 4 마커: 재고 난 뒤의 자세(끄덕임 54.7도)에서, 설계 쪽(cad/)이 계산해 둔 좌표와 같은가
        m = get("/api/marker"); g = json.load(open(os.path.join(PI, "marker.json"), encoding="utf-8"))
        assert abs(m["pitch"] - 54.7356) < 0.01 and m["dictionary"] == "DICT_4X4_50" and len(m["markers"]) == 2 and not m["moving"], m
        import math
        a = math.radians(m["yaw"])
        for mk in m["markers"]:
            ref = g["corners_base_mm_at_register"][str(mk["id"])]          # 받침 0도일 때의 값: 지금 받침 각만큼 돌려 견준다
            for cgot, cref in zip(mk["corners"], ref):
                x = (cref[0] * math.cos(a) - cref[1] * math.sin(a)) / 1000.0; y = (cref[0] * math.sin(a) + cref[1] * math.cos(a)) / 1000.0
                assert abs(cgot[0] - x) < 2e-5 and abs(cgot[1] - y) < 2e-5 and abs(cgot[2] - cref[2] / 1000.0) < 2e-5, (cgot, cref)
        side = [sum((p - q) ** 2 for p, q in zip(m["markers"][0]["corners"][i], m["markers"][0]["corners"][(i + 1) % 4])) ** 0.5 for i in range(4)]
        assert all(abs(v - 0.034) < 1e-4 for v in side), side
        assert abs(m["normal"][2] - math.sin(math.radians(54.7356))) < 1e-3                                         # 등판이 위를 55도 올려다본다
        assert post("/api/present", {"yaw": 30}).get("ok"); wait("마커 보여 주기"); m2 = get("/api/marker"); assert abs(m2["yaw"] - 30) < 0.03, m2["yaw"]      # 모터 한 칸(0.022도) 단위로 맞는다: 실물과 같게 적어 둔 위치에서 읽는다
        assert post("/api/park").get("ok"); wait("세우기"); m3 = get("/api/marker"); assert m3["pitch"] == 0 and abs(m3["normal"][2]) < 1e-9
        print("마커: 받침 %.0f도, 끄덕임 %.1f도에서 0번 마커 가운데 높이 %.3f m, 면이 향한 쪽 %s" % (m["yaw"], m["pitch"], sum(c[2] for c in m["markers"][0]["corners"]) / 4, m["normal"]))
        print("node 가짜 모드 검사 통과 (%s)" % PY)
    finally:
        p.terminate()
        try:
            out = p.communicate(timeout=5)[0].decode("utf-8", "replace")
        except Exception:
            p.kill(); out = ""
        if "Traceback" in out:
            print(out[-3000:])


if __name__ == "__main__":
    main()
