#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""core/score.py 검사: docs/의심 점수.md 6절의 상황을 가짜 탐지 표로 만들어, 점수·등급·확인 요청이 문서대로 나오는지 본다.
    python3 tests/test_score.py"""
import os, random, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "core"))
import score as S

rnd = random.Random(3)
HOTEL_AP, CAM, PHONE, LAPTOP = "a4:2b:b0:00:00:a6", "c0:49:ef:11:22:33", "da:a1:19:77:88:99", "3c:22:fb:44:55:66"
HALL_AP, HOTSPOT, SPYCAM = "00:1d:aa:10:20:30", "6a:11:22:33:44:55", "c0:49:ef:66:77:88"


def steady(rate, n=10):
    return [max(0, int(rnd.gauss(rate * 0.5, 2))) or 1 for _ in range(n)]


def periodic(every_s, n=10, size=12):
    k = int(every_s / 0.5); return [size if i % k == 0 else 0 for i in range(n)]


def row(mac, role="sta", bssid="", ssid="", per_s=0.0, mean_len=0, rssi=-50.0, series=None, beacon=False, ch=6):
    cand = per_s >= 30 and mean_len >= 300
    return dict(mac=mac, role=role, bssid=bssid or mac, ssid=ssid, ch=ch, n=0 if beacon else int(per_s * 5), per_s=per_s, mean_len=mean_len, rssi=rssi,
                candidate=cand, beacon_only=beacon, series=series or [])


def test_pattern():
    assert S.pattern([steady(60)])[0] == "steady"
    p, per = S.pattern([periodic(3.0, 20), periodic(3.0, 20)])
    assert p == "periodic" and abs(per - 3.0) < 0.01, (p, per)
    assert S.pattern([[0, 0, 0, 9, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0]])[0] == "burst"
    assert S.pattern([[0] * 10])[0] == "quiet"
    assert S.pattern([])[0] == "none"


def hidden_cam():
    """몰카 (숙소·화장실 등): 점검 전부터 켜져 있다 (새로 나타남 없음). 영상 통화 중인 폰도 비슷하게 보인다 -> 둘 다 '의심', 확인 요청"""
    T = S.Tracker()
    for k in range(3):
        T.update([row(HOTEL_AP, "ap", HOTEL_AP, "room-507", 140, 1200, -52, beacon=False),
                  row(CAM, bssid=HOTEL_AP, per_s=63, mean_len=730, rssi=-48, series=steady(63)),
                  row(PHONE, bssid=HOTEL_AP, per_s=80, mean_len=900, rssi=-45, series=steady(80)),
                  row(LAPTOP, bssid=HOTEL_AP, per_s=8, mean_len=400, rssi=-58, series=[[0, 0, 9, 0, 0, 0, 0, 0, 6, 0], [7, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 5, 8, 0, 0, 0, 0]][k])], 100.0 + 30 * k)
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[CAM]["level"] == "suspect" and d[PHONE]["level"] == "suspect", (d[CAM]["score"], d[PHONE]["score"])
    assert r["need_confirm"] and set(r["confirm"]) == {CAM, PHONE}
    T.walked(CAM, "camera"); T.walked(PHONE, "suspect")
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[CAM]["level"] == "camera" and d[PHONE]["level"] == "suspect" and not r["need_confirm"]
    assert d[HOTEL_AP]["level"] == "camera" and CAM in d[HOTEL_AP]["clients"]     # 허브(여기서는 그 방의 공유기)도 같이 보인다
    T.mine(PHONE, True); d = dict((x["mac"], x) for x in T.result()["devices"])
    assert d[PHONE]["level"] == "mine"
    return d[CAM]


def hidden_cam_alone():
    """몰카, 다른 업로더 없음: 내 폰은 '내 기기'. 많이 올려 보내는 기기가 카메라 하나뿐 -> 걷지 않고 바로 '카메라'"""
    T = S.Tracker(); T.mine(PHONE, True)
    for k in range(3):
        T.update([row(HOTEL_AP, "ap", HOTEL_AP, "room-507", 140, 1200, -52),
                  row(CAM, bssid=HOTEL_AP, per_s=63, mean_len=730, rssi=-48, series=steady(63)),
                  row(PHONE, bssid=HOTEL_AP, per_s=80, mean_len=900, rssi=-45, series=steady(80))], 100.0 + 30 * k)
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[CAM]["level"] == "camera" and "only" in d[CAM]["keys"] and not r["need_confirm"], (d[CAM]["score"], d[CAM]["reasons"])
    return d[CAM]


def presses(n_press, cells=20):
    """버튼을 누를 때마다 사진 한 장이 몰려 나간다 (칸 두세 개), 나머지는 연결 유지 신호 정도"""
    w = [0] * cells
    for i in rnd.sample(range(0, cells - 3, 4), n_press):
        w[i], w[i + 1] = 30, 18
    return w


def exam():
    """시험장: 시작 때는 건물 공유기 비컨뿐. 시험 중에 핫스팟과 몇 초마다 사진을 올리는 카메라가 나타난다 -> 확인 요청 없이 '카메라'"""
    T = S.Tracker()
    T.update([row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1)], 0.0)
    for k in range(10):                                                # 사람이 없는 동안: 아무 일도 없음
        T.update([row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1)], 30.0 * (k + 1))
    r = T.result()
    assert not r["top"] and not r["need_confirm"], r["top"]               # 잘못 울림 없음
    for k in range(4):
        T.update([row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1),
                  row(HOTSPOT, "ap", HOTSPOT, "", rssi=-55, beacon=True, ch=6),
                  row(SPYCAM, bssid=HOTSPOT, per_s=5.0, mean_len=900, rssi=-56, series=periodic(3.0, 16), ch=6)], 400.0 + 30 * k)
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[SPYCAM]["level"] == "camera", (d[SPYCAM]["score"], d[SPYCAM]["reasons"])
    assert d[HOTSPOT]["level"] == "camera" and SPYCAM in d[HOTSPOT]["clients"]
    assert d[HALL_AP]["level"] == "ignore" and not r["need_confirm"]
    assert any(e["mac"] == SPYCAM for e in r["events"])
    return d[SPYCAM], d[HOTSPOT]


def exam_on_from_start():
    """시험장, 카메라와 핫스팟을 켠 채로 들어와 기준선에 이미 들어감 (새로 나타남 없음). 응시자가 마음대로 누를 때 사진이 나간다
    -> 누르기 전: 핫스팟만 '의심' (기기가 붙어 있음). 몇 번 누른 뒤: 카메라가 '카메라 가능성 높음'"""
    T = S.Tracker()
    base = lambda: [row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1), row(HOTSPOT, "ap", HOTSPOT, "", rssi=-55, beacon=True, ch=6)]
    T.update(base() + [row(SPYCAM, bssid=HOTSPOT, per_s=0.1, mean_len=60, rssi=-56, series=[0] * 20, ch=6)], 0.0)
    for k in range(4):                                                  # 시험 시작, 아직 안 누름: 연결 유지 신호뿐
        T.update(base() + [row(SPYCAM, bssid=HOTSPOT, per_s=0.1, mean_len=60, rssi=-56, series=[0] * 19 + [1], ch=6)], 30.0 * (k + 1))
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[SPYCAM]["level"] == "ignore", (d[SPYCAM]["score"], d[SPYCAM]["reasons"])
    assert d[HOTSPOT]["level"] == "suspect" and "hubcl" in d[HOTSPOT]["keys"], (d[HOTSPOT]["score"], d[HOTSPOT]["reasons"])
    assert d[HALL_AP]["level"] == "ignore" and not r["need_confirm"]
    quiet = d[HOTSPOT]
    for k in range(4):                                                  # 시험지를 받고 가끔 누름
        T.update(base() + [row(SPYCAM, bssid=HOTSPOT, per_s=4.0, mean_len=900, rssi=-56, series=presses(1 if k % 2 else 2), ch=6)], 200.0 + 30 * k)
    r = T.result(); d = dict((x["mac"], x) for x in r["devices"])
    assert d[SPYCAM]["level"] == "camera" and "bursts" in d[SPYCAM]["keys"], (d[SPYCAM]["score"], d[SPYCAM]["reasons"])
    assert d[HOTSPOT]["level"] == "camera" and not r["need_confirm"]
    return quiet, d[SPYCAM]


def exam_button():
    """버튼을 누를 때만 찍는 카메라: 평소에는 핫스팟 비컨만 새로 보인다 -> 핫스팟이 '의심 기기'"""
    T = S.Tracker()
    T.update([row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1)], 0.0)
    T.update([row(HALL_AP, "ap", HALL_AP, "KOREATECH", rssi=-62, beacon=True, ch=1), row(HOTSPOT, "ap", HOTSPOT, "iPhone", rssi=-55, beacon=True)], 30.0)
    d = dict((x["mac"], x) for x in T.result()["devices"])
    assert d[HOTSPOT]["level"] == "suspect", (d[HOTSPOT]["score"], d[HOTSPOT]["reasons"])
    return d[HOTSPOT]


if __name__ == "__main__":
    test_pattern(); c = hidden_cam(); a = hidden_cam_alone(); e, h = exam(); q, o = exam_on_from_start(); b = exam_button()
    show = lambda d: "%d %s · %s" % (d["score"], S.LEVEL_NAME[d["level"]], " + ".join(d["reasons"]))
    print("몰카 (영상 통화 폰과 같이):", show(c))
    print("몰카 (혼자 올림):", show(a))
    print("시험장 카메라:", e["score"], S.LEVEL_NAME[e["level"]], "·", " + ".join(e["reasons"]))
    print("  그 핫스팟:", h["score"], S.LEVEL_NAME[h["level"]], "·", " + ".join(h["reasons"]))
    print("처음부터 켠 핫스팟 (누르기 전):", show(q))
    print("처음부터 켠 카메라 (몇 번 누른 뒤):", show(o))
    print("버튼 카메라의 핫스팟:", b["score"], S.LEVEL_NAME[b["level"]], "·", " + ".join(b["reasons"]))
    print("점수 검사 통과")
