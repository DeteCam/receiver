#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문서에 넣을 폰 화면 그림을 찍는다 (장치 없이).   python3 screens.py <그림을 둘 폴더>
수신기 프로그램을 가짜 모드(node.py --dry --fast)로 띄우고, 폰 크기(390 x 844)의 브라우저로 열어 단계마다 한 장씩 찍는다.
playwright 가 있어야 한다 (pip install playwright, playwright install chromium). 수신기에는 필요 없다: 문서를 만들 때만 쓴다."""
import os, subprocess, sys, time
from urllib.request import urlopen
from playwright.sync_api import sync_playwright

HERE = os.path.dirname(os.path.abspath(__file__)); PI = os.path.join(HERE, "..", "pi")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "screens"); os.makedirs(OUT, exist_ok=True)
PORT = 8791; URL = "http://127.0.0.1:%d" % PORT


def wait_idle(page, t=60.0):
    """하던 일이 끝날 때까지 (상태가 idle 이 될 때까지) 기다린다"""
    end = time.time() + t
    while time.time() < end:
        if page.evaluate("S && S.status && S.status.state") == "idle":
            return True
        page.wait_for_timeout(150)
    raise RuntimeError("끝나지 않았다")


def shot(page, name, top=None, dy=10):
    """보이는 화면을 그대로 찍는다. top = 맨 위에 오게 할 것 (CSS 선택자), dy = 그 위로 남길 높이"""
    if top:
        page.evaluate("([s, dy]) => { var e = document.querySelector(s); window.scrollTo(0, e.getBoundingClientRect().top + window.scrollY - dy); }", [top, dy])
        page.wait_for_timeout(250)
    page.screenshot(path=os.path.join(OUT, name)); print("찍음", name, flush=True)


srv = subprocess.Popen([sys.executable, os.path.join(PI, "node.py"), "--dry", "--fast", "--port", str(PORT)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
try:
    for _ in range(100):
        try:
            urlopen(URL + "/api/status", timeout=1).read(); break
        except Exception:
            time.sleep(0.2)
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        page.on("dialog", lambda d: d.accept())
        page.goto(URL); page.wait_for_function("S !== null")
        # 1 탐지
        page.click("text=둘레 기기 찾기 (약 30초)"); page.wait_for_timeout(300); wait_idle(page); page.wait_for_selector("#survey tr >> nth=1")
        shot(page, "web_1_detect.png")
        # 2 판별: 후보 줄을 눌러 고르고, 하는 중과 결과
        page.click("#survey tr:has-text('카메라 후보') >> nth=0"); page.wait_for_timeout(300)
        page.click("#iw"); page.wait_for_selector("#prompt", state="visible", timeout=15000)
        page.wait_for_function("document.getElementById('ptext').textContent.length > 0", timeout=15000)
        shot(page, "web_2_prompt.png", "#prompt")
        wait_idle(page); page.wait_for_timeout(500)
        shot(page, "web_3_result.png", "#iw")
        # 3 위치: 세 자리 재기
        for x, y, h in ((1.0, 1.0, 0), (2.6, 1.2, 90), (1.2, 3.4, -60)):
            for k, v in (("#px", x), ("#py", y), ("#pz", 0), ("#ph", h)):
                page.fill(k, str(v))
            page.click("#go"); page.wait_for_timeout(300); wait_idle(page, 120)
        page.wait_for_timeout(800)
        shot(page, "web_4_measure.png", "#px", 150)
        shot(page, "web_5_zone.png", "#zmsg", 70)
        # 장치: 손으로 움직이기 (펼친 채로), 움직이는 중의 멈춤
        page.click("summary"); page.wait_for_timeout(300)
        shot(page, "web_6_control.png", ".jog", 120)
        b.close()
finally:
    srv.terminate()
    try:
        srv.wait(timeout=5)
    except Exception:
        srv.kill()

# 하는 중의 '멈춤' 단추: 모터 명령이 실제 시간이 걸리게 해 놓고(DETECAM_DRY_SLEEP) 재는 도중에 찍고, 멈춤을 눌러 본다
srv = subprocess.Popen([sys.executable, os.path.join(PI, "node.py"), "--dry", "--port", str(PORT + 1)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=dict(os.environ, DETECAM_DRY_SLEEP="1"))
try:
    for _ in range(100):
        try:
            urlopen("http://127.0.0.1:%d/api/status" % (PORT + 1), timeout=1).read(); break
        except Exception:
            time.sleep(0.2)
    with sync_playwright() as p:
        b = p.chromium.launch(); page = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        page.goto("http://127.0.0.1:%d" % (PORT + 1)); page.wait_for_function("S !== null")
        for k, v in (("#px", 1.0), ("#py", 1.0), ("#pz", 0), ("#ph", 0)):
            page.fill(k, str(v))
        page.click("#go"); page.wait_for_function("S && S.status && S.status.state === 'busy'", timeout=10000); page.wait_for_timeout(1500)
        page.evaluate("window.scrollTo(0, 0)"); page.wait_for_timeout(200)
        page.screenshot(path=os.path.join(OUT, "web_7_moving.png"), clip={"x": 0, "y": 0, "width": 390, "height": 330}); print("찍음 web_7_moving.png", flush=True)
        page.click("#stopbar button"); page.wait_for_function("S && S.status && S.status.state !== 'busy'", timeout=20000); page.wait_for_timeout(1500)
        page.screenshot(path=os.path.join(OUT, "web_8_stopped.png"), clip={"x": 0, "y": 0, "width": 390, "height": 330}); print("찍음 web_8_stopped.png:", page.evaluate("S.status.message"), flush=True)
        b.close()
finally:
    srv.terminate()
    try:
        srv.wait(timeout=5)
    except Exception:
        srv.kill()
print("그림 폴더:", OUT)
