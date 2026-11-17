#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect.py 의 5GHz 채널 고르기 검사: iw phy 출력 읽기, 바퀴마다 훑을 채널 (ai 저장소의 docs/의심 점수.md 8절).   python3 tests/test_band.py"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "pi"))
import detect as D

PHY = """Wiphy phy1
	Band 1:
		Frequencies:
			* 2412 MHz [1] (20.0 dBm)
			* 2437 MHz [6] (20.0 dBm)
			* 2472 MHz [13] (20.0 dBm)
			* 2484 MHz [14] (disabled)
	Band 2:
		Frequencies:
			* 5180 MHz [36] (20.0 dBm)
			* 5200 MHz [40] (20.0 dBm)
			* 5260 MHz [52] (20.0 dBm) (no IR, radar detection)
			* 5745 MHz [149] (20.0 dBm)
			* 5865 MHz [173] (disabled)
"""


def main():
    c = D.parse_phy_channels(PHY)
    assert c == (1, 6, 13, 36, 40, 52, 149), c                       # 꺼진 14·173 은 빼고, 레이더 채널(no IR)은 듣기만 하므로 넣는다
    assert D.parse_phy_channels("Band 1:\n * 2412 MHz [1] (20.0 dBm)\n") == (1,)
    assert D.plan((1, 6, 11)) == (1, 6, 11)                            # 2.4GHz 동글: 그대로
    assert D.plan(c, None, 0) == c                                      # 처음 바퀴: 모두
    assert D.plan(c, {149}, 1) == (1, 6, 13, 149)                       # 그 뒤: AP 가 들린 5GHz 채널만
    assert D.plan(c, set(), 2) == (1, 6, 13)
    assert D.plan(c, {149}, D.FULL5) == c                               # FULL5 바퀴마다 다시 모두
    rows = [dict(ch=6), dict(ch=149), dict(ch=36), dict(ch=None)]
    assert D.ap_channels5(rows) == {36, 149}
    print("5GHz 채널 고르기 검사 통과: 처음 %d채널, 그 뒤 %d채널" % (len(D.plan(c, None, 0)), len(D.plan(c, {149}, 1))))


if __name__ == "__main__":
    main()
