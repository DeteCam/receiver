# -*- coding: utf-8 -*-
"""다른 폴더의 코드를 찾아 sys.path 에 넣는다. 이 폴더의 스크립트는 맨 먼저 `import empaths` 를 한다.
   저장소 배치:  sim/em (여기) · sim (방 모형, 실험) · firmware/core (계산 코드) · cad (치수 p2.py, params.py)
   작업용 배치:  rx2/em (여기) · rx2/sim (방 모형 + 계산 코드) · rx2, 그 위 (치수)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))


def _find(cands, must):
    for c in cands:
        p = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(p, must)):
            return p
    sys.exit("%s 를 찾지 못했습니다. 찾아본 곳: %s" % (must, ", ".join(cands)))


SIM = _find(["..", "../sim"], "montecarlo.py")                                   # 방 모형과 실험
CORE = _find(["../../firmware/core", "../sim"], "estimator.py")                  # 계산 코드 (수신기와 같은 것)
CAD = _find(["../../cad", ".."], "p2.py")                                        # 치수
BASE = _find(["../../cad", "../.."], "params.py")                                # 파이·모터 치수
for _p in (BASE, CAD, CORE, SIM, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
