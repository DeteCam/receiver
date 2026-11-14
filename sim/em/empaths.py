# -*- coding: utf-8 -*-
"""전파 계산 폴더의 길잡이. 이 폴더의 스크립트는 맨 먼저 `import empaths` 를 한다.
   sim/em/model    수신기의 전파 모형 (device, necgeo), 전파 표 (run_tables, tables)
   sim/em/studies  표로 재는 과정을 흉내 내는 실험들과 결과 json
   sim/em/tools    실물 곡선과 견주는 도구
   sim/em/figures  그림
   sim/em/tables   저장소에 넣어 둔 줄인 표,  sim/em/cache  NEC2 로 푼 원래 표 (저장소에는 없다)
   다른 곳: firmware/core (계산 코드), cad/model (치수 p2.py, params.py), sim/direction 등 (방 모형)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))          # sim/em
SIM = os.path.dirname(HERE)
ROOT = os.path.dirname(SIM)
MODEL, STUDIES, TOOLS, FIGS, FDTD = (os.path.join(HERE, d) for d in ("model", "studies", "tools", "figures", "fdtd"))
TABLES = os.path.join(HERE, "tables"); CACHE = os.path.join(HERE, "cache")


def _need(p, must):
    if not os.path.exists(os.path.join(p, must)):
        sys.exit("%s 를 찾지 못했습니다 (%s). 저장소를 통째로 받았는지 봅니다." % (must, p))
    return p


CORE = _need(os.path.join(ROOT, "firmware", "core"), "estimator.py")              # 계산 코드 (수신기와 같은 것)
CAD = _need(os.path.join(ROOT, "cad", "model"), "p2.py")                           # 치수 (p2.py, params.py)
WORK = os.path.join(os.path.dirname(ROOT), "out2")
IMG = os.path.join(WORK, "img") if os.path.isdir(WORK) else os.path.join(ROOT, "docs", "images")
for _p in (CAD, CORE, os.path.join(SIM, "identify"), os.path.join(SIM, "location"), os.path.join(SIM, "direction"), SIM, FIGS, TOOLS, STUDIES, MODEL, HERE):
    if _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)
