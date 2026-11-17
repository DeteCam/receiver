# -*- coding: utf-8 -*-
"""시뮬레이션 폴더의 길잡이. `identify/` 의 스크립트는 맨 먼저 `import paths` 를 한다.
  - 계산 코드(identify, score, panel)는 firmware/core 한 곳에만 둔다 (수신기와 시뮬레이션이 같은 코드를 쓴다).
  - 주제별 폴더(identify)의 모듈을 서로 부를 수 있게 sys.path 에 넣는다.
  - IMG: 그림을 쓰는 곳 (저장소에서는 docs/images, 작업용 배치에서는 out2/img).
돌판 수신기 때의 폴더(direction, location, em)는 receiver-legacy 저장소에 있다."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))          # sim/
ROOT = os.path.dirname(HERE)
CORE = None
for _c in ("../firmware/core", "../core"):
    _p = os.path.normpath(os.path.join(HERE, _c))
    if os.path.exists(os.path.join(_p, "identify.py")):
        CORE = _p
        break
if CORE is None:
    sys.exit("계산 코드(identify.py 등)를 찾지 못했습니다. sim 폴더 옆에 firmware/core 가 있어야 합니다 (저장소를 통째로 받았는지 봅니다).")
for _p in (CORE, os.path.join(HERE, "identify")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
WORK = os.path.join(os.path.dirname(ROOT), "out2")
IMG = os.path.join(WORK, "img") if os.path.isdir(WORK) else os.path.join(ROOT, "docs", "images")
os.makedirs(IMG, exist_ok=True)
