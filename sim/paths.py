# -*- coding: utf-8 -*-
"""계산 코드(estimator, model, fusion, locate, identify, analyze3)가 있는 곳을 찾아 sys.path 에 넣는다.
이 폴더(sim)는 실험만 담고, 계산 코드는 firmware/core 한 곳에만 둔다 (수신기와 시뮬레이션이 같은 코드를 쓴다)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
for _c in (".", "../firmware/core", "../core"):
    _p = os.path.normpath(os.path.join(HERE, _c))
    if os.path.exists(os.path.join(_p, "locate.py")):
        CORE = _p
        if _p not in sys.path:
            sys.path.insert(0, _p)
        break
else:
    sys.exit("계산 코드(locate.py 등)를 찾지 못했습니다. 이 폴더 옆에 firmware/core 가 있어야 합니다 (저장소를 통째로 받았는지 봅니다).")
