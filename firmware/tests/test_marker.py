# -*- coding: utf-8 -*-
"""마커 좌표 파일이 설계와 같은지 본다.   python3 tests/test_marker.py
수신기가 폰에 알려 주는 마커 귀의 좌표는 pi/marker.json 에서 나온다. 이 파일은 CAD(cad/marker.py)가 만드는 마커/marker.json 의 복사본이다.
등판이나 마커 자리를 바꾸고 한쪽만 고치면 AR 표시가 틀어지므로, 두 파일이 같은지와 수신기 코드가 같은 값을 내는지 확인한다."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); PI = os.path.normpath(os.path.join(HERE, "..", "pi"))
sys.path.insert(0, PI)

mine = os.path.join(PI, "marker.json")
J = json.load(open(mine, encoding="utf-8"))
assert J["ids"] == [0, 1] and len(J["corners_head_mm"]) == 2 and all(len(v) == 4 for v in J["corners_head_mm"].values()), "marker.json 의 모양이 다르다"

for c in ("../../마커/marker.json", "../../out2/marker/marker.json"):          # 저장소 배치, 작업용 배치
    cad = os.path.normpath(os.path.join(HERE, c))
    if os.path.exists(cad):
        assert open(cad, "rb").read() == open(mine, "rb").read(), "pi/marker.json 이 %s 와 다르다. cad/marker.py 를 돌린 뒤 복사한다" % cad
        print("설계의 마커 파일과 같다:", cad)
        break
else:
    print("설계의 마커 파일을 찾지 못해 견주지 못했다 (수신기에만 복사해 둔 경우에는 정상)")

# 수신기 코드가 '재는 자세'에서 내는 좌표가 파일에 적어 둔 값과 같은지 (같은 식을 쓰는지)
import math
K = J["kinematics"]; reg = J["register_pose"]
def head_to_base(p, yaw, pitch):
    x, y, z = p; t = math.radians(pitch); a = math.radians(yaw)
    xy = -(y - K["Y_T"]) * math.cos(t) + (z - K["S_T"]) * math.sin(t); zy = K["H_T"] + (y - K["Y_T"]) * math.sin(t) + (z - K["S_T"]) * math.cos(t)
    return [xy * math.cos(a) - x * math.sin(a), xy * math.sin(a) + x * math.cos(a), zy]
worst = 0.0
for k, pts in J["corners_head_mm"].items():
    for p, q in zip(pts, J["corners_base_mm_at_register"][k]):
        worst = max(worst, max(abs(u - v) for u, v in zip(head_to_base(p, 0.0, reg["pitch"]), q)))
assert worst < 0.02, "적어 둔 좌표와 식이 %.3f mm 다르다" % worst
print("마커 귀 8개: 식으로 계산한 값과 적어 둔 값의 차이 %.3f mm" % worst)
print("마커 검사 통과")
