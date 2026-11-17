"""패널의 AR 마커 자리를 marker/marker.json 에 적는다. 마커 띠(marker/AR_마커.pdf, 34mm 마커 둘)는 돌판 때의 것을 그대로 쓴다.

    python3 export/marker_panel.py
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import panel as P
import paths

J = dict(dictionary="DICT_4X4_50", ids=[0, 1], size_mm=P.MARK_SIZE, strip_mm=list(P.MARK),
         where="패널 앞덮개의 아래쪽 표시선 안. 띠의 가운데가 앞덮개 가운데 줄에 오게 붙인다",
         panel_frame="원점 = 안테나 배열 보드 앞면의 한가운데, x = 앞에서 볼 때 오른쪽, y = 위, z = 앞 (방 쪽)",
         corners_panel_mm={str(k): [[round(float(u), 2) for u in p] for p in v] for k, v in P.marker_corners().items()},
         corner_order="왼쪽 위, 오른쪽 위, 오른쪽 아래, 왼쪽 아래 (마커를 바로 볼 때. OpenCV ArUco 와 같다)",
         antenna=dict(rows=4, cols=4, pitch_mm=P.PITCH, note="안테나 자리는 원점을 가운데로 한 4 x 4 격자 (z = 0). 간격은 보드를 받은 뒤 재서 고친다"),
         use="폰이 마커 둘의 귀 여덟 개를 보고 패널의 자리와 방향을 구한다. 방향은 패널끼리 서로를 재서 한 번 더 맞춘다")
json.dump(J, open(os.path.join(paths.MARKER, "marker.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("marker.json:", J["corners_panel_mm"]["0"][0], J["corners_panel_mm"]["1"][2])
