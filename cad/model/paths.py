"""파일을 어디에 쓰는지 한곳에 모았다.
   저장소 (이 파일 = cad/model/paths.py):  STL -> stl/, 그림 -> docs/images/, 계산 결과(json) -> cad/results/, 슬라이싱 결과 -> print/, 마커 -> marker/
   작업용 배치 (저장소 폴더 바로 옆에 out2 폴더가 있을 때):  out2/stl, out2/img, out2/*.json, out2/print, out2/marker"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
CAD = os.path.dirname(HERE)                         # cad/
ROOT = os.path.dirname(CAD)                         # 저장소 맨 위
ASSETS = os.path.join(CAD, "assets")                # 그림에 쓰는 화면 갈무리 등
WORK = os.path.join(os.path.dirname(ROOT), "out2")
if os.path.isdir(WORK):
    RES = WORK; STL = os.path.join(WORK, "stl"); IMG = os.path.join(WORK, "img"); SLICE = os.path.join(WORK, "print"); MARKER = os.path.join(WORK, "marker")
else:
    RES = os.path.join(CAD, "results"); STL = os.path.join(ROOT, "stl"); SLICE = os.path.join(ROOT, "print"); MARKER = os.path.join(ROOT, "marker")
    IMG = os.path.join(ROOT, "docs", "images")
TMP = os.environ.get("DETECAM_TMP") or os.path.join(CAD, "_tmp")          # 임시 파일 (렌더 중간 그림, 유한요소 그물)
for d in (RES, STL, IMG, TMP, MARKER):
    os.makedirs(d, exist_ok=True)
