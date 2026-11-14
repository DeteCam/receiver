"""파일을 어디에 쓰는지 한곳에 모았다.
   저장소 (cad 폴더 안에서 실행):          STL -> ../stl, 그림 -> ../docs/그림, 계산 결과(json) -> ./결과, 슬라이싱 결과 -> ../슬라이스, 마커 -> ../마커
   작업용 배치 (옆에 out2 폴더가 있을 때):  out2/stl, out2/img, out2/*.json, out2/슬라이스, out2/marker"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
UP = os.path.dirname(HERE)
if os.path.isdir(os.path.join(UP, "out2")):
    RES = os.path.join(UP, "out2"); STL = os.path.join(RES, "stl"); IMG = os.path.join(RES, "img"); SLICE = os.path.join(RES, "슬라이스"); MARKER = os.path.join(RES, "marker")
else:
    RES = os.path.join(HERE, "결과"); STL = os.path.join(UP, "stl"); SLICE = os.path.join(UP, "슬라이스"); MARKER = os.path.join(UP, "마커")
    IMG = os.path.join(UP, "docs", "그림") if os.path.isdir(os.path.join(UP, "docs")) else os.path.join(UP, "그림")
TMP = os.environ.get("DETECAM_TMP") or os.path.join(HERE, "_tmp")          # 임시 파일 (렌더 중간 그림)
for d in (RES, STL, IMG, TMP, MARKER):
    os.makedirs(d, exist_ok=True)
