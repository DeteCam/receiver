"""수신기 전체(재는 자세, 판 0도)를 STL 한 장으로 내보낸다: 깃허브에서 이 파일을 누르면 3D 로 돌려 볼 수 있다.
    python3 export_assembly.py        ->  그림 폴더의 수신기_3D.stl   (출력용이 아니다. 출력할 부품은 stl/ 폴더)
보기 좋게 겉에서 보이는 부품만 넣는다 (머리 속 기판·전선은 뺀다). 크기를 줄이려고 둥근 면을 조금 성기게 한다."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import manifold3d as m3
m3.set_min_circular_angle(9.0)
m3.set_min_circular_edge_length(1.5)
import parts2 as P
import paths
from p2 import *

SHOW = ("tub", "cover", "clip", "lan_cap", "yoke", "pinion1", "pinion2", "m2_ring", "pin_r", "pin_l", "head_shell", "head_lid", "crown", "sector", "rotor_bar", "coin_cap",
        "plate_frame", "plate_clamp", "marker_frame", "keeper_r", "keeper_l", "plate", "motor3", "motor3_cap", "motor2", "motor2_cap", "motor1", "motor1_cap", "dongle", "dongle2", "ant_clip", "xiao_cap", "xiao", "fpc")


def write_stl(path, tris):
    """이진 STL"""
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec["n"] = n; rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"DeteCam receiver (measuring pose)".ljust(80, b" "))
        f.write(np.uint32(len(tris)).tobytes()); f.write(rec.tobytes())


if __name__ == "__main__":
    A = P.assembly(0.0, TILT3, 0.0)
    tris = []
    for k in SHOW:
        if k in A:
            me = A[k][0].to_mesh(); v = np.asarray(me.vert_properties)[:, :3]; t = np.asarray(me.tri_verts)
            tris.append(v[t])
    tris = np.concatenate(tris).astype(np.float32)
    out = os.path.join(paths.IMG, "수신기_3D.stl")
    write_stl(out, tris)
    print("%s: 삼각형 %d개, %.1f MB" % (out, len(tris), os.path.getsize(out) / 1e6))
