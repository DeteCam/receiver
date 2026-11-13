"""'전선: 고치기 전 / 고친 뒤' 그림에서 두 쪽이 같이 쓰는 값 (카메라, 자세, 전선 색)과 작은 도우미."""
import numpy as np
from cadlib import M, union

CAM = dict(cam_az=174.0, cam_el=20.0, dist=640.0, focus=(-10, 0, 100), size=(1100, 1000))
POSE = (0.0, 54.7356, 90.0)            # 받침 0도, 끄덕임 54.7도(재는 자세), 판 90도
CABLE = "#F08A3C"                      # 전선 표시색 (주황)


def rz(p, deg):
    a = np.radians(deg); c, s = np.cos(a), np.sin(a)
    return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]])


def HP_factory(Y_T, S_T, H_T):
    def HP(p, yaw=0.0, pitch=0.0):
        """머리 좌표의 점 -> 세계 좌표"""
        p = np.asarray(p, float)
        x, y, z = p[0], p[1] - Y_T, p[2] - S_T
        x, y = -y, x
        pr = np.radians(pitch)
        x, z = x * np.cos(pr) + z * np.sin(pr), -x * np.sin(pr) + z * np.cos(pr)
        return rz(np.array([x, y, z + H_T]), yaw)
    return HP


def tube(pts, r=1.6):
    """점들을 잇는 굵은 선 (전선 표시용)"""
    out = []
    pts = [np.asarray(p, float) for p in pts]
    for a, b in zip(pts[:-1], pts[1:]):
        d = b - a; L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        z = d / L
        x = np.cross([0, 0, 1.0], z)
        x = np.array([1.0, 0, 0]) if np.linalg.norm(x) < 1e-6 else x / np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.column_stack([x, y, z, a])
        out.append(M.cylinder(L, r, r, 16).transform(T))
        out.append(M.sphere(r, 16).translate(list(b)))
    return union(out)
