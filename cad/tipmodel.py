"""넘어짐 계산에 쓰는 무게 모형: 장치를 네 덩어리(받침, 요크, 머리, 회전부)로 나누고, 덩어리마다 '무엇이 몇 g, 어디에'를 적는다.
tipover.py(가만히 있을 때의 계산)와 tip_mujoco.py(물리 엔진 MuJoCo 로 움직여 보는 계산)가 같이 쓴다.

출력 부품의 무게는 슬라이서(Bambu Studio)가 계산한 값을 쓰고, 없으면 부피 x 밀도 x 채움 어림을 쓴다. 사는 부품은 실제 무게(어림)를 쓴다.
좌표: 세계 (바닥 z = 0, 받침 회전축 = Z). 머리 좌표 -> 세계는 parts2.pose_head 와 같은 계산.
"""
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np
import parts2 as P
import mass as MS
import paths
from p2 import *

RHO = MS.RHO
BATT_G = {"S": 200.0, "L": 400.0}            # 보조배터리 무게 어림: 10000mAh 얇은 것 약 200g, 20000mAh 약 400g
SUPPORT_IN = 1.5                             # 받침 바닥 모서리가 깎여 있어 바닥에 닿는 원은 받침 반지름보다 이만큼 작다


def slice_g():
    """슬라이서가 계산한 부품 무게 (g). 파일이 없으면 빈 표"""
    f = os.path.join(paths.SLICE, "요약.json")
    try:
        return {k: v["g"] for k, v in json.load(open(f, encoding="utf-8"))["parts"].items()}
    except Exception:
        return {}


def part(name, m, key, fill, SG):
    v, c = MS.props(m)
    return (name, float(SG.get(key, v * RHO * fill)), np.array(c, float))


def base_items(size="S", battery=True, batt_g=None):
    SG = slice_g(); it = []
    it.append(part("받침 통", P.tub(size), "01_base_tub_%s" % size, 0.45, SG))
    it.append(part("받침 덮개", P.cover(size), "02_base_cover_%s" % size, 0.6, SG))
    it.append(("고리·마개·나사", 6.0, np.array([0.0, 0.0, 34.0])))
    if battery:
        v, c = MS.props(P.battery_dummy(size)); it.append(("보조배터리", float(BATT_G[size] if batt_g is None else batt_g), np.array(c, float)))
        it.append(("USB 플러그와 선", 15.0, np.array([0.0, -BASE[size]["R"] + 12.0, 12.0])))
    return it


def yoke_items():
    SG = slice_g(); it = [part("요크", P.yoke(), "04_yoke", 0.6, SG)]
    for nm, T in (("모터 1", P.motor1_T()), ("모터 2", P.motor2_T())):
        it.append((nm, 34.0, (T @ np.array([-M_OFFSET, 0.0, -M_BODY_H / 2, 1.0]))[:3]))
    it.append(("작은 기어·링·핀·막이·나사", 14.0, np.array([-20.0, 30.0, 75.0])))
    it.append(("모터선·전원선 (상자 안)", 18.0, np.array([-40.0, 0.0, 65.0])))
    return it


def head_items(plate_t=AL_T, length=150.0, n_coin=None):
    """(회전부가 아닌 머리 것들, 회전부 것들). 머리 좌표."""
    M = MS.report(fill=dict(plate_t=plate_t), quiet=True)
    n = P.COINS[(float(length), float(plate_t))] if n_coin is None else n_coin
    rot = [(nm, float(g), np.array(c, float)) for nm, g, c in MS.rotor_items(length, plate_t, n)]
    rn = set(nm for nm, g, c in MS.rotor_items(150.0, plate_t, P.COINS[(150.0, float(plate_t))]))
    fix = [(d["name"], d["g"], np.array(d["c"], float)) for d in M["head_items3"] if d["name"] not in rn]
    return fix, rot


def head_to_world(c, yaw, pitch, spin=None):
    c = np.asarray(c, float)
    if spin is not None:
        a = np.radians(spin); c = np.array([c[0] * np.cos(a) - c[1] * np.sin(a), c[0] * np.sin(a) + c[1] * np.cos(a), c[2]])
    x, y, z = -(c[1] - Y_T), c[0], c[2] - S_T                      # 옮기고 Z 둘레로 90도
    p = np.radians(pitch); x, z = x * np.cos(p) + z * np.sin(p), -x * np.sin(p) + z * np.cos(p)
    z += H_T
    a = np.radians(yaw)
    return np.array([x * np.cos(a) - y * np.sin(a), x * np.sin(a) + y * np.cos(a), z])


def yoke_to_world(c, yaw):
    a = np.radians(yaw); c = np.asarray(c, float)
    return np.array([c[0] * np.cos(a) - c[1] * np.sin(a), c[0] * np.sin(a) + c[1] * np.cos(a), c[2]])


class Device:
    def __init__(self, size="S", battery=True, plate_t=AL_T, length=150.0, batt_g=None, n_coin=None):
        self.size = size; self.R = BASE[size]["R"] - SUPPORT_IN
        self.base = base_items(size, battery, batt_g); self.yoke = yoke_items(); self.head, self.rotor = head_items(plate_t, length, n_coin)
        self.length = length; self.plate_t = plate_t

    def mass(self):
        return dict((k, sum(g for _, g, c in v)) for k, v in (("base", self.base), ("yoke", self.yoke), ("head", self.head), ("rotor", self.rotor)))

    def points(self, yaw, pitch, spin):
        """모든 것의 (무게 g, 세계 좌표) 목록"""
        out = [(g, c) for _, g, c in self.base]
        out += [(g, yoke_to_world(c, yaw)) for _, g, c in self.yoke]
        out += [(g, head_to_world(c, yaw, pitch)) for _, g, c in self.head]
        out += [(g, head_to_world(c, yaw, pitch, spin)) for _, g, c in self.rotor]
        return out

    def com(self, yaw, pitch, spin):
        pts = self.points(yaw, pitch, spin); m = sum(g for g, c in pts)
        return m, sum(g * c for g, c in pts) / m

    def top(self, yaw, pitch, spin):
        """가장 높은 점 (알루미늄 판의 네 귀 가운데 가장 높은 것)과 그 높이"""
        best = None
        for y in (-PLATE_W / 2, PLATE_W / 2):
            for z in (BL_Z0, BL_Z0 + self.length):
                p = head_to_world([FL_X1 + self.plate_t, y, z], yaw, pitch, spin)
                if best is None or p[2] > best[2]:
                    best = p
        return best

    def stability(self, yaw, pitch, spin):
        m, c = self.com(yaw, pitch, spin); r = float(np.hypot(c[0], c[1])); margin = self.R - r
        top = self.top(yaw, pitch, spin)
        return dict(mass_g=m, com=c, off=r, margin=margin, tip_deg=float(np.degrees(np.arctan2(margin, c[2]))), push_top_N=float(m * 9.81e-3 * margin / top[2]), top_h=float(top[2]))


if __name__ == "__main__":
    for size in ("S", "L"):
        d = Device(size); print(size, {k: round(v, 1) for k, v in d.mass().items()}, "합계 %.0f g" % sum(d.mass().values()))
        for nm, items in (("받침", d.base), ("요크", d.yoke)):
            for n_, g, c in items:
                print("   %-4s %-22s %6.1f g  (%6.1f, %6.1f, %6.1f)" % (nm, n_, g, c[0], c[1], c[2]))
        for p in (0.0, TILT3, 85.0):
            s = d.stability(0.0, p, 0.0); print("   끄덕임 %4.1f: 무게중심 (%.1f, %.1f, %.1f), 받침 끝까지 %.1f mm, 넘어가는 기울기 %.1f도, 꼭대기(%.0f mm)를 미는 힘 %.2f N" % (
                p, s["com"][0], s["com"][1], s["com"][2], s["margin"], s["tip_deg"], s["top_h"], s["push_top_N"]))
