"""3축 화면에 넣을 데이터를 만든다: 3D 메시(좌표 0.01mm 정수, 법선 8비트), 회절 표, 시뮬레이션 요약.
    xvfb-run -a python3 viewer_data3.py        ->  이 폴더의 data.json   (pip install manifold3d pyvista numpy)
모델(../../cad 의 p2.py, parts2.py)이나 시뮬레이션 결과(../../sim 의 result.json)가 바뀌었을 때만 다시 돌리면 된다. 그 뒤 build3.py 로 화면을 다시 만든다.
메시는 무리별 자기 좌표로 둔다 (화면에서 관절 각도대로 돌린다).
  B 받침(세계)  Y 요크  P1 작은 기어 1(모터 1 축 중심)  P2 작은 기어 2(모터 2 축 중심)  H 머리  R 회전부(머리 좌표, 판 0도)"""
import os, sys, json, base64
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data.json")          # 아래의 "from ... import *" 가 HERE 를 바꿔 놓기 전에 정해 둔다


def _find(cands, name):
    """name 파일이 들어 있는 폴더. 저장소(console/src)와 작업용 배치 둘 다에서 찾는다."""
    for c in cands:
        d = os.path.normpath(os.path.join(HERE, c))
        if os.path.exists(os.path.join(d, name)):
            return d
    sys.exit("%s 를 찾지 못했습니다. 찾아본 곳: %s" % (name, ", ".join(cands)))


# 저장소(console/src)에서는 옆 폴더들을 찾는다: cad, firmware/core, sim
CAD = _find(["../../cad/model"], "parts2.py")
CORE = _find(["../../firmware/core"], "model.py")                                  # 회절 표를 만드는 계산
SIM = _find(["../../sim/direction"], "result.json")                                          # 미리 돌려 둔 시뮬레이션 결과
for d in (CORE, CAD):
    sys.path.insert(0, d)
import numpy as np
import manifold3d as m3
m3.set_min_circular_angle(7.5)
m3.set_min_circular_edge_length(1.2)
import pyvista as pv
import parts2 as P
import parts as v1
from cadlib import *
from p2 import *
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def pack(m, decimate=0.0):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    pd = pv.PolyData(v, np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel())
    if decimate > 0:
        pd = pd.clean().decimate_pro(decimate, feature_angle=30.0, preserve_topology=True, splitting=False, boundary_vertex_deletion=False)
    pd = pd.compute_normals(split_vertices=True, feature_angle=38.0, auto_orient_normals=False, cell_normals=False, point_normals=True)
    Pt = np.asarray(pd.points); N = np.asarray(pd.point_data['Normals']); F = np.asarray(pd.faces).reshape(-1, 4)[:, 1:]
    assert len(Pt) < 65536, len(Pt)
    assert np.abs(Pt).max() < 327.0, np.abs(Pt).max()
    return dict(p=b64(np.round(Pt * 100).astype(np.int16)), n=b64(np.clip(np.round(N * 127), -127, 127).astype(np.int8)), i=b64(F.astype(np.uint16)), nv=int(len(Pt)), nt=int(len(F)))


def build_meshes():
    C = P.COL
    mb, mc = v1.motor_body_dummy(); pb, pc = v1.pi_dummy(); db, dc = v1.driver_dummy()
    sx, sy = P.m1_shaft(); px, pz = P.pin2_xz()
    items = [
        ("tub", P.tub("S"), "B", C["body"], "받침 통", 0), ("cover", P.cover("S"), "B", C["accent"], "받침 덮개", 0.35), ("clip", P.clip(), "B", C["accent"], "고리", 0),
        ("battery", P.battery_dummy("S"), "B", C["batt"], "배터리", 0),
        ("lan_cap", P.lan_cap(), "B", C["accent"], "랜선 홈 마개", 0), ("m2_ring", P.m2_ring(), "Y", C["accent"], "맞춤 링", 0),
        ("dongle", P.dongle_dummy(), "H", "#C9772E", "무선 동글 (폰과 통신)", 0), ("dongle2", P.dongle_dummy(P.DONGLE2), "H", "#1F9E8F", "탐지 동글 (둘레 기기 엿듣기)", 0), ("button", P.button_dummy()[0] + P.button_dummy()[1], "H", C["dark"], "전원 버튼", 0),
        ("yoke", P.yoke(), "Y", C["body"], "요크", 0), ("pin_r", P.pin(1), "Y", C["accent"], "핀", 0), ("pin_l", P.pin(-1), "Y", C["accent"], "핀", 0),
        ("motor1", mb.transform(P.motor1_T()), "Y", C["metal"], "모터 1", 0), ("motor1_cap", mc.transform(P.motor1_T()), "Y", C["cap"], "모터 1", 0),
        ("motor2", mb.transform(P.motor2_T()), "Y", C["metal"], "모터 2", 0), ("motor2_cap", mc.transform(P.motor2_T()), "Y", C["cap"], "모터 2", 0),
        ("pinion1", P.pinion(8.0), "P1", C["accent"], "작은 기어 1", 0), ("pinion2", P.pinion(0.0).rotate([90, 0, 0]), "P2", C["accent"], "작은 기어 2", 0),
        ("head_shell", P.head_shell(), "H", C["body"], "머리 몸통", 0), ("head_lid", P.head_lid(), "H", C["body"], "머리 뚜껑", 0), ("crown", P.crown(), "H", C["accent"], "모터 자리", 0),
        ("sector", P.sector_head(), "H", C["accent"], "부채꼴 기어", 0.3),
        ("motor3", v1._m1(mb), "H", C["metal"], "모터 3", 0), ("motor3_cap", v1._m1(mc), "H", C["cap"], "모터 3", 0),
        ("pi_board", pb, "H", C["pcb"], "라즈베리파이", 0), ("pi_parts", pc, "H", C["metal"], "라즈베리파이", 0), ("usb_plug", P.usb_plug_dummy(), "H", C["dark"], "전원선", 0),
        ("rotor_bar", P.rotor_bar(), "R", C["accent"], "회전 막대", 0), ("plate", P.al_plate(150.0), "R", C["foil"], "알루미늄 판 (100 × 150 × %gT)" % P.AL_T, 0),
        ("plate_frame", P.plate_frame(), "R", C["accent"], "판 틀 (판이 들어가는 홈)", 0), ("plate_clamp", P.plate_clamp(), "R", C["accent"], "판 누름쇠", 0),
        ("rotor_hw", P.rotor_hw(150.0, P.COINS[(150.0, P.AL_T)]), "R", C["screw"], "판 틀·누름쇠 나사와 너트, 축 고정 나사", 0), ("coins", P.coins(P.COINS[(150.0, P.AL_T)]), "R", C["metal"], "균형추 (100원 동전 %d개)" % P.COINS[(150.0, P.AL_T)], 0),
        ("coin_cap", P.coin_cap(P.COINS[(150.0, P.AL_T)]), "R", C["accent"], "균형추 마개", 0),
        ("marker_frame", P.marker_frame(), "H", C["accent"], "AR 마커 틀", 0), ("keeper_r", P.pin_keeper(1), "Y", C["accent"], "핀 막이", 0), ("keeper_l", P.pin_keeper(-1), "Y", C["accent"], "핀 막이", 0),
    ]
    for k in range(3):
        items.append(("driver%d" % (3 - k), db.transform(P.driver_T(k)), "H", "#3B8F5E", "드라이버 %d" % (3 - k), 0))
        items.append(("driver%d_parts" % (3 - k), dc.transform(P.driver_T(k)), "H", C["dark"], "드라이버 %d" % (3 - k), 0))
    out = []; tv = tt = 0
    for name, m, g, col, ko, dec in items:
        d = pack(m, dec); d.update(name=name, g=g, c=col, ko=ko); out.append(d); tv += d["nv"]; tt += d["nt"]
        print("  %-14s %s 꼭짓점 %6d 삼각형 %6d" % (name, g, d["nv"], d["nt"]))
    size = sum(len(d["p"]) + len(d["n"]) + len(d["i"]) for d in out)
    print("메시 %d개, 꼭짓점 %d, 삼각형 %d, base64 %.2f MB" % (len(out), tv, tt, size / 1e6))
    geo = dict(H_T=H_T, S_T=S_T, Y_T=Y_T, p1=[sx, sy, DECK_Z0 - 6.0], p2=[px, ARM_Y0 + 1.8, pz], baseR=BASE["S"]["R"], rFoil=R_FOIL, blade=[BL_Z0, BL_Z0 + 150.0, PLATE_W])
    return out, geo


def build_tables():
    from model import table
    tabs = {}; SC = 16384.0
    for dist in (1500.0, 3000.0, 5000.0):
        betas, dphis, F = table(dist=dist)
        assert np.abs(F.real).max() < 1.99 and np.abs(F.imag).max() < 1.99, (np.abs(F.real).max(), np.abs(F.imag).max())
        tabs[str(int(dist))] = dict(re=b64(np.round(F.real * SC).astype(np.int16)), im=b64(np.round(F.imag * SC).astype(np.int16)))
    return dict(beta0=float(betas[0]), db=float(betas[1] - betas[0]), nb=int(len(betas)), phi0=float(dphis[0]), dp=float(dphis[1] - dphis[0]), np=int(len(dphis)), scale=SC, tabs=tabs)


def build_summary():
    S = json.load(open(os.path.join(SIM, "result.json"), encoding="utf-8"))["summary"]
    cons = json.load(open(os.path.join(SIM, "consistency.json"), encoding="utf-8"))
    rows = []
    for name, label in (("세워서 1번", "세워서 1번 (논문 방식)"), ("기울여 3번 (+필요하면 1번 더)", "기울여 3번 (+필요하면 1번 더)"), ("세워서 1번 + 기울여 3번", "세워서 1번 + 기울여 3번"), ("기울여 4번", "60° 기울여 4번"), ("기울여 6번", "기울여 6번 (좁은 방용)")):
        r = dict(name=label)
        for key, tag in (("W·보통", "w"), ("V·보통", "v"), ("W·나쁨", "wb"), ("V·나쁨", "vb")):
            v = S[key][name]["전체"]; r[tag] = dict(med=v["med"], over30=v["over30"], scans=v["scans"])
        rows.append(r)
    return dict(n=1500, rows=rows, cons=dict(v=cons["V·보통"], w=cons["W·보통"]))


if __name__ == "__main__":
    meshes, geo = build_meshes()
    data = dict(meshes=meshes, geo=geo, table=build_tables(), mc=build_summary())
    s = json.dumps(data, ensure_ascii=False, separators=(',', ':'))
    open(OUT, "w", encoding="utf-8").write(s)
    print("data.json %.2f MB" % (len(s.encode()) / 1e6))
