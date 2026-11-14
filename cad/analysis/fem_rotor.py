"""회전 막대(판 받이 + 균형추 통)를 유한요소법으로 푼다: 손으로 한 강도 계산(strength.py)을 전문 도구로 다시 본다.
    python3 fem_rotor.py        ->  결과 폴더의 fem_rotor.json, fem/ 폴더에 그물과 풀이 파일

도구: fTetWild (그물 만들기. pip install pytetwild) + CalculiX ccx (선형 정적 해석. apt install calculix-ccx). 둘 다 공개 도구다.
모형: 회전 막대 한 덩어리를 2차 사면체(C3D10)로 나눈다. 재료는 PLA (탄성 계수 2500 MPa, 푸아송 비 0.35), 축 구멍 면을 붙잡는다.
힘 (알루미늄 판은 휘지 않는 지렛대로 보고, 판에 건 힘을 볼트 4개 자리의 힘으로 바꿔 판 받이에 건다)
  1) 판 아래끝을 머리 쪽으로 5N   2) 판 옆 모서리를 머리 쪽으로 5N   3) 판 위끝을 아래로 5N   4) 동전 12개의 무게 (누운 자세, 옆으로)
보는 것: 가장 큰 응력, 층 사이를 벌리는 응력(출력할 때 층이 쌓이는 방향 = Z), 판 끝이 움직이는 양.
"""
import os, sys, json, subprocess, re
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import numpy as np
import trimesh
import parts2 as P
import paths
from p2 import *

W = os.path.join(paths.TMP, "fem"); os.makedirs(W, exist_ok=True)          # 그물과 풀이 파일 (다시 만들 수 있어 저장소에는 넣지 않는다)
E_PLA, NU = 2500.0, 0.35
S_XY, S_Z = 40.0, 15.0
F = 5.0
SIZE = float(os.environ.get("FEM_SIZE", "1.5"))


def mesh():
    """회전 막대의 겉면(삼각형)에서 속이 찬 사면체 그물을 만든다. fTetWild(pytetwild): 겉면 삼각형이 길쭉하거나 가늘어도 되는 튼튼한 방법.
    (Gmsh 로 먼저 해 봤는데, 동전 통과 나사 받이가 만나는 가는 틈에서 겉면이 서로 뚫고 지나가 실패했다. 그물 품질 확인에는 Gmsh 를 쓴다.)
    만든 1차 사면체에 모서리 가운데 절점을 더해 2차 사면체(C3D10)로 바꾼다."""
    import pytetwild
    m = P.rotor_bar().to_mesh(); v = np.asarray(m.vert_properties)[:, :3].astype(float); f = np.asarray(m.tri_verts).astype(np.int32)
    trimesh.Trimesh(v, f, process=True).export(os.path.join(W, "rotor.stl"))
    diag = float(np.linalg.norm(v.max(0) - v.min(0)))
    here = os.getcwd(); os.chdir(W)                 # pytetwild 는 지금 폴더에 중간 파일(__tracked_surface.stl)을 남긴다: 임시 폴더에서 돌린다
    try:
        pts, tets = pytetwild.tetrahedralize(v, f, optimize=True, edge_length_fac=SIZE / diag)
    finally:
        os.chdir(here)
    pts = np.asarray(pts, float); tets = np.asarray(tets, int)
    a, b, c, d = pts[tets[:, 0]], pts[tets[:, 1]], pts[tets[:, 2]], pts[tets[:, 3]]
    vol = np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a) / 6.0
    neg = vol < 0; tets[neg] = tets[neg][:, [0, 2, 1, 3]]; vol = np.abs(vol)
    keep = vol > 1e-9; tets = tets[keep]
    used = np.unique(tets); remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used)); pts = pts[used]; tets = remap[tets]
    # 2차로: 모서리 (1-2, 2-3, 3-1, 1-4, 2-4, 3-4) 가운데에 절점을 하나씩
    edges = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]; mid = {}; extra = []; el = []
    for t in tets:
        row = list(t)
        for i, j in edges:
            k = (min(t[i], t[j]), max(t[i], t[j]))
            if k not in mid:
                mid[k] = len(pts) + len(extra); extra.append((pts[k[0]] + pts[k[1]]) / 2.0)
            row.append(mid[k])
        el.append(row)
    allp = np.vstack([pts, np.array(extra)])
    nodes = {i + 1: list(map(float, q)) for i, q in enumerate(allp)}
    elems = [[i + 1] + [int(k) + 1 for k in row] for i, row in enumerate(el)]
    return nodes, elems, float(vol[keep].sum())


def near_axis(X, p, d, r, lo, hi):
    """축(점 p, 방향 d) 둘레 반지름 r 안, 축 방향 좌표가 lo~hi 인 점들"""
    d = np.asarray(d, float); q = X - np.asarray(p, float); t = q @ d; rad = np.linalg.norm(q - np.outer(t, d), axis=1)
    return (rad <= r) & (t >= lo) & (t <= hi)


def loads(ids, X):
    """힘 네 가지 -> {이름: [(절점, 방향, 힘)]}. 판에 건 힘을 볼트 자리의 힘으로 바꾼다."""
    xf = FL_XH - FL_T; ys = sorted(set(y for y, z in PLATE_BOLTS)); zs = sorted(set(z for y, z in PLATE_BOLTS)); zc = (zs[0] + zs[1]) / 2
    dz = zs[1] - zs[0]; dy = ys[1] - ys[0]

    def bolt_nodes(y, z, pull):
        # 당기는 나사: 너트가 앉은 자리(안쪽 면에서 2.2 들어간 면) 둘레.  누르는 나사: 판 틀(뒤) 받침이 닿는 바깥면의 구멍 둘레
        if pull:
            m = near_axis(X, [0, y, z], [1, 0, 0], 3.3, xf + 2.0, xf + 2.4)
        else:
            m = near_axis(X, [0, y, z], [1, 0, 0], 5.0, FL_XH - 0.2, FL_XH + 0.2)
        return ids[m]

    def spread(out, y, z, fx=0.0, fy=0.0, fz=0.0):
        if abs(fx) > 1e-9:
            n = bolt_nodes(y, z, fx > 0)
            out += [(int(k), 1, fx / len(n)) for k in n]
        for comp, val in ((2, fy), (3, fz)):
            if abs(val) > 1e-9:
                n = ids[near_axis(X, [0, y, z], [1, 0, 0], 2.3, xf + 2.2, FL_XH)]            # 옆으로 미는 힘은 나사 구멍 벽이 받는다
                out += [(int(k), comp, val / len(n)) for k in n]

    L = {}
    # 1) 판 틀 아래끝(z = BL_Z0 - FR_END)을 -X 로 5N: 합력 -5, Y 축 둘레 모멘트 = 5 x (나사 줄 가운데 - 아래끝)
    out = []; Mm = F * (zc - (BL_Z0 - FR_END))
    for y in ys:
        spread(out, y, zs[1], fx=(+Mm / dz - F / 2) / 2); spread(out, y, zs[0], fx=(-Mm / dz - F / 2) / 2)
    L["판 아래끝을 머리 쪽으로 5N"] = out
    # 2) 판 옆 모서리(y = +50, 가운데 높이)를 -X 로 5N: Z 축 둘레 모멘트 = 5 x 50
    out = []; Mm = F * PLATE_W / 2
    for z in zs:
        spread(out, ys[1], z, fx=(-Mm / dy - F / 2) / 2); spread(out, ys[0], z, fx=(+Mm / dy - F / 2) / 2)
    L["판 옆 모서리를 머리 쪽으로 5N"] = out
    # 3) 판 위끝을 아래로 5N: 볼트 4개가 나눠 받는다 (판이 받이에 붙어 있어 모멘트 팔은 판 두께쯤이라 무시)
    out = []
    for y, z in PLATE_BOLTS:
        spread(out, y, z, fz=-F / 4)
    L["판 위끝을 아래로 5N"] = out
    # 4) 동전 12개 + 마개 (0.72N) 가 옆(+Y)으로: 누운 자세에서 통 벽이 받는다
    n = ids[near_axis(X, [POD_XC, 0, 0], [0, 0, 1], POD_RI + 0.3, POD_Z0, POD_Z1)]
    L["동전 12개의 무게 (누운 자세)"] = [(int(k), 2, (COIN_G * COIN_SLOTS + 8.0) * 9.81e-3 / len(n)) for k in n]
    return L


def write_ccx(nodes, elems, fixed, L):
    with open(os.path.join(W, "rotor_ccx.inp"), "w") as f:
        f.write("*NODE, NSET=NALL\n" + "".join("%d,%.6f,%.6f,%.6f\n" % (k, *v) for k, v in nodes.items()))
        f.write("*ELEMENT, TYPE=C3D10, ELSET=EALL\n" + "".join(",".join(str(t) for t in e) + "\n" for e in elems))
        f.write("*NSET, NSET=FIX\n" + "".join("%d,\n" % k for k in fixed))
        f.write("*MATERIAL, NAME=PLA\n*ELASTIC\n%g,%g\n*SOLID SECTION, ELSET=EALL, MATERIAL=PLA\n*BOUNDARY\nFIX,1,3\n" % (E_PLA, NU))
        for name, ld in L.items():
            f.write("*STEP\n*STATIC\n*CLOAD, OP=NEW\n" + "".join("%d,%d,%.8e\n" % t for t in ld) + "*NODE FILE\nU\n*EL FILE\nS\n*END STEP\n")


def read_frd(path, n_steps):
    """frd 파일에서 단계마다 절점 변위와 응력을 읽는다"""
    disp = []; stress = []; cur = None; kind = None
    for line in open(path):
        if line.startswith(" -4"):
            name = line[5:13].strip(); kind = "D" if name.startswith("DISP") else ("S" if name.startswith("STRESS") else None); cur = {} if kind else None
            continue
        if line.startswith(" -3"):
            if cur is not None:
                (disp if kind == "D" else stress).append(cur)
            cur = None; kind = None; continue
        if cur is not None and line.startswith(" -1"):
            nid = int(line[3:13]); vals = [float(line[13 + 12 * i:25 + 12 * i]) for i in range(3 if kind == "D" else 6)]
            cur[nid] = vals
    return disp[:n_steps], stress[:n_steps]


def vm(s):
    sx, sy, sz, sxy, syz, szx = s.T
    return np.sqrt(0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2) + 3 * (sxy ** 2 + syz ** 2 + szx ** 2))


if __name__ == "__main__":
    nodes, elems, vol = mesh()
    print("그물의 부피 %.2f cm3 (모델 %.2f cm3)" % (vol / 1000, P.rotor_bar().volume() / 1000))
    ids = np.array(list(nodes.keys())); X = np.array([nodes[k] for k in ids])
    fixed = ids[near_axis(X, [0, 0, 0], [0, 0, 1], D_BORE_D / 2 + 0.15, M1_ZT + 3.9, M1_ZT + 10.1)]                 # 축 구멍(납작한 부분)의 벽
    L = loads(ids, X)
    for name, ld in L.items():
        tot = np.zeros(3)
        for k, c, v in ld:
            tot[c - 1] += v
        print("힘: %-26s 절점 %4d개에 나눠 건다, 합 (%.2f, %.2f, %.2f) N" % (name, len(set(k for k, c, v in ld)), *tot))
    print("그물: 절점 %d개, 사면체 %d개 (크기 %.1f mm), 붙잡은 절점 %d개" % (len(nodes), len(elems), SIZE, len(fixed)), flush=True)
    write_ccx(nodes, elems, fixed, L)
    r = subprocess.run(["ccx", "rotor_ccx"], cwd=W, capture_output=True, text=True, timeout=3000)
    if not os.path.exists(os.path.join(W, "rotor_ccx.frd")):
        raise RuntimeError("풀지 못했습니다:\n" + r.stdout[-2000:])
    disp, stress = read_frd(os.path.join(W, "rotor_ccx.frd"), len(L))
    res = dict(tool="fTetWild (pytetwild 0.4.2) + CalculiX 2.21",
               mesh=dict(nodes=len(nodes), elements=len(elems), size_mm=SIZE, type="C3D10"), material=dict(E=E_PLA, nu=NU), cases={})
    loaded = set(k for ld in L.values() for k, c, v in ld) | set(int(k) for k in fixed)
    xf = FL_XH - FL_T
    for (name, ld), D, S in zip(L.items(), disp, stress):
        sid = np.array([k for k in S if k not in loaded]); s = np.array([S[k] for k in sid]); Xs = np.array([nodes[k] for k in sid])
        v = vm(s); i = int(np.argmax(v))
        # 층 사이를 벌리는 응력: 층이 Z 방향으로 쌓인다 -> SZZ (당기는 쪽만). 판 받이(x > 60)와 막대 전체에서 각각
        szz = s[:, 2]; j = int(np.argmax(szz))
        flange = Xs[:, 0] > 60.0; jf = int(np.argmax(np.where(flange, szz, -1e9)))
        u = np.array([D[k] for k in ids]); umax = float(np.max(np.linalg.norm(u, axis=1)))
        # 판 받이가 기운 각 -> 판 끝이 움직이는 양: 볼트 자리 절점들의 x 변위를 높이에 대해 맞춘다
        fb = (X[:, 0] > xf) & (np.abs(X[:, 1]) < FL_HW)
        A = np.column_stack([np.ones(fb.sum()), X[fb, 2], X[fb, 1]]); co = np.linalg.lstsq(A, u[fb, 0], rcond=None)[0]
        end = co[0] + co[1] * (BL_Z0 - FR_END)
        res["cases"][name] = dict(vm_max=round(float(v[i]), 2), vm_at=[round(float(t), 1) for t in Xs[i]], vm_p999=round(float(np.percentile(v, 99.9)), 2),
                                  szz_max=round(float(szz[j]), 2), szz_at=[round(float(t), 1) for t in Xs[j]], szz_flange=round(float(szz[jf]), 2), szz_flange_at=[round(float(t), 1) for t in Xs[jf]],
                                  sf_vm=round(S_XY / float(v[i]), 1), sf_layer=round(S_Z / max(float(szz[j]), 1e-6), 1), u_max_mm=round(umax, 3), plate_end_mm=round(float(end), 3))
        c = res["cases"][name]
        print("%-26s 가장 큰 응력 %.2f MPa (%s) -> 여유 %.1f배 | 층을 벌리는 응력 %.2f MPa (%s) -> 여유 %.1f배 | 가장 많이 움직인 곳 %.3f mm, 판 아래끝 %.2f mm" % (
            name, c["vm_max"], c["vm_at"], c["sf_vm"], c["szz_max"], c["szz_at"], c["sf_layer"], c["u_max_mm"], c["plate_end_mm"]), flush=True)
    res["min_sf"] = min(min(c["sf_vm"], c["sf_layer"]) for c in res["cases"].values())
    json.dump(res, open(os.path.join(paths.RES, "fem_rotor.json"), "w"), ensure_ascii=False, indent=1)
    print("가장 작은 여유 %.1f배" % res["min_sf"])
