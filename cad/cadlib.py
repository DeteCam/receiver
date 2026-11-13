"""작은 CAD 도우미 (manifold3d 기반). 단위 mm."""
import numpy as np
import manifold3d as m3
from manifold3d import Manifold as M, CrossSection as CS, JoinType

m3.set_min_circular_angle(3.0)
m3.set_min_circular_edge_length(0.5)

EMPTY = M()

def box(x0, x1, y0, y1, z0, z1):
    return M.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])

def cz(r, z0, z1, x=0.0, y=0.0, seg=0):
    return M.cylinder(z1 - z0, r, r, seg).translate([x, y, z0])

def cy(r, y0, y1, x=0.0, z=0.0, seg=0):
    return M.cylinder(y1 - y0, r, r, seg).rotate([-90, 0, 0]).translate([x, y0, z])

def cx(r, x0, x1, y=0.0, z=0.0, seg=0):
    return M.cylinder(x1 - x0, r, r, seg).rotate([0, 90, 0]).translate([x0, y, z])

def union(parts):
    parts = [p for p in parts if p is not None]
    return M.batch_boolean(parts, m3.OpType.Add) if parts else M()

def rrect(x0, x1, y0, y1, r):
    """모서리가 둥근 사각형 단면"""
    r = min(r, (x1 - x0) / 2 - 1e-3, (y1 - y0) / 2 - 1e-3)
    cs = CS.square([x1 - x0 - 2 * r, y1 - y0 - 2 * r]).translate([x0 + r, y0 + r])
    return cs.offset(r, JoinType.Round)

def poly(pts):
    p = np.asarray(pts, dtype=np.float64)
    area2 = np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1])
    if area2 < 0:
        p = p[::-1]
    return CS([p])

def sector(r0, r1, a0, a1, n=24):
    """고리 조각 단면 (각도는 도, x축에서 반시계)"""
    a = np.radians(np.linspace(a0, a1, n))
    outer = np.c_[r1 * np.cos(a), r1 * np.sin(a)]
    inner = np.c_[r0 * np.cos(a[::-1]), r0 * np.sin(a[::-1])]
    return poly(np.vstack([outer, inner]))

def extrude_z(cs, z0, z1):
    return M.extrude(cs, z1 - z0).translate([0, 0, z0])

def prism_xz(cs, y0, y1):
    """단면 (u,w)->(X,Z), Y 방향으로 y0..y1 돌출"""
    return M.extrude(cs, y1 - y0).rotate([90, 0, 0]).translate([0, y1, 0])

def prism_xy_to_yz(cs, x0, x1):
    """단면 (u,w)->(Y,Z), X 방향으로 x0..x1 돌출"""
    T = np.array([[0, 0, 1, x0], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    return M.extrude(cs, x1 - x0).transform(T)

def revolve_profile(pts, seg=0):
    """(r,z) 단면을 Z축 둘레로 회전"""
    return M.revolve(poly(pts), seg)

def d_bore_cs(d=5.15, flat=3.25):
    """28BYJ-48 축용 양면 D 구멍 단면 (원 지름 d, 평면 사이 flat)"""
    return CS.circle(d / 2) ^ CS.square([d + 2, flat], True)

def gear_cs(mod, z, alpha_deg=20.0, shift=0.0, backlash=0.15, n=10):
    """인벌류트 평기어 단면. shift=전위계수, backlash=이 두께 감소량(mm)"""
    al = np.radians(alpha_deg)
    r = mod * z / 2.0
    rb = r * np.cos(al)
    ra = r + mod * (1.0 + shift)
    rf = r - mod * (1.25 - shift)
    s = mod * (np.pi / 2 + 2 * shift * np.tan(al)) - backlash   # 피치원 이 두께
    inv = lambda a: np.tan(a) - a
    def half(rho):
        phi = np.arccos(min(1.0, rb / rho))
        return s / (2 * r) + inv(al) - inv(phi)
    r_start = max(rb, rf)
    rs = np.linspace(r_start, ra, n)
    flank = [(rho, half(rho)) for rho in rs]
    if flank[-1][1] <= 0.01:
        raise ValueError("이끝이 너무 뾰족함")
    pts = []
    pitch = 2 * np.pi / z
    for i in range(z):
        c = i * pitch
        h0 = flank[0][1]
        # 이뿌리(왼쪽) -> 왼쪽 치면 -> 이끝 -> 오른쪽 치면 -> 이뿌리
        if rf < rb:
            pts.append((rf * np.cos(c - h0), rf * np.sin(c - h0)))
        for rho, h in flank:
            pts.append((rho * np.cos(c - h), rho * np.sin(c - h)))
        for rho, h in flank[::-1]:
            pts.append((rho * np.cos(c + h), rho * np.sin(c + h)))
        if rf < rb:
            pts.append((rf * np.cos(c + h0), rf * np.sin(c + h0)))
        # 다음 이까지 이뿌리 원호
        a0, a1 = c + h0, c + pitch - h0
        for a in np.linspace(a0, a1, 4)[1:-1]:
            pts.append((rf * np.cos(a), rf * np.sin(a)))
    return poly(pts), dict(r=r, rb=rb, ra=ra, rf=rf)

def teardrop_cs(r, toward_deg):
    """물방울 단면: 원 + 45도 꼭지. 옆으로 누운 구멍·못을 받침 없이 출력하려고 쓴다.
    toward_deg = 꼭지가 향하는 방향(도). 구멍이면 출력할 때 위쪽, 못이면 아래쪽을 향하게 준다."""
    a = np.radians(toward_deg)
    tip = np.array([np.cos(a), np.sin(a)]) * r * np.sqrt(2.0)
    p1 = np.array([np.cos(a + np.pi / 4), np.sin(a + np.pi / 4)]) * r
    p2 = np.array([np.cos(a - np.pi / 4), np.sin(a - np.pi / 4)]) * r
    return CS.circle(r) + poly([p1, tip, p2])

def rot_z(m, deg):
    return m.rotate([0, 0, deg])

def to_trimesh(m):
    import trimesh
    mesh = m.to_mesh()
    return trimesh.Trimesh(vertices=np.asarray(mesh.vert_properties)[:, :3],
                           faces=np.asarray(mesh.tri_verts), process=False)
