"""pyvista 렌더링 도우미"""
import numpy as np, pyvista as pv
pv.OFF_SCREEN = True

def to_pv(m):
    mesh = m.to_mesh()
    v = np.asarray(mesh.vert_properties)[:, :3].astype(np.float32)
    f = np.asarray(mesh.tri_verts).astype(np.int64)
    faces = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    pd = pv.PolyData(v, faces)
    return pd.compute_normals(split_vertices=True, feature_angle=40.0, auto_orient_normals=False)

MAT = {"metal": dict(specular=0.6, specular_power=40, ambient=0.25, diffuse=0.7),
       "plate": dict(specular=0.9, specular_power=60, ambient=0.3, diffuse=0.65)}

def render(A, fname, cam_az=35.0, cam_el=18.0, dist=640.0, focus=(0, 0, -30), size=(1400, 1400),
           hide=(), ghost=(), bg="#F4F2EE", fov=28.0, ground=True, zoom=1.0, ghosts=None, ssao=True,
           label_pts=None, ghost_opacity=0.22, ground_z=-150.4):
    """ghosts: 반투명으로 겹쳐 그릴 조립 목록 [(A2, 이름들)], label_pts: {이름: (x,y,z)} -> 화면 좌표 반환"""
    p = pv.Plotter(off_screen=True, window_size=size, lighting="none")
    p.set_background(bg)
    def add(name, m, col, opacity=None):
        kw = dict(color=col, smooth_shading=True, specular=0.25, specular_power=18, ambient=0.28, diffuse=0.82)
        if name in ("foil", "plate", "coins", "rotor_hw"):
            kw.update(MAT["plate"])
        if name.startswith("motor") and not name.endswith("cap"):
            kw.update(MAT["metal"])
        if opacity is not None:
            kw.update(opacity=opacity)
        p.add_mesh(to_pv(m), **kw)
    for name, (m, grp, col, printed) in A.items():
        if name in hide:
            continue
        add(name, m, col, ghost_opacity if name in ghost else None)
    for A2, names in (ghosts or []):
        for name in names:
            if name in A2:
                add(name, A2[name][0], A2[name][2], ghost_opacity)
    if ground:
        g = pv.Disc(center=(0, 0, ground_z), inner=0, outer=900, normal=(0, 0, 1), c_res=96)
        p.add_mesh(g, color="#E7E3DC", ambient=0.55, diffuse=0.55, specular=0.0)
    az, el = np.radians(cam_az), np.radians(cam_el)
    def L(a, e, inten, color="white"):
        a, e = np.radians(a), np.radians(e)
        pos = 1000 * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
        lt = pv.Light(position=tuple(pos), focal_point=focus, color=color, intensity=inten, light_type="scene light")
        lt.positional = False
        p.add_light(lt)
    L(cam_az - 35, 48, 0.95)
    L(cam_az + 60, 20, 0.45, "#FFF4E6")
    L(cam_az + 170, 35, 0.35, "#E6F0FF")
    L(cam_az, 85, 0.25)
    cam = np.array(focus) + dist * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    p.camera.position = tuple(cam)
    p.camera.focal_point = focus
    p.camera.up = (0, 0, 1)
    p.camera.view_angle = fov
    p.camera.zoom(zoom)
    if ghosts or ghost:
        try:
            p.enable_depth_peeling(8)
        except Exception:
            pass
    elif ssao:
        try:
            p.enable_ssao(radius=14, bias=0.5, kernel_size=128, blur=True)
        except Exception:
            pass
    try:
        p.enable_anti_aliasing("ssaa")
    except Exception:
        pass
    p.screenshot(fname)
    out = {}
    if label_pts:
        ren = p.renderer
        for k, w in label_pts.items():
            ren.SetWorldPoint(float(w[0]), float(w[1]), float(w[2]), 1.0)
            ren.WorldToDisplay()
            d = ren.GetDisplayPoint()
            out[k] = (d[0], size[1] - d[1])
    p.close()
    return out
