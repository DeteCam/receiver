"""패널 수신기의 빠른 확인 그림.  xvfb-run -a python3 view_panel.py 이름 az el [숨길 것,...] [panel|hub|foot]"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export")]
import panel as P, render, paths
name, az, el = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); hide = sys.argv[4].split(',') if len(sys.argv) > 4 and sys.argv[4] else []
what = sys.argv[5] if len(sys.argv) > 5 else "panel"
A = P.hub_assembly() if what == "hub" else P.panel_assembly("foot" if what == "foot" else "plate")
if what != "hub":                                            # 패널을 세운다: 패널의 y(위) -> 화면의 z, 패널의 z(앞) -> 화면의 -y
    A = {k: (m.rotate([90, 0, 0]), g, c, pr) for k, (m, g, c, pr) in A.items()}
render.render(A, os.path.join(paths.TMP, name + '.png'), cam_az=az, cam_el=el, size=(1000, 1000), dist=760 if what != "hub" else 520, focus=(0, 0, -10) if what != "hub" else (0, 0, 28), ground=False, hide=hide)
print(os.path.join(paths.TMP, name + '.png'))
