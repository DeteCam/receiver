"""빠른 확인용 그림.  xvfb-run -a python3 view.py 이름 yaw pitch spin az el [hide,...]"""
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE] + [os.path.join(os.path.dirname(HERE), d) for d in ("model", "export", "checks", "analysis", "figures")]      # cad 의 하위 폴더를 모두 찾는다
import parts2 as P, render
name, yaw, pitch, spin, az, el = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
hide = sys.argv[7].split(',') if len(sys.argv) > 7 else []
ghost = sys.argv[8].split(',') if len(sys.argv) > 8 else []
A = P.assembly(yaw, pitch, spin)
import paths
OUT = paths.TMP + '/'
render.render(A, OUT + name + '.png', cam_az=az, cam_el=el, size=(1100, 1100), dist=820, focus=(10, 0, 140), ground=False, hide=hide, ghost=ghost)
print(OUT + name + '.png')
