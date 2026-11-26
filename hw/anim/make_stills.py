"""Still pictures of the assembled receiver (no captions) for the docs and slides.  Design values.
xvfb-run -a -s "-screen 0 1280x720x24" python3 hw/anim/make_stills.py  ->  hw/img/수신기_외형_*.png"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_video as V  # noqa: E402
import vtk  # noqa: E402

OUT = os.path.join(V.HERE, "..", "img")
V.REN = vtk.vtkRenderer()
V.REN.SetBackground(1.0, 1.0, 1.0)
V.REN.SetBackground2(0.90, 0.93, 0.97)
V.REN.GradientBackgroundOn()
V.RW = vtk.vtkRenderWindow()
V.RW.SetOffScreenRendering(1)
V.RW.AddRenderer(V.REN)
V.RW.SetSize(V.W, V.H)
V.RW.SetMultiSamples(8)
V.build()
st = V.State()
V.apply(st, 1.0, V.STEPS[0], None)
shots = {"정면": V.cam((0, 25, -100), 18, 16, 700), "비스듬히": V.cam((0, 25, -100), 40, 26, 720),
         "뒤": V.cam((0, 25, -100), 215, 22, 720)}
for name, view in shots.items():
    V.set_cam(view, None, 1.0, None)
    V.grab().save(os.path.join(OUT, f"수신기_외형_{name}.png"))
# on the wall bracket
st.visible |= {"wall", "W1_wall_bracket", "cup_0", "cup_1", "lanyard"}
V.apply(st, 1.0, dict(ops=[], dur=1.0), None)
V.set_cam(V.cam((-40, 25, -170), 35, 14, 980), None, 1.0, None)
V.grab().save(os.path.join(OUT, "수신기_외형_벽.png"))
print("ok")


def transparent(view, path, size=(1600, 1200)):
    """Same picture with a transparent background (for slides)."""
    from vtk.util.numpy_support import vtk_to_numpy
    from PIL import Image
    V.RW.SetSize(*size)
    V.RW.SetAlphaBitPlanes(1)
    V.REN.GradientBackgroundOff()
    V.REN.SetBackground(1.0, 1.0, 1.0)
    V.REN.SetBackgroundAlpha(0.0)
    for n in ("wall", "W1_wall_bracket", "cup_0", "cup_1", "lanyard"):
        V.SCENE[n].actor.VisibilityOff()
    V.set_cam(view, None, 1.0, None)
    V.RW.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(V.RW)
    w2i.SetInputBufferTypeToRGBA()
    w2i.ReadFrontBufferOff()
    w2i.Update()
    im = w2i.GetOutput()
    a = vtk_to_numpy(im.GetPointData().GetScalars()).reshape(size[1], size[0], 4)[::-1]
    img = Image.fromarray(np.ascontiguousarray(a), "RGBA")
    img = img.crop(img.getbbox())
    img.save(path)


transparent(V.cam((0, 25, -100), 40, 26, 640), os.path.join(OUT, "수신기_외형_투명.png"))
print("ok2")
