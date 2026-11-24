"""Map-aware likelihood with large furniture: the room planes of palm.MapModel plus rectangular reflecting faces (wardrobe side, bed top, ...).
A face is used for a candidate only if the mirror point really falls on the face and the camera is on the receiver's side of it."""
import numpy as np
from palm import MapModel, steer, C0

class MapModelX(MapModel):
    def __init__(self, grid, node_pos, pos, axis, B, dims, faces=(), refl=0.5, second=True, lam=0.08, tc=np.arange(-45e-9, 45e-9 + 1e-12, 3e-9)):
        L, W, H = dims; self.B = B; self.tc = tc; C = len(grid)
        planes = [(0, 0.0), (0, L), (1, 0.0), (1, W), (2, H), (2, 0.0)]
        rho1 = [refl[p] if isinstance(refl, dict) else refl for p in planes]
        imgs = [grid] + [self._mirror(grid, ax, val) for ax, val in planes]; rho = [1.0] + rho1
        if second:
            for i in range(6):
                for j in range(i + 1, 6):
                    if planes[i][0] == planes[j][0]: continue
                    imgs.append(self._mirror(self._mirror(grid, *planes[i]), *planes[j])); rho.append(rho1[i] * rho1[j])
        valid = [np.ones(C, bool)] * len(imgs)
        for ax, val, lo, hi, rf in faces:                                   # (axis, value, lower corner, upper corner, amplitude reflectivity)
            g = self._mirror(grid, ax, val); den = g[:, ax] - node_pos[ax]; t = (val - node_pos[ax]) / np.where(np.abs(den) < 1e-9, 1e-9, den)
            p = node_pos[None] + t[:, None] * (g - node_pos[None]); ok = (t > 0) & (t < 1) & (np.sign(grid[:, ax] - val) == np.sign(node_pos[ax] - val))
            for j in range(3):
                if j != ax: ok &= (p[:, j] >= lo[j] - 0.05) & (p[:, j] <= hi[j] + 0.05)
            imgs.append(g); rho.append(rf); valid.append(ok)
        R = len(imgs); M = len(pos)
        self.S = np.zeros((C, R, M), np.complex64); self.dt = np.zeros((C, R)); self.w = np.zeros((C, R))
        d0 = np.linalg.norm(grid - node_pos, axis=1)
        for i, g in enumerate(imgs):
            v = g - node_pos; d = np.linalg.norm(v, axis=1); U = v / d[:, None]
            self.S[:, i] = steer(U, pos, axis, B) * valid[i][:, None]; self.dt[:, i] = (d - d0) / C0; self.w[:, i] = (rho[i] * d0 / d) ** 2
        Fq = np.exp(-2j * np.pi * B.df[None, None, :] * self.dt[:, :, None])
        Gs = np.einsum("crm,csm->crs", self.S.conj(), self.S); Gf = np.einsum("crk,csk->crs", Fq.conj(), Fq)
        self.G = Gs * Gf; self.Fq = Fq; self.lam = lam
        self.Ftc = np.exp(2j * np.pi * B.df[:, None] * tc[None])

def box_faces(lo, hi, which, rho=0.5):
    """Faces of a furniture box; which = e.g. ["x-", "y-", "z+"] (outward side)."""
    lo, hi = np.array(lo, float), np.array(hi, float); out = []
    for wch in which:
        ax = "xyz".index(wch[0]); val = hi[ax] if wch[1] == "+" else lo[ax]; out.append((ax, float(val), lo, hi, rho))
    return out

# large furniture of the lodging room that a user can reasonably measure (the TV panel and the fridge stay unknown, as in the nominal map)
MOTEL_FACES = (box_faces((3.8, 2.0, 0.0), (4.5, 3.7, 2.1), ["x-", "y-", "z+"]) + box_faces((0.3, 1.8, 0.0), (2.3, 3.6, 0.55), ["z+"])
               + box_faces((1.2, 0.0, 0.0), (3.2, 0.45, 0.85), ["z+", "y+"]))
