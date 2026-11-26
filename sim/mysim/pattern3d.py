"""NEC2 active-element patterns for heads whose elements are not all in one horizontal plane (stacked rings,
nodded heads).  Same interpolation as channel_sim.NEC2ArrayPattern (geometric phase stripped before bilinear
interpolation), but the stripped phase uses the full 3-D element position."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
from channel_sim import NEC2ArrayPattern, C0  # noqa: E402


class NEC3DArrayPattern(NEC2ArrayPattern):
    def __init__(self, pat_dict, fc_hz):
        self.fc = float(fc_hz)
        self.k0 = 2.0 * np.pi * self.fc / C0
        self.n_elem = int(len(pat_dict["eth"]))
        self.xyz = np.asarray(pat_dict["elem_xyz_m"], dtype=np.float64)
        self.xy = self.xyz[:, :2]
        self.ref_dz = float(pat_dict["ref_dz_m"])
        th = np.radians(np.arange(0, 181, 5))
        ph = np.radians(np.arange(0, 360, 5))
        ux = np.sin(th)[None, :] * np.cos(ph)[:, None]
        uy = np.sin(th)[None, :] * np.sin(ph)[:, None]
        uz = np.cos(th)[None, :] * np.ones((72, 1))
        self.eth_env = np.zeros_like(pat_dict["eth"], dtype=np.complex64)
        self.eph_env = np.zeros_like(pat_dict["eph"], dtype=np.complex64)
        for m in range(self.n_elem):
            nom = np.exp(1j * self.k0 * (self.xyz[m, 0] * ux + self.xyz[m, 1] * uy + self.xyz[m, 2] * uz))
            self.eth_env[m] = (pat_dict["eth"][m] / nom).astype(np.complex64)
            self.eph_env[m] = (pat_dict["eph"][m] / nom).astype(np.complex64)
        nom_ref = np.exp(1j * self.k0 * (self.ref_dz * uz))
        self.ref_eth_env = (pat_dict["ref_eth"] / nom_ref).astype(np.complex64)
        self.ref_eph_env = (pat_dict["ref_eph"] / nom_ref).astype(np.complex64)

    def element_responses(self, th, ph, aV=None, aH=None):
        th = np.asarray(th, dtype=np.float64)
        ph = np.asarray(ph, dtype=np.float64)
        ux = np.sin(th) * np.cos(ph)
        uy = np.sin(th) * np.sin(ph)
        uz = np.cos(th)
        geo = np.exp(1j * self.k0 * (self.xyz[:, 0:1] * ux.ravel()[None, :] + self.xyz[:, 1:2] * uy.ravel()[None, :]
                                     + self.xyz[:, 2:3] * uz.ravel()[None, :]))
        geo = geo.reshape((self.n_elem,) + th.shape)
        geo_ref = np.exp(1j * self.k0 * (self.ref_dz * uz))
        et = self._bil(self.eth_env, th, ph) * geo
        ep = self._bil(self.eph_env, th, ph) * geo
        rt = self._bil(self.ref_eth_env, th, ph) * geo_ref
        rp = self._bil(self.ref_eph_env, th, ph) * geo_ref
        if aV is None:
            return et, rt
        return et * aV[None, ...] + ep * aH[None, ...], rt * aV + rp * aH


def make_pattern(pat_dict, fc_hz):
    """3-D class only when some element sits off the z = 0 plane (keeps old results bit-identical)."""
    xyz = pat_dict.get("elem_xyz_m")
    if xyz is not None and np.max(np.abs(np.asarray(xyz)[:, 2])) > 1e-6:
        return NEC3DArrayPattern(pat_dict, fc_hz)
    return NEC2ArrayPattern(pat_dict, fc_hz)
