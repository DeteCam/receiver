"""Rotating-arm synthetic aperture (one fixed reference antenna + one antenna on a motor-driven arm), approximated from
single-point path data with a per-path plane-wave model. Rough estimate only (no cable flex, no motion during the sweep)."""
import numpy as np
from rt_eval import KW, DF, GRID, axes, udir, front

def arm_local(radius=0.20, npos=36):
    ph = np.arange(npos) * 2 * np.pi / npos
    return np.stack([radius * np.cos(ph), np.zeros(npos), radius * np.sin(ph)], 1)          # in the wall plane (local x-z)

def csi_arm(d, r, n, node, P):
    ex, ey, ez = axes(node["bore"]); u = udir(d["th"][r, n], d["ph"][r, n]); ul = np.stack([u @ ex, u @ ey, u @ ez], -1)
    st = np.exp(1j * KW * (P @ ul.T)); ph_f = np.exp(-2j * np.pi * d["tau"][r, n][:, None] * DF)
    return (st * (d["a"][r, n, 0] * front(u, ey))[None]) @ ph_f

def arm_steer(node, P):
    """plane-wave steering from the node centre to every grid point -> (C, M), unit norm"""
    ex, ey, ez = axes(node["bore"]); v = GRID - node["pos"]; v /= np.linalg.norm(v, axis=1, keepdims=True); ul = np.stack([v @ ex, v @ ey, v @ ez], 1)
    return np.exp(1j * KW * (ul @ P.T)) / np.sqrt(len(P))
