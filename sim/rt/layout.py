"""Pick a 12-element sparse planar layout (discrete antennas on a 3-D printed frame) by random search on peak sidelobe."""
import numpy as np
C0 = 299792458.0
def psl(xy, f, n=61):
    k = 2 * np.pi * f / C0
    u = np.linspace(-1.6, 1.6, n); U, V = np.meshgrid(u, u)            # difference of direction cosines (scan +-53 deg)
    m = (U ** 2 + V ** 2) <= 1.6 ** 2
    af = np.abs(np.exp(1j * k * (np.outer(U[m], xy[:, 0]) + np.outer(V[m], xy[:, 1]))).sum(1)) / len(xy)
    r = np.hypot(U[m], V[m]); bw = 0.9 * (C0 / f) / (np.ptp(xy, axis=0).max() + 1e-9)
    return af[r > 1.3 * bw].max()
def search(n_el=12, half=0.12, dmin=0.040, trials=4000, seed=3):
    rng = np.random.default_rng(seed); best = None
    for t in range(trials):
        pts = []
        while len(pts) < n_el:
            p = rng.uniform(-half, half, 2)
            if all(np.hypot(*(p - q)) >= dmin for q in pts): pts.append(p)
        xy = np.array(pts); s = max(psl(xy, 2.437e9), psl(xy, 5.22e9))
        if best is None or s < best[0]: best = (s, xy)
    return best
if __name__ == "__main__":
    s, xy = search()
    nano = np.array([[-25.2, 39], [-4.5, 39], [23.8, 39], [-22.5, -39], [0.8, -39], [21.8, -39], [-39, -26.5], [-39, -5.8], [-39, 21.2], [39, -27.2], [39, -3.0], [39, 26.8]]) * 1e-3
    print("frame12 peak sidelobe (worst of both bands):", round(20 * np.log10(s), 1), "dB | nano84:", round(20 * np.log10(max(psl(nano, 2.437e9), psl(nano, 5.22e9))), 1), "dB")
    np.save("frame12.npy", xy); print(np.round(xy * 1000).astype(int).tolist())
