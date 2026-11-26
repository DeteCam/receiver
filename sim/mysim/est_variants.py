"""Estimator variants for the realistic-room study (software candidates; same API as RingMapEstimator)."""
import os
import sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "v1_ring"))
from estimators import RingMapEstimator  # noqa: E402


class EstWV(RingMapEstimator):
    """Same estimator with another prior weight of the 'direct path visible' hypothesis / effective sample size."""
    def __init__(self, *a, w_vis=0.72, n_eff=10.0, **k):
        super().__init__(*a, **k)
        self._w_vis, self._n_eff = float(w_vis), float(n_eff)

    def log_likelihood(self, X, n_eff=None, w_vis=None):
        return super().log_likelihood(X, n_eff=self._n_eff, w_vis=self._w_vis)


class EstFeat(RingMapEstimator):
    """Same likelihood as RingMapEstimator (identical numbers); also keeps per-measurement evidence values that a
    reliability rule or model can use: the explained-energy scores with / without a direct path, and how much the
    reference chain's level changes over the subcarriers (frequency selectivity)."""
    def log_likelihood(self, X, n_eff=10.0, w_vis=0.72):
        X_use = X.astype(np.complex64)
        col_raw = np.linalg.norm(X_use, axis=0)
        if self.robust_huber:
            med = float(np.median(col_raw) + 1e-12)
            thr = med * (1.0 + self.huber_delta * 3.0)
            scale = np.where(col_raw > thr, thr / (col_raw + 1e-12), 1.0)
            X_use = X_use * scale[None, :].astype(np.float32)
        x_norm2 = float(np.sum(np.abs(X_use) ** 2))
        b_all = ((self._S_conj @ X_use) * self._Fq_conj) @ self.Ftc
        R = self.S.shape[1]
        s1 = self._score_precomputed(b_all, self._inv_all, np.arange(R), x_norm2)
        s0 = self._score_precomputed(b_all, self._inv_nlos, np.arange(1, R), x_norm2)
        # direct path alone (no wall images): matched filter of the first column
        b0 = b_all[:, 0, :]
        g00 = np.real(self.G[:, 0, 0]) + 1e-12
        sd = np.clip(np.max(np.abs(b0) ** 2, axis=1) / (g00 * (x_norm2 + 1e-30)), 0.0, 0.999)
        if self.robust_huber:
            nu = 6.0
            l1 = -0.5 * (nu + 1.0) * np.log(1.0 + (2.0 * n_eff / nu) * np.maximum(1.0 - s1, 5e-3))
            l0 = -0.5 * (nu + 1.0) * np.log(1.0 + (2.0 * n_eff / nu) * np.maximum(1.0 - s0, 5e-3))
        else:
            l1 = -n_eff * np.log(1.0 - s1 + 2e-3)
            l0 = -n_eff * np.log(1.0 - s0 + 2e-3)
        ll = np.logaddexp(np.log(w_vis) + l1, np.log(1.0 - w_vis) + l0)
        c = int(np.argmax(ll))
        ref_row = np.abs(X[-1]).astype(np.float64)
        ring = np.abs(X[:-1]).astype(np.float64)
        self.last = np.array([
            float(s1.max()), float(s0.max()), float(sd.max()), float(s1[c]), float(s0[c]), float(sd[c]),
            float(l1[c] - l0[c]),                                         # evidence for a visible direct path at the peak
            float(np.std(ref_row) / (np.mean(ref_row) + 1e-12)),          # frequency selectivity of the reference chain
            float(np.std(ring.mean(axis=1)) / (ring.mean() + 1e-12)),     # level spread over the ring elements
            float(ring.mean() / (ref_row.mean() + 1e-12)),                # ring / reference level
        ], dtype=np.float32)
        return ll, self.azimuth_spread(ll)
