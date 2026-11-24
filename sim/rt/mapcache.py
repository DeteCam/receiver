"""Pre-factorised map-aware likelihood used by rt_eval.py (direct path + wall images, Gaussian path gains)."""
import numpy as np


class Cache:
    """Per (group, subarray) pre-factorised model for the rt estimators."""

    def __init__(self, A, P, alpha=0.12, blk_db=15.0):
        self.A = A                                        # (N, C, K, F)
        self.N = A.shape[0]
        C, K = P.shape
        self.s0 = alpha + 1e-3
        self.AHA = np.einsum("nckf,nclf->cfkl", np.conj(A), A)
        self.variants = {}
        for name, Pm in self._variants(P, blk_db).items():
            hyps = []
            for Pv in Pm:
                Pn = Pv / (Pv.sum(1, keepdims=True) + 1e-12)
                valid = Pn > 1e-9
                Pinv = np.where(valid, 1.0 / np.maximum(Pn, 1e-9), 1e9)
                Mk = self.AHA + self.s0 * (Pinv[:, None, :, None] * np.eye(K)[None, None])
                Minv = np.linalg.inv(Mk)
                PA = (np.where(valid, Pn, 0.0)[:, None, :, None] * self.AHA) / self.s0
                _, ld = np.linalg.slogdet(np.eye(K)[None, None] + PA)
                hyps.append((Minv, self.N * np.log(self.s0) + ld))
            self.variants[name] = hyps

    @staticmethod
    def _variants(P, blk_db):
        Pd = np.zeros_like(P)
        Pd[:, 0] = np.maximum(P[:, 0], 1e-9)
        P2 = P.copy()
        P2[:, 0] *= 10 ** (-blk_db / 10)
        return {"direct": [Pd], "full": [P, P2]}

    def ll(self, y, variant, f_eff=3.0, w_vis=0.7):
        B, N, F = y.shape
        yy = np.sum(np.abs(y) ** 2, axis=1)                             # (B, F)
        rhs = np.einsum("nckf,bnf->cfkb", np.conj(self.A), y)           # (C, F, K, B)
        outs = []
        for Minv, logdet in self.variants[variant]:
            sol = np.einsum("cfkl,cflb->cfkb", Minv, rhs)
            quad = np.real(np.sum(np.conj(rhs) * sol, axis=2))          # (C, F, B)
            q = np.maximum((yy.T[None] - quad) / self.s0, 1e-12)
            chat = q.sum(axis=(1, 2)) / (N * F * B)
            outs.append((-N * F * B * np.log(chat) - B * logdet.sum(1)) * (f_eff / F))
        if len(outs) == 1:
            return outs[0]
        return np.logaddexp(np.log(w_vis) + outs[0], np.log(1 - w_vis) + outs[1])
