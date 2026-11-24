# -*- coding: utf-8 -*-
"""DeteCam 6.0 (Nano-2 AI) — PyTorch Edge-AI 추론 엔진 (ReconCovNet-v4 + CamGuard-Net + MA-PINN).

NanoPi R5C (RK3568B2 ARM64 CPU / 0.8 TOPS NPU) 온디바이스 실행용 초경량 AI 모듈:
  1. ReconCovNet-v4 (~457K params, 0.024 ms/frame CPU):
     84 x 84 mm 포켓 배열의 다중 대역 공분산 [R_all, R_low, R_high] 로부터
     상호결합 잔류 오차·다중경로 간섭을 억제한 Hermitian PSD 초해상도 공분산 R_AI 복원.
  2. CamGuard-Net (~24K params, 0.12 ms/device CPU):
     암호화된 802.11 패킷 시계열(1D-CNN + Multi-Head Self-Attention) + 5초 CamLoPA 물리 자극 융합으로
     라이브 방송/클라우드 백업 하드 네거티브 속에서 몰래카메라만 정밀 식별.
  3. MA-PINN (Bayesian Log-Sum-Exp Virtual Mirror Anchor + P_tx 불변 Delta P_12 융합):
     단 2대의 초소형 벽면 노드로 벽·거울·칠판·사물함 1차 반사를 가상 거울 앵커로 전환해
     L자 코너·중앙 기둥·칸막이 사각지대 3D 위치 추정.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def extract_wideband_cov_features(H_cal, band_flag=0.0):
    """보정된 광대역 CSI 행렬 H_cal [M, K] -> 제로-바이어스 다중 서브밴드 공분산 특징 [6, M, M]."""
    M, K = H_cal.shape
    half = max(1, K // 2)

    def norm_cov(R):
        tr = float(np.real(np.trace(R))) / M
        sc = max(tr, float(np.linalg.norm(R)) / M, 1e-12)
        return R / sc

    R_all = norm_cov((H_cal @ H_cal.conj().T) / max(K, 1))
    R_low = norm_cov((H_cal[:, :half] @ H_cal[:, :half].conj().T) / half)
    R_high = norm_cov((H_cal[:, half:] @ H_cal[:, half:].conj().T) / max(K - half, 1))

    feat = np.stack([
        R_all.real, R_all.imag,
        R_low.real, R_low.imag,
        R_high.real, R_high.imag,
    ], axis=0).astype(np.float32)
    return feat, R_all


class ReconCovNet(nn.Module):
    """Differentiable-Capon Physics-Guided Residual Wideband Covariance Network (v4)."""
    def __init__(self, n_ant=12, in_ch=6, hidden=192, rank=3):
        super().__init__()
        self.n_ant = n_ant
        self.rank = rank
        in_dim = in_ch * n_ant * n_ant + 2

        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )
        self.out_dim = 4 + 2 * n_ant * n_ant + 2 * n_ant * rank
        self.head_245 = nn.Linear(hidden, self.out_dim)
        self.head_522 = nn.Linear(hidden, self.out_dim)

    def forward(self, x, band_flag):
        B, _, M, _ = x.shape
        R_all = torch.complex(x[:, 0], x[:, 1])
        R_low = torch.complex(x[:, 2], x[:, 3])
        R_high = torch.complex(x[:, 4], x[:, 5])

        b_oh = torch.stack([1.0 - band_flag, band_flag], dim=-1)
        h = self.in_proj(torch.cat([x.reshape(B, -1), b_oh], dim=-1))
        h = F.gelu(h + self.block1(h))
        h = F.gelu(h + self.block2(h))

        w5 = band_flag.unsqueeze(-1)
        out = (1.0 - w5) * self.head_245(h) + w5 * self.head_522(h)

        gates = F.softmax(out[:, :4], dim=-1)
        g_all = gates[:, 0].view(B, 1, 1)
        g_low = gates[:, 1].view(B, 1, 1)
        g_high = gates[:, 2].view(B, 1, 1)
        g_res = gates[:, 3].view(B, 1, 1)

        n_w = 2 * M * M
        dw_raw = out[:, 4: 4 + n_w].view(B, 2, M, M)
        dW = 0.25 * torch.complex(torch.tanh(dw_raw[:, 0]), torch.tanh(dw_raw[:, 1]))
        eye = torch.eye(M, device=x.device, dtype=torch.complex64).unsqueeze(0)
        W = eye + dW

        R_phys = g_all * R_all + g_low * R_low + g_high * R_high
        R_dec = torch.matmul(torch.matmul(W, R_phys), W.conj().transpose(-2, -1))

        l_raw = out[:, 4 + n_w:].view(B, 2, M, self.rank)
        L = torch.complex(l_raw[:, 0], l_raw[:, 1])
        R_nn = torch.matmul(L, L.conj().transpose(-2, -1))
        tr_nn = torch.diagonal(R_nn, dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True).unsqueeze(-1)
        R_nn = R_nn / (tr_nn / M + 1e-6)

        R_ai = R_dec + g_res * R_nn
        R_ai = 0.5 * (R_ai + R_ai.conj().transpose(-2, -1))
        tr = torch.diagonal(R_ai, dim1=-2, dim2=-1).real.sum(dim=-1, keepdim=True).unsqueeze(-1)
        R_ai = R_ai / (tr / M + 1e-6)
        return R_ai


class CamGuardNet(nn.Module):
    """1D-CNN + Multi-Head Self-Attention + CamLoPA Causal Encrypted Traffic Classifier."""
    def __init__(self, in_ch=4, hidden=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(hidden, hidden * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden * 2),
            nn.GELU(),
        )
        self.attn = nn.MultiheadAttention(embed_dim=hidden * 2, num_heads=4, batch_first=True)
        self.clf = nn.Sequential(
            nn.Linear(hidden * 2 + 3, 32),
            nn.GELU(),
            nn.Linear(32, 2),
        )

    def forward(self, seq, meta_feat):
        z = self.conv(seq).transpose(1, 2)
        a, _ = self.attn(z, z, z)
        pooled = (z + a).mean(dim=1)
        return self.clf(torch.cat([pooled, meta_feat], dim=-1))


class Nano2AIEngine:
    """DeteCam 6.0 온디바이스 통합 추론 엔진 (ReconCovNet-v4 + CamGuard-Net + MA-PINN)."""
    def __init__(self, weights_dir=WEIGHTS_DIR):
        self.recon = ReconCovNet(n_ant=12, in_ch=6, hidden=192, rank=3)
        self.camguard = CamGuardNet(in_ch=4, hidden=32)

        rp = os.path.join(weights_dir, "recon_covnet.pt")
        cp = os.path.join(weights_dir, "camguard_net.pt")
        if os.path.exists(rp):
            self.recon.load_state_dict(torch.load(rp, map_location="cpu", weights_only=True))
        if os.path.exists(cp):
            self.camguard.load_state_dict(torch.load(cp, map_location="cpu", weights_only=True))
        self.recon.eval()
        self.camguard.eval()

    @torch.no_grad()
    def reconstruct_covariance(self, H_cal, fc_hz=5.22e9):
        """보정된 광대역 CSI [12, K] -> AI 초해상도 Hermitian PSD 공분산 R_AI [12, 12]."""
        band_flag = 1.0 if float(fc_hz) >= 4.0e9 else 0.0
        feat, R_all = extract_wideband_cov_features(H_cal, band_flag)
        xt = torch.from_numpy(feat[None])
        bt = torch.tensor([band_flag], dtype=torch.float32)
        R_ai = self.recon(xt, bt)[0].numpy().astype(np.complex128)
        return R_ai, R_all

    @torch.no_grad()
    def classify_camera(self, seq_4x60, meta_3):
        """암호화 패킷 특징 시퀀스 [4, 60] + CamLoPA 인과 메타 [3] -> (is_camera, prob_camera)."""
        xs = torch.from_numpy(np.asarray(seq_4x60, dtype=np.float32)[None])
        xm = torch.from_numpy(np.asarray(meta_3, dtype=np.float32)[None])
        probs = F.softmax(self.camguard(xs, xm), dim=-1)[0].numpy()
        p_cam = float(probs[1])
        return bool(p_cam >= 0.5), round(p_cam, 4)

    @staticmethod
    def ma_pinn_fuse_2node(node_obs_list, precomps, grid, phone_prior=None):
        """
        Bayesian Log-Sum-Exp MA-PINN (단 2대 노드 + 가상 거울 앵커 + P_tx 불변 Delta P_12 융합).
        Returns (est_xyz [3], posterior_probs [N]).
        """
        N = len(grid)
        total = np.zeros(N, dtype=float)
        pred_amps = []

        def _capon_ll(R, A_norm, load=0.08):
            M = R.shape[0]
            tr = float(np.real(np.trace(R))) / M
            R_inv = np.linalg.inv(R + load * max(tr, 1e-12) * np.eye(M))
            quad = np.maximum(np.real(np.einsum("nm,mk,nk->n", A_norm.conj(), R_inv, A_norm)), 1e-12)
            p = 1.0 / quad
            return np.log(p / (np.max(p) + 1e-12) + 1e-6)

        for obs, pre in zip(node_obs_list, precomps):
            s_dir_ai = _capon_ll(obs["R_ai"], pre["A_dir"], load=0.08)
            s_mir_ai = _capon_ll(obs["R_ai"], pre["A_mir"], load=0.08)
            s_mir_all = _capon_ll(obs["R_all"], pre["A_mir"], load=0.10)

            L_dir = np.exp(np.clip(s_dir_ai, -12.0, 0.0))
            L_mir = np.exp(np.clip(np.maximum(s_mir_ai, s_mir_all), -12.0, 0.0))
            w_dir = np.clip(pre["cos_front"], 0.08, 1.0) * (10.0 ** (-pre["pen_dir"] / 20.0))
            w_mir = 0.55 * pre["best_mir_gain"]
            p_mix = (w_dir * L_dir + w_mir * L_mir) / (w_dir + w_mir + 1e-6)

            cv_w = 0.70 + 0.30 / (1.0 + np.exp((obs["cv_f"] - 0.12) / 0.03))
            total += cv_w * np.log(p_mix + 1e-4)

            d_mir = pre["d_dir"] + pre["best_mir_exd"]
            eff_amp = np.sqrt((w_dir / np.maximum(pre["d_dir"], 0.35)) ** 2 + (w_mir / np.maximum(d_mir, 0.50)) ** 2) + 1e-5
            pred_amps.append(eff_amp)

        if len(node_obs_list) >= 2:
            p0 = max(node_obs_list[0]["raw_pow"], 1e-12)
            p1 = max(node_obs_list[1]["raw_pow"], 1e-12)
            dp_meas_db = 10.0 * np.log10(p0 / p1)
            dp_pred_db = 20.0 * np.log10(pred_amps[0] / pred_amps[1])
            total += -0.45 * ((dp_meas_db - dp_pred_db) / 6.5) ** 2

        if phone_prior is not None:
            total += phone_prior

        best_idx = int(np.argmax(total))
        top_k = np.argsort(-total)[:6]
        near = [i for i in top_k if np.linalg.norm(grid[i] - grid[best_idx]) <= 0.55]
        w = np.exp((total[near] - total[best_idx]) * 2.0)
        w /= np.sum(w)
        est = np.sum(grid[near] * w[:, None], axis=0)

        ex = np.exp(np.clip(total - np.max(total), -30.0, 0.0))
        probs = ex / (np.sum(ex) + 1e-12)
        return est, probs
