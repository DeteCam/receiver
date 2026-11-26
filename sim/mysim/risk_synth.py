"""Switched-ring CSI synthesis with hardware / driver risk models (software-only risk study).

Same physics and the same random-draw order as channel_sim.synthesize_ring_csi, so with every risk switched off the
output is bit-identical to the repository synthesiser (checked in exp_risk.py).  Every risk-specific random draw comes
from a separate generator derived from the call's generator state, so switching one risk on does not change the noise,
people or calibration draws of the baseline (paired comparison).

Risk models (all off by default):
  p_swap / swap_fix          AX210 reports chain 0 and chain 1 swapped for a packet.  swap_fix: decide per packet from
                             the per-chain energy fields (1 dB resolution): the anchored chain must be the stronger one.
  agc (dict)                 per-chain gain states (step_db wide, chosen from the packet level), each state with its
                             own complex gain error (ph_sig_deg, amp_sig_db).  mode "indep" | "joint".
                             raw=True: the reported CSI also carries the nominal gain step (not compensated by firmware).
                             lvl_jit_db: packet-to-packet transmit level change of the camera (rate / power control).
                             fix: None | "table" -> per-state complex-gain table learnt from all-off slots, state read from
                             the energy fields (table error tab_ph_deg / tab_amp_db).
  p_mis / mis_guard          packet labelled with the wrong switch port (neighbour port).  mis_guard: packets near a
                             switching edge are flagged and dropped (drop 20 %, residual mislabel rate / 10).
  agg ("mean"|"median"|"trim")  per-port aggregation over sweeps (trim keeps the 70 % closest to the median).
  leak_db                    switch isolation: every port also carries all other elements at leak_db (fixed phases).
  alpha_err_deg / alpha_err_db   residual error of the anchor calibration (default 3 deg / 0.25 dB as the repository).
  iq_conj / iq_selftest      capture tool reads I/Q in the wrong order (= complex conjugate); self-test: sign of the
                             anchor delay slope across subcarriers (anchor_delay_ns) -> conjugate back.
"""
import zlib
import numpy as np

C0 = 299792458.0


def _risk_rng(rng):
    st = repr(rng.bit_generator.state).encode()
    return np.random.default_rng(zlib.crc32(st) ^ 0x5A5A1234)


def _agg(r, mode, keep=None):
    """r: (T, N, K) per-packet port ratios; keep: (T, N) bool mask of usable packets."""
    T, N, K = r.shape
    if keep is None:
        keep = np.ones((T, N), dtype=bool)
    out = np.zeros((N, K), dtype=np.complex128)
    for m in range(N):
        rr = r[keep[:, m], m, :]
        if rr.shape[0] == 0:
            rr = r[:, m, :]
        if mode == "mean" or rr.shape[0] < 3:
            out[m] = rr.mean(axis=0)
        elif mode == "median":
            out[m] = np.median(rr.real, axis=0) + 1j * np.median(rr.imag, axis=0)
        elif mode == "trim":
            med = np.median(rr.real, axis=0) + 1j * np.median(rr.imag, axis=0)
            d = np.sqrt(np.mean(np.abs(rr - med[None]) ** 2, axis=1))
            nk = max(2, int(np.ceil(0.7 * rr.shape[0])))
            out[m] = rr[np.argsort(d)[:nk]].mean(axis=0)
        else:
            raise ValueError(mode)
    return out


def packet_features(Y0, R_raw, ratios, metric, dE, P0, P1, df_hz, anchor_delay_ns):
    """Per-packet features that a real receiver can compute (T, N, 8):
    0 energy difference of the two chains (dB, from the energy fields)
    1 median log|Y1/Y0| over subcarriers
    2 phase slope of Y1/Y0 across subcarriers in units of the designed anchor delay (+1 = as designed)
    3 deviation of the anchor-corrected ratio from its port's median over sweeps (relative to the port's typical one)
    4 reference-chain magnitude-shape change against the median over all packets (1 - correlation)
    5, 6 level of each chain against its median (dB)
    7 sign-decision confidence (|metric| relative to its median)"""
    T, N, K = R_raw.shape
    f = df_hz - df_hz.mean()
    ph = np.unwrap(np.angle(R_raw), axis=2)
    slope = np.sum((ph - ph.mean(axis=2, keepdims=True)) * f, axis=2) / np.sum(f ** 2)
    slope_n = slope / (2 * np.pi * anchor_delay_ns * 1e-9) * -1.0
    med = np.median(ratios.real, axis=0) + 1j * np.median(ratios.imag, axis=0)
    dev = np.sqrt(np.mean(np.abs(ratios - med[None]) ** 2, axis=2))
    dev_n = dev / (np.median(dev, axis=0, keepdims=True) + 1e-12)
    a0 = np.abs(Y0)
    ma = np.median(a0.reshape(T * N, K), axis=0)
    a0c = a0 - a0.mean(axis=2, keepdims=True)
    mac = ma - ma.mean()
    corr = np.sum(a0c * mac[None, None, :], axis=2) / (np.linalg.norm(a0c, axis=2) * np.linalg.norm(mac) + 1e-12)
    lm = np.median(np.log(np.abs(R_raw) + 1e-12), axis=2)
    am = np.abs(metric[:, :, 0])
    conf = am / (np.median(am) + 1e-12)
    return np.stack([dE, lm, slope_n, dev_n, 1.0 - corr, P0 - np.median(P0), P1 - np.median(P1), conf], axis=2)


def synth_risk(paths_rn, cam_pos, node_pos, arr_pat, df_hz, rng, n_sweep=10, dyn_frac=0.35, dyn_std_deg=45.0,
               cal_phase_deg=4.5, cal_amp_db=0.4, snr_db=20.0, flip_mode="sign_anchor", anchor_alpha=3.5,
               agg_mode="ratio", include_ref=True, weight_by_ref_amp=True,
               p_swap=0.0, swap_fix=False, agc=None, p_mis=0.0, mis_guard=False, agg="mean", leak_db=None,
               alpha_err_deg=3.0, alpha_err_db=0.25, anchor_delay_ns=1.5, iq_conj=False, iq_selftest=False,
               cal_in=None, order_model=None, ref_pad_db=0.0, order_proc="stronger", drop_margin_db=3.0,
               rss_read_noise_db=0.3, collide=None, qc_model=None, qc_thr=0.5, qc_rule=False, return_pkt=False,
               sw_pad_db=0.0):
    """cal_in: calibration residuals of another sweep of the SAME receiver (port_cal, anchor error, reference row);
    the draws are still made (stream alignment) but replaced, as one device's calibration does not change between
    two head orientations.  The residuals used are returned in stats['cal_out']."""
    assert flip_mode == "sign_anchor" and agg_mode in ("ratio", "conj")
    aV, aH, tau, th, ph = paths_rn
    pw = np.abs(aV) ** 2 + np.abs(aH) ** 2
    valid = pw > 1e-24
    if not np.any(valid):
        M = arr_pat.n_elem + (1 if include_ref else 0)
        return np.zeros((M, len(df_hz)), dtype=np.complex64), dict(sign_acc=1.0, los_frac=0.0)
    aV, aH, tau, th, ph, pw = aV[valid], aH[valid], tau[valid], th[valid], ph[valid], pw[valid]
    d_los = float(np.linalg.norm(cam_pos - node_pos))
    is_los = np.abs(tau - d_los / C0) < 0.08e-9
    los_frac = float(pw[is_los].sum() / (pw.sum() + 1e-30))
    G_ring, G_ref = arr_pat.element_responses(th, ph, aV, aH)
    Fq = np.exp(-2j * np.pi * tau[:, None] * df_hz[None, :])
    N = arr_pat.n_elem
    K = len(df_hz)
    T = n_sweep

    # ---- identical draw order to channel_sim.synthesize_ring_csi ----
    dyn_mask = (~is_los) & (rng.random(len(tau)) < dyn_frac) if (dyn_frac > 0 and dyn_std_deg > 0) else np.zeros(len(tau), dtype=bool)
    n_dyn = int(dyn_mask.sum())
    if cal_phase_deg > 0 or cal_amp_db > 0:
        port_cal = (10.0 ** (rng.normal(0.0, cal_amp_db, N) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, cal_phase_deg, N)))
    else:
        port_cal = np.ones(N, dtype=np.complex128)
    H_ref_static = G_ref @ Fq
    ref_pwr = float(np.mean(np.abs(H_ref_static) ** 2) + 1e-30)
    noise_std = np.sqrt(ref_pwr * (10.0 ** (-snr_db / 10.0)) / 2.0) if snr_db < 90.0 else 0.0
    alpha_true = anchor_alpha * np.exp(1j * 0.4) * np.exp(-2j * np.pi * anchor_delay_ns * 1e-9 * df_hz)
    if cal_phase_deg > 0:
        z_a, z_p = rng.normal(0.0, 1.0), rng.normal(0.0, 1.0)
        a_err = (10.0 ** (z_a * alpha_err_db / 20.0)) * np.exp(1j * np.radians(z_p * alpha_err_deg))
    else:
        a_err = 1.0 + 0j
    if cal_in is not None:
        port_cal, a_err = cal_in["port_cal"], cal_in["a_err"]
    alpha_cal = alpha_true * a_err
    if n_dyn > 0:
        stat_mask = ~dyn_mask
        H_ring_stat = (port_cal[:, None] * G_ring[:, stat_mask]) @ Fq[stat_mask]
        H_ref_stat = G_ref[stat_mask] @ Fq[stat_mask]
        Fq_dyn = Fq[dyn_mask]
        ph_dyn = np.exp(1j * np.radians(rng.normal(0.0, dyn_std_deg, (T, N, n_dyn))))
        G_ring_dyn = port_cal[:, None] * G_ring[:, dyn_mask]
        Hm_all = H_ring_stat[None, :, :] + (ph_dyn * G_ring_dyn[None, :, :]) @ Fq_dyn
        Hr_all = H_ref_stat[None, None, :] + (ph_dyn * G_ref[None, None, dyn_mask]) @ Fq_dyn
    else:
        H_ring_stat = (port_cal[:, None] * G_ring) @ Fq
        Hm_all = np.broadcast_to(H_ring_stat, (T, N, K)).copy()
        Hr_all = np.broadcast_to(H_ref_static[None, None, :], (T, N, K)).copy()
    pkt_ph = np.exp(1j * rng.uniform(0.0, 2.0 * np.pi, (T, N, 1)))
    if noise_std > 0:
        n0 = noise_std * (rng.standard_normal((T, N, K)) + 1j * rng.standard_normal((T, N, K)))
        n1 = noise_std * (rng.standard_normal((T, N, K)) + 1j * rng.standard_normal((T, N, K)))
    else:
        n0 = n1 = 0.0
    s_true = np.where(rng.random((T, N, 1)) < 0.5, 1.0, -1.0)
    # c_ref is drawn after aggregation in the repository code; draw it now from a copy so the order is kept
    rr = _risk_rng(rng)
    st_extra = {}

    # ---- risk: switch isolation (other elements leak into every port) ----
    if leak_db is not None:
        eps = 10.0 ** (leak_db / 20.0) * np.exp(1j * rr.uniform(0, 2 * np.pi, (N, N)))
        np.fill_diagonal(eps, 0.0)
        Hm_all = Hm_all + (eps @ H_ring_stat)[None]
    # ---- risk: packet labelled with a neighbouring port ----
    keep = np.ones((T, N), dtype=bool)
    mis_mask = np.zeros((T, N), dtype=bool)
    if p_mis > 0:
        pm = p_mis / 10.0 if mis_guard else p_mis
        mis = rr.random((T, N)) < pm
        shift = np.where(rr.random((T, N)) < 0.5, 1, -1)
        src = (np.arange(N)[None, :] + np.where(mis, shift, 0)) % N
        Hm_all = np.take_along_axis(Hm_all, src[:, :, None], axis=1)
        mis_mask = mis
        if mis_guard:
            keep = rr.random((T, N)) >= 0.20
    elif mis_guard:
        keep = rr.random((T, N)) >= 0.20

    pad = 10.0 ** (-ref_pad_db / 20.0)          # fixed attenuator on the reference port (signal only; noise stays)
    Y0 = pkt_ph * (pad * Hr_all) + n0
    pad1 = 10.0 ** (-sw_pad_db / 20.0)         # fixed attenuator at the chain-B input (anchor + ring together; noise stays)
    Y1 = pkt_ph * (s_true * (pad1 * (Hm_all + alpha_true[None, None, :] * Hr_all))) + n1
    if ref_pad_db:
        alpha_cal = alpha_cal / pad               # the all-off calibration sees c / p
    if sw_pad_db:
        alpha_cal = alpha_cal * pad1
    bad = np.zeros((T, N), dtype=bool)          # ground truth: packet corrupted by some risk (for the QC study)
    if collide:
        # another transmitter overlaps the packet: an independent smooth frequency-selective term on each chain
        col = rr.random((T, N)) < collide.get("p", 0.05)
        sir = rr.uniform(collide.get("sir_lo", 0.0), collide.get("sir_hi", 15.0), (T, N))
        nt = 6
        tau_i = rr.uniform(0.0, 150e-9, (2, T, N, nt))
        g_i = (rr.standard_normal((2, T, N, nt)) + 1j * rr.standard_normal((2, T, N, nt))) / np.sqrt(2 * nt)
        I = np.einsum("ctnp,ctnpk->ctnk", g_i, np.exp(-2j * np.pi * tau_i[..., None] * df_hz[None, None, None, None, :]))
        p0 = np.mean(np.abs(Y0) ** 2, axis=2, keepdims=True)
        p1 = np.mean(np.abs(Y1) ** 2, axis=2, keepdims=True)
        sc = 10.0 ** (-sir[:, :, None] / 20.0) * col[:, :, None]
        Y0 = Y0 + sc * np.sqrt(p0) * I[0]
        Y1 = Y1 + sc * np.sqrt(p1) * I[1]
        bad |= col

    # ---- risk: per-chain AGC gain states ----
    P0 = 10 * np.log10(np.mean(np.abs(Y0) ** 2, axis=2) + 1e-30)   # input level per chain (what energy_a/b report)
    P1 = 10 * np.log10(np.mean(np.abs(Y1) ** 2, axis=2) + 1e-30)
    if agc:
        step = float(agc.get("step_db", 2.0))
        jit = rr.normal(0.0, float(agc.get("lvl_jit_db", 2.0)), (T, N, 1))
        g_lin = 10.0 ** (jit / 20.0)
        Y0, Y1 = Y0 * g_lin, Y1 * g_lin          # same packet -> same transmit level on both chains
        P0, P1 = P0 + jit[:, :, 0], P1 + jit[:, :, 0]
        u = rr.uniform(0.0, 1.0, 2)
        if agc.get("mode", "indep") == "joint":
            Pm = np.maximum(P0, P1)
            g0 = g1 = np.floor((Pm - np.median(Pm)) / step + u[0]).astype(int)
        else:
            g0 = np.floor((P0 - np.median(P0)) / step + u[0]).astype(int)
            g1 = np.floor((P1 - np.median(P1)) / step + u[1]).astype(int)
        ns = 41
        tab = (10.0 ** (rr.normal(0.0, agc.get("amp_sig_db", 0.3), (2, ns)) / 20.0)
               * np.exp(1j * np.radians(rr.normal(0.0, agc.get("ph_sig_deg", 7.0), (2, ns)))))
        tab[:, ns // 2] = 1.0                       # calibration state = state 0 (anchor / ports calibrated there)
        gi0, gi1 = np.clip(g0 + ns // 2, 0, ns - 1), np.clip(g1 + ns // 2, 0, ns - 1)
        c0, c1 = tab[0, gi0], tab[1, gi1]
        # raw=True: the reported CSI carries the analogue gain, i.e. each chain is normalised to the same target
        # level (the +6..11 dB anchor excess of chain 1 disappears from |Y1/Y0|) and moves in step_db steps
        off1 = float(np.median(P1) - np.median(P0))
        if agc.get("raw", False):
            c0 = c0 * 10.0 ** (-(g0 * step) / 20.0)
            c1 = c1 * 10.0 ** (-(g1 * step + off1) / 20.0)
        Y0 = Y0 * c0[:, :, None]
        Y1 = Y1 * c1[:, :, None]
        agc_fix = None
        if agc.get("fix") == "table":
            # per-state complex gain table learnt from all-off slots (table error e_ph / e_am); the state of each
            # packet is read from energy field vs raw CSI level, wrong by one step when that reading is off by more
            # than half a step (state_err_db = rms error of the reading)
            e_ph, e_am = agc.get("tab_ph_deg", 1.0), agc.get("tab_amp_db", 0.1)
            tab_hat = tab * (10.0 ** (rr.normal(0.0, e_am, (2, ns)) / 20.0)) * np.exp(1j * np.radians(rr.normal(0.0, e_ph, (2, ns))))
            tab_hat[:, ns // 2] = 1.0
            se = float(agc.get("state_err_db", 0.0))
            gh0 = g0 + (np.round(rr.normal(0.0, se, g0.shape) / step).astype(int) if se > 0 else 0)
            gh1 = g1 + (np.round(rr.normal(0.0, se, g1.shape) / step).astype(int) if se > 0 else 0)
            h0 = tab_hat[0, np.clip(gh0 + ns // 2, 0, ns - 1)]
            h1 = tab_hat[1, np.clip(gh1 + ns // 2, 0, ns - 1)]
            if agc.get("raw", False):
                h0 = h0 * 10.0 ** (-(gh0 * step) / 20.0)
                h1 = h1 * 10.0 ** (-(gh1 * step + off1) / 20.0)
            agc_fix = (h0, h1)                       # applied after the chain-swap check (needs chain identity)
            st_extra["state_ok"] = float(np.mean((gh0 == g0) & (gh1 == g1)))
        st_extra["agc_states"] = float(np.mean((g0 != 0) | (g1 != 0)))
        agc_bad = ((g0 != 0) | (g1 != 0)) if agc.get("fix") != "table" else ((gh0 != g0) | (gh1 != g1))
    else:
        agc_fix = None
        agc_bad = np.zeros((T, N), dtype=bool)

    # ---- AX210 RSS-ordered antenna selection (Liang et al. 2024, IAX 2025): the first RF chain takes the port with
    # the larger RSS (integer dB readings, ties either way); the CSI rows come out in chain order ----
    if order_model == "rss":
        off = rr.uniform(0.0, 1.0, (T, N))                    # where the integer boundaries fall for this packet
        rS = np.floor(P1 + rr.normal(0.0, rss_read_noise_db, (T, N)) + off)
        rR = np.floor(P0 + rr.normal(0.0, rss_read_noise_db, (T, N)) + off)
        sw_first = (rS > rR) | ((rS == rR) & (rr.random((T, N)) < 0.5))
        A = np.where(sw_first[:, :, None], Y1, Y0)
        B = np.where(sw_first[:, :, None], Y0, Y1)
        EA, EB = np.maximum(rS, rR), np.minimum(rS, rR)
        st_extra["order_flip"] = float(np.mean(~sw_first))
        if order_proc in ("naive", "refdrop"):            # the first (stronger) row is taken as the reference chain
            Y0, Y1 = A, B
            if order_proc == "refdrop":
                ok = (EA - EB) >= drop_margin_db
                keep = keep & ok
                st_extra["dropped"] = float(np.mean(~ok))
            st_extra["ref_wrong"] = float(np.mean(sw_first))            # first row is really the switch chain
            st_extra["ref_wrong_kept"] = float(np.mean(sw_first & keep))
        else:                                             # design rule: chain A (stronger) = switch port
            Y1, Y0 = A, B
            if order_proc == "drop":
                ok = (EA - EB) >= drop_margin_db
                keep = keep & ok
                st_extra["dropped"] = float(np.mean(~ok))
        if order_proc not in ("naive", "refdrop"):
            st_extra["kept_wrong"] = float(np.mean((~sw_first) & keep))
            bad |= ~sw_first
        dE_read = EA - EB
    else:
        dE_read = np.round(P1) - np.round(P0)

    # ---- risk: chain swap ----
    swp = np.zeros((T, N), dtype=bool)
    if p_swap > 0:
        swp = rr.random((T, N)) < p_swap
        Y0s = np.where(swp[:, :, None], Y1, Y0)
        Y1s = np.where(swp[:, :, None], Y0, Y1)
        Y0, Y1 = Y0s, Y1s
        if swap_fix:
            if swap_fix == "mag":   # raw CSI magnitude ratio (fails if per-chain AGC rescales the CSI)
                E0 = 10 * np.log10(np.mean(np.abs(Y0) ** 2, axis=2) + 1e-30)
                E1 = 10 * np.log10(np.mean(np.abs(Y1) ** 2, axis=2) + 1e-30)
            else:                   # energy fields: input power per chain label, 1 dB steps (+ 0.5 dB reading noise)
                E0 = np.round(np.where(swp, P1, P0) + rr.normal(0, 0.5, (T, N)))
                E1 = np.round(np.where(swp, P0, P1) + rr.normal(0, 0.5, (T, N)))
            dec = E1 < E0
            st_extra["swap_det_acc"] = float(np.mean(dec == swp))
            Y0b = np.where(dec[:, :, None], Y1, Y0)
            Y1b = np.where(dec[:, :, None], Y0, Y1)
            Y0, Y1 = Y0b, Y1b
    if agc_fix is not None:
        Y0 = Y0 / agc_fix[0][:, :, None]
        Y1 = Y1 / agc_fix[1][:, :, None]

    # ---- risk: capture tool conjugates the CSI (I/Q order) ----
    if iq_conj:
        Y0, Y1 = np.conj(Y0), np.conj(Y1)
    if iq_conj or iq_selftest:
        if iq_selftest:
            # anchor dominates Y1/Y0: phase slope across subcarriers = -2 pi tau_anchor (correct order) or + (conjugated)
            R = Y1 / (Y0 + 1e-30)
            ph_u = np.unwrap(np.angle(R), axis=2)
            f = df_hz - df_hz.mean()
            slope = np.sum((ph_u - ph_u.mean(axis=2, keepdims=True)) * f[None, None, :], axis=2) / np.sum(f ** 2)
            conj_hat = np.median(slope) > 0
            st_extra["iq_ok"] = float(conj_hat == bool(iq_conj))
            if conj_hat:
                Y0, Y1 = np.conj(Y0), np.conj(Y1)
                # a conjugating tool would also have conjugated the anchor calibration; after the self-test both agree

    if iq_conj and not (iq_selftest and st_extra.get("iq_ok", 0.0) == 1.0):
        alpha_cal = np.conj(alpha_cal)          # the anchor was calibrated with the same (conjugating) tool
    ref_amps = np.abs(Y0)
    R_raw = Y1 / (Y0 + 1e-30)
    w_mf = ref_amps ** 2
    metric = np.real(np.sum(w_mf * np.conj(alpha_cal)[None, None, :] * R_raw, axis=2, keepdims=True))
    s_hat = np.where(metric >= 0.0, 1.0, -1.0)
    sign_acc = float(np.mean(s_hat == s_true))
    ratios = (s_hat * R_raw - alpha_cal[None, None, :]) * (pad / pad1)
    bad = bad | mis_mask | agc_bad
    if return_pkt or qc_model is not None or qc_rule:
        F = packet_features(Y0, R_raw, ratios, metric, dE_read, P0, P1, df_hz, anchor_delay_ns)
        if qc_model is not None:
            pb = qc_model.predict_proba(F.reshape(T * N, -1))[:, 1].reshape(T, N)
            flag = pb >= qc_thr
        elif qc_rule:
            flag = (F[:, :, 3] > 3.0) | (F[:, :, 0] < 3.0) | (F[:, :, 2] < 0.0)
        else:
            flag = np.zeros((T, N), dtype=bool)
        if qc_model is not None or qc_rule:
            # never empty a port: keep its least suspicious packet
            for m in range(N):
                if not np.any(~flag[:, m] & keep[:, m]):
                    flag[np.argmin(F[:, m, 3]), m] = False
            st_extra["qc_flag"] = float(np.mean(flag))
            st_extra["qc_recall"] = float(np.mean(flag[bad])) if bad.any() else 1.0
            st_extra["qc_fpr"] = float(np.mean(flag[~bad])) if (~bad).any() else 0.0
            keep = keep & ~flag
        if return_pkt:
            st_extra["pkt"] = dict(F=F.reshape(T * N, -1).astype(np.float32), y=bad.reshape(-1))
    st_extra["bad_frac"] = float(np.mean(bad))
    if agg == "mean" and agg_mode == "conj":
        conj_num = (s_hat * Y1 - alpha_cal[None, None, :] * Y0) * np.conj(Y0)
        if keep.all():
            X_ring = np.mean(conj_num, axis=0) / (np.mean(ref_amps ** 2, axis=0) + 1e-30)
        else:
            kk = keep[:, :, None]
            X_ring = np.sum(conj_num * kk, axis=0) / (np.sum(ref_amps ** 2 * kk, axis=0) + 1e-30)
        X_ring = X_ring * (pad / pad1)
    elif agg == "mean" and keep.all():
        X_ring = np.mean(ratios, axis=0)
    else:
        X_ring = _agg(ratios, agg, keep)

    c_ref = 1.0 + 0j
    if include_ref:
        if cal_phase_deg > 0:
            c_ref = (10.0 ** (rng.normal(0.0, 0.2) / 20.0)) * np.exp(1j * np.radians(rng.normal(0.0, 2.0)))
        if cal_in is not None:
            c_ref = cal_in["c_ref"]
        X = np.vstack([X_ring, np.full((1, K), c_ref, dtype=np.complex128)])
    else:
        X = X_ring
    if weight_by_ref_amp:
        w_k = np.sqrt(np.mean(ref_amps ** 2, axis=(0, 1)))
        w_k = w_k / (np.sqrt(np.mean(w_k ** 2)) + 1e-30)
        X = X * w_k[None, :]
    st = dict(sign_acc=sign_acc, los_frac=los_frac)
    st.update(st_extra)
    st["cal_out"] = dict(port_cal=port_cal, a_err=a_err, c_ref=c_ref)
    return X.astype(np.complex64), st
