"""Switch state <-> CSI packet association (plan sim/plan_20261126_risk3.md section 1).  [Monte Carlo, not a measurement]
The host steps the SP16T through 17 states (all-off + 16 elements), dwell D each.  A packet is assigned to the state
that was commanded at its time stamp; packets within g of a switching instant are dropped.
  "host"   : time stamp = host arrival time (delay + jitter)
  "marker" : NIC time stamps (exact packet spacing, unknown constant offset to the host clock); the offset is found from
             the packets that look like the all-off state (|B/A - alpha| / |alpha| < 0.2), then the same rule is used
Run: python3 mysim/risk_switch_sync.py"""
import numpy as np

N_STATE = 17
RATES = (10.0, 26.0, 50.0, 94.0)
DWELL = (0.060, 0.100, 0.150, 0.250)
GUARD = (0.0, 0.005, 0.010, 0.020)
T_SW = 0.0005          # switch settling (assumption)


def arrivals(rate, t_end, burst, rng):
    if not burst:
        n = rng.poisson(rate * t_end)
        return np.sort(rng.uniform(0.0, t_end, n))
    fps = 15.0                                   # frames of a video stream, each sent as a short burst
    per = max(1, int(round(rate / fps)))
    t0 = np.arange(0.0, t_end, 1.0 / fps) + rng.uniform(0, 1.0 / fps)
    t = (t0[:, None] + np.sort(rng.uniform(0.0, 0.002, (len(t0), per)), axis=1)).ravel()
    return np.sort(t[t < t_end])


def host_delay(n, bad, rng):
    d = rng.exponential(0.003 if bad else 0.001, n)
    if bad:
        late = rng.random(n) < 0.05
        d = d + late * rng.uniform(0.010, 0.060, n)
    return d


def run(rate, D, g, mode, bad, burst, rng, n_cycles=40):
    t_end = n_cycles * N_STATE * D
    t = arrivals(rate, t_end, burst, rng)
    true_state = (np.floor(t / D).astype(int)) % N_STATE          # state 0 = all-off
    in_transition = (t % D) < T_SW                                  # packet received while the switch settles
    if mode == "host":
        ts = t + host_delay(len(t), bad, rng)
        ts = ts - (0.003 if bad else 0.001)                         # the mean delay is known and removed
    else:
        off = rng.uniform(0.0, N_STATE * D)                         # unknown offset of the NIC clock
        tn = t + off
        # all-off look: residual |B/A - alpha| / |alpha|
        resid = np.where(true_state == 0, 0.0, 0.5 * rng.rayleigh(1.0 / np.sqrt(2.0), len(t)) * np.sqrt(2.0))
        resid = np.abs(resid + 0.1 * (rng.standard_normal(len(t)) + 1j * rng.standard_normal(len(t))) / np.sqrt(2.0))
        looks_off = resid < 0.2
        # offset search: the phase of the cycle that puts most all-off looking packets into the all-off slot
        cyc = N_STATE * D
        cand = np.arange(0.0, cyc, 0.001)
        ph = (tn[looks_off][None, :] - cand[:, None]) % cyc
        score = np.sum((ph > g) & (ph < D - g), axis=1)
        best = cand[np.flatnonzero(score == score.max())]
        est = float(np.mean(best)) if (best.max() - best.min()) < D else float(best[0])
        ts = tn - est
    est_state = (np.floor(ts / D).astype(int)) % N_STATE
    pos = ts % D
    keep = (pos >= g) & (pos <= D - g)
    wrong = keep & ((est_state != true_state) | in_transition)
    k = keep & (est_state > 0)
    mis = float(np.sum(wrong & (est_state > 0)) / max(1, np.sum(k)))
    drop = 1.0 - float(np.mean(keep))
    # time until every element has >= 5 kept packets (in whole cycles)
    need = np.full(n_cycles, np.nan)
    cyc_idx = np.floor(ts / (N_STATE * D)).astype(int)
    cnt = np.zeros(N_STATE - 1, dtype=int)
    done = np.nan
    for c in range(n_cycles):
        m = k & (cyc_idx == c)
        if np.any(m):
            cnt += np.bincount(est_state[m] - 1, minlength=N_STATE - 1)
        if np.all(cnt >= 5):
            done = (c + 1) * N_STATE * D
            break
    return mis, drop, done


def main():
    rng = np.random.default_rng(0)
    print("# rate/s  arrival  jitter  mode    D(ms)  g(ms)  wrong%  dropped%  time for 5 pkts/element: median / p99 (s)")
    for burst in (False, True):
        for bad in (False, True):
            for mode in ("host", "marker"):
                for rate in RATES:
                    for D in DWELL:
                        for g in GUARD:
                            if 2 * g >= D:
                                continue
                            r = np.array([run(rate, D, g, mode, bad, burst, rng) for _ in range(60)])
                            done = r[:, 2]
                            ok = np.isfinite(done)
                            med = np.median(done[ok]) if ok.any() else float("nan")
                            p99 = np.percentile(done[ok], 99) if ok.mean() > 0.99 else float("nan")
                            print(f"{rate:5.0f}  {'burst' if burst else 'poisson':7s}  {'bad' if bad else 'good':4s}  {mode:6s}  "
                                  f"{1e3 * D:5.0f}  {1e3 * g:4.0f}  {100 * r[:, 0].mean():6.2f}  {100 * r[:, 1].mean():6.1f}   "
                                  f"{med:6.1f} / {p99:6.1f}")


if __name__ == "__main__" and len(__import__("sys").argv) == 1:
    main()


def adaptive(rate, g, bad, burst, rng, need=5, need_off=3, t_max=120.0):
    """POST-HOC (after the fixed-dwell table): the host moves on as soon as it has seen `need` kept packets of the
    present state (`need_off` for the all-off state).  Association by host time stamps with guard g."""
    t = arrivals(rate, t_max, burst, rng)
    h = t + host_delay(len(t), bad, rng)                 # time at which the host sees the packet
    mean_d = 0.003 if bad else 0.001
    order = np.argsort(h)
    sw_t = [0.0]                                         # switching instants (true time); state k lasts sw_t[k]..sw_t[k+1]
    state = 0
    cnt = 0
    wrong = kept = 0
    for i in order:
        ts = h[i] - mean_d
        if ts < sw_t[-1] + g:                            # too close to (or before) the last switching instant: dropped
            continue
        true_state = int(np.searchsorted(sw_t, t[i], side="right") - 1)
        settled = (t[i] - sw_t[true_state]) >= T_SW
        kept += 1
        if true_state != state or not settled:
            wrong += 1
        cnt += 1
        if cnt >= (need_off if state == 0 else need):
            state += 1
            cnt = 0
            sw_t.append(h[i])                            # the host switches when it sees this packet
            if state == N_STATE:
                return wrong / max(1, kept), h[i]
    return wrong / max(1, kept), float("nan")


def main_adaptive():
    rng = np.random.default_rng(1)
    print("# ADAPTIVE (post-hoc): move on after 5 kept packets (3 in the all-off state)")
    print("# rate/s  arrival  jitter  g(ms)  wrong%   time per pose: median / p99 (s)")
    for burst in (False, True):
        for bad in (False, True):
            for rate in RATES:
                for g in (0.005, 0.010, 0.020):
                    r = np.array([adaptive(rate, g, bad, burst, rng) for _ in range(300)])
                    print(f"{rate:5.0f}  {'burst' if burst else 'poisson':7s}  {'bad' if bad else 'good':4s}  {1e3 * g:4.0f}  "
                          f"{100 * r[:, 0].mean():6.2f}   {np.nanmedian(r[:, 1]):6.2f} / {np.nanpercentile(r[:, 1], 99):6.2f}")


if __name__ == "__main__" and len(__import__("sys").argv) > 1 and __import__("sys").argv[1] == "adaptive":
    main_adaptive()
