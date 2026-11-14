/* DeteCam 3축 화면의 계산 부분 (화면과 무관). 브라우저에서는 window.DC3, node 에서는 module.exports.
   파이썬 원본을 그대로 옮긴 것이다. 고칠 때는 원본과 같이 고치고 console/src/test_core3.js 를 돌린다.
     firmware/core/estimator.py   방향 약속 (frame, to_scan, sphere)
     firmware/core/model.py       판 뒤 신호 표 (Plate.look)
     sim/direction/montecarlo.py  방 모형 (World)
     firmware/core/fusion.py      골 찾기와 방향 합치기 (trough, Fusion, consistency, next_scan)
     firmware/pi/rx3.py           모터 시간 모형 (step, move_to, scan, home)
     firmware/host/scan3.py       재는 순서
   각도는 도. yaw = 받침(J1), pitch = 끄덕임(J2), spin = 판(J3). */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.DC3 = factory();
})(this, function () {
  "use strict";
  var D2R = Math.PI / 180, R2D = 180 / Math.PI;
  var TILT3 = 54.7356;
  // ---- 장치 값 (rx3.py 와 같다) ----
  var CFG = {
    PER_REV: 4076,
    RATIO: { yaw: 4.0, pitch: 11.2, spin: 1.0 },
    LIMIT: { yaw: [-174.0, 174.0], pitch: [-5.0, 85.0] },
    STOP: { yaw: [-175.4, 175.4], pitch: [-5.3, 86.9] },
    STEP_S: { yaw: 0.0020, pitch: 0.0030, spin: 0.0015 },
    FULL_STEP: { yaw: false, pitch: true, spin: false },
    RAMP: { yaw: [80, 2.0], pitch: [80, 2.0], spin: [240, 4.0] },      // 관절마다 (천천히 가는 칸 수, 처음에 몇 배 느리게). 판은 알루미늄 판과 균형추가 무거워 더 길게, 더 느리게 출발한다 (rx3.RAMP)
    BACKLASH: 3.0, SETTLE: 0.4, TILT3: TILT3,
    PLANS: {
      "tilt3": [[-120.0, TILT3], [0.0, TILT3], [120.0, TILT3]],
      "up+tilt3": [[-120.0, 0.0], [-120.0, TILT3], [0.0, TILT3], [120.0, TILT3]],
      "tilt4": [[-135.0, 60.0], [-45.0, 60.0], [45.0, 60.0], [135.0, 60.0]],
      "tilt6": [[-150.0, TILT3], [-90.0, TILT3], [-30.0, TILT3], [30.0, TILT3], [90.0, TILT3], [150.0, TILT3]],
      "up": [[0.0, 0.0]]
    },
    B_LO: -36.0, B_HI: 49.0,      // 판이 직선으로 가리는 범위 (도는 면에서 벗어난 각)
    BIN: 2.0,                     // 곡선을 2도 칸으로 묶는다 (montecarlo.py 와 같다)
    EL_MIN: -80.0                 // 이보다 아래(바로 밑)는 받침에 가려 시뮬레이션에서도 뺐다
  };
  var ENV = {                     // 반사 세기(직접파 대비 dB 범위), 누설 최대값, 초당 패킷, 패킷 흔들림(dB)
    normal: { k: [6.0, 14.0], leak: 0.25, rate: 50.0, pkt: 0.8 },
    bad: { k: [1.0, 8.0], leak: 0.45, rate: 30.0, pkt: 1.2 }
  };

  function mod(a, m) { return ((a % m) + m) % m; }
  function wrap180(a) { return mod(a + 180, 360) - 180; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
  function unit(az, el) { var a = az * D2R, e = el * D2R; return [Math.cos(e) * Math.cos(a), Math.cos(e) * Math.sin(a), Math.sin(e)]; }
  function azel(c) { return [Math.atan2(c[1], c[0]) * R2D, Math.asin(clamp(c[2], -1, 1)) * R2D]; }
  function angle(a, b) { return Math.acos(clamp(dot(a, b), -1, 1)) * R2D; }
  // 스캔 하나의 좌표: a = 판 회전축, ex = 판 각 0도 방향, ey = 90도 방향 (세계 좌표)
  function frame(yaw, tilt) {
    var y = yaw * D2R, t = tilt * D2R;
    var a = [Math.sin(t) * Math.cos(y), Math.sin(t) * Math.sin(y), Math.cos(t)], ex = [-Math.sin(y), Math.cos(y), 0];
    return { a: a, ex: ex, ey: [a[1] * ex[2] - a[2] * ex[1], a[2] * ex[0] - a[0] * ex[2], a[0] * ex[1] - a[1] * ex[0]] };
  }
  // 방향 c 가 그 스캔에서 보이는 [면 안의 각 psi, 면에서 벗어난 각 beta]
  function toScan(c, fr) { return [Math.atan2(dot(c, fr.ey), dot(c, fr.ex)) * R2D, Math.asin(clamp(dot(c, fr.a), -1, 1)) * R2D]; }
  // 그 스캔에서 (psi, beta) 로 보이는 방향을 세계 좌표로
  function fromScan(psi, beta, fr) {
    var p = psi * D2R, b = beta * D2R, cp = Math.cos(b) * Math.cos(p), sp = Math.cos(b) * Math.sin(p), s = Math.sin(b);
    return [cp * fr.ex[0] + sp * fr.ey[0] + s * fr.a[0], cp * fr.ex[1] + sp * fr.ey[1] + s * fr.a[1], cp * fr.ex[2] + sp * fr.ey[2] + s * fr.a[2]];
  }
  function sphere(n, elMin) {
    var out = [], zmin = Math.sin((elMin == null ? -88 : elMin) * D2R), g = Math.PI * (1 + Math.sqrt(5));
    for (var k = 0; k < n; k++) {
      var i = k + 0.5, z = 1 - 2 * i / n, r = Math.sqrt(1 - z * z), th = g * i;
      if (z >= zmin) out.push([r * Math.cos(th), r * Math.sin(th), z]);
    }
    return out;
  }

  // ---- 난수 (같은 씨앗이면 같은 결과) ----
  function rng(seed) {
    var s = seed >>> 0;
    return function () { s += 0x6D2B79F5; var t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  }
  function gauss(r) { var u = 0, v; while (u === 0) u = r(); v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function expo(r) { var u = 0; while (u === 0) u = r(); return -Math.log(u); }

  // ---- 판 뒤 신호 표: F[beta, dphi] 복소수, 거리 세 가지 ----
  function bytes(s) {
    var u, i;
    if (typeof atob === "function") { var b = atob(s); u = new Uint8Array(b.length); for (i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); }
    else { var buf = Buffer.from(s, "base64"); u = new Uint8Array(buf.length); u.set(buf); }
    return u;
  }
  function Table(tb) {
    this.nb = tb.nb; this.np = tb.np; this.db = tb.db; this.dp = tb.dp; this.inv = 1 / tb.scale; this.tabs = {};
    for (var k in tb.tabs) this.tabs[k] = { re: new Int16Array(bytes(tb.tabs[k].re).buffer), im: new Int16Array(bytes(tb.tabs[k].im).buffer) };
  }
  Table.prototype.look = function (key, beta, dphi, out) {
    var T = this.tabs[key], nb = this.nb, np = this.np;
    var fb = (clamp(beta, -90, 90) + 90) / this.db, i0 = clamp(Math.floor(fb), 0, nb - 2), wb = fb - i0;
    var fp = mod(dphi + 180, 360) / this.dp, f0 = Math.floor(fp), j0 = f0 % np, j1 = (j0 + 1) % np, wp = fp - f0;
    var a = i0 * np + j0, b = i0 * np + j1, c = a + np, d = b + np, re = T.re, im = T.im;
    out[0] = ((1 - wb) * ((1 - wp) * re[a] + wp * re[b]) + wb * ((1 - wp) * re[c] + wp * re[d])) * this.inv;
    out[1] = ((1 - wb) * ((1 - wp) * im[a] + wp * im[b]) + wb * ((1 - wp) * im[c] + wp * im[d])) * this.inv;
    return out;
  };

  // ---- 방 모형: 직접파 + 반사 4개 + 누설. montecarlo.World 와 같다 ----
  // 씨앗 하나가 방 하나다. 환경·골 모양·거리를 바꿔도 반사파의 방향과 위상은 그대로 두려고, 난수는 늘 같은 순서로 뽑는다.
  function World(r, c, shape, env, table, dist) {
    var e = ENV[env], n = 4, i, u;
    this.shape = shape; this.env = env; this.rate = e.rate; this.pkt = e.pkt; this.table = table;
    u = r(); this.dist = dist != null ? String(dist) : ["1500", "3000", "5000"][Math.min(2, Math.floor(u * 3))];
    var kdb = e.k[0] + (e.k[1] - e.k[0]) * r(), amp = [], s2 = 0, az = [], el = [];
    for (i = 0; i < n; i++) { u = 0; while (u === 0) u = r(); amp.push(Math.sqrt(-2 * Math.log(u))); s2 += amp[i] * amp[i]; }
    var sc = Math.sqrt(Math.pow(10, -kdb / 10) / s2);
    for (i = 0; i < n; i++) az.push(360 * r());
    for (i = 0; i < n; i++) el.push(clamp(22 * gauss(r), -70, 70));
    u = r(); el[0] = (u < 0.5 ? -1 : 1) * (25 + 40 * r());
    this.kdb = kdb;
    this.rays = [{ re: 1, im: 0, d: c }];
    for (i = 0; i < n; i++) { var ph = 2 * Math.PI * r(), m = amp[i] * sc; this.rays.push({ re: m * Math.cos(ph), im: m * Math.sin(ph), d: unit(az[i], el[i]), az: az[i], el: el[i] }); }
    var lm = e.leak * r(), lp = 2 * Math.PI * r();
    this.leak = [lm * Math.cos(lp), lm * Math.sin(lp)];
    this.back = 0.15 + 0.40 * r();
    this.v = { D: 8 + 5 * r(), sig: 18 + 12 * r(), lo: -38 + 16 * r(), hi: 38 + 14 * r(), w: 5 + 4 * r() };
  }
  World.prototype.field = function (beta, dphi, out) {
    if (this.shape === "W") return this.table.look(this.dist, beta, dphi, out);
    var v = this.v, s = 1 / (1 + Math.exp(-(beta - v.lo) / v.w)) / (1 + Math.exp((beta - v.hi) / v.w)), d = wrap180(dphi) / v.sig;
    out[0] = Math.pow(10, -v.D * s * Math.exp(-d * d) / 20); out[1] = 0; return out;
  };

  // ---- 스캔 하나: 실제 자세 오차와 판 0도 오차는 여기서 정해지고, 추정하는 쪽은 모른다 ----
  function Scan(world, r, yaw, tilt, exact) {
    this.world = world; this.yaw = yaw; this.tilt = tilt; this.fr = frame(yaw, tilt);
    var ty = exact ? yaw : yaw + gauss(r), tt = exact ? tilt : tilt + gauss(r);
    this.zero = exact ? 0 : 2 * gauss(r);
    var tf = frame(ty, tt); this.trueFr = tf;
    this.rays = world.rays.map(function (q) { var s = toScan(q.d, tf); return { re: q.re, im: q.im, psi: s[0], beta: s[1] }; });
    this.cos2 = Math.pow(Math.cos(this.rays[0].beta * D2R), 2);
    this.nb = Math.round(360 / CFG.BIN); this.sum = new Float64Array(this.nb); this.cnt = new Int32Array(this.nb);
    this.pk = []; this.swept = 0; this.done = false; this.obs = null;
  }
  // 판 각 phi 에서의 세기 (dB, 판이 없을 때 직접파 = 0). 잡음은 넣지 않은 값
  Scan.prototype.level = function (phi) {
    var w = this.world, re = 0, im = 0, f = [0, 0];
    for (var k = 0; k < this.rays.length; k++) {
      var q = this.rays[k], dphi = phi + this.zero - q.psi;
      w.field(q.beta, dphi, f);
      if (k === 0) { var x = (mod(dphi, 360) - 180) / 45; f[0] += w.back * Math.exp(-x * x) * this.cos2; }
      re += q.re * f[0] - q.im * f[1]; im += q.re * f[1] + q.im * f[0];
    }
    re += w.leak[0]; im += w.leak[1];
    return 20 * Math.log(Math.max(Math.sqrt(re * re + im * im), 1e-4)) / Math.LN10;
  };
  Scan.prototype.packet = function (phi, r) {
    var p = mod(phi, 360), lv = this.level(p) + this.world.pkt * gauss(r), b = Math.min(this.nb - 1, Math.floor(p / CFG.BIN));
    this.sum[b] += lv; this.cnt[b]++; this.pk.push([p, lv]);
  };
  // 칸마다 평균. 빈 칸은 NaN
  Scan.prototype.curve = function () {
    var phi = new Array(this.nb), lv = new Array(this.nb);
    for (var i = 0; i < this.nb; i++) { phi[i] = (i + 0.5) * CFG.BIN; lv[i] = this.cnt[i] ? this.sum[i] / this.cnt[i] : NaN; }
    return { phi: phi, lv: lv };
  };
  Scan.prototype.filled = function () { var n = 0; for (var i = 0; i < this.nb; i++) if (this.cnt[i]) n++; return n; };

  // ---- 골 찾기 (fusion.trough) ----
  function median(a) {
    var s = Array.prototype.slice.call(a).sort(function (x, y) { return x - y; }), n = s.length;
    return n ? (n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2) : NaN;
  }
  function fillNaN(x) {
    var n = x.length, out = new Array(n), ok = [], i;
    for (i = 0; i < n; i++) if (isFinite(x[i])) ok.push(i);
    if (!ok.length) return null;
    for (i = 0; i < n; i++) {
      if (isFinite(x[i])) { out[i] = x[i]; continue; }
      var dp = 1, dq = 1;
      while (!isFinite(x[mod(i - dp, n)])) dp++;
      while (!isFinite(x[mod(i + dq, n)])) dq++;
      var a = x[mod(i - dp, n)], b = x[mod(i + dq, n)];
      out[i] = a + (b - a) * dp / (dp + dq);
    }
    return out;
  }
  function circSmooth(x, w) {
    w = Math.max(1, Math.floor(w) | 1);
    var n = x.length, h = (w - 1) / 2, out = new Array(n);
    for (var i = 0; i < n; i++) { var s = 0; for (var k = -h; k <= h; k++) s += x[mod(i + k, n)]; out[i] = s / w; }
    return out;
  }
  /* 세기가 꺼진 구간의 가운데 각(-180~180), 깊이(dB), 흔들림(dB).
     돌려주는 값에 그림용으로 s1(고르게 한 곡선), base(가운데값), lo·hi(구간의 양 끝 각)도 넣는다. */
  function trough(phi, lv, gap, maxWidth) {
    gap = gap == null ? 36 : gap; maxWidth = maxWidth == null ? 150 : maxWidth;
    var n = phi.length, step = phi[1] - phi[0], x = fillNaN(lv), i, q;
    if (!x) return { psi: 0, depth: 0, noise: 0, s1: null, base: 0, empty: true };
    var s1 = circSmooth(x, 10 / step), base = median(s1), att = new Array(n), i0 = 0;
    for (i = 0; i < n; i++) { att[i] = base - s1[i]; if (att[i] > att[i0]) i0 = i; }
    var depth = att[i0], thr = clamp(0.3 * depth, 2, 4), low = att.map(function (v) { return v >= thr; });
    var out = { psi: wrap180(phi[i0]), depth: depth, noise: 0, s1: s1, base: base, lo: null, hi: null };
    if (!low[i0]) return out;
    var j = -1;
    for (i = 0; i < n; i++) if (!low[i]) { j = i; break; }
    if (j < 0) return out;
    var runs = [], k = 0;
    while (k < n) {
      if (low[(j + k) % n]) { var a0 = k; while (k < n && low[(j + k) % n]) k++; runs.push([a0, k - a0]); }
      else k++;
    }
    var pos0 = mod(i0 - j, n), m = 0;
    for (i = 0; i < runs.length; i++) if (runs[i][0] <= pos0 && pos0 < runs[i][0] + runs[i][1]) { m = i; break; }
    var lo = runs[m][0], hi = runs[m][0] + runs[m][1], g = Math.round(gap / step), wmax = Math.round(maxWidth / step), grown = true;
    while (grown) {
      grown = false;
      for (i = 0; i < runs.length; i++) {
        var ra = runs[i][0], ln = runs[i][1], dmax = -Infinity;
        for (q = ra; q < ra + ln; q++) dmax = Math.max(dmax, att[(j + q) % n]);
        if (dmax < 0.35 * depth) continue;
        if (ra >= hi && ra - hi <= g && (ra + ln - lo) <= wmax) { hi = ra + ln; grown = true; }
        else if (ra + ln <= lo && lo - (ra + ln) <= g && (hi - ra) <= wmax) { lo = ra; grown = true; }
      }
    }
    var mid = (lo + hi - 1) / 2, centre = phi[0] + mod(j + mid, n) * step;
    var inside = new Array(n);
    for (q = lo; q < hi; q++) inside[(j + q) % n] = true;
    var rest = [];
    for (i = 0; i < n; i++) if (!inside[i]) rest.push(s1[i] - base);
    if (rest.length) { var mr = median(rest); out.noise = 1.4826 * median(rest.map(function (v) { return Math.abs(v - mr); })); }
    out.psi = wrap180(centre);
    out.lo = mod(phi[0] + (j + lo) * step - step / 2, 360); out.hi = mod(phi[0] + (j + hi - 1) * step + step / 2, 360);
    return out;
  }
  function pSeen(beta) { return 1 / (1 + Math.exp(-(beta - CFG.B_LO - 4) / 5)) / (1 + Math.exp((beta - CFG.B_HI + 4) / 5)); }

  // ---- 여러 스캔을 합쳐 방향 구하기 (fusion.Fusion) ----
  function Fusion(nGrid, minDepth, kNoise) {
    this.G = sphere(nGrid || 8000); this.minDepth = minDepth == null ? 4.5 : minDepth; this.kNoise = kNoise == null ? 3.0 : kNoise;
  }
  Fusion.prototype.readOne = function (phi, lv) {
    var t = trough(phi, lv);
    t.seen = !t.empty && t.depth >= Math.max(this.minDepth, this.kNoise * t.noise);
    return t;
  };
  Fusion.prototype.read = function (curves) { var self = this; return curves.map(function (c) { return self.readOne(c.phi, c.lv); }); };
  function prep(obs, frames) {
    return obs.map(function (o, i) {
      var f = frame(frames[i][0], frames[i][1]);
      return { a: f.a, ex: f.ex, ey: f.ey, seen: o.seen, psi: o.psi, sig: 6 * Math.sqrt(8 / Math.max(o.depth, 4)) };
    });
  }
  function score(c, P) {
    var s = 0;
    for (var k = 0; k < P.length; k++) {
      var p = P[k], ps = pSeen(Math.asin(clamp(dot(c, p.a), -1, 1)) * R2D);
      if (p.seen) { var d = wrap180(Math.atan2(dot(c, p.ey), dot(c, p.ex)) * R2D - p.psi) / p.sig; s += -0.5 * d * d + Math.log(ps + 1e-6); }
      else s += Math.log(1 - 0.95 * ps);
    }
    return s;
  }
  Fusion.prototype.solveObs = function (obs, frames) {
    var nSeen = 0, i, j, G = this.G;
    obs.forEach(function (o) { if (o.seen) nSeen++; });
    if (!nSeen) return null;
    var P = prep(obs, frames), best = -Infinity, c0 = null, s;
    for (i = 0; i < G.length; i++) { s = score(G[i], P); if (s > best) { best = s; c0 = G[i]; } }
    [[6.0, 1.0], [1.5, 0.25]].forEach(function (q) {
      var sp = q[0], st = q[1], ae = azel(c0), d = [], v, cf = Math.max(Math.cos(ae[1] * D2R), 0.15), b2 = -Infinity, c1 = c0;
      for (v = -sp; v <= sp + 1e-9; v += st) d.push(v);
      for (i = 0; i < d.length; i++) for (j = 0; j < d.length; j++) {
        var c = unit(ae[0] + d[j] / cf, clamp(ae[1] + d[i], -89.5, 89.5)); s = score(c, P);
        if (s > b2) { b2 = s; c1 = c; }
      }
      c0 = c1;
    });
    return c0;
  };
  Fusion.prototype.solve = function (curves, frames) { var obs = this.read(curves); return { est: this.solveObs(obs, frames), obs: obs }; };
  // 가장 좋은 방향 둘레에서 점수가 drop 안쪽인 후보들이 퍼진 각. 크면 한 스캔으로만 정해져 불확실하다는 뜻
  Fusion.prototype.spread = function (obs, frames, c, drop) {
    var P = prep(obs, frames), G = this.G, sc = new Array(G.length), mx = -Infinity, i, out = 0;
    for (i = 0; i < G.length; i++) { sc[i] = score(G[i], P); if (sc[i] > mx) mx = sc[i]; }
    for (i = 0; i < G.length; i++) if (sc[i] >= mx - (drop == null ? 3 : drop)) out = Math.max(out, angle(G[i], c));
    return out;
  };
  // 찾은 방향이 스캔 하나하나와 맞는지
  function consistency(est, obs, frames, tol, margin) {
    tol = tol == null ? 15 : tol; margin = margin == null ? 8 : margin;
    var out = [], ok = true;
    obs.forEach(function (o, i) {
      var s = toScan(est, frame(frames[i][0], frames[i][1])), psi = s[0], beta = s[1], d;
      if (o.seen) {
        var dd = Math.abs(wrap180(o.psi - psi));
        d = { seen: true, resid: dd, beta: beta, ok: dd <= tol && (CFG.B_LO - margin <= beta && beta <= CFG.B_HI + margin) };
      } else d = { seen: false, resid: null, beta: beta, ok: !(CFG.B_LO + margin < beta && beta < CFG.B_HI - margin) };
      out.push(d); ok = ok && d.ok;
    });
    return { ok: ok, detail: out };
  }
  // 골이 한 스캔에서만 보였을 때, 그 반쪽 평면을 가로질러 재는 다음 스캔의 받침 각
  function nextScan(obs, frames, tilt) {
    var k = 0, bd = -Infinity, i;
    for (i = 0; i < obs.length; i++) { var v = obs[i].seen ? obs[i].depth : -1; if (v > bd) { bd = v; k = i; } }
    var c = fromScan(obs[k].psi, 8.0, frame(frames[k][0], frames[k][1])), ae = azel(c);
    var x = -Math.tan(ae[1] * D2R) / Math.tan(tilt * D2R), d = Math.acos(clamp(x, -1, 1)) * R2D;
    var cands = [wrap180(ae[0] + d), wrap180(ae[0] - d)];
    function far(y) { var m = Infinity; frames.forEach(function (f) { m = Math.min(m, Math.abs(wrap180(y - f[0]))); }); return m; }
    return [far(cands[1]) > far(cands[0]) ? cands[1] : cands[0], tilt];
  }
  // 그 방향이 계획의 스캔 몇 개에서 골로 보일 자리인가 (판이 가리는 범위 안)
  function coverage(c, frames) {
    var n = 0;
    frames.forEach(function (f) { var b = toScan(c, frame(f[0], f[1]))[1]; if (b > CFG.B_LO && b < CFG.B_HI) n++; });
    return n;
  }

  // ---- 모터 시간 모형 (rx3.Rig.step) ----
  function perDeg(j) { return CFG.PER_REV * CFG.RATIO[j] / 360; }
  function cumTimes(count, base, ramp, j) {
    var c = new Float64Array(count), acc = 0, rp = CFG.RAMP[j || "yaw"], R = rp[0], slow = rp[1];
    for (var i = 0; i < count; i++) {
      var dt = base;
      if (ramp) { var e = Math.min(i, count - 1 - i); if (e < R) dt = base * (slow - (slow - 1) * e / R); }
      acc += dt; c[i] = acc;
    }
    return c;
  }
  // 시작한 뒤 tau 초에 지나간 칸 수 (소수 포함)
  function stepsAt(cum, tau) {
    var n = cum.length;
    if (!n || tau <= 0) return 0;
    if (tau >= cum[n - 1]) return n;
    var lo = 0, hi = n - 1;                              // cum[hi] > tau 인 가장 작은 hi
    while (lo < hi) { var mid = (lo + hi) >> 1; if (cum[mid] > tau) hi = mid; else lo = mid + 1; }
    var t0 = hi ? cum[hi - 1] : 0;
    return hi + (tau - t0) / (cum[hi] - t0);
  }

  // ---- 장치 시뮬레이터 ----
  var PLAN_NAME = { "tilt3": "기울여 3번", "up+tilt3": "세워서 1번 + 기울여 3번", "tilt4": "60° 기울여 4번", "tilt6": "기울여 6번", "up": "세워서 1번" };
  function Sim(table, fusion) {
    this.table = table; this.fus = fusion || new Fusion();
    this.pos = { yaw: 0, pitch: 0, spin: 0 }; this.plan_ = { yaw: 0, pitch: 0, spin: 0 };   // 반 스텝 수 (지금 / 줄 선 동작이 끝난 뒤)
    this.t = 0; this.mode = "idle"; this.label = ""; this.homed = true;
    this.queue = []; this.script = []; this.cur = null;
    this.cam = { az: 35, el: -42 }; this.dist = "3000"; this.env = "normal"; this.shape = "V"; this.plan = "tilt3";
    this.roomSeed = 1; this.runSeed = 0; this.noExtra = false; this.exact = false;
    this.scans = []; this.result = null; this.world = null; this.r = rng(1); this.task = null; this.blocked = null;
    this.on = function () {};
  }
  Sim.prototype.deg = function (j) { return this.pos[j] / perDeg(j); };
  Sim.prototype.busy = function () { return !!this.cur || this.queue.length > 0 || this.script.length > 0; };
  Sim.prototype._mode = function (m, label) { if (this.mode !== m || this.label !== (label || "")) { this.mode = m; this.label = label || ""; this.on("mode", m); } };
  Sim.prototype.log = function (text) { this.on("log", { t: this.t, text: text }); };
  Sim.prototype.makeWorld = function () {
    return new World(rng(this.roomSeed * 7919 + 13), unit(this.cam.az, this.cam.el), this.shape, this.env, this.table, this.dist);
  };
  Sim.prototype.stop = function (why) {
    var was = this.busy(), op = this.cur;
    if (op && op.k === "step") {                                 // 돌던 모터는 지금 칸에서 선다
      var k = Math.floor(stepsAt(op.cum, op.tau - 0.02)), p = op.pos0 + op.sgn * op.unit * k;
      this.pos[op.j] = op.floor != null ? Math.max(op.floor, p) : p;
    }
    this.cur = null; this.queue = []; this.script = []; this.task = null; this.blocked = null;
    this.plan_ = { yaw: this.pos.yaw, pitch: this.pos.pitch, spin: this.pos.spin };
    if (why !== "quiet") { this._mode("stopped"); if (was) this.log("정지"); }
  };
  // 반 스텝 n 칸 (부호 = 관절 각이 커지는 쪽이 +)
  Sim.prototype._step = function (j, n, opt) {
    opt = opt || {};
    if (!n) return;
    var unitN = CFG.FULL_STEP[j] ? 2 : 1, count = Math.floor(Math.abs(n) / unitN), sgn = n > 0 ? 1 : -1;
    var cum = cumTimes(count, CFG.STEP_S[j] * unitN, opt.ramp !== false, j);
    this.queue.push({ k: "step", j: j, sgn: sgn, unit: unitN, count: count, cum: cum, T: 0.02 + (count ? cum[count - 1] : 0) + 0.05,
                      mode: opt.mode || "move", label: opt.label, floor: opt.floor, scan: opt.scan || null });
    this.plan_[j] += sgn * count * unitN;
  };
  Sim.prototype._wait = function (s, mode, label) { this.queue.push({ k: "wait", T: s, mode: mode || "move", label: label }); };
  Sim.prototype._moveTo = function (j, target, opt) {
    opt = opt || {};
    if (CFG.LIMIT[j] && opt.check !== false) target = clamp(target, CFG.LIMIT[j][0], CFG.LIMIT[j][1]);
    var pd = perDeg(j), n = Math.round(target * pd) - this.plan_[j];
    if (j === "yaw" && n < 0 && CFG.BACKLASH > 0) {              // - 방향으로 갈 때는 조금 지나쳤다가 + 로 돌아온다
      var over = Math.round(CFG.BACKLASH * pd), loSteps = Math.round(CFG.LIMIT.yaw[0] * pd);
      over = Math.max(0, Math.min(over, this.plan_[j] + n - loSteps));
      this._step(j, n - over, opt); this._step(j, over, opt);
    } else this._step(j, n, opt);
  };
  Sim.prototype._goto = function (yaw, pitch, opt) { if (yaw != null) this._moveTo("yaw", yaw, opt); if (pitch != null) this._moveTo("pitch", pitch, opt); };
  Sim.prototype.goto = function (yaw, pitch) {
    var self = this;
    this.stop("quiet"); this.task = "goto";
    this.script = [function () { self._goto(yaw, pitch, { mode: "move" }); }, function () { self.task = null; self._mode("idle"); }];
  };
  // 관절 하나를 조금 움직인다 (rx3 의 jog). 범위 밖으로는 가지 않는다
  Sim.prototype.jog = function (j, deg) {
    var self = this;
    if (this.task !== "jog") { this.stop("quiet"); this.task = "jog"; }
    this.script = this.script.filter(function (f) { return !f.tail; });
    this.script.push(function () {
      var pd = perDeg(j), want = Math.round(deg * pd), n = want;
      if (CFG.LIMIT[j]) n = clamp(self.plan_[j] + n, Math.round(CFG.LIMIT[j][0] * pd), Math.round(CFG.LIMIT[j][1] * pd)) - self.plan_[j];
      self.blocked = n !== want ? j : null;                      // 범위 끝에 걸렸다
      self._step(j, n, { mode: "jog" });
    });
    var tail = function () { self.task = null; self._mode("idle"); };
    tail.tail = true; this.script.push(tail);
  };
  // 원점 잡기 (rx3.Rig.home): 끄덕임과 받침을 - 쪽 멈춤턱까지 넉넉히 밀었다가 0도로 돌아온다
  Sim.prototype.home = function () {
    var self = this;
    this.stop("quiet"); this.task = "home"; this.homed = false; this.log("원점 잡기 시작");
    this.script = [];
    [["pitch", 8.0, "끄덕임"], ["yaw", 10.0, "받침"]].forEach(function (q) {
      var j = q[0], pd = perDeg(j), lo = CFG.STOP[j][0], hi = CFG.STOP[j][1];
      self.script.push(function () {
        var travel = (hi - lo) + q[1], fl = Math.round(lo * pd);
        self.log(q[2] + ": − 쪽 멈춤턱까지 " + travel.toFixed(0) + "° 만큼 민다");
        self._step(j, -Math.round(travel * pd), { mode: "home", label: q[2] + "을 멈춤턱까지 미는 중", floor: fl });
      });
      self.script.push(function () {
        self.pos[j] = self.plan_[j] = Math.round(lo * pd);
        self.log(q[2] + ": 멈춤턱 " + fmt1(lo) + "에 닿음. 0°로 돌아간다");
        self._moveTo(j, 0.0, { check: false, mode: "home", label: q[2] + "을 0°로" });
      });
    });
    this.script.push(function () { self.homed = true; self.task = null; self.log("원점 잡기 끝. 받침 0°, 끄덕임 0°"); self._mode("idle"); });
  };
  // 자세를 잡고 판을 한 바퀴 (rx3.Rig.scan)
  Sim.prototype._scan = function (yaw, pitch, i, extra) {
    var self = this, name = "스캔 " + (i + 1);
    this.script.unshift(
      function () {
        self._goto(yaw, pitch, { mode: "move", label: name + ": 자세 잡는 중" });
        self._wait(CFG.SETTLE, "move", name + ": 자세 잡는 중");
      },
      function () {
        var sc = new Scan(self.world, self.r, self.deg("yaw"), self.deg("pitch"), self.exact);
        sc.i = i; sc.extra = !!extra; sc.spin0 = mod(self.deg("spin"), 360); sc.t0 = self.t;
        self.scans.push(sc); self.on("scan", sc);
        self.log(name + ": 받침 " + fmt1(sc.yaw) + ", 끄덕임 " + fmt1(sc.tilt) + "에서 판 한 바퀴");
        self._step("spin", CFG.PER_REV, { mode: "scan", label: name + ": 판 돌리는 중", scan: sc });
      },
      function () {
        var sc = self.scans[self.scans.length - 1], c = sc.curve();
        sc.done = true; sc.swept = 360; sc.t1 = self.t;
        sc.obs = self.fus.readOne(c.phi, c.lv);
        self.log(name + ": 패킷 " + sc.pk.length + "개. " + (sc.obs.seen ? "골 보임 (가운데 " + mod(sc.obs.psi, 360).toFixed(0) + "°, 깊이 " + sc.obs.depth.toFixed(1) + " dB)" : "골 안 보임 (가장 낮은 곳 " + sc.obs.depth.toFixed(1) + " dB)"));
        self.on("scanDone", sc);
      });
  };
  // 스캔들을 합쳐 방향을 구한다 (analyze3.solve)
  Sim.prototype._solve = function () {
    var self = this, S = this.scans, frames = S.map(function (s) { return [s.yaw, s.tilt]; });
    var usable = [];
    S.forEach(function (s, i) { if (s.filled() >= Math.floor(s.nb / 2)) usable.push(i); });
    var res = { frames: frames, scans: S.map(function (s) { return { yaw: s.yaw, pitch: s.tilt, packets: s.pk.length, seen: false, depth: null, centre: null, noise: null }; }),
                nSeen: 0, az: null, el: null, est: null, truth: { az: this.truthCam.az, el: this.truthCam.el }, single: S.length === 1, plan: this.plan };
    if (!usable.length) return res;
    var obs = usable.map(function (i) { return S[i].obs; }), fr = usable.map(function (i) { return frames[i]; });
    var est = this.fus.solveObs(obs, fr);
    usable.forEach(function (i, q) {
      var o = obs[q], d = res.scans[i];
      d.seen = o.seen; d.depth = o.depth; d.centre = mod(o.psi, 360); d.noise = o.noise; if (o.seen) res.nSeen++;
    });
    res.obs = obs; res.usable = usable;
    if (est) {
      var ae = azel(est), ct = unit(res.truth.az, res.truth.el);
      res.est = est; res.az = ae[0]; res.el = ae[1];
      res.spread = this.fus.spread(obs, fr, est);
      var cs = consistency(est, obs, fr);
      usable.forEach(function (i, q) { var d = cs.detail[q]; res.scans[i].fits = d.ok; res.scans[i].beta = d.beta; res.scans[i].resid = d.resid; });
      res.consistent = cs.ok && res.nSeen >= 2;
      res.err = angle(est, ct);
      res.errFlat = angle(unit(ae[0], 0), ct);                   // 한 번만 재면 높이를 알 수 없다 -> 수평이라고 볼 때의 오차
    }
    return res;
  };
  // 재는 순서 (scan3.py): 계획대로 재고, 골이 한 스캔에서만 보였으면 그 방향을 가로질러 한 번 더 잰다
  Sim.prototype.measure = function () {
    var self = this, frames = CFG.PLANS[this.plan].slice(), t0 = 0, yaw0, flip;
    this.stop("quiet"); this.task = "measure"; this.scans = []; this.result = null; this.runSeed += 1;
    yaw0 = this.deg("yaw");                                       // 받침이 가까운 끝부터 잰다 (scan3.order)
    flip = this.plan !== "up+tilt3" && Math.abs(yaw0 - frames[frames.length - 1][0]) < Math.abs(yaw0 - frames[0][0]);
    if (flip) frames.reverse();
    this.truthCam = { az: this.cam.az, el: this.cam.el };
    this.world = this.makeWorld(); this.r = rng(this.roomSeed * 104729 + this.runSeed * 31 + 7);
    this.on("clear");
    function finish(res) {
      res.total = self.t - t0; self.result = res;
      if (res.az == null) self.log("결과: 어느 스캔에서도 골이 보이지 않았다");
      else self.log("결과: 방위 " + fmt0(res.az) + ", 높이 " + fmt0(res.el) + " · " + (res.nSeen >= 2 ? "두 스캔 이상에서 보임" : "한 스캔에서만 보임") + " · 걸린 시간 " + res.total.toFixed(1) + "초");
      self.task = null; self.on("result", res); self._mode("done");
    }
    this.script = [function () { t0 = self.t; self.log("재기 시작: " + PLAN_NAME[self.plan] + (flip ? " (받침이 가까운 쪽 " + fmt0(frames[0][0]) + "부터)" : "")); }];
    frames.forEach(function (f, i) { self.script.push(function () { self._scan(f[0], f[1], i); }); });
    this.script.push(function () {
      var res = self._solve();
      if (!self.noExtra && self.plan !== "up" && res.nSeen === 1 && res.obs) {
        var fr = res.usable.map(function (i) { return res.frames[i]; }), nx = nextScan(res.obs, fr, TILT3), yaw = clamp(nx[0], -174, 174);
        self.log("한 스캔에서만 골이 보였다. 받침 " + fmt0(yaw) + "에서 한 번 더 잰다");
        self.result = res; res.partial = true; self.on("partial", res);
        self.script.unshift(function () { self._scan(yaw, nx[1], self.scans.length, true); }, function () { finish(self._solve()); });
      } else finish(res);
    });
  };
  // 시간 dt(초)만큼 진행
  Sim.prototype.advance = function (dt) {
    var guard = 0;
    while (dt > 1e-12 && guard++ < 100000) {
      if (!this.cur) {
        if (this.queue.length) {
          var op = this.queue.shift(); op.tau = 0;
          if (op.k === "step") { op.pos0 = this.pos[op.j]; if (op.scan) op.next = 0.02 + expo(this.r) / this.scan_rate(op.scan); }
          this.cur = op; this._mode(op.mode, op.label);
        } else if (this.script.length) { this.script.shift().call(this); continue; }
        else break;
      }
      var c = this.cur, h = Math.min(dt, c.T - c.tau);
      c.tau += h; this.t += h; dt -= h;
      if (c.k === "step") {
        var k = stepsAt(c.cum, c.tau - 0.02), p = c.pos0 + c.sgn * c.unit * k;
        this.pos[c.j] = c.floor != null ? Math.max(c.floor, p) : p;
        if (c.scan) {                                            // 판이 도는 동안 패킷이 띄엄띄엄 들어온다
          var sc = c.scan, end = 0.02 + (c.count ? c.cum[c.count - 1] : 0), rate = this.scan_rate(sc), n = 0;
          while (c.next <= c.tau && c.next <= end) {
            sc.packet(sc.spin0 + 360 * stepsAt(c.cum, c.next - 0.02) / CFG.PER_REV, this.r);
            c.next += expo(this.r) / rate; n++;
          }
          sc.swept = 360 * k / CFG.PER_REV;
          if (n) this.on("packet", sc);
        }
      }
      if (c.tau >= c.T - 1e-12) {
        if (c.k === "step") this.pos[c.j] = c.floor != null ? Math.max(c.floor, c.pos0 + c.sgn * c.unit * c.count) : c.pos0 + c.sgn * c.unit * c.count;
        this.cur = null;
      }
    }
  };
  Sim.prototype.scan_rate = function (sc) { return sc.world.rate; };
  // 끝날 때까지 한꺼번에 돌린다 (처음 화면의 예시, 시험)
  Sim.prototype.finish = function () { var n = 0; while (this.busy() && n++ < 100000) this.advance(5); };

  function fmt0(v) { var r = Math.round(v); return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r) + "°"; }
  function fmt1(v) { var r = Math.round(v * 10) / 10; return (r > 0 ? "+" : r < 0 ? "−" : "") + Math.abs(r).toFixed(1) + "°"; }

  return { CFG: CFG, ENV: ENV, PLAN_NAME: PLAN_NAME, TILT3: TILT3, mod: mod, wrap180: wrap180, clamp: clamp, unit: unit, azel: azel, angle: angle, frame: frame, toScan: toScan, fromScan: fromScan,
           sphere: sphere, rng: rng, gauss: gauss, Table: Table, World: World, Scan: Scan, median: median, fillNaN: fillNaN, circSmooth: circSmooth, trough: trough, pSeen: pSeen,
           Fusion: Fusion, consistency: consistency, nextScan: nextScan, coverage: coverage, perDeg: perDeg, cumTimes: cumTimes, stepsAt: stepsAt, Sim: Sim, fmt0: fmt0, fmt1: fmt1 };
});
