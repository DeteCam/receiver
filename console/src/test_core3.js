/* core3.js 가 파이썬 원본과 같은 값을 내는지 본다.
     node test_core3.js            값 비교 (testvec.json 은 make_testvec.py 가 만든다)
     node test_core3.js mc [횟수]   화면이 쓰는 경로 그대로 여러 방향을 돌려 파이썬 몬테카를로 요약과 견준다 */
"use strict";
const fs = require("fs"), path = require("path");
const DC = require("./core3.js");
const DATA = JSON.parse(fs.readFileSync(path.join(__dirname, "data.json"), "utf8"));
const table = new DC.Table(DATA.table), fus = new DC.Fusion();
let bad = 0;
function check(name, ok, info) { if (!ok) bad++; console.log((ok ? "  ok  " : "  FAIL ") + name + (info ? "  " + info : "")); }
const mx = (a) => a.reduce((m, v) => Math.max(m, v), 0);

function values() {
  const V = JSON.parse(fs.readFileSync(path.join(__dirname, "testvec.json"), "utf8"));
  // 1) 표
  let e = [];
  V.look.forEach((L) => L.beta.forEach((b, i) => { const o = table.look(L.key, b, L.dphi[i], [0, 0]); e.push(Math.abs(o[0] - L.re[i]), Math.abs(o[1] - L.im[i])); }));
  check("표 꺼내기 " + e.length / 2 + "곳", mx(e) < 1e-4, "가장 큰 차이 " + mx(e).toExponential(2));
  // 2) 좌표
  const G = fus.G, S = V.sphere, d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
  check("구 격자", G.length === S.n && d3(G[0], S.first) < 1e-12 && d3(G[G.length - 1], S.last) < 1e-12 && d3(G[4000], S.mid) < 1e-12, "점 " + G.length + "개");
  e = [];
  V.frames.forEach((f) => { const fr = DC.frame(f.yaw, f.tilt), s = DC.toScan(f.c, fr); e.push(d3(fr.a, f.a), d3(fr.ex, f.ex), d3(fr.ey, f.ey), Math.abs(DC.wrap180(s[0] - f.psi)), Math.abs(s[1] - f.beta));
    e.push(d3(DC.fromScan(s[0], s[1], fr), f.c)); });
  check("스캔 좌표", mx(e) < 1e-9, mx(e).toExponential(2));
  e = V.p_seen.beta.map((b, i) => Math.abs(DC.pSeen(b) - V.p_seen.p[i]));
  check("골이 보일 가능성", mx(e) < 1e-12);
  // 3) 잡음 없는 곡선
  e = [];
  V.levels.forEach((c) => {
    const w = { shape: c.shape, table: table, dist: c.dist, rays: c.rays, leak: c.leak, back: c.back, v: c.v, pkt: 0, rate: 50, field: DC.World.prototype.field };
    const sc = new DC.Scan(w, null, c.yaw, c.tilt, true); sc.zero = c.zero;
    c.phi.forEach((p, i) => e.push(Math.abs(sc.level(p) - c.lv[i])));
  });
  check("세기 곡선 " + V.levels.length + "개", mx(e) < 0.02, "가장 큰 차이 " + mx(e).toFixed(4) + " dB");
  // 4) 골 찾기, 방향 합치기
  let et = [], ea = [], es = [], flags = 0, nx = [], nEst = 0, nNext = 0;
  V.cases.forEach((c) => {
    const curves = c.lv.map((lv) => ({ phi: c.phi, lv: lv.map((v) => (v == null ? NaN : v)) }));
    const r = fus.solve(curves, c.frames);
    r.obs.forEach((o, i) => { const p = c.obs[i]; et.push(Math.abs(DC.wrap180(o.psi - p.psi)), Math.abs(o.depth - p.depth), Math.abs(o.noise - p.noise)); if (o.seen !== p.seen) flags++; });
    if ((r.est == null) !== (c.est == null)) { flags++; return; }
    if (!r.est) return;
    nEst++;
    ea.push(DC.angle(r.est, c.est)); es.push(Math.abs(fus.spread(r.obs, c.frames, r.est) - c.spread));
    const cs = DC.consistency(c.est, r.obs, c.frames);
    if (cs.ok !== c.cons_ok) flags++;
    cs.detail.forEach((d, i) => { if (d.ok !== c.cons[i].ok || Math.abs(d.beta - c.cons[i].beta) > 1e-6 || (d.resid == null) !== (c.cons[i].resid == null) || (d.resid != null && Math.abs(d.resid - c.cons[i].resid) > 1e-6)) flags++; });
    if (c.next) { nNext++; const n = DC.nextScan(r.obs, c.frames, DC.TILT3); nx.push(Math.abs(DC.wrap180(n[0] - c.next[0])), Math.abs(n[1] - c.next[1])); }
  });
  check("골 가운데·깊이·흔들림 " + et.length / 3 + "개", mx(et) < 1e-6, "가장 큰 차이 " + mx(et).toExponential(2));
  check("찾은 방향 " + nEst + "개", mx(ea) < 0.3, "가장 큰 차이 " + mx(ea).toFixed(3) + "°");
  check("퍼짐", mx(es) < 0.5, "가장 큰 차이 " + mx(es).toFixed(3) + "°");
  check("보임/맞음 판정", flags === 0, "다른 것 " + flags + "개");
  check("한 번 더 잴 받침 각 " + nNext + "개", mx(nx) < 1e-6);
  e = [];
  V.trough5.forEach((c) => { const t = DC.trough(c.phi, c.lv.map((v) => (v == null ? NaN : v))); e.push(Math.abs(DC.wrap180(t.psi - c.psi)), Math.abs(t.depth - c.depth), Math.abs(t.noise - c.noise)); });
  check("5도 칸 곡선의 골", mx(e) < 1e-6, mx(e).toExponential(2));
  // 5) 모터 시간
  const T = V.timing;
  ["tilt3", "up+tilt3", "tilt4", "tilt6", "up"].forEach((p) => {
    const sim = new DC.Sim(table, fus); sim.plan = p; sim.noExtra = true; sim.measure(); sim.finish();
    const e2 = [];
    sim.scans.forEach((s, i) => { const r = T[p][i]; e2.push(Math.abs(s.t1 - r.t_end), Math.abs(s.yaw - r.yaw), Math.abs(s.tilt - r.pitch), Math.abs(s.t0 + 0.02 - r.t0), Math.abs(s.spin0 - r.spin0)); });
    check("시간 " + p, sim.scans.length === T[p].length && mx(e2) < 2e-3, "끝 " + sim.t.toFixed(2) + "초 (파이썬 " + T[p][T[p].length - 1].t_end.toFixed(2) + "), 가장 큰 차이 " + mx(e2).toExponential(1));
  });
  let sim = new DC.Sim(table, fus); sim.home(); sim.finish();
  check("원점 잡기 시간", Math.abs(sim.t - T.home) < 1e-6 && sim.pos.yaw === 0 && sim.pos.pitch === 0, sim.t.toFixed(2) + "초");
  sim = new DC.Sim(table, fus); sim.noExtra = true; sim.measure(); sim.finish(); let t0 = sim.t; sim.measure(); sim.finish();
  check("이어서 한 번 더 (가까운 쪽부터)", Math.abs(sim.t - t0 - T.tilt3_again) < 1e-6 && Math.abs(sim.scans[0].yaw - T.tilt3_again_first) < 0.05, (sim.t - t0).toFixed(2) + "초, 첫 스캔 받침 " + sim.scans[0].yaw.toFixed(0) + "°");
  const cum = DC.cumTimes(4076, 0.0015, true, "spin");
  e = T.spin.k.map((k, i) => Math.abs((k ? cum[k - 1] : 0) + 0.02 - T.spin.t[i]));
  e.push(Math.abs(DC.stepsAt(cum, cum[999]) - 1000), Math.abs(DC.stepsAt(cum, (cum[999] + cum[1000]) / 2) - 1000.5), DC.stepsAt(cum, -1), Math.abs(DC.stepsAt(cum, 1e9) - 4076));
  check("판 한 바퀴 시간표", mx(e) < 2e-6, "한 바퀴 " + (cum[4075] + 0.07).toFixed(3) + "초");
  e = [];
  T.moves.forEach((m) => {
    const s = new DC.Sim(table, fus), pd = DC.perDeg(m.j); s.pos[m.j] = s.plan_[m.j] = Math.round(m.a * pd);
    s.script = [function () { s._moveTo(m.j, m.b); }]; s.finish();
    e.push(Math.abs(s.t - m.t), Math.abs(s.pos[m.j] - m.pos));
  });
  check("관절 하나 움직이기 " + T.moves.length + "가지", mx(e) < 1e-6, T.moves.map((m) => m.t.toFixed(1)).join(" "));
  // 6) 멈춤, 손으로 움직이기
  sim = new DC.Sim(table, fus); sim.measure(); sim.advance(30); sim.stop();
  check("도중에 멈춤", !sim.busy() && Number.isInteger(sim.pos.yaw) && Number.isInteger(sim.pos.pitch) && sim.mode === "stopped", "받침 " + sim.deg("yaw").toFixed(2) + "°, 끄덕임 " + sim.deg("pitch").toFixed(2) + "°");
  sim.jog("pitch", 200); sim.finish(); const top = sim.deg("pitch"); sim.jog("pitch", -200); sim.finish();
  check("손으로 움직이기는 범위 안에서만", Math.abs(top - 85) < 0.02 && Math.abs(sim.deg("pitch") + 5) < 0.02 && sim.mode === "idle", top.toFixed(2) + " / " + sim.deg("pitch").toFixed(2));
  console.log(bad ? "\n실패 " + bad + "개" : "\n모두 통과");
  process.exit(bad ? 1 : 0);
}

function monte(n) {
  const R = DC.rng(424242), row = (k) => DATA.mc.rows.find((r) => r.name.indexOf(k) === 0);
  const ref = { "tilt3": row("기울여 3번"), "tilt4": row("60° 기울여 4번"), "up": row("세워서 1번 (논문") };
  const cams = [];
  while (cams.length < n) { const v = [DC.gauss(R), DC.gauss(R), DC.gauss(R)], m = Math.hypot(v[0], v[1], v[2]); if (v[2] / m > Math.sin(-80 * Math.PI / 180)) cams.push(DC.azel([v[0] / m, v[1] / m, v[2] / m])); }
  const sim = new DC.Sim(table, fus);
  console.log("화면과 같은 경로로 " + n + "방향. 괄호는 파이썬 몬테카를로(1500방향) 값");
  [["V", "normal", "v"], ["W", "normal", "w"], ["V", "bad", "vb"], ["W", "bad", "wb"]].forEach((q) => {
    ["tilt3", "tilt4", "up"].forEach((plan) => {
      const t0 = Date.now(), err = []; let scans = 0, cons = 0, consBad = 0, flagged = 0, flaggedBad = 0;
      sim.shape = q[0]; sim.env = q[1]; sim.plan = plan; sim.dist = null; sim.noExtra = plan !== "tilt3";
      cams.forEach((c, i) => {
        sim.cam = { az: c[0], el: c[1] }; sim.roomSeed = 1000 + i; sim.runSeed = 0; sim.measure(); sim.finish();
        const r = sim.result, e = r.az == null ? 180 : plan === "up" ? r.errFlat : r.err;
        err.push(e); scans += sim.scans.length;
        if (r.az != null && r.consistent) { cons++; if (e > 30) consBad++; } else { flagged++; if (e > 30) flaggedBad++; }
      });
      const s = err.slice().sort((a, b) => a - b), med = (s[(n >> 1) - 1 + (n % 2)] + s[n >> 1]) / 2, over = err.filter((v) => v > 30).length / n * 100, p = ref[plan][q[2]];
      console.log("  [" + q[0] + "·" + (q[1] === "normal" ? "보통" : "나쁨") + "] " + (DC.PLAN_NAME[plan] + "          ").slice(0, 12) + " 가운데값 " + med.toFixed(1) + "° (" + p.med + ")  30° 넘음 " + over.toFixed(1) + "% (" + p.over30 + ")  평균 스캔 " +
        (scans / n).toFixed(2) + " (" + p.scans + ")" + (plan === "tilt3" ? "  | 점검 통과 " + (cons / n * 100).toFixed(0) + "% 중 틀린 것 " + (cons ? consBad / cons * 100 : 0).toFixed(1) + "%, 경고 " + (flagged / n * 100).toFixed(0) + "% 중 틀린 것 " + (flagged ? flaggedBad / flagged * 100 : 0).toFixed(0) + "%" : "") + "  " + ((Date.now() - t0) / 1000).toFixed(1) + "초");
    });
  });
  console.log("점검 참고(파이썬, 기울여 3번): V 통과한 것 중 틀린 것 " + DATA.mc.cons.v.fail_when_pass + "%, 경고 " + DATA.mc.cons.v.flagged + "% · W " + DATA.mc.cons.w.fail_when_pass + "%, 경고 " + DATA.mc.cons.w.flagged + "%");
}

if (process.argv[2] === "mc") monte(+process.argv[3] || 1500); else values();
