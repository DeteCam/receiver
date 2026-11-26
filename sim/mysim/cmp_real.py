"""Compact comparison of several designs on the realistic rooms.
Usage: python3 mysim/cmp_real.py <band> <ref cfg@tag> <cfg@tag> [<cfg@tag> ...]"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import analyze_real as A
import real_scenes as R

band = sys.argv[1]
items = [x.split("@") for x in sys.argv[2:]]
cache = {}
def res(tag):
    if tag not in cache:
        cache[tag] = A.load(tag)
    return cache[tag]
ref_cfg, ref_tag = items[0]
print(f"band {band}; reference {ref_cfg}@{ref_tag}; bearing = four-combo spots, outside rooms; position = 4 spots / 2 spots within 0.5 m; exam = top-3 desks (pair)   [simulation]")
print(f"{'cfg':10s} {'mean':>6s} {'med':>5s} {'p90':>5s} {'p95':>5s} | {'diff':>6s} {'rel':>6s} {'room CI':>16s} {'roomxseed CI':>16s} | {'exam mean':>9s} | {'4sp lodg':>8s} {'rest':>5s} {'lock':>5s} {'2sp all':>7s} | {'ex top3':>7s} | per-room mean")
for cfg, tag in items:
    r = res(tag)
    rows = A.bearing_rows(r, cfg, band, "four")
    if not rows:
        print(f"{cfg:10s} (no data)"); continue
    ro = [x for x in rows if x["scene"] not in R.EXAM]
    v = A.cat(ro)
    refrows = [x for x in A.bearing_rows(res(ref_tag), ref_cfg, band, "four") if x["scene"] not in R.EXAM]
    b = A.room_boot(ro, refrows)
    ex = A.cat([x for x in rows if x["scene"] in R.EXAM])
    pr = A.pos_rows(r, cfg, band, "four")
    def within(kind):
        w = A.cat([x for x in pr if R.KIND[x["scene"]] == kind])
        return f"{100*np.mean(w<=0.5):5.1f}" if len(w) else "    -"
    p2 = A.cat([x for x in A.pos_rows(r, cfg, band, "pair") if x["scene"] not in R.EXAM])
    e3 = [x["seat3"] for x in A.pos_rows(r, cfg, band, "pair") if x["seat3"] is not None]
    per = {sc: np.mean(A.cat([x for x in rows if x["scene"] == sc])) for sc in sorted({x["scene"] for x in rows}, key=R.REAL_SCENES.index)}
    print(f"{cfg:10s} {v.mean():6.2f} {np.median(v):5.1f} {np.percentile(v,90):5.1f} {np.percentile(v,95):5.1f} | "
          f"{b['mean_d']:+6.2f} {100*b['rel']:+5.1f}% [{b['ci']['room'][0]:+6.2f},{b['ci']['room'][1]:+6.2f}] [{b['ci']['room_seed'][0]:+6.2f},{b['ci']['room_seed'][1]:+6.2f}] | "
          f"{(ex.mean() if len(ex) else float('nan')):9.2f} | {within('lodging'):>8s} {within('restroom'):>5s} {within('locker'):>5s} {(100*np.mean(p2<=0.5) if len(p2) else float('nan')):7.1f} | "
          f"{(100*np.mean(np.concatenate(e3)) if e3 else float('nan')):7.1f} | " + " ".join(f"{k[:8]} {x:.1f}" for k, x in per.items()))
