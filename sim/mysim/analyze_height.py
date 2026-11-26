import sys, numpy as np
sys.path.insert(0, '/opt/detecam/sim/mysim')
import analyze as A
base = A.load('/opt/detecam/sim/results/exp_base.pkl')
tags = sys.argv[1:] if len(sys.argv) > 1 else ['z120']
ALL = A.DEV_SCENES + A.EVAL_SCENES
for tag in tags:
    ext = A.load(f'/opt/detecam/sim/results/exp_{tag}.pkl')
    comb = {}
    for k, (m, o) in ext.items():
        if k in base:
            comb[k] = (m, {'EXT': o['R2'], 'BASE': base[k][1]['R2']})
    scenes = sorted({k[0] for k in comb})
    print(f"\n===== {tag} vs 기준 위치 (같은 카메라, {len(comb)} 파일)")
    for band in ('2g4', '5g'):
        sc = [s for s in ALL if s in scenes]
        for metric in ('one', 'diag_equal'):
            r = A.paired_boot(comb, 'EXT', 'BASE', band, sc, metric, n_boot=1000)
            if r: print(f"  [{band}] {metric:10s} | {A.fmt_cmp(r)}")
        for pn in ('pos_diag_equal', 'pos_adj_equal', 'pos_p02_equal'):
            e1, _ = A.collect(comb, 'EXT', band, sc, pn); e0, _ = A.collect(comb, 'BASE', band, sc, pn)
            print(f"  [{band}] 위치오차 {pn[4:-6]:5s}: {e1.mean():.2f} m vs 기준 {e0.mean():.2f} m")
        row = []
        for s in sc:
            e1, _ = A.collect(comb, 'EXT', band, [s], 'diag_equal'); e0, _ = A.collect(comb, 'BASE', band, [s], 'diag_equal')
            row.append(f"{s} {e1.mean():.1f}/{e0.mean():.1f}")
        print(f"  [{band}] 방별 대각(변형/기준): " + ", ".join(row))
    notes = [(k, m.get('node_notes')) for k, (m, o) in ext.items() if m.get('node_notes')] if isinstance(next(iter(ext.values()))[0], dict) else []
