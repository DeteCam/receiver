import sys, numpy as np
sys.path.insert(0, '/opt/detecam/sim/mysim')
import analyze as A
res = A.load('/opt/detecam/sim/results/exp_base.pkl')
ALL = A.DEV_SCENES + A.EVAL_SCENES

def cmp(a, b, band, scenes, metric):
    r = A.paired_boot(res, a, b, band, scenes, metric, n_boot=1500, level="camera")
    return A.fmt_cmp(r)

print("=== E7 부호 앵커 비용 / 약한 신호, E5 측정 바퀴 수, 상자 모델 민감도 (평가 9장면, A vs B, 음수=A가 좋음)")
for band in ("2g4", "5g"):
    print(f"\n[{band}]")
    for name, a, b in [("이상(반전 없음) vs 앵커 α=3.5", "E7_noflip", "R2"),
                       ("앵커 α=2 vs α=3.5", "E7_a2", "R2"),
                       ("앵커 α=5 vs α=3.5", "E7_a5", "R2"),
                       ("SNR 10dB vs 20dB (앵커)", "E7_snr10", "R2"),
                       ("SNR10 이상 vs SNR10 앵커", "E7_noflip_snr10", "E7_snr10"),
                       ("바퀴 5 vs 10", "E5_n5", "R2"),
                       ("바퀴 30 vs 10", "E5_n30", "R2"),
                       ("거친 상자 모델 vs 촘촘한 상자", "BOX_coarse", "R2")]:
        for m in ("one", "diag_equal"):
            print(f"  {name:28s} {m:10s} | {cmp(a, b, band, A.EVAL_SCENES, m)}")
    for c in ("R2", "E7_a2", "E7_a5", "E7_snr10"):
        sa = np.concatenate([res[k][1][c]['sign_acc'].ravel() for k in res if k[1] == band and k[0] in A.EVAL_SCENES])
        print(f"  부호복원 {c:10s}: 평균 {100*sa.mean():.2f}% | 50% 이하로 실패한 표본 {100*(sa<0.6).mean():.2f}%")
