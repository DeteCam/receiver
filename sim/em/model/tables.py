"""전파 표를 저장소에 넣을 만한 크기로 눌러 담는다 / 다시 푼다.
    python3 tables.py pack 이름 [이름 ...]      cache/이름.npz (복소수 64비트, 약 2.8MB) -> tables/이름.npz (16비트 실수·허수, 약 1.4MB 이하)
표 하나 = E[판 각 72, theta 37, phi 72, 편파 2]. 눌러 담을 때 생기는 오차는 가장 큰 값의 0.03%: 방향 계산에는 영향이 없다.
cache/ 는 run_tables.py 가 NEC2 로 다시 만들 수 있으므로 저장소에 넣지 않는다 (220MB)."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))); import empaths      # noqa: E401  sim/em 의 길잡이
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); DIR = empaths.TABLES; CACHE = empaths.CACHE
# 저장소에 넣는 표: 이름 -> 무엇인가
SHIP = {
    "ideal_z": "이상적인 안테나 (세로 전기 쌍극자) + 판만",
    "geo_mz_w100_R80": "이상적인 안테나 (세로 자기 쌍극자) + 판만",
    "geo_slot_w100_R80": "파이 기판 + 틈 안테나 + 단자 + 판",
    "head_mono": "파이 기판 + 막대 안테나 + 머리 속 부품 + 판",
    "head": "파이 기판 + 홈 안테나 + 머리 속 부품 + 판 (균형추 없음)",
    "dev_y+000": "지금 설계 전체 (홈 안테나, 머리·요크·받침·배터리·전선, 동전 균형추), 받침 0도",
    "dev_y+120": "지금 설계 전체, 받침 +120도",
    "dev_y-120": "지금 설계 전체, 받침 -120도",
}


def pack(name):
    d = np.load(os.path.join(CACHE, name + ".npz")); E = d["E"]; s = float(np.abs(E).max())
    q = np.stack([E.real / s, E.imag / s], -1).astype(np.float16)
    os.makedirs(DIR, exist_ok=True); f = os.path.join(DIR, name + ".npz"); np.savez_compressed(f, q=q, s=s, spins=d["spins"])
    back, _ = load(name); err = float(np.abs(back - E).max() / s)
    return os.path.getsize(f), err


def load(name):
    d = np.load(os.path.join(DIR, name + ".npz")); q = d["q"].astype(np.float32)
    return ((q[..., 0] + 1j * q[..., 1]) * float(d["s"])).astype(np.complex64), d["spins"]


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "pack":
        for nm in (sys.argv[2:] or list(SHIP)):
            if os.path.exists(os.path.join(CACHE, nm + ".npz")):
                size, err = pack(nm); print("%-20s %7.2f MB  오차 %.1e  %s" % (nm, size / 1e6, err, SHIP.get(nm, "")))
            else:
                print("%-20s cache 에 없음" % nm)
    else:
        print(__doc__)
