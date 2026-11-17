"""place_t16.npz (손이 닿는 스무 곳, 0~19) + place_g16b.npz (grid 102곳, 20~121) -> place_g16.npz.
같은 카메라(씨앗 20000부터, 종류마다 400대)와 같은 사람 배치라 곳 축으로 이으면 122곳을 한 번에 잰 것과 같다."""
import os, json
import numpy as np
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "study")
a = np.load(os.path.join(OUT, "place_t16.npz")); b = np.load(os.path.join(OUT, "place_g16b.npz"))
assert (a["seat"] == b["seat"]).all() and (a["kind"] == b["kind"]).all() and (a["owner"] == b["owner"]).all(), "카메라가 다르다"
assert a["sel"].tolist() == list(range(20)) and b["sel"].tolist() == list(range(20, 122)), (a["sel"], b["sel"])
out = dict(seat=a["seat"], kind=a["kind"], owner=a["owner"], seats_xy=a["seats_xy"], spots=np.concatenate([a["spots"], b["spots"]]), sel=np.arange(122),
           sets=b["sets"], opt=json.dumps(dict(json.loads(str(b["opt"])), spots=list(range(122)))))
for k in ("A", "W8", "W4"):
    out[k] = np.concatenate([a[k], b[k]], axis=1)
out["L1"] = np.concatenate([a["L1"], b["L1"]], axis=2)
np.savez(os.path.join(OUT, "place_g16.npz"), **out)
print("이음 place_g16.npz", {k: v.shape for k, v in out.items() if hasattr(v, "shape")}, flush=True)
