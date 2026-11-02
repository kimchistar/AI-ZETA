import sys, os, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd

CASES = {"21_V19_1A": (6660, 6800, 6698)}


def load_clean(case):
    fn = f"rk_out/{case}_clean_delta.json"
    return set(json.load(open(fn, encoding="utf-8"))["clean"]) if os.path.exists(fn) else None


def compute(case, F0, F1, IMG):
    from common import load
    from prep import preprocess
    from rk_kinetics import disk_sum
    from track03_kinetics import frame_times
    L = pd.read_csv(f"rk_out/{case}_labels.csv"); L = L[L.label.astype(str).str.len() > 0]
    D = L[(L.phase == "delta") & (L.first_frame <= IMG) & (L.last_frame >= IMG)].sort_values("label")
    AP = {r.label: [(r.cy, r.cx, r.area)] for r in D.itertuples()}
    fn = f"rk_out/{case}_site_merges.json"
    if os.path.exists(fn):
        s2l = {str(r.site): r.label for r in D.itertuples()}
        for m in json.load(open(fn)):
            if str(m["keep"]) in s2l:
                AP[s2l[str(m["keep"])]].append((m["cy"], m["cx"], m["area"]))
    mk = np.load(f"mask_{case}.npy"); tf = frame_times(case); fps = 1 / np.median(np.diff(tf))
    hold = max(2, int(round(0.3 * fps))); w3 = max(3, int(round(3 * fps)))
    fr = np.arange(F0, F1 + 1); I = {k: np.zeros(len(fr)) for k in AP}
    for j, f in enumerate(fr):
        R = preprocess(load(case, int(f)), mk)["res"]
        for k, aps in AP.items():
            for cy, cx, a in aps:
                ri = int(min(30, np.sqrt(max(a, 20) / np.pi) + 2)); I[k][j] += disk_sum(R, cy, cx, r_in=ri, r1=ri + 2, r2=ri + 8)
    sm = lambda y: pd.Series(y).rolling(3, center=True, min_periods=1).median().values
    frac = lambda y: (sm(y) - np.median(sm(y)[-w3:])) / max(sm(y).max() - np.median(sm(y)[-w3:]), 1e-9)
    def fs(cond, frm):
        for k in range(frm, len(cond) - hold + 1):
            if cond[k:k + hold].all():
                return k
    Xc = frac(sum(I.values())); t = tf[fr] - tf[fr[fs(Xc < 0.95, int(np.argmax(Xc)))]]
    pk = {k: sm(v).max() for k, v in I.items()}; pmax = max(pk.values())
    rows = []
    for k, v in I.items():
        x = frac(v)
        rows.append(dict(label=k, peak_rel=round(pk[k] / pmax, 3), pre_std=round(float(np.std(x[(t >= -1.5) & (t < -0.1)])), 3),
                         post_mean=round(float(np.mean(x[(t >= 0.5) & (t <= 2.5)])), 3)))
    Q = pd.DataFrame(rows)
    Q["clean"] = (Q.peak_rel >= 0.05) & (Q.pre_std <= 0.05) & (Q.post_mean.abs() <= 0.1)
    out = dict(case=case, window=[int(F0), int(F1)], img=int(IMG), criteria="peak ≥ 5 % of max, pre-std ≤ 0.05, |post mean| ≤ 0.1",
               clean=sorted(Q.label[Q.clean].tolist(), key=lambda s: int(s[1:])), table=Q.to_dict("records"))
    json.dump(out, open(f"rk_out/{case}_clean_delta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(Q.sort_values("peak_rel", ascending=False).to_string(index=False)); print("clean:", out["clean"])
    return out


if __name__ == "__main__":
    case = sys.argv[1] if len(sys.argv) > 1 else "21_V19_1A"
    F0, F1, IMG = [int(x) for x in sys.argv[2:5]] if len(sys.argv) >= 5 else CASES[case]
    compute(case, F0, F1, IMG)
