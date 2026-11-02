import sys, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from common import load
from prep import preprocess
from rk_kinetics import disk_sum
from rk_thermal import temperature_on_frames
from track03_kinetics import frame_times
from clean_delta import load_clean
import refine_timing as RTm
import table1_paper_delta as T1

RT = json.load(open("refine_timing.json", encoding="utf-8"))
OUT = {}
for case in sys.argv[1:] or ["8_TRIP2", "2_TRIP1", "31_A003", "26_V19_5A", "21_V19_1A"]:
    tf = frame_times(case); fps = 1 / np.median(np.diff(tf)); T = temperature_on_frames(case)["T"]
    L = pd.read_csv(f"rk_out/{case}_labels.csv"); L = L[L.label.astype(str).str.len() > 0]
    g1 = RT[case]["gamma"]["first_gamma"] or RT[case]["stage1"]["g0"]
    cl = load_clean(case)
    D = L[(L.phase == "delta") & (L.first_frame < g1) & (L.last_frame >= g1 - 3 * fps)]
    if cl is not None:
        D = D[D.label.isin(cl)]
    D = D.assign(_n=D.label.str[1:].astype(int)).sort_values("_n"); labs = list(D.label)
    w0, w1 = T1.WIN[case]
    S1 = dict(fps=fps, w0=w0, w1=w1, deltas=T1.deltas(case, labs))
    tr = RTm.delta_fraction(case, S1); fr = np.array(tr["curve"]["frames"])
    mk = np.load(f"mask_{case}.npy"); I = {l: np.zeros(len(fr)) for l in labs}
    for j, f in enumerate(fr):
        R = preprocess(load(case, int(f)), mk)["res"]
        for d in S1["deltas"]:
            l = d["label"].rstrip("'"); ri = int(min(30, np.sqrt(max(d["area"], 20) / np.pi) + 2))
            I[l][j] += disk_sum(R, d["cy"], d["cx"], r_in=ri, r1=ri + 2, r2=ri + 8)
    w3 = max(3, int(round(3 * fps))); hold = max(2, int(round(0.3 * fps)))
    sm = lambda y: pd.Series(y).rolling(3, center=True, min_periods=1).median().values
    def frac(y):
        y = sm(y); fl = np.median(y[-w3:]); return (y - fl) / max(y.max() - fl, 1e-9)
    def fs(cond, frm):
        for k in range(frm, len(cond) - hold + 1):
            if cond[k:k + hold].all():
                return k
    per = {}
    for l, v in I.items():
        x = frac(v); s = fs(x < 0.95, int(np.argmax(x)))
        e = fs(x <= max(0.01, 3 * float(np.std(x[-w3:]))), s) if s is not None else None
        rtd = next((d.get("t_dis") for d in RT[case]["deltas"] if d["label"] == l), "n/a")
        per[l] = dict(start95=int(fr[s]) if s is not None else None, end=int(fr[e]) if e is not None else None,
                      end_s=round(float(tf[fr[e]] - tf[g1]), 3) if e is not None else None,
                      T_end=round(float(T[fr[e]]), 1) if e is not None else None, refine_json_t_dis=rtd,
                      label_last_frame=int(D[D.label == l].last_frame.iloc[0]))
    ends = [p["end"] for p in per.values() if p["end"] is not None]
    OUT[case] = dict(fps=round(float(fps), 2), first_gamma=int(g1), deltas=labs, per_delta=per,
                     combined=dict(start=tr["start"], complete=tr["complete"], full_disappear=tr["full_disappear"], t05=tr["t05"], n=tr["n"]),
                     last_delta_end=max(ends) if ends else None,
                     last_delta_end_s=round(float(tf[max(ends)] - tf[g1]), 3) if ends else None,
                     refine_json_delta_disappear=RT[case].get("delta_disappear"))
    print(case, json.dumps(OUT[case], ensure_ascii=False, default=int), flush=True)
json.dump(OUT, open("delta_end.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
