import sys, os, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from rk_thermal import temperature_on_frames
from track03_kinetics import frame_times
import refine_timing as RTm

PAPER_D = {"8_TRIP2": ["δ1"], "2_TRIP1": ["δ1", "δ2"], "31_A003": ["δ1", "δ2", "δ3"], "26_V19_5A": ["δ1"], "21_V19_1A": ["δ3", "δ1"]}
WIN = {"8_TRIP2": (3990, 4084), "2_TRIP1": (1166, 1854), "31_A003": (2039, 2120), "26_V19_5A": (2740, 3129), "21_V19_1A": (6680, 6800)}
VAR = {}
RT = json.load(open("refine_timing.json", encoding="utf-8"))


def deltas(case, labs):
    L = pd.read_csv(f"rk_out/{case}_labels.csv"); D = L[(L.phase == "delta") & L.label.isin(labs)]
    out = [dict(label=r.label, cy=float(r.cy), cx=float(r.cx), area=float(r.area)) for r in D.itertuples()]
    fn = f"rk_out/{case}_site_merges.json"
    if os.path.exists(fn):
        s2l = {int(r.site): r.label for r in D.itertuples()}
        for m in json.load(open(fn)):
            if int(m["keep"]) in s2l:
                out.append(dict(label=s2l[int(m["keep"])] + "'", cy=m["cy"], cx=m["cx"], area=m["area"]))
    return out


def run(case, labs):
    tf = frame_times(case); fps = 1 / np.median(np.diff(tf)); w0, w1 = WIN[case]
    S1 = dict(fps=fps, w0=w0, w1=w1, deltas=deltas(case, labs))
    r = RTm.delta_fraction(case, S1); T = temperature_on_frames(case)["T"]
    g = RT[case]["gamma"]["first_gamma"] or RT[case]["stage1"]["g0"]
    return dict(labels=labs, n_apertures=len(S1["deltas"]), start=r["start"], T_start=float(T[r["start"]]) if r["start"] else None,
                t0_frame=r["t0_frame"], T_t0=float(T[r["t0_frame"]]) if r["t0_frame"] else None,
                first_gamma=g, T_first_gamma=float(T[g]), complete=r["complete"], duration_s=r["duration_s"],
                full_disappear=r["full_disappear"], full_s=r["full_s"], t05=r["t05"], n=r["n"],
                frames_in_transition=r["frames_in_transition"], curve=r["curve"])


if __name__ == "__main__":
    OUT = {}
    for c, labs in PAPER_D.items():
        OUT[c] = run(c, labs)
        if c in VAR:
            OUT[c + "_var"] = run(c, VAR[c])
    for k, v in OUT.items():
        print(k, {q: (round(x, 3) if isinstance(x, float) else x) for q, x in v.items() if q != "curve"}, flush=True)
    json.dump(OUT, open("table1_paper_delta.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
