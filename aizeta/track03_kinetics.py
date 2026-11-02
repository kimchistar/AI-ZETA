import sys, os, re, json
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from datetime import datetime
sys.path.insert(0, ".")
from common import files, load
from prep import preprocess

TS = re.compile(r"_(\d{8})\.(\d{6})\.(\d{1,3})")


def frame_times(case):
    t = []
    for f in files(case):
        m = TS.search(os.path.basename(f))
        if not m:
            t.append(np.nan); continue
        d, hms, ms = m.groups()
        t.append(datetime.strptime(d + hms, "%Y%m%d%H%M%S").timestamp() + int(ms.ljust(3, "0")) / 1000)
    t = np.array(t, float)
    return t - np.nanmin(t)


def avrami(t, t05, n):
    return 1 - np.exp(-np.log(2) * np.power(np.clip(t, 1e-9, None) / t05, n))


def t05_interp(t, Xd):
    t, Xd = np.asarray(t, float), np.asarray(Xd, float)
    for k in range(1, len(Xd)):
        if Xd[k - 1] > 0.5 >= Xd[k] and t[k] > 0:
            return float(t[k - 1] + (Xd[k - 1] - 0.5) / (Xd[k - 1] - Xd[k]) * (t[k] - t[k - 1]))
    return None


def n_from_t05(t, Xd, t05):
    t, Xd = np.asarray(t, float), np.asarray(Xd, float)
    k = next(k for k in range(len(t)) if t[k] > t05)
    return float(np.log(-np.log(Xd[k]) / np.log(2)) / np.log(t[k] / t05))


def fit_avrami(t, X):
    ok = np.isfinite(t) & np.isfinite(X) & (t > 0)
    t, X = t[ok], np.clip(X[ok], 0, 1)
    if len(t) < 8 or X.max() < 0.3:
        return np.nan, np.nan, np.nan
    p0 = [t[np.argmax(X >= 0.5)] if (X >= 0.5).any() else np.median(t), 1.5]
    try:
        p, _ = curve_fit(avrami, t, X, p0=p0, bounds=([1e-3, 0.2], [1e3, 8]), maxfev=40000)
    except Exception:
        return np.nan, np.nan, np.nan
    rmse = float(np.sqrt(np.mean((X - avrami(t, *p)) ** 2)))
    return float(p[0]), float(p[1]), rmse


def smooth(y, w=5):
    return pd.Series(y).rolling(w, center=True, min_periods=1).median().values


def run(case):
    T = pd.read_csv(f"track_obs_{case}.csv")
    S = pd.read_csv(f"track_sum_{case}.csv")
    info = json.load(open(f"track_info_{case}.json"))
    tfull = frame_times(case)
    frames = np.array(sorted(T.frame.unique()), dtype=int)
    onset = int(info["onset"]); t0 = float(tfull[onset]); t = tfull[frames] - t0
    curves = {}
    for ph in ("delta", "gamma", "static"):
        sub = T[T.phase == ph]
        curves[ph] = dict(
            I=sub.groupby("frame").I_sum.sum().reindex(frames, fill_value=0.0).values,
            N=sub.groupby("frame").tid.nunique().reindex(frames, fill_value=0).values)
    pre = frames <= onset
    Id = smooth(curves["delta"]["I"])
    ref_d = np.median(Id[pre][-30:]) if pre.sum() > 5 else Id[0]
    Xd = np.clip(1 - Id / max(ref_d, 1e-9), 0, 1)
    Ig = smooth(curves["gamma"]["I"])
    plateau = np.median(np.sort(Ig)[-int(max(5, 0.1 * len(Ig))):])
    Xg = np.clip(Ig / max(plateau, 1e-9), 0, 1)
    fits = dict(ours_gamma=fit_avrami(t[t > 0], Xg[t > 0]))
    mk = np.load(f"mask_{case}.npy")
    Sd = S[S.phase == "delta"].sort_values("I_mean", ascending=False)
    fr = frames
    res_cache = {}
    def res_of(f):
        if f not in res_cache:
            res_cache[f] = np.clip(preprocess(load(case, int(f)), mk)["res"], 0, None)
        return res_cache[f]
    def roi_sum(f, cy, cx, h):
        R = res_of(f)
        return float(R[max(int(cy)-h,0):int(cy)+h+1, max(int(cx)-h,0):int(cx)+h+1].sum())
    fixed, tracked = {}, {}
    for _, r in Sd.iterrows():
        obs = T[T.tid == r.tid].sort_values("frame")
        h = max(6, int(np.sqrt(r.area)) + 5)
        pos = pd.DataFrame(dict(cy=obs.cy.values, cx=obs.cx.values), index=obs.frame.values.astype(int))
        pos = pos.reindex(fr).ffill().bfill()
        y0, x0 = float(obs.iloc[0].cy), float(obs.iloc[0].cx)
        fixed[int(r.tid)] = [roi_sum(f, y0, x0, h) for f in fr]
        tracked[int(r.tid)] = [roi_sum(f, pos.cy.loc[f], pos.cx.loc[f], h) for f in fr]
    tt = tfull[fr] - t0
    def norm_fit(v):
        v = np.array(v, float); prem = fr <= onset
        ref2 = np.median(v[prem][-20:]) if prem.sum() > 3 else v[0]
        X2 = np.clip(1 - smooth(v) / max(ref2, 1e-9), 0, 1)
        return X2, fit_avrami(tt[tt > 0], X2[tt > 0])
    per, per_tracked = {}, {}
    for tid in fixed:
        X2, f2 = norm_fit(fixed[tid]);  per[int(tid)] = dict(fit=f2, curve=fixed[tid], X=X2.tolist())
        X3, f3 = norm_fit(tracked[tid]); per_tracked[int(tid)] = dict(fit=f3, curve=tracked[tid], X=X3.tolist())
    tot = np.sum([np.array(tracked[t]) for t in tracked], axis=0)
    Xd, fit_d = norm_fit(tot)
    fits["ours_delta"] = fit_d
    out = dict(case=case, onset_frame=onset, onset_time=t0, fps=float(1 / np.median(np.diff(tfull))),
               frames=frames.tolist(), t=t.tolist(), Xd=Xd.tolist(), Xg=Xg.tolist(),
               I=dict((k, curves[k]["I"].tolist()) for k in curves),
               N=dict((k, curves[k]["N"].tolist()) for k in curves),
               counts={k:int(v) for k,v in S.phase.value_counts().items()}, fits={k: list(v) for k, v in fits.items()},
               roi_frames=[int(x) for x in fr], roi_t=tt.tolist(),
               published_fixed={str(k): v for k, v in per.items()},
               published_tracked={str(k): v for k, v in per_tracked.items()},
               info=info)
    json.dump(out, open(f"track_kin_{case}.json", "w"), default=float)
    print(f"{case}: onset frame {onset} | tracks delta {out['counts'].get('delta',0)} "
          f"gamma {out['counts'].get('gamma',0)} static {out['counts'].get('static',0)}")
    print(f"  ours  delta-decay t0.5={fits['ours_delta'][0]:.2f}s n={fits['ours_delta'][1]:.2f} rmse={fits['ours_delta'][2]:.3f}")
    print(f"  ours  gamma-growth t0.5={fits['ours_gamma'][0]:.2f}s n={fits['ours_gamma'][1]:.2f} rmse={fits['ours_gamma'][2]:.3f}")
    for tid, v in per.items():
        a, b, c = v["fit"]; d, e, g = per_tracked[tid]["fit"]
        print(f"  spot {tid}: fixed ROI t0.5={a:.2f}s n={b:.2f} (rmse {c:.3f})   tracked t0.5={d:.2f}s n={e:.2f} (rmse {g:.3f})")


if __name__ == "__main__":
    run(sys.argv[1])
