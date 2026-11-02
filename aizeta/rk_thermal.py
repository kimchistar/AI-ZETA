import os, re, glob
import numpy as np, pandas as pd
from datetime import datetime
from rk_params import P

from config import TEMP_FILES
REL_SYNC = {"31_A003": dict(frame=1235, t_log=101.8 + 44.0)}
TS = re.compile(r"_(\d{8})\.(\d{6})\.(\d{1,3})")


def laue_abs_times(case):
    from common import files
    out = []
    for f in files(case):
        m = TS.search(os.path.basename(f))
        d, hms, ms = m.groups()
        out.append(datetime.strptime(d + hms, "%Y%m%d%H%M%S").timestamp() + int(ms.ljust(3, "0")) / 1000)
    return np.array(out)


def read_temp(case):
    fn = TEMP_FILES.get(case)
    if fn is None or not os.path.exists(fn):
        return None, None
    d = pd.read_excel(fn, sheet_name="Sheet2", header=None)
    hdr_row = next(i for i in range(min(10, len(d))) if any("Time" in str(v) for v in d.iloc[i].values))
    hdr = [str(v) for v in d.iloc[hdr_row].values]
    tcol = next(j for j, h in enumerate(hdr) if ("B type" in h or "Sample Temp" in h))
    body = d.iloc[hdr_row + 1:]
    T = pd.to_numeric(body.iloc[:, tcol], errors="coerce").values
    ts = None
    for j in range(body.shape[1]):
        v = body.iloc[:, j].astype(str)
        if v.str.match(r"20\d\d-\d\d-\d\d").mean() > 0.9:
            ts = pd.to_datetime(v, errors="coerce").map(lambda x: x.timestamp() if pd.notna(x) else np.nan).values
            break
    if ts is None and case in REL_SYNC:
        jt = next((j for j, h in enumerate(hdr) if "sec" in h.lower()), None)
        if jt is not None:
            rel = pd.to_numeric(body.iloc[:, jt], errors="coerce").values
            lt = laue_abs_times(case); rs = REL_SYNC[case]
            off = pd.Timestamp("2000-01-01").timestamp() - datetime(2000, 1, 1).timestamp()
            ts = rel - rs["t_log"] + lt[rs["frame"] - files_offset(case)] + off
    ok = np.isfinite(T) & (np.isfinite(ts) if ts is not None else True)
    return (ts[ok] if ts is not None else None), T[ok]


def files_offset(case):
    from common import CASES
    return 0


T_CLOCK_S = {"21_V19_1A": 1.0}


def temperature_on_frames(case):
    ts, T = read_temp(case)
    if T is None:
        return dict(synced=False, T=None, events=[], reason="no_file")
    if ts is None:
        return dict(synced=False, T=None, events=[], reason="no_absolute_time")
    lt = laue_abs_times(case)
    off = pd.Timestamp("2000-01-01").timestamp() - datetime(2000, 1, 1).timestamp()
    ts = ts - off - T_CLOCK_S.get(case, 0.0)
    if lt[0] > ts[-1] or lt[-1] < ts[0]:
        return dict(synced=False, T=None, events=[], reason="no_overlap")
    Tf = np.interp(lt, ts, T, left=np.nan, right=np.nan)
    Ts = pd.Series(T).rolling(3, center=True, min_periods=1).mean().values
    dTdt = np.gradient(Ts, ts)
    trend = pd.Series(dTdt).rolling(151, center=True, min_periods=20).median().values
    resid = dTdt - trend
    mad = np.nanmedian(np.abs(resid - np.nanmedian(resid))) * 1.4826 + 1e-6
    hot = np.abs(resid) > P["k_thermal"] * mad
    cooling = trend < 0
    ev_t = []
    i = 0
    while i < len(hot):
        if hot[i]:
            j = i
            while j + 1 < len(hot) and (hot[j + 1] or (j + 2 < len(hot) and hot[j + 2])):
                j += 1
            if j - i + 1 >= 2:
                k = i + int(np.argmax(np.abs(resid[i:j + 1])))
                ev_t.append((ts[k], bool(cooling[k]), float(resid[k])))
            i = j + 1
        else:
            i += 1
    ev = [(int(np.argmin(np.abs(lt - t))), c, r) for t, c, r in ev_t if lt[0] <= t <= lt[-1]]
    ev_frames = sorted(set(e[0] for e in ev))
    cool_frames = np.interp(lt, ts, cooling.astype(float)) > 0.5
    return dict(synced=True, T=Tf, events=ev_frames, events_detail=ev, cooling=cool_frames, reason="ok")


def max_rise(T, i0, i1):
    seg = np.asarray(T[max(i0, 0):i1 + 1], float)
    if len(seg) < 2 or not np.isfinite(seg).any():
        return 0.0, None, None
    seg = np.where(np.isfinite(seg), seg, np.nanmean(seg))
    run_min = np.minimum.accumulate(seg); gain = seg - run_min
    j = int(np.argmax(gain)); i = int(np.argmin(seg[:j + 1])) if j > 0 else 0
    return float(gain[j]), max(i0, 0) + i, max(i0, 0) + j


def cooling_arrest(T, f, fps, pre=(10, 3), post=(-1, 3)):
    def slope(a, b):
        idx = np.arange(int(f + a * fps), int(f + b * fps) + 1)
        idx = idx[(idx >= 0) & (idx < len(T))]
        y = np.asarray(T, float)[idx]; ok = np.isfinite(y)
        return np.polyfit(idx[ok] / fps, y[ok], 1)[0] if ok.sum() >= 3 else np.nan
    s0, s1 = slope(-pre[0], -pre[1]), slope(post[0], post[1])
    return float(s1 / s0) if np.isfinite(s0) and s0 < 0 else np.nan


def scan_recalescence(T, fps, min_rise, win_s=3.0):
    T = np.asarray(T, float); w = int(win_s * fps); out = []; i = 0
    while i < len(T) - 1:
        r, a, b = max_rise(T, i, i + w)
        if r >= min_rise and a is not None:
            out.append((a, b, r)); i = b + w
        else:
            i += max(w // 2, 1)
    return out


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    for c in TEMP_FILES:
        r = temperature_on_frames(c)
        if r["synced"]:
            Tf = r["T"]
            print(f"{c:11s} synced  T {np.nanmin(Tf):.0f}-{np.nanmax(Tf):.0f} C  events {len(r['events'])}: {r['events'][:8]}")
        else:
            print(f"{c:11s} NOT synced ({r['reason']})")
