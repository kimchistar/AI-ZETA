import sys, os, json
import numpy as np, pandas as pd, cv2
sys.path.insert(0, ".")
from rk_params import P
from common import load
from prep import preprocess


def _smooth(y, w):
    return pd.Series(y).rolling(w, center=True, min_periods=1).median().values


def profile_stats(I, w=P["w_smooth"]):
    I = np.asarray(I, float)
    ok = np.isfinite(I)
    if ok.sum() < 5:
        return dict(t_pk_idx=None, rise=np.nan, fall=np.nan, unimodal=False)
    y = _smooth(np.where(ok, I, np.nan), w)
    y = pd.Series(y).interpolate(limit_direction="both").values
    k = int(np.nanargmax(y))
    lo_b = np.nanmin(y[:k + 1]); lo_a = np.nanmin(y[k:])
    rise = y[k] / max(lo_b, 1e-9); fall = y[k] / max(lo_a, 1e-9)
    from scipy.signal import find_peaks
    rng = np.nanmax(y) - np.nanmin(y)
    pk, _ = find_peaks(np.r_[-np.inf, y, -np.inf], prominence=0.3 * rng if rng > 0 else 1)
    return dict(t_pk_idx=k, rise=float(rise), fall=float(fall), unimodal=bool(len(pk) == 1))


def units_detected(T, frames):
    f0, f1 = int(frames[0]), int(frames[-1])
    rows, series = [], {}
    for tid, g in T.groupby("tid"):
        g = g.sort_values("frame")
        s = pd.Series(g.I_sum.values, index=g.frame.values.astype(int)).reindex(frames)
        ps = profile_stats(s.values)
        disp = np.hypot(g.cy - g.cy.iloc[0], g.cx - g.cx.iloc[0]).max()
        rows.append(dict(uid=int(tid), **({"site": int(g.site.iloc[0])} if "site" in g.columns else {}),
                         cy=float(g.cy.median()), cx=float(g.cx.median()),
                         ignite=int(g.frame.min()), extinguish=int(g.frame.max()),
                         present_start=int(g.frame.min()) <= f0, alive_end=int(g.frame.max()) >= f1,
                         area=float(g.area.median()), ecc=float(g.ecc.median()),
                         flat=float(np.nanstd(s.values) / max(np.nanmean(s.values), 1e-9)),
                         n_obs=len(g), max_disp=float(disp), I_med=float(g.I_sum.median()),
                         t_peak=int(frames[ps["t_pk_idx"]]) if ps["t_pk_idx"] is not None else None,
                         rise=ps["rise"], fall=ps["fall"], unimodal=ps["unimodal"]))
        series[int(tid)] = s.values
    return pd.DataFrame(rows), series


def _disk(r):
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    m = yy ** 2 + xx ** 2 <= r * r
    return yy[m], xx[m]


def _objarr(lst):
    a = np.empty(len(lst), dtype=object)
    for i, v in enumerate(lst):
        a[i] = v
    return a


def measure_sites(case, T, frames, cache_dir="rk_cache", tag="v6"):
    from scipy.spatial import cKDTree
    os.makedirs(cache_dir, exist_ok=True)
    fn = f"{cache_dir}/{case}_sites_{tag}.npz"
    mk = np.load(f"mask_{case}.npy")
    sites = []
    for tid, g in T.groupby("site"):
        w = g.I_sum.clip(lower=1).values
        cy = float(np.average(g.cy, weights=w)); cx = float(np.average(g.cx, weights=w))
        amed = float(g.area.median())
        big = amed >= P["A_sap"] / 2
        r = int(np.ceil(np.sqrt(amed / np.pi))) + P["margin"]
        sites.append((int(tid), cy, cx, int(min(r, 45 if big else 15)), big))
    dsk = [_disk(int(min(12, np.ceil(np.sqrt(float(g.area.median()) / np.pi)) + 2))) for _, g in T.groupby("site")]
    if os.path.exists(fn):
        Z = np.load(fn, allow_pickle=True)
        if list(Z["tids"]) == [s[0] for s in sites] and len(Z["frames"]) == len(frames):
            return dict(Z)
    C = np.array([[s[1], s[2]] for s in sites])
    yy, xx = np.mgrid[:800, :800]
    _, near = cKDTree(C).query(np.c_[yy.ravel(), xx.ravel()], k=1)
    near = near.reshape(800, 800)
    bigmask = np.zeros((800, 800), bool)
    for k, (tid, cy, cx, r, big) in enumerate(sites):
        if big:
            bigmask |= (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    ap, an = [], []
    for k, (tid, cy, cx, r, big) in enumerate(sites):
        y0, y1 = max(int(cy) - r, 0), min(int(cy) + r + 1, 800); x0, x1 = max(int(cx) - r, 0), min(int(cx) + r + 1, 800)
        sub_y, sub_x = yy[y0:y1, x0:x1], xx[y0:y1, x0:x1]
        m = ((sub_y - cy) ** 2 + (sub_x - cx) ** 2 <= r * r) & (near[y0:y1, x0:x1] == k) & mk[y0:y1, x0:x1]
        if not big:
            m &= ~bigmask[y0:y1, x0:x1]
        py, px = sub_y[m], sub_x[m]
        if len(py) < 6:
            py, px = _disk(2); py = py + int(round(cy)); px = px + int(round(cx))
        ap.append((py.astype(np.int32) - int(round(cy)), px.astype(np.int32) - int(round(cx))))
        R2 = r + 7
        y0b, y1b = max(int(cy) - R2, 0), min(int(cy) + R2 + 1, 800); x0b, x1b = max(int(cx) - R2, 0), min(int(cx) + R2 + 1, 800)
        by_, bx_ = yy[y0b:y1b, x0b:x1b], xx[y0b:y1b, x0b:x1b]
        dd = (by_ - cy) ** 2 + (bx_ - cx) ** 2
        mb = (dd >= (r + 2) ** 2) & (dd <= R2 ** 2) & mk[y0b:y1b, x0b:x1b]
        if not big:
            mb &= ~bigmask[y0b:y1b, x0b:x1b]
        an.append((by_[mb].astype(np.int32) - int(round(cy)), bx_[mb].astype(np.int32) - int(round(cx))))
    n, m = len(frames), len(sites)
    I = np.zeros((m, n), np.float32); S = np.zeros((m, n), np.float32); A = np.zeros((m, n), np.float32)
    SD = np.zeros((m, n), np.float32)
    drift = np.zeros((n, 2), np.float32)
    ref = None
    bigk = [k for k, s in enumerate(sites) if s[4]]
    crops = {k: [] for k in bigk}
    for j, f in enumerate(frames):
        Pp = preprocess(load(case, int(f)), mk)
        R = np.clip(Pp["res"], 0, None); Sm = Pp["S"]
        u = Pp["u"].astype(np.float32)
        Rpad = np.pad(Pp["res"], 120) if bigk else None
        if ref is None:
            ref = u; win = cv2.createHanningWindow(u.shape[::-1], cv2.CV_32F)
        (dx, dy), _ = cv2.phaseCorrelate(ref, u, win)
        if abs(dy) > 3 or abs(dx) > 3:
            dy = dx = 0.0
        drift[j] = (dy, dx)
        sy, sx = int(round(dy)), int(round(dx))
        for k, (tid, cy, cx, r, big) in enumerate(sites):
            oy, ox = ap[k]
            py = np.clip(int(round(cy)) + sy + oy, 0, 799); px = np.clip(int(round(cx)) + sx + ox, 0, 799)
            by_, bx_ = an[k]
            if len(by_) >= 8:
                qy = np.clip(int(round(cy)) + sy + by_, 0, 799); qx = np.clip(int(round(cx)) + sx + bx_, 0, 799)
                bR = float(np.median(Pp["res"][qy, qx])); bS = float(np.median(Sm[qy, qx]))
            else:
                bR = bS = 0.0
            I[k, j] = np.clip(Pp["res"][py, px] - bR, 0, None).sum()
            sv = Sm[py, px] - bS
            S[k, j] = np.sort(sv)[-6:].mean() if len(sv) >= 6 else sv.max()
            A[k, j] = (sv >= P["theta_area"]).sum()
            dy_, dx_ = dsk[k]
            qy2 = np.clip(int(round(cy)) + sy + dy_, 0, 799); qx2 = np.clip(int(round(cx)) + sx + dx_, 0, 799)
            SD[k, j] = np.sort(Sm[qy2, qx2])[-6:].mean()
            if k in crops:
                y0 = int(np.clip(round(cy) + sy, 0, 799)) + 120; x0 = int(np.clip(round(cx) + sx, 0, 799)) + 120
                crops[k].append(Rpad[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1].astype(np.float32))
    out = dict(tids=np.array([s[0] for s in sites]), cy=np.array([s[1] for s in sites]),
               cx=np.array([s[2] for s in sites]), rad=np.array([s[3] for s in sites]),
               frames=np.asarray(frames), I=I, S=S, A=A, SD=SD, drift=drift,
               crop_keys=np.array(bigk, dtype=int),
               crops=_objarr([np.stack(crops[k]) for k in bigk]))
    np.savez_compressed(fn, **out)
    return out


def hysteresis_all(Sk, on=P["theta_on"], off=P["theta_off"], m_on=P["m_on"], m_off=P["m_off"]):
    out, i, n = [], 0, len(Sk)
    while i < n:
        run, ig = 0, None
        while i < n:
            run = run + 1 if Sk[i] >= on else 0
            i += 1
            if run >= m_on:
                ig = i - m_on; break
        if ig is None:
            break
        run, ex = 0, None
        while i < n:
            run = run + 1 if Sk[i] < off else 0
            i += 1
            if run >= m_off:
                ex = i - m_off; break
        out.append((ig, ex))
        if ex is None:
            break
    return out


def units_continuous(case, T, frames, tag="v6"):
    M = measure_sites(case, T, frames, tag=tag)
    fr = M["frames"]; fidx = {int(f): i for i, f in enumerate(fr)}
    f0, f1 = int(fr[0]), int(fr[-1])
    kof = {int(s): k for k, s in enumerate(M["tids"])}
    rows, series = [], {}
    for site, gs in T.groupby("site"):
        k = kof.get(int(site))
        if k is None:
            continue
        eps = [g for _, g in sorted(gs.groupby("tid"), key=lambda kv: kv[1].frame.min())]
        for j, g in enumerate(eps):
            ig, ex = fidx[int(g.frame.min())], fidx[int(g.frame.max())]
            nxt = fidx[int(eps[j + 1].frame.min())] if j + 1 < len(eps) else len(fr)
            s = M["I"][k].astype(float).copy()
            if nxt < len(fr):
                s[nxt:] = s[nxt - 1]
            s[:ig] = s[ig]
            seg = s[ig:nxt]
            ps = profile_stats(seg) if len(seg) >= 5 else dict(t_pk_idx=None, rise=np.nan, fall=np.nan, unimodal=False)
            uid = int(g.tid.iloc[0])
            rows.append(dict(uid=uid, site=int(site), cy=float(M["cy"][k]), cx=float(M["cx"][k]),
                             ignite=int(g.frame.min()), extinguish=int(g.frame.max()),
                             present_start=int(g.frame.min()) <= f0, alive_end=int(g.frame.max()) >= f1,
                             area=float(g.area.median()), ecc=float(g.ecc.median()),
                             flat=(float(np.std(M["I"][k][ig:ex + 1]) / max(np.mean(M["I"][k][ig:ex + 1]), 1e-9))
                                   if ("merged" in g.columns and bool(g.merged.any()))
                                   else float(g.I_sum.std() / max(g.I_sum.mean(), 1e-9))),
                             flat_cont=float(np.std(M["I"][k][ig:ex + 1]) / max(np.mean(M["I"][k][ig:ex + 1]), 1e-9)),
                             n_obs=len(g), max_disp=float(np.hypot(g.cy - M["cy"][k], g.cx - M["cx"][k]).max()),
                             I_med=float(g.I_sum.median()),
                             t_peak=int(fr[ig + ps["t_pk_idx"]]) if ps["t_pk_idx"] is not None else None,
                             rise=ps["rise"], fall=ps["fall"], unimodal=ps["unimodal"], rad=int(M["rad"][k])))
            series[uid] = s
    return pd.DataFrame(rows), series, M
