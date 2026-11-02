import sys, json, os
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from rk_params import P
from rk_trackers import TRACKERS
from rk_units import units_detected, units_continuous
from rk_rules import run_rules
from rk_rsd import rsd_children
from rk_thermal import temperature_on_frames
from track03_kinetics import frame_times, fit_avrami

ARMS = ["site_full"]


def kinetics(U, series, frames, tsec, onset):
    if onset is None:
        return dict(delta=[np.nan] * 3, gamma=[np.nan] * 3)
    tt = np.array([tsec[int(f)] for f in frames]) - tsec[int(onset)]
    def total(ph):
        ids = U.uid[U.phase == ph].values
        arr = [np.nan_to_num(np.asarray(series[int(i)], float)) for i in ids if int(i) in series]
        return np.sum(arr, axis=0) if arr else np.zeros(len(frames))
    sm = lambda y: pd.Series(y).rolling(5, center=True, min_periods=1).median().values
    Id = sm(total("delta")); Ig = sm(total("gamma"))
    pre = tt <= 0
    ref = np.median(Id[pre][-20:]) if pre.sum() > 3 else (Id[0] if len(Id) else 1)
    Xd = np.clip(1 - Id / max(ref, 1e-9), 0, 1)
    plateau = np.median(np.sort(Ig)[-max(5, len(Ig) // 10):]) if len(Ig) else 1
    Xg = np.clip(Ig / max(plateau, 1e-9), 0, 1)
    return dict(delta=list(fit_avrami(tt[tt > 0], Xd[tt > 0])), gamma=list(fit_avrami(tt[tt > 0], Xg[tt > 0])),
                curve_t=tt.tolist(), Xd=Xd.tolist(), Xg=Xg.tolist())


def gap_fill(T, M, max_gap):
    fr = np.asarray(M["frames"]); idx = {int(f): i for i, f in enumerate(fr)}
    kof = {int(s): k for k, s in enumerate(M["tids"])}
    T = T.copy(); ext = {}
    for sid, g in T.groupby("site"):
        k = kof.get(int(sid))
        if k is None:
            continue
        sd = pd.Series(np.asarray(M["SD"][k], float)).rolling(3, center=True, min_periods=1).median().values
        det_i = np.array(sorted({idx[int(f)] for f in g.frame if int(f) in idx})); set_det = set(det_i.tolist())
        bright = sd >= P["theta_off"]; pres = bright.copy(); pres[det_i] = True
        keep = np.zeros(len(pres), bool); i = 0
        while i < len(pres):
            if pres[i]:
                j = i
                while j + 1 < len(pres) and pres[j + 1]:
                    j += 1
                if ((det_i >= i) & (det_i <= j)).any():
                    keep[i:j + 1] = True
                i = j + 1
            else:
                i += 1
        if len(det_i):
            keep[:det_i.min()] = False
        last_det = -1; ref = None
        for i in range(len(keep)):
            if i in set_det:
                last_det = i; ref = sd[i]
            elif keep[i] and ref is not None and sd[i] > P["rho_bridge"] * max(ref, P["theta_off"]):
                j = i
                while j < len(keep) and j not in set_det:
                    keep[j] = False; j += 1
                ref = None
        on = np.where(keep)[0]
        if not len(on):
            continue
        ep = np.r_[0, np.cumsum(np.diff(on) > max_gap)]
        ep_of = dict(zip(on, ep))
        new = {}
        for e_ in np.unique(ep):
            ii = on[ep == e_]; tid = int(sid) * 1000 + int(e_)
            nd = int(((det_i >= ii.min()) & (det_i <= ii.max())).sum())
            ext[tid] = (int(fr[ii.min()]), int(fr[ii.max()]), int(len(ii) - nd))
        T.loc[g.index, "tid"] = [int(sid) * 1000 + int(ep_of.get(idx.get(int(f), -1), 0)) for f in g.frame]
    return T, ext


MERGED_INFO = []


def merge_sites(T, M, max_gap):
    from math import acos, sqrt, pi
    kof = {int(s): k for k, s in enumerate(M["tids"])}
    st = T.groupby("site").agg(cy=("cy", "median"), cx=("cx", "median"), area=("area", "median"),
                               f0=("frame", "min"), f1=("frame", "max"))
    st["r"] = np.sqrt(st.area.clip(lower=6) / np.pi)
    W_ = T.groupby("site").I_sum.sum()
    fset = {s: set(g.frame) for s, g in T.groupby("site")}
    ell = {s: (float(g.major.median()), float(g.minor.median()), float(g.orient.median())) for s, g in T.groupby("site")} \
        if {"major", "minor", "orient"} <= set(T.columns) else {s: (0.0, 0.0, 0.0) for s in st.index}
    lg = {s: np.log(np.clip(np.asarray(M["I"][k], float), 1, None)) for s, k in kof.items()}
    dyn = {s: float(np.percentile(np.asarray(M["I"][k], float), 95) / max(np.percentile(np.asarray(M["I"][k], float), 5), 1.0)) >= P["rho_dyn"]
           for s, k in kof.items()}

    def circ_overlap(r1, r2, d):
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return pi * min(r1, r2) ** 2
        a1 = r1 * r1 * acos((d * d + r1 * r1 - r2 * r2) / (2 * d * r1)); a2 = r2 * r2 * acos((d * d + r2 * r2 - r1 * r1) / (2 * d * r2))
        return a1 + a2 - 0.5 * sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))

    ids = list(st.index); parent = {s: s for s in ids}
    def root(s):
        while parent[s] != s:
            s = parent[s]
        return s
    merged = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if i not in lg or j not in lg:
                continue
            ri, rj = st.r[i], st.r[j]; d = float(np.hypot(st.cy[i] - st.cy[j], st.cx[i] - st.cx[j]))
            big, small = (i, j) if st.area[i] >= st.area[j] else (j, i)
            contain = circ_overlap(ri, rj, d) / (pi * min(ri, rj) ** 2)
            adjacent = (d <= ri + rj + P["merge_adj"] * max(ri, rj)) and st.area[big] >= P["A_sap"] / 2
            if dyn[i] and dyn[j] and d <= ri + rj + max(ri, rj):
                fs, fb = fset[small], fset[big]
                sh, lo = (small, big) if st.f1[small] - st.f0[small] <= st.f1[big] - st.f0[big] else (big, small)
                o = ell[big]; dy, dx = st.cy[small] - st.cy[big], st.cx[small] - st.cx[big]
                u = dy * np.cos(o[2]) + dx * np.sin(o[2]); v = -dy * np.sin(o[2]) + dx * np.cos(o[2])
                inside = (u / max(o[0] / 2, 1)) ** 2 + (v / max(o[1] / 2, 1)) ** 2 <= 1.0
                within = st.f0[sh] >= st.f0[lo] - max_gap and st.f1[sh] <= st.f1[lo] + max_gap
                if inside and within and len(fs & fb) <= 0.05 * min(len(fs), len(fb)):
                    keep_, drop_ = (i, j) if W_[i] >= W_[j] else (j, i)
                    ra, rb = root(keep_), root(drop_)
                    if ra != rb:
                        parent[rb] = ra; merged.append((int(drop_), int(keep_), round(d, 1), round(contain, 2), "turns"))
                        MERGED_INFO.append(dict(drop=int(drop_), keep=int(keep_), cy=float(st.cy[drop_]), cx=float(st.cx[drop_]), area=float(st.area[drop_])))
                    continue
            if contain < P["merge_contain"] and not adjacent:
                continue
            ov = min(st.f1[i], st.f1[j]) - max(st.f0[i], st.f0[j])
            if dyn[i] != dyn[j]:
                continue
            if not dyn[i] and not dyn[j]:
                ov = max(ov, 1); co_ok = True
            else:
                co_ok = ov > 0 and ov >= P["merge_coalive"] * min(st.f1[i] - st.f0[i], st.f1[j] - st.f0[j])
            if not co_ok:
                continue
            if dyn[i] and contain < P["merge_contain"] and ov < P["merge_coalive_long"] * max(st.f1[i] - st.f0[i], st.f1[j] - st.f0[j]):
                continue
            if dyn[i] != dyn[j]:
                continue
            r = float(np.corrcoef(lg[i], lg[j])[0, 1])
            if dyn[i] and r < P["merge_corr"]:
                continue
            keep_, drop_ = (i, j) if W_[i] >= W_[j] else (j, i)
            ra, rb = root(keep_), root(drop_)
            if ra != rb:
                parent[rb] = ra; merged.append((int(drop_), int(keep_), round(d, 1), round(contain, 2), round(r, 3)))
                MERGED_INFO.append(dict(drop=int(drop_), keep=int(keep_), cy=float(st.cy[drop_]), cx=float(st.cx[drop_]), area=float(st.area[drop_])))
    if not merged:
        return T, []
    T = T.copy(); T["site"] = T.site.map(lambda s: root(s))
    roots = {root(b) for _, b, *_ in merged} | {root(a) for a, *_ in merged}
    T["merged"] = T.site.isin(roots)
    T = T.sort_values("I_sum", ascending=False).drop_duplicates(["site", "frame"]).sort_values(["site", "frame"])
    fr = np.sort(T.frame.unique()); idx = {f: i for i, f in enumerate(fr)}
    si = T.frame.map(idx)
    gap = si.groupby(T.site).diff().fillna(0) > max_gap
    T["tid"] = T.site.astype(np.int64) * 1000 + gap.groupby(T.site).cumsum().astype(np.int64)
    return T.sort_values("frame").reset_index(drop=True), merged


def drop_union_masks(case, U, T):
    st = U[U.phase.isin(["delta", "gamma"]) & U.site.notna()].groupby("site").agg(
        cy=("cy", "median"), cx=("cx", "median"), ph=("phase", "first"), f0=("ignite", "min"), f1=("extinguish", "max"))
    ell = {s: (float(g.major.median()), float(g.minor.median()), float(g.orient.median())) for s, g in T[T.site.isin(st.index)].groupby("site")}
    drop = []
    for a in st.index:
        if a not in ell:
            continue
        M, m, o = ell[a]; inside = []
        for b in st.index:
            if b == a or st.ph[b] != st.ph[a]:
                continue
            dy, dx = st.cy[b] - st.cy[a], st.cx[b] - st.cx[a]
            u = dy * np.cos(o) + dx * np.sin(o); v = -dy * np.sin(o) + dx * np.cos(o)
            if (u / max(M / 2, 1)) ** 2 + (v / max(m / 2, 1)) ** 2 <= 1 and (st.f1[b] - st.f0[b]) > (st.f1[a] - st.f0[a]):
                inside.append(int(b))
        if len(inside) >= 2:
            drop.append((int(a), inside))
    if drop:
        U = U.copy()
        for a, ins in drop:
            mm = U.site == a
            U.loc[mm, "phase"] = "union"; U.loc[mm, "evidence"] = "R-union: SAM mask over sites " + ",".join(map(str, ins))
        print(f"  {case} union masks removed (site -> contained sites): {drop}", flush=True)
    return U


def merge_delta_growth(case, U, T, tsec):
    D = U[(U.phase == "delta") & U.site.notna()]
    if not len(D):
        return U
    st = D.groupby("site").agg(cy=("cy", "median"), cx=("cx", "median"), area=("area", "median"), f0=("ignite", "min"), f1=("extinguish", "max"))
    ell = {}
    for s, g in T[T.site.isin(st.index)].groupby("site"):
        ell[s] = (float(g.major.median()), float(g.minor.median()), float(g.orient.median())) if "major" in g else (0.0, 0.0, 0.0)
    parent = {s: s for s in st.index}
    def root(s):
        while parent[s] != s:
            s = parent[s]
        return s
    for a in st.sort_values("f0").index:
        for b in st.index:
            if a == b or st.area[a] < st.area[b] or st.f0[b] <= st.f0[a] or a not in ell:
                continue
            M, m, o = ell[a]; dy, dx = st.cy[b] - st.cy[a], st.cx[b] - st.cx[a]; rb = np.sqrt(st.area[b] / np.pi)
            u = dy * np.cos(o) + dx * np.sin(o); v = -dy * np.sin(o) + dx * np.cos(o)
            if (u / (M / 2 + rb + 3)) ** 2 + (v / (m / 2 + rb + 3)) ** 2 > 1:
                continue
            if abs(tsec[int(st.f1[a])] - tsec[int(st.f1[b])]) > 1.0:
                continue
            ra, rb_ = root(a), root(b)
            if ra != rb_:
                keep, drop = (ra, rb_) if st.f0[ra] <= st.f0[rb_] else (rb_, ra)
                parent[drop] = keep
    moves = {s: root(s) for s in st.index if root(s) != s}
    if not moves:
        return U
    U = U.copy(); m_ = U.site.isin(list(moves)) & (U.phase == "delta")
    U.loc[m_, "evidence"] = U.loc[m_, "evidence"].astype(str) + " R-δ-grow (merged into δ site " + U.loc[m_, "site"].map(lambda s: str(int(moves[s]))) + ")"
    U.loc[m_, "site"] = U.loc[m_, "site"].map(moves)
    fn = f"rk_out/{case}_site_merges.json"
    MG = json.load(open(fn)) if os.path.exists(fn) else []
    for s, k in moves.items():
        MG.append(dict(drop=int(s), keep=int(k), cy=float(st.cy[s]), cx=float(st.cx[s]), area=float(st.area[s]), rule="δ-grow"))
    json.dump(MG, open(fn, "w"))
    print(f"  {case} δ growth merge (drop -> keep): {[(int(s), int(k)) for s, k in moves.items()]}", flush=True)
    return U


def run_case(case, G):
    det = pd.read_csv(f"track_det_{case}.csv").sort_values("frame").reset_index(drop=True)
    if G and G.get("window"):
        det = det[(det.frame >= G["window"][0]) & (det.frame <= G["window"][1])].reset_index(drop=True)
    frames = np.array(sorted(det.frame.unique()), dtype=int)
    tf = frame_times(case); tsec = {i: float(tf[i]) for i in range(len(tf))}
    thermal = temperature_on_frames(case)
    det = det.assign(det_id=np.arange(len(det)))

    built = {}
    for arm in ARMS:
        step = float(np.median(np.diff(np.sort(det.frame.unique()))))
        fps_ = 1.0 / float(np.median(np.diff(tf)))
        g_ = max(1, int(round(P["gap_s"] * fps_ / step))); m_ = max(2, int(round(P["min_obs_s"] * fps_ / step)))
        if P.get("gap_mode", "samples") == "samples":
            g_, m_ = 6, P["min_obs"]
        kw = dict(memory=g_, min_obs=m_) if arm == "trackpy" else dict(max_gap=g_, min_obs=m_)
        T = TRACKERS["site" if arm.startswith("site") else arm](det, **kw)
        if arm == "site_full":
            U, series, M = units_continuous(case, T, frames)
            tag = "v6"
            if P.get("site_merge", False):
                MERGED_INFO.clear()
                T, merged = merge_sites(T, M, g_)
                os.makedirs("rk_out", exist_ok=True)
                json.dump(MERGED_INFO, open(f"rk_out/{case}_site_merges.json", "w"))
                if merged:
                    print(f"  {case} site merge (small -> large, dist, overlap, corr): {merged}", flush=True)
                    tag = "v6m"; U, series, M = units_continuous(case, T, frames, tag=tag)
            if P.get("gap_fill", False) and "SD" in M:
                T, ext = gap_fill(T, M, g_)
                U, series, M = units_continuous(case, T, frames, tag=tag)
                for i in U.index:
                    e0 = ext.get(int(U.at[i, "uid"]))
                    if e0:
                        U.at[i, "ignite"], U.at[i, "extinguish"] = min(int(U.at[i, "ignite"]), e0[0]), max(int(U.at[i, "extinguish"]), e0[1])
                U["filled"] = U.uid.map(lambda u: ext.get(int(u), (0, 0, 0))[2])
        else:
            U, series = units_detected(T, frames); M = None
        built[arm] = (T, U, series, M)

    T_s, U_s, ser_s, M_s = built["site_full"]
    mask = np.load(f"mask_{case}.npy")
    U0, s0_ = run_rules(U_s, tsec, thermal, None, mask)
    sap_uids = set(U0.site[U0.phase == "sapphire"].dropna().astype(int)) if "site" in U0 else set(U0.uid[U0.phase == "sapphire"].astype(int))
    D0 = U0[(U0.phase == "delta") & ~U0.present_start]
    win_ = None
    if s0_.get("onset_frame") is not None:
        fps_w = 1.0 / float(np.median(np.diff(tf))); Wc = P["W_fast"] if s0_.get("burst") == "fast" else P["W_slow"]
        on_ = int(s0_["onset_frame"]); win_ = (on_ - int(round((P["D_co"] + 1.0) * fps_w)), on_ + int(round(Wc * fps_w)))
    children, ch_series = rsd_children(M_s, sap_uids, frames, thermal, tsec, f_min=int(D0.ignite.min()) if len(D0) else None, win=win_)
    if not P.get("rsd_on", False):
        children, ch_series = children.iloc[:0], {}

    res = {}
    for arm in ARMS:
        T, U, series, M = built[arm]
        series = dict(series); series.update(ch_series)
        site_S = None
        if arm == "site_full" and M is not None:
            site_S = {int(s): (np.asarray(M["frames"]), np.asarray(M["S"][k])) for k, s in enumerate(M["tids"])}
            site_I = {int(s): (np.asarray(M["frames"]), np.asarray(M["I"][k])) for k, s in enumerate(M["tids"])}
            site_SD = {int(s): (np.asarray(M["frames"]), np.asarray(M["SD"][k])) for k, s in enumerate(M["tids"])} if "SD" in M else None
        else:
            site_I = site_SD = None
        U2, summ = run_rules(U, tsec, thermal, children if len(children) else None, mask, site_S, site_I, site_SD=site_SD)
        if arm == "site_full" and P.get("delta_growth_merge", True):
            U2 = merge_delta_growth(case, U2, T, tsec)
        if arm == "site_full":
            U2 = drop_union_masks(case, U2, T)
        kin = kinetics(U2, series, frames, tsec, summ["onset_frame"])
        r = dict(summary=summ, kin={k: v for k, v in kin.items() if k in ("delta", "gamma")})
        os.makedirs("rk_out", exist_ok=True)
        U2.to_csv(f"rk_out/{case}_{arm}_units.csv", index=False)
        T.to_csv(f"rk_out/{case}_{arm}_obs.csv", index=False)
        if "curve_t" in kin:
            json.dump(dict(t=kin["curve_t"], Xd=kin["Xd"], Xg=kin["Xg"]), open(f"rk_out/{case}_{arm}_kin.json", "w"))
        res[arm] = r
        print(f"  {case:9s} {arm:10s} onset {summ['onset_frame']} ({summ['burst']}, n={summ['n_burst']}) "
              f"case {summ['case']}  δ{summ['n_delta']} γ{summ['n_gamma']} sap{summ['n_sapphire']} st{summ['n_static']}"
              + f" | t0.5δ {kin['delta'][0]:.2f} n {kin['delta'][1]:.2f}", flush=True)
    res["_children"] = children.drop(columns=[c for c in children.columns if c in ()], errors="ignore").to_dict("records") if len(children) else []
    res["_thermal"] = dict(synced=thermal["synced"], events=thermal.get("events", []), reason=thermal["reason"])
    return res


if __name__ == "__main__":
    RUNS = json.load(open("runs.json", encoding="utf-8"))
    cases = sys.argv[1:] or list(RUNS)
    out = json.load(open("rk_results.json", encoding="utf-8")) if os.path.exists("rk_results.json") else {}
    for c in cases:
        out[c] = run_case(c, RUNS.get(c))
        json.dump(out, open("rk_results.json", "w"), default=lambda o: None if o is None else (float(o) if np.isscalar(o) else str(o)), indent=1)
