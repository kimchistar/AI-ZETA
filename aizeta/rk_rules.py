import numpy as np, pandas as pd
from rk_params import P
from rk_thermal import max_rise, cooling_arrest, scan_recalescence


def run_rules(U, tsec, thermal=None, children=None, mask=None, site_S=None, site_I=None, site_SD=None):
    U = U.copy()
    if children is not None and len(children):
        U = pd.concat([U, children], ignore_index=True)
    for col in ("phase", "flag", "evidence"):
        U[col] = ""
    U["confidence"] = "high"
    t = lambda f: tsec[int(f)]
    f_start = int(U.ignite.min()) if len(U) else 0
    U["present_start"] = U.present_start | (U.ignite.map(t) <= t(f_start) + 1.0)
    U["edge"] = False
    if mask is not None:
        from scipy import ndimage as ndi
        dist = ndi.distance_transform_edt(mask)
        for i, r in U.iterrows():
            y, x = int(np.clip(round(r.cy), 0, mask.shape[0] - 1)), int(np.clip(round(r.cx), 0, mask.shape[1] - 1))
            U.loc[i, "edge"] = dist[y, x] <= P["m_edge"] + 0.5 * np.sqrt(max(r.area, 1) / np.pi)

    flat = U.flat <= P["eps_flat"] * 3
    through = U.present_start & U.alive_end
    is_parent_ok = U.get("parent", pd.Series([None] * len(U))).isna()
    sap = through & flat & (U.area >= P["A_sap"]) & (U.ecc <= P["e_sap_bar"]) & is_parent_ok
    stat = through & flat & ~sap & is_parent_ok
    if "site" in U.columns:
        f_end = int(U.extinguish.max())
        for s_id, g in U[U.site.notna() & is_parent_ok].groupby("site"):
            cov = (g.extinguish - g.ignite).sum() / max(t(f_end) - t(f_start), 1e-9) *                   (t(f_end) - t(f_start)) / max(f_end - f_start, 1)
            span = g.present_start.any() and g.extinguish.max() >= f_end - 2
            if span and cov >= 0.5 and g.area.median() >= P["A_sap"] and g.ecc.median() <= P["e_sap_bar"]                     and (g.flat.median() <= P["eps_flat"] * 3):
                sap.loc[g.index] = True
    sap = sap | ((U.area >= P["A_sap"]) & (U.ecc <= P["e_sap_bar"]) & (U.flat <= P["cv_sap_flat"]) & is_parent_ok)
    U.loc[sap, "phase"] = "sapphire"; U.loc[sap, "evidence"] = "R-SAP-1"
    U.loc[stat, "phase"] = "static"; U.loc[stat, "evidence"] = "R-SAP-3"
    diffuse = (U.area >= 2 * P["A_sap"]) & (U.ecc > P["e_sap"]) & (U.phase == "") & is_parent_ok & U.present_start
    U.loc[diffuse, "phase"] = "static"; U.loc[diffuse, "evidence"] = "R-BG-diffuse"
    big = U[U.phase.isin(["sapphire", "static"])]
    if len(big):
        bc = big[["cy", "cx"]].values; br = np.sqrt(big.area.values / np.pi)
        for i in U.index[(U.phase == "") & is_parent_ok]:
            d = np.hypot(bc[:, 0] - U.at[i, "cy"], bc[:, 1] - U.at[i, "cx"])
            inside = (d <= br) & (big.area.values > U.at[i, "area"])
            if inside.any():
                U.loc[i, "phase"] = big.phase.values[np.argmax(inside)]
                U.loc[i, "evidence"] = "R-SAP-frag"
    if "site" in U.columns:
        f_lo, f_hi = int(U.ignite.min()), int(U.extinguish.max())
        span_all = max(f_hi - f_lo, 1)
        for s_id, g in U[U.site.notna() & is_parent_ok & (U.phase == "")].groupby("site"):
            cov = float((g.extinguish - g.ignite).sum()) / span_all
            if cov >= P["cov_bg"] and g.flat.median() <= P["cv_bg"]:
                U.loc[g.index, "phase"] = "static"; U.loc[g.index, "evidence"] = "R-BG2-persistent"
    if P["assume_liquid_start"]:
        if "site" in U.columns:
            s0 = set(U.site[U.present_start & U.site.notna()].astype(int))
            m = U.site.isin(s0) & (U.phase == "") & is_parent_ok
        else:
            m = U.present_start & (U.phase == "") & is_parent_ok
        U.loc[m, "phase"] = "sapphire"; U.loc[m, "evidence"] = "R-SAP-start"
    persist_sites = set()
    if site_S is not None and "site" in U.columns:
        f_lo_w, f_hi_w = int(U.ignite.min()), int(U.extinguish.max())
        first_det = U[U.site.notna()].groupby("site").ignite.min()
        for s_id in U.site.dropna().unique():
            v = site_S.get(int(s_id))
            if v is None:
                continue
            if t(first_det[s_id]) - t(f_lo_w) > 0.05 * (t(f_hi_w) - t(f_lo_w)):
                continue
            lit = v[1] >= P["theta_on"]
            n0 = max(3, int(0.05 * len(lit)))
            if lit[:n0].mean() >= 0.8 and lit.mean() >= P["persist_frac"]:
                persist_sites.add(int(s_id))
        m = U.site.isin(persist_sites) & is_parent_ok & (U.phase == "")
        U.loc[m, "phase"] = "static"; U.loc[m, "evidence"] = "R-PERSIST-continuous"
    if thermal is not None and thermal.get("synced") and thermal.get("T") is not None:
        Tt = np.asarray(thermal["T"], float); fmax = int(np.nanargmax(Tt))
        after = np.where(Tt[fmax:] < Tt[fmax] - P["dT_liq"])[0]
        fmax = fmax + (int(after[0]) if len(after) else 0)
        if "site" in U.columns:
            fs = U[U.site.notna()].groupby("site").ignite.min()
            early = set(fs[fs <= fmax].index.astype(int))
            m = U.site.isin(early) & (U.phase == "") & is_parent_ok
        else:
            m = (U.ignite <= fmax) & (U.phase == "") & is_parent_ok
        U.loc[m, "phase"] = "sapphire"; U.loc[m, "evidence"] = "R-SAP-liquid"
    base = U[U.evidence.isin(["R-SAP-start", "R-SAP-liquid"])]
    if len(base):
        bc = base[["cy", "cx"]].values; ba = base.area.values; br = np.sqrt(ba / np.pi)
        for i in U.index[(U.phase == "") & is_parent_ok]:
            ri = np.sqrt(max(U.at[i, "area"], 1) / np.pi)
            d = np.hypot(bc[:, 0] - U.at[i, "cy"], bc[:, 1] - U.at[i, "cx"])
            same = (d <= 0.8 * (br + ri)) & (U.at[i, "area"] >= 0.5 * ba)
            if same.any():
                if site_I is not None and "site" in U.columns and int(U.at[i, "site"]) in site_I:
                    fr_s, I_s = site_I[int(U.at[i, "site"])]
                    w = (np.asarray(fr_s) >= U.at[i, "ignite"]) & (np.asarray(fr_s) <= U.at[i, "extinguish"])
                    if w.sum() >= 5:
                        v = np.asarray(I_s, float)[w]
                        if np.percentile(v, 95) / max(np.percentile(v, 5), 1.0) >= P["rho_dyn"]:
                            continue
                U.loc[i, "phase"] = "sapphire"; U.loc[i, "evidence"] = "R-SAP-redetect"
    if "site" in U.columns:
        for s_id, g in U[U.site.notna() & (U.phase == "") & is_parent_ok].groupby("site"):
            if g.area.median() >= 2 * P["A_sap"] and g.ecc.median() > P["e_sap"] and g.flat.median() <= P["cv_sap_flat"]:
                if not g.present_start.any():
                    s_int = int(s_id)
                    if site_I is None or s_int not in site_I:
                        continue
                    fr_i, I_i = site_I[s_int]
                    sel = np.asarray(fr_i) >= int(g.ignite.min())
                    Ii = pd.Series(np.asarray(I_i, float)[sel]).rolling(5, center=True, min_periods=1).median().values
                    if len(Ii) >= 10 and np.percentile(Ii, 95) >= P["rho_dyn"] * max(np.percentile(Ii, 5), 1.0):
                        U.loc[g.index, "evidence"] = "R-BG-dyn: brightness changes -> not background"
                        continue
                U.loc[g.index, "phase"] = "static"; U.loc[g.index, "evidence"] = "R-BG-diffuse-site"
    if site_SD is not None and "site" in U.columns and P.get("bg_step", True):
        for i in U.index[(U.phase == "") & ~U.present_start & U.site.notna()]:
            sid = int(U.at[i, "site"])
            if sid not in site_SD:
                continue
            fr_s, sd = site_SD[sid]; tt = np.array([t(int(f)) for f in fr_s])
            ta, tb = t(int(U.at[i, "ignite"])), t(int(U.at[i, "extinguish"]))
            prev = U[(U.site == U.at[i, "site"]) & (U.extinguish < U.at[i, "ignite"])]
            p0 = max(ta - 3.0, (t(int(prev.extinguish.max())) + 0.3) if len(prev) else -1e9)
            pre = sd[(tt >= p0) & (tt <= ta - 0.3)]; dur = sd[(tt >= ta) & (tt <= min(tb, ta + 5.0))]
            if len(pre) < 2 or len(dur) < 1 or (ta - 0.3) - p0 < 0.5:
                continue
            if np.median(dur) - np.median(pre) < P["dS_new"]:
                life = sd[(tt >= ta) & (tt <= tb)]
                if len(life) >= 5:
                    ls = pd.Series(life).rolling(5, center=True, min_periods=1).median().values
                    if np.percentile(ls, 95) >= P["rho_dyn"] * max(np.percentile(ls, 5), 1.0):
                        continue
                U.loc[i, "phase"] = "static"; U.loc[i, "evidence"] = "R-BG-step: no brightness rise at appearance, flat"
    edge_bg = U.edge & through & (U.phase == "")
    U.loc[edge_bg, "phase"] = "static"; U.loc[edge_bg, "evidence"] = "Q4-edge"
    U.loc[U.edge & (U.phase == ""), "flag"] = "AMB_EDGE"

    C = U[U.phase == ""]
    ign = C[~C.present_start & ~C.edge].sort_values("ignite")
    if "site" in ign.columns and ign.site.notna().all():
        ign = ign.drop_duplicates("site", keep="first")
    ig_t = np.array([t(f) for f in ign.ignite])

    burst_f, burst, n_burst, T_split = None, None, 0, np.inf
    for i in range(len(ig_t)):
        w = (ig_t >= ig_t[i]) & (ig_t <= ig_t[i] + P["W_slow"])
        if int(w.sum()) >= P["K_gamma"]:
            k = np.where(w)[0][P["K_gamma"] - 1]
            burst_f = int(ign.ignite.iloc[k]); n_burst = int(w.sum()); T_split = ig_t[i]
            n_fast = int(((ig_t >= ig_t[i]) & (ig_t <= ig_t[i] + P["W_fast"])).sum())
            burst = "fast" if n_fast >= P["K_gamma"] else "slow"
            break
    cand = C[~C.edge]
    if "site" in cand.columns and cand.site.notna().all():
        n0 = int(cand[(cand.ignite <= f_start) & (cand.extinguish >= f_start)].site.nunique())
    else:
        n0 = int(((cand.ignite <= f_start) & (cand.extinguish >= f_start)).sum())
    censored = n0 >= P["K_gamma"]
    if censored:
        burst_f, burst, n_burst, T_split = f_start, "state", n0, t(f_start)
    therm_onset = None
    if __import__("os").environ.get("RK_DEBUG"):
        print("THERM-ONSET pre", "censored", censored, "n_ig", len(ig_t), "T_split", T_split, "synced", thermal is not None and thermal.get("synced"), flush=True)
    if (not censored and thermal is not None and thermal.get("synced") and thermal.get("T") is not None
            and len(ig_t) and np.isfinite(T_split)):
        Tt_ = np.asarray(thermal["T"], float); fr_ = np.array(sorted(tsec))
        fps_r = 1.0 / np.median(np.diff([tsec[k] for k in fr_])) * np.median(np.diff(fr_))
        for a_, b_, r_ in scan_recalescence(Tt_, fps_r, P["dT_rec_min"]):
            if not (fr_[0] <= a_ <= fr_[-1]):
                continue
            pre_i = np.arange(max(int(a_ - 10 * fps_r), 0), max(int(a_ - fps_r), 1))
            ok_ = np.isfinite(Tt_[pre_i])
            if ok_.sum() < 3 or np.polyfit(pre_i[ok_] / fps_r, Tt_[pre_i][ok_], 1)[0] >= 0:
                continue
            ta = t(int(fr_[np.argmin(np.abs(fr_ - a_))]))
            win = (ig_t >= ta - P["rec_img_before"]) & (ig_t <= ta + P["rec_img_after"])
            if __import__("os").environ.get("RK_DEBUG"):
                print("R-THERM-onset dbg", int(a_), round(ta, 2), "burst", int(win.sum()), "T_split", round(T_split, 2), flush=True)
            if int(win.sum()) < P["K_gamma"] or ig_t[win].min() <= T_split + P["W_slow"]:
                continue
            first_b = float(ig_t[win].min())
            crowd = ign[(ig_t >= T_split) & (ig_t < first_b)]
            if "site" in U.columns and crowd.site.notna().all():
                last_ext = U.groupby("site").extinguish.max()
                crowd = crowd.assign(extinguish=crowd.site.map(last_ext).values)
            crowd = crowd[crowd.extinguish.map(t) >= first_b - 3.0]
            if not len(crowd):
                continue
            die = (np.abs(crowd.extinguish.map(t) - first_b) <= 3.0)
            if site_SD is not None and "site" in crowd.columns:
                dz = []
                for sid in crowd.site:
                    if pd.isna(sid) or int(sid) not in site_SD:
                        dz.append(False); continue
                    fr_s, sd = site_SD[int(sid)]; tt = np.array([t(int(f)) for f in fr_s])
                    b4 = sd[(tt >= first_b - 2.0) & (tt <= first_b - 0.5)]; af = sd[(tt >= first_b + 0.5) & (tt <= first_b + 2.0)]
                    dz.append(bool(len(b4) and len(af) and np.median(af) < 0.5 * np.median(b4)))
                die = die | np.array(dz)
            if __import__("os").environ.get("RK_DEBUG"):
                print("  crowd", len(crowd), "die", int(die.sum()), flush=True)
            if die.mean() >= P["rec_die_frac"]:
                therm_onset = dict(frame=int(a_), rise=round(float(r_), 2), crowd_was=int(burst_f), n_crowd=int(len(crowd)),
                                   crowd_die=int(die.sum()))
                k = np.where(win)[0][P["K_gamma"] - 1]
                burst_f = int(ign.ignite.iloc[k]); n_burst = int(win.sum()); T_split = first_b; burst = "fast"
                break
    onset_f = burst_f
    T0 = T_split

    if np.isfinite(T0):
        weak = U.evidence.isin(["R-SAP-start", "R-SAP-liquid", "R-SAP-redetect"])
        key = "site" if "site" in U.columns else None
        groups = U[weak].groupby(key) if key else [(i, U.loc[[i]]) for i in U.index[weak]]
        f_endw = int(U.extinguish.max())
        for _, g in groups:
            g = g.sort_values("ignite")
            dur = g.extinguish.map(t) - g.ignite.map(t)
            hit = g[(np.abs(g.extinguish.map(t) - T0) <= 3.0) & (dur >= 10.0) & (g.extinguish < f_endw - 2)]
            if len(hit) and site_S is not None and key and int(g.site.iloc[0]) in site_S:
                fr_s, S_s = site_S[int(g.site.iloc[0])]
                tt = np.array([t(f) for f in fr_s])
                post = (tt >= T0 + 1.0) & (tt <= T0 + 10.0)
                if post.any() and (S_s[post] >= P["theta_on"]).mean() >= 0.3:
                    hit = hit.iloc[:0]
            if len(hit):
                cut = int(hit.extinguish.max())
                upto = g[g.extinguish <= cut]
                U.loc[upto.index, "phase"] = "delta"; U.loc[upto.index, "evidence"] = "δ-vanish-at-onset"
                later = g[g.ignite > cut]
                U.loc[later.index, "phase"] = ""; U.loc[later.index, "evidence"] = ""
    C = U[U.phase == ""]
    pre = (C.present_start) | (C.ignite.map(t) < T0 - P["D_co"])
    if censored:
        pre = pre & (C.extinguish.map(t) < T0)
    post = ~pre
    U.loc[C.index[pre], "phase"] = "delta"
    U.loc[C.index[post], "phase"] = "gamma"
    U.loc[C.index[pre], "evidence"] += " P0/P2"
    if "site" in U.columns and np.isfinite(T0) and "I_med" in U.columns:
        for s_id, g in U[U.site.notna() & U.phase.isin(["delta", "gamma"])].groupby("site"):
            g = g.sort_values("ignite")
            b4 = g[(g.phase == "delta") & (g.ignite.map(t) < T0)]
            af = g[g.ignite.map(t) >= T0 - P["D_co"]]
            if not len(b4) or not len(af):
                continue
            last, first = b4.iloc[-1], af.iloc[0]
            if t(first.ignite) - t(last.extinguish) > P["bridge_s"]:
                continue
            r = first.I_med / max(last.I_med, 1e-9)
            if r <= P["rho_bridge"]:
                U.loc[af.index, "phase"] = "delta"; U.loc[af.index, "evidence"] = "R-bridge (same δ)"
    if site_I is not None and "site" in U.columns:
        for s_id, g in U[U.site.notna() & (U.phase == "delta")].groupby("site"):
            v = site_I.get(int(s_id))
            if v is None:
                continue
            fr_i, I_i = v
            I_i = pd.Series(I_i).rolling(5, center=True, min_periods=1).median().values
            f_a, f_b = int(g.ignite.min()), int(g.extinguish.max())
            life = (fr_i >= f_a) & (fr_i <= f_b); prev = fr_i < f_a
            if life.sum() < 3:
                continue
            base = np.median(I_i[prev]) if prev.sum() >= 5 else np.percentile(I_i[life], 10)
            peak = np.percentile(I_i[life], 90)
            if peak / max(base, 1e-9) < P["rho_grow"] or peak < P["I_grow_min"]:
                U.loc[g.index, "phase"] = "static"; U.loc[g.index, "evidence"] = "R-DL-flat (no growth)"
    if "site" in U.columns and np.isfinite(T0) and "I_med" in U.columns:
        bgev = ("R-SAP-liquid", "R-SAP-start", "R-SAP-redetect", "R-PERSIST-continuous", "R-DL-flat (no growth)")
        for s_id, g in U[U.site.notna() & U.evidence.isin(bgev)].groupby("site"):
            g_pre = g[g.extinguish.map(t) < T0]
            g_post = g[g.ignite.map(t) >= T0 - P["D_co"]]
            if not len(g_pre) or not len(g_post):
                continue
            ref = float(g_pre.I_med.median())
            hit = g_post[g_post.I_med >= P["rho_q1"] * max(ref, 1e-9)]
            if len(hit):
                later = g_post[g_post.ignite >= hit.ignite.min()]
                U.loc[later.index, "phase"] = "gamma"; U.loc[later.index, "evidence"] = "R-γ-on-background"
    if "site" in U.columns and np.isfinite(T0):
        Dv = U[(U.phase == "delta") & (U.ignite.map(t) < T0) & (U.extinguish.map(t) >= T0 - 3.0)]
        t_ends = [T0] + [t(int(f)) for f in Dv.groupby("site").extinguish.max()] if "site" in Dv else [T0]
        f_endw = int(U.extinguish.max())
        for s_id, g in U[U.site.notna() & U.evidence.str.startswith("R-BG-step")].groupby("site"):
            if g.present_start.any() or (U[(U.site == s_id)].phase == "sapphire").any():
                continue
            ta, te = t(int(g.ignite.min())), t(int(g.extinguish.max()))
            if ta >= T0 or te - ta < P["covanish_life_s"] or int(g.extinguish.max()) >= f_endw - 2:
                continue
            if min(abs(te - x) for x in t_ends) > P["covanish_s"]:
                continue
            if site_SD is not None and int(s_id) in site_SD:
                fr_s, sd = site_SD[int(s_id)]; tt = np.array([t(int(f)) for f in fr_s])
                b4 = sd[(tt >= te - 2.0) & (tt <= te - 0.3)]; af = sd[(tt >= te + 0.5) & (tt <= te + 3.0)]
                if not (len(b4) and len(af) and np.median(af) < 0.5 * np.median(b4)):
                    continue
            U.loc[g.index, "phase"] = "delta"; U.loc[g.index, "evidence"] = "R-δ-covanish: vanishes with the δ at the onset"
    if "site" in U.columns and np.isfinite(T0) and burst in ("fast", "slow"):
        Wc = P["W_fast"] if burst == "fast" else P["W_slow"]
        for s_id, g in U[U.site.notna() & (U.evidence == "R-BG-diffuse-site")].groupby("site"):
            if g.present_start.any():
                continue
            ta = t(int(g.ignite.min()))
            if T0 - P["D_co"] <= ta <= T0 + Wc:
                U.loc[g.index, "phase"] = "gamma"; U.loc[g.index, "evidence"] = "R-γ-crowd-diffuse: appears with the γ crowd"
    if site_SD is not None and "site" in U.columns and np.isfinite(T0):
        for i in U.index[U.evidence.astype(str).str.startswith("R-BG-step") & (U.ignite.map(t) >= T0 - P["D_co"]) & ~U.edge & U.site.notna()]:
            sid = int(U.at[i, "site"])
            if sid not in site_SD:
                continue
            fr_s, sd = site_SD[sid]; tt = np.array([t(int(f)) for f in fr_s])
            pre = sd[(tt >= T0 - 3.0) & (tt <= T0 - 0.3)]
            if len(pre) < 2:
                continue
            hi = sd >= np.median(pre) + P["late_rise"]; best, t_on, k_on, k_first = 0.0, None, None, None
            for k in range(len(hi)):
                if tt[k] < T0 - P["D_co"]:
                    continue
                if hi[k]:
                    if t_on is None:
                        t_on, k_on = tt[k], k
                    if tt[k] - t_on >= P["late_hold_s"] and k_first is None:
                        k_first = k_on
                    best = max(best, tt[k] - t_on)
                else:
                    t_on = None
            if best >= P["late_hold_s"]:
                U.loc[i, "phase"] = "gamma"; U.loc[i, "evidence"] = "R-γ-late-detected: brighter than before the γ onset"
                if k_first is not None and int(fr_s[k_first]) < int(U.at[i, "ignite"]):
                    U.loc[i, "ignite"] = int(fr_s[k_first])
    if site_SD is not None and "site" in U.columns and np.isfinite(T0):
        for i in U.index[(U.phase == "gamma") & U.site.notna() & (U.ignite.map(t) > T0)]:
            sid = int(U.at[i, "site"])
            if sid not in site_SD:
                continue
            fr_s, sd = site_SD[sid]; tt = np.array([t(int(f)) for f in fr_s])
            pre = sd[(tt >= T0 - 3.0) & (tt <= T0 - 0.3)]
            if len(pre) < 2:
                continue
            thr = np.median(pre) + P["dS_new"]
            if site_I is not None and sid in site_I:
                fr_s, sd = site_I[sid]; fr_s = np.asarray(fr_s); sd = np.asarray(sd, float); tt = np.array([t(int(f)) for f in fr_s])
                pre = sd[(tt >= T0 - 3.0) & (tt <= T0 - 0.3)]
                post_ = sd[(tt >= T0) & (tt <= T0 + 5.0)]
                if len(pre) < 2 or not len(post_):
                    continue
                thr = np.median(pre) + max(5 * np.std(pre), 0.02 * (post_.max() - np.median(pre)))
            k = int(np.searchsorted(fr_s, int(U.at[i, "ignite"]))) - 1; k_start = None
            while k >= 0 and tt[k] >= T0 - P["D_co"] and sd[k] >= thr:
                k_start = k; k -= 1
            if k_start is not None:
                U.loc[i, "ignite"] = int(fr_s[k_start]); U.loc[i, "evidence"] = str(U.at[i, "evidence"]) + " +start-back"
    if "site" in U.columns:
        Dd = U[U.phase == "delta"]
        life = Dd.extinguish.map(t) - Dd.ignite.map(t)
        sustained = set(Dd.site[life >= P["min_delta_s"]].dropna().astype(int))
        fl = Dd.index[(life < P["min_delta_s"]) & ~Dd.site.isin(sustained)]
        U.loc[fl, "phase"] = "static"; U.loc[fl, "evidence"] = "flicker"
    U.loc[C.index[post], "evidence"] += " A3"

    pre_ig = ig_t[ig_t < T0 - P["D_co"]]
    for x in pre_ig:
        n = int(((pre_ig >= x) & (pre_ig <= x + P["W_fast"])).sum())
        if P["K_delta"] < n < P["K_gamma"]:
            m = U.phase.eq("delta") & U.ignite.map(t).between(x, x + P["W_fast"]) & ~U.present_start
            U.loc[m, "flag"] = "AMB_COUNT_GAP"; U.loc[m, "confidence"] = "low"

    D = U[(U.phase == "delta") & ~U.edge]
    case, rule = None, None
    history, transition = None, None
    if onset_f is not None:
        if len(D) == 0:
            case, rule = "R", "P5"
        else:
            ext_t = np.array([t(f) for f in D.extinguish])
            late = (D.extinguish.map(t) > T0 + P["persist_s"]) | D.alive_end
            fading = (D.fall >= P["rho_fall"]) & (D.flat > P["eps_flat"] * 1.5)
            alive_late = (late & fading).any()
            coincide = (np.abs(ext_t - T0) <= max(P["D_co"], 2.0 / 10)).any() if len(ext_t) else False
            decay_s = None
            if site_I is not None and "site" in D.columns and D.site.notna().any():
                alive = D[(D.ignite.map(t) < T0) & (D.extinguish.map(t) >= T0 - 3.0)]
                sids = [int(x) for x in alive.site.dropna().unique() if int(x) in site_I]
                if sids:
                    fr_s = np.asarray(site_I[sids[0]][0]); tt = np.array([t(int(f)) for f in fr_s])
                    Isum = pd.Series(np.sum([np.asarray(site_I[k][1], float) for k in sids], axis=0)).rolling(3, center=True, min_periods=1).median().values
                    pre = (tt >= T0 - 10.0) & (tt <= T0)
                    if pre.any():
                        ip = np.where(pre)[0][np.argmax(Isum[pre])]; plateau = Isum[ip]
                        tail = tt >= tt.max() - 2.0
                        floor = float(np.median(Isum[tail])) if tail.any() else 0.0
                        X = (Isum - floor) / max(plateau - floor, 1e-9)
                        after = np.arange(len(tt)) >= ip
                        s_i = np.where(after & (X < 0.95))[0]
                        if len(s_i):
                            e_i = np.where((np.arange(len(tt)) > s_i[0]) & (X <= 0.05))[0]
                            decay_s = float(tt[e_i[0]] - tt[s_i[0]]) if len(e_i) else float("inf")
                            coincide = decay_s <= P["fast_s"]; alive_late = not coincide
            if coincide and not alive_late:
                case, rule = 1, "P1"
            elif alive_late:
                case, rule = 4, "P4"
            elif coincide:
                case, rule = 1, "P1"
            else:
                case, rule = "?", "AMB_CASE"
            transition = case
            Dh = D
            if "site" in D.columns and D.site.notna().all():
                gg = D.groupby("site"); Dh = D.loc[gg.ignite.idxmin()].copy()
                Dh["extinguish"] = gg.extinguish.max().reindex(Dh.site).values
            life = Dh.extinguish.map(t) - Dh.ignite.map(t)
            Ds = Dh[(life >= P["min_delta_s"]) & (Dh.ignite.map(t) < T0 - P["D_co"])].sort_values("ignite")
            liquid_start = not bool(((U.phase == "delta") & U.present_start & ~U.edge).any())
            if liquid_start and len(Ds) == 1:
                history = 1
            if liquid_start and 2 <= len(Ds) and (len(Ds) <= 1 + P["K_delta"] or therm_onset is not None):
                first = Ds.iloc[0]
                later = Ds.iloc[1:][~Ds.iloc[1:].present_start]
                if len(later):
                    l0 = later.iloc[0]
                    if t(first.extinguish) >= t(l0.ignite) - P["D_co"]:
                        history = 2
                    elif abs(t(first.extinguish) - t(l0.ignite)) <= 1.0:
                        history = 3

    for i in U.index[U.phase == "delta"]:
        r = U.loc[i]
        okp = bool(r.unimodal) and r.rise >= P["rho_rise"] and r.fall >= P["rho_fall"]
        if okp:
            U.loc[i, "evidence"] += " D3"
        else:
            U.loc[i, "flag"] = (U.loc[i, "flag"] + " DL_INCOMPLETE").strip()
            if U.loc[i, "confidence"] == "high":
                U.loc[i, "confidence"] = "medium"
        if onset_f is not None and r.t_peak is not None and not pd.isna(r.t_peak):
            if t(r.t_peak) > T0 + P["D_co"] and t(r.ignite) > T0:
                U.loc[i, "phase"] = "gamma"; U.loc[i, "evidence"] += " D3-timing"

    U["id"] = ""
    for ph, pref in (("delta", "delta"), ("gamma", "gamma"), ("sapphire", "sap"), ("static", "st")):
        idx = U.index[U.phase == ph]
        order = U.loc[idx].sort_values(["ignite", "cy", "cx"]).index
        for k, i in enumerate(order, 1):
            U.loc[i, "id"] = f"{pref}{k}"
    liquid0 = not bool(((U.phase == "delta") & U.present_start & ~U.edge).any())
    if onset_f is not None and case == "R":
        history = "R"
    tags = [x for x in dict.fromkeys([history, transition]) if x is not None]
    case = history if history in (2, 3, "R") else transition
    summary = dict(therm_onset=therm_onset, delta_decay_s=(locals().get("decay_s")), onset_frame=onset_f, burst=burst, n_burst=n_burst, case=case, rule=rule,
                   case_tags=tags, history=history, transition=transition,
                   gamma_first_frame=None if not np.isfinite(T0) else int(min(tsec, key=lambda k: abs(tsec[k] - T0))),
                   onset_censored=bool(censored),
                   window_start="liquid" if liquid0 else "solid",
                   n_delta=int((U.phase == "delta").sum()), n_gamma=int((U.phase == "gamma").sum()),
                   n_sapphire=int((U.phase == "sapphire").sum()), n_static=int((U.phase == "static").sum()),
                   delta_times=[dict(id=r.id, ignite=int(r.ignite), peak=None if r.t_peak is None or pd.isna(r.t_peak) else int(r.t_peak),
                                     extinguish=int(r.extinguish), confidence=r.confidence)
                                for _, r in U[U.phase == "delta"].iterrows()])
    if thermal is not None and thermal.get("synced") and thermal.get("T") is not None and np.isfinite(T0):
        Tt = np.asarray(thermal["T"], float); fr_all = np.array(sorted(tsec))
        fps_ = 1.0 / np.median(np.diff([tsec[k] for k in fr_all])) * np.median(np.diff(fr_all))
        f0 = int(min(tsec, key=lambda k: abs(tsec[k] - T0)))
        rise, fa, fb = max_rise(Tt, f0 - int(P["rec_before"] * fps_), f0 + int(P["rec_after"] * fps_))
        arrest = cooling_arrest(Tt, f0, fps_)
        mode = "fast" if rise >= P["dT_rec_min"] else "slow"
        stage2 = []
        for a_, b_, r_ in scan_recalescence(Tt, fps_, P["dT_rec_min"]):
            if a_ <= f0 + P["rec_after"] * fps_ or not (fr_all[0] <= a_ <= fr_all[-1]):
                continue
            ta = t(int(fr_all[np.argmin(np.abs(fr_all - a_))]))
            dl = U[(U.phase == "delta") & (U.extinguish.map(lambda f: abs(t(int(f)) - ta) <= P["rec_before"] + 1.0))]
            alive = U[(U.phase == "delta") & (U.ignite.map(lambda f: t(int(f)) < ta))
                      & (U.extinguish.map(lambda f: t(int(f)) >= ta - P["rec_before"]))]
            if len(alive):
                U.loc[dl.index, "evidence"] = U.loc[dl.index, "evidence"].astype(str) + "+R-THERM"
                stage2.append(dict(frame=int(a_), peak_frame=int(b_), rise=round(float(r_), 2),
                                   T=round(float(Tt[a_]), 1), n_delta_consumed=int(len(dl))))
        summary.update(thermal_mode=mode, recalescence_dT=round(rise, 2),
                       recalescence_frame=None if fa is None or rise < P["dT_rec_min"] else int(fa),
                       recalescence_peak_frame=None if fb is None or rise < P["dT_rec_min"] else int(fb),
                       T_onset=round(float(Tt[f0]), 1) if np.isfinite(Tt[f0]) else None,
                       cooling_arrest=None if not np.isfinite(arrest) else round(arrest, 2),
                       massive_stage2=stage2)
    if thermal is not None and thermal.get("synced") and onset_f is not None:
        ev = np.array(thermal["events"])
        if len(ev):
            j = int(np.argmin(np.abs(ev - onset_f)))
            summary["thermal_nearest_frame"] = int(ev[j])
            summary["thermal_offset_s"] = float(t(int(ev[j])) - T0) if int(ev[j]) in tsec else None
    return U, summary
