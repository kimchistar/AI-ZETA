import numpy as np, pandas as pd
from skimage import measure
from rk_params import P
from rk_units import profile_stats


def _liquid_end(thermal, frames):
    if thermal is None or not thermal.get("synced") or thermal.get("T") is None:
        return int(frames[0])
    Tt = np.asarray(thermal["T"], float); fmax = int(np.nanargmax(Tt))
    after = np.where(Tt[fmax:] < Tt[fmax] - P["dT_liq"])[0]
    return fmax + (int(after[0]) if len(after) else 0)


def rsd_children(M, sap_uids, frames, thermal=None, tsec=None, f_min=None, win=None):
    rows, series = [], {}
    keys = list(M["crop_keys"]); frames = np.asarray(frames)
    f_liq = _liquid_end(thermal, frames)
    if f_min is not None:
        f_liq = max(f_liq, int(f_min))
    nb_hi, nb_lo = P["rsd_nb_hi"], P["rsd_nb_lo"]
    for k_site, tid in enumerate(M["tids"]):
        if int(tid) not in sap_uids or k_site not in keys:
            continue
        stack = M["crops"][keys.index(k_site)].astype(np.float32)
        n = len(stack)
        if stack.ndim != 3 or n < nb_hi + P["m_on"] + 2:
            continue
        taken = np.zeros(stack.shape[1:], bool); n_child = 0
        i = max(nb_hi, int(np.searchsorted(frames, f_liq)))
        i_end = n - P["m_on"]
        if win is not None:
            i = max(i, int(np.searchsorted(frames, win[0]))); i_end = min(i_end, int(np.searchsorted(frames, win[1], side="right")))
        while i < i_end:
            B = np.median(stack[i - nb_hi:i - nb_lo], axis=0)
            E = stack[i] - B
            sig = 1.4826 * np.median(np.abs(E - np.median(E))) + 1e-6
            if np.median(E / sig) > 2.0:
                i += 1; continue
            hot = (E / sig >= P["theta_jump"]) & ~taken
            lab = measure.label(hot)
            new = False
            for p in measure.regionprops(lab):
                if p.area < P["rsd_min_px"]:
                    continue
                msk = lab == p.label
                zz = [((stack[j] - B)[msk] / sig).max() for j in range(i, min(i + P["m_on"], n))]
                if min(zz) < P["theta_jump"]:
                    continue
                b_ = max(float(B[msk].mean()), 3 * sig)
                bc_ = max(float(B.mean()), 3 * sig)
                rr = min((float(stack[j][msk].mean()) / b_) / max(float(stack[j].mean()) / bc_, 1e-6)
                         for j in range(i, min(i + P["m_on"], n)))
                j2 = min(i + P["m_on"], n) - 1
                raw_rr = float(stack[j2][msk].mean()) / b_
                pix_rr = float(np.percentile(stack[j2][msk] / np.maximum(B[msk], 3 * sig), 90))
                if rr < P["rsd_rise"]:
                    continue
                zmax = ((stack[i:] - B)[:, msk] / sig).max(axis=1)
                ext, c = None, 0
                for j in range(len(zmax)):
                    c = c + 1 if zmax[j] < P["theta_jump"] / 2 else 0
                    if c >= P["m_off"]:
                        ext = i + j - P["m_off"] + 1; break
                f_e = int(frames[ext]) if ext is not None else int(frames[-1])
                if tsec is not None and tsec[f_e] - tsec[int(frames[i])] < P["rsd_life_s"]:
                    continue
                I = np.clip(stack[:, msk] - B[msk], 0, None).sum(axis=1)
                ps = profile_stats(I)
                cy0 = float(M["cy"][k_site]) - stack.shape[1] // 2 + p.centroid[0]
                cx0 = float(M["cx"][k_site]) - stack.shape[2] // 2 + p.centroid[1]
                uid = int(1e6 + int(tid) * 100 + n_child); n_child += 1
                flag, ev = "", "D4-image"
                if thermal is None or not thermal.get("synced"):
                    flag = "AMB_NO_TEMP"
                else:
                    tf = tsec[int(frames[i])]
                    if [e for e in thermal["events"] if abs(tsec.get(int(e), 1e9) - tf) <= P["D_T"]]:
                        ev += " D4-thermal"
                    else:
                        flag = "SD_NO_THERMAL"
                I_life = I[i:ext if ext is not None else n]
                rows.append(dict(uid=uid, site=uid, cy=cy0, cx=cx0, ignite=int(frames[i]),
                                 extinguish=int(frames[ext]) if ext is not None else int(frames[-1]),
                                 present_start=False, alive_end=ext is None, area=float(p.area),
                                 ecc=float(p.eccentricity), flat=float(np.std(I_life) / max(np.mean(I_life), 1e-9)),
                                 n_obs=int((zmax >= P["theta_jump"]).sum()), max_disp=0.0,
                                 I_med=float(np.median(I_life)) if len(I_life) else 0.0,
                                 t_peak=int(frames[ps["t_pk_idx"]]) if ps["t_pk_idx"] is not None else None,
                                 rise=ps["rise"], fall=ps["fall"], unimodal=ps["unimodal"],
                                 rise_ratio=round(rr, 2), raw_rr=round(raw_rr, 2), pix_rr=round(pix_rr, 2), z_pk=round(float(zmax[:P['m_on']].min()), 1), parent=int(tid), sd_flag=flag, sd_evidence=ev))
                series[uid] = I
                taken |= msk; new = True
            i += 1
    C = pd.DataFrame(rows)
    if len(C):
        C = C.sort_values("ignite").reset_index(drop=True)
        keep = []
        for i_, r in C.iterrows():
            hit = [k for k in keep if C.at[k, "parent"] == r.parent and np.hypot(C.at[k, "cy"] - r.cy, C.at[k, "cx"] - r.cx) <= P["rsd_merge_px"]]
            if hit:
                k = hit[0]; a, b = C.at[k, "area"], r.area
                C.at[k, "cy"] = (C.at[k, "cy"] * a + r.cy * b) / (a + b); C.at[k, "cx"] = (C.at[k, "cx"] * a + r.cx * b) / (a + b)
                C.at[k, "area"] = a + b; C.at[k, "extinguish"] = max(C.at[k, "extinguish"], r.extinguish)
                C.at[k, "alive_end"] = C.at[k, "alive_end"] or r.alive_end
                series[int(C.at[k, "uid"])] = series[int(C.at[k, "uid"])] + series.pop(int(r.uid))
            else:
                keep.append(i_)
        C = C.loc[keep].reset_index(drop=True)
    return C, series
