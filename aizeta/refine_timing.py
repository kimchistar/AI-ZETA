import sys, os, json, time; sys.path.insert(0, ".")
import numpy as np, pandas as pd
from skimage import measure
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from rk_params import P
from track03_kinetics import frame_times

CASES = ["8_TRIP2", "2_TRIP1", "31_A003", "26_V19_5A", "21_V19_1A"]
R = json.load(open("rk_results.json", encoding="utf-8"))
E = json.load(open("rk_events.json", encoding="utf-8"))
WIN = {"fast": (3.0, 3.0), "slow": (10.0, 5.0)}
MAX_DELTA, BOX, R_DELTA = 3, 64, 6.0


def stage1(case):
    s = R[case]["site_full"]["summary"]; tf = frame_times(case); fps = 1 / np.median(np.diff(tf))
    ev = {e["label"]: e["frame"] for e in E.get(case, [])}
    g0 = min(s["gamma_first_frame"], s["onset_frame"]); on = s["onset_frame"]
    t_dis = ev.get("δ disappear") or on
    mode = s.get("thermal_mode") or ("fast" if abs(t_dis - on) / fps <= 1.0 else "slow")
    L = pd.read_csv(f"rk_out/{case}_labels.csv")
    D = L[L.phase == "delta"].assign(life=lambda d: d.last_frame - d.first_frame)
    alive = D[(D.first_frame < g0) & (D.last_frame >= g0 - 3 * fps)]
    if not len(alive):
        alive = D[D.last_frame >= w0]
    alive = alive.sort_values("life", ascending=False).head(MAX_DELTA)
    extra = []
    mfn = f"rk_out/{case}_site_merges.json"
    if os.path.exists(mfn):
        MG = json.load(open(mfn))
        roots = {}
        for m_ in MG:
            roots.setdefault(int(m_["keep"]), []).append(m_)
        for r in alive.itertuples():
            for m_ in roots.get(int(r.site), []):
                extra.append(dict(label=f"{r.label}'", cy=m_["cy"], cx=m_["cx"], area=m_["area"]))
    w0, w1, dyn = dynamic_window(case, alive, g0, fps, len(tf) - 1)
    return dict(fps=fps, g0=int(g0), onset=int(on), t_dis=int(t_dis), mode=mode, w0=w0, w1=w1, dyn=dyn,
                deltas=[dict(label=r.label, cy=float(r.cy), cx=float(r.cx), area=float(r.area)) for r in alive.itertuples()] + extra,
                others=L[L.phase.isin(["sapphire", "static", "delta"])][["cy", "cx", "area"]].values.tolist(),
                sapphire=L[L.phase == "sapphire"][["cy", "cx", "area"]].values.tolist())


def dynamic_window(case, alive, g0, fps, last):
    from common import load
    from prep import preprocess
    from rk_kinetics import disk_sum
    mk = np.load(f"mask_{case}.npy"); step = max(1, int(round(fps / 4)))
    def level(f):
        R_ = preprocess(load(case, int(f)), mk)["res"]; v = 0.0
        for r in alive.itertuples():
            ri = int(min(30, np.sqrt(max(r.area, 20) / np.pi) + 2))
            v += disk_sum(R_, r.cy, r.cx, r_in=ri, r1=ri + 2, r2=ri + 8)
        return v
    lo = int(max(g0 - 30 * fps, 0))
    fr = list(range(lo, int(g0 + 5 * fps), step)); L = [level(f) for f in fr]
    Ls = pd.Series(L).rolling(5, center=True, min_periods=1).median().values
    ip = int(np.argmax(Ls)); peak = Ls[ip]
    drop = next((k for k in range(ip, len(Ls)) if Ls[k] < 0.9 * peak), None)
    if drop is None:
        start = int(g0)
    else:
        k = drop
        while k > ip and Ls[k] < 0.95 * peak:
            k -= 1
        start = int(fr[k])
    f, low, end = int(max(start, g0)), 0, last
    hist = []
    while f <= last:
        v = level(f); hist.append((f, v))
        low = low + 1 if v < 0.1 * peak else 0
        if low * step >= 2 * fps:
            end = int(f - 2 * fps); break
        f += step
    yy, xx = np.mgrid[-12:13, -12:13]; dd = np.hypot(yy, xx)
    def cores(f):
        Sm = preprocess(load(case, int(f)), mk)["S"]; out = []
        for r in alive.itertuples():
            y, x = int(round(r.cy)), int(round(r.cx)); rr = min(12, np.sqrt(max(r.area, 20) / np.pi) + 2)
            sub = Sm[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]
            out.append(np.sort(sub[dd[:sub.shape[0], :sub.shape[1]] <= rr])[-6:].mean() if sub.size else 0.0)
        return max(out) if out else 0.0
    f, low = int(max(end, g0)), 0
    while f <= last:
        low = low + 1 if cores(f) < P["theta_off"] else 0
        if low * step >= 2 * fps:
            end = max(end, int(f - 2 * fps)); break
        f += step
    else:
        end = last
    w0 = int(max(min(start, g0) - fps, 0)); w1 = int(min(end + 2 * fps, last))
    return w0, w1, dict(decay_start=start, below10=end, plateau=float(peak))


def crop_detect(case, S1, amg):
    from common import load
    from prep import preprocess
    out = f"track_det_refine_{case}.csv"; rows = []; done = set()
    if os.path.exists(out):
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(old.frame.unique())
    mk = np.load(f"mask_{case}.npy")
    boxes = [(max(int(d["cy"]) - BOX, 0), min(int(d["cy"]) + BOX, 800), max(int(d["cx"]) - BOX, 0), min(int(d["cx"]) + BOX, 800)) for d in S1["deltas"]]
    frames = [f for f in range(S1["w0"], S1["w1"] + 1) if f not in done]
    t0 = time.time()
    for n, f in enumerate(frames):
        Pp = preprocess(load(case, f), mk); n_before = len(rows)
        for bi, (y0, y1, x0, x1) in enumerate(boxes):
            m = mk[y0:y1, x0:x1]
            if m.sum() < 100:
                continue
            lab, sc = amg.run(Pp["u"][y0:y1, x0:x1], m, spacing=3, iou_thr=0.5, max_area_frac=0.3)
            Sc = Pp["S"][y0:y1, x0:x1]
            for p in measure.regionprops(lab, intensity_image=Sc):
                core = float(np.sort(Sc[p.coords[:, 0], p.coords[:, 1]])[-6:].mean())
                if p.area < 6 or sc[p.label] < 0.88 or (p.intensity_mean < 3.0 and core < P["core_gate"]):
                    continue
                rows.append(dict(frame=f, box=bi, cy=p.centroid[0] + y0, cx=p.centroid[1] + x0, area=int(p.area),
                                 snr_mean=float(p.intensity_mean), score=float(sc[p.label])))
        if len(rows) == n_before:
            rows.append(dict(frame=f, box=-1, cy=np.nan, cx=np.nan, area=0, snr_mean=np.nan, score=np.nan))
        if n % 25 == 0:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  refine-crop {case} {n + 1}/{len(frames)} ({(time.time() - t0) / (n + 1):.1f}s/f)", flush=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    return pd.read_csv(out)


def delta_disappearance(S1, C, t_gamma, case=None):
    from common import load
    from prep import preprocess
    fps = S1["fps"]; gap = int(round(P["gap_s"] * fps)); out = []
    rad = 10.0 if S1["mode"] == "slow" else R_DELTA
    Cd = C[C.box >= 0]; tg = t_gamma if t_gamma is not None else S1["g0"]
    mk = np.load(f"mask_{case}.npy"); frs = np.arange(S1["w0"], S1["w1"] + 1)
    yy, xx = np.mgrid[-12:13, -12:13]; dd = np.hypot(yy, xx)
    core = {i: [] for i in range(len(S1["deltas"]))}
    for f in frs:
        Sm = preprocess(load(case, int(f)), mk)["S"]
        for i, d in enumerate(S1["deltas"]):
            y, x = int(round(d["cy"])), int(round(d["cx"])); r = min(12, np.sqrt(max(d.get("area", 20), 20) / np.pi) + 2)
            sub = Sm[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]
            core[i].append(np.sort(sub[dd[:sub.shape[0], :sub.shape[1]] <= r])[-6:].mean() if sub.size else np.nan)
    for i, d in enumerate(S1["deltas"]):
        cs = pd.Series(core[i]).rolling(3, center=True, min_periods=1).median().values
        lit = frs[cs >= P["theta_off"]]
        t_ph = None
        for k, f in enumerate(lit):
            nxt = lit[k + 1] if k + 1 < len(lit) else None
            if f < tg - 1:
                continue
            if (nxt is None and S1["w1"] - f >= gap) or (nxt is not None and nxt - f > gap):
                t_ph = int(f); break
        if t_ph is None and len(lit) and lit.max() < tg:
            t_ph = int(lit.max())
        litd = np.sort(Cd[np.hypot(Cd.cy - d["cy"], Cd.cx - d["cx"]) <= rad].frame.unique())
        t_det = int(litd[litd >= tg - 1][np.r_[np.diff(litd[litd >= tg - 1]) > gap, True]][0]) if (litd >= tg - 1).any() else None
        out.append(dict(d, t_dis=t_ph, t_dis_det=t_det, core_snr=[round(float(v), 2) for v in cs[::max(1, int(fps / 4))]],
                        note="" if t_ph is not None else "still above noise at window end"))
    return out


def new_locations(case, S1):
    from common import load
    from prep import preprocess
    fps = S1["fps"]; w0, w1 = S1["w0"], S1["w1"]
    det = pd.read_csv(f"track_det_{case}.csv"); det = det[(det.frame >= w0) & (det.frame <= w1)].reset_index(drop=True)
    xy = det[["cy", "cx"]].values
    pairs = cKDTree(xy).query_pairs(P["r_site"], output_type="ndarray")
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, site = connected_components(G, directed=False); det["site"] = site
    det = det.sort_values(["site", "frame"])
    pos0 = det.groupby("site")[["cy", "cx"]].median()
    mk0 = np.load(f"mask_{case}.npy"); frs = np.arange(w0, w1 + 1)
    yy0, xx0 = np.mgrid[-6:7, -6:7]; d0 = np.hypot(yy0, xx0) <= 6
    core = np.zeros((len(pos0), len(frs)), np.float32); P0 = np.round(pos0.values).astype(int)
    for j, f in enumerate(frs):
        Sm = preprocess(load(case, int(f)), mk0)["S"]; Sp = np.pad(Sm, 6)
        for i, (y, x) in enumerate(P0):
            core[i, j] = np.sort(Sp[y:y + 13, x:x + 13][d0])[-6:].mean()
    gap_s = P["gap_s"] * fps; ep_col = []
    for i, (sid, g) in enumerate(det.groupby("site", sort=True)):
        bright = pd.Series(core[i]).rolling(3, center=True, min_periods=1).median().values >= P["theta_off"]
        fs = g.frame.values; e, last, eps = 0, None, []
        for f in fs:
            if last is not None and f - last > gap_s and not bright[(frs > last) & (frs < f)].all():
                e += 1
            eps.append(e); last = f
        ep_col.append(pd.Series(eps, index=g.index))
    det["ep"] = pd.concat(ep_col)
    E = det.groupby(["site", "ep"]).agg(first=("frame", "min"), last=("frame", "max"), cy=("cy", "median"), cx=("cx", "median"),
                                         area=("area", "median")).reset_index()
    E = E.sort_values(["site", "ep"]); E["prev_end"] = E.groupby("site")["last"].shift(1)
    E = E[E["first"] > w0 + fps]
    sap = np.array([o for o in S1["sapphire"]]) if S1.get("sapphire") else np.zeros((0, 3))
    if len(sap) and len(E):
        d, i = cKDTree(sap[:, :2]).query(E[["cy", "cx"]].values)
        E = E[d > np.sqrt(sap[i, 2] / np.pi) + 4]
    if not len(E):
        return E.assign(dS=[]), det
    mk = np.load(f"mask_{case}.npy"); need = {}
    for k, r in E.iterrows():
        for f in range(int(r["first"] - 3 * fps), int(min(r["last"], r["first"] + 5 * fps)) + 1):
            if w0 <= f <= w1:
                need.setdefault(f, []).append(k)
    yy, xx = np.mgrid[-12:13, -12:13]; dd = np.hypot(yy, xx)
    val = {k: {} for k in E.index}
    for f in sorted(need):
        S = preprocess(load(case, int(f)), mk)["S"]
        for k in need[f]:
            r = E.loc[k]; y, x = int(round(r.cy)), int(round(r.cx)); rr = min(12, np.sqrt(max(r.area, 6) / np.pi) + 2)
            sub = S[max(y - 12, 0):y + 13, max(x - 12, 0):x + 13]
            if sub.shape != dd.shape:
                continue
            v = np.sort(sub[dd <= rr])[-6:]; val[k][f] = float(v.mean())
    dS = []
    for k, r in E.iterrows():
        p0 = r["first"] - 3 * fps if pd.isna(r["prev_end"]) else max(r["first"] - 3 * fps, r["prev_end"] + 0.3 * fps)
        pre = [v for f, v in val[k].items() if p0 <= f <= r["first"] - 0.3 * fps]
        post = [v for f, v in val[k].items() if r["first"] <= f <= min(r["last"], r["first"] + 5 * fps)]
        dS.append(np.median(post) - np.median(pre) if pre and post else np.nan)
    E = E.assign(dS=dS)
    E = E[(E.dS >= P["dS_new"]) | E.dS.isna()]
    return E.sort_values("first"), det


def delta_fraction(case, S1):
    from common import load, files
    from prep import preprocess
    from rk_kinetics import disk_sum
    from track03_kinetics import fit_avrami
    fps = S1["fps"]; tf = frame_times(case); mk = np.load(f"mask_{case}.npy"); N = len(files(case))
    def inten(f):
        R_ = preprocess(load(case, int(f)), mk)["res"]; v = 0.0
        for d in S1["deltas"]:
            ri = int(min(30, np.sqrt(max(d.get("area", 20), 20) / np.pi) + 2))
            v += disk_sum(R_, d["cy"], d["cx"], r_in=ri, r1=ri + 2, r2=ri + 8)
        return v
    frs = list(range(S1["w0"], S1["w1"] + 1)); I = [inten(f) for f in frs]
    w3 = max(3, int(round(3 * fps)))
    while frs[-1] < N - 1:
        Is_ = pd.Series(I).rolling(3, center=True, min_periods=1).median().values
        if len(Is_) >= 2 * w3 and (np.median(Is_[-2 * w3:-w3]) - np.median(Is_[-w3:])) < 0.005 * Is_.max():
            break
        for f in range(frs[-1] + 1, min(frs[-1] + w3, N - 1) + 1):
            frs.append(f); I.append(inten(f))
    frs = np.array(frs)
    Is = pd.Series(I).rolling(3, center=True, min_periods=1).median().values
    ip = int(np.argmax(Is)); plateau = Is[ip]
    floor = float(np.median(Is[-w3:]))
    X = (Is - floor) / max(plateau - floor, 1e-9)
    noise = max(0.01, 3 * float(np.std(X[-w3:])))
    hold = max(2, int(round(0.3 * fps)))
    def first_sustained(cond, frm):
        for k in range(frm, len(cond) - hold + 1):
            if cond[k:k + hold].all():
                return k
        return None
    s_i = first_sustained(X < 0.95, ip)
    e_i = first_sustained(X <= 0.05, s_i) if s_i is not None else None
    z_i = first_sustained(X <= noise, s_i) if s_i is not None else None
    t = tf[frs]
    dur = float(t[e_i] - t[s_i]) if (s_i is not None and e_i is not None) else None
    t05 = n = None
    if s_i is not None:
        k0 = max(s_i - 1, 0); m = np.arange(len(X)) >= k0
        t05, n, _ = fit_avrami(t[m] - t[k0], np.clip(1 - X[m], 0, 1))
        if t05 != t05 or n != n or t05 <= 2e-3 or not (0.2 < n < 8):
            t05 = n = None
    return dict(start=int(frs[s_i]) if s_i is not None else None, complete=int(frs[e_i]) if e_i is not None else None,
                duration_s=dur, transition=("fast" if dur is not None and dur <= P["fast_s"] else "slow"),
                full_disappear=int(frs[z_i]) if z_i is not None else None,
                full_s=float(t[z_i] - t[s_i]) if (s_i is not None and z_i is not None) else None, noise=noise,
                t0_frame=int(frs[max(s_i - 1, 0)]) if s_i is not None else None,
                t05=t05, n=n, frames_in_transition=int(e_i - s_i) if (s_i is not None and e_i is not None) else None,
                curve=dict(frames=[int(f) for f in frs], X=[round(float(v), 4) for v in X]))


def gamma_onset(case, S1):
    fps = S1["fps"]
    NL = new_locations(case, S1)[0]
    keep = NL["first"] if len(NL) else pd.Series([], dtype=int)
    ig = np.sort(keep.values); W = (P["W_fast"] if S1["mode"] == "fast" else P["W_slow"]) * fps
    Wt = P["W_tog"] * fps
    for k in range(len(ig)):
        n = int(((ig >= ig[k]) & (ig <= ig[k] + W)).sum())
        n_tog = int(((ig >= ig[k]) & (ig <= ig[k] + Wt)).sum())
        if n >= P["K_gamma"] and n_tog >= P["K_tog"]:
            return dict(first_gamma=int(ig[k]), crowd_complete=int(ig[k + P["K_gamma"] - 1]), n_in_window=n)
    return dict(first_gamma=None, crowd_complete=None, n_in_window=0)


if __name__ == "__main__":
    from sam_amg import AMG
    amg = AMG("sam", fp16=True)
    res = json.load(open("refine_timing.json", encoding="utf-8")) if os.path.exists("refine_timing.json") else {}
    for c in (sys.argv[1:] or CASES):
        S1 = stage1(c)
        C = crop_detect(c, S1, amg)
        go = gamma_onset(c, S1); dd = delta_disappearance(S1, C, go["first_gamma"], c)
        dfr = delta_fraction(c, S1)
        tds = [x["t_dis"] for x in dd if x["t_dis"] is not None]
        res[c] = dict(stage1=dict(g0=S1["g0"], onset=S1["onset"], t_dis=S1["t_dis"], mode=S1["mode"], window=[S1["w0"], S1["w1"]]),
                      gamma=go, deltas=dd, delta_disappear=max(tds) if tds else None, transformation=dfr,
                      fps=round(S1["fps"]))
        print(c, json.dumps({k: res[c][k] for k in ("stage1", "gamma", "delta_disappear")}, ensure_ascii=False, default=int), flush=True)
        json.dump(res, open("refine_timing.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=int)
    print("REFINE_DONE", flush=True)
