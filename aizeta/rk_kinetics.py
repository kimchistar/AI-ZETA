import sys, json, os
sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load
from prep import preprocess
from rk_params import P
from rk_thermal import temperature_on_frames, max_rise, cooling_arrest
from track03_kinetics import frame_times, fit_avrami, avrami
from datetime import datetime

R = json.load(open("rk_results.json", encoding="utf-8")); EV = json.load(open("rk_events.json", encoding="utf-8"))
CASES = ["8_TRIP2", "2_TRIP1", "31_A003", "26_V19_5A", "21_V19_1A"]
PUB = {
 "8_TRIP2": dict(ref="Zamanian 2025 New, steel T2 (Table 4; Figs. 5–6)", t05=0.14, n=1.2, onset=1363.4, eq=1453.1, dT=89.7, rec=7.3, fps=5,
                 delta_h=1, note="30 °C/min; the first γ panel (t = 0.2 s, 1368 °C) is already within the recalescence, so the onset temperature is that of t = 0 (1363.4 °C)",
                 mapping="same run: data folder 'New insights into massive transformation', 1 A/min; onset temperature and recalescence agree"),
 "2_TRIP1": dict(ref="Zamanian 2025 New, steel T1 (Table 3; Figs. 3–4)", t05=18.4, n=1.6, onset=1372.6, eq=1445.6, dT=73.0, rec=None, fps=10,
                 delta_h=2, note="110 °C/min; onset temperature and undercooling taken at the first γ in the Laue figure (t = 0.8 s, 1372.6 °C)",
                 mapping="same steel and condition: data folder 'New insights…', 5 A/min, measured cooling ≈ 94 °C/min"),
 "31_A003": dict(ref="Zamanian 2024 Analysis, ULC at 2.2 °C/s (Table 3; Figs. 3–4)", t05=0.055, n=1.9, onset=1340.8, eq=1340.8 + 56, dT=56, rec=3.1, fps=None,
                 delta_h=3, note="all five temperatures printed in Fig. 3 agree with our log within 2 °C",
                 mapping="same run: data folder 'Analysis of high temperature…', measured cooling 2.15 °C/s"),
 "26_V19_5A": dict(ref="Zamanian 2025 Comparison, steel C, small undercooling (Table 3; Figs. 4–5)", t05=0.94, n=0.85, onset=1468.5, eq=1488.6, dT=20.1, rec=None, fps=5,
                 delta_h=1, note="110 °C/min; onset temperature and undercooling taken at the first γ in the Laue figure (1468.5 °C)",
                 mapping="same steel and condition: data folder 'Comparison of transformation behavior…', 5 A/min, measured cooling ≈ 85 °C/min"),
 "21_V19_1A": dict(ref="Zamanian 2025 Comparison, steel C, large undercooling (Table 4; Figs. 6–7)", t05=0.096, n=2.1, onset=1410.1, eq=1488.6, dT=78.5, rec=7.1, fps=5,
                 delta_h=2, note="30 °C/min; onset temperature and undercooling taken at the first γ in the Laue figure (1410.1 °C)",
                 mapping="same run: data folder 'Comparison…', 1 A/min, measured cooling ≈ 32 °C/min; temperature and recalescence agree"),
}


def disk_sum(R_, cy, cx, r_in=6, r1=8, r2=12):
    y, x = int(round(cy)), int(round(cx))
    yy, xx = np.mgrid[-r2:r2 + 1, -r2:r2 + 1]; d = np.hypot(yy, xx)
    sub = R_[max(y - r2, 0):y + r2 + 1, max(x - r2, 0):x + r2 + 1]
    if sub.shape != d.shape:
        return np.nan
    ring = np.median(sub[(d >= r1) & (d <= r2)])
    return float((sub[d <= r_in] - ring).clip(0).sum())


def temp_events(case, f_on):
    th = temperature_on_frames(case)
    if not th["synced"]:
        return dict(T_on=None, dT_rec=None, f_rec=None, f_rec_peak=None, arrest=None)
    T = np.asarray(th["T"], float); fps = 1 / np.median(np.diff(frame_times(case)))
    rise, fa, fb = max_rise(T, f_on - int(P["rec_before"] * fps), f_on + int(P["rec_after"] * fps))
    big = rise >= P["dT_rec_min"]
    return dict(T_on=float(T[f_on]), dT_rec=round(rise, 2), f_rec=int(fa) if big else None,
                f_rec_peak=int(fb) if big else None, T_rec_peak=float(T[fb]) if big else None,
                arrest=cooling_arrest(T, f_on, fps))


def run(case, g0_over=None, horizon_s=None):
    U = pd.read_csv(f"rk_out/{case}_site_full_units.csv")
    s = R[case]["site_full"]["summary"]
    tf = frame_times(case); fps = 1 / np.median(np.diff(tf))
    t = lambda f: tf[int(f)]
    on = s["onset_frame"]; g0 = g0_over or s.get("gamma_first_frame") or on
    ev = {e["label"]: e["frame"] for e in EV.get(case, [])}
    D = U[(U.phase == "delta") & ~U.edge & U.site.notna()]
    if not len(D):
        return dict(case=case, n_delta=0)
    sites = D.groupby("site").agg(cy=("cy", "median"), cx=("cx", "median"), a=("ignite", "min"), b=("extinguish", "max"))
    t_dis = ev.get("δ disappear", int(sites.b.max())) if not g0_over else int(sites.b.max())
    alive = sites[(sites.a <= g0) & (sites.b >= g0 - int(3 * fps))]
    if not len(alive):
        alive = sites[sites.b == sites.b.max()]
    t0f = g0
    end = int(min(max(t_dis, alive.b.max()) + max(3 * fps, 0.3 * (t(alive.b.max()) - t(t0f)) * fps), len(tf) - 1))
    if horizon_s:
        end = int(min(t0f + horizon_s * fps, len(tf) - 1))
    beg = int(max(t0f - 6 * fps, 0))
    step = max(1, (end - beg + 1) // 450)
    frames = np.arange(beg, end + 1, step)
    cf = f"rk_cache/{case}_kin_v1{'_' + str(g0_over) if g0_over else ''}.npz"
    if os.path.exists(cf) and np.array_equal(np.load(cf)["frames"], frames):
        Isum = np.load(cf)["I"]
    else:
        mk = np.load(f"mask_{case}.npy"); Isum = []
        for f in frames:
            R_ = preprocess(load(case, int(f)), mk)["res"]
            Isum.append(np.nansum([disk_sum(R_, r.cy, r.cx) for r in alive.itertuples()]))
        Isum = np.array(Isum); np.savez(cf, frames=frames, I=Isum)
    tt = tf[frames] - t(t0f)
    Ism = pd.Series(Isum).rolling(3, center=True, min_periods=1).median().values
    pw = (tt >= -6) & (tt <= -2)
    I_pre = np.median(Ism[pw]) if pw.sum() >= 3 else np.median(Ism[tt <= 0])
    tail = tt >= tt.max() - max(1.0, 0.1 * tt.max())
    I_post = np.median(Ism[tail])
    X = np.clip((I_pre - Ism) / max(I_pre - I_post, 1e-9), 0, 1)

    def cross(level):
        above = X >= level
        idx = [i for i in range(len(X)) if above[i:].mean() > 0.8 and above[i]]
        if not idx:
            return np.nan
        i = idx[0]
        if i == 0:
            return float(tt[0])
        return float(np.interp(level, [X[i - 1], X[i]], [tt[i - 1], tt[i]]))
    t10, t50, t90 = cross(0.1), cross(0.5), cross(0.9)
    dt_fr = float(np.median(np.diff(tt)))
    n_trans = int(((tt >= t10) & (tt <= t90)).sum()) if np.isfinite(t10) and np.isfinite(t90) else 0
    n_ok = n_trans >= 5
    ts0 = tt[(X < 0.05) & (tt <= t10)].max() if np.isfinite(t10) and ((X < 0.05) & (tt <= t10)).any() else 0.0
    m = tt > ts0
    t05, n, rmse = fit_avrami(tt[m] - ts0, X[m]) if n_ok else (np.nan, np.nan, np.nan)
    te = temp_events(case, t0f)
    pub = PUB[case + ("@" + str(g0_over) if g0_over else "")] if (case + ("@" + str(g0_over) if g0_over else "")) in PUB else PUB[case]
    dT_uc = (pub["eq"] - te["T_on"]) if (pub.get("eq") and te["T_on"]) else None
    out = dict(case=case, fps=round(fps, 1), n_delta=int(len(alive)), delta_sites=[(round(r.cy), round(r.cx)) for r in alive.itertuples()],
               gamma_first=int(g0), onset_rule=int(on), delta_disappear=int(t_dis),
               delta_life_after_gamma_s=round(float(t(alive.b.max()) - t(g0)), 2),
               I_pre=float(I_pre), I_post=float(I_post), drop_frac=float(1 - I_post / max(I_pre, 1e-9)),
               t05=t05, n=n, rmse=rmse, t_decay0=float(ts0), t10=t10, t50=t50, t90=t90, dt_frame=dt_fr * step / step,
               n_transition_frames=int(n_trans), n_reliable=bool(n_ok),
               **te, dT_uc=dT_uc, pub=pub,
               curve=dict(t=tt.round(3).tolist(), X=X.round(4).tolist()))
    return out


if __name__ == "__main__":
    res = {}
    for c in (sys.argv[1:] or CASES):
        r = run(c.split("@")[0], int(c.split("@")[1]), 6.0) if "@" in c else run(c); r["case"] = c; res[c] = r
        print(f"{c:10s} δ {r.get('n_delta')} | t0.5 {r.get('t05')} n {r.get('n')} trans {r.get('n_transition_frames')} | "
              f"t10/50/90 {r.get('t10')}/{r.get('t50')}/{r.get('t90')} T_on {r.get('T_on')} rec {r.get('dT_rec')} | ΔT_uc {r.get('dT_uc')}", flush=True)
    json.dump(res, open("rk_kinetics.json", "w", encoding="utf-8"), ensure_ascii=False, default=float)
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    for A, c in zip(ax.ravel(), CASES):
        r = res.get(c)
        if not r or "curve" not in r:
            A.set_title(f"{c}: no confirmed δ"); A.axis("off"); continue
        tt = np.array(r["curve"]["t"]); X = np.array(r["curve"]["X"])
        A.plot(tt, X, ".", ms=3, color="#e03b3b", label="δ transformed fraction (1 − I/I₀)")
        if r["n_reliable"] and np.isfinite(r["t05"]) and r["rmse"] > 0.05:
            x = np.linspace(1e-3, tt.max() - r["t_decay0"], 300)
            A.plot(x + r["t_decay0"], avrami(x, r["t05"], r["n"]), "k--", lw=0.8,
                   label=f"single Avrami fits poorly (rmse {r['rmse']:.2f}): two-stage decay")
        elif r["n_reliable"] and np.isfinite(r["t05"]):
            x = np.linspace(1e-3, tt.max() - r["t_decay0"], 300)
            A.plot(x + r["t_decay0"], avrami(x, r["t05"], r["n"]), "k-", lw=1.2,
                   label=f"Avrami (t=0 at decay start) t₀.₅={r['t05']:.2f}s n={r['n']:.2f}")
        else:
            A.plot([], [], " ", label=("10→90 % between two frames" if r["n_transition_frames"] == 0 else f"10→90 % within {r['n_transition_frames']} frames") + ": frame-limited")
        if np.isfinite(r["t50"]):
            A.axvline(r["t50"], color="#e03b3b", ls=":", lw=1); A.text(r["t50"], 0.5, f" X=0.5 @ {r['t50']:+.2f}s", color="#e03b3b", fontsize=8)
        A.axvline(0, color="#3b6ee0", ls="--", lw=1); A.text(0, 1.02, " γ appear", color="#3b6ee0", fontsize=8)
        p = r["pub"]
        if isinstance(p.get("t05"), (int, float)):
            A.axvline(p["t05"], color="#999", ls=":", lw=1.2); A.text(p["t05"], 0.12, f" pub t₀.₅ {p['t05']}", color="#777", fontsize=8)
        A.set_ylim(-0.05, 1.1); A.set_xlabel("time from first γ spot [s]"); A.set_ylabel("X")
        A.set_title(f"{c}  ({r['fps']} fps, {r['n_delta']} confirmed δ)", fontsize=10); A.legend(fontsize=7, loc="lower right")
        if tt.max() > 20:
            A.set_xscale("symlog", linthresh=1)
    os.makedirs("fig", exist_ok=True); plt.tight_layout(); plt.savefig("fig/kinetics.png", dpi=90)
