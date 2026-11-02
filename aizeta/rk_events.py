import sys, os, json
sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common import load
from prep import preprocess
from rk_params import P
from rk_thermal import temperature_on_frames
from track03_kinetics import frame_times

COL = {"delta": "#e03b3b", "gamma": "#3b6ee0", "sapphire": "#9aa6af", "static": "#e3b35a"}
EVC = {"no γ": "#6b7a86", "δ appear": "#e03b3b", "γ appear": "#3b6ee0", "δ disappear": "#a33", "γ disappear": "#24479a", "γ appear 2": "#7b4fd6"}


def events(case, arm="site_full"):
    U = pd.read_csv(f"rk_out/{case}_{arm}_units.csv")
    R = json.load(open("rk_results.json", encoding="utf-8"))[case][arm]["summary"]
    tf = frame_times(case); t = lambda f: tf[int(f)]
    f0, f1 = int(U.ignite.min()), int(U.extinguish.max())
    on = R["onset_frame"]
    D = U[(U.phase == "delta") & ~U.edge]
    D = D[(D.extinguish.map(t) - D.ignite.map(t)) >= P["min_delta_s"]].sort_values("ignite")
    ev = []
    liquid = not bool(((U.phase == "delta") & U.present_start & ~U.edge).any())
    first_d = int(D.ignite.min()) if len(D) else None
    ref_end = first_d if (liquid and first_d) else (on if on else f1)
    nog = int(f0 + 0.5 * (ref_end - f0)) if ref_end and ref_end > f0 else f0
    ev.append(("no γ", nog, "liquid, no pattern" if (liquid and first_d and nog < first_d) else "δ only, no γ"))
    if liquid:
        last = -1e9
        Dl = D[(D.ignite < (on if on else f1))]
        if "site" in Dl.columns:
            gg = Dl.groupby("site"); Dl = Dl.loc[gg.ignite.idxmin()].copy()
            Dl["extinguish"] = gg.extinguish.max().reindex(Dl.site).values
        Dl = Dl[(Dl.extinguish.map(t) - Dl.ignite.map(t)) >= 10.0].sort_values("ignite")
        grp = []
        for k, (_, r) in enumerate(Dl.iterrows()):
            if grp and t(r.ignite) - t(grp[-1][-1][1].ignite) <= 10.0:
                grp[-1].append((k, r))
            else:
                grp.append([(k, r)])
        for g in grp:
            k0, r0 = g[0]
            if len(g) == 1:
                ev.append(("δ appear", int(r0.ignite), f"δ{k0+1} at ({r0.cy:.0f},{r0.cx:.0f})"))
            else:
                ev.append(("δ appear", int(r0.ignite), f"δ{k0+1}-δ{g[-1][0]+1}: {len(g)} new δ in {t(g[-1][1].ignite) - t(r0.ignite):.0f} s, first at ({r0.cy:.0f},{r0.cx:.0f})"))
    if on is not None:
        gf = R.get("gamma_first_frame")
        ev.append(("γ appear", int(on), (f"first new spot #{gf}, " if gf is not None and gf != on else "") + f"{R['n_burst']} new in 10 s"))
    if on is not None and len(D):
        Da = D[(D.ignite <= on) & (D.extinguish >= on - int(2 * (1 / max(np.median(np.diff(tf)), 1e-3))))]
        if len(Da):
            fe = int(Da.extinguish.max())
            if fe < f1 - 1:
                ev.append(("δ disappear", fe, f"{(t(fe)-t(on)):+.1f} s after γ appear"))
            else:
                ev.append(("δ disappear", f1, "still present at window end (Case 4)"))
    G = U[(U.phase == "gamma") & ~U.edge]
    if on is not None and len(G):
        ig = np.sort(G.ignite[G.ignite.map(t) > t(on) + P["W_slow"]].values)
        tt = np.array([t(f) for f in ig])
        for i in range(len(ig)):
            n1 = int(((tt >= tt[i]) & (tt <= tt[i] + P["W_tog"])).sum())
            n10 = int(((tt >= tt[i]) & (tt <= tt[i] + P["W_slow"])).sum())
            if n1 < P["K_tog"] or n10 < P["K_gamma"]:
                continue
            b2 = int(ig[i])
            g1 = G[(G.ignite < b2) & (G.extinguish.map(t) >= t(b2) - 1.0)]
            die = g1[np.abs(g1.extinguish.map(t) - t(b2)) <= 3.0]
            if len(g1) and len(die) >= 0.5 * len(g1):
                ev.append(("γ disappear", int(np.median(die.extinguish)), f"{len(die)} of {len(g1)} first-cohort γ vanish"))
                ev.append(("γ appear 2", b2, f"{n10} new spots in 10 s (2nd cohort)"))
                break
    ev.sort(key=lambda e: e[1])
    return ev, U, R, (f0, f1)


def figure(case, arm="site_full"):
    ev, U, R, (f0, f1) = events(case, arm)
    mk = np.load(f"mask_{case}.npy"); ys, xs = np.where(mk)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    tf = frame_times(case); th = temperature_on_frames(case)
    n = len(ev)
    fig = plt.figure(figsize=(3.1 * max(n, 4), 7.2))
    gs = fig.add_gridspec(2, max(n, 4), height_ratios=[1.25, 0.8], hspace=0.38)
    for j, (lab, f, note) in enumerate(ev):
        A = fig.add_subplot(gs[0, j])
        u = preprocess(load(case, f), mk)["u"]
        A.imshow(u[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=0.8)
        al = U[(U.ignite <= f) & (U.extinguish >= f)]
        for _, r in al.iterrows():
            if r.phase in ("delta", "gamma"):
                A.add_patch(plt.Circle((r.cx - x0, r.cy - y0), max(7, 1.1 * np.sqrt(max(r.area, 20) / np.pi)),
                                       fill=False, ec=COL[r.phase], lw=1.5))
        A.set_xticks([]); A.set_yticks([])
        for s in A.spines.values():
            s.set_edgecolor(EVC[lab]); s.set_linewidth(3)
        Tt = f" · {th['T'][f]:.0f}°C" if th["synced"] else ""
        A.set_xlabel(f"{lab}\nframe {f}{Tt}\n{note}", fontsize=10, color=EVC[lab], fontweight="bold")
    B = fig.add_subplot(gs[1, :])
    fr = np.arange(f0, f1 + 1)
    for ph in ("delta", "gamma"):
        S = U[(U.phase == ph) & ~U.edge]
        B.plot(tf[fr] - tf[f0], [int(((S.ignite <= f) & (S.extinguish >= f)).sum()) for f in fr], color=COL[ph], lw=1.6,
               label=f"{'δ' if ph=='delta' else 'γ'} spots alive")
    ymax = B.get_ylim()[1]; span = tf[f1] - tf[f0]; lastx, lvl = -1e9, 0
    for lab, f, note in ev:
        x = tf[f] - tf[f0]
        lvl = lvl + 1 if x - lastx < 0.04 * span else 0
        B.axvline(x, color=EVC[lab], ls="--", lw=1.1)
        B.text(x + 0.012 * span * lvl, ymax * (0.97 - 0.30 * lvl), f" {lab}", color=EVC[lab], fontsize=9, va="top",
               rotation=90, fontweight="bold")
        lastx = x
    B.set_xlabel(f"time from frame {f0} [s]"); B.set_ylabel("spots"); B.legend(loc="center left", fontsize=9)
    if th["synced"]:
        B2 = B.twinx(); B2.plot(tf[fr] - tf[f0], th["T"][fr], color="#999", lw=0.8); B2.set_ylabel("T [°C]", color="#777")
    fig.suptitle(f"{case} — automatic event timeline (rule engine, {arm}); case {R['case']}", fontsize=11)
    os.makedirs("fig", exist_ok=True); plt.savefig(f"fig/events_{case}.png", dpi=88, bbox_inches="tight"); plt.close()
    return ev


if __name__ == "__main__":
    out = {}
    for c in sys.argv[1:]:
        ev = figure(c)
        out[c] = [dict(label=a, frame=b, note=d) for a, b, d in ev]
        print(c, [(a, b) for a, b, _ in ev])
    try:
        old = json.load(open("rk_events.json", encoding="utf-8"))
    except Exception:
        old = {}
    old.update(out); json.dump(old, open("rk_events.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
