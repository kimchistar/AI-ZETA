import sys, os, json
import numpy as np, pandas as pd


def export(case, arm="site_full", cut_at_completion=True):
    U = pd.read_csv(f"rk_out/{case}_{arm}_units.csv")
    obs = pd.read_csv(f"rk_out/{case}_{arm}_obs.csv")
    det_frames = np.sort(obs.frame.unique())
    if cut_at_completion and os.path.exists("refine_timing.json"):
        tr = json.load(open("refine_timing.json", encoding="utf-8")).get(case, {}).get("transformation", {})
        if tr.get("complete") is not None:
            U = U.copy(); m = (U.phase == "delta") & (U.ignite <= tr["complete"])
            U.loc[m, "extinguish"] = np.minimum(U.loc[m, "extinguish"], tr["complete"])
    U = U.copy()
    if os.path.exists(f"track_det_{case}.csv"):
        full_ = pd.read_csv(f"track_det_{case}.csv", usecols=["frame", "cy", "cx"])
        step_ = max(1, int(np.median(np.diff(det_frames))))
        for i_ in U.index[U.phase.isin(["delta", "gamma"])]:
            f_ = int(U.at[i_, "ignite"])
            while True:
                nb = full_[(full_.frame >= f_ - step_ - 1) & (full_.frame < f_)]
                hit = nb.frame[np.hypot(nb.cy - U.at[i_, "cy"], nb.cx - U.at[i_, "cx"]) <= 6]
                if not len(hit):
                    break
                f_ = int(hit.min())
            U.at[i_, "ignite"] = f_
    if os.path.exists(f"rk_out/{case}_sap_gamma.csv"):
        SG = pd.read_csv(f"rk_out/{case}_sap_gamma.csv")
        if len(SG):
            add = pd.DataFrame(dict(uid=900000 + np.arange(len(SG)), site=900000 + np.arange(len(SG)), phase="gamma",
                                    ignite=SG.first_frame.values, extinguish=SG.last_frame.values, cy=SG.cy.values, cx=SG.cx.values,
                                    area=SG.area.values, edge=False, present_start=False, evidence="R-SD-diff: γ on a sapphire"))
            U = pd.concat([U, add], ignore_index=True)
    U = U.copy()
    key = U["site"] if "site" in U.columns else U["uid"]
    U["lockey"] = key.astype("Int64").astype(str) + "|" + U.phase
    PFX = {"delta": "δ", "gamma": "γ", "sapphire": "S", "static": "B"}
    lab = {}
    for ph, p in PFX.items():
        firsts = U[U.phase == ph].groupby("lockey").ignite.min().sort_values()
        for k, lk in enumerate(firsts.index, 1):
            lab[lk] = f"{p}{k}"
    U["label"] = U.lockey.map(lab).fillna("")
    rows = []
    for lk, g in U[U.label != ""].groupby("lockey"):
        rows.append(dict(label=g.label.iloc[0], phase=g.phase.iloc[0], site=lk.split("|")[0],
                         cy=round(float(g.cy.median()), 1), cx=round(float(g.cx.median()), 1),
                         first_frame=int(g.ignite.min()), last_frame=int(g.extinguish.max()), n_episodes=len(g),
                         area=round(float(g.area.median()), 0), edge=bool(g.edge.any()),
                         evidence=";".join(sorted(set(str(e).strip() for e in g.evidence if str(e).strip())))))
    order = {"δ": 0, "γ": 1, "S": 2, "B": 3}
    Lt = pd.DataFrame(rows)
    if len(Lt):
        Lt["_o"] = Lt.label.str[0].map(order); Lt["_n"] = Lt.label.str[1:].astype(int)
        Lt = Lt.sort_values(["_o", "_n"]).drop(columns=["_o", "_n"])
    return Lt


if __name__ == "__main__":
    for case in sys.argv[1:]:
        export(case).to_csv(f"rk_out/{case}_labels.csv", index=False)
