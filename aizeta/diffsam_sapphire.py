import sys, os, json; sys.path.insert(0, ".")
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import maximum_filter
from common import load
from prep import preprocess
from rk_params import P
from track03_kinetics import frame_times
from sam_amg import AMG

CASES = sys.argv[1:] or ["31_A003", "8_TRIP2", "2_TRIP1", "26_V19_5A", "21_V19_1A"]
R = json.load(open("rk_results.json", encoding="utf-8")); amg = AMG("sam", fp16=True); B = 64
os.makedirs("fig/diffsam", exist_ok=True); allc = []
yy, xx = np.mgrid[:800, :800]
for case in CASES:
    s = R[case]["site_full"]["summary"]; on = int(s["onset_frame"]); tf = frame_times(case); fps = 1 / np.median(np.diff(tf))
    mode = s.get("thermal_mode") or s["burst"]
    W = P["W_fast"] if mode == "fast" else P["W_slow"]; N = len(tf)
    f_lo, f_hi = on - int(round(1.3 * fps)), min(on + int(round(W * fps)), N - 1)
    step = 1 if mode == "fast" else max(1, int(round(0.5 * fps)))
    mk = np.load(f"mask_{case}.npy")
    L = pd.read_csv(f"rk_out/{case}_labels.csv"); L = L[L.label.astype(str).str.len() > 0]
    SAP = L[L.phase == "sapphire"]
    DG = L[L.phase.isin(["delta", "gamma"]) & ~L.evidence.astype(str).str.contains("R-SD-diff")]
    DF = [(r.cy, r.cx, np.sqrt(r.area / np.pi) + 2) for r in L[L.phase == "delta"].itertuples()]
    if os.path.exists(f"rk_out/{case}_site_merges.json"):
        ds = set(L.site[L.phase == "delta"].astype(str))
        DF += [(m["cy"], m["cx"], np.sqrt(m["area"] / np.pi) + 2) for m in json.load(open(f"rk_out/{case}_site_merges.json")) if str(m["keep"]) in ds]
    bf = [preprocess(load(case, g), mk) for g in range(on - int(round(3 * fps)), f_lo, max(1, step))]
    Rb0 = np.median([b["res"] for b in bf], axis=0); sig_b = float(np.median([b["sigma"] for b in bf]))
    Rb = maximum_filter(Rb0, size=5)
    Dimg = lambda f: np.clip((preprocess(load(case, f), mk)["res"] - Rb) / sig_b, 0, None)
    hits = []
    for f in range(f_lo, f_hi + 1, step):
        D = Dimg(f); uD = np.arcsinh(D / 3) / np.arcsinh(40 / 3)
        for r in SAP.itertuples():
            y0, y1, x0, x1 = max(int(r.cy) - B, 0), min(int(r.cy) + B, 800), max(int(r.cx) - B, 0), min(int(r.cx) + B, 800)
            m = mk[y0:y1, x0:x1]
            if m.sum() < 100:
                continue
            lab, sc = amg.run(uD[y0:y1, x0:x1], m, spacing=3, iou_thr=0.5, max_area_frac=0.3)
            rs = np.sqrt(r.area / np.pi) + 3
            foot = (yy[y0:y1, x0:x1] - r.cy) ** 2 + (xx[y0:y1, x0:x1] - r.cx) ** 2 <= rs * rs
            Dc = D[y0:y1, x0:x1]
            for p in measure.regionprops(lab, intensity_image=Dc):
                if p.area < 6 or p.area > 2000 or sc[p.label] < 0.88 or p.intensity_mean < 5 or p.minor_axis_length < 3:
                    continue
                if foot[p.coords[:, 0], p.coords[:, 1]].sum() < 10:
                    continue
                v = Dc[p.coords[:, 0], p.coords[:, 1]]
                if np.sort(v)[-6:].mean() < 10:
                    continue
                py_, px_ = p.coords[:, 0] + y0, p.coords[:, 1] + x0
                if any(((py_ - a) ** 2 + (px_ - b) ** 2 <= rr * rr).any() for a, b, rr in DF):
                    continue
                if len(DG) and any(((np.abs(py_ - a) < 1.5) & (np.abs(px_ - b) < 1.5)).any() for a, b in zip(DG.cy, DG.cx)):
                    continue
                hits.append(dict(sap=r.label, frame=f, cy=float(py_.mean()), cx=float(px_.mean()), area=int(p.area),
                                 Dmean=float(v.mean()), score=float(sc[p.label]), py=py_, px=px_))
        print(case, f, len(hits), flush=True)
    C = []
    for h in sorted(hits, key=lambda h: h["frame"]):
        k = next((c for c in C if np.hypot(c["cy"] - h["cy"], c["cx"] - h["cx"]) <= 15), None)
        if k is None:
            C.append(dict(h, first=h["frame"], last=h["frame"], n=1, saps={h["sap"]}))
        else:
            k["n"] += 1; k["last"] = h["frame"]; k["saps"].add(h["sap"])
            if h["Dmean"] * h["area"] > k["Dmean"] * k["area"]:
                k.update(py=h["py"], px=h["px"], area=h["area"], Dmean=h["Dmean"], cy=h["cy"], cx=h["cx"], frame=h["frame"])
    C = [c for c in C if c["n"] >= 3]
    from scipy import ndimage as ndi_
    for c in C:
        M_ = np.zeros((800, 800), bool); M_[c["py"], c["px"]] = True
        for _ in range(3):
            grow = ndi_.binary_dilation(M_, iterations=2)
            for h in hits:
                if grow[h["py"], h["px"]].any() and len(h["py"]) <= 2000:
                    M_[h["py"], h["px"]] = True
        ys_, xs_ = np.where(M_); c["py"], c["px"] = ys_, xs_; c["area"] = int(M_.sum()); c["cy"], c["cx"] = float(ys_.mean()), float(xs_.mean())
    STEP_S, STEP, TAIL_S, TAIL = 5.0, 0.6, 10.0, 0.9
    for c in C:
        gs = list(range(max(c["first"] - int(round(2 * fps)), 0), N, max(1, int(round(fps)))))
        ser = pd.Series([float(Dimg(g)[c["py"], c["px"]].mean()) for g in gs]).rolling(3, center=True, min_periods=1).median().values
        k = int(np.argmax(ser)); pk = float(ser[k])
        c["t_s"] = [float(tf[g] - tf[c["first"]]) for g in gs]; c["ser"] = ser; c["peak_frame"] = int(gs[k])
        tt = np.asarray(c["t_s"]); early = ser[(tt >= 0) & (tt <= STEP_S)]
        c["step_ratio"] = float(early.max() / max(pk, 1e-6)) if len(early) else 0.0
        tail = ser[tt >= tt[-1] - TAIL_S]
        c["tail_ratio"] = float(tail.mean() / max(pk, 1e-6))
        c["ok"] = c["step_ratio"] >= STEP and c["tail_ratio"] < TAIL
        e = next((gs[j] for j in range(k, len(gs)) if ser[j] < 3), N - 1)
        c["extinguish"] = int(e)
        print("   cand", "/".join(sorted(c["saps"])), c["first"], "peak", c["peak_frame"], "step", round(c["step_ratio"], 2), "tail", round(c["tail_ratio"], 2),
              "ext", c["extinguish"], "PASS" if c["ok"] else "fail", flush=True)
    CALL = C; C = [c for c in C if c["ok"]]
    rows = [dict(case=case, label_hint="γ on " + "/".join(sorted(c["saps"])), cy=round(c["cy"], 1), cx=round(c["cx"], 1),
                 first_frame=int(c["first"]), last_frame=int(c["extinguish"]), area=int(c["area"]), Dmean=round(c["Dmean"], 1),
                 n_seen=int(c["n"])) for c in C]
    pd.DataFrame(rows, columns=["case", "label_hint", "cy", "cx", "first_frame", "last_frame", "area", "Dmean", "n_seen"]).to_csv(
        f"rk_out/{case}_sap_gamma.csv", index=False)
    np.savez_compressed(f"rk_out/{case}_sap_gamma_masks.npz", cy=np.array([r["cy"] for r in rows], float),
                        cx=np.array([r["cx"] for r in rows], float),
                        **{f"py{i}": np.asarray(c["py"], np.int16) for i, c in enumerate(C)},
                        **{f"px{i}": np.asarray(c["px"], np.int16) for i, c in enumerate(C)})
    allc += rows
    print(case, "onset", on, s["burst"], "window", f_lo, f_hi, "step", step, "σ_b", round(sig_b, 1), "sapphires", len(SAP),
          "γ on sapphires", len(C), [(r["label_hint"], r["first_frame"], r["last_frame"]) for r in rows], flush=True)
    if os.path.exists(f"fig/paper/diffsam/{case}.png"):
        os.remove(f"fig/paper/diffsam/{case}.png")
    if not CALL:
        continue
    fig, ax = plt.subplots(len(CALL), 4, figsize=(12.5, 3.1 * len(CALL)), squeeze=False, gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    for i, c in enumerate(CALL):
        a = ax[i, 3]; a.plot(c["t_s"], c["ser"], color="k", lw=1.2); a.axhline(c["ser"].max() * STEP, color="#d62728", ls=":", lw=1); a.axvspan(0, STEP_S, color="#3b8bff", alpha=0.08); a.axvspan(c["t_s"][-1] - TAIL_S, c["t_s"][-1], color="#d62728", alpha=0.08)
        a.axvline(0, color="#3b8bff", ls="--", lw=0.8); a.set_xlabel("time from first detection (s)", fontsize=8); a.set_ylabel("mean D in mask", fontsize=8)
        a.tick_params(labelsize=7)
        a.set_title(f"within {STEP_S:.0f} s: {c['step_ratio']:.2f} of peak (≥ {STEP}?), last {TAIL_S:.0f} s: {c['tail_ratio']:.2f} of peak (< {TAIL}?)" + ("  → KEEP" if c["ok"] else "  → DROP"),
                    fontsize=8, color="k" if c["ok"] else "#d62728")
        fa = c["frame"]; Ra = preprocess(load(case, fa), mk)["res"]; Da = np.clip((Ra - Rb) / sig_b, 0, None)
        y, x, h = int(c["cy"]), int(c["cx"]), 40
        ya, xa = max(y - h, 0), max(x - h, 0); sl = (slice(ya, y + h), slice(xa, x + h))
        vm = np.percentile(np.r_[Rb0[sl].ravel(), Ra[sl].ravel()], 99.5)
        Mk = np.zeros((800, 800), bool); Mk[c["py"], c["px"]] = True
        for j, (img, tt, v) in enumerate([(Rb0[sl], f"before (median {on - int(round(3 * fps))}–{f_lo})", vm), (Ra[sl], f"after (frame {fa})", vm),
                                          (Da[sl], "(after − before) / σ_before", max(np.percentile(Da[sl], 99.5), 1))]):
            a = ax[i, j]; a.imshow(np.clip(img / v, 0, 1), cmap="gray")
            for cc in measure.find_contours(Mk[sl].astype(float), 0.5):
                a.plot(cc[:, 1], cc[:, 0], color="#3b8bff", lw=1.2)
            a.set_xticks([]); a.set_yticks([]); a.set_title(tt, fontsize=8)
        ax[i, 0].set_ylabel(f"{'/'.join(sorted(c['saps']))}  #{i + 1}\nframes {c['first']}–{c['extinguish']} (seen {c['n']})\narea {c['area']}, D {c['Dmean']:.0f}", fontsize=8)
    plt.suptitle(f"{case}: γ on sapphires (difference SAM, residual / σ_before = {sig_b:.0f}), onset {on} ({s['burst']})", fontsize=10)
    plt.tight_layout(); plt.savefig(f"fig/diffsam/{case}.png", dpi=90); plt.close()
pd.DataFrame(allc).to_csv("fig/paper/diffsam/diffsam_candidates.csv", index=False)
