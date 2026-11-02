"""Detection videos: every labeled reflection drawn as its SAM ellipse, colored by label.
A δ that is not detected in a frame but has not yet disappeared is drawn as a dashed circle.
Part 1 shows the whole run once per second; part 2 shows every frame around the transformation.
Run from the folder that holds rk_out/, the mask_<run>.npy files and delta_end.json."""
import sys, os, json; sys.path[:0] = [".", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aizeta")]
import numpy as np, pandas as pd, cv2
import matplotlib
from PIL import Image, ImageDraw, ImageFont
from common import load
from prep import preprocess
from rk_thermal import temperature_on_frames
from track03_kinetics import frame_times

RUNS = {"2_TRIP1": "TRIP T1, 110 °C/min", "8_TRIP2": "TRIP T2, 30 °C/min", "31_A003": "ULC, 132 °C/min",
        "26_V19_5A": "Steel C, 110 °C/min", "21_V19_1A": "Steel C, 30 °C/min"}
COL = {"delta": (224, 59, 59), "gamma": (59, 110, 224), "other": (150, 150, 150)}
FONT = os.path.join(matplotlib.get_data_path(), "fonts", "ttf", "DejaVuSans.ttf")
SIDE = 720


def phases(case):
    U = pd.read_csv(f"rk_out/{case}_site_full_units.csv")
    L = pd.read_csv(f"rk_out/{case}_labels.csv")
    lab = {(str(r.site), r.phase): r.label for r in L.itertuples()}
    END = json.load(open("delta_end.json", encoding="utf-8"))[case]
    clean = None
    if os.path.exists(f"rk_out/{case}_clean_delta.json"):
        clean = set(json.load(open(f"rk_out/{case}_clean_delta.json", encoding="utf-8"))["clean"])
    info = {}
    for u in U.itertuples():
        ph = u.phase if u.phase in ("delta", "gamma") else "other"
        name = lab.get((str(int(u.site)), u.phase), "")
        last = None
        if ph == "delta":
            if clean is not None and name not in clean:
                continue
            last = END["per_delta"].get(name, {}).get("end")
            if last is None:
                last = int(L.loc[L.label == name, "last_frame"].iloc[0]) if name in set(L.label) else None
        info[int(u.uid)] = (ph, name if ph == "delta" else "", last)
    return info, END, U


def render(case, f, tf, T, t_star, mk, crop, D, alive, extra, font, small):
    y0, y1, x0, x1 = crop
    u = preprocess(load(case, int(f)), mk)["u"][y0:y1, x0:x1]
    s = SIDE / max(u.shape)
    img = cv2.resize((np.clip(u / 0.8, 0, 1) * 255).astype(np.uint8), None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    texts = []
    for r in D.get(f, []):
        ph, name, cy, cx, M, m, o = r
        c = (int(round((cx - x0) * s)), int(round((cy - y0) * s)))
        ax = (max(2, int(round(M / 2 * s))), max(2, int(round(m / 2 * s))))
        cv2.ellipse(img, c, ax, 90 - np.degrees(o), 0, 360, COL[ph], 2 if ph != "other" else 1, cv2.LINE_AA)
        if name:
            texts.append((c[0] + min(ax) + 4, c[1] - 10, name, COL[ph]))
    for name, cy, cx, rad in alive.get(f, []):
        c = (int(round((cx - x0) * s)), int(round((cy - y0) * s))); rr = max(3, int(round(rad * s)))
        for a in range(0, 360, 30):
            cv2.ellipse(img, c, (rr, rr), 0, a, a + 15, COL["delta"], 2, cv2.LINE_AA)
        texts.append((c[0] + rr + 3, c[1] - 10, name, COL["delta"]))
    for cy, cx, rad in extra.get(f, []):
        cv2.circle(img, (int(round((cx - x0) * s)), int(round((cy - y0) * s))), max(3, int(round(rad * s))), COL["gamma"], 2, cv2.LINE_AA)
    pil = Image.new("RGB", (img.shape[1], img.shape[0] + 72), (0, 0, 0))
    pil.paste(Image.fromarray(img), (0, 72))
    d = ImageDraw.Draw(pil)
    for x, y, t, c in texts:
        d.text((x, y + 72), t, fill=c, font=small)
    dt = tf[int(f)] - tf[int(t_star)]
    temp = "" if T is None or not np.isfinite(T[int(f)]) else f"   {T[int(f)]:.0f} °C"
    d.text((10, 6), f"{RUNS[case]}   frame {int(f)}   t − t* = {dt:+.2f} s{temp}", fill=(255, 255, 255), font=font)
    xx = 10
    for k, nm in (("delta", "δ"), ("gamma", "γ"), ("other", "non-specimen")):
        d.ellipse((xx, 42, xx + 16, 58), outline=COL[k], width=2); d.text((xx + 22, 40), nm, fill=(255, 255, 255), font=small); xx += 60 if k != "other" else 0
    return np.asarray(pil)


class writer:
    def __init__(self, path, size, fps):
        import imageio_ffmpeg
        self.g = imageio_ffmpeg.write_frames(path, size, fps=fps, codec="libx264", quality=5, macro_block_size=8,
                                             output_params=["-pix_fmt", "yuv420p"])
        self.g.send(None)

    def write(self, frame):
        self.g.send(np.ascontiguousarray(frame))

    def release(self):
        self.g.close()


def make(case, out_dir):
    info, END, U = phases(case)
    obs = pd.read_csv(f"rk_out/{case}_site_full_obs.csv")
    D = {}
    for r in obs.itertuples():
        i = info.get(int(r.tid))
        if i is None or (i[2] is not None and r.frame > i[2]):
            continue
        D.setdefault(int(r.frame), []).append((i[0], i[1], r.cy, r.cx, r.major, r.minor, r.orient))
    alive = {}
    for u in U.itertuples():
        i = info.get(int(u.uid))
        if i is None or i[0] != "delta":
            continue
        seen = set(obs.frame[obs.tid == u.uid])
        for f in range(int(u.ignite), int(i[2] if i[2] is not None else u.extinguish) + 1):
            if f not in seen:
                alive.setdefault(f, []).append((i[1], u.cy, u.cx, np.sqrt(u.area / np.pi)))
    extra = {}
    if os.path.exists(f"rk_out/{case}_sap_gamma.csv"):
        for g in pd.read_csv(f"rk_out/{case}_sap_gamma.csv").itertuples():
            for f in range(int(g.first_frame), int(g.last_frame) + 1):
                extra.setdefault(f, []).append((g.cy, g.cx, np.sqrt(g.area / np.pi)))
    mk = np.load(f"mask_{case}.npy"); ys, xs = np.where(mk); crop = (ys.min(), ys.max() + 1, xs.min(), xs.max() + 1)
    tf = frame_times(case); fps = int(round(1 / np.median(np.diff(tf))))
    th = temperature_on_frames(case); T = np.asarray(th["T"]) if th.get("synced") else None
    t_star = int(END["first_gamma"])
    RT = json.load(open("refine_timing.json", encoding="utf-8"))[case]
    w0 = min(t_star, int(RT["transformation"]["start"])) - 3 * fps
    w1 = int(max(END["last_delta_end"], t_star)) + 2 * fps
    f_all = np.sort(obs.frame.unique())
    font = ImageFont.truetype(FONT, 20); small = ImageFont.truetype(FONT, 16)
    parts = {"overview": (f_all[::fps], 10), "transformation": (np.arange(max(w0, f_all[0]), min(w1, f_all[-1]) + 1), 5)}
    for nm, (frames, play) in parts.items():
        w = None
        for f in frames:
            fr = render(case, f, tf, T, t_star, mk, crop, D, alive, extra, font, small)
            if w is None:
                w = writer(os.path.join(out_dir, f"{case}_{nm}.mp4"), (fr.shape[1], fr.shape[0]), play)
            w.write(fr)
        w.release()
        print(case, nm, len(frames), "frames", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    for case in sys.argv[2:] or list(RUNS):
        make(case, out)
