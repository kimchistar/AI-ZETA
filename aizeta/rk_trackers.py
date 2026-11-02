import numpy as np, pandas as pd
from scipy.optimize import linear_sum_assignment
from rk_params import P


def _finish(rows, min_obs):
    T = pd.DataFrame(rows)
    keep = T.groupby("tid").frame.transform("size") >= min_obs
    return T[keep].reset_index(drop=True)


def track_hungarian(det, gate=12.0, max_gap=6, min_obs=P["min_obs"]):
    rows, active, last, nxt = [], {}, {}, 0
    for f, D in det.groupby("frame", sort=True):
        ids = [t for t, lf in active.items() if f - lf <= max_gap]
        Q = D[["cy", "cx"]].values
        used = set()
        if ids and len(D):
            Pp = np.array([last[t] for t in ids])
            C = np.linalg.norm(Pp[:, None] - Q[None], axis=2)
            C[C > gate] = 1e6
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6:
                    t = ids[i]; r = D.iloc[j].to_dict(); r["tid"] = t
                    rows.append(r); active[t] = f; last[t] = Q[j]; used.add(j)
        for j in range(len(D)):
            if j in used:
                continue
            r = D.iloc[j].to_dict(); r["tid"] = nxt
            rows.append(r); active[nxt] = f; last[nxt] = Q[j]; nxt += 1
    return _finish(rows, min_obs)


class _KF:
    def __init__(self, y, x):
        self.s = np.array([y, x, 0.0, 0.0])
        self.P = np.diag([4.0, 4.0, 25.0, 25.0])
        self.Q = np.diag([0.5, 0.5, 1.0, 1.0])
        self.R = np.diag([1.0, 1.0])
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)

    def predict(self, dt):
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        self.s = F @ self.s
        self.P = F @ self.P @ F.T + self.Q * dt
        return self.s[:2]

    def update(self, z):
        y = z - self.H @ self.s
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P


def track_sort(det, gate=12.0, max_gap=6, min_obs=P["min_obs"]):
    rows, kf, lastf, nxt = [], {}, {}, 0
    for f, D in det.groupby("frame", sort=True):
        ids = [t for t in kf if f - lastf[t] <= max_gap]
        Q = D[["cy", "cx"]].values
        used = set()
        if ids and len(D):
            pred = np.array([kf[t].predict(f - lastf[t]) for t in ids])
            C = np.linalg.norm(pred[:, None] - Q[None], axis=2)
            C[C > gate] = 1e6
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < 1e6:
                    t = ids[i]; kf[t].update(Q[j]); lastf[t] = f
                    r = D.iloc[j].to_dict(); r["tid"] = t; rows.append(r); used.add(j)
        for j in range(len(D)):
            if j in used:
                continue
            kf[nxt] = _KF(*Q[j]); lastf[nxt] = f
            r = D.iloc[j].to_dict(); r["tid"] = nxt; rows.append(r); nxt += 1
    return _finish(rows, min_obs)


def track_trackpy(det, search_range=12.0, memory=6, min_obs=P["min_obs"]):
    import trackpy as tp
    tp.quiet()
    d = det.rename(columns={"cy": "y", "cx": "x"}).copy()
    L = tp.link_df(d, search_range=search_range, memory=memory)
    L = L.rename(columns={"y": "cy", "x": "cx", "particle": "tid"})
    keep = L.groupby("tid").frame.transform("size") >= min_obs
    return L[keep].reset_index(drop=True)


def track_site(det, r_site=P["r_site"], min_obs=P["min_obs"], max_gap=6):
    from scipy.spatial import cKDTree
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    xy = det[["cy", "cx"]].values
    n = len(xy)
    pairs = cKDTree(xy).query_pairs(r_site, output_type="ndarray")
    G = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _, lab = connected_components(G, directed=False)
    out = det.assign(site=lab)
    out = out.sort_values("I_sum", ascending=False).drop_duplicates(["site", "frame"]).sort_values(["site", "frame"])
    gap = out.groupby("site").frame.diff().fillna(0) > max_gap
    ep = gap.groupby(out.site).cumsum()
    out["tid"] = out.site.astype(np.int64) * 1000 + ep.astype(np.int64)
    out = out.sort_values("frame")
    keep = out.groupby("tid").frame.transform("size") >= min_obs
    return out[keep].reset_index(drop=True)


def _on_sample_index(fn):
    def wrapped(det, **kw):
        fr = np.sort(det.frame.unique())
        idx = {f: i for i, f in enumerate(fr)}
        d = det.assign(frame_raw=det.frame.values, frame=det.frame.map(idx).values)
        out = fn(d, **kw)
        out["frame"] = out["frame_raw"].astype(int)
        return out.drop(columns=["frame_raw"])
    return wrapped


TRACKERS = dict(hungarian=_on_sample_index(track_hungarian), sort=_on_sample_index(track_sort),
                trackpy=_on_sample_index(track_trackpy), site=_on_sample_index(track_site))
