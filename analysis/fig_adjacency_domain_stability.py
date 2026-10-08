

import argparse
import os
import sys
import itertools

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_

DOMAINS = ["FD001", "FD002", "FD003", "FD004"]


def time_cosine_adjacency(x):

    x = x - x.mean(dim=-1, keepdim=True)
    norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    xn = x / norm
    return torch.einsum("bcn,bdn->bcd", xn, xn)


def per_band_coherence(x, n_bands):

    X = torch.fft.rfft(x, dim=-1)
    F_len = X.shape[-1]
    bin_size = max(1, F_len // n_bands)
    mats = []
    for k in range(n_bands):
        Xb = X[..., k * bin_size:(k + 1) * bin_size]
        cross = torch.einsum("bif,bjf->bij", Xb, Xb.conj()).abs()
        auto = (Xb * Xb.conj()).real.sum(dim=-1)
        denom = (auto.unsqueeze(2) * auto.unsqueeze(1)).clamp(min=1e-8).sqrt()
        mats.append(cross / denom)
    return torch.stack(mats, dim=1)


def offdiag_vec(A):

    C = A.shape[-1]
    mask = ~np.eye(C, dtype=bool)
    return np.asarray(A)[mask]


def cos_sim(u, v):

    u = u - u.mean()
    v = v - v.mean()
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return np.nan
    return float(u @ v / (nu * nv))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="D:/Datasets/CMAPSS")
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--maxlife", type=int, default=125)
    ap.add_argument("--n_bands", type=int, default=4)
    ap.add_argument("--n_bins", type=int, default=5)
    ap.add_argument("--max_windows", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)


    mean_adj = {}
    bin_edges = np.linspace(0.0, args.maxlife, args.n_bins + 1)
    bin_edges[-1] += 1e-6

    for dom in DOMAINS:
        print(f"[load] {dom}", flush=True)
        X_train, y_train, _, X_vali, y_vali, _, _, _, _, _ = get_cmapss_data_(
            data_path=args.data_path, Data_id=dom, test_Data_id=dom,
            sequence_length=args.seq_len, MAXLIFE=args.maxlife,
            is_difference=False, validation=0.1)

        X = np.concatenate([X_train, X_vali], axis=0)
        y = np.concatenate([y_train, y_vali], axis=0).reshape(-1)

        if len(X) > args.max_windows:
            sel = rng.choice(len(X), args.max_windows, replace=False)
            X, y = X[sel], y[sel]

        xt = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        with torch.no_grad():
            A_time = time_cosine_adjacency(xt).numpy()
            A_band = per_band_coherence(xt, args.n_bands).numpy()

        for b in range(args.n_bins):
            m = (y >= bin_edges[b]) & (y < bin_edges[b + 1])
            if m.sum() < 10:
                continue
            mean_adj[(dom, b, "time")] = A_time[m].mean(axis=0)
            mean_adj[(dom, b, "band_mix")] = A_band[m].mean(axis=(0, 1))
            for k in range(args.n_bands):
                mean_adj[(dom, b, f"band_{k}")] = A_band[m, k].mean(axis=0)
        print(f"[done] {dom}: {len(X)} windows", flush=True)

    descs = ["time", "band_mix"] + [f"band_{k}" for k in range(args.n_bands)]
    pairs = list(itertools.combinations(DOMAINS, 2))


    rows = []
    for a, b in pairs:
        for desc in descs:
            sims = []
            for bin_i in range(args.n_bins):
                ka, kb = (a, bin_i, desc), (b, bin_i, desc)
                if ka in mean_adj and kb in mean_adj:
                    sims.append(cos_sim(offdiag_vec(mean_adj[ka]),
                                        offdiag_vec(mean_adj[kb])))
            rows.append({"pair": f"{a[2:]}-{b[2:]}", "descriptor": desc,
                         "cross_domain_cos": float(np.nanmean(sims)),
                         "n_bins_used": len(sims)})


    snr_rows = []
    for desc in descs:
        sig, nui = [], []
        for dom in DOMAINS:
            for b1, b2 in itertools.combinations(range(args.n_bins), 2):
                k1, k2 = (dom, b1, desc), (dom, b2, desc)
                if k1 in mean_adj and k2 in mean_adj:
                    sig.append(1 - cos_sim(offdiag_vec(mean_adj[k1]),
                                           offdiag_vec(mean_adj[k2])))
        for a, b in pairs:
            for bin_i in range(args.n_bins):
                ka, kb = (a, bin_i, desc), (b, bin_i, desc)
                if ka in mean_adj and kb in mean_adj:
                    nui.append(1 - cos_sim(offdiag_vec(mean_adj[ka]),
                                           offdiag_vec(mean_adj[kb])))
        s, n = float(np.nanmean(sig)), float(np.nanmean(nui))
        snr_rows.append({"descriptor": desc, "lifecycle_signal": s,
                         "domain_nuisance": n,
                         "signal_to_nuisance": s / max(n, 1e-9)})

    import pandas as pd
    snr_df = pd.DataFrame(snr_rows)
    snr_csv = os.path.join(args.out, "adjacency_lifecycle_snr.csv")
    snr_df.to_csv(snr_csv, index=False)
    print("\nlifecycle signal vs domain nuisance (1 - Pearson r of pattern):")
    print(snr_df.round(4).to_string(index=False))
    print(f"saved: {snr_csv}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, "adjacency_domain_stability.csv")
    df.to_csv(csv_path, index=False)

    piv = df.pivot(index="pair", columns="descriptor", values="cross_domain_cos")
    print("\ncross-domain cosine similarity of mean adjacency (higher = more stable)")
    print(piv.round(4).to_string())
    print(f"\nmean over pairs:\n{piv.mean().round(4).to_string()}")


    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10,
                         "mathtext.fontset": "stix"})

    DESC_LABEL = {"time": "time-domain\ncosine",
                  "band_mix": "band coherence\n(uniform mix)"}
    DESC_LABEL.update({f"band_{k}": f"band {k}" for k in range(args.n_bands)})
    COL_TIME, COL_BAND, COL_MIX = "#8896AB", "#7CB342", "#2E7D32"

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.0, 3.6),
                                   gridspec_kw={"width_ratios": [1.15, 1]})


    LABEL_OFFSET = {"time": (8, -3), "band_mix": (-10, -4),
                    "band_0": (8, -3), "band_1": (8, 2),
                    "band_2": (-14, 8), "band_3": (8, -2)}
    for _, r in snr_df.iterrows():
        d = r["descriptor"]
        c = COL_TIME if d == "time" else (COL_MIX if d == "band_mix" else COL_BAND)
        marker = "s" if d == "time" else ("D" if d == "band_mix" else "o")
        axA.scatter(r["domain_nuisance"], r["lifecycle_signal"],
                    s=55, color=c, marker=marker, zorder=3)
        axA.annotate(DESC_LABEL[d].replace("\n", " "),
                     (r["domain_nuisance"], r["lifecycle_signal"]),
                     textcoords="offset points",
                     xytext=LABEL_OFFSET.get(d, (6, 4)), fontsize=8,
                     ha="right" if LABEL_OFFSET.get(d, (6, 4))[0] < 0 else "left")
    lim = max(snr_df["domain_nuisance"].max(),
              snr_df["lifecycle_signal"].max()) * 1.15
    for ratio, ls in [(0.25, ":"), (0.5, "--")]:
        axA.plot([0, lim], [0, lim * ratio], ls, color="0.75", lw=0.8, zorder=1)
        axA.annotate(f"ratio {ratio}", (lim * 0.97, lim * ratio * 0.97),
                     fontsize=7, color="0.45", ha="right", va="bottom")
    axA.set_xlim(0, lim)
    axA.set_ylim(0, snr_df["lifecycle_signal"].max() * 1.35)
    axA.set_xlabel("domain nuisance  (pattern shift across domains,\nsame lifecycle bin)")
    axA.set_ylabel("lifecycle signal\n(pattern shift across lifecycle bins,\nsame domain)")
    axA.set_title("(a) Adjacency substrate: lifecycle signal vs domain nuisance")
    axA.spines[["top", "right"]].set_visible(False)


    order = ["time", "band_mix"] + [f"band_{k}" for k in range(args.n_bands)]
    vals = [snr_df.set_index("descriptor").loc[d, "signal_to_nuisance"]
            for d in order]
    cols = [COL_TIME] + [COL_MIX] + [COL_BAND] * args.n_bands
    xpos = np.arange(len(order))
    axB.bar(xpos, vals, 0.62, color=cols)
    for x, v in zip(xpos, vals):
        axB.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                     xytext=(0, 2), ha="center", fontsize=8)
    axB.set_xticks(xpos)
    axB.set_xticklabels(["time-domain\ncosine", "band\nmix"]
                        + [f"band\n{k}" for k in range(args.n_bands)],
                        fontsize=8)
    axB.set_ylim(0, max(vals) * 1.12)
    axB.set_ylabel("lifecycle signal / domain nuisance")
    axB.set_title("(b) Lifecycle information\nper unit of domain shift")
    axB.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig_path = os.path.join(args.out, "fig_adjacency_lifecycle_snr.png")
    fig.savefig(fig_path, dpi=300)
    fig.savefig(fig_path.replace(".png", ".pdf"))
    print(f"\nsaved: {csv_path}\nsaved: {fig_path} (+pdf)")


if __name__ == "__main__":
    main()
