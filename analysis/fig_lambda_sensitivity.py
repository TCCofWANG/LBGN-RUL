

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

from fig_error_distribution import apply_house_style

INFO_RE = re.compile(r"lmar([\d.]+)_lda([\d.]+)_s(\d+)")


def load_all(logs_dir):
    files = sorted(glob.glob(os.path.join(logs_dir, "*experimental_logs.csv")))
    if not files:
        raise FileNotFoundError(f"no *experimental_logs.csv files found under {logs_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        df["pair"] = os.path.basename(f).replace("experimental_logs.csv", "")
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)

    parsed = full["info"].astype(str).str.extract(INFO_RE)
    parsed.columns = ["lam_mar", "lam_da", "seed"]
    full = pd.concat([full, parsed], axis=1)
    full = full.dropna(subset=["lam_mar", "lam_da"])
    full["lam_mar"] = full["lam_mar"].astype(float)
    full["lam_da"] = full["lam_da"].astype(float)


    n_before = len(full)
    full = full.drop_duplicates(subset=["pair", "lam_mar", "lam_da", "seed"], keep="last")
    n_dropped = n_before - len(full)
    if n_dropped:
        print(f"[load_all] dropped {n_dropped} duplicate (pair, lam_mar, lam_da, seed) row(s)")

    return full


def make_bar_panel(plt, df, value_col, ylabel, title, out_path):
    grouped = df.groupby(["lam_mar", "lam_da"])[value_col].agg(["mean", "std", "count"]).reset_index()
    grouped = grouped.sort_values(["lam_mar", "lam_da"]).reset_index(drop=True)

    x = np.arange(len(grouped))
    labels = [f"$\\lambda_{{mar}}$={r.lam_mar:g}\n$\\lambda_{{da}}$={r.lam_da:g}" for r in grouped.itertuples()]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.bar(x, grouped["mean"], yerr=grouped["std"], width=0.55,
           color="#4C72B0", edgecolor="#4C72B0", alpha=0.85,
           capsize=6, error_kw={"elinewidth": 1.5, "capthick": 1.5})

    for xi, m in zip(x, grouped["mean"]):
        ax.annotate(f"{m:.1f}", (xi, m), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=14, weight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=16, weight="bold")
    ax.set_title(title)
    n = int(grouped["count"].iloc[0])
    ax.text(0.5, -0.22, f"mean $\\pm$ std over {n} CMAPSS transfer pairs, seed 42",
            transform=ax.transAxes, ha="center", fontsize=12)

    plt.tight_layout()
    fig.savefig(out_path + ".png", dpi=300)
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".eps")
    plt.close(fig)
    return grouped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs_lambda_sensitivity")
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_house_style(plt)

    os.makedirs(args.out, exist_ok=True)
    full = load_all(args.logs_dir)

    print(f"Loaded {len(full)} rows across {full['pair'].nunique()} CMAPSS pairs, "
          f"{full[['lam_mar','lam_da']].drop_duplicates().shape[0]} (lam_mar, lam_da) points, "
          f"seeds {sorted(full['seed'].unique())}")

    rmse_path = os.path.join(args.out, "fig_lambda_sensitivity_rmse")
    score_path = os.path.join(args.out, "fig_lambda_sensitivity_score")

    rmse_grouped = make_bar_panel(plt, full, "best_last_RMSE", "RMSE",
                                   "CMAPSS: LBGN-RUL sensitivity to $\\lambda_{mar}$/$\\lambda_{da}$", rmse_path)
    score_grouped = make_bar_panel(plt, full, "score", "Score",
                                    "CMAPSS: LBGN-RUL sensitivity to $\\lambda_{mar}$/$\\lambda_{da}$", score_path)

    print(f"\nsaved: {rmse_path}.png / .pdf / .eps")
    print(f"saved: {score_path}.png / .pdf / .eps")

    print("\nRMSE by (lam_mar, lam_da):")
    print(rmse_grouped.round(3).to_string(index=False))
    print("\nScore by (lam_mar, lam_da):")
    print(score_grouped.round(3).to_string(index=False))

    rmse_range = rmse_grouped["mean"].max() - rmse_grouped["mean"].min()
    rmse_rel = 100 * rmse_range / rmse_grouped["mean"].min()
    print(f"\nRMSE mean range across the 5 tested points: {rmse_range:.3f} "
          f"({rmse_rel:.1f}% relative to the lowest)")
    print("NOTE: range redesigned (0.5,0.1) -> (5.0,5.0) to include the actual "
          "production values lam_mar=5, lam_da=3 from configs/hparams.py, tested "
          "here as the (5.0, 3.0) point.")


if __name__ == "__main__":
    main()
