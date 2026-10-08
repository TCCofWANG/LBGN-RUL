

import argparse
import glob
import os

import numpy as np
import pandas as pd

MODEL_LABELS = {
    "PDMN": "PDMN",
    "LBGN_RUL": "FiLMHeavy (ours)",
    "NDC_PDMN": "NDC-PDMN",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True,
                    help="folder containing *experimental_logs.csv")
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--models", nargs="*", default=list(MODEL_LABELS),
                    help="models to include (default: PDMN FiLMHeavy NDC)")
    args = ap.parse_args()

    files = glob.glob(os.path.join(args.logs_dir, "**", "*experimental_logs.csv"),
                      recursive=True)
    if not files:
        raise SystemExit(f"no *experimental_logs.csv found under {args.logs_dir}")
    print(f"found {len(files)} log files")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


    df = df[df.train_dataset.str.contains("_", na=False)].copy()
    df = df[df.model.isin(args.models)]


    best = (df.groupby(["train_dataset", "model", "hidden_dim"])["best_last_RMSE"]
              .min().reset_index())


    agg = (best.groupby(["model", "hidden_dim"])["best_last_RMSE"]
               .agg(mean="mean", worst="max", n="count").reset_index())

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "rmse_vs_capacity.csv")
    agg.to_csv(csv_path, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for model, g in agg.groupby("model"):
        g = g.sort_values("hidden_dim")
        label = MODEL_LABELS.get(model, model)
        axes[0].plot(g.hidden_dim, g["mean"], marker="o", linewidth=2, label=label)
        axes[1].plot(g.hidden_dim, g["worst"], marker="s", linewidth=2, label=label)

    axes[0].set_ylabel("RMSE (best per pair, averaged over pairs)")
    axes[0].set_title("Mean over transfer pairs")
    axes[1].set_ylabel("RMSE (worst transfer pair)")
    axes[1].set_title("Worst-case transfer pair")
    for ax in axes:
        ax.set_xlabel("hidden_dim")
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(agg.hidden_dim.unique()))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.grid(alpha=0.3)
        ax.legend(frameon=False)
    fig.suptitle("Structured lifecycle conditioning vs. model capacity")
    fig.tight_layout()

    png = os.path.join(args.out, "rmse_vs_capacity.png")
    fig.savefig(png, dpi=300)
    fig.savefig(png.replace(".png", ".pdf"))
    print(f"saved: {png} / .pdf / {csv_path}")
    print()
    print(agg.pivot_table(index="hidden_dim", columns="model",
                          values="mean").round(2).to_string())


if __name__ == "__main__":
    main()
