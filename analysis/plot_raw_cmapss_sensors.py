

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

COLUMN_NAMES = (
    ["engine_id", "cycle", "setting1", "setting2", "setting3"]
    + [f"s{i}" for i in range(1, 22)]
)
SIGNAL_COLUMNS = ["setting1", "setting2", "setting3"] + [f"s{i}" for i in range(1, 22)]

SMALL_SIZE = 18
MEDIUM_SIZE = 22
LARGE_SIZE = 26
X_LARGE_SIZE = 32


def apply_style():
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['font.weight'] = 'bold'
    plt.rc('font', weight='bold')
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=LARGE_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=MEDIUM_SIZE)
    plt.rc('figure', titlesize=SMALL_SIZE)


def load_raw(data_path, dataset, split):
    fpath = os.path.join(data_path, f"{split}_{dataset}.txt")
    df = pd.read_table(fpath, header=None, sep=r"\s+")
    df = df.iloc[:, :len(COLUMN_NAMES)]
    df.columns = COLUMN_NAMES
    return df


def plot_all_signals(df, signal_cols=SIGNAL_COLUMNS, ncols=4, title=None):
    nrows = int(np.ceil(len(signal_cols) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for ax, col in zip(axes, signal_cols):
        for _, g in df.groupby("engine_id"):
            ax.plot(g["cycle"].values, g[col].values, linewidth=0.8)
        ax.set_title(col)
        ax.set_xlabel("time")


        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    for ax in axes[len(signal_cols):]:
        ax.axis("off")

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="D:/Datasets/CMAPSS")
    ap.add_argument("--dataset", default="FD001", choices=["FD001", "FD002", "FD003", "FD004"])
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out", default=None, help="output image path (default: analysis/out/raw_<dataset>_<split>.png)")
    ap.add_argument("--show", action="store_true", help="also open an interactive window")
    args = ap.parse_args()

    apply_style()
    df = load_raw(args.data_path, args.dataset, args.split)

    fig = plot_all_signals(
        df, ncols=args.ncols,
        title=f"{args.dataset} ({args.split}) -- raw signal trajectories, all engines",
    )

    out = args.out or os.path.join("analysis", "out", f"raw_{args.dataset}_{args.split}.png")
    out_dir = os.path.dirname(out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    stem, _ = os.path.splitext(out)

    png_path = stem + ".png"
    eps_path = stem + ".eps"
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(eps_path, format="eps", bbox_inches="tight")
    print(f"saved -> {png_path}")
    print(f"saved -> {eps_path}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
