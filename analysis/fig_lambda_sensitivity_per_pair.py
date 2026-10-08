

import argparse
import os

import numpy as np

from fig_error_distribution import apply_house_style
from fig_lambda_sensitivity import load_all

PAIRS = [
    ("FD001", "FD002"), ("FD001", "FD003"), ("FD001", "FD004"),
    ("FD002", "FD001"), ("FD002", "FD003"), ("FD002", "FD004"),
    ("FD003", "FD001"), ("FD003", "FD002"), ("FD003", "FD004"),
    ("FD004", "FD001"), ("FD004", "FD002"), ("FD004", "FD003"),
]


COMBO_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]


COMBO_HATCHES = ["", "///", "xxx", "...", "\\\\\\"]


def abbr(src, tgt):
    return f"F{src[-1]}$\\to$F{tgt[-1]}"


def make_grouped_panel(plt, full, value_col, ylabel, out_path):
    combos = sorted(full[["lam_mar", "lam_da"]].drop_duplicates().itertuples(index=False, name=None))
    n_combos = len(combos)
    pair_keys = [f"{s}_{t}" for s, t in PAIRS]
    pair_labels = [abbr(s, t) for s, t in PAIRS]

    data = np.full((n_combos, len(PAIRS)), np.nan)
    for ci, (lm, ld) in enumerate(combos):
        sub = full[(full["lam_mar"] == lm) & (full["lam_da"] == ld)]
        by_pair = sub.groupby("pair")[value_col].mean()
        for pi, pk in enumerate(pair_keys):
            if pk in by_pair.index:
                data[ci, pi] = by_pair[pk]

    n_groups = len(PAIRS)
    group_width = 0.8
    bar_width = group_width / n_combos
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(18, 8))
    for ci, (lm, ld) in enumerate(combos):
        offset = (ci - (n_combos - 1) / 2) * bar_width
        ax.bar(x + offset, data[ci], width=bar_width * 0.92,
               color=COMBO_COLORS[ci % len(COMBO_COLORS)],
               edgecolor="black", linewidth=0.7,
               hatch=COMBO_HATCHES[ci % len(COMBO_HATCHES)],
               label=f"{lm:g}, {ld:g}")

    ax.set_xticks(x)
    ax.set_xticklabels(pair_labels, fontsize=20)
    ax.tick_params(axis="y", labelsize=20)
    ax.set_ylabel(ylabel, fontsize=26, weight="bold")
    ax.legend(title="$\\lambda_{mar}$, $\\lambda_{da}$", ncol=n_combos, loc="upper center",
               bbox_to_anchor=(0.5, 1.17), fontsize=20, title_fontsize=20, frameon=False)

    plt.tight_layout()
    fig.savefig(out_path + ".png", dpi=300)
    fig.savefig(out_path + ".pdf")
    fig.savefig(out_path + ".eps")
    plt.close(fig)
    return combos, data, pair_labels


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

    rmse_path = os.path.join(args.out, "fig_lambda_sensitivity_per_pair_rmse")
    score_path = os.path.join(args.out, "fig_lambda_sensitivity_per_pair_score")

    combos, rmse_data, pair_labels = make_grouped_panel(plt, full, "best_last_RMSE", "RMSE", rmse_path)
    _, score_data, _ = make_grouped_panel(plt, full, "score", "Score", score_path)

    print(f"saved: {rmse_path}.png / .pdf / .eps")
    print(f"saved: {score_path}.png / .pdf / .eps")

    print("\nRMSE per pair per (lam_mar, lam_da):")
    header = "pair".ljust(10) + "".join(f"({lm:g},{ld:g})".rjust(12) for lm, ld in combos)
    print(header)
    for pi, pl in enumerate(pair_labels):
        row = pl.replace("$\\to$", "->").ljust(10) + "".join(f"{rmse_data[ci,pi]:12.2f}" for ci in range(len(combos)))
        print(row)


if __name__ == "__main__":
    main()
