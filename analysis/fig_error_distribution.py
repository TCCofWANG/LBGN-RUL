

import argparse
import os

import numpy as np
from matplotlib.transforms import Bbox


CONFIGS = {
    "cmapss": [
        ("best", "CMAPSS", "F4$\\to$F1 (best)", "FD004_FD001", [("LBGN_RUL", "LBGN (ours)"), ("DAST_RUL", "DAST")]),
        ("worst", "CMAPSS", "F2$\\to$F3 (worst)", "FD002_FD003", [("LBGN_RUL", "LBGN (ours)"), ("OCS_DANN", "OCS-DANN")]),
    ],
    "ncmapss": [
        ("best", "N-CMAPSS", "D4$\\to$D1 (best)", "DS04_DS01", [("LBGN_RUL", "LBGN (ours)"), ("CCDG_RUL", "CCDG")]),
        ("worst", "N-CMAPSS", "D1$\\to$D3 (worst)", "DS01_DS03", [("LBGN_RUL", "LBGN (ours)"), ("OCS_DANN", "OCS-DANN")]),
    ],
}


DESIRED_TICKS = [-52, -39, -26, -13, 0, 10, 20, 30, 40, 50]


def apply_house_style(plt):
    SMALL_SIZE = 18
    MEDIUM_SIZE = 22
    LARGE_SIZE = 26
    X_LARGE_SIZE = 32
    plt.rcParams['font.family'] = ['Times New Roman', 'serif']
    plt.rcParams['font.weight'] = 'bold'
    plt.rc('font', weight='bold')
    plt.rc('font', size=SMALL_SIZE)
    plt.rc('axes', titlesize=SMALL_SIZE)
    plt.rc('axes', labelsize=SMALL_SIZE)
    plt.rc('xtick', labelsize=SMALL_SIZE)
    plt.rc('ytick', labelsize=SMALL_SIZE)
    plt.rc('legend', fontsize=SMALL_SIZE)
    plt.rc('figure', titlesize=SMALL_SIZE)
    return LARGE_SIZE


def _deconflict_labels(fig, anns, margin_px=4, max_passes=10):

    anns = [a for a in anns if a is not None]
    if not anns:
        return
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for _ in range(max_passes):
        boxes = [(a, a.get_window_extent(renderer)) for a in anns]
        boxes.sort(key=lambda t: t[1].x0)
        moved = False
        for i in range(len(boxes) - 1):
            a, bbox_a = boxes[i]
            b, bbox_b = boxes[i + 1]
            expanded_a = Bbox.from_extents(bbox_a.x0, bbox_a.y0, bbox_a.x1 + margin_px, bbox_a.y1)
            if expanded_a.overlaps(bbox_b):
                dx, dy = b.xyann
                b.xyann = (dx, dy + bbox_b.height + margin_px)
                moved = True
        if not moved:
            break
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()


def load_errors(logs_dir, pair_dir, model_dir, seed):
    path = os.path.join(logs_dir, pair_dir, model_dir, f"final_s{seed}", "result.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path)
    preds = data["test_preds"].reshape(-1)
    trues = data["test_trues"].reshape(-1)
    return preds - trues


def make_panel_figure(plt, dataset_tag, which, dataset_disp, pair_disp, pair_dir, models, logs_dir, seed, out_dir):
    (ours_dir, ours_label), (base_dir, base_label) = models
    error_all = load_errors(logs_dir, pair_dir, ours_dir, seed)
    error_baseline_all = load_errors(logs_dir, pair_dir, base_dir, seed)

    bins = np.array(DESIRED_TICKS)
    counts, _ = np.histogram(error_all, bins=bins)
    baseline_counts, _ = np.histogram(error_baseline_all, bins=bins)


    percentages = 100 * counts / len(error_all)
    baseline_percentages = 100 * baseline_counts / len(error_baseline_all)

    bin_widths = np.diff(bins)
    bar_widths = bin_widths * 0.42
    gap = bin_widths * 0.08

    fig, ax = plt.subplots(figsize=(9, 6.5))

    ax.axvspan(-13, 10, color='#36AE7C', alpha=0.2, zorder=0)


    bars_model = ax.bar(
        bins[:-1] + gap / 2, percentages, width=bar_widths,
        color='#4C72B0', edgecolor='black', linewidth=0.7, hatch='',
        align='edge', alpha=0.75, label=ours_label,
    )
    bars_baseline = ax.bar(
        bins[:-1] + bar_widths + gap, baseline_percentages, width=bar_widths,
        color='#EB5353', edgecolor='black', linewidth=0.7, hatch='///',
        align='edge', alpha=0.75, label=base_label,
    )

    label_anns = []
    for bar, pct in zip(bars_model, percentages):
        if pct > 0:
            label_anns.append(ax.annotate(f'{pct:.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 4), ha='center', fontsize=14, weight='bold'))
    for bar, pct in zip(bars_baseline, baseline_percentages):
        if pct > 0:
            label_anns.append(ax.annotate(f'{pct:.1f}%', (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       textcoords="offset points", xytext=(0, 4), ha='center', fontsize=14, weight='bold'))


    ax.set_xlabel("Prediction error", fontsize=16, weight='bold')
    ax.set_xticks(DESIRED_TICKS)
    ax.set_xlim([-54, 52])
    ax.tick_params(axis='x', which='major', labelsize=14)
    ax.set_title(f"{dataset_disp}: {pair_disp}")

    ax.set_ylabel("Predictions (%)")
    ax.set_yticks([])

    ax.legend(fontsize=16, frameon=False, loc='upper right')
    plt.tight_layout()
    _deconflict_labels(fig, label_anns)

    base_path = os.path.join(out_dir, f"fig_error_distribution_{dataset_tag}_{which}")
    fig.savefig(base_path + ".png", dpi=300)
    fig.savefig(base_path + ".pdf")
    fig.savefig(base_path + ".eps")
    plt.close(fig)

    shaded_idx = [i for i, (lo, hi) in enumerate(zip(bins[:-1], bins[1:])) if lo >= -13 and hi <= 10]
    shaded_ours = float(percentages[shaded_idx].sum())
    shaded_base = float(baseline_percentages[shaded_idx].sum())
    return base_path, (ours_label, shaded_ours, base_label, shaded_base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cmapss_logs_dir", required=True)
    ap.add_argument("--ncmapss_logs_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="analysis/out")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    apply_house_style(plt)

    os.makedirs(args.out, exist_ok=True)
    logs_dirs = {"cmapss": args.cmapss_logs_dir, "ncmapss": args.ncmapss_logs_dir}
    all_reports = []

    for dataset_tag, panels in CONFIGS.items():
        for which, dataset_disp, pair_disp, pair_dir, models in panels:
            base_path, stats = make_panel_figure(
                plt, dataset_tag, which, dataset_disp, pair_disp, pair_dir, models,
                logs_dirs[dataset_tag], args.seed, args.out,
            )
            print(f"saved: {base_path}.png / .pdf / .eps")
            all_reports.append((dataset_tag, pair_disp, *stats))

    print("\nIn-[-13,10]-region proportions (normalized by TRUE total, not in-range count):")
    for dataset_tag, pair_disp, ours_label, ours_pct, base_label, base_pct in all_reports:
        print(f"  {dataset_tag:9s} {pair_disp:22s} {ours_label}={ours_pct:.1f}%  {base_label}={base_pct:.1f}%")


if __name__ == "__main__":
    main()
