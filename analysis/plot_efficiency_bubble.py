

import csv
import os
import statistics as stats

import matplotlib.pyplot as plt
from adjustText import adjust_text
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO_ROOT, "figures")
FLOPS_CSV = os.path.join(REPO_ROOT, "analysis", "out", "flops_per_model.csv")


SMALL_SIZE = 18
MEDIUM_SIZE = 22
LARGE_SIZE = 26
X_LARGE_SIZE = 32

plt.rcParams["font.family"] = ["Times New Roman", "serif"]
plt.rcParams["font.weight"] = "bold"
plt.rc("font", weight="bold")
plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=MEDIUM_SIZE)
plt.rc("figure", titlesize=SMALL_SIZE)

EDGE_COLOR = "#52514e"


DISP = {"LBGN_RUL": "LBGN", "DAGCN_RUL": "DAGCN", "EviAdaptRUL": "EviAdapt",
        "DAST_RUL": "DAST", "CADA_RUL": "CADA", "TACDA_RUL": "TACDA",
        "OCS_DANN": "OCS-DANN", "MDAN_RUL": "MDAN", "CCDG_RUL": "CCDG"}


RMSE = {"LBGN_RUL": 18.15, "DAGCN_RUL": 26.61, "EviAdaptRUL": 25.86, "DAST_RUL": 25.74,
        "CADA_RUL": 25.95, "TACDA_RUL": 26.61, "OCS_DANN": 25.38, "MDAN_RUL": 25.73,
        "CCDG_RUL": 31.12}
SCORE = {"LBGN_RUL": 2007.75, "DAGCN_RUL": 31993.92, "EviAdaptRUL": 6093.46,
         "DAST_RUL": 4496.76, "CADA_RUL": 5467.99, "TACDA_RUL": 6546.00,
         "OCS_DANN": 9935.86, "MDAN_RUL": 5932.85, "CCDG_RUL": 22760.47}


def load_flops_stats():

    by_model = {}
    with open(FLOPS_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_model.setdefault(r["model"], []).append(r)
    rows = {}
    for model, recs in by_model.items():
        rows[model] = {
            "params_mean": stats.mean(float(r["params"]) for r in recs),
            "flops": stats.mean(float(r["total_macs_incl_fft"]) * 2 for r in recs),
        }
    return rows


def make_plot(y_values, y_label, out_stem, y_log=False):
    flops_data = load_flops_stats()
    models = list(DISP.keys())

    flops_m = [flops_data[m]["flops"] / 1e6 for m in models]
    parameters = [flops_data[m]["params_mean"] for m in models]
    y = [y_values[m] for m in models]
    bubble_size = [p / 100 for p in parameters]

    fig, ax = plt.subplots(figsize=(8, 6))

    scatter = ax.scatter(
        flops_m, y, s=bubble_size, alpha=0.6,
        c=parameters, cmap="plasma", edgecolors="k",
    )

    ax.set_xlabel("Total FLOPs (M)", fontsize=22)
    ax.set_ylabel(y_label, fontsize=22)
    if y_log:
        ax.set_yscale("log")


        ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(1, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="both", labelsize=14)

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Parameters Count", fontsize=22)
    cbar.ax.tick_params(labelsize=14)

    ax.grid(False)


    x_margin = (max(flops_m) - min(flops_m)) * 0.1
    y_margin = (max(y) - min(y)) * 0.35
    ax.set_xlim(min(flops_m) - x_margin, max(flops_m) + x_margin)
    if not y_log:
        ax.set_ylim(min(y) - y_margin, max(y) + y_margin)
    else:
        ax.set_ylim(min(y) / 1.8, max(y) * 2.4)


    texts = []
    for i in range(len(models)):
        offset = (bubble_size[i] ** 0.5) * 0.12
        texts.append(ax.text(flops_m[i], y[i] + offset, DISP[models[i]],
                              ha="center", va="bottom", fontsize=14))
    adjust_text(
        texts, ax=ax,
        expand_points=(1.8, 2.0), expand_text=(1.3, 1.5), force_text=(0.7, 0.9),
        force_points=(0.3, 0.35),
        arrowprops=dict(arrowstyle="-", color="#333333", lw=1.3, shrinkA=2, shrinkB=7),
    )

    fig.tight_layout()
    png_path = os.path.join(OUT_DIR, f"{out_stem}.png")
    eps_path = os.path.join(OUT_DIR, f"{out_stem}.eps")
    fig.savefig(png_path, format="png", dpi=300)
    fig.savefig(eps_path, format="eps", dpi=300)
    plt.close(fig)
    print(f"Saved: {png_path}")
    print(f"Saved: {eps_path}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    make_plot(RMSE, "Average RMSE", "Inference_complexity_rmse")
    make_plot(SCORE, "Average Score (log scale)", "Inference_complexity_score", y_log=True)


if __name__ == "__main__":
    main()
