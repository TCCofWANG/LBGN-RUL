

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

PAIR_RE = re.compile(r"^(FD00\d)_(FD00\d)$")


def score_compute(pred, gt):

    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    diff = pred - gt
    score_list = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.sum(score_list))


def rmse_compute(pred, gt):
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def find_runs(logs_dir, models, tag_prefix):

    tag_re = re.compile(r"^" + re.escape(tag_prefix) + r"(\d+)$")
    for pair_dir in sorted(glob.glob(os.path.join(logs_dir, "FD00?_FD00?"))):
        pair = os.path.basename(pair_dir)
        if not PAIR_RE.match(pair):
            continue
        for model_dir in sorted(glob.glob(os.path.join(pair_dir, "*"))):
            model = os.path.basename(model_dir)
            if models and model not in models:
                continue
            for run_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
                tag = os.path.basename(run_dir)
                m = tag_re.match(tag)
                if not m:
                    continue
                npz_path = os.path.join(run_dir, "result.npz")
                hparam_path = os.path.join(run_dir, "hparam.yaml")
                if not os.path.isfile(npz_path):
                    continue
                seed = int(m.group(1))
                yield {
                    "pair": pair,
                    "model": model,
                    "seed": seed,
                    "tag": tag,
                    "npz_path": npz_path,
                    "hparam_path": hparam_path,
                }


def load_metrics(run):
    d = np.load(run["npz_path"], allow_pickle=True)
    pred, gt = d["test_preds"], d["test_trues"]
    return {
        **run,
        "RMSE": rmse_compute(pred, gt),
        "score": score_compute(pred, gt),
    }


def fmt_cell(vals, is_best):
    mean, std = np.mean(vals), np.std(vals)
    s = f"{mean:.2f}±{std:.2f}"
    return s + " *" if is_best else s


def build_table(df, metric):
    pairs = sorted(df["pair"].unique())
    models = sorted(df["model"].unique())
    rows = []
    for pair in pairs:
        row = {"train_dataset": pair}
        sub = df[df["pair"] == pair]
        means = {}
        for model in models:
            vals = sub[sub["model"] == model][metric].values
            means[model] = np.mean(vals) if len(vals) else np.inf
        best_model = min(means, key=means.get) if means else None
        for model in models:
            vals = sub[sub["model"] == model][metric].values
            if len(vals) == 0:
                row[model] = "-"
            else:
                row[model] = fmt_cell(vals, model == best_model and np.isfinite(means[model]))
        rows.append(row)
    return pd.DataFrame(rows, columns=["train_dataset"] + models)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--models", nargs="*", default=None,
                     help="restrict to these model dir names (default: all found)")
    ap.add_argument("--tag_prefix", default="final_s",
                     help="folder-name prefix identifying the campaign to pool "
                          "(e.g. 'final_s' for the main campaign, 'ablation_r1_s' "
                          "for the NDC_PDMN r=1 ablation)")
    ap.add_argument("--out", default="analysis/out/local_final_table")
    args = ap.parse_args()

    runs = list(find_runs(args.logs_dir, args.models, args.tag_prefix))
    if not runs:
        print(f"No runs found under {args.logs_dir}/*/*/{args.tag_prefix}*/result.npz "
              f"(models filter: {args.models})")
        return

    rows = [load_metrics(r) for r in runs]
    df = pd.DataFrame(rows)


    counts = (df.groupby(["pair", "model"])["seed"]
                .agg(lambda s: ",".join(map(str, sorted(s))))
                .reset_index()
                .rename(columns={"seed": "seeds"}))

    rmse_table = build_table(df, "RMSE")
    score_table = build_table(df, "score")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rmse_table.to_csv(args.out + "_rmse.csv", index=False)
    score_table.to_csv(args.out + "_score.csv", index=False)
    counts.to_csv(args.out + "_seed_counts.csv", index=False)

    print(f"n runs loaded: {len(df)}  ({df['pair'].nunique()} pairs, {df['model'].nunique()} models)")
    print(f"Wrote {args.out}_rmse.csv")
    print(f"Wrote {args.out}_score.csv")
    print(f"Wrote {args.out}_seed_counts.csv  (check this for uneven seed pools before trusting the table)")


if __name__ == "__main__":
    main()
