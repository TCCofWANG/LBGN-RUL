

import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

try:
    from scipy.stats import wilcoxon
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

MULTI_CONDITION_SOURCES = {"FD002", "FD004"}
PAIR_RE = re.compile(r"^(FD00\d)_(FD00\d)$")


def score_compute(pred, gt):
    pred, gt = np.asarray(pred).reshape(-1), np.asarray(gt).reshape(-1)
    diff = pred - gt
    score_list = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp(diff / 10) - 1)
    return float(np.sum(score_list))


def rmse_compute(pred, gt):
    pred, gt = np.asarray(pred).reshape(-1), np.asarray(gt).reshape(-1)
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def collect(logs_dir, tag_re):
    rows = []
    for pair_dir in sorted(glob.glob(os.path.join(logs_dir, "FD00?_FD00?"))):
        pair = os.path.basename(pair_dir)
        if not PAIR_RE.match(pair):
            continue
        model_dir = os.path.join(pair_dir, "NDC_PDMN")
        if not os.path.isdir(model_dir):
            continue
        for run_dir in sorted(glob.glob(os.path.join(model_dir, "*"))):
            tag = os.path.basename(run_dir)
            m = tag_re.match(tag)
            if not m:
                continue
            npz_path = os.path.join(run_dir, "result.npz")
            if not os.path.isfile(npz_path):
                continue
            d = np.load(npz_path, allow_pickle=True)
            pred, gt = d["test_preds"], d["test_trues"]
            rows.append({
                "pair": pair,
                "seed": int(m.group(1)),
                "RMSE": rmse_compute(pred, gt),
                "score": score_compute(pred, gt),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    args = ap.parse_args()

    full = collect(args.logs_dir, re.compile(r"^final_s(\d+)$"))
    abl = collect(args.logs_dir, re.compile(r"^ablation_r1_s(\d+)$"))

    print(f"full model rows   : {len(full)}  ({full['pair'].nunique() if len(full) else 0} pairs, "
          f"seeds {sorted(full['seed'].unique()) if len(full) else []})")
    print(f"r=1 ablation rows : {len(abl)}  ({abl['pair'].nunique() if len(abl) else 0} pairs, "
          f"seeds {sorted(abl['seed'].unique()) if len(abl) else []})\n")

    if len(abl) == 0:
        print("No ablation_r1_s* runs found yet -- run scripts/run_ablation_ndc_r1.sh first.")
        return

    for metric in ["RMSE", "score"]:
        print(f"=== {metric} ===")
        full_p = full.set_index(["pair", "seed"])[metric]
        abl_p = abl.set_index(["pair", "seed"])[metric]
        joined = pd.concat([full_p.rename("full"), abl_p.rename("ablated_r1")], axis=1).dropna()

        joined["source"] = [p.split("_")[0] for p in joined.index.get_level_values(0)]
        joined["delta_pct"] = (joined["ablated_r1"] - joined["full"]) / joined["full"] * 100

        print(f"n paired = {len(joined)}")
        print(f"  full model mean       : {joined['full'].mean():.2f}")
        print(f"  r=1 ablation mean     : {joined['ablated_r1'].mean():.2f}")
        print(f"  mean %% worse w/o clock: {joined['delta_pct'].mean():+.1f}%")

        if HAVE_SCIPY and len(joined) >= 2:
            stat, p = wilcoxon(joined["full"], joined["ablated_r1"])
            sig = "SIGNIFICANT" if p < 0.05 else "not significant"
            direction = "better" if joined["full"].mean() < joined["ablated_r1"].mean() else "worse"
            print(f"  Wilcoxon: full model {direction} than r=1 ablation, p={p:.4f}  [{sig}]")

        multi = joined[joined["source"].isin(MULTI_CONDITION_SOURCES)]
        single = joined[~joined["source"].isin(MULTI_CONDITION_SOURCES)]
        print("\n  Falsifiable prediction check (gains should concentrate on FD002/FD004 sources):")
        print(f"    FD002/FD004 (multi-condition)  mean %% worse w/o clock: {multi['delta_pct'].mean():+.1f}%  (n={len(multi)})")
        print(f"    FD001/FD003 (near-single-cond) mean %% worse w/o clock: {single['delta_pct'].mean():+.1f}%  (n={len(single)})")
        if HAVE_SCIPY and len(multi) >= 2 and len(single) >= 2:
            stat, p = wilcoxon(multi["full"], multi["ablated_r1"])
            print(f"    Wilcoxon within FD002/FD004 only: p={p:.4f}")
            stat, p = wilcoxon(single["full"], single["ablated_r1"])
            print(f"    Wilcoxon within FD001/FD003 only: p={p:.4f}")
        print()


if __name__ == "__main__":
    main()
