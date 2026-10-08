

import argparse
import glob
import os
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    csv_paths = glob.glob(os.path.join(args.logs_dir, "*experimental_logs.csv"))
    df = pd.concat([pd.read_csv(f) for f in csv_paths], ignore_index=True)
    df = df[df["info"].fillna("").str.contains("_final")].copy()
    df["seed"] = df["info"].str.extract(r"seed(\d+)")
    df["med_score"] = df.groupby(["train_dataset", "model"])["score"].transform("median")
    df["ratio"] = df["score"] / df["med_score"].clip(lower=1)
    flagged = df[df["ratio"] > args.ratio].sort_values("ratio", ascending=False)

    print(f"Flagged {len(flagged)} anomalous run(s) (score > {args.ratio}x pair/model median):")
    print(flagged[["train_dataset", "model", "seed", "best_last_RMSE", "score", "ratio"]]
          .round(2).to_string(index=False))

    if args.dry_run:
        print("\n--dry_run: nothing deleted.")
        return

    n_deleted_dirs = 0
    rows_to_drop = []
    for idx, row in flagged.iterrows():
        pair, model, seed = row.train_dataset, row.model, row.seed
        d = os.path.join(args.logs_dir, pair, model, f"final_s{seed}")
        if os.path.isdir(d):
            shutil.rmtree(d)
            n_deleted_dirs += 1
        rows_to_drop.append((pair, model, seed))


    for pair in flagged.train_dataset.unique():
        csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
        if not os.path.exists(csv_path):
            continue
        full = pd.read_csv(csv_path)
        full_seed = full["info"].fillna("").str.extract(r"seed(\d+)")[0]
        drop_keys = {(m, s) for p, m, s in rows_to_drop if p == pair}
        mask = full.apply(lambda r: False, axis=1)
        for m, s in drop_keys:
            mask |= (full.model == m) & (full_seed == s) & full["info"].fillna("").str.contains("_final")
        full[~mask].to_csv(csv_path, index=False)

    print(f"\nDeleted {n_deleted_dirs} checkpoint dirs and their matching log rows.")
    print("Rerun scripts/run_final_FD00X.sh -- resume-safe, will only regenerate these.")


if __name__ == "__main__":
    main()
