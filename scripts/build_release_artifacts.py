

import argparse
import glob
import os
import re
import shutil

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    ap.add_argument("--out", default="./release_artifacts")
    ap.add_argument("--model", default="LBGN_RUL")
    ap.add_argument("--seeds", nargs="+", default=None,
                    help="if given, copy these fixed seeds per pair instead of "
                         "auto-selecting the best-performing seed")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)


    csv_out = os.path.join(args.out, "logs_csv")
    os.makedirs(csv_out, exist_ok=True)
    csv_paths = glob.glob(os.path.join(args.logs_dir, "**", "*experimental_logs.csv"), recursive=True)
    for f in csv_paths:
        shutil.copy2(f, os.path.join(csv_out, os.path.basename(f)))
    print(f"copied {len(csv_paths)} log CSVs -> {csv_out}")


    ck_out = os.path.join(args.out, "checkpoints", args.model)
    manifest = []

    if args.seeds is not None:

        pair_dirs = sorted(glob.glob(os.path.join(args.logs_dir, "FD*_FD*")))
        for pair_dir in pair_dirs:
            pair = os.path.basename(pair_dir)
            model_dir = os.path.join(pair_dir, args.model)
            if not os.path.isdir(model_dir):
                continue
            for seed in args.seeds:
                src = os.path.join(model_dir, f"final_s{seed}")
                if not os.path.isdir(src):
                    continue
                dst = os.path.join(ck_out, pair, f"seed{seed}")
                os.makedirs(dst, exist_ok=True)
                for fn in ["best_checkpoint.pth", "hparam.yaml"]:
                    sp = os.path.join(src, fn)
                    if os.path.exists(sp):
                        shutil.copy2(sp, os.path.join(dst, fn))
                manifest.append({"pair": pair, "seed": seed, "rmse": None, "mode": "fixed"})
        print(f"copied {len(manifest)} {args.model} checkpoint(s) (fixed seeds {args.seeds}) -> {ck_out}")

    else:

        if not csv_paths:
            raise SystemExit("no experimental_logs.csv found; can't auto-select best seed")
        df = pd.concat([pd.read_csv(f) for f in csv_paths], ignore_index=True)
        df = df[(df.model == args.model) & (df["info"].fillna("").str.contains("_final"))].copy()
        df["seed"] = df["info"].str.extract(r"seed(\d+)")

        for pair, sub in df.groupby("train_dataset"):
            if sub.empty:
                continue
            best = sub.loc[sub["best_last_RMSE"].idxmin()]
            seed = best["seed"]
            rmse = float(best["best_last_RMSE"])
            src = os.path.join(args.logs_dir, pair, args.model, f"final_s{seed}")
            if not os.path.isdir(src):
                print(f"  WARNING: best seed for {pair} (seed {seed}, rmse {rmse:.2f}) "
                      f"not found on disk at {src} -- skipping")
                continue
            dst = os.path.join(ck_out, pair, f"best_seed{seed}")
            os.makedirs(dst, exist_ok=True)
            for fn in ["best_checkpoint.pth", "hparam.yaml"]:
                sp = os.path.join(src, fn)
                if os.path.exists(sp):
                    shutil.copy2(sp, os.path.join(dst, fn))
            manifest.append({"pair": pair, "seed": seed, "rmse": round(rmse, 4), "mode": "best"})
            print(f"  {pair:14s} best seed={seed:<4} rmse={rmse:.2f}")

        print(f"copied {len(manifest)} {args.model} checkpoint(s) "
              f"(auto-selected best seed per pair) -> {ck_out}")

    manifest_path = os.path.join(args.out, f"{args.model}_checkpoint_manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)
    print(f"wrote manifest -> {manifest_path}")

    total_size = sum(os.path.getsize(os.path.join(dp, f))
                      for dp, _, fs in os.walk(args.out) for f in fs)
    print(f"total release_artifacts size: {total_size/1e6:.2f} MB")


if __name__ == "__main__":
    main()
