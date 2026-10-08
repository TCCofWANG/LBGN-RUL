

import argparse
import glob
import os
import re
import shutil
import sys

import pandas as pd
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.hparams import get_configs

STALE_MODELS_BY_SOURCE = {
    "FD001": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL"],
    "FD002": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL", "TACDA_RUL"],
    "FD003": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL", "LBGN_RUL", "NDC_PDMN", "PDMN"],
    "FD004": ["DAGCN_RUL", "TACDA_RUL"],
}


PROMOTE_FROM_DIAG = {
    ("FD001", "LBGN_RUL"): "conservative",
}


def promote_diagnostic_runs(logs_dir, src, model, variant):
    n_dirs_deleted = n_rows_removed = n_copied = n_rows_added = 0
    for tgt in ["FD001", "FD002", "FD003", "FD004"]:
        if tgt == src:
            continue
        pair = f"{src}_{tgt}"
        csv_path = os.path.join(logs_dir, f"{pair}experimental_logs.csv")
        model_dir = os.path.join(logs_dir, pair, model)
        if not os.path.exists(csv_path) or not os.path.isdir(model_dir):
            print(f"  SKIP {pair} {model}: csv or model dir not found")
            continue

        df = pd.read_csv(csv_path)
        info = df["info"].fillna("")


        diag_mask = (df.model == model) & info.str.contains(f"_{variant}_") & info.str.contains("_diag")
        diag_rows = df[diag_mask]
        if diag_rows.empty:
            print(f"  SKIP {pair} {model}: no '{variant}' diagnostic rows found")
            continue
        target_lr = diag_rows["LR"].iloc[0]

        final_mask = (df.model == model) & info.str.contains("_final")
        final_rows = df[final_mask]
        already_promoted = (not final_rows.empty and
                            set(final_rows["info"].str.extract(r"seed(\d+)")[0]) ==
                            set(diag_rows["info"].str.extract(r"seed(\d+)")[0]) and
                            (abs(final_rows["LR"] - target_lr) < 1e-9).all())
        if already_promoted:
            print(f"  OK {pair:14s} {model:24s} -> already promoted, no change")
            continue


        for seed_val in info[final_mask].str.extract(r"seed(\d+)")[0].dropna().unique():
            d = os.path.join(model_dir, f"final_s{seed_val}")
            if os.path.isdir(d):
                shutil.rmtree(d)
                n_dirs_deleted += 1
        n_rows_removed += int(final_mask.sum())
        df = df[~final_mask]

        for _, row in diag_rows.iterrows():
            m = re.search(r"seed(\d+)", row["info"])
            if not m:
                continue
            seed = m.group(1)
            rel = row.savepath.replace("./logs/", "").replace("/", os.sep)
            src_dir = os.path.join(logs_dir, rel)
            dst_dir = os.path.join(model_dir, f"final_s{seed}")
            if os.path.isdir(src_dir) and not os.path.isdir(dst_dir):
                os.makedirs(dst_dir, exist_ok=True)
                for fn in ["best_checkpoint.pth", "hparam.yaml"]:
                    sp = os.path.join(src_dir, fn)
                    if os.path.exists(sp):
                        shutil.copy2(sp, os.path.join(dst_dir, fn))
                n_copied += 1

            new_row = row.copy()
            new_row["savepath"] = f"./logs/{pair}/{model}/final_s{seed}"
            new_row["info"] = f"{model}_{src}_{tgt}_seed{seed}_final"
            df = pd.concat([df, pd.DataFrame([new_row])[df.columns]], ignore_index=True)
            n_rows_added += 1

        df.to_csv(csv_path, index=False)
        print(f"  OK {pair:14s} {model:24s} -> promoted {len(diag_rows)} '{variant}' diagnostic run(s) to final_s*")

    return n_dirs_deleted, n_rows_removed, n_copied, n_rows_added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    args = ap.parse_args()

    n_deleted_dirs = n_copied = n_rows_removed = n_rows_added = 0

    print("=== Strategy B: promoting validated diagnostic runs ===")
    for (src, model), variant in PROMOTE_FROM_DIAG.items():
        d, r, c, a = promote_diagnostic_runs(args.logs_dir, src, model, variant)
        n_deleted_dirs += d; n_rows_removed += r; n_copied += c; n_rows_added += a

    print("\n=== Strategy A: reusing grid-search runs at the corrected LR ===")
    for src, models in STALE_MODELS_BY_SOURCE.items():
        for model in models:
            cfg = get_configs("CMAPSS", src).train_params[model]
            cfg_hd = get_configs("CMAPSS", src).alg_hparams[model].get("hidden_dim")
            cfg_lr = cfg["learning_rate"]

            for tgt in ["FD001", "FD002", "FD003", "FD004"]:
                if tgt == src:
                    continue
                pair = f"{src}_{tgt}"
                csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
                model_dir = os.path.join(args.logs_dir, pair, model)
                if not os.path.exists(csv_path) or not os.path.isdir(model_dir):
                    continue

                df = pd.read_csv(csv_path)
                seed_col = df["info"].fillna("").str.extract(r"seed(\d+)")[0]


                grid_rows = df[(df.model == model) & (~df["info"].fillna("").str.contains("_final"))]
                cand = grid_rows[abs(grid_rows.LR - cfg_lr) < 1e-9]
                if model == "NDC_PDMN":
                    cand = cand[cand.hidden_dim == cfg_hd]
                if len(cand) == 0:
                    print(f"  SKIP {pair} {model}: no grid row at lr={cfg_lr} found (nothing touched)")
                    continue
                row = cand.iloc[0]
                rel = row.savepath.replace("./logs/", "").replace("/", os.sep)
                src_dir = os.path.join(args.logs_dir, rel)
                yaml_path = os.path.join(src_dir, "hparam.yaml")
                if not os.path.isdir(src_dir) or not os.path.exists(yaml_path):
                    print(f"  SKIP {pair} {model}: grid checkpoint/hparam missing at {src_dir} (nothing touched)")
                    continue
                with open(yaml_path) as f:
                    seed = yaml.safe_load(f).get("seed", "1")


                stale_mask = (df.model == model) & df["info"].fillna("").str.contains("_final") & seed_col.isin(["0", "42", "64"])
                for s in ["0", "42", "64"]:
                    d = os.path.join(model_dir, f"final_s{s}")
                    if os.path.isdir(d):
                        shutil.rmtree(d)
                        n_deleted_dirs += 1
                n_rows_removed += int(stale_mask.sum())
                df = df[~stale_mask]

                dst_dir = os.path.join(model_dir, f"final_s{seed}")
                already_final = ((df.model == model) & df["info"].fillna("").str.contains(f"seed{seed}_final")).any()
                if not os.path.isdir(dst_dir):
                    os.makedirs(dst_dir, exist_ok=True)
                    for fn in ["best_checkpoint.pth", "hparam.yaml"]:
                        sp = os.path.join(src_dir, fn)
                        if os.path.exists(sp):
                            shutil.copy2(sp, os.path.join(dst_dir, fn))
                    n_copied += 1

                if not already_final:
                    new_row = row.copy()
                    new_row["savepath"] = f"./logs/{pair}/{model}/final_s{seed}"
                    new_row["info"] = f"{model}_{src}_{tgt}_seed{seed}_final"
                    df = pd.concat([df, pd.DataFrame([new_row])[df.columns]], ignore_index=True)
                    n_rows_added += 1

                df.to_csv(csv_path, index=False)
                print(f"  OK {pair:14s} {model:24s} -> final_s{seed} (reused grid run at lr={cfg_lr})")

    print(f"\nSummary: deleted {n_deleted_dirs} stale dirs, removed {n_rows_removed} stale rows, "
          f"reused {n_copied} grid checkpoints, added {n_rows_added} new log rows.")
    print("Remaining fresh seeds still needed per combo: whatever's missing from {0,42,64,7,123} "
          "(scripts/run_final_FD00X.sh will fill these in — resume-safe).")


if __name__ == "__main__":
    main()
