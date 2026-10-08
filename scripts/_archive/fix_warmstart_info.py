

import argparse
import glob
import os
import re
import shutil

import pandas as pd


def derive_correct_info(row):
    savepath = str(row.get("savepath", ""))
    m = re.search(r"[\\/]([^\\/]+)_([^\\/]+)[\\/]([^\\/]+)[\\/]warmstart_s(\d+)\s*$", savepath.replace("\\", "/"))
    if not m:

        m = re.search(r"([A-Za-z0-9_]+)_([A-Za-z0-9]+)/([A-Za-z0-9_]+)/warmstart_s(\d+)", savepath.replace("\\", "/"))
    if not m:
        return None
    src, tgt, model, seed = m.groups()
    return f"{model}_{src}_{tgt}_seed{seed}_warmstart"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    csv_paths = glob.glob(os.path.join(args.logs_dir, "*experimental_logs.csv"))
    total_fixed = 0

    for csv_path in csv_paths:
        try:
            df = pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print(f"  ⚠ skipping empty/corrupt CSV: {csv_path}")
            continue

        is_warmstart = df["savepath"].fillna("").str.contains("warmstart_s")
        if not is_warmstart.any():
            continue

        changed = 0
        for idx in df[is_warmstart].index:
            row = df.loc[idx]
            correct_info = derive_correct_info(row)
            if correct_info is None:
                print(f"  ⚠ could not parse savepath, skipping: {row['savepath']}")
                continue
            if row["info"] != correct_info:
                print(f"  {os.path.basename(csv_path)}: '{row['info']}' -> '{correct_info}'")
                if args.apply:
                    df.at[idx, "info"] = correct_info
                changed += 1

        if changed and args.apply:
            backup_path = csv_path + ".bak"
            if not os.path.exists(backup_path):
                shutil.copy2(csv_path, backup_path)
            df.to_csv(csv_path, index=False)

        total_fixed += changed

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: {total_fixed} row(s) "
          f"{'fixed' if args.apply else 'would be fixed'}")
    if not args.apply and total_fixed:
        print("Re-run with --apply to write the corrected CSVs (backups saved as *.csv.bak).")


if __name__ == "__main__":
    main()
