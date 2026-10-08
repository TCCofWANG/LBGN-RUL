

import argparse
import os
import shutil
import time as time_mod

import pandas as pd


BAD_ROWS = [
    ("DS01_DS02", "./logs_NCMAPSS/DS01_DS02/NDC_PDMN/final_s1"),
    ("DS01_DS04", "./logs_NCMAPSS/DS01_DS04/DAGCN_RUL/final_s1234"),
    ("DS02_DS01", "./logs_NCMAPSS/DS02_DS01/DAST_RUL/final_s64"),
    ("DS03_DS02", "./logs_NCMAPSS/DS03_DS02/DAGCN_RUL/final_s2026"),
    ("DS03_DS02", "./logs_NCMAPSS/DS03_DS02/DAST_RUL/final_s0"),
    ("DS03_DS02", "./logs_NCMAPSS/DS03_DS02/DAST_RUL/final_s1"),
    ("DS03_DS02", "./logs_NCMAPSS/DS03_DS02/DAST_RUL/final_s41"),
    ("DS03_DS02", "./logs_NCMAPSS/DS03_DS02/DAST_RUL/final_s7"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/DAGCN_RUL/final_s41"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/DAST_RUL/final_s0"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/DAST_RUL/final_s1234"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/DAST_RUL/final_s41"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/DAST_RUL/final_s64"),
    ("DS03_DS04", "./logs_NCMAPSS/DS03_DS04/LBGN_RUL/final_s41"),
    ("DS04_DS01", "./logs_NCMAPSS/DS04_DS01/LBGN_RUL/final_s0"),
    ("DS04_DS02", "./logs_NCMAPSS/DS04_DS02/LBGN_RUL/final_s1"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs_NCMAPSS")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    by_pair = {}
    for pair, savepath in BAD_ROWS:
        by_pair.setdefault(pair, []).append(savepath)

    total_removed = 0
    for pair, savepaths in sorted(by_pair.items()):
        csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
        if not os.path.exists(csv_path):
            print(f"⚠ {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        target = set(savepaths)
        is_bad = df["savepath"].isin(target)
        n_found = int(is_bad.sum())
        n_expected = len(savepaths)

        print(f"\n{pair}: expected {n_expected} bad row(s), found {n_found}")
        for _, row in df[is_bad].iterrows():
            print(f"    {row['model']:12s} {row['savepath']:48s} score={row['score']}")
        if n_found != n_expected:
            missing = target - set(df.loc[is_bad, "savepath"])
            for sp in missing:
                print(f"    ⚠ expected but not found in this CSV: {sp}")

        if n_found == 0:
            continue

        if args.apply:
            backup_path = f"{csv_path}.{time_mod.strftime('%Y%m%d-%H%M%S')}.bak"
            shutil.copy2(csv_path, backup_path)
            df[~is_bad].to_csv(csv_path, index=False)
            print(f"  -> removed {n_found} row(s), backup at {backup_path}")

        total_removed += n_found

    print(f"\n{'='*70}")
    print(f"{'APPLIED' if args.apply else 'DRY RUN'}: {total_removed}/{len(BAD_ROWS)} bad rows "
          f"{'removed' if args.apply else 'would be removed'}")
    if not args.apply:
        print("Re-run with --apply to actually remove them.")
    else:
        print("\nNext: re-run fix_orphaned_csv_rows.py --dataset_name N_CMAPSS "
              "--logs_dir ./logs_NCMAPSS --apply to correctly re-backfill these.")


if __name__ == "__main__":
    main()
