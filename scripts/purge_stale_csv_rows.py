

import argparse
import glob
import os
import shutil
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs", type=str,
                     help="root containing {pair}experimental_logs.csv files and the {pair}/{model}/{tag}/ tree")
    ap.add_argument("--models", nargs="+", default=None,
                     help="restrict to these model names (default: check every model found)")
    ap.add_argument("--apply", action="store_true",
                     help="actually rewrite the CSVs (default is dry-run/report only)")
    args = ap.parse_args()

    csv_paths = sorted(glob.glob(os.path.join(args.logs_dir, "*experimental_logs.csv")))
    if not csv_paths:
        raise SystemExit(f"no experimental_logs.csv found under {args.logs_dir}")

    total_rows = total_stale = total_files_changed = 0

    for csv_path in csv_paths:
        basename = os.path.basename(csv_path)
        pair = basename[:-len("experimental_logs.csv")] if basename.endswith("experimental_logs.csv") else basename

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"⚠ skipping unreadable CSV: {csv_path} ({e})")
            continue
        if df.empty:
            continue

        total_rows += len(df)

        def row_is_stale(row):
            savepath = str(row.get("savepath", "") or "")
            tag = os.path.basename(savepath.rstrip("/\\"))
            model = str(row.get("model", "") or "")
            if not tag or not model:
                return False
            ckpt_dir = os.path.join(args.logs_dir, pair, model, tag)
            return not os.path.isdir(ckpt_dir)

        is_stale = df.apply(row_is_stale, axis=1)
        if args.models:
            is_stale = is_stale & df["model"].isin(args.models)

        n_stale = int(is_stale.sum())
        if n_stale == 0:
            continue

        total_stale += n_stale
        print(f"\n{pair}: {n_stale} stale row(s) out of {len(df)}")
        for _, row in df[is_stale].iterrows():
            tag = os.path.basename(str(row.get("savepath", "") or "").rstrip("/\\"))
            print(f"    {row.get('model'):24s} {tag:20s} "
                  f"RMSE={row.get('best_last_RMSE')}  score={row.get('score')}  "
                  f"info={row.get('info')}")

        if args.apply:
            backup_path = f"{csv_path}.{time.strftime('%Y%m%d-%H%M%S')}.bak"
            shutil.copy2(csv_path, backup_path)
            df[~is_stale].to_csv(csv_path, index=False)
            total_files_changed += 1
            print(f"  -> removed, backup at {backup_path}")

    print(f"\n{'='*70}")
    print(f"{'APPLIED' if args.apply else 'DRY RUN'}")
    print(f"  total rows scanned : {total_rows}")
    print(f"  stale rows found   : {total_stale}")
    if args.apply:
        print(f"  CSV files rewritten: {total_files_changed}")
    else:
        print("\n  Re-run with --apply to actually remove these rows "
              "(a .bak copy of each changed CSV is written first).")


if __name__ == "__main__":
    main()
