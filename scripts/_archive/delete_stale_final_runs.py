

import argparse
import csv
import os
import re
import shutil

LOGS_ROOT = "./logs"


STALE_ROWS = [
    ("FD001_FD002", "LBGN_RUL", "0",   "0.005", None),
    ("FD001_FD002", "LBGN_RUL", "7",   "0.005", None),
    ("FD001_FD002", "LBGN_RUL", "42",  "0.005", None),
    ("FD001_FD002", "LBGN_RUL", "64",  "0.005", None),
    ("FD001_FD002", "LBGN_RUL", "123", "0.005", None),
    ("FD001_FD002", "PDMN",                  "0",   "0.0005", None),
    ("FD001_FD002", "PDMN",                  "7",   "0.0005", None),
    ("FD001_FD002", "PDMN",                  "42",  "0.0005", None),
    ("FD001_FD003", "LBGN_RUL", "0",   "0.005", None),
    ("FD001_FD003", "LBGN_RUL", "7",   "0.005", None),
    ("FD001_FD003", "LBGN_RUL", "42",  "0.005", None),
    ("FD001_FD003", "LBGN_RUL", "64",  "0.005", None),
    ("FD001_FD003", "LBGN_RUL", "123", "0.005", None),
    ("FD001_FD003", "PDMN",                  "0",   "0.0005", None),
    ("FD001_FD003", "PDMN",                  "7",   "0.0005", None),
    ("FD001_FD003", "PDMN",                  "42",  "0.0005", None),
    ("FD001_FD004", "PDMN",                  "0",   "0.0005", None),
    ("FD001_FD004", "PDMN",                  "7",   "0.0005", None),
    ("FD001_FD004", "PDMN",                  "42",  "0.0005", None),
    ("FD002_FD001", "LBGN_RUL", "0",   "0.0005", None),
    ("FD002_FD001", "LBGN_RUL", "7",   "0.0005", None),
    ("FD002_FD001", "LBGN_RUL", "42",  "0.0005", None),
    ("FD002_FD001", "NDC_PDMN",              "0",   "0.0005", "64"),
    ("FD002_FD001", "PDMN",                  "0",   "0.0005", None),
    ("FD002_FD001", "PDMN",                  "7",   "0.0005", None),
    ("FD002_FD001", "PDMN",                  "42",  "0.0005", None),
    ("FD002_FD003", "LBGN_RUL", "0",   "0.0005", None),
    ("FD002_FD003", "LBGN_RUL", "7",   "0.0005", None),
    ("FD002_FD003", "LBGN_RUL", "42",  "0.0005", None),
    ("FD002_FD003", "PDMN",                  "0",   "0.0005", None),
    ("FD002_FD003", "PDMN",                  "7",   "0.0005", None),
    ("FD002_FD003", "PDMN",                  "42",  "0.0005", None),
    ("FD002_FD004", "LBGN_RUL", "0",   "0.0005", None),
    ("FD002_FD004", "LBGN_RUL", "7",   "0.0005", None),
    ("FD002_FD004", "LBGN_RUL", "42",  "0.0005", None),
    ("FD002_FD004", "PDMN",                  "0",   "0.0005", None),
    ("FD002_FD004", "PDMN",                  "7",   "0.0005", None),
    ("FD002_FD004", "PDMN",                  "42",  "0.0005", None),
]

assert len(STALE_ROWS) == 38, f"expected 38 rows, got {len(STALE_ROWS)}"


def norm_lr(s):
    try:
        return repr(float(s))
    except (TypeError, ValueError):
        return s


def load_csv_rows(csv_path):
    if not os.path.exists(csv_path):
        return None, []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv_rows(csv_path, fieldnames, rows):
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    apply = args.apply

    deleted = skipped_mismatch = skipped_missing = 0

    for pair, model, seed, stale_lr, stale_hd in STALE_ROWS:
        csv_path = os.path.join(LOGS_ROOT, f"{pair}experimental_logs.csv")
        ckpt_dir = os.path.join(LOGS_ROOT, pair, model, f"final_s{seed}")

        fieldnames, rows = load_csv_rows(csv_path)
        if fieldnames is None:
            print(f"⚠ SKIP missing-csv     {pair}/{model} seed{seed}  ({csv_path} not found)")
            skipped_missing += 1
            continue

        match_idx = None
        for i, row in enumerate(rows):
            if row["model"] != model:
                continue
            m = re.search(r"seed(\d+)", row.get("info", "") or "")
            if not m or m.group(1) != seed:
                continue
            match_idx = i
            break

        if match_idx is None:
            print(f"⚠ SKIP no-csv-row      {pair}/{model} seed{seed}  (already removed?)")
            skipped_missing += 1
            continue

        row = rows[match_idx]
        actual_lr_ok = norm_lr(row["LR"]) == norm_lr(stale_lr)
        actual_hd_ok = stale_hd is None or row.get("hidden_dim") == stale_hd
        if not (actual_lr_ok and actual_hd_ok):
            print(f"⚠ SKIP already-fixed   {pair}/{model} seed{seed}  "
                  f"(row now has LR={row['LR']} hd={row.get('hidden_dim')}, no longer matches the stale snapshot -- "
                  f"looks like it was already retrained)")
            skipped_mismatch += 1
            continue

        print(f"✓ DELETE  {pair}/{model} seed{seed}  LR={row['LR']} hd={row.get('hidden_dim')}  "
              f"score={row['score']}  ({ckpt_dir})")
        deleted += 1

        if apply:
            if os.path.isdir(ckpt_dir):
                shutil.rmtree(ckpt_dir)
            remaining = rows[:match_idx] + rows[match_idx + 1:]
            write_csv_rows(csv_path, fieldnames, remaining)

    print("\n────────────────────────────────────────")
    print(f"{'APPLIED' if apply else 'DRY RUN'}")
    print(f"  deleted              : {deleted}")
    print(f"  skip (already fixed) : {skipped_mismatch}")
    print(f"  skip (missing)       : {skipped_missing}")
    if not apply:
        print("\nRe-run with --apply to actually delete checkpoints and rewrite CSV rows.")
    else:
        print("\nNext: re-run the affected run_final_FD001.sh / run_final_FD002.sh -- "
              "the resume-safe skip logic will now retrain exactly these 38 seeds under the current hparams.py.")


if __name__ == "__main__":
    main()
