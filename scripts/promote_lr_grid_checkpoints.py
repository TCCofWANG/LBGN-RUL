

import argparse
import csv
import os
import re
import shutil

LR_LOGS_ROOT = "./lr_logs"
LOGS_ROOT = "./logs"


WINNERS = {
    ("FD001", "PDMN"):                   {"LR": "0.005",  "hd": None},
    ("FD001", "NDC_PDMN"):                {"LR": "0.0005", "hd": "32"},
    ("FD002", "PDMN"):                   {"LR": "0.005",  "hd": None},
    ("FD002", "NDC_PDMN"):                {"LR": "0.0005", "hd": "32"},
    ("FD002", "LBGN_RUL"):   {"LR": "0.005",  "hd": None},


}

SEEDS_AVAILABLE = ("123", "64")


def find_pair_csvs():
    for fname in sorted(os.listdir(LR_LOGS_ROOT)):
        if fname.endswith("experimental_logs.csv"):
            yield os.path.join(LR_LOGS_ROOT, fname)


def row_key(row):

    info = row.get("info", "") or ""
    if "_final" not in info:
        return None
    m = re.search(r"seed(\d+)", info)
    if not m:
        return None
    return (row["train_dataset"], row["model"], m.group(1))


def drop_matching_row(dest_csv_path, target_key):

    if not os.path.exists(dest_csv_path):
        return False
    with open(dest_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)
    kept = [r for r in rows if row_key(r) != target_key]
    dropped = len(kept) != len(rows)
    if dropped:
        with open(dest_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(kept)
    return dropped


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="preview only, no changes")
    g.add_argument("--apply", action="store_true", help="actually copy checkpoints + append CSV rows")
    args = ap.parse_args()
    apply = args.apply

    promoted = overwritten = skipped_missing_src = skipped_not_winner = 0

    for csv_path in find_pair_csvs():
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        for row in rows:
            src = row["train_dataset"].split("_")[0]
            model = row["model"]
            key = (src, model)
            if key not in WINNERS:
                skipped_not_winner += 1
                continue

            win = WINNERS[key]
            if row["LR"] != win["LR"]:
                continue
            if win["hd"] is not None and row.get("hidden_dim") != win["hd"]:
                continue

            m = re.search(r"seed(\d+)", row.get("info", ""))
            if not m or m.group(1) not in SEEDS_AVAILABLE:
                continue
            seed = m.group(1)

            pair = row["train_dataset"]
            tgt = row["test_dataset"]
            src_dir = row["savepath"]
            src_ckpt = os.path.join(src_dir, "best_checkpoint.pth")

            dest_dir = os.path.join(LOGS_ROOT, pair, model, f"final_s{seed}")
            dest_ckpt = os.path.join(dest_dir, "best_checkpoint.pth")
            dest_csv_path = os.path.join(LOGS_ROOT, f"{pair}experimental_logs.csv")

            if not os.path.exists(src_ckpt):
                print(f"⚠ SKIP missing-source  {pair}/{model} seed{seed}  ({src_ckpt} not found)")
                skipped_missing_src += 1
                continue

            will_overwrite = os.path.exists(dest_ckpt)
            action = "OVERWRITE" if will_overwrite else "PROMOTE  "
            print(f"✓ {action}  {pair}/{model} seed{seed}  LR={row['LR']} hd={row.get('hidden_dim')}  "
                  f"RMSE={row['best_last_RMSE']} score={row['score']}  ({src_dir} -> {dest_dir})")
            promoted += 1
            if will_overwrite:
                overwritten += 1

            if apply:
                if os.path.exists(dest_dir):
                    shutil.rmtree(dest_dir)
                os.makedirs(dest_dir, exist_ok=True)
                for fname in os.listdir(src_dir):
                    shutil.copy2(os.path.join(src_dir, fname), os.path.join(dest_dir, fname))

                new_row = dict(row)
                new_row["savepath"] = dest_dir
                new_row["info"] = f"{model}_{src}_{tgt}_seed{seed}_final"
                new_row["train"] = "True"
                new_row["resume"] = "False"
                new_row["resumepath"] = ""

                drop_matching_row(dest_csv_path, (pair, model, seed))

                write_header = not os.path.exists(dest_csv_path)
                with open(dest_csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if write_header:
                        writer.writeheader()
                    writer.writerow(new_row)

    print("\n────────────────────────────────────────")
    print(f"{'APPLIED' if apply else 'DRY RUN'}")
    print(f"  promoted (total)     : {promoted}")
    print(f"  of which overwritten : {overwritten}")
    print(f"  skip (missing ckpt)  : {skipped_missing_src}")
    if not apply:
        print("\nRe-run with --apply to actually copy checkpoints and rewrite CSV rows.")


if __name__ == "__main__":
    main()
