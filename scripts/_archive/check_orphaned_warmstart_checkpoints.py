

import argparse
import csv
import glob
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    args = ap.parse_args()

    ckpt_paths = glob.glob(os.path.join(args.logs_dir, "*", "*", "warmstart_s*", "best_checkpoint.pth"))
    print(f"{len(ckpt_paths)} warmstart_s* checkpoint(s) found on disk under {args.logs_dir}\n")

    orphaned = []
    for ckpt in ckpt_paths:
        ckpt_dir = os.path.dirname(ckpt)
        tag = os.path.basename(ckpt_dir)
        model = os.path.basename(os.path.dirname(ckpt_dir))
        pair = os.path.basename(os.path.dirname(os.path.dirname(ckpt_dir)))
        csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")

        found = False
        if os.path.exists(csv_path):
            with open(csv_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    sp = (row.get("savepath") or "").replace("\\", "/").rstrip("/")
                    if sp.endswith(f"{pair}/{model}/{tag}"):
                        found = True
                        break
        if not found:
            orphaned.append((pair, model, tag, ckpt_dir))

    print(f"{len(orphaned)} orphaned (checkpoint exists, but no matching CSV row):")
    for pair, model, tag, ckpt_dir in orphaned:
        print(f"  {pair:14s} {model:24s} {tag:16s}  {ckpt_dir}")

    if orphaned:
        print("\nThese trained successfully (or at least far enough to save a checkpoint) but never "
              "logged a CSV row -- likely interrupted between checkpoint-save and CSV-write. "
              "The resume-safe scripts will skip them forever as-is since best_checkpoint.pth exists.")


if __name__ == "__main__":
    main()
