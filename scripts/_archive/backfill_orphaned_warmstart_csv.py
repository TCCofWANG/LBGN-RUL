

import argparse
import csv
import glob
import os
import re
import subprocess
import sys

import numpy as np
import yaml

COLUMNS = [
    "train_dataset", "test_dataset", "model", "trainable_parameters", "time",
    "inference_time", "LR", "batch_size", "loss_type", "best_last_RMSE", "score",
    "windowsize", "hidden_dim", "train", "savepath", "resume", "resumepath", "info",
]


def score_compute(pred, gt):

    d = pred - gt
    score_list = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    return float(np.sum(score_list))


def rmse_compute(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def find_orphaned(logs_dir):
    ckpt_paths = glob.glob(os.path.join(logs_dir, "*", "*", "warmstart_s*", "best_checkpoint.pth"))
    orphaned = []
    for ckpt in ckpt_paths:
        ckpt_dir = os.path.dirname(ckpt)
        tag = os.path.basename(ckpt_dir)
        model = os.path.basename(os.path.dirname(ckpt_dir))
        pair = os.path.basename(os.path.dirname(os.path.dirname(ckpt_dir)))
        csv_path = os.path.join(logs_dir, f"{pair}experimental_logs.csv")

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
    return orphaned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    orphaned = find_orphaned(args.logs_dir)
    print(f"{len(orphaned)} orphaned checkpoint(s) found\n")

    backfilled = failed = 0

    for pair, model, tag, ckpt_dir in orphaned:
        src, tgt = pair.split("_", 1)
        m = re.match(r"warmstart_s(\d+)", tag)
        seed = m.group(1) if m else "0"

        ckpt_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
        hparam_path = os.path.join(ckpt_dir, "hparam.yaml")
        npz_path = os.path.join(ckpt_dir, "result.npz")

        print(f"▶ {pair} | {model} | {tag}")

        if not args.apply:
            print("  (dry run -- would re-eval and append a CSV row)")
            continue

        cmd = [
            py, "CMAPSS_DA.py",
            "--model_name", model,
            "--dataset_name", "CMAPSS",
            "--Data_id_CMAPSS", src,
            "--Data_id_CMAPSS_test", tgt,
            "--seed", seed,
            "--train", "False",
            "--resume", "True",
            "--warm_start_checkpoint", ckpt_path,
            "--warm_start_hparam_yaml", hparam_path,
            "--save_path", tag,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            print(f"  ✗ eval subprocess failed for {pair}/{model}/{tag}")
            failed += 1
            continue

        if os.path.exists(npz_path):

            data = np.load(npz_path)
            preds, trues = data["test_preds"].reshape(-1), data["test_trues"].reshape(-1)
            rmse = rmse_compute(preds, trues)
            score = score_compute(preds, trues)
        else:


            rmse_m = re.search(r"of enc overall is:\s*([\-\d.eE]+)", result.stdout)
            score_m = re.search(r"ofenc\s*([\-\d.eE]+)", result.stdout)
            if not (rmse_m and score_m):
                print(f"  ✗ neither result.npz nor a parseable RMSE/score line found for {pair}/{model}/{tag}")
                failed += 1
                continue
            rmse = float(rmse_m.group(1))
            score = float(score_m.group(1))
            print(f"  (result.npz missing -- parsed rmse={rmse} score={score} from stdout instead)")

        hp = {}
        if os.path.exists(hparam_path):
            with open(hparam_path, "r", encoding="utf-8") as f:
                hp = yaml.load(f.read(), Loader=yaml.FullLoader) or {}

        row = {
            "train_dataset": pair,
            "test_dataset": tgt,
            "model": model,
            "trainable_parameters": "",
            "time": "",
            "inference_time": "",
            "LR": hp.get("learning_rate", ""),
            "batch_size": hp.get("batch_size", ""),
            "loss_type": hp.get("loss_type", ""),
            "best_last_RMSE": rmse,
            "score": score,
            "windowsize": hp.get("input_length", ""),
            "hidden_dim": hp.get("hidden_dim", ""),
            "train": True,
            "savepath": ckpt_dir.replace("\\", "/"),
            "resume": False,
            "resumepath": "",
            "info": f"{model}_{src}_{tgt}_seed{seed}_warmstart",
        }

        csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"  ✓ rmse={rmse:.2f} score={score:.2f} -> appended to {csv_path}")
        backfilled += 1

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: backfilled {backfilled}, failed {failed}, "
          f"total orphaned {len(orphaned)}")
    if not args.apply:
        print("Re-run with --apply to actually re-evaluate checkpoints and append CSV rows.")


if __name__ == "__main__":
    main()
