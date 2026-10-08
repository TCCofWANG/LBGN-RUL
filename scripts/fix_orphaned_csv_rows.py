

import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys

import numpy as np
import yaml

COLUMNS = [
    "train_dataset", "test_dataset", "model", "trainable_parameters", "time",
    "inference_time", "LR", "batch_size", "loss_type", "best_last_RMSE", "score",
    "windowsize", "hidden_dim", "train", "savepath", "resume", "resumepath", "info",
]


def score_compute(pred, gt, dataset_name):

    d = pred - gt
    score_list = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
    if dataset_name == "CMAPSS":
        return float(np.sum(score_list))
    return float(np.mean(score_list))


def rmse_compute(pred, gt):
    return float(np.sqrt(np.mean((pred - gt) ** 2)))


def find_orphaned(logs_dir, tag_glob, source=None):

    pair_glob = f"{source}_*" if source else "*"
    pattern = os.path.join(logs_dir, pair_glob, "*", tag_glob, "best_checkpoint.pth")
    orphaned, total = [], 0
    for ckpt in sorted(glob.glob(pattern)):
        total += 1
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
    return orphaned, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    ap.add_argument("--source", default=None,
                    help="only repair pairs from this source, e.g. FD001 (default: all sources). "
                         "Use this to repair one source at a time, matching the per-source campaign runs.")
    ap.add_argument("--tag_glob", default="final_s*",
                    help="checkpoint folder pattern, e.g. 'final_s*', 'warmstart_s*', 'abl_*'")
    ap.add_argument("--info_suffix", default="final",
                    help="suffix used in the rebuilt info tag: {model}_{src}_{tgt}_seed{N}_{suffix}")
    ap.add_argument("--mode", choices=["backfill", "delete"], default="backfill",
                    help="backfill (default): re-evaluate and append a CSV row, keep the checkpoint. "
                         "delete: remove the orphaned checkpoint directory outright, no re-evaluation.")
    ap.add_argument ( '--dataset_name' , default = 'CMAPSS' , choices = ['CMAPSS', 'N_CMAPSS'] , type = str ,
                     help = 'which --Data_id_* flags to pass to the eval subprocess (see main()) -- '
                            'must match whatever --logs_dir actually holds' )
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    orphaned, total = find_orphaned(args.logs_dir, args.tag_glob, args.source)

    scope = f"source {args.source}" if args.source else "all sources"
    print(f"Scope: {scope}  |  mode: {args.mode}")
    print(f"{total} checkpoint(s) matching '{args.tag_glob}' under {args.logs_dir}")
    print(f"{len(orphaned)} orphaned (checkpoint exists, no CSV row)\n")
    if not orphaned:
        print("Nothing to do.")
        return


    by_model = {}
    for pair, model, tag, _ in orphaned:
        by_model.setdefault(model, []).append(f"{pair}/{tag}")
    print("Breakdown by model:")
    for model in sorted(by_model):
        print(f"  {model:24s} {len(by_model[model]):3d}")
    print()

    if args.mode == "delete":
        deleted = 0
        for pair, model, tag, ckpt_dir in orphaned:
            label = "DELETE" if args.apply else "would delete"
            print(f"  {label:12s} {pair:14s} {model:24s} {tag:20s}  {ckpt_dir}")
            if args.apply:
                shutil.rmtree(ckpt_dir)
                deleted += 1
        print()
        if args.apply:
            plural = "y" if deleted == 1 else "ies"
            print(f"Deleted {deleted} orphaned checkpoint director{plural}.")
        else:
            print(f"{len(orphaned)} checkpoint(s) would be deleted. Re-run with --apply to actually delete them.")
        return

    backfilled = failed = 0
    for pair, model, tag, ckpt_dir in orphaned:
        src, tgt = pair.split("_", 1)
        m = re.search(r"_s(\d+)$", tag) or re.search(r"(\d+)$", tag)
        seed = m.group(1) if m else "0"

        ckpt_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
        hparam_path = os.path.join(ckpt_dir, "hparam.yaml")
        npz_path = os.path.join(ckpt_dir, "result.npz")

        print(f"> {pair} | {model} | {tag} (seed {seed})")
        if not args.apply:
            continue


        if args.dataset_name == "N_CMAPSS":
            id_flags = ["--Data_id_N_CMAPSS", src, "--Data_id_N_CMAPSS_test", tgt]
        else:
            id_flags = ["--Data_id_CMAPSS", src, "--Data_id_CMAPSS_test", tgt]

        cmd = [
            py, "CMAPSS_DA.py",
            "--model_name", model,
            "--dataset_name", args.dataset_name,
            *id_flags,
            "--seed", seed,
            "--train", "False",
            "--resume", "True",
            "--warm_start_checkpoint", ckpt_path,


            "--logs_root", args.logs_dir,
            "--save_path", tag,
        ]
        if os.path.exists(hparam_path):
            cmd += ["--warm_start_hparam_yaml", hparam_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  x eval failed (rc={result.returncode})")
            tail = (result.stderr or "").strip().splitlines()[-3:]
            for line in tail:
                print(f"     {line}")
            failed += 1
            continue

        if os.path.exists(npz_path):
            data = np.load(npz_path)
            preds, trues = data["test_preds"].reshape(-1), data["test_trues"].reshape(-1)
            rmse, score = rmse_compute(preds, trues), score_compute(preds, trues, args.dataset_name)
        else:


            rmse_m = re.search(r"of enc overall is:\s*([\-\d.eE]+)", result.stdout)
            score_m = re.search(r"ofenc\s*([\-\d.eE]+)", result.stdout)
            if not (rmse_m and score_m):
                print("  x no result.npz and no parseable RMSE/score in stdout")
                failed += 1
                continue
            rmse, score = float(rmse_m.group(1)), float(score_m.group(1))

        hp = {}
        if os.path.exists(hparam_path):
            with open(hparam_path, "r", encoding="utf-8") as f:
                hp = yaml.load(f.read(), Loader=yaml.FullLoader) or {}

        row = {
            "train_dataset": src, "test_dataset": tgt, "model": model,
            "trainable_parameters": "", "time": "", "inference_time": "",
            "LR": hp.get("learning_rate", ""), "batch_size": hp.get("batch_size", ""),
            "loss_type": hp.get("loss_type", ""),
            "best_last_RMSE": rmse, "score": score,
            "windowsize": hp.get("input_length", ""), "hidden_dim": hp.get("hidden_dim", ""),
            "train": True, "savepath": ckpt_dir.replace("\\", "/"),
            "resume": False, "resumepath": "",
            "info": f"{model}_{src}_{tgt}_seed{seed}_{args.info_suffix}",
        }

        csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            if write_header:
                w.writeheader()
            w.writerow(row)

        print(f"  ok rmse={rmse:.2f} score={score:.2f} -> {os.path.basename(csv_path)}")
        backfilled += 1

    print(f"\n{'APPLIED' if args.apply else 'DRY RUN'}: "
          f"{backfilled} backfilled, {failed} failed, {len(orphaned)} orphaned total")
    if not args.apply:
        print("Re-run with --apply to re-evaluate these checkpoints and append their CSV rows.")


if __name__ == "__main__":
    main()
