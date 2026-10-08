

import argparse
import os
import sys
import time

import pandas as pd
import torch
import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

COLUMNS = [
    "train_dataset", "test_dataset", "model", "trainable_parameters", "time",
    "inference_time", "LR", "batch_size", "loss_type", "best_last_RMSE", "score",
    "windowsize", "hidden_dim", "train", "savepath", "resume", "resumepath", "info",
]


def count_params(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    return int(sum(t.numel() for t in state_dict.values() if hasattr(t, "numel")))


def load_hparams(hparam_path):
    if not os.path.exists(hparam_path):
        return {}
    with open(hparam_path, "r", encoding="utf-8") as f:
        return yaml.load(f.read(), Loader=yaml.FullLoader) or {}


def build_info(model, pair, seed, kind):
    src, tgt = pair.split("_", 1)
    if kind == "ablation":
        return f"{model}_ablr1_{src}_{tgt}_seed{seed}_ablation"
    return f"{model}_{src}_{tgt}_seed{seed}_final"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_best_dir", default="../logs_best")
    args = ap.parse_args()

    manifest_path = os.path.join(args.logs_best_dir, "best_checkpoint_manifest.csv")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"manifest not found: {manifest_path} "
                          f"(run scripts/copy_best_checkpoints.py first)")
    try:
        manifest = pd.read_csv(manifest_path)
    except pd.errors.EmptyDataError:
        raise SystemExit(
            f"{manifest_path} is empty -- copy_best_checkpoints.py likely failed or was "
            f"interrupted before writing it. Regenerate it first:\n"
            f"  python scripts/copy_best_checkpoints.py --logs_dir ./logs --out {args.logs_best_dir} --force"
        )
    if "kind" not in manifest.columns:
        raise SystemExit(
            f"{manifest_path} is missing the 'kind' column -- it was written by an older "
            f"version of copy_best_checkpoints.py. Regenerate it first:\n"
            f"  python scripts/copy_best_checkpoints.py --logs_dir ./logs --out {args.logs_best_dir} --force"
        )

    written = 0
    for pair, group in manifest.groupby("pair"):
        _, tgt = pair.split("_", 1)
        rows = []

        for _, row in group.iterrows():
            model = row["model"]
            kind = row["kind"]
            seed = row["seed"]
            folder = model + ("_ablation" if kind == "ablation" else "")
            ckpt_dir = os.path.join(args.logs_best_dir, pair, folder, f"seed{seed}")
            ckpt_path = os.path.join(ckpt_dir, "best_checkpoint.pth")
            hparam_path = os.path.join(ckpt_dir, "hparam.yaml")

            if not os.path.exists(ckpt_path):
                print(f"  ⚠ missing checkpoint, skipping: {ckpt_path}")
                continue

            hp = load_hparams(hparam_path)
            try:
                n_params = count_params(ckpt_path)
            except Exception as e:
                print(f"  ⚠ could not load {ckpt_path}: {e} -- trainable_parameters left blank")
                n_params = ""

            mtime = time.strftime("%Y%m%d-%H%M%S", time.localtime(os.path.getmtime(ckpt_path)))

            def pick(manifest_key, hparam_key):


                v = row.get(manifest_key, "")
                if v == "" or pd.isna(v):
                    return hp.get(hparam_key, "")
                return v

            rows.append({
                "train_dataset": pair,
                "test_dataset": tgt,
                "model": model,
                "trainable_parameters": n_params,
                "time": mtime,
                "inference_time": "",
                "LR": pick("LR", "learning_rate"),
                "batch_size": pick("batch_size", "batch_size"),
                "loss_type": pick("loss_type", "loss_type"),
                "best_last_RMSE": row["rmse"],
                "score": row["score"],
                "windowsize": pick("windowsize", "input_length"),
                "hidden_dim": pick("hidden_dim", "hidden_dim"),
                "train": True,
                "savepath": ckpt_dir.replace("\\", "/"),
                "resume": False,
                "resumepath": "",
                "info": build_info(model, pair, seed, kind),
            })

        if not rows:
            continue

        out_path = os.path.join(args.logs_best_dir, f"{pair}experimental_logs.csv")
        pd.DataFrame(rows, columns=COLUMNS).to_csv(out_path, index=False)
        print(f"  wrote {len(rows)} rows -> {out_path}")
        written += 1

    print(f"\n{written} CSV file(s) written under {args.logs_best_dir}")


if __name__ == "__main__":
    main()
