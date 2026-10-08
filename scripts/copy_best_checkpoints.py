

import argparse
import glob
import os
import shutil
import sys

import pandas as pd
import yaml


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs.hparams import get_configs


EXPERIMENTAL_LOG_COLUMNS = [
    "train_dataset", "test_dataset", "model", "trainable_parameters", "time",
    "inference_time", "LR", "batch_size", "loss_type", "best_last_RMSE", "score",
    "windowsize", "hidden_dim", "train", "savepath", "resume", "resumepath", "info",
]

_alg_hparams_cache = {}


def current_alg_hparams(dataset_name, source, model):

    key = (dataset_name, source, model)
    if key not in _alg_hparams_cache:
        try:
            hparams_class = get_configs(dataset_name, source)
            _alg_hparams_cache[key] = hparams_class.alg_hparams.get(model)
        except Exception as e:
            print(f"  ⚠ could not load current hparams for {source}/{model}: {e} -- "
                  f"architecture check skipped for this group")
            _alg_hparams_cache[key] = None
    return _alg_hparams_cache[key]


SHAPE_CRITICAL_KEYS = {
    "num_nodes", "input_length", "hidden_dim", "K", "ndc_hidden",
    "LCE_dim", "n_bands", "z_dim", "lstm_hid", "lstm_n_layers",
    "d_model", "nhead", "nlayers",
}


def architecture_matches(checkpoint_dir, expected):

    if expected is None:
        return True
    yaml_path = os.path.join(checkpoint_dir, "hparam.yaml")
    if not os.path.exists(yaml_path):
        return False
    with open(yaml_path, "r", encoding="utf-8") as f:
        saved = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
    for key, expected_val in expected.items():
        if key not in SHAPE_CRITICAL_KEYS:
            continue
        if key not in saved:
            continue
        if saved[key] != expected_val:
            return False
    return True


def pick_best_seed(sub):

    rmse_winner = sub.loc[sub["best_last_RMSE"].idxmin()]
    score_winner = sub.loc[sub["score"].idxmin()]

    if rmse_winner["seed"] == score_winner["seed"]:
        return rmse_winner, "rmse_and_score_agree"

    rmse_mean = sub["best_last_RMSE"].mean()
    score_mean = sub["score"].mean()

    def avg_improvement(row):
        rmse_impr = (rmse_mean - row["best_last_RMSE"]) / rmse_mean * 100
        score_impr = (score_mean - row["score"]) / score_mean * 100
        return (rmse_impr + score_impr) / 2

    return (rmse_winner, "conflict_avg_improvement") if avg_improvement(rmse_winner) >= avg_improvement(score_winner)\
        else (score_winner, "conflict_avg_improvement")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./external_data/lr_logs")
    ap.add_argument("--out", default="./logs_best")
    ap.add_argument("--dataset_name", default="CMAPSS", choices=["CMAPSS", "N_CMAPSS"],
                    help="which configs/hparams.py dataset class to check candidate "
                         "architectures against (source id prefix, e.g. FD00x vs DS0x, "
                         "doesn't change -- only which hparams.py class is consulted)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="restrict to these model names (default: every model found in the logs)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing exp0 dir instead of skipping it")
    args = ap.parse_args()

    csv_paths = glob.glob(os.path.join(args.logs_dir, "*experimental_logs.csv"))
    if not csv_paths:
        raise SystemExit(f"no experimental_logs.csv found under {args.logs_dir}")

    frames = []
    for f in csv_paths:
        try:
            frames.append(pd.read_csv(f))
        except pd.errors.EmptyDataError:
            print(f"  ⚠ skipping empty/corrupt CSV: {f}")
    if not frames:
        raise SystemExit("every experimental_logs.csv found was empty/corrupt -- nothing to select from")

    df_all = pd.concat(frames, ignore_index=True)
    df_all["seed"] = df_all["info"].fillna("").str.extract(r"seed(\d+)")


    df_all["pair"] = df_all.apply(
        lambda r: r["train_dataset"] if r["train_dataset"] == r["test_dataset"]
        else f"{r['train_dataset']}_{r['test_dataset']}",
        axis=1,
    )

    if args.models:
        df_all = df_all[df_all.model.isin(args.models)]

    os.makedirs(args.out, exist_ok=True)
    manifest = []
    best_rows_by_pair = {}
    copied = skipped_exists = skipped_missing = skipped_arch_mismatch = 0

    has_final_tags = df_all["info"].fillna("").str.contains("_final").any()

    if has_final_tags:

        KINDS = {
            "final": ("_final", ""),
            "ablation": ("_ablation", "_ablation"),
        }
    else:


        KINDS = {"grid": (None, "")}

    for kind, (info_substr, dst_suffix) in KINDS.items():
        if info_substr is None:
            df = df_all.copy()
        else:
            df = df_all[df_all["info"].fillna("").str.contains(info_substr)].copy()
            if kind == "final":
                df = df[~df["info"].fillna("").str.contains("_ablation")]

        for (pair, model), sub in df.groupby(["pair", "model"]):
            sub = sub.dropna(subset=["best_last_RMSE", "score"]).copy()
            if sub.empty:
                continue

            def row_src(row):


                sp = str(row.get("savepath", "")).strip().replace("\\", "/").rstrip("/")
                if not sp:
                    return ""
                if os.path.isabs(sp) or sp.startswith(args.logs_dir.replace("\\", "/")):
                    return sp


                exp_tag = os.path.basename(sp)
                return os.path.join(args.logs_dir, pair, model, exp_tag)

            sub["__src"] = sub.apply(row_src, axis=1)


            source = pair.split("_")[0]
            expected_arch = current_alg_hparams(args.dataset_name, source, model)
            arch_ok = sub["__src"].apply(lambda s: bool(s) and os.path.isdir(s)
                                          and architecture_matches(s, expected_arch))
            n_excluded = int((~arch_ok).sum())
            if n_excluded:
                print(f"  ⚠ {pair}/{model}: excluding {n_excluded} candidate(s) with mismatched "
                      f"architecture (stale checkpoint from a different sweep)")
                skipped_arch_mismatch += n_excluded
            sub = sub[arch_ok]
            if sub.empty:
                print(f"  ⚠ {pair}/{model}: no architecture-consistent candidates left -- skipping")
                continue

            best, rule = pick_best_seed(sub)
            seed = best["seed"]
            rmse = float(best["best_last_RMSE"])
            score = float(best["score"])
            src = best["__src"]
            if not src or not os.path.isdir(src):
                print(f"  ⚠ WARNING: best {kind} seed for {pair}/{model} (seed {seed}, rmse {rmse:.2f}) "
                      f"not found on disk at {src} -- skipping")
                skipped_missing += 1
                continue

            model_dir = os.path.join(args.out, pair, model + dst_suffix)
            dst = os.path.join(model_dir, f"seed{seed}")


            if os.path.isdir(model_dir):
                for entry in os.listdir(model_dir):
                    stale_path = os.path.join(model_dir, entry)
                    if entry.startswith("seed") and stale_path != dst and os.path.isdir(stale_path):
                        print(f"  ⚠ {pair}/{model}{dst_suffix}: removing stale {entry} "
                              f"(no longer the best seed)")
                        shutil.rmtree(stale_path)

            already_exists = os.path.isdir(dst)
            if already_exists and not args.force:
                print(f"  ⏭ skip {pair}/{model}{dst_suffix} (seed{seed} already exists, use --force to overwrite)")
                skipped_exists += 1
            else:
                if already_exists:
                    shutil.rmtree(dst)
                os.makedirs(dst, exist_ok=True)

                n_files = 0
                for fn in ["best_checkpoint.pth", "hparam.yaml", "result.npz"]:
                    sp = os.path.join(src, fn)
                    if os.path.exists(sp):
                        shutil.copy2(sp, os.path.join(dst, fn))
                        n_files += 1

                flag = "" if rule == "rmse_and_score_agree" else "  [conflict, tie-broken by avg improvement]"
                print(f"  {pair:14s} {model + dst_suffix:24s} best seed={seed:<4} rmse={rmse:.2f}  score={score:.2f}  "
                      f"({n_files} files) -> {dst}{flag}")
                copied += 1


            manifest.append({"pair": pair, "model": model, "kind": kind, "seed": seed,
                              "rmse": round(rmse, 4), "score": round(score, 4), "rule": rule,
                              "batch_size": best.get("batch_size", ""),
                              "loss_type": best.get("loss_type", ""),
                              "windowsize": best.get("windowsize", ""), "hidden_dim": best.get("hidden_dim", ""),
                              "src": src})

            row = {col: best.get(col, "") for col in EXPERIMENTAL_LOG_COLUMNS}
            row["model"] = model + dst_suffix
            row["savepath"] = dst.replace("\\", "/")
            best_rows_by_pair.setdefault(pair, []).append(row)

    manifest_path = os.path.join(args.out, "best_checkpoint_manifest.csv")
    pd.DataFrame(manifest).to_csv(manifest_path, index=False)

    for pair, rows in best_rows_by_pair.items():
        csv_path = os.path.join(args.out, f"{pair}experimental_logs.csv")
        pd.DataFrame(rows, columns=EXPERIMENTAL_LOG_COLUMNS).to_csv(csv_path, index=False)

    print(f"\ncopied {copied}, skipped (exists) {skipped_exists}, skipped (missing on disk) {skipped_missing}, "
          f"candidates excluded (architecture mismatch) {skipped_arch_mismatch}")
    print(f"manifest -> {manifest_path}")
    print(f"per-pair experimental_logs.csv written for {len(best_rows_by_pair)} pair(s) under {args.out}/")


if __name__ == "__main__":
    main()
