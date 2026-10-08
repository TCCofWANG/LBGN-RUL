

import argparse
import csv
import glob
import os
import re
import statistics as stats
import subprocess
import sys

RESULT_RE = re.compile(
    r"INFERENCE_TIME_RESULT model=(?P<model>\S+) params=(?P<params>\d+) "
    r"mean_ms=(?P<mean_ms>[\d.]+) std_ms=(?P<std_ms>[\d.]+) "
    r"n_batches=(?P<n_batches>\d+) n_samples=(?P<n_samples>\d+) num_runs=(?P<num_runs>\d+)"
)

PER_CKPT_COLUMNS = ["pair", "model", "seed", "params", "mean_ms", "std_ms",
                     "n_batches", "n_samples", "num_runs"]


PER_MODEL_COLUMNS = ["model", "params_mean", "params_min", "params_max",
                      "mean_ms", "std_ms", "n_checkpoints"]


def discover_checkpoints(logs_dir, tag_glob, models_filter, max_seeds_per_pair):

    pattern = os.path.join(logs_dir, "*", "*", tag_glob, "best_checkpoint.pth")
    by_pair_model = {}
    for ckpt in sorted(glob.glob(pattern)):
        ckpt_dir = os.path.dirname(ckpt)
        tag = os.path.basename(ckpt_dir)
        model = os.path.basename(os.path.dirname(ckpt_dir))
        pair = os.path.basename(os.path.dirname(os.path.dirname(ckpt_dir)))
        if models_filter and model not in models_filter:
            continue
        m = re.search(r"_s(\d+)$", tag) or re.search(r"(\d+)$", tag)
        seed = m.group(1) if m else "0"
        yaml_path = os.path.join(ckpt_dir, "hparam.yaml")
        by_pair_model.setdefault((pair, model), []).append((seed, ckpt, yaml_path))

    out = []
    for (pair, model), entries in sorted(by_pair_model.items()):
        entries.sort(key=lambda e: (e[0] != "42", e[0]))
        for seed, ckpt, yaml_path in entries[:max_seeds_per_pair]:
            out.append((pair, model, seed, ckpt, yaml_path))
    return out


def run_one(py, ckpt, yaml_path, pair, model, seed, num_runs, warmup):
    src, tgt = pair.split("_", 1)
    cmd = [
        py, "CMAPSS_DA.py",
        "--model_name", model,
        "--dataset_name", "CMAPSS",
        "--Data_id_CMAPSS", src,
        "--Data_id_CMAPSS_test", tgt,
        "--seed", seed,
        "--train", "False",
        "--resume", "True",
        "--warm_start_checkpoint", ckpt,
        "--save_path", f"final_s{seed}",
        "--inference_time_only", "True",
        "--inference_time_runs", str(num_runs),
        "--inference_time_warmup", str(warmup),
    ]
    if os.path.exists(yaml_path):
        cmd += ["--warm_start_hparam_yaml", yaml_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    m = RESULT_RE.search(result.stdout)
    if not m:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-5:]
        return None, tail
    d = m.groupdict()
    return {
        "pair": pair, "model": model, "seed": seed,
        "params": int(d["params"]), "mean_ms": float(d["mean_ms"]),
        "std_ms": float(d["std_ms"]), "n_batches": int(d["n_batches"]),
        "n_samples": int(d["n_samples"]), "num_runs": int(d["num_runs"]),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    ap.add_argument("--out_dir", default="./analysis/out")
    ap.add_argument("--tag_glob", default="final_s*")
    ap.add_argument("--models", nargs="*", default=None,
                     help="restrict to these model_name values (default: all found)")
    ap.add_argument("--max_seeds_per_pair", type=int, default=1,
                     help="how many seeds of each (pair, model) to benchmark (default 1, "
                          "seed 42 preferred first) -- see module docstring's SCOPE section")
    ap.add_argument("--num_runs", type=int, default=20,
                     help="timed forward-pass repeats per checkpoint (default 100)")
    ap.add_argument("--warmup", type=int, default=10)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    ckpts = discover_checkpoints(args.logs_dir, args.tag_glob, set(args.models or []),
                                  args.max_seeds_per_pair)

    print(f"{len(ckpts)} checkpoint(s) selected "
          f"(max_seeds_per_pair={args.max_seeds_per_pair}, num_runs={args.num_runs} each "
          f"-> {len(ckpts) * args.num_runs} total timed forward passes)")
    by_model = {}
    for pair, model, seed, _, _ in ckpts:
        by_model.setdefault(model, []).append(f"{pair}/s{seed}")
    for model in sorted(by_model):
        print(f"  {model:14s} n={len(by_model[model])}")
    print()

    if args.dry_run:
        for pair, model, seed, ckpt, _ in ckpts:
            print(f"  would run  {pair:14s} {model:14s} seed={seed}  {ckpt}")
        print(f"\n{len(ckpts)} would be measured. Re-run with --apply to actually launch them.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    per_ckpt_path = os.path.join(args.out_dir, "inference_time_per_checkpoint.csv")
    per_model_path = os.path.join(args.out_dir, "inference_time_per_model.csv")

    rows, failed = [], []
    for i, (pair, model, seed, ckpt, yaml_path) in enumerate(ckpts, 1):
        print(f"[{i}/{len(ckpts)}] {pair} | {model} | seed {seed} ... ", end="", flush=True)
        row, err_tail = run_one(py, ckpt, yaml_path, pair, model, seed, args.num_runs, args.warmup)
        if row is None:
            print("FAILED")
            for line in err_tail:
                print(f"    {line}")
            failed.append((pair, model, seed))
            continue
        rows.append(row)
        print(f"{row['mean_ms']:.3f}ms +/- {row['std_ms']:.3f}ms  ({row['n_samples']} samples)")

    with open(per_ckpt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_CKPT_COLUMNS)
        w.writeheader()
        w.writerows(rows)

    by_model_rows = {}
    for r in rows:
        by_model_rows.setdefault(r["model"], []).append(r)
    summary = []
    for model, rs in sorted(by_model_rows.items()):
        means = [r["mean_ms"] for r in rs]
        param_vals = [r["params"] for r in rs]
        if len(set(param_vals)) > 1:
            print(f"NOTE: {model}'s parameter count varies across checkpoints "
                  f"({min(param_vals)}-{max(param_vals)}) -- architecture is tuned per source "
                  f"domain, not fixed. See params_min/params_max, not a single 'params' value.")
        summary.append({
            "model": model,
            "params_mean": round(stats.mean(param_vals), 1),
            "params_min": min(param_vals), "params_max": max(param_vals),
            "mean_ms": stats.mean(means),
            "std_ms": stats.stdev(means) if len(means) > 1 else 0.0,
            "n_checkpoints": len(rs),
        })
    with open(per_model_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PER_MODEL_COLUMNS)
        w.writeheader()
        w.writerows(summary)

    print(f"\n{len(rows)} measured, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)
    print(f"Saved: {per_ckpt_path}")
    print(f"Saved: {per_model_path}")


if __name__ == "__main__":
    main()
