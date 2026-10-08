

import argparse
import csv
import os
import re
import sys

import subprocess

RESULT_RE = re.compile(
    r"FLOPS_RESULT model=(?P<model>\S+) params=(?P<params>\d+) "
    r"thop_layer_macs=(?P<thop_layer_macs>\d+) "
    r"einsum_matmul_macs=(?P<einsum_matmul_macs>\d+) "
    r"fft_macs_estimated=(?P<fft_macs_estimated>[\d.]+) "
    r"total_macs_excl_fft=(?P<total_macs_excl_fft>\d+) "
    r"total_macs_incl_fft=(?P<total_macs_incl_fft>\d+)"
)

ALL_MODELS = ["LBGN_RUL", "DAGCN_RUL", "EviAdaptRUL", "DAST_RUL", "CADA_RUL",
              "TACDA_RUL", "OCS_DANN", "MDAN_RUL", "CCDG_RUL"]


CHECKPOINTS = [(m, "FD001_FD002", "42") for m in ALL_MODELS] +\
              [("LBGN_RUL", "FD002_FD001", "42")]

COLUMNS = ["model", "pair", "seed", "params", "thop_layer_macs",
           "einsum_matmul_macs", "fft_macs_estimated",
           "total_macs_excl_fft", "total_macs_incl_fft"]


def run_one(py, logs_dir, model, pair, seed):
    src, tgt = pair.split("_", 1)
    ckpt_dir = os.path.join(logs_dir, pair, model, f"final_s{seed}")
    ckpt = os.path.join(ckpt_dir, "best_checkpoint.pth")
    yaml_path = os.path.join(ckpt_dir, "hparam.yaml")
    if not os.path.exists(ckpt):
        return None, [f"checkpoint not found: {ckpt}"]

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
        "--flops_only", "True",
    ]
    if os.path.exists(yaml_path):
        cmd += ["--warm_start_hparam_yaml", yaml_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    m = RESULT_RE.search(result.stdout)
    if not m:
        tail = (result.stderr or result.stdout or "").strip().splitlines()[-6:]
        return None, tail
    d = m.groupdict()
    return {
        "model": model, "pair": pair, "seed": seed,
        "params": int(d["params"]),
        "thop_layer_macs": int(d["thop_layer_macs"]),
        "einsum_matmul_macs": int(d["einsum_matmul_macs"]),
        "fft_macs_estimated": float(d["fft_macs_estimated"]),
        "total_macs_excl_fft": int(d["total_macs_excl_fft"]),
        "total_macs_incl_fft": int(d["total_macs_incl_fft"]),
    }, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    ap.add_argument("--out_dir", default="./analysis/out")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    print(f"{len(CHECKPOINTS)} checkpoint(s) to profile (1 per model, LBGN_RUL x2 "
          f"for its two architecture variants):")
    for model, pair, seed in CHECKPOINTS:
        print(f"  {model:14s} {pair:14s} seed={seed}")
    if args.dry_run:
        print("\nRe-run with --apply to actually profile these.")
        return

    py = sys.executable
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "flops_per_model.csv")

    rows, failed = [], []
    for model, pair, seed in CHECKPOINTS:
        print(f"[{model} | {pair} | seed {seed}] ... ", end="", flush=True)
        row, err = run_one(py, args.logs_dir, model, pair, seed)
        if row is None:
            print("FAILED")
            for line in err:
                print(f"    {line}")
            failed.append((model, pair, seed))
            continue
        rows.append(row)
        gmacs_excl = row["total_macs_excl_fft"] / 1e9
        print(f"params={row['params']:,}  MACs(excl.FFT)={row['total_macs_excl_fft']:,} "
              f"({gmacs_excl:.4f} GMACs)  fft_macs~={row['fft_macs_estimated']:,.0f}")

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    print(f"\n{len(rows)} measured, {len(failed)} failed.")
    if failed:
        print("Failed:", failed)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
