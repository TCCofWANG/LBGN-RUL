

import argparse
import os
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = "CMAPSS_DA.py"

ALL_DATASETS = ["FD001", "FD002", "FD003", "FD004"]

MODELS = ["DAGCN_RUL", "EviAdaptRUL", "DAST_RUL", "CADA_RUL", "TACDA_RUL",
          "OCS_DANN", "MDAN_RUL", "CCDG_RUL", "LBGN_RUL", "NDC_PDMN", "LBGN_RUL_v2", "DLGNet", "AIDGN"]
SEEDS_BY_SOURCE = {
    "FD001": [0, 1, 7, 41, 64, 1234, 2026],
    "FD002": [0, 1, 7, 41, 64, 1234, 2026],
    "FD003": [0, 1, 7, 41, 64, 1234, 2026],
    "FD004": [0, 1, 7, 41, 64, 1234, 2026],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=ALL_DATASETS, choices=ALL_DATASETS,
                     help="which datasets to use as source (default: all 4)")
    args = ap.parse_args()


    jobs = []
    for src in args.sources:
        targets = [t for t in ALL_DATASETS if t != src]
        for tgt in targets:
            for seed in SEEDS_BY_SOURCE[src]:
                for model in MODELS:
                    jobs.append((src, tgt, model, seed))

    total = len(jobs)
    done_count = 0

    print("=" * 66)
    print(f"  Final runs  |  sources: {args.sources}  |  {total} experiments")
    print(f"  Models : {MODELS}")
    for src in args.sources:
        print(f"  Seeds ({src}): {SEEDS_BY_SOURCE[src]}")
    print("=" * 66)

    for src, tgt, model, seed in jobs:
        done_count += 1
        tag = f"final_s{seed}"
        ckpt = os.path.join(REPO_ROOT, "logs", f"{src}_{tgt}", model, tag, "best_checkpoint.pth")
        source_ckpt = os.path.join("logs", f"{src}_{tgt}", model, "final_s42", "best_checkpoint.pth")
        source_path = os.path.join("logs", f"{src}_{tgt}", model, "final_s42")

        if os.path.isfile(ckpt):
            print(f"skip  [{done_count}/{total}] {src} -> {tgt} | {model} | seed {seed} (done)")
            continue

        print()
        print(f"run   [{done_count}/{total}] {src} -> {tgt} | {model} | seed {seed}")

        cmd = [
            sys.executable, SCRIPT,
            "--model_name", model,
            "--dataset_name", "CMAPSS",
            "--Data_id_CMAPSS", src,
            "--Data_id_CMAPSS_test", tgt,
            "--seed", str(seed),
            "--logs_root", "./logs",
            "--resume", str(True),
            "--save_path", tag,
            "--resume_path", source_path,
            "--warm_start_checkpoint" , source_ckpt,
            "--info", f"{model}_{src}_{tgt}_seed{seed}_final",
        ]
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"FAIL  {src} -> {tgt} | {model} | seed {seed}")

    print()
    print("=" * 66)
    print(f"  Sources {args.sources} complete.  Results in: ./logs/")
    print("=" * 66)


if __name__ == "__main__":
    main()
