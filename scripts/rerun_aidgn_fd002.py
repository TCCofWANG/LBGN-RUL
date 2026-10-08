

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = "CMAPSS_DA.py"
MODEL = "AIDGN"
LOGS_ROOT = "./logs"

SRC = "FD002"
TARGETS = ["FD003", "FD004"]
SEEDS = [0, 1, 7, 41, 42, 64, 1234, 2026]


def ckpt_path(tgt, seed):
    return PROJECT_ROOT / "logs" / f"{SRC}_{tgt}" / MODEL / f"final_s{seed}" / "best_checkpoint.pth"


def main():
    total = len(TARGETS) * len(SEEDS)
    done = 0

    print("=" * 66)
    print(f"  AIDGN rerun  |  SOURCE: {SRC}  |  targets: {TARGETS}  |  {total} experiments")
    print(f"  Seeds: {SEEDS}")
    print("=" * 66)

    for tgt in TARGETS:
        for seed in SEEDS:
            done += 1
            ckpt = ckpt_path(tgt, seed)

            if ckpt.exists():
                print(f"[skip] [{done}/{total}] {SRC} -> {tgt} | seed {seed} (done)")
                continue

            print(f"\n[run ] [{done}/{total}] {SRC} -> {tgt} | seed {seed}")

            cmd = [
                sys.executable, SCRIPT,
                "--model_name", MODEL,
                "--dataset_name", "CMAPSS",
                "--Data_id_CMAPSS", SRC,
                "--Data_id_CMAPSS_test", tgt,
                "--seed", str(seed),
                "--logs_root", LOGS_ROOT,
                "--save_path", f"final_s{seed}",
                "--info", f"{MODEL}_{SRC}_{tgt}_seed{seed}_final",
            ]
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"[FAIL] {SRC} -> {tgt} | seed {seed}")

    print()
    print("=" * 66)
    print(f"  AIDGN rerun complete.  Results in: {LOGS_ROOT}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
