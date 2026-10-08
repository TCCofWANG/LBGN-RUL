

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = "CMAPSS_DA.py"
MODEL = "DLGNet"
LOGS_ROOT = "./logs"

PAIRS = [
    ("FD001", "FD002"),
    ("FD001", "FD004"),
    ("FD003", "FD002"),
    ("FD003", "FD004"),
]
SEEDS = [0, 1, 7, 41, 42, 64, 1234, 2026]


def ckpt_path(src, tgt, seed):
    return PROJECT_ROOT / "logs" / f"{src}_{tgt}" / MODEL / f"final_s{seed}" / "best_checkpoint.pth"


def main():
    total = len(PAIRS) * len(SEEDS)
    done = 0

    print("=" * 66)
    print(f"  DLGNet rerun  |  pairs: {PAIRS}  |  {total} experiments")
    print(f"  Seeds: {SEEDS}")
    print("=" * 66)

    for src, tgt in PAIRS:
        for seed in SEEDS:
            done += 1
            ckpt = ckpt_path(src, tgt, seed)

            if ckpt.exists():
                print(f"[skip] [{done}/{total}] {src} -> {tgt} | seed {seed} (done)")
                continue

            print(f"\n[run ] [{done}/{total}] {src} -> {tgt} | seed {seed}")

            cmd = [
                sys.executable, SCRIPT,
                "--model_name", MODEL,
                "--dataset_name", "CMAPSS",
                "--Data_id_CMAPSS", src,
                "--Data_id_CMAPSS_test", tgt,
                "--seed", str(seed),
                "--logs_root", LOGS_ROOT,
                "--save_path", f"final_s{seed}",
                "--info", f"{MODEL}_{src}_{tgt}_seed{seed}_final",
            ]
            result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                print(f"[FAIL] {src} -> {tgt} | seed {seed}")

    print()
    print("=" * 66)
    print(f"  DLGNet rerun complete.  Results in: {LOGS_ROOT}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
