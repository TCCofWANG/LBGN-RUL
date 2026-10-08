

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = "CMAPSS_DA.py"
MODEL = "LBGN_RUL"
LOGS_ROOT = "./logs_abl"

SOURCES = ["FD003", "FD004"]
TARGETS_BY_SOURCE = {
    "FD001": ["FD002", "FD003", "FD004"],
    "FD002": ["FD001", "FD003", "FD004"],
    "FD003": ["FD001", "FD002", "FD004"],
    "FD004": ["FD001", "FD002", "FD003"],
}

VARIANTS = ["source_only"]

COLD_SEED = 42
WARM_SEEDS = [0, 1, 7, 41, 64, 1234, 2026]
ALL_SEEDS = [COLD_SEED] + WARM_SEEDS


def run_one(src, tgt, variant, seed, resume, resume_path, tag):
    cmd = [
        sys.executable, SCRIPT,
        "--model_name", MODEL,
        "--dataset_name", "CMAPSS",
        "--Data_id_CMAPSS", src,
        "--Data_id_CMAPSS_test", tgt,
        "--seed", str(seed),
        "--logs_root", LOGS_ROOT,
        "--save_path", tag,
        "--resume", str(resume),
        "--resume_path", resume_path,
        "--lbgn_ablation", variant,
        "--info", f"{MODEL}_{src}_{tgt}_seed{seed}_ablation_{variant}",
    ]
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return result.returncode == 0


def ckpt_path(src, tgt, tag):
    return PROJECT_ROOT / LOGS_ROOT.lstrip("./") / f"{src}_{tgt}" / MODEL / tag / "best_checkpoint.pth"


def main():
    total = sum(len(TARGETS_BY_SOURCE[s]) for s in SOURCES) * len(VARIANTS) * len(ALL_SEEDS)
    done = 0

    print("=" * 66)
    print(f"  LBGN source_only  |  ALL SOURCES  |  {total} experiments  |  seeds: {ALL_SEEDS}")
    print("=" * 66)

    for src in SOURCES:
        for tgt in TARGETS_BY_SOURCE[src]:
            for variant in VARIANTS:
                cold_tag = f"abl_{variant}_s{COLD_SEED}"
                cold_ckpt = ckpt_path(src, tgt, cold_tag)

                done += 1
                if cold_ckpt.exists():
                    print(f"[skip] [{done}/{total}] {src} -> {tgt} | {variant} | seed {COLD_SEED} (done)")
                else:
                    print(f"\n[run ] [{done}/{total}] {src} -> {tgt} | {variant} | seed {COLD_SEED} (cold start)")
                    ok = run_one(src, tgt, variant, COLD_SEED, resume=False, resume_path=cold_tag, tag=cold_tag)
                    if not ok:
                        print(f"[FAIL] {src} -> {tgt} | {variant} | seed {COLD_SEED}")

                if not cold_ckpt.exists():
                    print(f"[warn] seed {COLD_SEED} checkpoint still missing for {src} -> {tgt} | {variant} "
                          f"-- skipping its {len(WARM_SEEDS)} warm-start seeds")
                    done += len(WARM_SEEDS)
                    continue

                for seed in WARM_SEEDS:
                    tag = f"abl_{variant}_s{seed}"
                    ckpt = ckpt_path(src, tgt, tag)

                    done += 1
                    if ckpt.exists():
                        print(f"[skip] [{done}/{total}] {src} -> {tgt} | {variant} | seed {seed} (done)")
                        continue

                    print(f"\n[run ] [{done}/{total}] {src} -> {tgt} | {variant} | seed {seed} (warm start from seed {COLD_SEED})")
                    ok = run_one(src, tgt, variant, seed, resume=True, resume_path=cold_tag, tag=tag)
                    if not ok:
                        print(f"[FAIL] {src} -> {tgt} | {variant} | seed {seed}")

    print()
    print("=" * 66)
    print(f"  source_only ablation complete.  Results in: {LOGS_ROOT}/")
    print("=" * 66)


if __name__ == "__main__":
    main()
