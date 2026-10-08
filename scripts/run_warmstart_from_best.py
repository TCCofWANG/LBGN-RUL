

import os
import subprocess
import sys

LOGS_BEST_DIR = "./logs_best"
SCRIPT = "CMAPSS_DA.py"

MODELS = ["LBGN_RUL", "PDMN", "NDC_PDMN", "DAGCN_RUL", "EviAdaptRUL",
          "DAST_RUL", "CADA_RUL", "TACDA_RUL", "OCS_DANN", "MDAN_RUL", "CCDG_RUL"]


SEEDS = [1, 64, 1001, 1002]

PAIRS = [
    ("FD001", "FD002"), ("FD001", "FD003"), ("FD001", "FD004"),
    ("FD002", "FD001"), ("FD002", "FD003"), ("FD002", "FD004"),
]


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    py = sys.executable
    total = len(PAIRS) * len(MODELS) * len(SEEDS)
    done_count = 0

    print("══════════════════════════════════════════════════════════════")
    print(f"  Warm-start from logs_best  |  {total} experiments")
    print(f"  Models : {' '.join(MODELS)}")
    print(f"  Seeds  : {' '.join(map(str, SEEDS))}")
    print("══════════════════════════════════════════════════════════════")

    for src, tgt in PAIRS:
        for model in MODELS:
            best_ckpt = os.path.join(LOGS_BEST_DIR, f"{src}_{tgt}", model, "exp0", "best_checkpoint.pth")
            best_hparam_yaml = os.path.join(LOGS_BEST_DIR, f"{src}_{tgt}", model, "exp0", "hparam.yaml")
            if not os.path.exists(best_ckpt):
                print(f"⚠  no best checkpoint for {src} → {tgt} | {model} at {best_ckpt} -- skipping all seeds")
                done_count += len(SEEDS)
                continue

            for seed in SEEDS:
                done_count += 1
                tag = f"warmstart_s{seed}"
                ckpt = os.path.join("./logs", f"{src}_{tgt}", model, tag, "best_checkpoint.pth")

                if os.path.exists(ckpt):
                    print(f"⏭  [{done_count}/{total}] skip {src} → {tgt} | {model} | seed {seed} (done)")
                    continue

                print()
                print(f"▶  [{done_count}/{total}] {src} → {tgt} | {model} | seed {seed}  "
                      f"(warm-start from {best_ckpt})")

                cmd = [
                    py, SCRIPT,
                    "--model_name", model,
                    "--dataset_name", "CMAPSS",
                    "--Data_id_CMAPSS", src,
                    "--Data_id_CMAPSS_test", tgt,
                    "--seed", str(seed),
                    "--train", "True",
                    "--resume", "True",
                    "--warm_start_checkpoint", best_ckpt,
                    "--warm_start_hparam_yaml", best_hparam_yaml,
                    "--save_path", tag,
                    "--info", f"{model}_{src}_{tgt}_seed{seed}_warmstart",
                ]
                result = subprocess.run(cmd)
                if result.returncode != 0:
                    print(f"✗  [FAIL] {src} → {tgt} | {model} | seed {seed}")

    print()
    print("════════════════════════════════════════════════════════════════")
    print("  Warm-start runs complete.  Results in: logs/  (info tag: _warmstart)")
    print("════════════════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
