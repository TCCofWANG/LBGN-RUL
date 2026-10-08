

import argparse
import os
import subprocess
import sys

ALL_SOURCES = ["FD001", "FD002", "FD003", "FD004"]


CONFIGS = [("0.5", "0.1"), ("1.5", "1.0"), ("3.0", "2.0"), ("5.0", "3.0"), ("5.0", "5.0")]
SEED = 42
LOGS_ROOT = "./logs_lambda_sensitivity"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="restrict to this source's targets, e.g. FD001")
    ap.add_argument("--target", default=None, help="with --source, restrict to this exact pair")
    args = ap.parse_args()

    pairs = []
    for src in ALL_SOURCES:
        if args.source and src != args.source:
            continue
        for tgt in ALL_SOURCES:
            if tgt == src:
                continue
            if args.target and tgt != args.target:
                continue
            pairs.append((src, tgt))

    total = len(pairs) * len(CONFIGS)
    print("==================================================================")
    print(f"  lam_mar/lam_da sensitivity  |  LBGN_RUL  |  CMAPSS  |  {total} experiments")
    print(f"  Pairs   : {[f'{s}_{t}' for s, t in pairs]}")
    print(f"  Configs : {CONFIGS}  (lam_mar,lam_da)")
    print(f"  Seed    : {SEED}")
    print(f"  Logs    : {LOGS_ROOT}")
    print("==================================================================")

    py = sys.executable
    done = 0
    for src, tgt in pairs:
        pair = f"{src}_{tgt}"
        for lmar, lda in CONFIGS:
            done += 1
            tag = f"lmar{lmar}_lda{lda}_s{SEED}"
            ckpt = os.path.join(LOGS_ROOT, pair, "LBGN_RUL", tag, "best_checkpoint.pth")

            if os.path.exists(ckpt):
                print(f"SKIP [{done}/{total}] {pair} | lam_mar={lmar} lam_da={lda} (done)")
                continue

            print(f"\nRUN  [{done}/{total}] {pair} | lam_mar={lmar} lam_da={lda}")
            cmd = [
                py, "CMAPSS_DA.py",
                "--model_name", "LBGN_RUL",
                "--dataset_name", "CMAPSS",
                "--Data_id_CMAPSS", src,
                "--Data_id_CMAPSS_test", tgt,
                "--loss_type", "DA",
                "--seed", str(SEED),
                "--lam_mar", lmar,
                "--lam_da", lda,
                "--logs_root", LOGS_ROOT,
                "--save_path", tag,
                "--info", f"LBGN_RUL_{pair}_{tag}_sensitivity",
            ]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print(f"FAIL [{done}/{total}] {pair} | lam_mar={lmar} lam_da={lda} (rc={result.returncode})")

    print(f"\n==================================================================")
    print(f"  Done.  {done}/{total} experiments attempted.  Results in: {LOGS_ROOT}/")
    print(f"==================================================================")


if __name__ == "__main__":
    main()
