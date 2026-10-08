

import argparse
import os
import pandas as pd

STALE_MODELS_BY_SOURCE = {
    "FD001": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL", "LBGN_RUL"],
    "FD002": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL", "TACDA_RUL"],
    "FD003": ["CADA_RUL", "DAGCN_RUL", "EviAdaptRUL", "LBGN_RUL", "NDC_PDMN", "PDMN"],
    "FD004": ["DAGCN_RUL", "TACDA_RUL"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="./logs")
    args = ap.parse_args()

    total_removed = 0
    for src, models in STALE_MODELS_BY_SOURCE.items():
        for tgt in ["FD001", "FD002", "FD003", "FD004"]:
            if tgt == src:
                continue
            pair = f"{src}_{tgt}"
            csv_path = os.path.join(args.logs_dir, f"{pair}experimental_logs.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path)
            seed = df["info"].str.extract(r"seed(\d+)")[0]
            is_stale = (df.model.isin(models) &
                        df["info"].fillna("").str.contains("_final") &
                        seed.isin(["0", "42", "64"]))
            n = int(is_stale.sum())
            if n:
                df[~is_stale].to_csv(csv_path, index=False)
                total_removed += n
                print(f"{pair}: removed {n} stale row(s)")

    print(f"\ntotal stale rows removed: {total_removed}")


if __name__ == "__main__":
    main()
