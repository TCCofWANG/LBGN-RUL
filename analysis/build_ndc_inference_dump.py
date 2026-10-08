
import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import MinMaxScaler

from Models_RUL.NDC_PDMN import NDC_PDMN


def load_test_with_regime(data_path, source, target, sequence_length, MAXLIFE, epsilon=1e-02):

    column_name = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3',
                   's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
                   's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19',
                   's20', 's21']

    train_FD = pd.read_table(f"{data_path}/train_{source}.txt", header=None, delim_whitespace=True)
    train_FD.columns = column_name
    test_FD = pd.read_table(f"{data_path}/test_{target}.txt", header=None, delim_whitespace=True)
    test_FD.columns = column_name
    RUL_FD = pd.read_table(f"{data_path}/RUL_{target}.txt", header=None, delim_whitespace=True)


    total_cycles = []
    for _id in set(train_FD['engine_id']):
        cyc = train_FD[train_FD['engine_id'] == _id]['cycle']
        total_cycles.append(cyc.max())
    avg_total_cycle_train = sum(total_cycles) / len(total_cycles)


    rul = []
    for _id_test in set(test_FD['engine_id']):
        true_rul = int(RUL_FD.iloc[_id_test - 1])
        block = test_FD[test_FD['engine_id'] == _id_test]
        cycle_list = block['cycle'].tolist()
        max_cycle = max(cycle_list) + true_rul
        knee_point = max_cycle - MAXLIFE
        kink_RUL = []
        for i in range(len(cycle_list)):
            kink_RUL.append(MAXLIFE if i < knee_point else max_cycle - i - 1)
        rul.extend(kink_RUL)
    test_FD["RUL"] = rul


    train_FD["RUL"] = 0.0

    col_to_drop = ['setting2', 'setting3', 's1', 's5', 's6', 's10', 's16', 's18', 's19']
    train_FD = train_FD.drop(col_to_drop, axis=1)
    test_FD = test_FD.drop(col_to_drop, axis=1)

    train_FD['setting1'] = train_FD['setting1'].round(1)
    test_FD['setting1'] = test_FD['setting1'].round(1)


    test_regime = test_FD['setting1'].copy()

    grouped_train = train_FD.groupby('setting1')
    grouped_test = test_FD.groupby('setting1')

    scaler = MinMaxScaler()
    test_normalized = pd.DataFrame(columns=test_FD.columns[3:])
    for train_idx, train in grouped_train:
        scaled_train = scaler.fit_transform(train.iloc[:, 3:])
        for test_idx, test in grouped_test:
            if train_idx == test_idx:
                scaled_test = scaler.transform(test.iloc[:, 3:])
                test_normalized = pd.concat([test_normalized, pd.DataFrame(
                    data=scaled_test, index=test.index, columns=test_FD.columns[3:])])
    test_normalized = test_normalized.sort_index()
    test_FD.iloc[:, 3:-1] = test_normalized.iloc[:, :]
    test_FD = test_FD.drop('setting1', axis=1)


    x_batch, occ_batch, y_batch = [], [], []
    engine_ids, regimes, final_cycles = [], [], []
    for _id in sorted(set(test_FD['engine_id'])):
        block = test_FD[test_FD['engine_id'] == _id]
        regime_block = test_regime.loc[block.index]
        if block.shape[0] >= sequence_length:
            window = block.iloc[-sequence_length:]
            regime_window = regime_block.iloc[-sequence_length:]
        else:
            num_pad = sequence_length - block.shape[0]
            window = block
            regime_window = regime_block
            for _ in range(num_pad):
                window = pd.concat([window.head(1), window], axis=0)
                regime_window = pd.concat([regime_window.head(1), regime_window], axis=0)

        x_batch.append(window.iloc[:, 2:-1].values)
        occ_batch.append((window.iloc[:, 1:2].values / avg_total_cycle_train) * epsilon)
        y_batch.append(window.iloc[-1:, -1].values)
        engine_ids.append(_id)


        regimes.append(regime_window.mode().iloc[0])
        final_cycles.append(window['cycle'].iloc[-1])

    return (np.array(x_batch, dtype=np.float32),
            np.array(occ_batch, dtype=np.float32),
            np.array(y_batch, dtype=np.float32),
            engine_ids, regimes, final_cycles)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_root", default="./logs")
    ap.add_argument("--data_path", default="./CMAPSS")
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--seed", required=True)
    ap.add_argument("--tag", default=None, help="checkpoint dir name; defaults to final_s{seed}")
    ap.add_argument("--out", default="analysis/out/ndc_rate_dump.csv")
    args = ap.parse_args()

    tag = args.tag or f"final_s{args.seed}"
    ckpt_dir = os.path.join(args.logs_root, f"{args.source}_{args.target}", "NDC_PDMN", tag)
    with open(os.path.join(ckpt_dir, "hparam.yaml")) as f:
        hp = yaml.safe_load(f)

    class Args:
        pass
    margs = Args()
    for k, v in hp.items():
        setattr(margs, k, v)
    margs.ndc_ablate_rate1 = False

    model = NDC_PDMN(margs)
    state_dict = torch.load(os.path.join(ckpt_dir, "best_checkpoint.pth"), map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    x, occ, y, engine_ids, regimes, final_cycles = load_test_with_regime(
        args.data_path, args.source, args.target,
        sequence_length=hp["input_length"], MAXLIFE=hp["MAXLIFE_CMAPSS"])

    x_t = torch.from_numpy(x)
    occ_t = torch.from_numpy(occ)
    with torch.no_grad():
        _, out = model(x_t, occ_t)
    rate = out["rate"].squeeze(-1).numpy()
    tau = out["tau"].squeeze(-1).numpy()
    pred_rul = out["gamma"].numpy()

    rows = []
    for i, eng_id in enumerate(engine_ids):
        rows.append({
            "source": args.source, "target": args.target, "seed": args.seed,
            "engine_id": eng_id, "regime": regimes[i], "final_cycle": final_cycles[i],
            "final_OCC": occ[i, -1, 0], "true_RUL": y[i, 0],
            "pred_RUL": pred_rul[i, 0],
            "rate_mean": rate[i].mean(), "rate_std": rate[i].std(),
            "rate_final": rate[i, -1], "tau_final": tau[i, -1],
        })
    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_header = not os.path.exists(args.out)
    df.to_csv(args.out, mode="a", header=write_header, index=False)
    print(f"appended {len(df)} rows -> {args.out}")
    print(df[["regime", "rate_mean", "rate_final"]].groupby("regime").agg(["mean", "std", "count"]))


if __name__ == "__main__":
    main()
