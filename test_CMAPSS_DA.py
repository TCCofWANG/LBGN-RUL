import argparse
import glob
import os
import warnings

import pandas as pd
import yaml

from configs.hparams import get_configs
from configs.data_model_configs import update_namespace
from Experiment.Experiment import Exp

warnings.filterwarnings("ignore")


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v == 'True':
        return True
    if v == 'False':
        return False


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument('--logs_dir', default='./external_data/CMAPSS final with different seeds', type=str,
                         help='root to scan for {pair}/{model}/{tag}/best_checkpoint.pth')
    parser.add_argument('--models', nargs='+', default=None,
                         help='restrict to these model names (default: every model found)')
    parser.add_argument('--pairs', nargs='+', default=None,
                         help='restrict to these pair folder names, e.g. FD001_FD003 (default: every pair found)')

    parser.add_argument('--dataset_name', default='CMAPSS', type=str, help='[CMAPSS]')
    parser.add_argument('--data_path_CMAPSS', default='./CMAPSS', type=str,
                         help='data path for CMAPSS ./CMAPSS or D:/Datasets/CMAPSS')
    parser.add_argument('--MAXLIFE_CMAPSS', default=125, type=int, help='maxlife for cmapss')
    parser.add_argument('--normalization_CMAPSS', default='minmax', type=str, help='way for norm')
    parser.add_argument('--input_length', default=50, type=int, help='input_length')
    parser.add_argument('--validation', default=0.1, type=float, help='validation')
    parser.add_argument('--batch_size', default=128, type=int,
                         help='not architecture-critical -- test() aggregates across batches regardless')


    parser.add_argument('--learning_rate', default=0.001, type=float, help='lr')
    parser.add_argument('--lam_mar', default=0.05, type=float, help='')
    parser.add_argument('--lam_da', default=0.1, type=float, help='')
    parser.add_argument('--LCE_dim', default=16, type=int, help='')
    parser.add_argument('--n_bands', default=4, type=int, help='')
    parser.add_argument('--hop', default=2, type=int, help='')
    parser.add_argument('--lam_struct', default=0.1, type=float, help='')
    parser.add_argument('--lam_rmmd', default=0.1, type=float, help='')
    parser.add_argument('--dropout', default=0.1, type=float, help='')
    parser.add_argument('--lbgn_ablation', default='none', type=str, choices=['none', 'time_adj', 'uniform_band', 'static_band', 'no_lifecycle',
                                  'no_input_film', 'shared_film', 'no_film',
                                  'plain_mse', 'no_mar', 'no_mmd', 'no_da'])
    parser.add_argument('--ndc_ablate_rate1', action='store_true')

    parser.add_argument('--is_diff', default=False, type=str2bool)
    parser.add_argument('--is_minmax', default=True, type=str2bool)
    parser.add_argument('--add_noise', default=False, type=str2bool)
    parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer to use')
    parser.add_argument('--learning_rate_patience', default=10, type=int)
    parser.add_argument('--learning_rate_factor', default=0.3, type=float)
    parser.add_argument('--early_stop_patience', default=25, type=int)

    parser.add_argument('--loss_type', default='DA', type=str,
                         help='Base supervised criterion: [DA, MSE, MAE, QUAN, Huber]')
    parser.add_argument('--save_test', default=False, type=str2bool,
                         help='leave False so this pass never overwrites the original result.npz')


    parser.add_argument('--test_all_subset', default=False, type=str2bool)

    parser.add_argument('--out_csv', default=None, type=str,
                         help='where to write the verification log (default: '
                              '<logs_dir>/checkpoint_verification.csv)')

    return parser


def discover_checkpoints(logs_dir, models=None, pairs=None):

    pattern = os.path.join(logs_dir, "*", "*", "*", "best_checkpoint.pth")
    for ckpt_path in sorted(glob.glob(pattern)):
        tag_dir = os.path.dirname(ckpt_path)
        tag = os.path.basename(tag_dir)
        model = os.path.basename(os.path.dirname(tag_dir))
        pair = os.path.basename(os.path.dirname(os.path.dirname(tag_dir)))
        if models and model not in models:
            continue
        if pairs and pair not in pairs:
            continue
        hparam_path = os.path.join(tag_dir, "hparam.yaml")
        if not os.path.exists(hparam_path):
            print(f"  [skip] {ckpt_path} -- no sibling hparam.yaml, can't verify safely")
            continue
        yield pair, model, tag, ckpt_path, hparam_path


def original_metrics(logs_dir, pair, model, tag):

    csv_path = os.path.join(logs_dir, f"{pair}experimental_logs.csv")
    if not os.path.exists(csv_path):
        return None, None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None, None
    savepath = df["savepath"].fillna("").str.rstrip("/\\")
    matches = df[(df["model"] == model) & savepath.str.endswith(tag)]
    if matches.empty:
        return None, None
    row = matches.iloc[-1]
    return row.get("best_last_RMSE"), row.get("score")


_NON_ARCH_KEYS = {
    "seed", "info", "train", "resume", "resume_path", "save_path",
    "dataset_name", "model_name", "logs_root", "save_test",
    "Data_id_CMAPSS", "Data_id_CMAPSS_test", "Data_id_N_CMAPSS", "Data_id_N_CMAPSS_test",
    "warm_start_checkpoint", "warm_start_hparam_yaml",
    "learning_rate", "batch_size", "train_epochs",
    "da_mode",
}


RMSE_TOL = 0.01
SCORE_TOL = 0.5


def main():
    args = build_parser().parse_args()

    checkpoints = list(discover_checkpoints(args.logs_dir, args.models, args.pairs))
    print(f"Found {len(checkpoints)} checkpoint(s) under {args.logs_dir}")

    records = []
    n_ok, n_fail = 0, 0
    for pair, model, tag, ckpt_path, hparam_path in checkpoints:
        src, _, tgt = pair.partition("_")[0], None, pair.partition("_")[2] or pair.partition("_")[0]

        print()
        print("=" * 90)
        print(f"  {pair} | {model} | {tag}")
        print("=" * 90)

        run_args = argparse.Namespace(**vars(args))
        run_args.model_name = model
        run_args.Data_id_CMAPSS = src
        run_args.Data_id_CMAPSS_test = tgt
        run_args.save_path = tag
        run_args.resume = True
        run_args.train = False
        run_args.resume_path = "exp0"
        run_args.warm_start_checkpoint = ckpt_path
        run_args.warm_start_hparam_yaml = hparam_path
        run_args.info = f"checkpoint_check_{model}_{pair}_{tag}"


        run_args.logs_root = args.logs_dir

        try:
            hparams_class = get_configs(run_args.dataset_name, run_args.Data_id_CMAPSS)
            update_namespace(run_args, hparams_class.train_params[model])
            update_namespace(run_args, hparams_class.alg_hparams[model])
        except Exception as e:
            print(f"  (no current hparams.py entry for {model}: {e} -- using CLI/checkpoint values only)")

        record = {
            "pair": pair, "model": model, "tag": tag, "ckpt_path": ckpt_path,
            "orig_rmse": None, "orig_score": None,
            "fresh_rmse": None, "fresh_score": None,
            "rmse_diff": None, "score_diff": None,
            "status": None, "error": "",
        }


        try:
            with open(hparam_path, "r", encoding="utf-8") as f:
                warm_start_hparams = yaml.load(f.read(), Loader=yaml.FullLoader) or {}
            warm_start_hparams = {k: v for k, v in warm_start_hparams.items() if k not in _NON_ARCH_KEYS}
            update_namespace(run_args, warm_start_hparams)

            orig_rmse, orig_score = original_metrics(args.logs_dir, pair, model, tag)
            record["orig_rmse"], record["orig_score"] = orig_rmse, orig_score
            if orig_rmse is not None:
                print(f"  originally recorded : RMSE={orig_rmse:.4f}  score={orig_score:.4f}")
            else:
                print(f"  originally recorded : (no matching row in {pair}experimental_logs.csv)")
            print(f"  re-running now      : (see RMSE/score below)")

            exp = Exp(run_args)
            fresh_rmse, fresh_score = exp.start()
            record["fresh_rmse"], record["fresh_score"] = fresh_rmse, fresh_score

            if orig_rmse is not None:
                rmse_diff = abs(fresh_rmse - orig_rmse)
                score_diff = abs(fresh_score - orig_score)
                record["rmse_diff"], record["score_diff"] = rmse_diff, score_diff
                record["status"] = ("same" if rmse_diff <= RMSE_TOL and score_diff <= SCORE_TOL
                                     else "different")
            else:
                record["status"] = "no_original_record"

            n_ok += 1
        except Exception as e:
            record["status"] = "error"
            record["error"] = str(e)
            print(f"  [ERROR] {model} | {pair} | {tag}: {e}")
            n_fail += 1

        records.append(record)

    out_csv = args.out_csv or os.path.join(args.logs_dir, "checkpoint_verification.csv")
    pd.DataFrame(records).to_csv(out_csv, index=False)

    print()
    print("=" * 90)
    print(f"  Done. {n_ok} checkpoint(s) re-ran successfully, {n_fail} errored out.")
    n_same = sum(1 for r in records if r["status"] == "same")
    n_diff = sum(1 for r in records if r["status"] == "different")
    n_norec = sum(1 for r in records if r["status"] == "no_original_record")
    print(f"  {n_same} same (within RMSE<={RMSE_TOL}/score<={SCORE_TOL}), "
          f"{n_diff} different, {n_norec} with no original record to compare, "
          f"{n_fail} errored.")
    print(f"  Full per-checkpoint log written to: {out_csv}")
    print("=" * 90)


if __name__ == "__main__":
    main()
