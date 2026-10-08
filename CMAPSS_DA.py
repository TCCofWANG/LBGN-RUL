import warnings
import argparse
from configs.hparams import get_configs
from configs.data_model_configs import update_namespace
from Experiment.Experiment import Exp
warnings.filterwarnings ( "ignore" )

def str2bool(v) :
    if isinstance ( v , bool ) :
        return v
    if v == 'True' :
        return True
    if v == 'False' :
        return False

def add_cmapss_args(parser) :
    parser.add_argument ( '--data_path_CMAPSS' , default = "./CMAPSS" , type = str ,help = 'data path for CMAPSS ./CMAPSS or D:/Datasets/CMAPSS' )
    parser.add_argument ( '--Data_id_CMAPSS' , default = "FD001" , type = str , help = 'CMAPSS subset for train' )
    parser.add_argument ( '--Data_id_CMAPSS_test' , default = "FD003" , type = str , help = 'CMAPSS for test' )
    parser.add_argument ( '--MAXLIFE_CMAPSS' , default = 125 , type = int , help = 'maxlife for cmapss' )
    parser.add_argument ( '--normalization_CMAPSS' , default = "minmax" , type = str , help = 'way for norm' )
    return parser


def add_ncmapss_args(parser) :
    parser.add_argument ( '--Data_id_N_CMAPSS' , default = "DS01" , type = str , help = 'for N_CMAPSS (source, when doing DA)' )
    parser.add_argument ( '--Data_id_N_CMAPSS_test' , default = None , type = str ,
                          help = 'N_CMAPSS DA target (e.g. DS02); if unset/equal to Data_id_N_CMAPSS, '
                                 'training is single-dataset (non-DA), matching the original behaviour' )
    parser.add_argument ( '--s' , type = int , default = 10 , help = 'stride of window' )
    parser.add_argument ( '--sampling' , type = int , default = 10 , help = 'sub sampling of the given data. If it is '
                                                                            '10, then this indicates that we assumes '
                                                                            '0.1Hz of data collection' )
    parser.add_argument ( '--change_len' , type = str2bool , default = True , help = 're-generate data when you change input_len' )
    parser.add_argument ( '--rate' , type = float , default = 0.8 , help = 'max_life related' )
    return parser


def add_noise_realted_args(parser) :
    parser.add_argument ( '--noise_level' , default = 0.05 , type = int , help = 'noise level or threshold' )
    parser.add_argument ( '--noise_level_train' , default = 0.05 , type = int , help = 'noise level or threshold' )
    parser.add_argument ( '--noise_level_test' , default = 0.1 , type = int , help = 'noise level or threshold' )


if __name__ == '__main__' :

    parser = argparse.ArgumentParser ( description = __doc__ )
    parser.add_argument ( '--model_name' , default = 'LBGN_RUL' , type = str ,
                          help = '[LBGN_RUL ]' )

    parser.add_argument ( '--info' , default = 'CMAPSS cross domain test' , type = str , help = 'extra information' )
    parser.add_argument ( '--train' , default = True , type = str2bool , help = 'Train or test' )
    parser.add_argument ( '--resume' , default = False , type = str2bool , help = 'load checkpoint or not' )
    parser.add_argument ( '--resume_path' , default = 'final_s42' , help = 'if resume is True, it will be useful' )
    parser.add_argument ( '--save_path' , default = None ,
                          help = 'If resume is ture the save_path should be None, it will be add 1 save model per training; otherwise use exp0 wiil be useful' )
    parser.add_argument ( '--warm_start_checkpoint' , default = None , type = str ,
                          help = 'if set (with --resume True), load weights from this exact '
                                 'best_checkpoint.pth path instead of the usual logs_root/pair/model/'
                                 'resume_path location -- e.g. a ./logs_best/{pair}/{model}/exp0/'
                                 'best_checkpoint.pth checkpoint, to warm-start further training '
                                 'with a new --seed rather than from scratch' )
    parser.add_argument ( '--warm_start_hparam_yaml' , default = None , type = str ,
                          help = 'path to the hparam.yaml sibling of --warm_start_checkpoint; its '
                                 'architecture keys (hidden_dim, K, ndc_hidden, ...) are applied '
                                 'BEFORE model construction so shapes always match the checkpoint '
                                 'being loaded, even if current configs/hparams.py has since '
                                 'changed (e.g. warm-starting from a grid-search checkpoint that '
                                 'used a different K/ndc_hidden than the current tuned value)' )
    parser.add_argument ( '--save_test' , default = True , type = str2bool ,
                          help = 'Save true and prededition for later usage like fiugre maping ' )
    parser.add_argument ( '--loss_type' , default = 'DA' , type = str , help = 'Base supervised criterion: [DA, MSE, MAE, QUAN, Huber]' )
    parser.add_argument ( '--logs_root' , default = None , type = str ,
                          help = 'redirect output away from ./logs/ (e.g. ./lr_logs) so exploratory '
                                 'sweeps never collide with the final campaign data' )
    parser.add_argument ( '--inference_time_only' , default = False , type = str2bool ,
                          help = 'with --train False --resume True: load the checkpoint, measure '
                                 'pure model-forward latency over the test set '
                                 '(Experiment.measure_pure_inference_time), print an '
                                 'INFERENCE_TIME_RESULT line, and return -- no RMSE/score, '
                                 'result.npz, or CSV row. See scripts/measure_inference_time_cmapss.py.' )
    parser.add_argument ( '--inference_time_runs' , default = 100 , type = int ,
                          help = 'timed repeats of a full test-set forward pass, used only with '
                                 '--inference_time_only True' )
    parser.add_argument ( '--inference_time_warmup' , default = 10 , type = int ,
                          help = 'untimed warmup passes before timing, used only with '
                                 '--inference_time_only True' )
    parser.add_argument ( '--flops_only' , default = False , type = str2bool ,
                          help = 'with --train False --resume True: load the checkpoint, count '
                                 'FLOPs/MACs on one batch-size-1 test sample '
                                 '(analysis/flop_utils.py -- thop for standard layers plus a '
                                 'monkey-patched einsum/matmul/fft count for the custom graph/'
                                 'spectral ops thop cannot see), print a FLOPS_RESULT line, and '
                                 'return -- no RMSE/score, result.npz, or CSV row. See '
                                 'scripts/measure_flops_cmapss.py.' )


    parser.add_argument ( '--dataset_name' , default = 'CMAPSS' , type = str , help = '[CMAPSS,N_CMAPSS]' )
    parser.add_argument ( '--input_length' , default = 50 , type = int , help = 'input_lenth' )
    parser.add_argument ( '--validation' , default = 0.1 , type = float , help = 'validation' )
    parser.add_argument ( '--seed' , type = int , default = None , help = "fix random seed; omit for a fresh random seed (see Experiment.set_seed)" )

    parser.add_argument ( '--learning_rate' , default = 0.001 , type = float , help = 'lr which is need for bash run and update from config' )

    parser.add_argument ( '--LCE_dim' , default = 16 , type = int , help = '' )
    parser.add_argument ( '--n_bands' , default = 4 , type = int , help = '' )
    parser.add_argument ( '--hop' , default = 2 , type = int , help = 'K-hop depth for LBGN_RUL/PDMN graph conv (was a dead --hope typo; getattr(args,"hop",2) never matched it)' )
    parser.add_argument ( '--lam_struct' , default = 1.25 , type = float , help = '' )
    parser.add_argument ( '--lam_rmmd' , default = 0.25 , type = float , help = '' )
    parser.add_argument ( '--lam_mar' , default = 5 , type = float , help = '' )
    parser.add_argument ( '--lam_da' , default = 3 , type = float , help = '' )
    parser.add_argument ( '--dropout' , default = 0.1 , type = float , help = '' )


    parser.add_argument ( '--add_noise' , default = False , type = str2bool , help = 'add noise to data' )

    parser.add_argument ( '--is_diff' , default = False , type = str2bool )
    parser.add_argument ( '--is_minmax' , default = True , type = str2bool )


    parser.add_argument ( '--optimizer' , default = 'Adam' , type = str , help = 'optimizer to use' )
    parser.add_argument ( '--learning_rate_patience' , default = 10 , type = int ,
                          help = 'number of epochs with no improvement before reducing learning rate' )
    parser.add_argument ( '--learning_rate_factor' , default = 0.3 , type = float ,
                          help = 'factor to multiply learning rate by' )
    parser.add_argument ( '--early_stop_patience' , default = 25 , type = int ,
                          help = 'number of epochs with no improvement before stopping' )


    parser.add_argument ( '--hidden_dim_override' , default = None , type = int ,
                          help = 'if set, overrides alg_hparams hidden_dim after config load (capacity ablation)' )
    parser.add_argument ( '--learning_rate_override' , default = None , type = float ,
                          help = 'if set, overrides train_params learning_rate after config load '
                                 '(per-pair fix for baselines where the source-level majority-vote '
                                 'LR is a poor fit for one specific target)' )


    parser.add_argument ( '--ndc_ablate_rate1' , action = 'store_true' ,
                          help = 'NDC_PDMN only: freeze damage-clock rate at 1 (tau=OCC) for the r≡1 ablation' )


    parser.add_argument ( '--lbgn_ablation' , default = 'none' , type = str ,
                          choices = [ 'none' , 'time_adj' , 'uniform_band' , 'static_band' ,
                                      'no_lifecycle' ,
                                      'no_input_film' , 'shared_film' , 'no_film' ,
                                      'no_spectral_view' , 'no_spectral_film' , 'no_spectral_diffusion' ,
                                      'uniform_band_no_spectral_diffusion' ,
                                      'plain_mse' , 'inverted_occ_weight' , 'no_mar' , 'no_mmd' , 'no_da' ,
                                      'source_only' ] ,
                          help = 'LBGN_RUL only: ablation variant (default none = full model)' )

    temp_args , _ = parser.parse_known_args ( )
    if temp_args.dataset_name == 'CMAPSS' :
        parser = add_cmapss_args ( parser )

    elif temp_args.dataset_name == 'N_CMAPSS' :
        parser = add_ncmapss_args ( parser )
    else :
        raise ValueError ( 'without corresponding dataset' )

    if temp_args.add_noise :
        parser = add_noise_realted_args ( parser )

    args = parser.parse_args ( )
    args.test_all_subset= False


    hparams_class = get_configs ( args.dataset_name , args.Data_id_CMAPSS if args.dataset_name == "CMAPSS" else args.Data_id_N_CMAPSS )
    hparams_class = get_configs ( args.dataset_name , args.Data_id_CMAPSS if args.dataset_name == "CMAPSS" else args.Data_id_N_CMAPSS )
    try :

        OVERRIDABLE = ['seed', 'learning_rate', 'lam_mar', 'lam_da', 'loss_type', 'lbgn_ablation']


        aux = argparse.ArgumentParser ( argument_default = argparse.SUPPRESS ,
                                        add_help = False , allow_abbrev = False )
        for _k in OVERRIDABLE :
            aux.add_argument ( '--' + _k )
        _explicit , _ = aux.parse_known_args ( )
        cli_supplied = {k: getattr ( args , k ) for k in vars ( _explicit )}

        train_configs = hparams_class.train_params[args.model_name]
        model_configs = hparams_class.alg_hparams[args.model_name]
        update_namespace ( args , train_configs )
        update_namespace ( args , model_configs )


        for k, v in cli_supplied.items():
            setattr(args, k, v)
    except Exception as X :
        print ( X )

    if args.hidden_dim_override is not None :
        args.hidden_dim = args.hidden_dim_override
        print ( f"[capacity ablation] hidden_dim overridden to {args.hidden_dim}" )

    if args.learning_rate_override is not None :
        args.learning_rate = args.learning_rate_override
        print ( f"[per-pair fix] learning_rate overridden to {args.learning_rate}" )


    if getattr ( args , 'warm_start_hparam_yaml' , None ) :
        import yaml as _yaml
        with open ( args.warm_start_hparam_yaml , 'r' , encoding = 'utf-8' ) as f :
            warm_start_hparams = _yaml.load ( f.read () , Loader = _yaml.FullLoader ) or {}


        _non_arch_keys = {
            'seed', 'info', 'train', 'resume', 'resume_path', 'save_path',
            'dataset_name', 'model_name', 'logs_root',
            'Data_id_CMAPSS', 'Data_id_CMAPSS_test', 'Data_id_N_CMAPSS', 'Data_id_N_CMAPSS_test',
            'warm_start_checkpoint', 'warm_start_hparam_yaml',
            'learning_rate', 'batch_size', 'train_epochs',
        }
        warm_start_hparams = {k : v for k, v in warm_start_hparams.items () if k not in _non_arch_keys}
        update_namespace ( args , warm_start_hparams )
        print ( f"[warm start] architecture overridden from {args.warm_start_hparam_yaml}: "
                f"{list ( warm_start_hparams.keys () )}" )

    exp = Exp ( args )
    exp.start ( )
