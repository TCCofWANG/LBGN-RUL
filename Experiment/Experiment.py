import math
import random
import sys
sys.path.append ( ".." )
import os
import numpy as np
import shutil
import torch
from torch import optim
import torch.nn as nn
from time import time
import time as time_module
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import yaml
from N_CMAPSS_Related.N_CMAPSS_load_data import get_n_cmapss_data_, get_n_cmapss_data_da_, N_CMAPSSData
from CMAPSS_Related.load_data_CMAPSS import get_cmapss_data_
from CMAPSS_Related.CMAPSS_Dataset import CMAPSSData

from torch.utils.data import DataLoader

from Models_RUL import *

from Experiment.Early_Stopping import EarlyStopping
from Experiment.learining_rate_adjust import adjust_learning_rate_class
from Experiment.HTS_Loss_Function import Weighted_MSE_Loss, MSE_Smoothness_Loss

from tool.Write_csv import *
import datetime
import torch.nn.functional as F

from tqdm import tqdm

"""
This file only used for CMAPSS Datase
"""


def set_seed ( seed=None ) :

    if seed is None :
        seed = int ( time_module.time_ns () ^ ( os.getpid () << 16 ) ) & 0xFFFFFFFF
    print ( f"[Experiment] Using Random Seed: {seed}" )

    random.seed ( seed )
    np.random.seed ( seed )
    torch.manual_seed ( seed )
    torch.cuda.manual_seed_all ( seed )


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    g = torch.Generator ()
    g.manual_seed ( seed )
    return seed , g


class Exp ( object ) :
    def __init__(self, args) :
        self.args = args


        seed , self.seed_generator = set_seed ( getattr ( args , 'seed' , None ) )
        args.seed = seed

        self.device = self._acquire_device ()

        self._get_path()


        self.train_data, self.train_loader, self.vali_data, self.vali_loader, self.test_data, self.test_loader, self.input_feature = self._get_data ()


        self.model = self._get_model


        self.optimizer_dict = {"Adam" : optim.Adam, "AdamW" : optim.AdamW}
        self.criterion_dict = {"MSE" : nn.MSELoss, "CrossEntropy" : nn.CrossEntropyLoss,
                               "WeightMSE" : Weighted_MSE_Loss, "smooth_mse" : MSE_Smoothness_Loss}


    def _acquire_device(self) :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            print(f'Using GPU: {torch.cuda.get_device_name(0)}')
            print(f'CUDA_VISIBLE_DEVICES = {os.environ.get("CUDA_VISIBLE_DEVICES", "Not set (using default GPU 0)")}')
        else :
            print ( 'Use CPU' )
        return device


    @property
    def _get_model(self) :
        if self.args.model_name == 'DAGCN_RUL':
            model = DAGCN_RUL ( self.args )

        elif self.args.model_name == 'EviAdaptRUL':
            model = EviAdaptRUL ( self.args )

        elif self.args.model_name == 'DAST_RUL':
            model = DAST_RUL ( self.args )

        elif self.args.model_name == 'CADA_RUL':
            model = CADA_RUL ( self.args )

        elif self.args.model_name == 'TACDA_RUL':
            model = TACDA_RUL ( self.args )

        elif self.args.model_name == 'LBGN_RUL':


            _arch_ablations = (
                'time_adj' , 'uniform_band' , 'static_band' , 'no_lifecycle' ,
                'no_input_film' , 'shared_film' , 'no_film' ,
                'no_spectral_view' , 'no_spectral_film' , 'no_spectral_diffusion' ,
                'uniform_band_no_spectral_diffusion' ,
            )
            if getattr ( self.args , 'lbgn_ablation' , 'none' ) in _arch_ablations :
                model = LBGN_RUL_ablation ( self.args )
            else :
                model = LBGN_RUL ( self.args )

        elif self.args.model_name == 'OCS_DANN':
            model = OCS_DANN ( self.args )

        elif self.args.model_name == 'MDAN_RUL':
            model = MDAN_RUL ( self.args )

        elif self.args.model_name == 'CCDG_RUL':
            model = CCDG_RUL ( self.args )

        self.trainable_parameters= np.sum ( [para.numel () for para in model.parameters ()] )
        print ( "Parameter :",  self.trainable_parameters)

        return model.to ( self.device )


    def _select_optimizer(self) :
        if self.args.optimizer not in self.optimizer_dict.keys () :
            raise NotImplementedError

        model_optim = self.optimizer_dict[self.args.optimizer] ( self.model.parameters (),
                                                                 lr = self.args.learning_rate )
        self.grad_clip = 1.0
        return model_optim


    def _select_criterion(self) :
        if self.args.criterion not in self.criterion_dict.keys () :
            raise NotImplementedError

        criterion = self.criterion_dict[self.args.criterion] ()
        return criterion


    @staticmethod
    def _resolve_da_mode(args) :

        explicit = getattr ( args , 'da_mode' , None )
        if explicit :
            return explicit
        _legacy = { 'DA' : 'da_loop' , 'EviAdapt' : 'eviadapt' , 'Adversarial' : 'adversarial' }
        return _legacy.get ( args.loss_type )

    def _get_data(self) :
        args = self.args


        da_mode = getattr ( self.args , 'da_mode' , None )
        if self.args.dataset_name == 'CMAPSS' and da_mode in ('da_loop', 'routing_mmd', 'eviadapt', 'adversarial', 'internal', 'dast', 'cada', 'tacda', 'lbgn', 'ocs_dann', 'mdan', 'ccdg') :
            return self._get_data_da ()


        if self.args.dataset_name == 'N_CMAPSS' and da_mode in ('da_loop', 'routing_mmd', 'eviadapt', 'adversarial', 'internal', 'dast', 'cada', 'tacda', 'lbgn', 'ocs_dann', 'mdan', 'ccdg') and getattr ( args , 'Data_id_N_CMAPSS_test' , None ) :
            return self._get_data_da_ncmapss ()


        if self.args.dataset_name == 'CMAPSS' :


            X_train, y_train, index_train, X_vali, y_vali, index_vali, X_test, y_test, index_test, self.global_input = get_cmapss_data_ (
                data_path = args.data_path_CMAPSS, Data_id = args.Data_id_CMAPSS, test_Data_id =args.Data_id_CMAPSS_test, sequence_length = args.input_length,
                MAXLIFE = args.MAXLIFE_CMAPSS, is_difference = args.is_diff, validation = args.validation )
            self.max_life = args.MAXLIFE_CMAPSS

        elif self.args.dataset_name == 'N_CMAPSS' :


            tgt = getattr ( args, 'Data_id_N_CMAPSS_test', None ) or args.Data_id_N_CMAPSS
            ( X_train, index_train, y_train, X_vali, index_vali, y_vali,
              _, _, _, _, _, _,
              X_test, index_test, y_test, self.max_life ) = get_n_cmapss_data_da_ (
                args = args, source = args.Data_id_N_CMAPSS, target = tgt )

        else :
            raise ValueError ( 'without corresponding dataset' )

        train_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_train, index_train, y_train )
        vali_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_vali, index_vali, y_vali )
        test_data_set = eval ( self.args.dataset_name + 'Data' ) ( X_test, index_test, y_test )

        input_fea = X_test.shape[-1]

        train_data_loader = DataLoader ( dataset = train_data_set, batch_size = args.batch_size, shuffle = True, num_workers = 0, drop_last = True, generator = self.seed_generator )
        vali_data_loader = DataLoader ( dataset = vali_data_set, batch_size = args.batch_size, shuffle = False, num_workers = 0, drop_last = True )
        test_data_loader = DataLoader ( dataset = test_data_set, batch_size = args.batch_size, shuffle = False, num_workers = 0, drop_last = False )

        return train_data_set, train_data_loader, vali_data_set, vali_data_loader, test_data_set, test_data_loader, input_fea

    def _get_data_da(self) :

        args = self.args
        kw = dict ( sequence_length=args.input_length , MAXLIFE=args.MAXLIFE_CMAPSS ,
                    is_difference=args.is_diff , validation=args.validation )


        X_s, y_s, idx_s, X_sv, y_sv, idx_sv, _, _, _, self.global_input = get_cmapss_data_ (
            data_path=args.data_path_CMAPSS ,
            Data_id=args.Data_id_CMAPSS ,
            test_Data_id=args.Data_id_CMAPSS , **kw )
        print("the information about of source subset:", args.Data_id_CMAPSS)
        print("the shape of X_train of is:", X_s.shape)
        print("the shape of y_train is:", y_s.shape)
        print("the shape of idx_train is:", idx_s.shape)

        X_t, y_t, idx_t, X_tv, y_tv, idx_tv, X_test, y_test, idx_test, _ = get_cmapss_data_ (
            data_path=args.data_path_CMAPSS ,
            Data_id=args.Data_id_CMAPSS_test ,
            test_Data_id=args.Data_id_CMAPSS_test , **kw )
        print("the information about of target subset:", args.Data_id_CMAPSS_test)
        print("the shape of X_test of is:", X_test.shape)
        print("the shape of y_test is:", y_test.shape)
        print("the shape of idx_test is:", idx_test.shape)
        self.max_life = args.MAXLIFE_CMAPSS

        src_set  = CMAPSSData ( X_s    , idx_s    , y_s    )


        if args.Data_id_CMAPSS == args.Data_id_CMAPSS_test :
            tgt_set = CMAPSSData ( X_test , idx_test , y_test )
        else :
            tgt_set = CMAPSSData ( X_t    , idx_t    , y_t   )
        vali_set = CMAPSSData ( X_tv , idx_tv , y_tv )
        test_set = CMAPSSData ( X_test , idx_test , y_test )

        bs = args.batch_size

        same_domain = args.Data_id_CMAPSS == args.Data_id_CMAPSS_test
        self.source_loader = DataLoader ( src_set  , batch_size=bs , shuffle=True  , num_workers=0 , drop_last=True , generator=self.seed_generator )
        self.target_train_loader = DataLoader ( tgt_set  , batch_size=bs , shuffle=True  , num_workers=0 , drop_last=not same_domain , generator=self.seed_generator )
        vali_loader = DataLoader ( vali_set , batch_size=bs , shuffle=False , num_workers=0 , drop_last=True )
        test_loader = DataLoader ( test_set , batch_size=bs , shuffle=False , num_workers=0 , drop_last=False )

        input_fea = X_test.shape[-1]

        return src_set , self.source_loader , vali_set , vali_loader , test_set , test_loader , input_fea

    def _get_data_da_ncmapss(self) :

        args = self.args
        src = args.Data_id_N_CMAPSS
        tgt = args.Data_id_N_CMAPSS_test

        ( X_s , idx_s , y_s , X_sv , idx_sv , y_sv ,
          X_t , idx_t , y_t , X_tv , idx_tv , y_tv ,
          X_test , idx_test , y_test , self.max_life ) = get_n_cmapss_data_da_ (
            args=args , source=src , target=tgt )

        src_set = N_CMAPSSData ( X_s , idx_s , y_s )


        same_domain = src == tgt
        if same_domain :
            tgt_set = N_CMAPSSData ( X_test , idx_test , y_test )
        else :
            tgt_set = N_CMAPSSData ( X_t , idx_t , y_t )
        vali_set = N_CMAPSSData ( X_tv , idx_tv , y_tv )
        test_set = N_CMAPSSData ( X_test , idx_test , y_test )

        bs = args.batch_size
        self.source_loader = DataLoader ( src_set , batch_size=bs , shuffle=True , num_workers=0 , drop_last=True , generator=self.seed_generator )
        self.target_train_loader = DataLoader ( tgt_set , batch_size=bs , shuffle=True , num_workers=0 , drop_last=not same_domain , generator=self.seed_generator )
        vali_loader = DataLoader ( vali_set , batch_size=bs , shuffle=False , num_workers=0 , drop_last=True )
        test_loader = DataLoader ( test_set , batch_size=bs , shuffle=False , num_workers=0 , drop_last=False )

        input_fea = X_test.shape[-1]
        return src_set , self.source_loader , vali_set , vali_loader , test_set , test_loader , input_fea


    def save_hparam(self) :

        value2save = {k : v for k, v in vars ( self.args ).items () if
                      not k.startswith ( '__' ) and not k.endswith ( '__' )}

        del_key = ['train', 'resume', 'save_path', 'resume_path', 'batch_size', 'train_epochs', 'learning_rate']
        for key in del_key :
            del value2save[key]

        with open ( os.path.join ( self.save_path, 'hparam.yaml' ), 'a+' ) as f :
            f.write ( yaml.dump ( value2save ) )

    def _get_path(self) :


        logs_root = getattr ( self.args , 'logs_root' , None ) or './logs'
        if not os.path.exists ( logs_root ) :
            os.makedirs ( logs_root )

        exp_id = self.args.save_path

        if self.args.dataset_name == 'CMAPSS' :
            src = self.args.Data_id_CMAPSS
            tgt = self.args.Data_id_CMAPSS_test
            data = f'{src}_{tgt}'
        elif self.args.dataset_name == 'N_CMAPSS' :
            tgt = getattr ( self.args , 'Data_id_N_CMAPSS_test' , None )


            data = f'{self.args.Data_id_N_CMAPSS}_{tgt}' if tgt else self.args.Data_id_N_CMAPSS
        else :
            data = self.args.dataset_name
        self.path = logs_root + '/' + data
        if not os.path.exists ( self.path ) :
            os.makedirs ( self.path )

        self.model_path = self.path + '/' + self.args.model_name
        if not os.path.exists ( self.model_path ) :
            os.makedirs ( self.model_path )

        if exp_id is not None and exp_id != 'None' and exp_id != 'none' :
            self.save_path = self.model_path + '/' + exp_id
            if self.args.train :
                if os.path.exists ( self.save_path ) :
                    shutil.rmtree ( self.save_path )
                os.makedirs ( self.save_path )

        else :

            path_list = os.listdir ( self.model_path )
            if path_list == [] :
                self.save_path = self.model_path + '/exp0'

            else :
                path_list = [int ( idx[3 :] ) for idx in path_list]
                self.save_path = self.model_path + '/exp' + str ( max ( path_list ) + 1 )

            os.makedirs ( self.save_path )

    def _load_checkpoint(self) :


        warm_start = getattr ( self.args , 'warm_start_checkpoint' , None )
        if warm_start :
            self.checkpoint_dir = warm_start
            yaml_dir = os.path.join ( os.path.dirname ( warm_start ) , 'hparam.yaml' )
        else :
            self.checkpoint_dir = self.model_path + '/' + self.args.resume_path + '/best_checkpoint.pth'
            yaml_dir = self.model_path + '/' + self.args.resume_path + '/hparam.yaml'

        if os.path.exists ( self.checkpoint_dir ) :
            check_point = torch.load ( self.checkpoint_dir , map_location=self.device )
            self.model.load_state_dict ( check_point )
        else :
            raise FileNotFoundError ( f"checkpoint not found: {self.checkpoint_dir}" )


        _NON_ARCH_KEYS = {
            "seed", "info", "train", "resume", "resume_path", "save_path",
            "dataset_name", "model_name", "logs_root", "save_test",
            "Data_id_CMAPSS", "Data_id_CMAPSS_test", "Data_id_N_CMAPSS", "Data_id_N_CMAPSS_test",
            "warm_start_checkpoint", "warm_start_hparam_yaml",
            "learning_rate", "batch_size", "train_epochs",
            "da_mode",
        }

        with open ( yaml_dir, 'r', encoding = 'utf-8' ) as f :
            yml = yaml.load ( f.read (), Loader = yaml.FullLoader )
            for k in yml.keys () :
                if k in _NON_ARCH_KEYS :
                    continue
                exec ( 'self.args.' + k + '=yml[k]' )


    def _setup_eviadapt(self) :

        import copy

        self.source_model = copy.deepcopy ( self.model )
        self.source_model.eval ()
        for p in self.source_model.parameters () :
            p.requires_grad_ ( False )


        if hasattr ( self.model , 'time_view' ) and hasattr ( self.model , 'spectral_view' ) :
            encoder_params = (list ( self.model.time_view.parameters () ) +
                              list ( self.model.spectral_view.parameters () ) +
                              list ( self.model.lifecycle_adjacency.parameters () ))
            for p in self.model.parameters () :
                p.requires_grad_ ( False )
            for p in encoder_params :
                p.requires_grad_ ( True )
        elif hasattr ( self.model , 'cnn' ) :
            encoder_params = list ( self.model.cnn.parameters () )
            for p in self.model.parameters () :
                p.requires_grad_ ( False )
            for p in encoder_params :
                p.requires_grad_ ( True )
        else :
            encoder_params = list ( self.model.parameters () )


        self.model_optim = self.optimizer_dict[self.args.optimizer] (
            encoder_params , lr = self.args.learning_rate )


    def _setup_adversarial(self) :

        h_dim = getattr ( self.model , 'hidden_dim' , None ) or getattr ( self.args , 'hidden_dim' , 64 )

        if isinstance ( self.model , DAGCN_RUL ) :
            h_dim = 256
        self.adv_net   = AdversarialNet ( in_feature=h_dim , hidden_size=256 ).to ( self.device )
        self.adv_optim = self.optimizer_dict[self.args.optimizer] (
            self.adv_net.parameters () , lr = self.args.learning_rate )


    def _setup_dast ( self ) :

        self.model_optim = torch.optim.SGD (
            self.model.parameters () , lr=self.args.learning_rate , momentum=0.9 )
        self._dast_scheduler = torch.optim.lr_scheduler.StepLR (
            self.model_optim , step_size=80 , gamma=0.5 )


    def training_dast ( self , epoch : int = 0 ) :

        import itertools
        start_time = time ()
        train_loss  = []
        bce         = nn.BCELoss ()
        lam_fea     = getattr ( self.args , 'lam_da_fea' , 0.1 )
        lam_out     = getattr ( self.args , 'lam_da_out' , 0.5 )
        da_warmup   = getattr ( self.args , 'da_warmup'  , 5   )
        use_da      = epoch >= da_warmup

        self.model.train ()
        tgt_iter = itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , _ = next ( tgt_iter )

            batch_xs = batch_xs.to ( self.device )
            batch_ys = batch_ys.to ( self.device )
            batch_xt = batch_xt.to ( self.device )

            self.model_optim.zero_grad ()


            _ , out_s = self.model ( batch_xs )
            pred_s    = out_s [ 'gamma' ]
            if self.args.is_minmax :
                loss_mse = F.mse_loss ( pred_s , batch_ys / self.max_life )
            else :
                loss_mse = F.mse_loss ( pred_s , batch_ys )

            loss = loss_mse


            if use_da :
                _ , out_t = self.model ( batch_xt )
                B_s  = pred_s.size (0)
                B_t  = out_t ['gamma'].size (0)
                lbl_s = torch.ones  ( B_s , 1 , device=self.device )
                lbl_t = torch.zeros ( B_t , 1 , device=self.device )


                seq_s    = out_s ['out_seq'].squeeze (-1)
                seq_t    = out_t ['out_seq'].squeeze (-1)
                d1_s     = self.model.D1 ( seq_s )
                d1_t     = self.model.D1 ( seq_t )
                loss_D1  = 0.5 * ( bce ( d1_s , lbl_s ) + bce ( d1_t , lbl_t ) )


                feat_s   = out_s ['h']
                feat_t   = out_t ['h']
                d2_s     = self.model.D2 ( feat_s )
                d2_t     = self.model.D2 ( feat_t )
                loss_D2  = 0.5 * ( bce ( d2_s , lbl_s ) + bce ( d2_t , lbl_t ) )

                loss = loss_mse + lam_fea * loss_D2 + lam_out * loss_D1

            train_loss.append ( loss_mse.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()


        self._dast_scheduler.step ()

        return np.average ( train_loss ) , time () - start_time


    def _setup_cada_adda ( self ) :

        import copy
        from Models_RUL.CADA_RUL import _ADDA_Disc , _NCE
        lstm_hid = getattr ( self.args , 'lstm_hid'  , 32   )
        bid      = getattr ( self.args , 'lstm_bid'  , True )
        feat_dim = lstm_hid * ( 2 if bid else 1 )


        self.cada_src = copy.deepcopy ( self.model )
        self.cada_src.eval ()
        for p in self.cada_src.parameters () :
            p.requires_grad_ ( False )

        da_lr  = getattr ( self.args , 'da_lr'   , 5e-5 )
        nce_lr = getattr ( self.args , 'nce_lr'  , 1e-2 )

        self.cada_disc = _ADDA_Disc ( feat_dim ).to ( self.device )
        self.cada_nce  = _NCE ( feat_dim , self.args.input_length , self.args.num_nodes ).to ( self.device )

        self.cada_disc_optim = torch.optim.AdamW ( self.cada_disc.parameters ()         , lr=da_lr  , betas=(0.5, 0.9) )
        self.cada_enc_optim  = torch.optim.AdamW ( self.model.encoder.parameters ()     , lr=da_lr  , betas=(0.5, 0.9) )
        self.cada_nce_optim  = torch.optim.AdamW ( self.cada_nce.parameters ()          , lr=nce_lr , betas=(0.5, 0.9) )


    def training_cada ( self , epoch : int = 0 ) :

        import itertools
        pretrain_epochs = getattr ( self.args , 'pretrain_epochs' , 100 )
        start_time      = time ()
        train_loss      = []

        if epoch == pretrain_epochs :
            self._setup_cada_adda ()


        if epoch < pretrain_epochs :
            self.model.train ()
            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                self.model_optim.zero_grad ()
                _ , out = self.model ( batch_xs )
                pred    = out [ 'gamma' ]
                y       = batch_ys / self.max_life if self.args.is_minmax else batch_ys
                loss    = F.mse_loss ( pred , y )
                train_loss.append ( loss.item () )
                loss.backward ()
                torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
                self.model_optim.step ()


        else :
            bce       = nn.BCEWithLogitsLoss ()
            alpha_nce = getattr ( self.args , 'alpha_nce' , 0.2 )
            tgt_iter  = itertools.cycle ( self.target_train_loader )

            self.model.train ()
            self.cada_disc.train ()

            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xt , idxs_xt , _ = next ( tgt_iter )
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                batch_xt = batch_xt.to ( self.device )


                with torch.no_grad () :
                    _ , src_out  = self.cada_src ( batch_xs )
                    src_feat     = src_out [ 'h' ].squeeze (1)
                    _ , tgt_out  = self.model   ( batch_xt )
                    tgt_feat     = tgt_out [ 'h' ].squeeze (1)
                all_feat = torch.cat ( [ src_feat , tgt_feat ] )
                lbl_disc = torch.cat ( [
                    torch.ones  ( src_feat.size (0) , 1 , device=self.device ) ,
                    torch.zeros ( tgt_feat.size (0) , 1 , device=self.device ) ] )
                loss_disc = bce ( self.cada_disc ( all_feat ) , lbl_disc )
                self.cada_disc_optim.zero_grad ()
                loss_disc.backward ()
                self.cada_disc_optim.step ()


                self.cada_enc_optim.zero_grad ()
                self.cada_nce_optim.zero_grad ()
                _ , tgt_out2  = self.model ( batch_xt )
                tgt_feat2     = tgt_out2 [ 'h' ].squeeze (1)
                lbl_ones      = torch.ones ( tgt_feat2.size (0) , 1 , device=self.device )
                loss_adv      = bce ( self.cada_disc ( tgt_feat2 ) , lbl_ones )
                nce_loss      = self.cada_nce ( tgt_feat2 , batch_xt )
                loss_enc      = loss_adv + alpha_nce * nce_loss
                loss_enc.backward ()
                torch.nn.utils.clip_grad_norm_ ( self.model.encoder.parameters () , 1.0 )
                self.cada_enc_optim.step ()
                self.cada_nce_optim.step ()


                with torch.no_grad () :
                    _ , src_pred_out = self.model ( batch_xs )
                    y   = batch_ys / self.max_life if self.args.is_minmax else batch_ys
                    train_loss.append ( F.mse_loss ( src_pred_out [ 'gamma' ] , y ).item () )

        return np.average ( train_loss ) , time () - start_time


    def _setup_tacda_adda ( self ) :

        import copy
        from Models_RUL.CADA_RUL import _ADDA_Disc
        lstm_hid = getattr ( self.args , 'lstm_hid'  , 32   )
        bid      = getattr ( self.args , 'lstm_bid'  , True )
        feat_dim = lstm_hid * ( 2 if bid else 1 )

        self.tacda_src = copy.deepcopy ( self.model )
        self.tacda_src.eval ()
        for p in self.tacda_src.parameters () :
            p.requires_grad_ ( False )

        da_lr = getattr ( self.args , 'da_lr' , 5e-5 )
        self.tacda_disc = _ADDA_Disc ( feat_dim ).to ( self.device )
        self.tacda_disc_optim = torch.optim.AdamW ( self.tacda_disc.parameters (), lr=da_lr , betas=(0.5, 0.9) )

        self.tacda_enc_dec_optim = torch.optim.AdamW (
            [ { 'params': self.model.encoder.parameters () } ,
              { 'params': self.model.decoder.parameters () , 'lr': 100 * da_lr } ] ,
            lr=da_lr , betas=(0.5, 0.9) )


    def training_tacda ( self , epoch : int = 0 ) :

        import itertools
        pretrain_epochs = getattr ( self.args , 'pretrain_epochs' , 100 )
        start_time      = time ()
        train_loss      = []

        if epoch == pretrain_epochs :
            self._setup_tacda_adda ()


        if epoch < pretrain_epochs :
            self.model.train ()
            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                self.model_optim.zero_grad ()
                _ , out = self.model ( batch_xs )
                y       = batch_ys / self.max_life if self.args.is_minmax else batch_ys
                loss    = F.mse_loss ( out [ 'gamma' ] , y )
                train_loss.append ( loss.item () )
                loss.backward ()
                torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
                self.model_optim.step ()


        else :
            bce       = nn.BCEWithLogitsLoss ()
            alpha_dtw = getattr ( self.args , 'alpha_dtw' , 0.1 )
            tgt_iter  = itertools.cycle ( self.target_train_loader )

            self.model.train ()
            self.tacda_disc.train ()

            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xt , idxs_xt , _ = next ( tgt_iter )
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                batch_xt = batch_xt.to ( self.device )


                with torch.no_grad () :
                    _ , src_out = self.tacda_src ( batch_xs )
                    src_feat    = src_out [ 'h' ].squeeze (1)
                    _ , tgt_out = self.model   ( batch_xt )
                    tgt_feat    = tgt_out [ 'h' ].squeeze (1)
                all_feat  = torch.cat ( [ src_feat , tgt_feat ] )
                lbl_disc  = torch.cat ( [
                    torch.ones  ( src_feat.size (0) , 1 , device=self.device ) ,
                    torch.zeros ( tgt_feat.size (0) , 1 , device=self.device ) ] )
                loss_disc = bce ( self.tacda_disc ( all_feat ) , lbl_disc )
                self.tacda_disc_optim.zero_grad ()
                loss_disc.backward ()
                self.tacda_disc_optim.step ()


                self.tacda_enc_dec_optim.zero_grad ()
                _ , tgt_out2 = self.model ( batch_xt )
                tgt_feat2    = tgt_out2 [ 'h' ].squeeze (1)
                recon        = tgt_out2 [ 'recon' ]
                lbl_ones     = torch.ones ( tgt_feat2.size (0) , 1 , device=self.device )
                loss_adv     = bce ( self.tacda_disc ( tgt_feat2 ) , lbl_ones )
                loss_recon   = F.mse_loss ( recon , batch_xt )
                loss_enc     = loss_adv + alpha_dtw * loss_recon
                loss_enc.backward ()
                torch.nn.utils.clip_grad_norm_ (
                    list ( self.model.encoder.parameters () ) +
                    list ( self.model.decoder.parameters () ) , 1.0 )
                self.tacda_enc_dec_optim.step ()


                with torch.no_grad () :
                    _ , src_pred_out = self.model ( batch_xs )
                    y   = batch_ys / self.max_life if self.args.is_minmax else batch_ys
                    train_loss.append ( F.mse_loss ( src_pred_out [ 'gamma' ] , y ).item () )

        return np.average ( train_loss ) , time () - start_time

    def add_sensor_noise(self,signal, noise_level=0.05, cutoff_ratio=0.5):

        B, T, D = signal.shape


        signal_std = signal.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)


        noise = torch.randn_like(signal) * self.noise_std ( signal_std , noise_level )


        noise_freq = torch.fft.rfft(noise, dim=1)
        freqs = torch.linspace(0, 1, noise_freq.shape[1], device=signal.device)


        mask = (freqs >= cutoff_ratio).float().view(1, -1, 1)
        noise_freq_filtered = noise_freq * mask


        noise_high = torch.fft.irfft(noise_freq_filtered, n=T, dim=1)


        noisy_signal = signal + noise_high
        return noisy_signal


    def noise_std(self, signal_std, noise_level):

        return noise_level * signal_std

    def _print_config(self) :

        print ( f"|{'=' * 101}|" )
        for key , value in self.args.__dict__.items () :
            print ( f"|{str ( key ):>50s}|{str ( value ):<50s}|" )
        print ( f"|{'=' * 101}|" )

    def start(self) :
        epoch_times = []
        if self.args.train :

            train_steps = len ( self.train_loader )
            vali_steps = len ( self.vali_loader )
            print ( "train_steps: ", train_steps )
            print ( "validaion_steps: ", vali_steps )


            early_stopping = EarlyStopping ( patience = self.args.early_stop_patience, verbose = True )


            learning_rate_adapter = adjust_learning_rate_class ( self.args, True )

            self.model_optim = self._select_optimizer ()


        self._da_mode = self._resolve_da_mode ( self.args )


        if self.args.loss_type in ('MSE', 'DA', 'EviAdapt', 'Adversarial'):
            self.loss_criterion = nn.MSELoss()
        elif self.args.loss_type == 'MAE':
            self.loss_criterion = nn.L1Loss()
        elif self.args.loss_type == 'QUAN':
            self.loss_criterion = QuantileLoss(quantile=0.3)
        elif self.args.loss_type == 'Huber':
            self.loss_criterion = HuberLoss(delta=1.0)

        if self.args.resume :
            print ( 'Resuming using stored checkpoint....' )
            self._load_checkpoint ()


            if getattr ( self.args , 'inference_time_only' , False ) :
                num_runs = getattr ( self.args , 'inference_time_runs' , 100 )
                warmup = getattr ( self.args , 'inference_time_warmup' , 10 )
                mean_ms, std_ms, n_batches, n_samples = self.measure_pure_inference_time (
                    num_runs=num_runs , warmup=warmup )
                print ( f"INFERENCE_TIME_RESULT model={self.args.model_name} "
                        f"params={self.trainable_parameters} mean_ms={mean_ms:.6f} "
                        f"std_ms={std_ms:.6f} n_batches={n_batches} n_samples={n_samples} "
                        f"num_runs={num_runs}" )
                return None , None


            if getattr ( self.args , 'flops_only' , False ) :
                import sys as _sys
                _repo_root = os.path.dirname ( os.path.dirname ( os.path.abspath ( __file__ ) ) )
                if _repo_root not in _sys.path :
                    _sys.path.insert ( 0 , _repo_root )
                _flop_utils_dir = os.path.join ( _repo_root , 'analysis' )
                if _flop_utils_dir not in _sys.path :
                    _sys.path.insert ( 0 , _flop_utils_dir )
                from flop_utils import count_flops

                batch_x , idx_x , _ = next ( iter ( self.test_loader ) )
                batch_x = batch_x [ :1 ].to ( self.device )
                idx_x = idx_x [ :1 ].to ( self.device )
                result = count_flops ( self.model , ( batch_x , idx_x ) )


                print ( f"FLOPS_RESULT model={self.args.model_name} "
                        f"params={self.trainable_parameters} "
                        f"thop_layer_macs={result [ 'thop_layer_macs' ]} "
                        f"einsum_matmul_macs={result [ 'einsum_matmul_macs' ]} "
                        f"fft_macs_estimated={result [ 'fft_macs_estimated' ]:.2f} "
                        f"total_macs_excl_fft={result [ 'total_macs_excl_fft' ]} "
                        f"total_macs_incl_fft={result [ 'total_macs_incl_fft' ]}" )
                return None , None
        else :
            print ( 'Starting fresh training ....' )

        self._print_config ()


        if self._da_mode == 'eviadapt' and self.args.train and self.args.model_name != 'EviAdaptRUL' :
            self._setup_eviadapt ()
        if self._da_mode == 'adversarial' and self.args.train :
            self._setup_adversarial ()
        if self._da_mode == 'dast' and self.args.train :
            self._setup_dast ()


        if self.args.train :
            self.save_hparam ()

            print ( "start training" )


            _ , self.seed_generator = set_seed ( self.args.seed )

            for epoch in range ( self.args.train_epochs ) :

                if self._da_mode in ( 'lbgn', 'da_loop' , 'routing_mmd' ) :
                    train_loss, epoch_time = self.training_da ( epoch )
                elif self._da_mode == 'eviadapt' :
                    train_loss, epoch_time = self.training_eviadapt ( epoch )
                elif self._da_mode == 'adversarial' :
                    train_loss, epoch_time = self.training_adversarial ( epoch )
                elif self._da_mode == 'dast' :
                    train_loss, epoch_time = self.training_dast ( epoch )
                elif self._da_mode == 'cada' :
                    train_loss, epoch_time = self.training_cada ( epoch )
                elif self._da_mode == 'tacda' :
                    train_loss, epoch_time = self.training_tacda ( epoch )
                elif self._da_mode == 'ocs_dann' :
                    train_loss, epoch_time = self.training_ocs_dann ( epoch )
                elif self._da_mode == 'mdan' :
                    train_loss, epoch_time = self.training_mdan ( epoch )
                elif self._da_mode == 'ccdg' :
                    train_loss, epoch_time = self.training_ccdg ( epoch )
                else :
                    train_loss, epoch_time = self.training ( epoch )
                epoch_times.append(epoch_time)

                vali_loss = self.validation ( self.vali_loader, self.loss_criterion )

                print ( "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}. it takse {4:.7f} seconds".format (epoch + 1, train_steps, train_loss, vali_loss, epoch_time ) )


                early_stopping ( vali_loss, self.model, self.save_path )
                if early_stopping.early_stop :
                    print ( "Early stopping" )
                    break

                if self._da_mode not in ( 'dast' , 'cada' , 'tacda' ) :
                    learning_rate_adapter ( self.model_optim, vali_loss )


            check_point = torch.load ( self.save_path + '/' + 'best_checkpoint.pth' , map_location=self.device )
            self.model.load_state_dict ( check_point )
        avg_epoch_time = sum(epoch_times) / len(epoch_times) if len(epoch_times) > 0 else 0
        print(f"Average epoch time: {avg_epoch_time:.2f}s")

        if torch.cuda.is_available () :
            torch.cuda.synchronize()
        start_time = time()
        if self.args.test_all_subset== True :
            average_enc_loss, average_enc_overall_loss, overall_score = self.test_all_cmapss ()
        else :
            average_enc_loss, average_enc_overall_loss, overall_score = self.test ( self.test_loader )
        if torch.cuda.is_available () :
            torch.cuda.synchronize()
        end_time = time()
        inference_time = end_time - start_time

        print ( f"{self.args.dataset_name}: Test performance in term of RMSE: ", average_enc_loss, " of enc overall is: ",
                average_enc_overall_loss, ', and score :', overall_score )
        if self.args.dataset_name == 'CMAPSS' :
            src = self.args.Data_id_CMAPSS
            tgt = self.args.Data_id_CMAPSS_test
            dataset      = src if src == tgt else f'{src}_{tgt}'
            test_dataset = tgt
        elif self.args.dataset_name == 'N_CMAPSS' :
            src = self.args.Data_id_N_CMAPSS
            tgt = getattr ( self.args , 'Data_id_N_CMAPSS_test' , None ) or src
            dataset      = src if src == tgt else f'{src}_{tgt}'
            test_dataset = tgt
        else :
            raise ValueError ( 'without corresponding dataset' )

        logs_root = getattr ( self.args , 'logs_root' , None ) or './logs'
        log_path = logs_root + '/' + dataset + 'experimental_logs.csv'
        if not os.path.exists ( log_path ) :
            table_head = [
                ['train_dataset','test_dataset', 'model', "trainable_parameters", 'time','inference_time', 'LR', 'batch_size', 'loss_type', 'best_last_RMSE', 'score',
                 'windowsize', 'hidden_dim','train', 'savepath', 'resume','resumepath', 'info']
            ]
            write_csv ( log_path, table_head, 'w+' )

        time_now = datetime.datetime.now ().strftime ( '%Y%m%d-%H%M%S' )

        resume_dir = self.checkpoint_dir if self.args.resume else None

        if self.args.train :
            def _base_row ( test_ds , rmse , score ) :
                return {
                    'train_dataset'       : src ,
                    'test_dataset'        : test_ds ,
                    'model'               : self.args.model_name ,
                    'trainable_parameters': self.trainable_parameters ,
                    'time'                : time_now ,
                    'inference_time'      : avg_epoch_time ,
                    'LR'                  : self.args.learning_rate ,
                    'batch_size'          : self.args.batch_size ,
                    'loss_type'           : self.args.loss_type ,
                    'best_last_RMSE'      : rmse ,
                    'score'               : score ,
                    'windowsize'          : self.args.input_length ,
                    'hidden_dim'          : self.args.hidden_dim ,
                    'train'               : self.args.train ,
                    'savepath'            : self.save_path ,
                    'resume'              : self.args.resume ,
                    'resumepath'          : resume_dir ,
                    'info'                : self.args.info
                }

            if hasattr ( self , '_all_results' ) :
                a_log = [ _base_row ( tid , r[0] , r[1] )
                          for tid , r in self._all_results.items () ]
            else :
                a_log = [ _base_row ( test_dataset , average_enc_loss , overall_score ) ]

            write_csv_dict ( log_path, a_log, 'a+' )

        return average_enc_loss , overall_score

    @staticmethod
    def _pred_from_outputs ( outputs ) :

        if isinstance ( outputs , dict ) :
            p = outputs.get ( 'gamma' , outputs.get ( 'mean' ) )
        else :
            p = outputs
        if p is not None and p.dim () == 1 :
            p = p.unsqueeze ( -1 )
        return p


    def training_ocs_dann ( self , epoch : int = 0 ) :

        import itertools
        from Models_RUL.OCS_DANN import occ_to_stage
        start_time  = time ()
        train_loss  = []
        bce         = nn.BCELoss ( reduction='none' )
        lam_da      = getattr ( self.args , 'lam_da'  , 0.1 )
        lam_oc      = getattr ( self.args , 'lam_oc'  , 0.1 )
        max_life    = self.max_life

        self.model.train ()
        tgt_iter = itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , _ = next ( tgt_iter )
            batch_xs  = batch_xs.to  ( self.device )
            batch_ys  = batch_ys.to  ( self.device )
            idxs_xs   = idxs_xs.to   ( self.device )
            batch_xt  = batch_xt.to  ( self.device )

            self.model_optim.zero_grad ()


            occ_s    = idxs_xs.mean ( dim=(1, 2) )
            stage_s  = occ_to_stage ( occ_s , max_life )


            pred_s , h_s , oc_logits_s , d_s , d_t = self.model.forward_da ( batch_xs , batch_xt )


            y_s = batch_ys / max_life if self.args.is_minmax else batch_ys
            loss_rul = F.mse_loss ( pred_s , y_s )


            loss_oc = F.cross_entropy ( oc_logits_s , stage_s )


            oc_prob_s = torch.softmax ( oc_logits_s.detach () , dim=1 )
            oc_weight_s = oc_prob_s.gather ( 1 , stage_s.unsqueeze (1) ).squeeze (1)

            lbl_s = torch.ones  ( d_s.size (0) , 1 , device=self.device )
            lbl_t = torch.zeros ( d_t.size (0) , 1 , device=self.device )
            loss_d_s = ( bce ( d_s , lbl_s ).squeeze (1) * oc_weight_s ).mean ()
            loss_d_t = bce ( d_t , lbl_t ).mean ()

            loss = loss_rul + lam_oc * loss_oc + lam_da * ( loss_d_s + loss_d_t )
            train_loss.append ( loss_rul.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()

        return np.average ( train_loss ) , time () - start_time


    def training_mdan ( self , epoch : int = 0 ) :

        import itertools, math, random
        start_time      = time ()
        train_loss      = []
        pretrain_epochs = getattr ( self.args , 'pretrain_epochs' , 30 )
        alpha_mix       = getattr ( self.args , 'alpha_nce'       , 1.0 )
        max_life        = self.max_life
        temp            = 0.05

        self.model.train ()


        if epoch < pretrain_epochs :
            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                self.model_optim.zero_grad ()
                _ , out = self.model ( batch_xs )
                y = batch_ys / max_life if self.args.is_minmax else batch_ys
                loss = F.mse_loss ( out['gamma'] , y )
                train_loss.append ( loss.item () )
                loss.backward ()
                torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
                self.model_optim.step ()
            return np.average ( train_loss ) , time () - start_time


        if not hasattr ( self , '_mdan_lbd' ) :
            import numpy as _np
            self._mdan_lbd = float ( _np.random.beta ( 2 , 2 ) )

        tgt_iter = itertools.cycle ( self.target_train_loader )
        num_iter = min ( len ( self.source_loader ) , len ( self.target_train_loader ) )
        step     = 0

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , batch_yt = next ( tgt_iter )
            batch_xs = batch_xs.to ( self.device )
            batch_ys = batch_ys.to ( self.device )
            batch_xt = batch_xt.to ( self.device )
            batch_yt = batch_yt.to ( self.device )
            step    += 1

            self.model_optim.zero_grad ()
            lbd = self._mdan_lbd


            _ , out_s = self.model ( batch_xs )
            feat_s    = out_s [ 'h' ]
            pred_s    = out_s [ 'gamma' ]

            _ , out_t = self.model ( batch_xt )
            feat_t    = out_t [ 'h' ]
            pred_t    = out_t [ 'gamma' ].detach ()


            y_s = batch_ys / max_life if self.args.is_minmax else batch_ys
            y_t = batch_yt / max_life if self.args.is_minmax else batch_yt
            pseudo_t = pred_t.squeeze ()
            y_s_sq   = y_s.squeeze ()


            x_mix   = lbd * batch_xs + ( 1 - lbd ) * batch_xt
            y_mix   = lbd * y_s_sq   + ( 1 - lbd ) * pseudo_t

            _ , out_mix_in  = self.model ( x_mix )
            pred_mix_in     = out_mix_in [ 'h' ]


            feat_mix  = lbd * feat_s + ( 1 - lbd ) * feat_t
            pred_mix_f = self.model.regress ( feat_mix ).squeeze ()


            loss = alpha_mix * (
                F.mse_loss ( out_mix_in [ 'gamma' ].squeeze () , y_mix ) +
                F.mse_loss ( pred_mix_f , y_mix ) )


            with torch.no_grad () :
                fs  = feat_s.mean ( dim=0 )
                ft  = feat_t.mean ( dim=0 )
                fm  = feat_mix.mean ( dim=0 )
                d_sm = ( fs - fm ).abs ().mean ().item ()
                d_tm = ( ft - fm ).abs ().mean ().item ()
                if d_sm + d_tm > 0 :
                    q   = math.exp ( -d_sm / ( d_sm + d_tm * temp + 1e-8 ) )
                else :
                    q   = 0.5
                lbd_new = ( step * ( 1 - q ) / num_iter ) + q * lbd
                lbd_new = float ( torch.clamp (
                    torch.as_tensor ( random.uniform ( lbd_new - 0.2 , lbd_new + 0.2 ) ) ,
                    min=0.0 , max=1.0 ) )
                self._mdan_lbd = lbd_new

            train_loss.append ( F.mse_loss ( pred_s , y_s ).item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()

        return np.average ( train_loss ) , time () - start_time


    def training_ccdg ( self , epoch : int = 0 ) :

        import itertools
        from Models_RUL.CCDG_RUL import ccdg_contrastive_loss, occ_to_stage
        start_time  = time ()
        train_loss  = []
        lam_align   = getattr ( self.args , 'lam_align' , 0.1 )
        max_life    = self.max_life

        self.model.train ()
        tgt_iter = itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , _ = next ( tgt_iter )
            batch_xs  = batch_xs.to  ( self.device )
            batch_ys  = batch_ys.to  ( self.device )
            idxs_xs   = idxs_xs.to   ( self.device )
            batch_xt  = batch_xt.to  ( self.device )

            self.model_optim.zero_grad ()


            _ , out_s = self.model ( batch_xs )
            pred_s    = out_s [ 'gamma' ]
            feat_s    = out_s [ 'h' ]

            y_s = batch_ys / max_life if self.args.is_minmax else batch_ys
            loss_rul = F.mse_loss ( pred_s , y_s )


            _ , out_t = self.model ( batch_xt )
            feat_t    = out_t [ 'h' ]
            pred_t    = out_t [ 'gamma' ].detach ()


            occ_s   = idxs_xs.mean ( dim=(1, 2) )
            stage_s = occ_to_stage ( occ_s , max_life )


            if self.args.is_minmax :
                occ_t_approx = ( 1.0 - pred_t.squeeze () ) * max_life
            else :
                occ_t_approx = max_life - pred_t.squeeze ()
            stage_t = occ_to_stage ( occ_t_approx.clamp ( 0 , max_life ) , max_life )


            all_feat  = torch.cat ( [ feat_s , feat_t ] , dim=0 )
            all_stage = torch.cat ( [ stage_s , stage_t ] , dim=0 )
            loss_cont = ccdg_contrastive_loss ( all_feat , all_stage )

            loss = loss_rul + lam_align * loss_cont
            train_loss.append ( loss_rul.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()

        return np.average ( train_loss ) , time () - start_time

    def training(self, epoch=0) :
        print("stated training using without DA------------------------------------")
        start_time = time ()

        iter_count = 0
        train_loss = []

        self.model.train ()
        for i, (batch_x, idxs_x, batch_y) in enumerate ( tqdm ( self.train_loader ) ) :
            iter_count += 1
            self.model_optim.zero_grad ()

            batch_x = batch_x.to ( self.device )
            batch_y = batch_y.to ( self.device )


            _, outputs = self.model(batch_x, idxs_x)
            pred = self._pred_from_outputs(outputs)

            if self.args.is_minmax:
                batch_y_norm = batch_y / self.max_life
                loss = F.mse_loss(pred, batch_y_norm)
            else:
                loss = F.mse_loss(pred, batch_y)

            train_loss.append(loss.item())
            loss.backward()
            self.model_optim.step()

        end_time = time ()
        epoch_time = end_time - start_time
        train_loss = np.average ( train_loss )

        return train_loss, epoch_time


    def training_da(self, epoch) :

        import itertools
        start_time = time ()
        train_loss  = []
        warmup_epochs = getattr(self.args, 'warmup_epochs', 5)


        lam_mar_max   = getattr(self.args, 'lam_mar', 0.01)
        lam_da_max    = getattr(self.args, 'lam_da', 0.1)

        if warmup_epochs > 0:
            progress = epoch / warmup_epochs

            ramp = 0.5 * (1 - math.cos(math.pi * min(1.0, progress)))
        else:
            ramp = 1.0

        lam_mar = lam_mar_max * ramp
        lam_da = lam_da_max * ramp


        _lbgn_abl = getattr ( self.args , 'lbgn_ablation' , 'none' )
        is_source_only = ( _lbgn_abl == 'source_only' )
        if _lbgn_abl == 'no_mar' :
            lam_mar = 0.0
        elif _lbgn_abl == 'no_mmd' :
            lam_da  = 0.0
        elif _lbgn_abl == 'no_da' or is_source_only :
            lam_mar = 0.0
            lam_da  = 0.0


        if epoch == 0 :
            print ( f"[DA weights] lam_mar_max={lam_mar_max}  lam_da_max={lam_da_max}  "
                    f"(ablation={_lbgn_abl}, warmup_epochs={warmup_epochs}; "
                    f"epoch-0 effective after ramp: lam_mar={lam_mar:.6g}, lam_da={lam_da:.6g})" )

        self.model.train ()


        tgt_iter = None if is_source_only else itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            if not is_source_only :
                batch_xt , idxs_xt , _ = next ( tgt_iter )

            batch_xs  = batch_xs.to  ( self.device )
            batch_ys  = batch_ys.to  ( self.device )
            idxs_xs   = idxs_xs.to   ( self.device )
            if not is_source_only :
                batch_xt  = batch_xt.to  ( self.device )
                idxs_xt   = idxs_xt.to   ( self.device )

            self.model_optim.zero_grad ()


            occ_s = idxs_xs.mean ( dim=(1,2) )
            if not is_source_only :
                occ_t = idxs_xt.mean ( dim=(1,2) )


            _ , out_s = self.model ( batch_xs , idxs_xs )
            pred_s    = self._pred_from_outputs ( out_s )


            _proposed_models = ( 'working_model_RUL' , 'LBGN_RUL' , 'LBGN_RUL_v2' )
            if self.args.model_name in _proposed_models and _lbgn_abl != 'plain_mse' :
                occ_w   = ( occ_s - occ_s.min () ) / ( occ_s.max () - occ_s.min () + 1e-8 )
                if _lbgn_abl == 'inverted_occ_weight' :


                    weight = ( 1.0 - 0.5 * occ_w ).detach ()
                else :
                    weight  = ( 0.5 + 0.5 * occ_w ).detach ()
                if self.args.is_minmax :
                    batch_ys_norm = batch_ys / self.max_life
                    per_s = F.mse_loss ( pred_s , batch_ys_norm , reduction='none' ).mean ( dim=-1 )
                else :
                    per_s = F.mse_loss ( pred_s , batch_ys , reduction='none' ).mean ( dim=-1 )
                loss_mse = ( weight * per_s ).mean ()
            else :
                if self.args.is_minmax :
                    loss_mse = F.mse_loss ( pred_s , batch_ys / self.max_life )
                else :
                    loss_mse = F.mse_loss ( pred_s , batch_ys )

            if is_source_only :


                loss = loss_mse
            else :

                _ , out_t = self.model ( batch_xt , idxs_xt )


                if self.args.model_name in ( 'PDMN' , 'NDC_PDMN' ) :
                    lam_struct = getattr ( self.args , 'lam_struct' , 0.1 ) * ramp
                    lam_rmmd   = getattr ( self.args , 'lam_rmmd'   , 0.1 ) * ramp
                    loss_struct = pdmn_struct_loss ( self.model )
                    acts_s      = out_s.get ( 'activations' )
                    acts_t      = out_t.get ( 'activations' )
                    loss_rmmd   = routing_mmd ( acts_s , acts_t ) if (acts_s is not None and acts_t is not None) else torch.tensor ( 0.0 , device=self.device )
                    loss = loss_mse + lam_struct * loss_struct + lam_rmmd * loss_rmmd


                    if 'rate' in out_s :
                        from Models_RUL.NDC_PDMN import ndc_calibration_loss
                        lam_cal = getattr ( self.args , 'lam_cal' , 0.1 )
                        pooled  = torch.cat ( [ out_s [ 'rate' ] , out_t [ 'rate' ] ] )
                        loss    = loss + lam_cal * ndc_calibration_loss ( pooled )


                        lam_eol  = getattr ( self.args , 'lam_eol' , 1.0 )
                        eol_tgt  = getattr ( self.args , 'ndc_eol_target' , 1e-2 )
                        eol_mask = ( batch_ys.squeeze ( -1 ) < 0.5 )
                        if eol_mask.any () :
                            tau_end  = out_s [ 'tau' ] [ eol_mask , -1 , 0 ]
                            loss     = loss + lam_eol * ( ( tau_end - eol_tgt ) ** 2 ).mean ()
                else :


                    if self.args.model_name in ( 'LBGN_RUL' , 'LBGN_RUL_v2' ) :
                        _mar_loss , _adjacency_mmd = lbgn_mar_loss , lbgn_adjacency_mmd
                    else :
                        _mar_loss , _adjacency_mmd = mar_loss , adjacency_mmd

                    has_adj = 'adj' in out_s and 'adj' in out_t


                    compute_mar = has_adj and _lbgn_abl not in ( 'no_mar' , 'no_da' )
                    compute_mmd = has_adj and _lbgn_abl not in ( 'no_mmd' , 'no_da' )
                    loss_mar = ( _mar_loss ( out_s['adj'] , occ_s ) + _mar_loss ( out_t['adj'] , occ_t )
                                 if compute_mar else torch.tensor ( 0.0 , device=self.device ) )
                    loss_mmd = ( _adjacency_mmd ( out_s['adj'] , occ_s , out_t['adj'] , occ_t ,
                                                 n_stages = getattr ( self.args , 'mmd_stages' , 3 ) )
                                 if compute_mmd else torch.tensor ( 0.0 , device=self.device ) )
                    loss = loss_mse + lam_mar * loss_mar + lam_da * loss_mmd

            train_loss.append ( loss.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()

        epoch_time = time () - start_time
        return np.average ( train_loss ) , epoch_time


    def training_eviadapt(self, epoch: int = 0):

        if isinstance ( self.model , EviAdaptRUL ) :
            return self._training_eviadapt_full ( epoch )
        return self._training_eviadapt_legacy ( epoch )

    def _training_eviadapt_full ( self , epoch : int ) :

        import copy , itertools
        from Experiment.eviadapt_losses import (
            quant_evi_loss , assign_src_stages , assign_tgt_stages ,
            compute_stage_alignment_loss )

        pretrain_epochs = getattr ( self.args , 'pretrain_epochs' , 100 )
        lam_align       = getattr ( self.args , 'lam_align'       , 0.1  )
        nig_coeff       = getattr ( self.args , 'nig_coeff'        , 0.01 )
        start_time      = time ()
        train_loss      = []


        if epoch == pretrain_epochs :

            self.source_model = copy.deepcopy ( self.model )
            self.source_model.eval ()
            for p in self.source_model.parameters () :
                p.requires_grad_ ( False )

            for p in self.model.parameters () :
                p.requires_grad_ ( False )
            for p in self.model.feature_extractor.parameters () :
                p.requires_grad_ ( True )

            da_lr = getattr ( self.args , 'da_lr' , 5e-5 )
            self.model_optim = torch.optim.AdamW (
                self.model.feature_extractor.parameters () ,
                lr = da_lr , betas = ( 0.5 , 0.9 ) )


        if epoch == 0 :

            self._eviadapt_scheduler = torch.optim.lr_scheduler.StepLR (
                self.model_optim , step_size=10 , gamma=0.5 )

        if epoch < pretrain_epochs :
            self.model.train ()
            self._eviadapt_scheduler.step ()
            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xs = batch_xs.to ( self.device )
                batch_ys = batch_ys.to ( self.device )
                self.model_optim.zero_grad ()

                _ , out_s = self.model ( batch_xs , idxs_xs )
                y_norm = batch_ys / self.max_life if self.args.is_minmax else batch_ys
                loss   = quant_evi_loss ( y_norm , out_s['mu'] , out_s['v'] ,
                                         out_s['alpha'] , out_s['beta'] ,
                                         quantiles=(0.25, 0.75) , coeff=nig_coeff )
                train_loss.append ( loss.item () )
                loss.backward ()
                torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
                self.model_optim.step ()


        else :
            self.model.train ()
            tgt_iter = itertools.cycle ( self.target_train_loader )

            for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
                batch_xt , idxs_xt , _ = next ( tgt_iter )

                batch_xs = batch_xs.to ( self.device )
                idxs_xs  = idxs_xs.to  ( self.device )
                batch_xt = batch_xt.to ( self.device )
                idxs_xt  = idxs_xt.to  ( self.device )

                self.model_optim.zero_grad ()


                with torch.no_grad () :
                    _ , out_s_ref = self.source_model ( batch_xs , idxs_xs )


                _ , out_t = self.model ( batch_xt , idxs_xt )


                occ_s      = idxs_xs.mean ( dim=(1,2) )
                src_stages = assign_src_stages ( occ_s )

                pred_t     = out_t['gamma'].detach ().squeeze ()

                q30        = torch.quantile ( pred_t , 0.3 )
                tgt_stages = assign_tgt_stages ( pred_t , q30 )

                loss_align = compute_stage_alignment_loss (
                    out_s_ref , out_t , src_stages , tgt_stages ,
                    active_stages=(1, 2) , device=self.device )

                loss = lam_align * loss_align
                train_loss.append ( loss.item () )
                loss.backward ()
                torch.nn.utils.clip_grad_norm_ (
                    self.model.feature_extractor.parameters () , 1.0 )
                self.model_optim.step ()

        return np.average ( train_loss ) , time () - start_time

    def _training_eviadapt_legacy ( self , epoch : int ) :

        import itertools
        start_time  = time ()
        train_loss  = []
        n_stages    = getattr ( self.args , 'eviadapt_stages' , 3 )
        lam_align   = getattr ( self.args , 'lam_da' , 0.1 )
        warmup      = getattr ( self.args , 'warmup_epochs' , 10 )
        ramp        = min ( 1.0 , epoch / max ( warmup , 1 ) )

        self.model.train ()
        tgt_iter = itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , _ = next ( tgt_iter )

            batch_xs = batch_xs.to ( self.device )
            batch_ys = batch_ys.to ( self.device )
            idxs_xs  = idxs_xs.to  ( self.device )
            batch_xt = batch_xt.to ( self.device )
            idxs_xt  = idxs_xt.to  ( self.device )

            self.model_optim.zero_grad ()

            occ_s = idxs_xs.mean ( dim=(1,2) )
            occ_t = idxs_xt.mean ( dim=(1,2) )

            _ , out_s = self.model ( batch_xs , idxs_xs )
            pred_s    = self._pred_from_outputs ( out_s )
            if self.args.is_minmax :
                loss_mse = F.mse_loss ( pred_s , batch_ys / self.max_life )
            else :
                loss_mse = F.mse_loss ( pred_s , batch_ys )

            _ , out_t = self.model ( batch_xt , idxs_xt )

            loss_align = torch.tensor ( 0.0 , device=self.device )
            h_s = out_s.get ( 'h' )
            h_t = out_t.get ( 'h' )
            if h_s is not None and h_t is not None :
                h_s_flat  = h_s.mean ( dim=1 ) if h_s.dim () == 3 else h_s
                h_t_flat  = h_t.mean ( dim=1 ) if h_t.dim () == 3 else h_t
                src_stage = ( occ_s * n_stages ).long ().clamp ( 0 , n_stages - 1 )
                tgt_stage = ( occ_t * n_stages ).long ().clamp ( 0 , n_stages - 1 )
                for s in range ( n_stages ) :
                    ms = ( src_stage == s )
                    mt = ( tgt_stage == s )
                    if ms.sum () >= 1 and mt.sum () >= 1 :
                        loss_align += F.mse_loss (
                            h_t_flat[mt].mean ( 0 ) ,
                            h_s_flat[ms].mean ( 0 ).detach () )

            loss = loss_mse + ramp * lam_align * loss_align
            train_loss.append ( loss_mse.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()

        return np.average ( train_loss ) , time () - start_time


    def training_adversarial(self, epoch: int = 0):

        import itertools
        start_time = time ()
        train_loss = []
        bce        = nn.BCELoss ()
        warmup     = getattr ( self.args , 'warmup_epochs' , 10 )
        ramp       = min ( 1.0 , epoch / max ( warmup , 1 ) )
        lam_mar    = getattr ( self.args , 'lam_mar' , 0.01 ) * ramp
        lam_da     = getattr ( self.args , 'lam_da'  , 0.1  ) * ramp


        if epoch == 0 :
            print ( f"[DA weights] lam_mar_max={getattr(self.args,'lam_mar',0.01)}  "
                    f"lam_da_max={getattr(self.args,'lam_da',0.1)}  (warmup_epochs={warmup}; "
                    f"epoch-0 effective after ramp: lam_mar={lam_mar:.6g}, lam_da={lam_da:.6g})" )

        self.model.train ()
        self.adv_net.train ()
        tgt_iter = itertools.cycle ( self.target_train_loader )

        for ( batch_xs , idxs_xs , batch_ys ) in tqdm ( self.source_loader ) :
            batch_xt , idxs_xt , _ = next ( tgt_iter )

            batch_xs = batch_xs.to ( self.device )
            batch_ys = batch_ys.to ( self.device )
            idxs_xs  = idxs_xs.to  ( self.device )
            batch_xt = batch_xt.to ( self.device )
            idxs_xt  = idxs_xt.to  ( self.device )

            self.model_optim.zero_grad ()
            self.adv_optim.zero_grad ()

            occ_s = idxs_xs.mean ( dim=(1,2) )
            occ_t = idxs_xt.mean ( dim=(1,2) )


            _ , out_s  = self.model ( batch_xs , idxs_xs )
            pred_s     = self._pred_from_outputs ( out_s )
            if self.args.is_minmax :
                loss_mse = F.mse_loss ( pred_s , batch_ys / self.max_life )
            else :
                loss_mse = F.mse_loss ( pred_s , batch_ys )


            _ , out_t = self.model ( batch_xt , idxs_xt )


            loss_mar = torch.tensor ( 0.0 , device=self.device )
            if 'adj' in out_s and 'adj' in out_t :
                loss_mar = mar_loss ( out_s['adj'] , occ_s ) + mar_loss ( out_t['adj'] , occ_t )


            loss_adv = torch.tensor ( 0.0 , device=self.device )
            loss_dan = torch.tensor ( 0.0 , device=self.device )
            h_s = out_s.get ( 'h' )
            h_t = out_t.get ( 'h' )
            if h_s is not None and h_t is not None :
                h_s_flat = h_s.mean ( dim=1 ) if h_s.dim () == 3 else h_s
                h_t_flat = h_t.mean ( dim=1 ) if h_t.dim () == 3 else h_t


                disc_s = self.adv_net ( h_s_flat )
                disc_t = self.adv_net ( h_t_flat )
                lbl_s  = torch.ones  ( disc_s.size (0) , 1 , device=self.device )
                lbl_t  = torch.zeros ( disc_t.size (0) , 1 , device=self.device )
                loss_adv = bce ( disc_s , lbl_s ) + bce ( disc_t , lbl_t )


                if self.args.model_name == 'DAGCN_RUL' :
                    from Experiment.eviadapt_losses import dan_mmd
                    loss_dan = dan_mmd ( h_s_flat , h_t_flat )

            loss = loss_mse + lam_mar * loss_mar + lam_da * ( loss_adv + loss_dan )

            train_loss.append ( loss_mse.item () )
            loss.backward ()
            torch.nn.utils.clip_grad_norm_ ( self.model.parameters () , 1.0 )
            self.model_optim.step ()
            self.adv_optim.step ()

        return np.average ( train_loss ) , time () - start_time


    def validation(self, vali_loader, criterion) :
        self.model.eval ()
        val_losses = []

        for i, (batch_x, idx_x, batch_y) in enumerate ( vali_loader ) :
            batch_x = batch_x.to ( self.device )
            batch_y = batch_y.to ( self.device )

            if self.args.is_minmax :
                batch_y_norm = batch_y / self.max_life
                _, outputs = self.model ( batch_x, idx_x )
                pred = self._pred_from_outputs ( outputs )
                loss = self.loss_criterion ( pred, batch_y_norm )

            else :
                _, outputs = self.model ( batch_x, idx_x )
                pred = self._pred_from_outputs ( outputs )
                loss = self.loss_criterion ( pred, batch_y )

            val_losses.append ( loss.item () )

        average_vali_loss = np.average ( val_losses )

        self.model.train ()
        return average_vali_loss


    def test_all_cmapss(self) :

        args   = self.args
        kw     = dict ( sequence_length=args.input_length , MAXLIFE=args.MAXLIFE_CMAPSS ,
                        is_difference=args.is_diff , validation=args.validation )
        all_ids = ['FD001','FD002','FD003','FD004']
        results = {}

        for tid in all_ids :
            _ , _ , _ , _ , _ , _ , X_t , y_t , idx_t , _ = get_cmapss_data_ (
                data_path    = args.data_path_CMAPSS ,
                Data_id      = tid ,
                test_Data_id = tid , **kw )
            loader = DataLoader (
                CMAPSSData ( X_t , idx_t , y_t ) ,
                batch_size = args.batch_size , shuffle=False , num_workers=0 , drop_last=False )
            rmse , _ , score = self.test ( loader )
            results[tid] = ( rmse , score )
            tag = '(in-domain)' if tid == args.Data_id_CMAPSS else ''
            print ( f"  {args.Data_id_CMAPSS} → {tid} {tag:12s} | RMSE {rmse:.4f}  Score {score:.1f}" )


        self._all_results = results


        primary = args.Data_id_CMAPSS_test
        return results[primary][0] , results[primary][0] , results[primary][1]

    def test(self, test_loader):
        self.model.eval()
        enc_pred, gt = [], []
        for i, (batch_x, idx_x, batch_y) in enumerate(test_loader):
            batch_x = batch_x.to(self.device)
            if self.args.add_noise:
                batch_x = self.add_sensor_noise(batch_x, self.args.noise_level_test)
            batch_y = batch_y.to(self.device)
            idx_x = idx_x.to(self.device)

            _, outputs = self.model(batch_x, idx_x)
            outputs = self._pred_from_outputs(outputs)
            if self.args.is_minmax:
                outputs = outputs * self.max_life

            batch_y = batch_y.detach().cpu().numpy()
            enc = outputs.detach().cpu().numpy()

            if not np.isfinite(enc).all():
                print(f"Invalid output detected in batch {i}, replacing")
                enc = np.nan_to_num(enc, nan=0.0, posinf=200.0, neginf=0.0)

            gt.append(batch_y)
            enc_pred.append(enc)

        gt = np.concatenate(gt).reshape(-1, 1)
        enc_pred = np.concatenate(enc_pred).reshape(-1, 1)
        enc_var = None

        if self.args.save_test:
            np.savez(self.save_path + '/result.npz',
                     test_preds=enc_pred, test_trues=gt)

        average_enc_loss = np.sqrt(mean_squared_error(enc_pred, gt))
        average_enc_overall_loss = average_enc_loss
        overall_score = self.score_compute(enc_pred, gt)


        return average_enc_loss, average_enc_overall_loss, overall_score


    def measure_pure_inference_time(self, num_runs=100, warmup=10):

        self.model.eval()
        batches = []
        for batch_x, idx_x, _ in self.test_loader:
            batches.append((batch_x.to(self.device), idx_x.to(self.device)))
        n_samples = sum(bx.shape[0] for bx, _ in batches)

        def _one_pass():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time()
            with torch.no_grad():
                for batch_x, idx_x in batches:
                    self.model(batch_x, idx_x)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return time() - t0

        for _ in range(warmup):
            _one_pass()

        times_ms = np.array([_one_pass() for _ in range(num_runs)]) * 1000.0
        return float(times_ms.mean()), float(times_ms.std()), len(batches), n_samples

    def measure_inference_time(self, num_runs=200, warmup=10):
        self.model.eval ()


        print(f"Warming up for {warmup} runs...")
        with torch.no_grad():
            for _ in range(warmup):

                _, _, _ = self.test(self.test_loader)


        inference_times = []

        with torch.no_grad():
            for run in range(num_runs):
                torch.cuda.synchronize()
                start_time = time()


                average_enc_loss, average_enc_overall_loss, overall_score = self.test(self.test_loader)

                torch.cuda.synchronize()
                end_time = time()

                inference_time = end_time - start_time
                inference_times.append(inference_time)


        inference_times = np.array(inference_times)
        inference_times_ms = inference_times * 1000
        mean_ms = np.mean(inference_times_ms)
        std_ms = np.std(inference_times_ms)

        print("\n" + "="*50)
        print(f"Inference Time Statistics of {self.args.model_name} for ({num_runs} runs):{mean_ms:.2f} {std_ms:.2f}")

        return inference_times
    def score_compute(self, pred, gt) :

        B = pred.shape
        score = 0
        score_list = np.where ( pred - gt < 0, np.exp ( -(pred - gt) / 13 ) - 1, np.exp ( (pred - gt) / 10 ) - 1 )
        if self.args.dataset_name == 'CMAPSS' :
            score = np.sum ( score_list )
        else :
            score = np.mean ( score_list )
        return score


    def calib_compute(self, pred, gt, var, levels=(0.5, 0.7, 0.9, 0.95)):
        z = {0.5: 0.674, 0.7: 1.036, 0.9: 1.645, 0.95: 1.960}
        sd = np.sqrt(np.maximum(var, 1e-8))
        errs, picp95, mpiw95 = [], None, None
        for q in levels:
            lo, hi = pred - z[q] * sd, pred + z[q] * sd
            cov = float(np.mean((gt >= lo) & (gt <= hi)))
            wid = float(np.mean(hi - lo))
            errs.append(abs(cov - q))
            if q == 0.95:
                picp95, mpiw95 = cov, wid
        return picp95, mpiw95, float(np.mean(errs))


class QuantileLoss(nn.Module):
    def __init__(self, quantile=0.1):
        super(QuantileLoss, self).__init__()
        self.quantile = quantile

    def forward(self, pred, target):
        errors = target - pred
        loss = torch.max(
            (self.quantile - 1) * errors,
            self.quantile * errors
        )
        return torch.mean(loss)

class HuberLoss(nn.Module):
    def __init__(self, delta=1.0):
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, pred, target):
        errors = target - pred
        abs_errors = torch.abs(errors)


        quadratic = 0.5 * errors ** 2
        linear = self.delta * (abs_errors - 0.5 * self.delta)

        loss = torch.where(abs_errors <= self.delta, quadratic, linear)
        return torch.mean(loss)
