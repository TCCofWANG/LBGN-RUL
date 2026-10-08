

def get_configs(dataset , dataset_id) :
    hparams_class = get_hparams_class ( dataset )
    hparams = hparams_class ( dataset_id )
    return hparams


def get_hparams_class(dataset_name) :

    if dataset_name not in globals ( ) :
        raise NotImplementedError ( "Dataset not found: {}".format ( dataset_name ) )
    return globals ( )[dataset_name]


class CMAPSS ( ) :
    def __init__(self , dataset_id) :
        super ( CMAPSS , self ).__init__ ( )

        if dataset_id == 'FD001' :
            self.train_params = {
                'PDMN'       : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'  : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL': {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                      'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'   : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'   : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'  : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2': {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'      : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'     : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }
            self.alg_hparams = {
                'PDMN'       : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'   : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'  : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL': {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'   : {'num_nodes' : 14 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'   : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'  : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'   : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 16, 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3 } ,
                'LBGN_RUL_v2': {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 16, 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3 } ,
                'OCS_DANN'   : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'   : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'   : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'      : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 64 , 'dropout' : 0.1 , 'AATE_dim' : 10 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'     : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }

        elif dataset_id == 'FD002' :
            self.train_params = {
                'PDMN'       : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'LBGN_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2': {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'DAGCN_RUL'  : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL': {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' , 'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'   : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'   : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' , 'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'  : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' , 'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'OCS_DANN'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'   : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'      : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'     : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }
            self.alg_hparams = {
                'PDMN'             : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'        : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'      : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'         : {'num_nodes' : 14 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'         : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'        : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_mar' : 5 , 'lam_da' : 3 } ,
                'LBGN_RUL_v2'      : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_mar' : 5 , 'lam_da' : 3 } ,
                'OCS_DANN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3} ,
                'MDAN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3} ,
                'AIDGN'            : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 64 , 'dropout' : 0.1 , 'AATE_dim' : 10 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'           : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }

        elif dataset_id == 'FD003' :
            self.train_params = {
                'PDMN'             : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'        : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'      : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                      'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'         : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'         : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'        : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'      : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'            : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }

            self.alg_hparams = {
                'PDMN'             : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'        : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'      : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'         : {'num_nodes' : 14 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'         : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'        : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'      : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3} ,
                'MDAN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.3} ,
                'AIDGN'            : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 64 , 'dropout' : 0.1 , 'AATE_dim' : 10 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'           : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }
        elif dataset_id == 'FD004' :
            self.train_params = {
                'NDC_PDMN'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'        : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'      : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                      'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'         : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'         : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'        : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                      'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'      : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'         : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'            : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }
            self.alg_hparams = {
                'PDMN'             : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 , 'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'        : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'      : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'         : {'num_nodes' : 14 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'         : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'        : {'num_nodes' : 14 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'      : {'num_nodes' : 14 , 'input_length' : 50 , 'LCE_dim' : 8 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'         : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'            : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 64 , 'dropout' : 0.1 , 'AATE_dim' : 10 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'           : {'num_nodes' : 14 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }
        else :
            raise ValueError ( 'No input dataset id for CMAPSS' )


class N_CMAPSS ( ) :
    def __init__(self , dataset_id=None) :
        super ( N_CMAPSS , self ).__init__ ( )
        if dataset_id == 'DS01' :
            self.train_params = {
                'PDMN'                  : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'             : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'           : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                          'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'              : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'              : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'             : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'                 : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'                : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }

            self.alg_hparams = {
                'PDMN'                  : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'             : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'           : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'              : {'num_nodes' : 20 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'              : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'             : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'           : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'                 : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 128 , 'dropout' : 0.1 , 'AATE_dim' : 16 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'                : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }

        elif dataset_id == 'DS02' :
            self.train_params = {
                'PDMN'                  : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'             : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'           : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                          'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'              : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'              : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'             : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'                 : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'                : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }

            self.alg_hparams = {
                'PDMN'                  : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'             : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'           : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'              : {'num_nodes' : 20 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'              : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'             : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'           : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'                 : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 128 , 'dropout' : 0.1 , 'AATE_dim' : 16 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'                : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }

        elif dataset_id == 'DS03' :
            self.train_params = {
                'PDMN'                  : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'             : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'           : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                          'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'              : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'              : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'             : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'                 : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'                : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }

            self.alg_hparams = {
                'PDMN'                  : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'             : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'           : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'              : {'num_nodes' : 20 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'              : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'             : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'           : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'                 : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 128 , 'dropout' : 0.1 , 'AATE_dim' : 16 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'                : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }

        elif dataset_id == 'DS04' :
            self.train_params = {
                'PDMN'                  : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'NDC_PDMN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'routing_mmd'} ,
                'DAGCN_RUL'             : {'train_epochs' : 150 , 'batch_size' : 64  , 'learning_rate' : 0.001, 'da_mode' : 'adversarial'} ,
                'EviAdaptRUL'           : {'train_epochs' : 150 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'eviadapt' ,
                                          'pretrain_epochs' : 30  , 'nig_coeff' : 0.3  , 'lam_align' : 0.1  , 'da_lr' : 5e-5} ,
                'DAST_RUL'              : {'train_epochs' : 240 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'dast'} ,
                'CADA_RUL'              : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'cada' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_nce' : 0.2 , 'nce_lr' : 1e-2} ,
                'TACDA_RUL'             : {'train_epochs' : 175 , 'batch_size' : 256 , 'learning_rate' : 0.001, 'da_mode' : 'tacda' ,
                                          'pretrain_epochs' : 100 , 'da_lr' : 5e-5 , 'alpha_dtw' : 0.1} ,
                'LBGN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'LBGN_RUL_v2'           : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'lbgn'} ,
                'OCS_DANN'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ocs_dann'} ,
                'MDAN_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'mdan' , 'pretrain_epochs' : 30} ,
                'CCDG_RUL'              : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'da_mode' : 'ccdg'} ,
                'AIDGN'                 : {'train_epochs' : 300 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
                'DLGNet'                : {'train_epochs' : 150 , 'batch_size' : 128 , 'learning_rate' : 0.001, 'loss_type' : 'MSE'} ,
            }

            self.alg_hparams = {
                'PDMN'                  : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1} ,
                'NDC_PDMN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'K' : 8 ,
                                          'lam_struct' : 0.1 , 'lam_rmmd' : 0.1 , 'lam_cal' : 0.05 , 'lam_eol' : 0.5 , 'ndc_hidden' : 8} ,
                'DAGCN_RUL'             : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_da' : 0.1} ,
                'EviAdaptRUL'           : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 32 , 'dropout' : 0.5 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'lstm_bid' : True} ,
                'DAST_RUL'              : {'num_nodes' : 20 , 'input_length' : 70 , 'hidden_dim' : 32 , 'd_model' : 32 , 'nhead' : 8 , 'nlayers' : 2 , 'dropout' : 0.1 , 'lam_da_fea' : 0.1 , 'lam_da_out' : 0.5 , 'da_warmup' : 5} ,
                'CADA_RUL'              : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'TACDA_RUL'             : {'num_nodes' : 20 , 'input_length' : 30 , 'hidden_dim' : 64 , 'lstm_hid' : 32 , 'lstm_n_layers' : 5 , 'dropout' : 0.5 , 'lstm_bid' : True} ,
                'LBGN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'LBGN_RUL_v2'           : {'num_nodes' : 20 , 'input_length' : 50 , 'LCE_dim' : 16 , 'n_bands' : 4 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lam_mar' : 5 , 'lam_da' : 3} ,
                'OCS_DANN'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'MDAN_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'lstm_n_layers' : 3 , 'lstm_bid' : True} ,
                'CCDG_RUL'              : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1} ,
                'AIDGN'                 : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 128 , 'dropout' : 0.1 , 'AATE_dim' : 16 , 'nlayer' : 1 , 'hop' : 1} ,
                'DLGNet'                : {'num_nodes' : 20 , 'input_length' : 50 , 'hidden_dim' : 32 , 'dropout' : 0.1 , 'LCE_dim' : 8 , 'z_dim' : 1} ,
            }
