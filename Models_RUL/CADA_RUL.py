

import torch
import torch.nn as nn
import torch.nn.functional as F


class _NCE ( nn.Module ) :

    def __init__ ( self , feat_dim=64 , seq_len=30 , out_dim=14 ) :
        super ().__init__ ()
        self.seq_len   = seq_len
        self.lsoftmax  = nn.LogSoftmax ( dim=1 )
        self.predictor = nn.Linear ( feat_dim , out_dim )

    def forward ( self , f_t , input_x ) :

        batch_size = input_x.size (0)
        preds = self.predictor ( f_t )
        nce   = 0.0
        for i in range ( self.seq_len ) :
            outs  = preds.permute ( 1 , 0 )
            total = torch.mm ( input_x [ : , i , : ] , outs )
            nce  += torch.sum ( torch.diag ( self.lsoftmax ( total ) ) )
        nce = nce / ( -1.0 * batch_size * self.seq_len )
        return nce


class _ADDA_Disc ( nn.Module ) :

    def __init__ ( self , feat_dim=64 ) :
        super ().__init__ ()
        self.net = nn.Sequential (
            nn.Linear ( feat_dim , feat_dim ) ,
            nn.ReLU () ,
            nn.Linear ( feat_dim , feat_dim // 2 ) ,
            nn.ReLU () ,
            nn.Linear ( feat_dim // 2 , 1 ) ,
        )

    def forward ( self , x ) :
        return self.net ( x )


class CADA_RUL ( nn.Module ) :

    def __init__ ( self , args ) :
        super ().__init__ ()
        C         = args.num_nodes
        lstm_hid  = getattr ( args , 'lstm_hid'      , 32   )
        n_layers  = getattr ( args , 'lstm_n_layers' , 5    )
        dropout   = getattr ( args , 'dropout'       , 0.5  )
        bid       = getattr ( args , 'lstm_bid'      , True )
        feat_dim  = lstm_hid * ( 2 if bid else 1 )

        lstm_drop = dropout if n_layers > 1 else 0.0
        self.encoder   = nn.LSTM ( C , lstm_hid , n_layers , dropout=lstm_drop ,
                                   batch_first=True , bidirectional=bid )
        self.regressor = nn.Sequential (
            nn.Linear  ( feat_dim , lstm_hid  ) , nn.ReLU () , nn.Dropout ( dropout ) ,
            nn.Linear  ( lstm_hid , lstm_hid // 2 ) , nn.ReLU () , nn.Dropout ( dropout ) ,
            nn.Linear  ( lstm_hid // 2 , 1 ) ,
        )

    def forward ( self , x , OCC=None ) :

        enc_out , _  = self.encoder ( x )
        features     = enc_out [ : , -1 , : ]
        pred         = self.regressor ( features )

        return None , {
            'gamma'  : pred ,
            'h'      : features.unsqueeze (1) ,
            'enc_out': enc_out ,
        }
