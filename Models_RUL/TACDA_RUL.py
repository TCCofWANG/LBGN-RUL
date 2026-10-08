

import torch
import torch.nn as nn
from Models_RUL.CADA_RUL import _ADDA_Disc


class TACDA_RUL ( nn.Module ) :

    def __init__ ( self , args ) :
        super ().__init__ ()
        C         = args.num_nodes
        lstm_hid  = getattr ( args , 'lstm_hid'      , 32   )
        n_layers  = getattr ( args , 'lstm_n_layers' , 5    )
        dropout   = getattr ( args , 'dropout'       , 0.5  )
        bid       = getattr ( args , 'lstm_bid'      , True )
        feat_dim  = lstm_hid * ( 2 if bid else 1 )
        dec_hid   = C // 2

        lstm_drop = dropout if n_layers > 1 else 0.0

        self.encoder   = nn.LSTM ( C , lstm_hid , n_layers , dropout=lstm_drop ,
                                   batch_first=True , bidirectional=bid )

        self.decoder   = nn.LSTM ( feat_dim , dec_hid , n_layers , dropout=lstm_drop ,
                                   batch_first=True , bidirectional=bid )
        self.regressor = nn.Sequential (
            nn.Linear  ( feat_dim , lstm_hid      ) , nn.ReLU () , nn.Dropout ( dropout ) ,
            nn.Linear  ( lstm_hid , lstm_hid // 2 ) , nn.ReLU () , nn.Dropout ( dropout ) ,
            nn.Linear  ( lstm_hid // 2 , 1 )       ,
        )

    def forward ( self , x , OCC=None ) :

        enc_out , _  = self.encoder ( x )
        features     = enc_out [ : , -1 , : ]
        pred         = self.regressor ( features )
        recon , _    = self.decoder   ( enc_out )

        return None , {
            'gamma'  : pred ,
            'h'      : features.unsqueeze (1) ,
            'enc_out': enc_out ,
            'recon'  : recon ,
        }
