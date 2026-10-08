

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseNormalGamma ( nn.Module ) :

    def __init__ ( self , input_dim : int , units : int ) :
        super ().__init__ ()
        self.units = int ( units )
        self.dense = nn.Linear ( input_dim , 4 * self.units )

    def forward ( self , x ) :
        out = self.dense ( x )
        mu , logv , logalpha , logbeta = torch.split ( out , self.units , dim=-1 )
        v     = F.softplus ( logv )
        alpha = F.softplus ( logalpha ) + 1.0
        beta  = F.softplus ( logbeta )
        return mu , v , alpha , beta


class EviAdaptRUL ( nn.Module ) :

    def __init__ ( self , args ) :
        super ().__init__ ()
        C         = args.num_nodes
        lstm_hid  = getattr ( args , 'lstm_hid'      , 32   )
        n_layers  = getattr ( args , 'lstm_n_layers' , 5    )
        bidirect  = getattr ( args , 'lstm_bid'      , True )
        dropout   = getattr ( args , 'dropout'       , 0.5  )
        feat_dim  = lstm_hid * ( 2 if bidirect else 1 )


        lstm_drop = dropout if n_layers > 1 else 0.0

        self.feature_extractor = nn.LSTM (
            input_size    = C ,
            hidden_size   = lstm_hid ,
            num_layers    = n_layers ,
            dropout       = lstm_drop ,
            batch_first   = True ,
            bidirectional = bidirect ,
        )

        self.evi_head = nn.Sequential (
            nn.Linear ( feat_dim , feat_dim ) ,
            nn.ReLU () ,
            nn.Dropout ( dropout ) ,
            nn.Linear ( feat_dim , feat_dim ) ,
            nn.ReLU () ,
            nn.Dropout ( dropout ) ,
            _DenseNormalGamma ( feat_dim , 2 ) ,
        )

    def forward ( self , x , OCC=None ) :

        enc_out , _ = self.feature_extractor ( x )
        h           = enc_out [ : , -1 , : ]

        mu , v , alpha , beta = self.evi_head ( h )


        pred = mu.mean ( dim=-1 , keepdim=True )

        return None , {
            'gamma' : pred ,
            'mu'    : mu   ,
            'v'     : v    ,
            'alpha' : alpha ,
            'beta'  : beta  ,
            'h'     : h.unsqueeze ( 1 ) ,
        }
