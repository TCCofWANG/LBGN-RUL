

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class _ReverseLayer ( Function ) :
    @staticmethod
    def forward ( ctx , x , alpha ) :
        ctx.alpha = alpha
        return x.view_as ( x )

    @staticmethod
    def backward ( ctx , grad_output ) :
        return grad_output.neg () * ctx.alpha , None


class _PositionalEncoding ( nn.Module ) :
    def __init__ ( self , d_model , dropout=0.1 , max_len=512 ) :
        super ().__init__ ()
        self.dropout = nn.Dropout ( p=dropout )
        pe            = torch.zeros ( max_len , d_model )
        position      = torch.arange ( max_len ).unsqueeze ( 1 ).float ()
        div_term      = torch.exp ( torch.arange ( 0 , d_model , 2 ).float () *
                                    ( -math.log ( 10000.0 ) / d_model ) )
        pe [ : , 0::2 ] = torch.sin ( position * div_term )
        pe [ : , 1::2 ] = torch.cos ( position * div_term )
        self.register_buffer ( 'pe' , pe )

    def forward ( self , x ) :

        x = x + self.pe [ :x.size (1) ].unsqueeze (0)
        return self.dropout ( x )


class _OutputDiscriminator ( nn.Module ) :
    def __init__ ( self , seq_len , alpha=1.0 ) :
        super ().__init__ ()
        self.alpha = alpha
        self.li    = nn.Sequential (
            nn.Linear    ( seq_len , 512 ) ,
            nn.BatchNorm1d ( 512 ) ,
            nn.ReLU      () ,
            nn.Linear    ( 512 , 1 ) ,
            nn.Sigmoid   () ,
        )

    def forward ( self , x ) :

        x = _ReverseLayer.apply ( x , self.alpha )
        return self.li ( x )


class _BackboneDiscriminator ( nn.Module ) :
    def __init__ ( self , seq_len , d_model , alpha=1.0 ) :
        super ().__init__ ()
        self.alpha = alpha
        self.li1   = nn.Linear ( d_model , 1 )
        self.li2   = nn.Sequential (
            nn.Linear    ( seq_len , 512 ) ,
            nn.BatchNorm1d ( 512 ) ,
            nn.ReLU      () ,
            nn.Linear    ( 512 , 1 ) ,
            nn.Sigmoid   () ,
        )

    def forward ( self , x ) :

        x    = _ReverseLayer.apply ( x , self.alpha )
        out1 = self.li1 ( x ).squeeze ( -1 )
        return self.li2 ( out1 )


class DAST_RUL ( nn.Module ) :

    def __init__ ( self , args ) :
        super ().__init__ ()
        C       = args.num_nodes
        d_model = getattr ( args , 'd_model'  , 32  )
        nhead   = getattr ( args , 'nhead'    , 8   )
        nlayers = getattr ( args , 'nlayers'  , 2   )
        dropout = getattr ( args , 'dropout'  , 0.1 )
        seq_len = args.input_length

        self.input_proj  = nn.Linear ( C , d_model )
        self.pos_enc     = _PositionalEncoding ( d_model , dropout=dropout , max_len=seq_len + 10 )
        enc_layer        = nn.TransformerEncoderLayer (
            d_model , nhead , dim_feedforward=512 ,
            dropout=dropout )
        self.transformer = nn.TransformerEncoder ( enc_layer , nlayers )
        self.drop        = nn.Dropout ( dropout )
        self.regressor   = nn.Linear ( d_model , 1 )


        self.D1 = _OutputDiscriminator  ( seq_len )
        self.D2 = _BackboneDiscriminator ( seq_len , d_model )

    def forward ( self , x , OCC=None ) :

        h       = self.input_proj ( x )
        h       = self.pos_enc ( h )
        feat    = self.transformer ( h )
        feat    = self.drop ( feat )
        out_seq = self.regressor ( feat )
        pred    = out_seq [ : , -1 , : ]

        return None , {
            'gamma'   : pred ,
            'out_seq' : out_seq ,
            'h'       : feat ,
        }
