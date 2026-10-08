

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_RUL.PDMN import PDMN


class DamageClock ( nn.Module ) :

    def __init__ ( self , feature_num , hidden=16 ) :
        super ().__init__ ()
        self.rate_net = nn.Sequential (
            nn.Linear ( feature_num , hidden ) , nn.ELU () ,
            nn.Linear ( hidden , 1 ) )


        nn.init.zeros_ ( self.rate_net [ -1 ].weight )
        nn.init.constant_ ( self.rate_net [ -1 ].bias , 0.5414 )

    def forward ( self , x , OCC ) :

        rate = F.softplus ( self.rate_net ( x ) )


        d_occ = torch.diff ( OCC , dim=1 ,
                             prepend=OCC [ : , :1 ] - ( OCC [ : , 1:2 ] - OCC [ : , :1 ] ) )
        d_occ = d_occ.clamp ( min=0.0 )


        window_damage = torch.cumsum ( rate * d_occ , dim=1 )


        hist_damage = OCC [ : , :1 ] * rate.mean ( dim=1 , keepdim=True )

        tau = hist_damage + window_damage
        return tau , rate


class NDC_PDMN ( PDMN ) :

    def __init__ ( self , args , supports=None ) :
        super ().__init__ ( args , supports )
        self.clock = DamageClock (
            args.num_nodes , hidden=getattr ( args , 'ndc_hidden' , 16 ) )


        self.ndc_ablate_rate1 = bool ( getattr ( args , 'ndc_ablate_rate1' , False ) )

    def forward ( self , x_in , OCC=None ) :
        OCC = OCC.to ( x_in.device )
        if self.ndc_ablate_rate1 :
            rate = torch.ones_like ( OCC )
            tau  = OCC
        else :
            tau , rate = self.clock ( x_in , OCC )
        _ , out    = super ().forward ( x_in , tau )
        out [ 'tau'  ] = tau
        out [ 'rate' ] = rate
        return None , out


def ndc_calibration_loss ( rate ) :

    return ( rate.mean () - 1.0 ).pow ( 2 )
