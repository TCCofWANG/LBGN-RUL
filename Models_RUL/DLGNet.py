import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SASRLayer ( nn.Module ) :
    def __init__(self , node_count , h_dim ,z_dim, kappa=0.1 , alpha=0.05) :
        super ( SASRLayer , self ).__init__ ( )
        self.node_count = node_count
        self.z_dim = z_dim
        self._kappa = nn.Parameter ( torch.tensor ( kappa ) )
        self._alpha = nn.Parameter ( torch.tensor ( alpha ) )


        self.r_phi = nn.Sequential ( nn.Linear ( self.z_dim , int(h_dim/2) ) , nn.ELU ( ) , nn.Linear ( int(h_dim/2) , self.z_dim ) )
        self.proj = nn.Linear ( self.z_dim , h_dim )

    def forward(self , A , dyn_prior) :

        B , C , z_dim = dyn_prior.shape


        M = A.shape[1]
        z = dyn_prior.unsqueeze(1).expand(-1, M, -1, -1)


        Az = torch.einsum('bmcd,bmde->bmce', A, z)
        Lz = (z - Az).mean(dim=1)

        kappa = F.softplus ( self._kappa )
        alpha = torch.sigmoid(self._alpha)

        r_out = self.r_phi ( dyn_prior.reshape(-1, self.z_dim) ).view(B, C, z_dim)
        z_new = dyn_prior - kappa * Lz + alpha * r_out

        corr = self.proj ( z_new )
        return corr

class ACE ( nn.Module ) :
    def __init__(self , sequence_len , LCE_dim ,dropout, single_path=True) :
        super ( ).__init__ ( )
        self.sequence_len = sequence_len
        self.LCE_dim = LCE_dim
        self.dropout = dropout


        self.ll1 = nn.Linear ( self.sequence_len , self.LCE_dim )
        self.ll2 = nn.Linear ( self.sequence_len , self.LCE_dim )
        self.m_gate1 = nn.Sequential ( nn.Linear ( self.sequence_len + self.LCE_dim , 1 ) , nn.ELU ( ) )
        self.m_gate2 = nn.Sequential ( nn.Linear ( self.sequence_len + self.LCE_dim , 1 ) , nn.ELU ( ) )

    def forward(self , x , AATE ) :

        m = self.m_gate1( torch.cat ( [x , AATE.permute ( 0 , 2 , 1 , 3 )] , dim = -1 ) )
        e = self.dropout ( F.softmax (  ( m * self.ll1 ( x ) ) , dim = -1 ) )
        e = AATE + e.permute ( 0 , 2 , 1 , 3 )

        eT = e.permute ( 0 , 1 , 3 , 2 )

        attn = torch.matmul ( e , eT )/ math.sqrt(self.LCE_dim)
        A = F.softmax(attn, dim=-1)

        return self.dropout(A)


class nconv ( nn.Module ) :
    def __init__(self) : super ( ).__init__ ( )

    def forward(self , x , A) :
        return torch.einsum ( 'bfnm,bmnv->bfvm' , (x , A) ).contiguous ( )


class linear ( nn.Module ) :
    def __init__(self , c_in , c_out) :
        super ( ).__init__ ( )
        self.mlp = nn.Conv2d ( c_in , c_out , kernel_size = (1 , 1) )

    def forward(self , x) : return self.mlp ( x )


class RGCN ( nn.Module ) :
    def __init__(self , c_in , c_out , dropout , feature_num , support_len=3 , order=2) :
        super ( ).__init__ ( )
        self.nconv = nconv ( )
        self.mlp = linear ( (order * support_len + 1) * c_in , c_out )
        self.dropout = dropout
        self.order = order

    def forward(self , x , supports ) :
        out = [x]
        for b in supports :
            x1 = self.nconv ( x , b )
            out.append ( x1 )
            for k in range ( 2 , self.order + 1 ) :
                x2 = self.nconv ( x1 , b )
                out.append ( x2 )
                x1 = x2

        h = self.mlp ( torch.cat ( out , dim = 1 ) )
        return h


class DLGNet ( nn.Module ) :
    def __init__(self , args , supports=None) :
        super ( ).__init__ ( )
        self.device = torch.device ( "cuda" if torch.cuda.is_available ( ) else "cpu" )
        self.feature_num = args.num_nodes
        self.sequence_len = args.input_length
        self.hidden_dim = args.hidden_dim
        self.dropout = nn.Dropout ( args.dropout )
        self.LCE_dim = args.LCE_dim
        self.z_dim = getattr ( args , "z_dim" , 1 )
        self.hop = getattr ( args , "hop" , 1 )
        self.M = getattr ( args , "M" , 1 )
        self.supports = supports if supports is not None else []
        self.supports_len = len ( self.supports ) + 1


        self.use_dynamic_encoder = getattr ( args , "use_dynamic_encoder" , True )
        if self.use_dynamic_encoder :
            self.dynamic_encoder = nn.Sequential (
                nn.Conv1d ( self.feature_num , self.hidden_dim , kernel_size = 3 , padding = 1 ) , nn.ELU ( ) ,
                nn.Conv1d ( self.hidden_dim , self.feature_num , kernel_size = 1 ) , nn.AdaptiveAvgPool1d ( self.z_dim ) )


        self.ACE = ACE(self.sequence_len, self.LCE_dim, self.dropout)

        self.rgcn = RGCN ( self.sequence_len , self.hidden_dim , self.dropout , self.feature_num ,
                                            support_len = self.supports_len , order = self.hop )
        self.sasr = SASRLayer ( node_count = self.feature_num , h_dim = self.hidden_dim, z_dim = self.z_dim )

        self.project_ate_dim = nn.Linear ( self.sequence_len , self.LCE_dim )
        self.fc = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.Sigmoid())

        self.channel_fc = nn.Sequential ( nn.Linear ( self.hidden_dim , self.hidden_dim ) , nn.ELU ( ) ,
                                       nn.Linear ( self.hidden_dim , 1 ) )
        self.final_fc =  nn.Sequential ( nn.Linear ( self.feature_num , self.feature_num ) , nn.ELU ( ) ,
                                       nn.Linear ( self.feature_num , 1 ) )


    def forward(self , x_in , OCC=None) :
        B , N , C = x_in.shape
        x = x_in.permute ( 0 , 2 , 1 ).unsqueeze ( 2 ).expand ( -1 , -1 , self.M , -1 )
        OCC = self.project_ate_dim ( OCC.transpose ( 1 , 2 ).to ( self.device ) )
        OCC = OCC.unsqueeze(1).expand(-1, self.M, C, -1)


        dyn_corr = None
        if self.use_dynamic_encoder :
            dyn_corr = self.dynamic_encoder ( x_in.permute ( 0 , 2 , 1 ) )

        adj = self.ACE( x, OCC)
        new_supports = self.supports + [adj]

        h_gcn = self.rgcn( x.permute ( 0 , 3 , 1 , 2 ) , new_supports ).permute ( 0 , 3 , 2 , 1 ).mean ( dim = 1 )
        h_diff = self.sasr(adj, dyn_corr)
        gate = self.fc(h_gcn + h_diff)
        h= gate * h_gcn + (1 - gate) * h_diff
        h = self.dropout(h)


        fc_output = self.channel_fc( h ).permute( 0, 2, 1 )
        fc_output = self.final_fc( fc_output ).permute( 0, 2, 1 )

        rul_prediction = F.softplus(fc_output.squeeze(1))

        return None , rul_prediction
