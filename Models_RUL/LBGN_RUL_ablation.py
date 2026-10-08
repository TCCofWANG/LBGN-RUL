

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_RUL.LBGN_RUL import (
    SensorMessagePassing ,
    NodeFeatureProjection ,
    LifecycleModulator ,
    LifecycleAdjacency ,
    HopwiseLifecycleFiLM ,
    SpectralGraphLayer ,
    AdditiveGatedFusion ,
)


class TimeCosineAdjacency ( nn.Module ) :

    def __init__ ( self , dropout ) :
        super ().__init__ ()
        self.dropout = dropout

    def forward ( self , x , OCC ) :
        xc  = x - x.mean ( dim=-1 , keepdim=True )
        xn  = xc / xc.norm ( dim=-1 , keepdim=True ).clamp ( min=1e-8 )
        adj = torch.einsum ( 'bcn,bdn->bcd' , xn , xn )
        return self.dropout ( F.softmax ( adj , dim=-1 ) )


class UniformBandAdjacency ( LifecycleAdjacency ) :

    def forward ( self , x , OCC ) :
        coh = self._per_band_coherence ( x )
        return self.dropout ( F.softmax ( coh.mean ( dim=1 ) , dim=-1 ) )


class StaticBandAdjacency ( LifecycleAdjacency ) :

    def __init__ ( self , *a , **kw ) :
        super ().__init__ ( *a , **kw )
        self.static_logits = nn.Parameter ( torch.zeros ( self.n_bands ) )

    def forward ( self , x , OCC ) :
        w   = F.softmax ( self.static_logits , dim=-1 )
        coh = self._per_band_coherence ( x )
        adj = torch.einsum ( 'k,bkij->bij' , w , coh )
        return self.dropout ( F.softmax ( adj , dim=-1 ) )


class SpectralGraphLayerNoDiffusion ( SpectralGraphLayer ) :

    def forward ( self , x , occ_emb=None ) :
        z_out = self.band_embed ( self._band_energy ( x ) )

        if occ_emb is not None :
            film    = self.occ_film ( occ_emb )
            sg , sb = film.chunk ( 2 , dim=-1 )
            z_out = ( 1 + sg ) * z_out + sb

        return self.output_proj ( z_out )


class TimeViewGraphConvFiLMHeavyAblation ( nn.Module ) :

    def __init__ ( self , c_in , c_out , dropout , LCE_dim , support_len=3 , hop=2 ,
                   use_input_film=True , hop_film_mode='depthwise' ) :
        super ().__init__ ()
        self.prop           = SensorMessagePassing ()
        self.use_input_film = use_input_film
        self.hop_film_mode  = hop_film_mode
        if use_input_film :
            self.input_film = LifecycleModulator ( LCE_dim , c_in )
        if hop_film_mode == 'depthwise' :
            self.hop_film = HopwiseLifecycleFiLM ( LCE_dim , c_in , hop )
        elif hop_film_mode == 'shared' :
            self.hop_film = LifecycleModulator ( LCE_dim , c_in )
        self.project    = NodeFeatureProjection ( (hop * support_len + 1) * c_in , c_out )
        self.dropout    = dropout
        self.hop        = hop

    def _hop_modulate ( self , x , OCC , depth ) :
        if self.hop_film_mode == 'depthwise' :
            return self.hop_film ( x , OCC , depth=depth )
        if self.hop_film_mode == 'shared' :
            return self.hop_film ( x , OCC )
        return x

    def forward ( self , x , supports , OCC ) :
        if self.use_input_film :
            x = self.input_film ( x , OCC )
        out = [x]
        for A in supports :
            xk = self.dropout ( self._hop_modulate ( self.prop ( x , A ) , OCC , depth=1 ) )
            out.append ( xk )
            for depth in range ( 2 , self.hop + 1 ) :
                xk = self.dropout ( self._hop_modulate ( self.prop ( xk , A ) , OCC , depth=depth ) )
                out.append ( xk )
        return self.project ( torch.cat ( out , dim=1 ) )


class LBGN_RUL_ablation ( nn.Module ) :

    def __init__ ( self , args , supports=None ) :
        super ().__init__ ()
        self.num_sensors  = args.num_nodes
        self.sequence_len = args.input_length
        self.hidden_dim   = args.hidden_dim
        self.dropout      = nn.Dropout ( args.dropout )
        self.LCE_dim      = getattr ( args , "LCE_dim" , 16 )
        self.hop          = getattr ( args , "hop"     , 2  )

        supports = supports if supports is not None else []
        self._num_static_supports = len ( supports )
        for i , s in enumerate ( supports ) :
            self.register_buffer ( f"static_support_{i}" , s )
        self.supports_len = self._num_static_supports + 1

        self.occ_projection = nn.Linear ( self.sequence_len , self.LCE_dim )


        self.ablation = getattr ( args , 'lbgn_ablation' , 'none' )
        n_bands = getattr ( args , 'n_bands' , 4 )

        if self.ablation == 'time_adj' :
            self.lifecycle_adjacency = TimeCosineAdjacency ( self.dropout )
        elif self.ablation in ( 'uniform_band' , 'no_lifecycle' , 'uniform_band_no_spectral_diffusion' ) :
            self.lifecycle_adjacency = UniformBandAdjacency (
                self.sequence_len , self.LCE_dim , self.dropout , n_bands = n_bands )
        elif self.ablation == 'static_band' :
            self.lifecycle_adjacency = StaticBandAdjacency (
                self.sequence_len , self.LCE_dim , self.dropout , n_bands = n_bands )
        else :
            self.lifecycle_adjacency = LifecycleAdjacency (
                self.sequence_len , self.LCE_dim , self.dropout , n_bands = n_bands )

        film_kw = { 'use_input_film' : self.ablation not in ( 'no_input_film' , 'no_film' ) ,
                    'hop_film_mode'  : ( 'shared' if self.ablation == 'shared_film'
                                         else 'none' if self.ablation == 'no_film'
                                         else 'depthwise' ) }

        self.time_view = TimeViewGraphConvFiLMHeavyAblation (
            self.sequence_len , self.hidden_dim , self.dropout , self.LCE_dim ,
            support_len = self.supports_len , hop = self.hop , **film_kw )


        if self.ablation == 'no_spectral_view' :
            self.spectral_view = None
            self.fusion         = None
        else :
            spectral_cls = SpectralGraphLayerNoDiffusion\
                           if self.ablation in ( 'no_spectral_diffusion' , 'uniform_band_no_spectral_diffusion' )\
                           else SpectralGraphLayer
            self.spectral_view = spectral_cls (
                node_count = self.num_sensors ,
                h_dim      = self.hidden_dim ,
                n_bands    = n_bands ,
                LCE_dim    = self.LCE_dim )
            self.fusion = AdditiveGatedFusion ( self.hidden_dim )

        self.fuse_norm = nn.LayerNorm ( self.hidden_dim )

        self.channel_fc = nn.Sequential (
            nn.Linear ( self.hidden_dim , self.hidden_dim ) , nn.ELU () ,
            nn.Linear ( self.hidden_dim , 1 ) )
        self.final_fc = nn.Sequential (
            nn.Linear ( self.num_sensors , self.num_sensors ) , nn.ELU () ,
            nn.Linear ( self.num_sensors , 1 ) )

    @property
    def supports ( self ) :
        return [ getattr ( self , f"static_support_{i}" ) for i in range ( self._num_static_supports ) ]

    def forward ( self , x_in , OCC=None ) :
        B , N , C = x_in.shape
        x = x_in.permute ( 0 , 2 , 1 )

        occ_emb = self.occ_projection (
            OCC.transpose ( 1 , 2 ).to ( x_in.device ) )
        occ_emb = occ_emb.expand ( -1 , C , -1 )

        adj_time     = self.lifecycle_adjacency ( x , occ_emb )
        all_supports = self.supports + [adj_time]
        h_time = self.time_view (
            x.permute ( 0 , 2 , 1 ) , all_supports , occ_emb
        ).permute ( 0 , 2 , 1 )

        if self.ablation == 'no_spectral_view' :


            h = self.fuse_norm ( self.dropout ( h_time ) )
        else :


            spec_occ = None if self.ablation in ( 'no_lifecycle' , 'no_spectral_film' ) else occ_emb
            h_spec = self.spectral_view ( x , spec_occ )
            h = self.fuse_norm ( self.dropout ( self.fusion ( h_time , h_spec ) ) )

        fc_out = self.channel_fc ( h ).permute ( 0 , 2 , 1 )
        fc_out = self.final_fc   ( fc_out ).permute ( 0 , 2 , 1 )
        pred   = F.softplus ( fc_out.squeeze (1) )

        return None , { 'gamma' : pred , 'adj' : adj_time , 'h' : h }
