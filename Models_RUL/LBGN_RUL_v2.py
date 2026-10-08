"""
LBGN_RUL_v2 -- candidate next version of the reference model, with the
spectral view's one-step Laplacian diffusion pass removed (kappa*Lz
smoothing + alpha*r(z) residual correction, reused from DLGNet's SASRLayer
in v1's SpectralGraphLayer).

Ablation evidence (--lbgn_ablation no_spectral_diffusion, see
analysis/build_ablation_results_table.py output) shows removing this step
does not hurt -- it had the best RMSE of the entire ablation table, beating
even the reference model -- while closing off a "just incremental to
DLGNet" reviewer concern, since that diffusion formula was reused verbatim
from DLGNet's SASRLayer.

Kept as a SEPARATE file from LBGN_RUL.py (the current, checkpoint-compatible
main-table reference model) on purpose: this is a candidate, not yet a
replacement. LBGN_RUL.py's SpectralGraphLayer is untouched, so every
existing CMAPSS/N-CMAPSS checkpoint stays valid. Promoting v2 to be THE
reference model later means retraining the main campaigns from scratch,
since v1's kappa/alpha/residual_net weights are tuned assuming the
diffusion computation consumes them -- skipping that computation with those
same weights loaded would silently change the model's behavior, not
preserve it.

Depends on LBGN_RUL.py for every OTHER shared building block (unchanged) --
same "depend on the reference file for shared pieces" pattern
LBGN_RUL_ablation.py already uses. Only the spectral view differs, and it's
genuinely leaner here (no kappa/alpha/residual_net/band-coherence at all),
not just skipping dead computation the way the ablation's
SpectralGraphLayerNoDiffusion does (which inherits v1's unused diffusion
parameters for checkpoint-shape compatibility with the other RQ5 variants).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Models_RUL.LBGN_RUL import (
    SensorMessagePassing ,
    NodeFeatureProjection ,
    LifecycleModulator ,
    LifecycleAdjacency ,
    HopwiseLifecycleFiLM ,
    TimeViewGraphConvFiLMHeavy ,
    AdditiveGatedFusion ,
    lbgn_mar_loss ,
    lbgn_adjacency_mmd ,
)


class SpectralGraphLayerV2 ( nn.Module ) :
    """
    Spectral graph view over N sensors -- diffusion-free.

    Node features : per-sensor log-band energies (B, C, n_bands)
    OCC is applied as lightweight FiLM directly on the band-embedded
    features. No Laplacian diffusion pass beforehand (unlike v1's
    SpectralGraphLayer) -- the band-coherence adjacency is not computed at
    all here, since nothing in this forward path would consume it.

    x       : (B, C, N)        raw windowed sensor signals
    occ_emb : (B, C, LCE_dim)  lifecycle embedding (optional)
    out     : (B, C, h_dim)
    """
    def __init__ ( self , node_count , h_dim , n_bands=4 , LCE_dim=16 ) :
        super ().__init__ ()
        self.n_bands = n_bands
        self.band_embed  = nn.Linear ( n_bands , n_bands )
        self.occ_film    = nn.Linear ( LCE_dim , 2 * n_bands )
        self.output_proj = nn.Linear ( n_bands , h_dim )

    def _band_energy ( self , x ) :
        """x: (B, C, N)  →  (B, C, n_bands)  log-scale band power."""
        X        = torch.fft.rfft ( x , dim=-1 )
        power    = ( X * X.conj () ).real
        F_len    = power.shape[-1]
        bin_size = max ( 1 , F_len // self.n_bands )
        bands    = [
            power [ ... , k * bin_size : (k + 1) * bin_size ].mean ( dim=-1 )
            for k in range ( self.n_bands )
        ]
        return torch.log1p ( torch.stack ( bands , dim=-1 ).clamp ( min=0 ) )

    def forward ( self , x , occ_emb=None ) :
        z_out = self.band_embed ( self._band_energy ( x ) )

        if occ_emb is not None :
            film    = self.occ_film ( occ_emb )
            sg , sb = film.chunk ( 2 , dim=-1 )
            z_out = ( 1 + sg ) * z_out + sb

        return self.output_proj ( z_out )


class LBGN_RUL_v2 ( nn.Module ) :
    """
    Candidate next version of LBGN_RUL: identical to the reference model
    except the spectral view's one-step Laplacian diffusion pass is removed
    entirely (see module docstring). Everything else -- lifecycle band
    adjacency, depth-wise FiLM (time view), spectral OCC-FiLM, additive
    gated fusion, output head -- is unchanged from LBGN_RUL.

    Training: same da_mode='lbgn' path as LBGN_RUL (Experiment.training_da
    dispatches on model_name == 'LBGN_RUL' for the loss functions; this
    class reuses LBGN_RUL.py's lbgn_mar_loss / lbgn_adjacency_mmd directly).
    """
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

        n_bands = getattr ( args , 'n_bands' , 4 )
        self.lifecycle_adjacency = LifecycleAdjacency (
            self.sequence_len , self.LCE_dim , self.dropout , n_bands = n_bands )

        self.time_view = TimeViewGraphConvFiLMHeavy (
            self.sequence_len , self.hidden_dim , self.dropout , self.LCE_dim ,
            support_len = self.supports_len , hop = self.hop )

        self.spectral_view = SpectralGraphLayerV2 (
            node_count = self.num_sensors ,
            h_dim      = self.hidden_dim ,
            n_bands    = n_bands ,
            LCE_dim    = self.LCE_dim )

        self.fusion    = AdditiveGatedFusion ( self.hidden_dim )
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

        h_spec = self.spectral_view ( x , occ_emb )

        h = self.fuse_norm ( self.dropout ( self.fusion ( h_time , h_spec ) ) )

        fc_out = self.channel_fc ( h ).permute ( 0 , 2 , 1 )
        fc_out = self.final_fc   ( fc_out ).permute ( 0 , 2 , 1 )
        pred   = F.softplus ( fc_out.squeeze (1) )

        return None , { 'gamma' : pred , 'adj' : adj_time , 'h' : h }
