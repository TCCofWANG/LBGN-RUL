"""
LBGN: Lifecycle Band Graph Network.

Novel contributions vs the AIDGN -> DLGNet lineage this model descends from:

  1. Spectral band coherence adjacency -- sensor coupling is computed from
     FFT cross-coherence in discrete frequency bands (n_bands bands via
     torch.fft.rfft), with OCC selecting which band dominates the adjacency
     matrix.  DLGNet's ACE uses attention in node-feature space; its SASRLayer
     is graph-Laplacian smoothing (z - kappa*Lz + alpha*r(z)) -- neither
     touches the frequency domain.  The FFT-based band-selective adjacency is
     new to this lineage.

  2. Input FiLM -- x is OCC-modulated before any propagation, so lifecycle
     stage shapes the raw per-sensor window, not only propagated versions.

  3. Depth-wise independent FiLM -- each K-hop depth gets its own FiLM
     parameters (gamma_k, beta_k), so the degree of lifecycle amplification
     can vary with neighbourhood reach.  DLGNet has no FiLM anywhere.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorMessagePassing ( nn.Module ) :
    """Single batched graph-propagation step:  x' = A . x."""
    def __init__(self) : super ().__init__ ()

    def forward ( self , x , A ) :
        """x: (B, f, C)   A: (B, C, C)  ->  (B, f, C)"""
        return torch.einsum ( 'bfn,bnv->bfv' , x , A ).contiguous ()


class NodeFeatureProjection ( nn.Module ) :
    """Point-wise Conv1d used as a node-wise feature mixer after K-hop stacking."""
    def __init__ ( self , c_in , c_out ) :
        super ().__init__ ()
        self.conv = nn.Conv1d ( c_in , c_out , kernel_size=1 )

    def forward ( self , x ) : return self.conv ( x )


class LifecycleAdjacency ( nn.Module ) :
    """
    Builds a sensor adjacency matrix from OCC-selected spectral band coherence.

    Motivation
    ----------
    Sensor co-variation can occur through slow drifts, transients, or
    oscillatory dynamics, which occupy different regions of the frequency
    spectrum, and the distribution of coupling across these regions can
    change as degradation progresses.  A full-band time-domain adjacency
    aggregates this structure away before any lifecycle conditioning is
    applied.  Which band matters at which lifecycle stage is LEARNED
    (softmax over bands from OCC), not assumed -- if coupling carries no
    band structure the weights can recover the uniform mix.

    Here we compute per-band cross-coherence matrices (one C×C matrix per
    frequency band) and let the OCC embedding produce a soft attention weight
    over bands.  The adjacency is then a lifecycle-weighted sum of those
    band-coherence matrices — the engine's age selects *which spectral coupling
    channel* to look through.

    Contrast with SpectralGraphLayer: that view applies OCC post-hoc via FiLM
    on node features after equal-weight coherence diffusion.  This view applies
    OCC ante-hoc — it shapes the adjacency itself before any propagation.

    x   : (B, C, N)        windowed sensor signals
    OCC : (B, C, LCE_dim)  lifecycle embedding
    out : (B, C, C)        row-softmax adjacency
    """
    def __init__ ( self , sequence_len , LCE_dim , dropout , n_bands=4 ) :
        super ().__init__ ()
        self.n_bands      = n_bands
        self.band_select  = nn.Linear ( LCE_dim , n_bands )
        self.dropout      = dropout

    def _per_band_coherence ( self , x ) :
        """x: (B, C, N)  →  (B, n_bands, C, C)  normalised coherence per band."""
        X        = torch.fft.rfft ( x , dim=-1 )
        F_len    = X.shape[-1]
        bin_size = max ( 1 , F_len // self.n_bands )
        matrices = []
        for k in range ( self.n_bands ) :
            Xb    = X [ ... , k * bin_size : (k + 1) * bin_size ]
            cross = torch.einsum ( 'bif,bjf->bij' , Xb , Xb.conj () ).abs ()
            auto  = ( Xb * Xb.conj () ).real.sum ( dim=-1 )
            denom = ( auto.unsqueeze (2) * auto.unsqueeze (1) ).clamp ( min=1e-8 ).sqrt ()
            matrices.append ( cross / denom )
        return torch.stack ( matrices , dim=1 )

    def forward ( self , x , OCC ) :
        occ_global   = OCC.mean ( dim=1 )
        band_w       = F.softmax ( self.band_select ( occ_global ) , dim=-1 )
        coh          = self._per_band_coherence ( x )
        adj          = torch.einsum ( 'bk,bkij->bij' , band_w , coh )
        return self.dropout ( F.softmax ( adj , dim=-1 ) )


class LifecycleModulator ( nn.Module ) :
    """
    Residual FiLM (Feature-wise Linear Modulation) conditioned on OCC.

        x_out = (1 + gamma(OCC)) * x + beta(OCC)

    The residual form (1 + gamma) initialises as identity when gamma, beta
    are close to 0, so the network starts from the un-modulated hop output
    and learns to amplify or suppress features according to lifecycle stage.

    x   : (B, feature_dim, C)
    OCC : (B, C, LCE_dim)
    out : (B, feature_dim, C)
    """
    def __init__ ( self , LCE_dim , feature_dim ) :
        super ().__init__ ()
        self.proj = nn.Linear ( LCE_dim , 2 * feature_dim )

    def forward ( self , x , OCC ) :
        film         = self.proj ( OCC )
        gamma , beta = film.chunk ( 2 , dim=-1 )
        gamma = gamma.permute ( 0 , 2 , 1 )
        beta  = beta.permute  ( 0 , 2 , 1 )
        return ( 1 + gamma ) * x + beta


class HopwiseLifecycleFiLM ( nn.Module ) :
    """Depth-indexed FiLM: an independent (gamma, beta) generator per hop distance."""
    def __init__ ( self , LCE_dim , feature_dim , hop ) :
        super ().__init__ ()
        self.films = nn.ModuleList ( [ LifecycleModulator ( LCE_dim , feature_dim ) for _ in range ( hop ) ] )

    def forward ( self , x , OCC , depth ) :
        """depth: 1-indexed hop distance."""
        return self.films [ depth - 1 ] ( x , OCC )


class TimeViewGraphConvFiLMHeavy ( nn.Module ) :
    """
    Time-view graph convolution with an input-stage FiLM before propagation,
    plus an independent FiLM at every hop depth (rather than one shared
    across all depths).

    x        : (B, c_in, C)        windowed sensor features
    supports : list[(B, C, C)]     adjacency matrices
    OCC      : (B, C, LCE_dim)     lifecycle embedding
    out      : (B, c_out, C)
    """
    def __init__ ( self , c_in , c_out , dropout , LCE_dim , support_len=3 , hop=2 ) :
        super ().__init__ ()
        self.prop        = SensorMessagePassing ()
        self.input_film  = LifecycleModulator ( LCE_dim , c_in )
        self.hop_film    = HopwiseLifecycleFiLM ( LCE_dim , c_in , hop )
        self.project     = NodeFeatureProjection ( (hop * support_len + 1) * c_in , c_out )
        self.dropout     = dropout
        self.hop         = hop

    def forward ( self , x , supports , OCC ) :
        x = self.input_film ( x , OCC )
        out = [x]
        for A in supports :
            xk = self.dropout ( self.hop_film ( self.prop ( x , A ) , OCC , depth=1 ) )
            out.append ( xk )
            for depth in range ( 2 , self.hop + 1 ) :
                xk = self.dropout ( self.hop_film ( self.prop ( xk , A ) , OCC , depth=depth ) )
                out.append ( xk )
        return self.project ( torch.cat ( out , dim=1 ) )


class SpectralGraphLayer ( nn.Module ) :
    """
    Spectral graph view over N sensors.

    Node features  : per-sensor log-band energies  (B, C, n_bands)
    Adjacency      : band-coherence matrix          (B, C, C)

    A graph-diffusion pass (Laplacian smoothing + residual correction) is run
    in n_bands-dimensional space.  OCC is applied as lightweight FiLM after
    diffusion so the spectral view shares the same lifecycle conditioning as
    the time view.

    x       : (B, C, N)        raw windowed sensor signals
    occ_emb : (B, C, LCE_dim)  lifecycle embedding (optional)
    out     : (B, C, h_dim)
    """
    def __init__ ( self , node_count , h_dim , n_bands=4 , LCE_dim=16 , kappa=0.1 , alpha=0.05 ) :
        super ().__init__ ()
        self.n_bands = n_bands
        self._kappa  = nn.Parameter ( torch.tensor ( kappa ) )
        self._alpha  = nn.Parameter ( torch.tensor ( alpha ) )

        half = int ( h_dim / 2 )
        self.band_embed   = nn.Linear ( n_bands , n_bands )
        self.residual_net = nn.Sequential (
            nn.Linear ( n_bands , half ) , nn.ELU () ,
            nn.Linear ( half , n_bands ) )

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

    def _band_coherence ( self , x ) :
        """x: (B, C, N)  →  (B, C, C)  row-softmax normalised coherence."""
        X     = torch.fft.rfft ( x , dim=-1 )
        cross = torch.einsum ( 'bif,bjf->bij' , X , X.conj () ).abs ()
        auto  = ( X * X.conj () ).real.sum ( dim=-1 )
        denom = ( auto.unsqueeze (2) * auto.unsqueeze (1) ).clamp ( min=1e-8 ).sqrt ()
        return F.softmax ( cross / denom , dim=-1 )

    def forward ( self , x , occ_emb=None ) :
        B , C , _ = x.shape
        z     = self.band_embed ( self._band_energy ( x ) )
        A     = self._band_coherence ( x )
        kappa = F.softplus   ( self._kappa )
        alpha = torch.sigmoid ( self._alpha )
        Az    = torch.einsum ( 'bcd,bde->bce' , A , z )
        Lz    = z - Az
        r     = self.residual_net ( z.reshape ( -1 , self.n_bands ) ).view ( B , C , self.n_bands )
        z_out = z - kappa * Lz + alpha * r


        if occ_emb is not None :
            film           = self.occ_film ( occ_emb )
            sg , sb        = film.chunk ( 2 , dim=-1 )
            z_out = ( 1 + sg ) * z_out + sb

        return self.output_proj ( z_out )


class AdditiveGatedFusion ( nn.Module ) :
    """
    Fuses time-view and spectral-view features with an additive gate.

        gate = sigmoid( Linear( h_time + h_spec ) )   in (0,1)^h_dim
        h    = gate * h_time  +  (1 − gate) * h_spec

    The element-wise sum h_time + h_spec gives the gate an implicit conflict
    signal: when both views agree on a feature (same sign / magnitude) the sum
    amplifies it and the gate is pushed decisively toward 0 or 1; when they
    disagree the sum cancels and the gate stays near 0.5, blending both views.
    This inductive bias is exactly what concatenation-based gates lack while
    costing twice as many parameters.
    """
    def __init__ ( self , h_dim ) :
        super ().__init__ ()
        self.gate = nn.Sequential ( nn.Linear ( h_dim , h_dim ) , nn.Sigmoid () )

    def forward ( self , h_time , h_spec ) :
        """h_time, h_spec : (B, C, h_dim)  →  (B, C, h_dim)"""
        gate = self.gate ( h_time + h_spec )
        return gate * h_time + ( 1 - gate ) * h_spec


class LBGN_RUL ( nn.Module ) :
    """
    Reference (non-ablated) LBGN architecture: OCC-conditioned band-coherence
    adjacency, input + depth-wise FiLM time view, spectral view, additive
    gated fusion, channel-then-sensor regression head.

    Training: da_mode='lbgn' (Experiment.training_da), using this
    file's own lbgn_mar_loss / lbgn_adjacency_mmd (below) for the MAR +
    staged-MMD domain-adaptation terms.
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


        self.spectral_view = SpectralGraphLayer (
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


def _rbf_mmd ( s , t ) :
    """Unbiased RBF-MMD with median-heuristic bandwidth.  s,t: (B, D)."""
    with torch.no_grad () :
        sigma = torch.cdist ( torch.cat ( [s,t] ) , torch.cat ( [s,t] ) ).median ().clamp ( min=1e-3 )
    def rbf ( x , y ) :
        return torch.exp ( -(x.unsqueeze(1)-y.unsqueeze(0)).pow(2).sum(-1) / (2*sigma**2) )
    return rbf(s,s).mean() + rbf(t,t).mean() - 2*rbf(s,t).mean()


def lbgn_adjacency_mmd ( adj_s , occ_s , adj_t , occ_t , n_stages=3 ) :
    """
    OCC-Staged Adjacency MMD  (lifecycle-matched domain alignment).

    Working assumption (motivates the design; validated empirically, not
    claimed as physical law): coupling STRUCTURE at a given lifecycle
    stage is more comparable across operating-condition domains than raw
    signal distributions are, so alignment should match like with like.

    Global MMD fails here because it aligns adjacency matrices without regard
    to lifecycle stage: a late-life source batch aligned with an early-life
    target batch forces wrong sensor-coupling correspondences, actively hurting
    prediction.

    OCC-staged MMD fixes this by partitioning each batch into n_stages lifecycle
    bins (quantile-based so bins are always populated) and computing MMD only
    between source and target samples at the *same* lifecycle stage.

    adj_s  : (B_s, C, C)  source lifecycle adjacency
    occ_s  : (B_s,)       source mean OCC per sample
    adj_t  : (B_t, C, C)  target lifecycle adjacency
    occ_t  : (B_t,)       target mean OCC per sample
    """
    C    = adj_s.shape[1]
    mask = ~torch.eye ( C , dtype=torch.bool , device=adj_s.device )
    s_feat = adj_s [ : , mask ].float ()
    t_feat = adj_t [ : , mask ].float ()


    occ_all = torch.cat ( [ occ_s , occ_t ] )
    q       = torch.linspace ( 0 , 1 , n_stages + 1 , device=occ_all.device )
    edges   = torch.quantile ( occ_all , q )

    total , count = torch.tensor ( 0.0 , device=adj_s.device ) , 0
    for k in range ( n_stages ) :
        lo , hi = edges[k] , edges[k+1]
        ms = ( occ_s >= lo ) & ( occ_s <= hi )
        mt = ( occ_t >= lo ) & ( occ_t <= hi )
        if ms.sum () >= 2 and mt.sum () >= 2 :
            total = total + _rbf_mmd ( s_feat[ms] , t_feat[mt] )
            count += 1

    return total / max ( count , 1 )


def lbgn_mar_loss ( adj_batch , occ_scalar , margin=0.05 ) :
    """
    Monotonicity prior (soft regularizer, not a hard physical law):
    sensor coupling is encouraged to be non-decreasing with lifecycle
    age (OCC).  Rationale: in run-to-failure trajectories without
    maintenance events, degradation tends to propagate across subsystems,
    so growing off-diagonal adjacency mass is a reasonable prior — but it
    is enforced only as a hinge penalty the data can override.

    For every pair (i, j) in the batch where OCC_i > OCC_j by at least
    `margin` FRACTION OF THE BATCH'S OWN OCC SPREAD, we penalise if
    coupling_i < coupling_j. Relative (not absolute) margin so this is
    invariant to whatever units/scale OCC happens to be encoded in --
    CMAPSS's loader emits OCC pre-scaled by its own `epsilon` (e.g. 1e-2),
    so an absolute margin=0.05 would silently never fire (occ_diff can
    never exceed a 0.05 threshold when the whole batch spans ~0.01-0.02).

    adj_batch  : (B, C, C)  lifecycle adjacency matrices (from forward pass)
    occ_scalar : (B,)       mean OCC per sample
    margin     : float      minimum OCC gap, as a fraction of this batch's
                            occ_scalar range, to enforce the constraint
                            (avoids penalising near-identical lifecycle stages)
    """
    B , C , _ = adj_batch.shape

    eye_mask = torch.eye ( C , dtype=torch.bool , device=adj_batch.device )
    coupling = adj_batch.masked_fill ( eye_mask.unsqueeze (0) , 0.0 ).sum ( dim=(-1,-2) )
    coupling = coupling / ( C * ( C - 1 ) )


    occ_diff  = occ_scalar.unsqueeze (1) - occ_scalar.unsqueeze (0)
    coup_diff = coupling.unsqueeze  (1) - coupling.unsqueeze  (0)


    occ_range  = ( occ_scalar.max () - occ_scalar.min () ).clamp ( min=1e-8 )
    older_mask = ( occ_diff > margin * occ_range ).float ()
    violation  = F.relu ( -coup_diff ) * older_mask

    return violation.mean ()
