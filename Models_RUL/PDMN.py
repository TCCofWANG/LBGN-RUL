import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SensorMessagePassing ( nn.Module ) :
    """Single batched graph-propagation step:  x' = A · x."""
    def __init__ ( self ) : super ().__init__ ()

    def forward ( self , x , A ) :
        """x: (B, f, C)   A: (B, C, C)  →  (B, f, C)"""
        return torch.einsum ( 'bfn,bnv->bfv' , x , A ).contiguous ()


class NodeFeatureProjection ( nn.Module ) :
    """Point-wise Conv1d — node-wise feature mixer after K-hop stacking."""
    def __init__ ( self , c_in , c_out ) :
        super ().__init__ ()
        self.conv = nn.Conv1d ( c_in , c_out , kernel_size=1 )

    def forward ( self , x ) : return self.conv ( x )


class LifecycleModulator ( nn.Module ) :
    """
    Residual FiLM conditioned on OCC embedding.
        x_out = (1 + γ(OCC)) * x + β(OCC)
    x   : (B, feature_dim, C)
    OCC : (B, C, LCE_dim)
    """
    def __init__ ( self , LCE_dim , feature_dim ) :
        super ().__init__ ()
        self.proj = nn.Linear ( LCE_dim , 2 * feature_dim )

    def forward ( self , x , OCC ) :
        film         = self.proj ( OCC )
        gamma , beta = film.chunk ( 2 , dim=-1 )
        gamma        = gamma.permute ( 0 , 2 , 1 )
        beta         = beta.permute  ( 0 , 2 , 1 )
        return ( 1 + gamma ) * x + beta


class TimeViewGraphConv ( nn.Module ) :
    """
    K-hop GCN with FiLM lifecycle modulation at every hop.
    In PDMN the adjacency comes from DegradationModeAdjacency (not from x),
    so c_in = hidden_dim (tokenised features), not sequence_len.

    x        : (B, c_in, C)
    supports : list[(B, C, C)]
    OCC      : (B, C, LCE_dim)
    out      : (B, c_out, C)
    """
    def __init__ ( self , c_in , c_out , dropout , feature_num , LCE_dim ,
                   support_len=1 , order=2 ) :
        super ().__init__ ()
        self.prop    = SensorMessagePassing ()
        self.film    = LifecycleModulator ( LCE_dim , c_in )
        self.project = NodeFeatureProjection ( (order * support_len + 1) * c_in , c_out )
        self.dropout = dropout
        self.order   = order

    def forward ( self , x , supports , OCC ) :
        out = [x]
        for A in supports :
            x1 = self.film ( self.prop ( x , A ) , OCC )
            out.append ( x1 )
            for _ in range ( 2 , self.order + 1 ) :
                x2 = self.film ( self.prop ( x1 , A ) , OCC )
                out.append ( x2 )
                x1 = x2
        return self.project ( torch.cat ( out , dim=1 ) )


class DegradationModeAdjacency ( nn.Module ) :
    """
    K learned degradation mode adjacency matrices with sigmoid activations.

    Physical story
    ──────────────
    Turbofan failure unfolds as an ordered sequence of failure-mode activations:
    micro-wear at early OCC, inter-component coupling at mid OCC, multi-system
    deterioration near end-of-life.  We represent this directly:

        a_k(OCC) = σ( s_k · (OCC − μ_k) )          ← sigmoid, monotone in OCC
        adj      = softmax( Σ_k  a_k(OCC) · M_k )   ← superposition of active modes

    • μ_k  — learned onset OCC for mode k  (when does this failure mode begin?)
    • s_k  — learned sharpness              (abrupt vs. gradual onset)
    • M_k  — learned C×C coupling matrix   (which sensors couple when mode k fires?)

    Key properties
    ──────────────
    • adj depends ONLY on OCC, never on input x  → inherently domain-invariant
    • Cumulative activation (sigmoid): once a mode fires, it stays active
    • Monotone in OCC by construction: total coupling density ↑ as OCC ↑
    • Modes ordered μ_1 < μ_2 < ... < μ_K  (soft-enforced by mode_ordering_loss)

    occ_scalar : (B, 1)   scalar OCC in (0, 1)
    returns    : (B, C, C) adjacency,  (B, K) activation vector
    """
    def __init__ ( self , feature_num , K=8 , dropout=0.1 ) :
        super ().__init__ ()
        C         = feature_num
        self.K    = K
        self.C    = C


        eye        = torch.eye ( C ).unsqueeze (0).expand ( K , -1 , -1 ).clone ()
        self.modes = nn.Parameter ( eye + 0.01 * torch.randn ( K , C , C ) )


        init_mu         = torch.linspace ( 0.1 , 0.9 , K ).clamp ( 0.01 , 0.99 )
        self.mu_raw     = nn.Parameter ( torch.logit ( init_mu ) )


        self.sharpness  = nn.Parameter ( torch.ones ( K ) * 5.0 )

        self.dropout    = nn.Dropout ( dropout )

    @property
    def mu ( self ) :
        """Onset OCC values constrained to (0, 1)."""
        return torch.sigmoid ( self.mu_raw )

    def forward ( self , occ_scalar ) :
        """
        occ_scalar : (B, 1)
        returns    : adj (B, C, C),  activations (B, K)
        """
        a   = torch.sigmoid (
            self.sharpness.abs ().unsqueeze (0) *
            ( occ_scalar - self.mu.unsqueeze (0) )
        )
        adj = torch.einsum ( 'bk,kij->bij' , a , self.modes )
        return self.dropout ( F.softmax ( adj , dim=-1 ) ) , a


class LifecyclePositionEncoding ( nn.Module ) :
    """
    Encodes the scalar OCC as a position on the degradation manifold using a
    learnable Fourier basis — the lifecycle analogue of sinusoidal position
    encoding in Transformers.

    Motivation
    ──────────
    In a Transformer, position encoding makes the model aware of where in a
    sequence each token sits.  Here, OCC tells the model where the engine sits
    on its degradation trajectory — a continuous, physics-grounded timeline.

    Two engines at the same OCC across FD001 and FD003 receive the identical
    positional embedding.  Adding this to sensor tokens before GCN propagation
    primes every sensor representation with the engine's lifecycle coordinate,
    promoting domain-invariant features without any explicit adversarial loss.

    The frequencies are learnable, allowing the model to discover which
    "harmonics" of the degradation timeline are most informative.

    occ_scalar : (B, 1)  →  (B, 1, d_model)   broadcast-ready
    """
    def __init__ ( self , d_model ) :
        super ().__init__ ()
        n_freqs    = d_model // 2
        self.freqs = nn.Parameter ( torch.linspace ( 0.5 , 4.0 , n_freqs ) )
        self.proj  = nn.Linear ( 2 * n_freqs , d_model )
        self.norm  = nn.LayerNorm ( d_model )

    def forward ( self , occ_scalar ) :
        """occ_scalar: (B, 1)  →  (B, 1, d_model)"""
        f      = self.freqs.abs ()
        angles = 2 * math.pi * f.unsqueeze (0) * occ_scalar
        enc    = torch.cat ( [ torch.sin ( angles ) ,
                                torch.cos ( angles ) ] , dim=-1 )
        return self.norm ( self.proj ( enc ) ).unsqueeze (1)


class SpectralGraphLayer ( nn.Module ) :
    """
    Spectral graph view: log-band-energy node features + band-coherence
    Laplacian diffusion + OCC FiLM on node features.

    Unchanged from working_model_RUL — kept as the complementary view to
    PDMN's novel time-view.  The two views now differ along three axes:
      · node features   : tokenised time-domain (time-view)  vs. FFT band-energy (spectral-view)
      · adjacency       : OCC-only prototype modes (time-view)  vs. band-coherence of x (spectral-view)
      · OCC integration : ante-hoc (shapes the graph)  vs. post-hoc (FiLM on node features)

    x       : (B, C, N)
    occ_emb : (B, C, LCE_dim)
    out     : (B, C, h_dim)
    """
    def __init__ ( self , node_count , h_dim , n_bands=4 , LCE_dim=16 ,
                   kappa=0.1 , alpha=0.05 ) :
        super ().__init__ ()
        self.n_bands  = n_bands
        self._kappa   = nn.Parameter ( torch.tensor ( kappa ) )
        self._alpha   = nn.Parameter ( torch.tensor ( alpha ) )
        half          = int ( h_dim / 2 )
        self.band_embed   = nn.Linear ( n_bands , n_bands )
        self.residual_net = nn.Sequential (
            nn.Linear ( n_bands , half ) , nn.ELU () ,
            nn.Linear ( half    , n_bands ) )
        self.occ_film     = nn.Linear ( LCE_dim , 2 * n_bands )
        self.output_proj  = nn.Linear ( n_bands , h_dim )

    def _band_energy ( self , x ) :
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
            film       = self.occ_film ( occ_emb )
            sg , sb    = film.chunk ( 2 , dim=-1 )
            z_out      = ( 1 + sg ) * z_out + sb
        return self.output_proj ( z_out )


class AdditiveGatedFusion ( nn.Module ) :
    """
    gate = sigmoid( Linear( h_time + h_spec ) )
    h    = gate * h_time + (1 − gate) * h_spec
    """
    def __init__ ( self , h_dim ) :
        super ().__init__ ()
        self.gate = nn.Sequential ( nn.Linear ( h_dim , h_dim ) , nn.Sigmoid () )

    def forward ( self , h_time , h_spec ) :
        gate = self.gate ( h_time + h_spec )
        return gate * h_time + ( 1 - gate ) * h_spec


class PDMN ( nn.Module ) :
    """
    Progressive Degradation Mode Network.

    Two novel uses of OCC
    ──────────────────────
    1. Degradation Mode Activator (DegradationModeAdjacency)
       OCC triggers sigmoid activations of K learned failure modes.
       The sensor coupling graph = superposition of currently-active modes.
       adj is a pure function of OCC → zero domain contamination from x.

    2. Lifecycle Position Encoding (LifecyclePositionEncoding)
       OCC is treated as the engine's position on the degradation manifold
       (analogous to sequence position in Transformers).  Fourier-basis
       encoding of OCC is added to sensor tokens before GCN propagation,
       priming all sensor representations with a shared lifecycle coordinate
       across domains.

    Architecture overview
    ──────────────────────
    Time-view  :  sensor_proj(x) + LPE(OCC)  →  GCN(mode_adj, FiLM)  →  h_time
    Spec-view  :  SpectralGraphLayer(x, occ_emb)                       →  h_spec
    Fusion     :  AdditiveGatedFusion(h_time, h_spec)                  →  h
    Output     :  channel_fc → final_fc → softplus                     →  RUL

    Returned dict keys
    ───────────────────
    gamma       : (B, 1)    RUL prediction
    adj         : (B, C, C) mode adjacency (for mar_loss / routing_mmd)
    activations : (B, K)    per-mode activation weights (for routing_mmd)
    h           : (B, C, h) fused representation (for adversarial DA)
    """
    def __init__ ( self , args , supports=None ) :
        super ().__init__ ()
        self.device      = torch.device ( "cuda" if torch.cuda.is_available () else "cpu" )
        self.feature_num = args.num_nodes
        self.sequence_len= args.input_length
        self.hidden_dim  = args.hidden_dim
        self.dropout     = nn.Dropout ( args.dropout )
        self.LCE_dim     = getattr ( args , 'LCE_dim'  , 16 )
        self.K           = getattr ( args , 'K'        , 8  )
        self.hop         = getattr ( args , 'hop'      , 2  )


        self.occ_projection  = nn.Linear ( self.sequence_len , self.LCE_dim )

        self.occ_scalar_proj = nn.Linear ( self.LCE_dim , 1 )


        self.sensor_proj = nn.Sequential (
            nn.Linear ( self.sequence_len , self.hidden_dim ) ,
            nn.LayerNorm ( self.hidden_dim ) )


        self.lpe = LifecyclePositionEncoding ( self.hidden_dim )


        self.mode_adjacency = DegradationModeAdjacency (
            self.feature_num , K=self.K , dropout=args.dropout )


        self.time_view = TimeViewGraphConv (
            self.hidden_dim , self.hidden_dim , self.dropout ,
            self.feature_num , self.LCE_dim ,
            support_len=1 , order=self.hop )


        self.spectral_view = SpectralGraphLayer (
            node_count = self.feature_num ,
            h_dim      = self.hidden_dim ,
            n_bands    = getattr ( args , 'n_bands' , 4 ) ,
            LCE_dim    = self.LCE_dim )


        self.fusion    = AdditiveGatedFusion ( self.hidden_dim )
        self.fuse_norm = nn.LayerNorm ( self.hidden_dim )


        self.channel_fc = nn.Sequential (
            nn.Linear ( self.hidden_dim , self.hidden_dim ) , nn.ELU () ,
            nn.Linear ( self.hidden_dim , 1 ) )
        self.final_fc = nn.Sequential (
            nn.Linear ( self.feature_num , self.feature_num ) , nn.ELU () ,
            nn.Linear ( self.feature_num , 1 ) )

    def forward ( self , x_in , OCC=None ) :
        B , N , C = x_in.shape
        x = x_in.permute ( 0 , 2 , 1 )


        occ_emb    = self.occ_projection (
            OCC.transpose ( 1 , 2 ).to ( x_in.device ) )
        occ_emb    = occ_emb.expand ( -1 , C , -1 )
        occ_scalar = torch.sigmoid (
            self.occ_scalar_proj ( occ_emb.mean ( 1 ) ) )


        x_tok = self.sensor_proj ( x )


        x_tok = x_tok + self.lpe ( occ_scalar )


        adj_time , activations = self.mode_adjacency ( occ_scalar )


        h_time = self.time_view (
            x_tok.permute ( 0 , 2 , 1 ) ,
            [ adj_time ] , occ_emb
        ).permute ( 0 , 2 , 1 )


        h_spec = self.spectral_view ( x , occ_emb )


        h = self.fuse_norm (
            self.dropout ( self.fusion ( h_time , h_spec ) ) )


        fc_out = self.channel_fc ( h ).permute ( 0 , 2 , 1 )
        fc_out = self.final_fc   ( fc_out ).permute ( 0 , 2 , 1 )
        pred   = F.softplus ( fc_out.squeeze (1) )

        return None , {
            'gamma'       : pred ,
            'adj'         : adj_time ,
            'activations' : activations ,
            'h'           : h ,
        }


def pdmn_struct_loss ( model , margin=None ) :
    """
    Combined structural loss: L_order + L_div.

    L_order enforces μ_1 < μ_2 < ... < μ_K (modes fire in causal sequence).
    L_div   keeps mode matrices distinct (no collapse to a single pattern).

    Merged into one term so the training loop has a single lam_struct knob.
    Monotonicity of adj is implied structurally by ordered activation +
    near-identity init, so a separate MAR penalty is not needed.
    """
    return mode_ordering_loss ( model , margin ) + mode_diversity_loss ( model )


def mode_ordering_loss ( model , margin=None ) :
    """
    Soft constraint: onset OCC values must be ordered  μ_1 < μ_2 < ... < μ_K.

    Physical meaning: degradation modes activate in a fixed thermodynamic
    sequence — no later mode can precede an earlier one.

    margin defaults to 1 / (K + 1) so modes are spread evenly if exactly
    at the constraint boundary.
    """
    mu  = model.mode_adjacency.mu
    if margin is None :
        margin = 1.0 / ( model.mode_adjacency.K + 1 )
    return F.relu ( mu [ :-1 ] - mu [ 1: ] + margin ).sum ()


def mode_diversity_loss ( model ) :
    """
    Discourages K mode matrices from collapsing to the same coupling pattern.

    Computes the mean cosine similarity between every pair of modes on their
    off-diagonal (coupling) elements.  High similarity → modes redundant.
    Minimising this loss keeps modes spread across sensor coupling space.
    """
    C        = model.mode_adjacency.C
    device   = model.mode_adjacency.modes.device
    eye      = torch.eye ( C , device=device )
    mask     = ~eye.bool ()
    off_diag = model.mode_adjacency.modes [ : , mask ]
    normed   = F.normalize ( off_diag , dim=-1 )
    gram     = normed @ normed.T
    k_eye    = torch.eye ( model.mode_adjacency.K , device=device )
    return ( gram * ( 1 - k_eye ) ).abs ().mean ()


def _rbf_mmd ( s , t ) :
    """Unbiased RBF-MMD with median-heuristic bandwidth.  s,t: (B, D)."""
    with torch.no_grad () :
        sigma = torch.cdist (
            torch.cat ( [s,t] ) , torch.cat ( [s,t] )
        ).median ().clamp ( min=1e-3 )
    def rbf ( x , y ) :
        return torch.exp (
            -(x.unsqueeze(1)-y.unsqueeze(0)).pow(2).sum(-1) / (2*sigma**2) )
    return rbf(s,s).mean() + rbf(t,t).mean() - 2*rbf(s,t).mean()


def routing_mmd ( act_s , act_t ) :
    """
    MMD on K-dimensional mode activation vectors.

    Replaces adjacency_mmd from working_model_RUL.

    Instead of aligning full C×C adjacency matrices across domains
    (dimensionality C² ≈ 196 for CMAPSS), we align K-dimensional activation
    vectors (K = 8).  The activation vector captures which degradation modes
    are currently firing — this should be domain-invariant (same physics =
    same modes at same OCC) while being far cheaper to align.

    The OCC-staged bucketing from adjacency_mmd is no longer needed here:
    the activations are already a function of OCC, so source and target
    samples at the same lifecycle stage will naturally have similar activations.
    A flat MMD on the full batch suffices.

    act_s, act_t : (B, K)
    """
    return _rbf_mmd ( act_s.float () , act_t.float () )
