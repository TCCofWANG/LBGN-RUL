"""
Loss functions and stage utilities for EviAdapt
(Li et al., IEEE TIM 2024).

Used exclusively by Experiment.training_eviadapt when
model_name == 'EviAdaptRUL'.  All other models use the
legacy hidden-feature alignment already in Experiment.py.

Key differences vs. a standard NIG implementation
──────────────────────────────────────────────────
• quant_evi_loss  : asymmetric quantile evidential NIG loss (Eq. 6-8 in the
                    paper), evaluated at quantiles [0.25, 0.75].  This
                    replaces the symmetric Amini et al. NIG NLL used in the
                    pre-session implementation.
• Stage alignment : Gaussian kernel MMD cross-term (Stage_Wise_Alignment from
                    utils.py in EviAdapt-main), computed per degradation
                    stage.  Replaces prototype MSE.
• Target threshold: 30th-percentile of predicted RUL (not batch median) to
                    define pseudo-stage boundaries, matching the original.
• DAN / MMD       : Full unbiased MMD (XX + YY - XY - YX) for DAGCN_RUL's
                    feature-space alignment term alongside adversarial loss.
"""

import math
import torch
import torch.nn.functional as F
import torch.distributions as dist


def _tilted_loss ( q : float , e : torch.Tensor ) -> torch.Tensor :
    """Pinball / quantile loss: max(q·e, (q-1)·e)."""
    return torch.maximum ( q * e , ( q - 1 ) * e )


def _nig_nll ( y , gamma , v , alpha , beta , w_mean , quantile ) :
    """
    Asymmetric NIG NLL for a single quantile q.

    Derived from Eq. (6) of Li et al., 2024.
    tau_two = 2/(q*(1-q))  transforms the NIG precision to match the
    asymmetric Laplace distribution implied by tilted loss.
    """
    tau_two    = 2.0 / ( quantile * ( 1.0 - quantile ) )
    twoBlambda = 2.0 * 2.0 * beta * ( 1.0 + tau_two * w_mean * v )
    twoBlambda = twoBlambda.clamp ( min=1e-8 )

    nll = (
          0.5 * torch.log ( math.pi / v.clamp ( min=1e-8 ) )
        - alpha * torch.log ( twoBlambda )
        + ( alpha + 0.5 ) * torch.log ( ( v * ( y - gamma ) ** 2 + twoBlambda ).clamp ( min=1e-8 ) )
        + torch.lgamma ( alpha )
        - torch.lgamma ( alpha + 0.5 )
    )
    return nll.mean ()


def _nig_reg ( y , gamma , v , alpha , beta , w_mean , quantile , omega=0.01 ) :
    """Evidence regularization via tilted loss × accumulated evidence."""
    error = _tilted_loss ( quantile , y - gamma )
    evi   = 2.0 * v + alpha + 1.0 / beta.clamp ( min=1e-8 )
    return ( error * evi ).mean ()


def quant_evi_loss ( y , mu , v , alpha , beta ,
                     quantiles=(0.25, 0.75) , coeff=0.3 ) :
    """
    Quantile evidential NIG loss summed over quantile columns.

    Column i of (mu, v, alpha, beta) — shape (B, n_quantiles) — corresponds
    to quantile quantiles[i].  Matches train_eval.py exactly:

        for i, q in enumerate(config.quantiles):
            loss += quant_evi_loss(y, mu[:,i:i+1], v[:,i:i+1],
                                   alpha[:,i:i+1], beta[:,i:i+1], q, coeff=3e-1)

    For each quantile q at column i:
      theta   = (1 - 2q) / (q(1-q))
      w_mean  = beta_i / (alpha_i - 1)   [E of inverse-gamma; detached]
      mu_q    = mu_i + theta * w_mean
      loss_i  = NIG_NLL(y, mu_q, v_i, alpha_i, beta_i, w_mean, q)
              + coeff * NIG_Reg(y, mu_i, v_i, alpha_i, beta_i, w_mean, q)

    Args:
        y           : (B, 1)     target RUL (normalised if is_minmax).
        mu, v, alpha, beta : (B, n_q) or (B, 1) NIG parameters.
        quantiles   : tuple of quantile levels; paper uses (0.25, 0.75).
        coeff       : regularization weight; paper uses 3e-1.
    Returns:
        scalar — sum over quantiles (not averaged, matching original accumulation).
    """
    total = torch.tensor ( 0.0 , device=y.device )
    n_q   = mu.size ( -1 )

    for i , q in enumerate ( quantiles ) :

        col    = i if n_q > 1 else 0
        mu_i   = mu    [ : , col : col + 1 ]
        v_i    = v     [ : , col : col + 1 ]
        a_i    = alpha [ : , col : col + 1 ]
        b_i    = beta  [ : , col : col + 1 ]

        theta  = ( 1.0 - 2.0 * q ) / ( q * ( 1.0 - q ) )
        w_mean = ( b_i / ( a_i - 1.0 ).clamp ( min=1e-8 ) ).detach ()
        mu_q   = mu_i + theta * w_mean

        total += _nig_nll ( y , mu_q , v_i , a_i , b_i , w_mean , q )
        total += coeff * _nig_reg ( y , mu_i , v_i , a_i , b_i , w_mean , q )

    return total


def _gaussian_kernel ( source , target , kernel_mul=2.0 , kernel_num=5 ) :
    """
    Multi-scale Gaussian kernel matrix over concatenated [source; target].
    Matches guassian_kernel() in Stage_Wise_Alignment / DAN.py.
    """
    n      = source.size ( 0 ) + target.size ( 0 )
    total  = torch.cat ( [ source , target ] , dim=0 )
    t0     = total.unsqueeze ( 0 ).expand ( n , n , total.size ( 1 ) )
    t1     = total.unsqueeze ( 1 ).expand ( n , n , total.size ( 1 ) )
    L2     = ( ( t0 - t1 ) ** 2 ).sum ( 2 )
    bw     = ( L2.data.sum () / max ( n * ( n - 1 ) , 1 ) ).clamp ( min=1e-8 )
    bw    /= kernel_mul ** ( kernel_num // 2 )
    return sum ( torch.exp ( -L2 / ( bw * kernel_mul ** i ) ) for i in range ( kernel_num ) )


def stage_wise_alignment_mmd ( source , target , kernel_mul=2.0 , kernel_num=5 ) :
    """
    Kernel MMD cross-term alignment.

    loss = -(E[k(s,t)] + E[k(t,s)])

    Minimising this pulls source and target feature distributions together.
    Matches Stage_Wise_Alignment.forward() from EviAdapt-main/utils.py exactly.

    Args:
        source, target : (N, D) feature tensors for samples in the same stage.
    """
    n_s = source.size ( 0 )
    K   = _gaussian_kernel ( source , target , kernel_mul , kernel_num )
    XY  = torch.mean ( K [ :n_s , n_s: ] )
    YX  = torch.mean ( K [ n_s: , :n_s ] )
    return - ( XY + YX )


def dan_mmd ( source , target , kernel_mul=2.0 , kernel_num=5 ) :
    """
    Full unbiased MMD: E[k(s,s)] + E[k(t,t)] - E[k(s,t)] - E[k(t,s)].

    Matches DAN() from DAGCN/loss/DAN.py.
    Used as the DAN structure-loss term in training_adversarial for DAGCN_RUL.

    Args:
        source, target : (B, D) feature tensors.
    """
    n_s = source.size ( 0 )
    K   = _gaussian_kernel ( source , target , kernel_mul , kernel_num )
    XX  = K [ :n_s , :n_s ]
    YY  = K [ n_s: , n_s: ]
    XY  = K [ :n_s , n_s: ]
    YX  = K [ n_s: , :n_s ]
    return torch.mean ( XX + YY - XY - YX )


def assign_src_stages ( occ ) :
    """
    3-stage lifecycle assignment for labelled source samples.

    occ: (B,) normalised [0, 1] OCC  →  stage (B,) ∈ {0, 1, 2}

    Thresholds match EviAdapt source RUL-normalisation (max_rul=130):
        stage 0 (early)  : OCC ≤ 0.15
        stage 1 (mid)    : 0.15 < OCC ≤ 0.66
        stage 2 (late)   : OCC > 0.66
    """
    s           = torch.zeros ( occ.shape[0] , dtype=torch.long , device=occ.device )
    s [ occ > 0.15 ] = 1
    s [ occ > 0.66 ] = 2
    return s


def assign_tgt_stages ( pred_rul , threshold ) :
    """
    2-stage pseudo-label for unlabelled target samples.

    pred_rul  : (B,) predicted RUL
    threshold : scalar — 30th-percentile of predicted RUL over the target
                training set (np.quantile(pred, 0.3) in the original paper).
                The 30th percentile yields a 30/70 split, biasing toward
                the healthier (majority) region.

    stage 2 = pred > threshold  (higher RUL → earlier / less degraded)
    stage 1 = pred ≤ threshold  (lower  RUL → later  / more degraded)

    Matches tgt_trainY_cluster assignment in EviAdapt.py.
    """
    s              = torch.ones ( pred_rul.shape[0] , dtype=torch.long , device=pred_rul.device )
    s [ pred_rul > threshold ] = 2

    return s


def compute_stage_alignment_loss ( out_s_ref , out_t , src_stages , tgt_stages ,
                                   active_stages=(1, 2) , device='cpu' ) :
    """
    Per-stage kernel MMD between target and (frozen) source DER params.

    For each active stage, collects all source samples in that stage and all
    target samples in that stage, then computes Stage_Wise_Alignment MMD on
    their concatenated (v, α, β) feature vectors.

    Requires ≥ 2 samples per stage in both domains (kernel MMD is undefined
    with a single point); skips stage if either domain has < 2 samples.

    out_s_ref : dict from frozen source_model forward  (has 'v','alpha','beta')
    out_t     : dict from target_model forward          (has 'v','alpha','beta')
    src_stages: (B,) LongTensor from assign_src_stages
    tgt_stages: (B,) LongTensor from assign_tgt_stages
    """
    loss      = torch.tensor ( 0.0 , device=device , requires_grad=True )
    n_aligned = 0

    for stage in active_stages :
        ms = ( src_stages == stage )
        mt = ( tgt_stages == stage )
        if ms.sum () < 2 or mt.sum () < 2 :
            continue

        src_DER = torch.cat ( [
            out_s_ref['v']    [ ms ] ,
            out_s_ref['alpha'][ ms ] ,
            out_s_ref['beta'] [ ms ] ,
        ] , dim=-1 )

        tgt_DER = torch.cat ( [
            out_t['v']    [ mt ] ,
            out_t['alpha'][ mt ] ,
            out_t['beta'] [ mt ] ,
        ] , dim=-1 )

        loss      += stage_wise_alignment_mmd ( src_DER , tgt_DER )
        n_aligned += 1

    return loss / max ( n_aligned , 1 )
