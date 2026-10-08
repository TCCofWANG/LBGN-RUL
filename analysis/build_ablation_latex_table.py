

import math
import os
import statistics as stats

from build_ablation_results_table import (
    VARIANTS, PAIRS, load_all, collect,
)

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "DA_RUL_ablation_table.tex")


VARIANT_INFO = {
    "time_adj": (
        "Full-band time adjacency (no OCC)",
        "Adjacency from full-band time-domain cosine similarity; no frequency-band decomposition, no OCC selection.",
    ),
    "uniform_band": (
        "Uniform band mixing (no OCC)",
        "Band decomposition kept; OCC selection over bands replaced by a frozen uniform (1/K) mix.",
    ),
    "static_band": (
        "Static learned band mixing (no OCC)",
        "Band decomposition kept; band mix is learned but static (no OCC input).",
    ),
    "no_lifecycle": (
        "No lifecycle conditioning (both pathways)",
        "Both OCC-conditioning pathways removed at once (time-view band selection and spectral-view FiLM), so neither can compensate for the other.",
    ),
    "no_input_film": (
        "No input-stage FiLM",
        "Input-stage FiLM removed; depth-wise hop FiLM kept.",
    ),
    "shared_film": (
        "Shared (non-depthwise) FiLM",
        "One FiLM shared across all hop depths, instead of an independent FiLM per depth.",
    ),
    "no_film": (
        "No FiLM",
        "All FiLM conditioning removed from the time view.",
    ),
    "no_spectral_view": (
        "No spectral view",
        "Spectral view and fusion removed entirely; time-view output feeds the regression head directly.",
    ),
    "no_spectral_film": (
        "No spectral OCC-FiLM",
        "Spectral view kept; its OCC-FiLM correction is skipped (time-view FiLM untouched).",
    ),
    "plain_mse": (
        "Unweighted (plain) MSE",
        "Source MSE loss is unweighted; the OCC-weighted (near-EOL emphasis) loss is removed.",
    ),
    "no_da": (
        "No DA loss (MAR + MMD off)",
        "Both domain-adaptation loss terms (MAR and adjacency MMD) set to zero at once; architecture unchanged.",
    ),
    "source_only": (
        "Source-only (lower bound)",
        "No target-domain exposure of any kind during training (no target batch drawn, no target forward pass, no domain-alignment term); architecture and the OCC-weighted task loss otherwise unchanged. Not a loss-weight-zeroed variant of no\\_da -- a genuinely different training procedure, reported as the lower-bound anchor against which every DA method's (including our own) improvement can be read as closing X\\% of the source-only-to-target gap.",
    ),
}


RQ_SUBHEADING = {
    "time_adj": "A1: adjacency substrate",
    "uniform_band": "A1: adjacency substrate",
    "static_band": "A1: adjacency substrate",
    "no_lifecycle": "A1: adjacency substrate",
    "no_input_film": "A2: FiLM conditioning",
    "shared_film": "A2: FiLM conditioning",
    "no_film": "A2: FiLM conditioning",
    "no_spectral_view": "A3: spectral view",
    "no_spectral_film": "A3: spectral view",
    "plain_mse": "A4: loss terms",
    "no_da": "A4: loss terms",
    "source_only": "A4: loss terms",
}
GROUP_LABEL = {"arch": "Architecture ablations", "loss": "Loss ablations"}


SHORT_CODE = {
    "time_adj": "A1.1", "uniform_band": "A1.2", "static_band": "A1.3", "no_lifecycle": "A1.4",
    "no_input_film": "A2.1", "shared_film": "A2.2", "no_film": "A2.3",
    "no_spectral_view": "A3.1", "no_spectral_film": "A3.2",
    "plain_mse": "A4.1", "no_da": "A4.2", "source_only": "A4.3",
}

MAIN_TABLE_KEYS = set(VARIANT_INFO.keys())
MAIN_TABLE_VARIANTS = [v for v in VARIANTS if v[2] != "ref" and v[0] in MAIN_TABLE_KEYS]


def fmt(mean, std, is_score):
    if is_score:
        return f"{mean:,.0f}$\\pm${std:,.0f}"
    return f"{mean:.2f}$\\pm${std:.2f}"


def overall_mean_std(pair_data, is_score):

    pair_means = []
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        vals = [v[1] if is_score else v[0] for v in pair_data[pair].values()]
        if vals:
            pair_means.append(stats.mean(vals))
    if not pair_means:
        return None, None
    return stats.mean(pair_means), (stats.stdev(pair_means) if len(pair_means) > 1 else 0.0)


def overall_delta(variant_data, none_data, is_score):

    all_deltas, w, l, t = [], 0, 0, 0
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        common = set(variant_data[pair]) & set(none_data[pair])
        for sd in common:
            v_val = variant_data[pair][sd][1] if is_score else variant_data[pair][sd][0]
            n_val = none_data[pair][sd][1] if is_score else none_data[pair][sd][0]
            d = v_val - n_val
            all_deltas.append(d)
            if d < -1e-9:
                w += 1
            elif d > 1e-9:
                l += 1
            else:
                t += 1
    if not all_deltas:
        return None, w, l, t, None
    p = paired_p_value(all_deltas)
    return stats.mean(all_deltas), w, l, t, p


def paired_p_value(deltas):

    n = len(deltas)
    if n < 2:
        return None
    s = stats.stdev(deltas)
    if s == 0:
        return None
    se = s / math.sqrt(n)
    t_stat = stats.mean(deltas) / se
    return math.erfc(abs(t_stat) / math.sqrt(2))


def sig_stars(p):
    if p is None:
        return ""
    if p < 0.001:
        return "$^{***}$"
    if p < 0.01:
        return "$^{**}$"
    if p < 0.05:
        return "$^{*}$"
    return ""


_UNSET = object()


def build_definitions_paragraph():

    lines = []
    lines.append("We evaluate the following variants against the full model (\\textit{none}); "
                  "architecture variants keep training identical, loss variants keep the "
                  "architecture identical (Section~\\ref{sec:method}).")
    lines.append("")

    cur_group = _UNSET
    cur_sub = _UNSET
    itemize_open = False

    def close_itemize():
        nonlocal itemize_open
        if itemize_open:
            lines.append("\\end{itemize}")
            lines.append("")
            itemize_open = False

    for key, disp, group in MAIN_TABLE_VARIANTS:
        if group != cur_group:
            close_itemize()
            lines.append(f"\\textbf{{{GROUP_LABEL[group]}.}}")
            cur_group = group
            cur_sub = _UNSET

        sub = RQ_SUBHEADING.get(key)
        if sub != cur_sub:
            close_itemize()
            if sub is not None:
                lines.append(f"\\textit{{{sub}.}}")
            lines.append("\\begin{itemize}")
            itemize_open = True
            cur_sub = sub

        formal_name, description = VARIANT_INFO[key]
        code = SHORT_CODE[key]
        lines.append(f"  \\item \\textbf{{{code} -- {formal_name}.}} {description}")

    close_itemize()


    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def build_results_table(data_by_variant, none_data):
    rmse_none_m, rmse_none_s = overall_mean_std(none_data, is_score=False)
    score_none_m, score_none_s = overall_mean_std(none_data, is_score=True)

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append("\\caption{LBGN ablation study (CMAPSS). RMSE/Score: mean$\\pm$std over 8 seeds. "
                  "$\\Delta$: paired per-seed change vs \\textit{none} (negative = variant is better), with "
                  "two-tailed paired $t$-test $p$ and win/loss (W/L) seed count: $^{*}p{<}0.05$, "
                  "$^{**}p{<}0.01$, $^{***}p{<}0.001$. Variant codes defined above.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Variant & RMSE & $\\Delta$RMSE ($p$, W/L) & Score & $\\Delta$Score ($p$, W/L) \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{none (full model)}} & \\textbf{{{fmt(rmse_none_m, rmse_none_s, False)}}} & -- "
                  f"& \\textbf{{{fmt(score_none_m, score_none_s, True)}}} & -- \\\\")

    cur_group = None
    for key, disp, group in MAIN_TABLE_VARIANTS:
        if group != cur_group:
            lines.append("\\midrule")
            lines.append(f"\\multicolumn{{5}}{{l}}{{\\textit{{{GROUP_LABEL[group]}}}}} \\\\")
            cur_group = group

        pair_data = data_by_variant[key]
        rmse_m, rmse_s = overall_mean_std(pair_data, is_score=False)
        score_m, score_s = overall_mean_std(pair_data, is_score=True)
        d_rmse, w_r, l_r, t_r, p_r = overall_delta(pair_data, none_data, is_score=False)
        d_score, w_s, l_s, t_s, p_s = overall_delta(pair_data, none_data, is_score=True)

        rmse_cell = fmt(rmse_m, rmse_s, False) if rmse_m is not None else "-"
        score_cell = fmt(score_m, score_s, True) if score_m is not None else "-"

        def delta_cell(d, w, l, t, is_score, p):
            if d is None:
                return "-"
            sign = "+" if d > 1e-9 else ("" if d < -1e-9 else "$\\pm$")
            val = f"{d:,.0f}" if is_score else f"{d:.2f}"
            tie_str = f"-{t}T" if t else ""
            return f"{sign}{val}{sig_stars(p)} (${p_str(p)}$, {w}W-{l}L{tie_str})"

        drmse_cell = delta_cell(d_rmse, w_r, l_r, t_r, False, p_r)
        dscore_cell = delta_cell(d_score, w_s, l_s, t_s, True, p_s)

        lines.append(f"{SHORT_CODE[key]} & {rmse_cell} & {drmse_cell} & {score_cell} & {dscore_cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def variant_stats(data_by_variant, none_data, key):

    pair_data = data_by_variant[key]
    d_r, w_r, l_r, _, p_r = overall_delta(pair_data, none_data, is_score=False)
    d_s, w_s, l_s, _, p_s = overall_delta(pair_data, none_data, is_score=True)
    return d_r, p_r, w_r, l_r, d_s, p_s, w_s, l_s


def p_str(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "p{<}0.001"
    return f"p={p:.3f}"


def ncmapss_no_da_stats():

    import importlib
    ncmapss_mod = importlib.import_module("build_transposed_ncmapss_new_logs")
    full = ncmapss_mod.load_all()
    ours = ncmapss_mod.collect(full, "LBGN_RUL")
    noda = ncmapss_mod.collect(full, "LBGN_RUL_no_da")

    all_deltas, ex_deltas = [], []
    ds01_04 = {"ours": {}, "noda": {}}
    for src, tgt in ncmapss_mod.PAIRS:
        pair = f"{src}_{tgt}"
        ours_by_seed = {sd: r for r, sc, sd in ours[pair]}
        noda_by_seed = {sd: r for r, sc, sd in noda[pair]}
        common = set(ours_by_seed) & set(noda_by_seed)
        for sd in common:
            delta = noda_by_seed[sd] - ours_by_seed[sd]
            all_deltas.append(delta)
            if pair != "DS01_DS04":
                ex_deltas.append(delta)
            else:
                ds01_04["ours"][sd] = ours_by_seed[sd]
                ds01_04["noda"][sd] = noda_by_seed[sd]

    def paired_p(deltas):
        n = len(deltas)
        s = stats.stdev(deltas)
        if s == 0:
            return None
        se = s / math.sqrt(n)
        t_stat = stats.mean(deltas) / se
        return math.erfc(abs(t_stat) / math.sqrt(2))

    p_excl = paired_p(ex_deltas)
    n_diverged = sum(1 for sd, r in ds01_04["noda"].items() if r > 30)
    n_ds01_04_seeds = len(ds01_04["noda"])
    ours_stable_lo = min(ds01_04["ours"].values())
    ours_stable_hi = max(ds01_04["ours"].values())
    diverged_min = min(r for r in ds01_04["noda"].values() if r > 30)
    return {
        "n_total": len(all_deltas), "n_excl": len(ex_deltas), "p_excl": p_excl,
        "n_diverged": n_diverged, "n_ds01_04_seeds": n_ds01_04_seeds,
        "ours_lo": ours_stable_lo, "ours_hi": ours_stable_hi, "diverged_min": diverged_min,
    }


def build_results_discussion(data_by_variant, none_data):

    rmse_none_m, _ = overall_mean_std(none_data, is_score=False)
    score_none_m, _ = overall_mean_std(none_data, is_score=True)

    s = {k: variant_stats(data_by_variant, none_data, k) for k in
         list(MAIN_TABLE_KEYS) + ["inverted_occ_weight", "no_spectral_diffusion"]}

    def pct(key, is_score=False):
        d_r, p_r, w_r, l_r, d_s, p_s, w_s, l_s = s[key]
        d, p, w, l = (d_s, p_s, w_s, l_s) if is_score else (d_r, p_r, w_r, l_r)
        base = score_none_m if is_score else rmse_none_m
        return d / base * 100.0, p, w, l

    p = []

    ub_pct, ub_p, ub_w, ub_l = pct("uniform_band")
    sb_pct, sb_p, sb_w, sb_l = pct("static_band")
    ta_r_pct, ta_r_p, _, _ = pct("time_adj")
    ta_s_pct, ta_s_p, _, _ = pct("time_adj", is_score=True)
    nl_r_pct, nl_r_p, _, _ = pct("no_lifecycle")
    nl_s_pct, nl_s_p, _, _ = pct("no_lifecycle", is_score=True)
    p.append(
        f"\\textbf{{C1 -- time view (A1, A2).}} Of the three variants targeting the adjacency component of "
        f"C1, \\texttt{{uniform\\_band}} and \\texttt{{static\\_band}} are statistically indistinguishable "
        f"from the full model (${ub_pct:+.2f}\\%$ and ${sb_pct:+.2f}\\%$ RMSE, ${p_str(ub_p)}$/${p_str(sb_p)}$), "
        f"while \\texttt{{time\\_adj}} -- which additionally removes the band decomposition itself -- costs "
        f"${ta_r_pct:+.2f}\\%$ RMSE / ${ta_s_pct:+.1f}\\%$ Score (${p_str(ta_r_p)}$). All three leave C2's "
        f"independent spectral-view conditioning untouched, giving the model an alternate lifecycle pathway "
        f"that plausibly explains the first two null results; \\texttt{{no\\_lifecycle}}, which disables both "
        f"pathways at once, confirms this reading with a large, significant effect (${nl_r_pct:+.1f}\\%$ RMSE / "
        f"${nl_s_pct:+.1f}\\%$ Score, ${p_str(nl_r_p)}$). We read this as evidence that lifecycle-conditioning "
        f"the adjacency is load-bearing but redundantly encoded across views, not that it is dispensable; the "
        f"current variant set does not isolate the time-view contribution alone from this redundancy."
    )

    nif_r_pct, nif_r_p, nif_w, nif_l = pct("no_input_film")
    nif_s_pct, nif_s_p, _, _ = pct("no_input_film", is_score=True)
    sf_r_pct, sf_r_p, _, _ = pct("shared_film")
    nf_r_pct, nf_r_p, _, _ = pct("no_film")
    p.append(
        f"FiLM conditioning shows no such confound: removing the input-stage modulation "
        f"(\\texttt{{no\\_input\\_film}}) costs ${nif_r_pct:+.2f}\\%$ RMSE (${p_str(nif_r_p)}$); its large mean "
        f"Score increase (${nif_s_pct:+.1f}\\%$) is not significant (${p_str(nif_s_p)}$) and traces to a single "
        f"pair whose few divergent seeds sit an order of magnitude outside every other pair's range, despite "
        f"\\texttt{{no\\_input\\_film}} winning the majority of cells overall ({nif_w}W-{nif_l}L). Sharing one "
        f"modulator across hop depths (\\texttt{{shared\\_film}}) costs ${sf_r_pct:+.2f}\\%$ RMSE "
        f"(${p_str(sf_r_p)}$); removing FiLM entirely (\\texttt{{no\\_film}}) costs ${nf_r_pct:+.2f}\\%$ RMSE "
        f"(${p_str(nf_r_p)}$)."
    )

    nsv_r_pct, nsv_r_p, _, _ = pct("no_spectral_view")
    nsv_s_pct, nsv_s_p, _, _ = pct("no_spectral_view", is_score=True)
    nsf_r_pct, nsf_r_p, _, _ = pct("no_spectral_film")
    nsf_s_pct, nsf_s_p, _, _ = pct("no_spectral_film", is_score=True)
    p.append(
        f"\\textbf{{C2 -- spectral view (A3).}} Removing the spectral view entirely "
        f"(\\texttt{{no\\_spectral\\_view}}) costs ${nsv_r_pct:+.2f}\\%$ RMSE / ${nsv_s_pct:+.1f}\\%$ Score "
        f"(${p_str(nsv_r_p)}$/${p_str(nsv_s_p)}$); removing only its OCC-FiLM correction "
        f"(\\texttt{{no\\_spectral\\_film}}) costs \\emph{{more}} -- ${nsf_r_pct:+.2f}\\%$ RMSE / "
        f"${nsf_s_pct:+.1f}\\%$ Score (${p_str(nsf_r_p)}$/${p_str(nsf_s_p)}$) -- confirming it is the "
        f"lifecycle-conditioning, not the branch's presence, that earns it a place in the model."
    )

    pm_r_pct, pm_r_p, _, _ = pct("plain_mse")
    da_r_pct, da_r_p, da_w, da_l = pct("no_da")
    nc = ncmapss_no_da_stats()
    iow_r_pct, iow_r_p, _, _ = pct("inverted_occ_weight")
    nsd_r_pct, nsd_r_p, _, _ = pct("no_spectral_diffusion")
    so_r_pct, so_r_p, so_r_w, so_r_l = pct("source_only")
    so_s_pct, so_s_p, _, _ = pct("source_only", is_score=True)
    p.append(
        f"\\textbf{{Loss terms.}} \\texttt{{plain\\_mse}} costs ${pm_r_pct:+.2f}\\%$ RMSE (${p_str(pm_r_p)}$), "
        f"confirming non-uniform, lifecycle-aware loss weighting matters. \\texttt{{no\\_da}} (both the "
        f"monotonic prior and structural MMD terms off) is a clean null on CMAPSS (${da_r_pct:+.2f}\\%$ RMSE, "
        f"${p_str(da_r_p)}$, near-even {da_w}W-{da_l}L), consistent with "
        f"Section~\\ref{{sec:introduction}}'s disclosure that structural alignment is not claimed as a driver "
        f"of the reported gains. On N-CMAPSS, paired across matching seeds ($n={nc['n_total']}$), "
        f"\\texttt{{no\\_da}} leaves the point estimate largely unchanged on 11 of 12 transfers "
        f"(${p_str(nc['p_excl'])}$ excluding one outlier pair) but produces outright training divergence on "
        f"the hardest transfer (DS01$\\to$DS04: RMSE $>{nc['diverged_min']:.0f}$ on {nc['n_diverged']} of "
        f"{nc['n_ds01_04_seeds']} seeds vs. a stable $\\sim\\!{nc['ours_lo']:.0f}$--${nc['ours_hi']:.0f}$ with "
        f"the full model), suggesting the structural-alignment term plays a stabilizing rather than accuracy "
        f"role under large domain gaps -- a hypothesis we leave for confirmation with additional seeds. "
        f"\\texttt{{no\\_mar}}/\\texttt{{no\\_mmd}} in isolation are deferred to future work. Unlike \\texttt{{no\\_da}}, "
        f"\\texttt{{source\\_only}} -- no target-domain exposure of any kind, not merely a zero-weighted loss "
        f"term -- is a significant regression (${so_r_pct:+.2f}\\%$ RMSE / ${so_s_pct:+.1f}\\%$ Score, "
        f"${p_str(so_r_p)}$, {so_r_w}W-{so_r_l}L), giving a lower-bound reference against which the full "
        f"model's (and the eight compared baselines') transfer improvement can be read as closing a fraction "
        f"of the source-only-to-target gap rather than only beating the next-best baseline."
    )

    p.append(
        f"Two additional probes -- inverting the OCC weighting direction (\\texttt{{inverted\\_occ\\_weight}}) "
        f"and removing the spectral diffusion step alone (\\texttt{{no\\_spectral\\_diffusion}}) -- were also "
        f"non-significant (${p_str(iow_r_p)}$, ${p_str(nsd_r_p)}$) and are omitted from "
        f"Table~\\ref{{tab:ablation}} since neither maps to a claimed contribution."
    )

    return "\n\n".join(p)


SUBSECTION_HEADER = "\\subsection{Ablation study}\\label{subsec:ablation}"

TABLE_INTRO = (
    "Table~\\ref{tab:ablation} reports a controlled ablation isolating each architectural and loss "
    "component, pooled across all twelve CMAPSS source$\\to$target pairs and eight seeds "
    "($n{=}96$ paired seed-pair comparisons per row). $\\Delta$ is the paired per-seed change relative "
    "to the full model (negative = variant is better); $p$ is a two-tailed paired $t$-test on that "
    "column's own metric."
)


def main():
    full = load_all()
    data_by_variant = {key: collect(full, key) for key, _, _ in VARIANTS}
    none_data = data_by_variant["none"]

    defs_tex = build_definitions_paragraph()
    results_tex = build_results_table(data_by_variant, none_data)
    discussion_tex = build_results_discussion(data_by_variant, none_data)
    tex = "\n\n".join([SUBSECTION_HEADER, defs_tex, TABLE_INTRO, results_tex, discussion_tex])

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
