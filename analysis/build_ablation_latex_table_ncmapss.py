

import math
import os
import statistics as stats

from build_ablation_results_table_ncmapss import PAIRS, VARIANTS, load_all, collect

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "DA_RUL_NCMAPSS_ablation_table.tex")

VARIANT_INFO = {
    "no_spectral_diffusion": (
        "No spectral OCC-diffusion",
        "Spectral view's one-step Laplacian diffusion skipped (same variant as the CMAPSS table's excluded probe; significant here, unlike on CMAPSS).",
    ),
    "no_da": (
        "No DA loss (MAR + MMD off)",
        "Both domain-adaptation loss terms (MAR and adjacency MMD) set to zero at once; architecture unchanged.",
    ),
    "source_only": (
        "Source-only (lower bound)",
        "No target-domain exposure of any kind during training; architecture and the OCC-weighted task loss otherwise unchanged. Full 8/8-seed coverage on all 12 pairs.",
    ),
}
MAIN_TABLE_VARIANTS = [v for v in VARIANTS if v[0] in VARIANT_INFO]


def fmt(mean, std, is_score):
    if is_score:
        return f"{mean:,.2f}$\\pm${std:,.2f}"
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
    return stats.mean(all_deltas), w, l, t, paired_p_value(all_deltas)


def main():
    full = load_all()
    data_by_variant = {key: collect(full, key) for key, _, _ in VARIANTS}
    none_data = data_by_variant["none"]

    rmse_none_m, rmse_none_s = overall_mean_std(none_data, is_score=False)
    score_none_m, score_none_s = overall_mean_std(none_data, is_score=True)

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{N-CMAPSS follow-up ablation check (LBGN\\_RUL, 12 DS0x$\\rightarrow$DS0y "
                  "pairs, up to 8 seeds $\\{0,1,7,41,42,64,1234,2026\\}$ pooled). $\\Delta$ and "
                  "significance as in Table~\\ref{tab:ablation}. \\textit{Caution on no\\_da}: "
                  "$\\sim$80\\% of its pooled $\\Delta$RMSE magnitude comes from two pairs "
                  "(DS01$\\rightarrow$DS04, DS02$\\rightarrow$DS04) with extreme seed-to-seed variance "
                  "(std$>$14 RMSE, a diverging-run artifact, not a smooth effect) -- the W/L split is "
                  "the more reliable read. \\textit{no\\_spectral\\_diffusion} is small but NOT "
                  "outlier-driven (loses on 67/92 seed-pairs) and contradicts its CMAPSS result "
                  "(Table~\\ref{tab:ablation}: $p{=}0.287$, null), a dataset-dependent finding under "
                  "further investigation. \\textit{source\\_only} is the cleanest-covered row here "
                  "(full 8/8 seeds, all 12 pairs, no outlier caveat) -- unlike no\\_da it removes ALL "
                  "target-domain exposure, not just the loss weight, giving a genuine lower-bound "
                  "reference. N\\_CMAPSS hyperparameters are untested placeholders copied "
                  "from CMAPSS FD001 -- treat every number here as a first look.}")
    lines.append("\\label{tab:ablation-ncmapss}")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Variant & RMSE & $\\Delta$RMSE (W/L) & Score & $\\Delta$Score (W/L) \\\\")
    lines.append("\\midrule")
    lines.append(f"\\textbf{{none (full model)}} & \\textbf{{{fmt(rmse_none_m, rmse_none_s, False)}}} & -- "
                  f"& \\textbf{{{fmt(score_none_m, score_none_s, True)}}} & -- \\\\")
    lines.append("\\midrule")

    for key, disp, group in MAIN_TABLE_VARIANTS:
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
            val = f"{d:,.2f}"
            tie_str = f"-{t}T" if t else ""
            return f"{sign}{val}{sig_stars(p)} ({w}W-{l}L{tie_str})"

        drmse_cell = delta_cell(d_rmse, w_r, l_r, t_r, False, p_r)
        dscore_cell = delta_cell(d_score, w_s, l_s, t_s, True, p_s)

        formal_name, _ = VARIANT_INFO[key]
        lines.append(f"{formal_name} & {rmse_cell} & {drmse_cell} & {score_cell} & {dscore_cell} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    tex = "\n".join(lines)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
