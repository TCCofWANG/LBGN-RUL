

import csv
import math
import os
import re
import statistics as stats
from collections import defaultdict

from build_ablation_results_table import load_all, collect, PAIRS, PAIR_LABELS

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out", "DA_RUL_lineage_comparison_table.tex")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

DLGNET_BROKEN = {"FD001_FD002", "FD001_FD004", "FD003_FD002", "FD003_FD004"}
AIDGN_UNSTABLE = {"FD002_FD003", "FD002_FD004"}

MODELS = ["LBGN (Full)", "AIDGN", "DLGNet"]


def load_baseline(model):

    out = defaultdict(dict)
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        fpath = os.path.join(LOGS_DIR, f"{pair}experimental_logs.csv")
        with open(fpath, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["model"] != model:
                continue
            sm = re.search(r"_s(\d+)$", r["savepath"])
            if not sm:
                continue
            seed, t = sm.group(1), r["time"]
            cur = out[pair].get(seed)
            if cur is None or t > cur[2]:
                out[pair][seed] = (float(r["best_last_RMSE"]), float(r["score"]), t)
    return out


def fmt_num(x, is_score):
    if not is_score:
        return f"{x:.2f}"
    if abs(x) >= 1e6:
        mant, exp = f"{x:.2e}".split("e")
        return f"{mant}\\times10^{{{int(exp)}}}"
    return f"{x:,.0f}"


def quartiles(values):

    s = sorted(values)
    n = len(s)
    def q(p):
        idx = p * (n - 1)
        lo, hi = int(idx), min(int(idx) + 1, n - 1)
        frac = idx - lo
        return s[lo] + (s[hi] - s[lo]) * frac
    return q(0.25), q(0.5), q(0.75)


def fmt_cell(values, is_score, flag="", use_median=False):

    if not values:
        return "-"
    n = len(values)
    if n == 1:
        return f"${fmt_num(values[0], is_score)}$"
    if use_median:
        q1, med, q3 = quartiles(values)
        return f"${fmt_num(med, is_score)}\\,[{fmt_num(q1, is_score)},{fmt_num(q3, is_score)}]$" + flag
    m, s = stats.mean(values), stats.stdev(values)
    return f"${fmt_num(m, is_score)}\\pm{fmt_num(s, is_score)}$" + flag


def pooled(pair_to_values, exclude=frozenset()):
    means = [stats.mean(v) for (src, tgt) in PAIRS
             for pair in [f"{src}_{tgt}"] if pair not in exclude
             for v in [pair_to_values[pair]] if v]
    return stats.mean(means), stats.stdev(means)


def paired_p(deltas):
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


def p_str(p):
    if p is None:
        return "n/a"
    if p < 0.001:
        return "$p{<}0.001$"
    return f"$p={p:.4f}$"


def build_per_pair_table(data_by_model, is_score, flags_by_model, caption, label):
    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "c" * len(PAIRS) + "c}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(PAIR_LABELS) + " & Overall \\\\")
    lines.append("\\midrule")

    for model in MODELS:
        d = data_by_model[model]
        flags = flags_by_model.get(model, {})
        cells = []
        all_vals = []
        for src, tgt in PAIRS:
            pair = f"{src}_{tgt}"
            vals = [v[1] if is_score else v[0] for v in d[pair].values()]
            all_vals.extend(vals)
            cells.append(fmt_cell(vals, is_score, flags.get(pair, ""), use_median=is_score))
        if is_score:
            q1, med, q3 = quartiles(all_vals)
            overall = f"$\\mathbf{{{fmt_num(med, is_score)}\\,[{fmt_num(q1, is_score)},{fmt_num(q3, is_score)}]}}$"
        else:
            pair_vals = {f"{s}_{t}": [v[1] if is_score else v[0] for v in d[f'{s}_{t}'].values()] for s, t in PAIRS}
            m, s_ = pooled(pair_vals)
            overall = f"$\\mathbf{{{fmt_num(m, is_score)}\\pm{fmt_num(s_, is_score)}}}$"
        label_tex = model.replace("_", "\\_")
        lines.append(f"{label_tex} & " + " & ".join(cells) + f" & {overall} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_per_pair_winloss_table(data_by_model, is_score, caption, label):

    lines = []
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{l" + "c" * len(PAIRS) + "c}")
    lines.append("\\toprule")
    lines.append("Model & " + " & ".join(PAIR_LABELS) + " & Overall \\\\")
    lines.append("\\midrule")

    none_data = data_by_model["LBGN (Full)"]
    for model in ["AIDGN", "DLGNet"]:
        d = data_by_model[model]
        cells = []
        total_w, total_l = 0, 0
        for src, tgt in PAIRS:
            pair = f"{src}_{tgt}"
            none_seeds = {sd: (v[1] if is_score else v[0]) for sd, v in none_data[pair].items()}
            other_seeds = {sd: (v[1] if is_score else v[0]) for sd, v in d[pair].items()}
            common = set(none_seeds) & set(other_seeds)
            w = sum(1 for sd in common if other_seeds[sd] < none_seeds[sd] - 1e-9)
            l = sum(1 for sd in common if other_seeds[sd] > none_seeds[sd] + 1e-9)
            total_w += w
            total_l += l
            cells.append(f"{w}W-{l}L" if common else "-")
        lines.append(f"{model} & " + " & ".join(cells) + f" & \\textbf{{{total_w}W-{total_l}L}} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")
    lines.append("\\end{table*}")
    return "\n".join(lines)


def build_significance_table(data_by_model, is_score, caption, label):

    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\footnotesize")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")

    if is_score:
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\toprule")
        lines.append("Model & Median $\\Delta$Score [IQR] & W/L \\\\")
        lines.append("\\midrule")
        for model in ["AIDGN", "DLGNet"]:
            deltas, w, l = [], 0, 0
            for src, tgt in PAIRS:
                pair = f"{src}_{tgt}"
                none_seeds = {sd: v[1] for sd, v in data_by_model["LBGN (Full)"][pair].items()}
                other_seeds = {sd: v[1] for sd, v in data_by_model[model][pair].items()}
                common = set(none_seeds) & set(other_seeds)
                for sd in common:
                    d = other_seeds[sd] - none_seeds[sd]
                    deltas.append(d)
                    if d < -1e-9:
                        w += 1
                    elif d > 1e-9:
                        l += 1
            q1, med, q3 = quartiles(deltas)
            lines.append(f"{model} & ${fmt_num(med, True)}\\,[{fmt_num(q1, True)},{fmt_num(q3, True)}]$ & {w}W-{l}L \\\\")
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        return "\n".join(lines)

    none_pair_vals = {f"{s}_{t}": [v[0] for v in data_by_model["LBGN (Full)"][f"{s}_{t}"].values()] for s, t in PAIRS}
    none_mean, _ = pooled(none_pair_vals)

    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & $\\Delta$RMSE & \\%$\\Delta$ & $p$ & W/L \\\\")
    lines.append("\\midrule")

    for model in ["AIDGN", "DLGNet"]:
        deltas, w, l = [], 0, 0
        for src, tgt in PAIRS:
            pair = f"{src}_{tgt}"
            none_seeds = {sd: v[0] for sd, v in data_by_model["LBGN (Full)"][pair].items()}
            other_seeds = {sd: v[0] for sd, v in data_by_model[model][pair].items()}
            common = set(none_seeds) & set(other_seeds)
            for sd in common:
                d = other_seeds[sd] - none_seeds[sd]
                deltas.append(d)
                if d < -1e-9:
                    w += 1
                elif d > 1e-9:
                    l += 1
        m = stats.mean(deltas)
        p = paired_p(deltas)
        pct = m / none_mean * 100
        pct_str = f"{fmt_num(pct, is_score)}\\%" if abs(pct) >= 1e6 else f"{pct:+.1f}\\%"
        lines.append(f"{model} & ${fmt_num(m, is_score)}${sig_stars(p)} & ${pct_str}$ & {p_str(p)} & {w}W-{l}L \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def main():
    full = load_all()
    none_data = collect(full, "none")
    aidgn = load_baseline("AIDGN")
    dlgnet = load_baseline("DLGNet")

    data_by_model = {"LBGN (Full)": none_data, "AIDGN": aidgn, "DLGNet": dlgnet}
    flags_by_model = {
        "AIDGN": {p: "\\textsuperscript{a}" for p in AIDGN_UNSTABLE},
        "DLGNet": {p: "\\textsuperscript{b}" for p in DLGNET_BROKEN},
    }

    caveat = (
        "\\textsuperscript{a}AIDGN: FD02$\\to$FD03/FD04 show high seed variance traced to a "
        "reproducible seed-42 instability. Confirmed via result.npz inspection: worst seeds produce "
        "genuinely out-of-range predictions (e.g.\\ 450.7 predicted vs.\\ RUL cap 125, seed 42) -- a "
        "handful of catastrophically miscalibrated samples, not distributed noise; these single-handedly "
        "explain both the inflated RMSE and the exponentially-amplified Score. "
        "\\textsuperscript{b}DLGNet: FD01$\\to$FD02/FD04 and FD03$\\to$FD02/FD04 collapse to an "
        "identical value across seeds AND across both source domains -- a confirmed constant-output "
        "collapse (all 259 test predictions on FD01$\\to$FD02 seed42 are exactly 0.0; RMSE equals "
        "$\\sqrt{E[y^2]}$ exactly, i.e.\\ determined entirely by the target's own label distribution). "
        "Root cause: CMAPSS\\_Related/load\\_data\\_CMAPSS.py fits a MinMaxScaler per operating condition "
        "on the source split only (verified: fit\\_transform never touches target data, so this is a "
        "coverage failure, not a leakage one) and transforms target rows only when their condition "
        "exactly matches a source condition -- for single-condition sources (FD001/FD003) against "
        "six-condition targets (FD002/FD004) this match fails almost entirely, leaving target inputs "
        "effectively unnormalized at test time. Confirmed not to affect LBGN or any DA baseline "
        "(Experiment.\\_get\\_data\\_da() calls the same loader source-vs-itself and target-vs-itself, "
        "so the cross-domain condition mismatch never triggers); scoped fix affects AIDGN/DLGNet only. "
        "Not rerun as of writing -- flagged cells excluded from the clean-subset comparison in "
        "Table~\\ref{tab:lineage-clean}."
    )

    rmse_tex = build_per_pair_table(
        data_by_model, is_score=False, flags_by_model=flags_by_model,
        caption="RMSE comparison against LBGN-RUL's direct architectural predecessors (AIDGN, DLGNet), "
                "C-MAPSS, mean$\\pm$std over 8 seeds. " + caveat,
        label="tab:lineage-rmse",
    )
    score_tex = build_per_pair_table(
        data_by_model, is_score=True, flags_by_model=flags_by_model,
        caption="PHM08 Score comparison against LBGN-RUL's direct architectural predecessors (AIDGN, "
                "DLGNet), C-MAPSS, \\textbf{median [Q1, Q3]} over 8 seeds -- reported as median, not "
                "mean, because the flagged cells' exponentially-amplified values make a pooled mean "
                "uninterpretable (see Table~\\ref{tab:lineage-sig-score} for why the mean "
                "was dropped entirely there). " + caveat,
        label="tab:lineage-score",
    )
    score_wl_tex = build_per_pair_winloss_table(
        data_by_model, is_score=True,
        caption="Per-transfer win/loss record vs LBGN (Full) on PHM08 Score (seed-level comparisons "
                "within each pair) -- robust to the tail by construction, since a win/loss count does not "
                "depend on how large the losing margin was.",
        label="tab:lineage-score-winloss",
    )
    rmse_sig_tex = build_significance_table(
        data_by_model, is_score=False,
        caption="Paired significance vs LBGN (Full) on RMSE, C-MAPSS, two-tailed paired $t$-test on "
                "matching (pair, seed) values: $^{*}p{<}0.05$, $^{**}p{<}0.01$, $^{***}p{<}0.001$. "
                "Includes the unstable/collapsed cells flagged in Table~\\ref{tab:lineage-rmse} -- "
                "descriptive of current data, not final reportable capability estimates.",
        label="tab:lineage-sig-rmse",
    )
    score_sig_tex = build_significance_table(
        data_by_model, is_score=True,
        caption="Median $\\Delta$Score vs LBGN (Full), C-MAPSS. Reported as median $\\pm$ IQR and "
                "win/loss rather than a mean/$t$-test: a mean of exponentially-saturated Score values "
                "(up to $10^{18}$) is not a weak signal, it is uninterpretable, so no $p$-value is given "
                "here -- see Table~\\ref{tab:lineage-score-winloss} for the per-transfer breakdown.",
        label="tab:lineage-sig-score",
    )

    clean_pairs = sorted({f"{s}_{t}" for s, t in PAIRS} - AIDGN_UNSTABLE - DLGNET_BROKEN)
    clean_deltas, clean_w, clean_l = [], 0, 0
    for pair in clean_pairs:
        none_seeds = {sd: v[0] for sd, v in none_data[pair].items()}
        dlg_seeds = {sd: v[0] for sd, v in dlgnet[pair].items()}
        aid_seeds = {sd: v[0] for sd, v in aidgn[pair].items()}
        for sd in set(dlg_seeds) & set(aid_seeds):
            d = dlg_seeds[sd] - aid_seeds[sd]
            clean_deltas.append(d)
            if d < -1e-9:
                clean_w += 1
            elif d > 1e-9:
                clean_l += 1
    clean_m = stats.mean(clean_deltas)
    clean_p = paired_p(clean_deltas)
    clean_tex = "\n".join([
        "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\caption{DLGNet vs AIDGN RMSE, restricted to the " + str(len(clean_pairs)) +
        " pairs where \\emph{neither} model is flagged in Table~\\ref{tab:lineage-rmse} -- the decisive "
        "check of whether ``DLGNet transfers worse'' survives once the normalization-bug-contaminated "
        "cells are removed. Pairs: " + ", ".join(p.replace("FD00", "F") for p in clean_pairs) +
        ". Two-tailed paired $t$-test, matching (pair, seed) RMSE.}",
        "\\label{tab:lineage-clean}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Comparison & Mean $\\Delta$RMSE & $p$ & DLGNet W/L & $n$ \\\\",
        "\\midrule",
        f"DLGNet $-$ AIDGN & ${clean_m:+.2f}${sig_stars(clean_p)} & {p_str(clean_p)} & "
        f"{clean_w}W-{clean_l}L & {len(clean_deltas)} \\\\",
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])

    tex = "\n\n".join([rmse_tex, score_tex, score_wl_tex, rmse_sig_tex, score_sig_tex, clean_tex])
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(tex + "\n")
    print(tex)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
