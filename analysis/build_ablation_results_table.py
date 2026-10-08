

import os
import re
import statistics as stats

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_abl")
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "DA_RUL_ablation_results_transposed.xlsx"
)

EXPECTED_SEED_COUNT = 8

COMMON_SEEDS = {"0", "1", "7", "41", "42", "64", "1234", "2026"}

PAIRS = [
    ("FD001", "FD002"), ("FD001", "FD003"), ("FD001", "FD004"),
    ("FD002", "FD001"), ("FD002", "FD003"), ("FD002", "FD004"),
    ("FD003", "FD001"), ("FD003", "FD002"), ("FD003", "FD004"),
    ("FD004", "FD001"), ("FD004", "FD002"), ("FD004", "FD003"),
]


VARIANTS = [
    ("none",                   "none (reference)",     "ref"),
    ("time_adj",               "time_adj",              "arch"),
    ("uniform_band",           "uniform_band",          "arch"),
    ("static_band",            "static_band",           "arch"),
    ("no_lifecycle",           "no_lifecycle",          "arch"),
    ("no_input_film",          "no_input_film",         "arch"),
    ("shared_film",            "shared_film",           "arch"),
    ("no_film",                "no_film",               "arch"),
    ("no_spectral_view",       "no_spectral_view",      "arch"),
    ("no_spectral_film",       "no_spectral_film",      "arch"),
    ("no_spectral_diffusion",  "no_spectral_diffusion", "arch"),
    ("uniform_band_no_spectral_diffusion", "uniform_band_no_spectral_diffusion", "arch"),
    ("plain_mse",              "plain_mse",             "loss"),
    ("inverted_occ_weight",    "inverted_occ_weight",   "loss"),
    ("no_mar",                 "no_mar",                "loss"),
    ("no_mmd",                 "no_mmd",                "loss"),
    ("no_da",                  "no_da",                 "loss"),
    ("source_only",            "source_only",           "loss"),
]
NON_REF_VARIANTS = [v for v in VARIANTS if v[2] != "ref"]

FONT_NAME = "Arial"
THIN_GRAY = Side(style="thin", color="BFBFBF")
BORDER_ALL = Border(left=THIN_GRAY, right=THIN_GRAY, top=THIN_GRAY, bottom=THIN_GRAY)

TITLE_FONT = Font(name=FONT_NAME, size=8, italic=True, color="595959")
TITLE_ALIGN = Alignment(horizontal="left", vertical="center", wrap_text=True)

HEADER_FONT = Font(name=FONT_NAME, size=9, bold=True, color="FFFFFF")
HEADER_FILL = PatternFill(fill_type="solid", fgColor="1F3864")
HEADER_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)

REF_FILL = PatternFill(fill_type="solid", fgColor="D9E1F2")
ARCH_FILL = PatternFill(fill_type=None)
LOSS_FILL = PatternFill(fill_type="solid", fgColor="FCE4D6")
ZEBRA_GRAY_FILL = PatternFill(fill_type="solid", fgColor="F5F5F5")

MODEL_NAME_FONT = Font(name=FONT_NAME, size=9, bold=True, color="000000")
MODEL_NAME_ALIGN = Alignment(horizontal="left", vertical="center")

DATA_FONT = Font(name=FONT_NAME, size=9, color="000000")
DATA_ALIGN = Alignment(horizontal="center", vertical="center", wrap_text=True)

MEAN_FONT = Font(name=FONT_NAME, size=9, bold=True, color="000000")

DASH_FONT = Font(name=FONT_NAME, size=9, italic=True, color="808080")
DASH_FILL = PatternFill(fill_type="solid", fgColor="F2F2F2")

DELTA_POS_FONT = Font(name=FONT_NAME, size=9, bold=True, color="CC0000")
DELTA_NEG_FONT = Font(name=FONT_NAME, size=9, bold=True, color="375623")
DELTA_ZERO_FONT = Font(name=FONT_NAME, size=9, color="595959")

SECTION_FONT = Font(name=FONT_NAME, size=9, bold=True, color="1F3864")
SECTION_FILL = PatternFill(fill_type="solid", fgColor="BDD7EE")
SECTION_ALIGN = Alignment(horizontal="left", vertical="center")


def abbr(src, tgt):
    return f"F{src[-1]}->F{tgt[-1]}"


PAIR_LABELS = [abbr(*p) for p in PAIRS]


def load_all():

    records = []
    n_shifted = 0
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        fpath = os.path.join(DATA_DIR, f"{src}_{tgt}experimental_logs.csv")
        if not os.path.exists(fpath):
            raise FileNotFoundError(fpath)
        with open(fpath, newline="", encoding="utf-8") as f:
            import csv
            for r in csv.DictReader(f):
                if r.get("info") is not None:
                    info_str, savepath = r["info"], r["savepath"]
                else:
                    info_str, savepath = r.get("resume"), r.get("hidden_dim")
                    n_shifted += 1
                if not info_str or not savepath:
                    continue
                vm = re.search(r"ablation_(.+)$", info_str)
                sm = re.search(r"_s(\d+)$", savepath)
                if not vm or not sm:
                    continue
                seed = sm.group(1)
                if seed not in COMMON_SEEDS:
                    continue
                records.append({
                    "pair": pair,
                    "variant": vm.group(1),
                    "seed": seed,
                    "best_last_RMSE": float(r["best_last_RMSE"]),
                    "score": float(r["score"]),
                })
    if n_shifted:
        print(f"Recovered {n_shifted} schema-shifted rows via the resume/hidden_dim fallback.")
    full = pd.DataFrame.from_records(records)

    n_before = len(full)


    full = full.drop_duplicates(subset=["pair", "variant", "seed"], keep="last").reset_index(drop=True)
    n_deduped = n_before - len(full)
    if n_deduped:
        print(f"Deduped {n_deduped} exact (pair, variant, seed) duplicate rows (kept LATEST occurrence).")
    return full


def collect(full, variant_key):

    out = {}
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        sub = full[(full["pair"] == pair) & (full["variant"] == variant_key)]
        out[pair] = {row.seed: (row.best_last_RMSE, row.score) for row in sub.itertuples()}
    return out


def fmt_mean_std(values, is_score):
    n = len(values)
    if n == 0:
        return "-", 0
    if n == 1:
        v = values[0]
        return (f"{v:,.0f} (n=1)" if is_score else f"{v:.2f} (n=1)"), 1
    m = stats.mean(values)
    s = stats.stdev(values)
    if is_score:
        return f"{m:,.0f}\u00b1{s:,.0f}", n
    return f"{m:.2f}\u00b1{s:.2f}", n


def apply_border_all(ws, max_row, max_col):
    for r in range(1, max_row + 1):
        for c in range(1, max_col + 1):
            ws.cell(row=r, column=c).border = BORDER_ALL


def fill_for(group, row_idx):
    if group == "ref":
        return REF_FILL
    if group == "loss":
        return LOSS_FILL
    return ZEBRA_GRAY_FILL if row_idx % 2 == 0 else ARCH_FILL


def build_raw_sheet(wb, sheet_name, title_text, data_by_variant, is_score):
    ws = wb.create_sheet(sheet_name)
    n_pairs = len(PAIRS)
    last_col = 2 + n_pairs

    ws.column_dimensions["A"].width = 24
    for c in range(2, last_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 14

    row = 1
    ws.cell(row=row, column=1, value=title_text)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    ws.cell(row=row, column=1).font = TITLE_FONT
    ws.cell(row=row, column=1).alignment = TITLE_ALIGN
    ws.row_dimensions[row].height = 28
    row += 1

    ws.cell(row=row, column=1, value="Variant")
    for i, label in enumerate(PAIR_LABELS):
        ws.cell(row=row, column=2 + i, value=label)
    ws.cell(row=row, column=last_col, value="Overall\nmean")
    for c in range(1, last_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
    ws.row_dimensions[row].height = 26
    row += 1

    for key, disp, group in VARIANTS:
        base_fill = fill_for(group, row)
        pair_data = data_by_variant[key]
        ws.cell(row=row, column=1, value=disp)
        ws.cell(row=row, column=1).font = MODEL_NAME_FONT
        ws.cell(row=row, column=1).alignment = MODEL_NAME_ALIGN
        ws.cell(row=row, column=1).fill = base_fill

        pair_means = []
        for i, (src, tgt) in enumerate(PAIRS):
            pair = f"{src}_{tgt}"
            vals = [v[1] if is_score else v[0] for v in pair_data[pair].values()]
            s, n = fmt_mean_std(vals, is_score)
            cell = ws.cell(row=row, column=2 + i, value=s)
            cell.alignment = DATA_ALIGN
            if s == "-":
                cell.font = DASH_FONT
                cell.fill = DASH_FILL
            else:
                cell.font = DATA_FONT
                cell.fill = base_fill
                pair_means.append(stats.mean(vals))
        overall = (f"{stats.mean(pair_means):,.0f}" if is_score else f"{stats.mean(pair_means):.2f}")\
            if pair_means else "-"
        mcell = ws.cell(row=row, column=last_col, value=overall)
        mcell.font = MEAN_FONT
        mcell.alignment = DATA_ALIGN
        mcell.fill = base_fill
        row += 1

    apply_border_all(ws, row - 1, last_col)
    ws.freeze_panes = "B3"
    return ws


def build_delta_sheet(wb, sheet_name, title_text, data_by_variant, is_score):
    ws = wb.create_sheet(sheet_name)
    n_pairs = len(PAIRS)
    last_col = 2 + n_pairs

    ws.column_dimensions["A"].width = 24
    for c in range(2, last_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16

    row = 1
    ws.cell(row=row, column=1, value=title_text)
    ws.merge_cells(start_row=row, start_column=1, end_row=row, end_column=last_col)
    ws.cell(row=row, column=1).font = TITLE_FONT
    ws.cell(row=row, column=1).alignment = TITLE_ALIGN
    ws.row_dimensions[row].height = 42
    row += 1

    ws.cell(row=row, column=1, value="Variant")
    for i, label in enumerate(PAIR_LABELS):
        ws.cell(row=row, column=2 + i, value=label)
    ws.cell(row=row, column=last_col, value="Overall\nmean \u0394 / W-L")
    for c in range(1, last_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
    ws.row_dimensions[row].height = 26
    row += 1

    none_data = data_by_variant["none"]

    for key, disp, group in NON_REF_VARIANTS:
        base_fill = fill_for(group, row)
        pair_data = data_by_variant[key]
        ws.cell(row=row, column=1, value=disp)
        ws.cell(row=row, column=1).font = MODEL_NAME_FONT
        ws.cell(row=row, column=1).alignment = MODEL_NAME_ALIGN
        ws.cell(row=row, column=1).fill = base_fill

        all_deltas, total_w, total_l, total_t = [], 0, 0, 0
        for i, (src, tgt) in enumerate(PAIRS):
            pair = f"{src}_{tgt}"
            common_seeds = sorted(set(pair_data[pair]) & set(none_data[pair]), key=lambda x: (len(x), x))
            deltas = []
            w = l = t = 0
            for sd in common_seeds:
                v_val = pair_data[pair][sd][1] if is_score else pair_data[pair][sd][0]
                n_val = none_data[pair][sd][1] if is_score else none_data[pair][sd][0]
                d = v_val - n_val
                deltas.append(d)
                if d < -1e-9:
                    w += 1
                elif d > 1e-9:
                    l += 1
                else:
                    t += 1
            cell = ws.cell(row=row, column=2 + i)
            cell.alignment = DATA_ALIGN
            if not deltas:
                cell.value = "-"
                cell.font = DASH_FONT
                cell.fill = DASH_FILL
                continue
            m = stats.mean(deltas)
            all_deltas.extend(deltas)
            total_w, total_l, total_t = total_w + w, total_l + l, total_t + t
            sign = "+" if m > 1e-9 else ("" if m < -1e-9 else "\u00b1")
            val_str = f"{m:,.0f}" if is_score else f"{m:.2f}"
            cell.value = f"{sign}{val_str}  ({w}W-{l}L{'-' + str(t) + 'T' if t else ''})"
            cell.fill = base_fill
            if m > 1e-9:
                cell.font = DELTA_POS_FONT
            elif m < -1e-9:
                cell.font = DELTA_NEG_FONT
            else:
                cell.font = DELTA_ZERO_FONT

        mcell = ws.cell(row=row, column=last_col)
        mcell.alignment = DATA_ALIGN
        mcell.fill = base_fill
        if all_deltas:
            om = stats.mean(all_deltas)
            sign = "+" if om > 1e-9 else ("" if om < -1e-9 else "\u00b1")
            val_str = f"{om:,.0f}" if is_score else f"{om:.2f}"
            mcell.value = f"{sign}{val_str}  ({total_w}W-{total_l}L{'-' + str(total_t) + 'T' if total_t else ''})"
            mcell.font = MEAN_FONT
        else:
            mcell.value = "-"
            mcell.font = DASH_FONT
        row += 1

    apply_border_all(ws, row - 1, last_col)
    ws.freeze_panes = "B3"
    return ws


def build_provenance_sheet(wb, data_by_variant):
    ws = wb.create_sheet("Seed provenance")
    n_pairs = len(PAIRS)
    last_col = 1 + n_pairs

    ws.column_dimensions["A"].width = 24
    for c in range(2, last_col + 1):
        ws.column_dimensions[get_column_letter(c)].width = 22

    ws.cell(row=1, column=1,
            value=f"Per-(pair, variant) count of seeds present in logs_abl/, out of the full "
                  f"{EXPECTED_SEED_COUNT}-seed set {{0,1,7,41,42,64,1234,2026}} common to every variant that "
                  f"was actually run (verified 2026-08-10 backfill: all 15 variants at 8/8 seeds across all "
                  f"12 pairs). Cells below {EXPECTED_SEED_COUNT} are flagged red -- expected for no_mar/no_mmd "
                  f"(0/{EXPECTED_SEED_COUNT}, not run in this campaign at all). '-' means no rows exist for "
                  f"that cell.")
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col)
    ws.cell(row=1, column=1).font = TITLE_FONT
    ws.cell(row=1, column=1).alignment = TITLE_ALIGN
    ws.row_dimensions[1].height = 40

    row = 2
    ws.cell(row=row, column=1, value="Variant")
    for i, label in enumerate(PAIR_LABELS):
        ws.cell(row=row, column=2 + i, value=label)
    for c in range(1, last_col + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = HEADER_FONT
        cell.fill = HEADER_FILL
        cell.alignment = HEADER_ALIGN
    row += 1

    for key, disp, group in VARIANTS:
        base_fill = fill_for(group, row)
        ws.cell(row=row, column=1, value=disp)
        ws.cell(row=row, column=1).font = MODEL_NAME_FONT
        ws.cell(row=row, column=1).alignment = MODEL_NAME_ALIGN
        ws.cell(row=row, column=1).fill = base_fill
        for i, (src, tgt) in enumerate(PAIRS):
            pair = f"{src}_{tgt}"
            seeds = sorted(data_by_variant[key][pair].keys(), key=lambda x: (len(x), x))
            n = len(seeds)
            if n == 0:
                s = "-"
            elif n < EXPECTED_SEED_COUNT:
                s = f"n={n} seeds={seeds}"
            else:
                s = f"n={n}"
            cell = ws.cell(row=row, column=2 + i, value=s)
            cell.alignment = DATA_ALIGN
            if n == 0:
                cell.font = DASH_FONT
                cell.fill = DASH_FILL
            elif n < EXPECTED_SEED_COUNT:
                cell.font = Font(name=FONT_NAME, size=9, color="CC0000", bold=True)
                cell.fill = base_fill
            else:
                cell.font = DATA_FONT
                cell.fill = base_fill
        row += 1

    apply_border_all(ws, row - 1, last_col)
    return ws


def main():
    full = load_all()
    data_by_variant = {key: collect(full, key) for key, _, _ in VARIANTS}

    print(f"\n=== Seed-count inventory (full seed set, default={EXPECTED_SEED_COUNT}: "
          "{0,1,7,41,42,64,1234,2026}) ===")
    for key, disp, _ in VARIANTS:
        counts = [len(data_by_variant[key][f"{s}_{t}"]) for s, t in PAIRS]
        incomplete = [f"{abbr(*PAIRS[i])}(n={counts[i]})" for i in range(len(PAIRS)) if counts[i] < EXPECTED_SEED_COUNT]
        print(f"{disp:24s} counts={counts}" + (f"  INCOMPLETE: {incomplete}" if incomplete else ""))

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    build_raw_sheet(
        wb, "RMSE (mean+-std)",
        "LBGN ablation study - RMSE (mean+/-std over the full 8-seed set common to every variant, "
        "{0,1,7,41,42,64,1234,2026}). Every variant, incl. 'none', is cold-started at seed 42 then "
        "warm-started at 0/1/7/41/64/1234/2026 from that variant's own seed-42 checkpoint (same protocol "
        "throughout). Source: logs_abl/. Blue = 'none' reference row; orange = loss-level variants "
        "(plain_mse/no_mar/no_mmd/no_da).",
        data_by_variant, is_score=False,
    )
    build_raw_sheet(
        wb, "Score (mean+-std)",
        "LBGN ablation study - CMAPSS Score (PHM08-style asymmetric formula, lower=better), mean+/-std "
        "over the full 8-seed set {0,1,7,41,42,64,1234,2026} per cell (see RMSE sheet note). Score is "
        "heavy-tailed; read alongside the Delta sheet's win/loss counts, not the mean alone.",
        data_by_variant, is_score=True,
    )
    build_delta_sheet(
        wb, "Delta RMSE vs none",
        "RMSE delta vs 'none', PAIRED on matching seeds only (variant_seed - none_seed, averaged only over "
        "seeds present in both). Negative/green = variant beats none; positive/red = variant worse. "
        "W-L = seeds where the variant beat/lost to none on that pair -- a ~50/50 split with a near-zero "
        "mean delta is the signature of a true null effect (no measurable contribution), not underpowering.",
        data_by_variant, is_score=False,
    )
    build_delta_sheet(
        wb, "Delta Score vs none",
        "CMAPSS Score delta vs 'none', paired on matching seeds only. Same convention as the RMSE delta sheet.",
        data_by_variant, is_score=True,
    )
    build_provenance_sheet(wb, data_by_variant)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    wb.save(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
