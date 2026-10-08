

import csv
import math
import os
import re
import statistics as stats
from collections import defaultdict

import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

NONE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs_NCMAPSS")
ABL_DIR = "./external_data/abl_logs_NCMAPSS"
OUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "out", "DA_RUL_NCMAPSS_ablation_results_transposed.xlsx"
)

SOURCES = ["DS01", "DS02", "DS03", "DS04"]
PAIRS = [(s, t) for s in SOURCES for t in SOURCES if t != s]
PAIR_LABELS = [f"{s}->{t}" for s, t in PAIRS]

VARIANTS = [
    ("none", "none (reference)", "ref"),
    ("no_spectral_diffusion", "no_spectral_diffusion", "arch"),
    ("no_da", "no_da", "loss"),
    ("source_only", "source_only", "loss"),
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


def load_variant_rows(dir_path, pairs):

    records = []
    for src, tgt in pairs:
        pair = f"{src}_{tgt}"
        fpath = os.path.join(dir_path, f"{src}_{tgt}experimental_logs.csv")
        if not os.path.exists(fpath):
            continue
        with open(fpath, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("model") != "LBGN_RUL":
                    continue
                info_str, savepath = r.get("info"), r.get("savepath")
                if not info_str or not savepath:
                    continue
                vm = re.search(r"ablation_(.+)$", info_str)
                variant = vm.group(1) if vm else ("none" if info_str.endswith("_final") else None)
                if variant is None:
                    continue
                sm = re.search(r"_s(\d+)$", savepath)
                if not sm:
                    continue
                seed = sm.group(1)
                try:
                    rmse, score = float(r["best_last_RMSE"]), float(r["score"])
                except (KeyError, ValueError):
                    continue
                records.append((pair, variant, seed, rmse, score, r.get("time", "")))

    latest = {}
    for pair, variant, seed, rmse, score, t in records:
        key = (pair, variant, seed)
        if key not in latest or t > latest[key][-1]:
            latest[key] = (rmse, score, t)

    out = defaultdict(dict)
    for (pair, variant, seed), (rmse, score, _t) in latest.items():
        out[(pair, variant)][seed] = (rmse, score)
    return out


def load_all():
    none_rows = load_variant_rows(NONE_DIR, PAIRS)
    abl_rows = load_variant_rows(ABL_DIR, PAIRS)
    merged = defaultdict(dict)
    for (pair, variant), seeds in none_rows.items():
        if variant == "none":
            merged[(pair, "none")].update(seeds)
    for (pair, variant), seeds in abl_rows.items():
        if variant in ("no_da", "no_spectral_diffusion", "source_only"):
            merged[(pair, variant)].update(seeds)
    return merged


def collect(full, variant_key):
    out = {}
    for src, tgt in PAIRS:
        pair = f"{src}_{tgt}"
        out[pair] = full.get((pair, variant_key), {})
    return out


def fmt_mean_std(values, is_score):
    n = len(values)
    if n == 0:
        return "-", 0
    if n == 1:
        v = values[0]
        return f"{v:.2f} (n=1)", 1
    m, s = stats.mean(values), stats.stdev(values)
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
    return ARCH_FILL


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
    ws.row_dimensions[row].height = 40
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
        overall = f"{stats.mean(pair_means):.2f}" if pair_means else "-"
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
            deltas, w, l, t = [], 0, 0, 0
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
            val_str = f"{m:.2f}"
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
            n = len(all_deltas)
            sd_ = stats.stdev(all_deltas) if n > 1 else 0.0
            se = sd_ / math.sqrt(n) if n > 1 else 0.0
            p = math.erfc(abs(om / se) / math.sqrt(2)) if se > 0 else None
            sign = "+" if om > 1e-9 else ("" if om < -1e-9 else "\u00b1")
            p_str = f", p={p:.4f}" if p is not None else ""
            mcell.value = f"{sign}{om:.2f}  ({total_w}W-{total_l}L{'-' + str(total_t) + 'T' if total_t else ''}{p_str})"
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
        ws.column_dimensions[get_column_letter(c)].width = 24

    ws.cell(row=1, column=1,
            value="Per-(pair, variant) count and seed list, LBGN_RUL rows only (see module "
                  "docstring -- unfiltered rows from other baseline models sharing the same CSV "
                  "would corrupt this). 'none' from logs_NCMAPSS/ (local project copy, confirmed "
                  "identical to the Desktop source); no_da/no_spectral_diffusion from the Desktop's "
                  "abl_logs_NCMAPSS/ (2026-08-11 run, not yet copied into the local project tree).")
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
            s = f"n={n} seeds={seeds}" if n else "-"
            cell = ws.cell(row=row, column=2 + i, value=s)
            cell.alignment = DATA_ALIGN
            if n == 0:
                cell.font = DASH_FONT
                cell.fill = DASH_FILL
            elif n < 8:
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

    print("=== Seed-count inventory ===")
    for key, disp, _ in VARIANTS:
        counts = [len(data_by_variant[key][f"{s}_{t}"]) for s, t in PAIRS]
        print(f"{disp:24s} counts={counts}")

    wb = openpyxl.Workbook()
    wb.remove(wb.active)

    build_raw_sheet(
        wb, "RMSE (mean+-std)",
        "N-CMAPSS LBGN ablation check - RMSE (mean+/-std over all seeds present per cell, up to the "
        "full 8-seed set {0,1,7,41,42,64,1234,2026}). 'none' from logs_NCMAPSS/, no_da/"
        "no_spectral_diffusion from the Desktop's abl_logs_NCMAPSS/ (2026-08-11). configs/hparams.py's "
        "N_CMAPSS entries are untested placeholders (copied from CMAPSS FD001) -- treat every number "
        "here as a first look, not a tuned result.",
        data_by_variant, is_score=False,
    )
    build_raw_sheet(
        wb, "Score (mean+-std)",
        "N-CMAPSS LBGN ablation check - PHM08 Score (lower=better), mean+/-std over all seeds present "
        "per cell. Score is heavy-tailed; read alongside the Delta sheet's win/loss counts.",
        data_by_variant, is_score=True,
    )
    build_delta_sheet(
        wb, "Delta RMSE vs none",
        "RMSE delta vs 'none', PAIRED on matching seeds only. Negative/green = variant beats none; "
        "positive/red = variant worse. Overall-mean cell includes a two-tailed paired t-test p-value.",
        data_by_variant, is_score=False,
    )
    build_delta_sheet(
        wb, "Delta Score vs none",
        "PHM08 Score delta vs 'none', paired on matching seeds only. Same convention as the RMSE "
        "delta sheet.",
        data_by_variant, is_score=True,
    )
    build_provenance_sheet(wb, data_by_variant)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    wb.save(OUT_PATH)
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
