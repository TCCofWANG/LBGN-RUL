

import argparse
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Models_RUL.LBGN_RUL import LBGN_RUL


def make_occ_window(p, seq_len=50, avg_max_cycle=200.0, epsilon=1e-2):

    end_cycle = p * avg_max_cycle
    cycles = np.arange(end_cycle - seq_len + 1, end_cycle + 1)
    cycles = np.clip(cycles, 1, None)
    return torch.tensor(cycles / avg_max_cycle * epsilon,
                        dtype=torch.float32).view(1, seq_len, 1)


def film_gain(film_module, occ_emb):

    with torch.no_grad():
        gamma, _ = film_module.proj(occ_emb).chunk(2, dim=-1)
    return (1 + gamma).abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to best_checkpoint.pth")
    ap.add_argument("--lce_dim", type=int, default=8)
    ap.add_argument("--hidden_dim", type=int, default=32)
    ap.add_argument("--seq_len", type=int, default=50)
    ap.add_argument("--n_bands", type=int, default=4)
    ap.add_argument("--num_nodes", type=int, default=14)
    ap.add_argument("--hop", type=int, default=2)
    ap.add_argument("--avg_max_cycle", type=float, default=200.0)
    ap.add_argument("--epsilon", type=float, default=1e-2)
    ap.add_argument("--out", default="analysis/out")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    margs = types.SimpleNamespace(
        num_nodes=args.num_nodes, input_length=args.seq_len,
        hidden_dim=args.hidden_dim, dropout=0.0, LCE_dim=args.lce_dim,
        n_bands=args.n_bands, hop=args.hop)
    model = LBGN_RUL(margs)
    state = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    positions = np.linspace(0.02, 1.0, 50)
    curves = {"input FiLM (pre-propagation)": []}
    for d in range(args.hop):
        curves[f"hop-{d + 1} FiLM (reach {d + 1})"] = []

    with torch.no_grad():
        for p in positions:
            occ = make_occ_window(p, args.seq_len, args.avg_max_cycle, args.epsilon)
            occ_emb = model.occ_projection(occ.transpose(1, 2))
            occ_emb = occ_emb.expand(-1, args.num_nodes, -1)

            curves["input FiLM (pre-propagation)"].append(
                film_gain(model.time_view.input_film, occ_emb))
            for d in range(args.hop):
                curves[f"hop-{d + 1} FiLM (reach {d + 1})"].append(
                    film_gain(model.time_view.hop_film.films[d], occ_emb))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.join(
        args.out, "film_gain_vs_occ_" +
        os.path.basename(os.path.dirname(os.path.dirname(args.ckpt))))


    import csv
    with open(stem + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lifecycle_position"] + list(curves.keys()))
        for i, p in enumerate(positions):
            w.writerow([f"{p:.4f}"] + [f"{curves[k][i]:.6f}" for k in curves])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for name, vals in curves.items():
        ax.plot(positions, vals, label=name, linewidth=2)
    ax.set_xlabel("Lifecycle position (fraction of consumed life)")
    ax.set_ylabel(r"Mean FiLM gain  $\overline{|1+\gamma|}$")
    ax.set_title(args.title or "Lifecycle-gated propagation reach")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(stem + ".png", dpi=300)
    fig.savefig(stem + ".pdf")
    print(f"saved: {stem}.png / .pdf / .csv")


    for name, vals in curves.items():
        slope = np.polyfit(positions, vals, 1)[0]
        print(f"  {name}: gain {vals[0]:.3f} -> {vals[-1]:.3f}  (slope {slope:+.3f})")


if __name__ == "__main__":
    main()
