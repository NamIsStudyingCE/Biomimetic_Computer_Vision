"""
FlyWire Optic Lobe Connectome Inspection & Visualization Script
Generates 300 DPI publication-grade figures and quantitative metrics.
Biomimetic Computer Vision Project
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

BASE_DIR = Path(r"D:\Biomimetic Computer Vision")
FIG_DIR = BASE_DIR / "results" / "figures"
LOG_DIR = BASE_DIR / "results" / "logs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_FILE = BASE_DIR / "data" / "flywire" / "flywire_visual_graph.pt"

def inspect_and_plot():
    print("=" * 70)
    print("  FLYWIRE CONNECTOME STRUCTURAL INSPECTION & VISUALIZATION")
    print("=" * 70)

    data = torch.load(GRAPH_FILE, map_location="cpu", weights_only=False)
    cell_names = data["cell_names"]
    metadata = data["metadata"]
    df: pd.DataFrame = data["synapse_df"]
    W = data["W_raw"].numpy()

    # 1. Quantitative Metrics
    stats = {
        "num_neurons": len(cell_names),
        "num_synapse_types": len(df),
        "total_synapse_count": int(df["syn_count"].sum()),
        "excitatory_count": int(df[df["sign"] > 0]["syn_count"].sum()),
        "inhibitory_count": int(df[df["sign"] < 0]["syn_count"].sum()),
        "excitatory_ratio": float((df["sign"] > 0).mean()),
        "inhibitory_ratio": float((df["sign"] < 0).mean()),
        "layer_breakdown": {}
    }

    layers = ["Retina", "Lamina", "Medulla", "Lobula_Plate"]
    for l in layers:
        cells = [c for c, m in metadata.items() if m["layer"] == l]
        syn_in = int(df[df["post_layer"] == l]["syn_count"].sum())
        syn_out = int(df[df["pre_layer"] == l]["syn_count"].sum())
        stats["layer_breakdown"][l] = {
            "cell_count": len(cells),
            "cells": cells,
            "synapses_in": syn_in,
            "synapses_out": syn_out
        }

    # Save JSON log
    log_file = LOG_DIR / "gd1_connectome_stats.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[OK] Saved statistical metrics log to: {log_file}")

    # 2. Publication Figure (3-panel, 300 DPI)
    plt.style.use("default")
    fig = plt.figure(figsize=(15, 5), dpi=300)

    # Panel A: Synaptic Connectivity Matrix Heatmap
    ax1 = fig.add_subplot(1, 3, 1)
    vmax = np.max(np.abs(W))
    im = ax1.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax1.set_xticks(range(len(cell_names)))
    ax1.set_yticks(range(len(cell_names)))
    ax1.set_xticklabels(cell_names, rotation=90, fontsize=8)
    ax1.set_yticklabels(cell_names, fontsize=8)
    ax1.set_title("(A) Canonical Synaptic Matrix $W_{ij}$\n(Red: ACh +, Blue: GABA/Glu -)", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Post-synaptic Target", fontsize=9)
    ax1.set_ylabel("Pre-synaptic Source", fontsize=9)
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8)

    # Panel B: Synapse Counts by Layer
    ax2 = fig.add_subplot(1, 3, 2)
    layer_names = ["Lamina", "Medulla", "Lobula Plate"]
    layer_keys = ["Lamina", "Medulla", "Lobula_Plate"]
    exc_counts = [df[(df["post_layer"] == l) & (df["sign"] > 0)]["syn_count"].sum() for l in layer_keys]
    inh_counts = [df[(df["post_layer"] == l) & (df["sign"] < 0)]["syn_count"].sum() for l in layer_keys]
    x = np.arange(len(layer_names))
    width = 0.35

    ax2.bar(x - width/2, exc_counts, width, label="Excitatory (ACh +)", color="#ef4444", alpha=0.85)
    ax2.bar(x + width/2, inh_counts, width, label="Inhibitory (GABA/Glu -)", color="#3b82f6", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layer_names, fontsize=9, fontweight="medium")
    ax2.set_ylabel("Cumulative Synapse Count", fontsize=9)
    ax2.set_title("(B) Synaptic Polarity by Target Layer", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8, framealpha=0.9)
    ax2.grid(axis="y", linestyle="--", alpha=0.3)

    # Panel C: Receptive Field Spatial Footprint of T4 Motion Channels
    ax3 = fig.add_subplot(1, 3, 3)
    t4_names = ["T4a (+x)", "T4b (-x)", "T4c (+y)", "T4d (-y)"]
    t4_keys = ["T4a", "T4b", "T4c", "T4d"]
    footprint = np.zeros((4, 3)) # [Lead, Trail, Opponent]
    for i, k in enumerate(t4_keys):
        fwd_lead = df[(df["post_cell"] == k) & (df["pre_cell"] == "Mi1")]["syn_count"].sum()
        fwd_trail = df[(df["post_cell"] == k) & (df["pre_cell"] == "Tm3")]["syn_count"].sum()
        opp_inh = df[(df["post_cell"] == k) & (df["sign"] < 0)]["syn_count"].sum()
        footprint[i] = [fwd_lead, fwd_trail, opp_inh]

    im3 = ax3.imshow(footprint, cmap="YlOrRd", aspect="auto")
    ax3.set_xticks([0, 1, 2])
    ax3.set_xticklabels(["Mi1 (Lead, dx=0)", "Tm3 (Trail, -dx)", "Mi4/9 (Opp, +dx)"], fontsize=8, rotation=15)
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(t4_names, fontsize=9, fontweight="medium")
    ax3.set_title("(C) T4 Directional Motion Circuit Offsets", fontsize=10, fontweight="bold")
    for r in range(4):
        for c in range(3):
            ax3.text(c, r, f"{int(footprint[r, c])}", ha="center", va="center", color="black" if footprint[r, c] < 50 else "white", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig_file = FIG_DIR / "gd1_connectome_structure.png"
    plt.savefig(fig_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[OK] Generated 300 DPI publication figure: {fig_file}")

    print("\n--- SUMMARY OF METRICS ---")
    print(f"Total Synapse Count: {stats['total_synapse_count']}")
    print(f"Excitatory Count   : {stats['excitatory_count']} ({stats['excitatory_count']/stats['total_synapse_count']*100:.1f}%)")
    print(f"Inhibitory Count   : {stats['inhibitory_count']} ({stats['inhibitory_count']/stats['total_synapse_count']*100:.1f}%)")
    for l, val in stats["layer_breakdown"].items():
        print(f"  {l:<12}: {val['cell_count']:>2} cells | Synapses In: {val['synapses_in']:>4} | Synapses Out: {val['synapses_out']:>4}")
    print("=" * 70)

if __name__ == "__main__":
    inspect_and_plot()
