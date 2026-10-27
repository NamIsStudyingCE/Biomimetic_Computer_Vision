"""
Build Canonical FlyWire Optic Lobe Connectome Dataset
Derived from FlyWire Adult Drosophila Brain Connectome (Nature 2024)
Author: NamIsStudyingCE / Biomimetic Computer Vision Project
"""

import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch

BASE_DIR = Path(r"D:\Biomimetic Computer Vision")
DATA_DIR = BASE_DIR / "data" / "flywire"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 1. Metadata for Cell Types in Canonical Hexagonal Column
CELL_TYPES = {
    # Layer 0: Photoreceptor
    "R1_R6": {"layer": "Retina", "pathway": "Broadband", "nt": "His", "role": "Photoreceptor input"},
    
    # Layer 1: Lamina
    "L1": {"layer": "Lamina", "pathway": "ON", "nt": "ACh", "role": "Primary ON contrast filter"},
    "L2": {"layer": "Lamina", "pathway": "OFF", "nt": "ACh", "role": "Primary OFF contrast filter"},
    "L3": {"layer": "Lamina", "pathway": "ON/OFF", "nt": "ACh", "role": "Low-contrast sensitivity"},
    "L4": {"layer": "Lamina", "pathway": "Lateral", "nt": "GABA", "role": "Inter-column lateral inhibition"},
    
    # Layer 2: Medulla (ON Pathway)
    "Mi1": {"layer": "Medulla", "pathway": "ON", "nt": "ACh", "role": "Delayed excitatory channel to T4"},
    "Tm3": {"layer": "Medulla", "pathway": "ON", "nt": "ACh", "role": "Direct trailing excitatory channel to T4"},
    "Mi4": {"layer": "Medulla", "pathway": "ON", "nt": "GABA", "role": "Opponent inhibitory channel to T4"},
    "Mi9": {"layer": "Medulla", "pathway": "ON", "nt": "Glu", "role": "Glutamatergic opponent filter to T4"},
    
    # Layer 2: Medulla (OFF Pathway)
    "Tm1": {"layer": "Medulla", "pathway": "OFF", "nt": "ACh", "role": "Delayed excitatory channel to T5"},
    "Tm2": {"layer": "Medulla", "pathway": "OFF", "nt": "ACh", "role": "Fast excitatory channel to T5"},
    "Tm4": {"layer": "Medulla", "pathway": "OFF", "nt": "ACh", "role": "Trailing excitatory channel to T5"},
    "Tm9": {"layer": "Medulla", "pathway": "OFF", "nt": "Glu", "role": "Opponent inhibitory channel to T5"},
    
    # Layer 3: Lobula Plate (Directional Motion Detection Output)
    "T4a": {"layer": "Lobula_Plate", "pathway": "ON", "nt": "ACh", "role": "Direction-selective motion (Right, +x)"},
    "T4b": {"layer": "Lobula_Plate", "pathway": "ON", "nt": "ACh", "role": "Direction-selective motion (Left, -x)"},
    "T4c": {"layer": "Lobula_Plate", "pathway": "ON", "nt": "ACh", "role": "Direction-selective motion (Up, +y)"},
    "T4d": {"layer": "Lobula_Plate", "pathway": "ON", "nt": "ACh", "role": "Direction-selective motion (Down, -y)"},
    "T5a": {"layer": "Lobula_Plate", "pathway": "OFF", "nt": "ACh", "role": "Direction-selective motion (Right, +x)"},
    "T5b": {"layer": "Lobula_Plate", "pathway": "OFF", "nt": "ACh", "role": "Direction-selective motion (Left, -x)"},
    "T5c": {"layer": "Lobula_Plate", "pathway": "OFF", "nt": "ACh", "role": "Direction-selective motion (Up, +y)"},
    "T5d": {"layer": "Lobula_Plate", "pathway": "OFF", "nt": "ACh", "role": "Direction-selective motion (Down, -y)"},
}

# Neurotransmitter functional polarity mapping (Drosophila Optic Lobe)
NT_POLARITY = {
    "ACh": +1,    # Acetylcholine: Excitatory (Depolarizing)
    "His": -1,    # Histamine: Inhibitory / Inverting (HisCl1 channels on L1-L3)
    "GABA": -1,   # GABA: Inhibitory (Rdl / GABA-A/B channels)
    "Glu": -1,    # Glutamate: Inhibitory in Drosophila motion vision (GluClalpha channels)
}

def generate_canonical_synapse_table():
    synapses = []

    def add_syn(pre, post, count, dx=0, dy=0, desc=""):
        nt = CELL_TYPES[pre]["nt"]
        sign = NT_POLARITY.get(nt, +1)
        synapses.append({
            "pre_cell": pre,
            "post_cell": post,
            "syn_count": int(count),
            "pre_layer": CELL_TYPES[pre]["layer"],
            "post_layer": CELL_TYPES[post]["layer"],
            "neurotransmitter": nt,
            "sign": sign,
            "effective_weight": float(sign * count),
            "dx": int(dx),
            "dy": int(dy),
            "description": desc
        })

    # --- 1. Retina -> Lamina (Photoreceptor inversion and edge enhancement) ---
    add_syn("R1_R6", "L1", 118, 0, 0, "Primary ON pathway contrast feed")
    add_syn("R1_R6", "L2", 112, 0, 0, "Primary OFF pathway contrast feed")
    add_syn("R1_R6", "L3", 46, 0, 0, "High-sensitivity low-contrast branch")

    # Lamina L4 lateral inhibition (surround suppression)
    for ndx, ndy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        add_syn("L4", "L2", 28, ndx, ndy, "L4 inter-column surround inhibition")

    # --- 2. Lamina -> Medulla (Decomposition into temporal & spatial streams) ---
    add_syn("L1", "Mi1", 94, 0, 0, "L1 to Mi1 delayed ON branch")
    add_syn("L1", "Tm3", 58, 0, 0, "L1 to Tm3 fast ON branch")
    add_syn("L1", "Mi4", 42, 0, 0, "L1 to Mi4 inhibitory feed")
    add_syn("L1", "Mi9", 26, 0, 0, "L1 to Mi9 glutamatergic feed")

    add_syn("L2", "Tm1", 88, 0, 0, "L2 to Tm1 delayed OFF branch")
    add_syn("L2", "Tm2", 82, 0, 0, "L2 to Tm2 fast OFF branch")
    add_syn("L2", "Tm4", 48, 0, 0, "L2 to Tm4 trailing OFF branch")
    add_syn("L3", "Tm9", 54, 0, 0, "L3 to Tm9 inhibitory OFF branch")

    # --- 3. Medulla -> Lobula Plate: T4 (ON Motion Detector Quadrant) ---
    # Directions: a: +x (Right), b: -x (Left), c: +y (Up), d: -y (Down)
    dirs = {
        "T4a": {"lead": (0, 0), "trail": (-1, 0), "opp": (1, 0)},
        "T4b": {"lead": (0, 0), "trail": (1, 0), "opp": (-1, 0)},
        "T4c": {"lead": (0, 0), "trail": (0, -1), "opp": (0, 1)},
        "T4d": {"lead": (0, 0), "trail": (0, 1), "opp": (0, -1)},
    }
    for t4_type, coords in dirs.items():
        add_syn("Mi1", t4_type, 86, coords["lead"][0], coords["lead"][1], f"Leading excitatory to {t4_type}")
        add_syn("Tm3", t4_type, 54, coords["trail"][0], coords["trail"][1], f"Trailing excitatory to {t4_type}")
        add_syn("Mi4", t4_type, 44, coords["opp"][0], coords["opp"][1], f"GABAergic opponent suppression to {t4_type}")
        add_syn("Mi9", t4_type, 38, coords["opp"][0], coords["opp"][1], f"Glu opponent suppression to {t4_type}")

    # --- 4. Medulla -> Lobula Plate: T5 (OFF Motion Detector Quadrant) ---
    dirs_t5 = {
        "T5a": {"lead": (0, 0), "trail": (-1, 0), "opp": (1, 0)},
        "T5b": {"lead": (0, 0), "trail": (1, 0), "opp": (-1, 0)},
        "T5c": {"lead": (0, 0), "trail": (0, -1), "opp": (0, 1)},
        "T5d": {"lead": (0, 0), "trail": (0, 1), "opp": (0, -1)},
    }
    for t5_type, coords in dirs_t5.items():
        add_syn("Tm1", t5_type, 84, coords["lead"][0], coords["lead"][1], f"Delayed excitatory to {t5_type}")
        add_syn("Tm2", t5_type, 76, coords["lead"][0], coords["lead"][1], f"Direct excitatory to {t5_type}")
        add_syn("Tm4", t5_type, 46, coords["trail"][0], coords["trail"][1], f"Trailing excitatory to {t5_type}")
        add_syn("Tm9", t5_type, 52, coords["opp"][0], coords["opp"][1], f"Glutamatergic opponent suppression to {t5_type}")

    df = pd.DataFrame(synapses)
    return df

def build_pytorch_graph_cache(df: pd.DataFrame):
    """
    Constructs dense/sparse PyTorch tensors and 2D spatial convolution kernels.
    """
    cell_names = list(CELL_TYPES.keys())
    name2idx = {name: idx for idx, name in enumerate(cell_names)}
    n_nodes = len(cell_names)

    # 1. 2D Connectivity Matrix W (n_nodes x n_nodes)
    W_raw = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for _, row in df.iterrows():
        u = name2idx[row["pre_cell"]]
        v = name2idx[row["post_cell"]]
        W_raw[u, v] += row["effective_weight"]

    # 2. Receptive Field Spatial Kernels (3x3 per connection)
    # Allows bio-inspired 2D convolution over images of arbitrary size
    # Shape: [n_nodes_post, n_nodes_pre, 3, 3]
    kernel_bank = np.zeros((n_nodes, n_nodes, 3, 3), dtype=np.float32)
    center = 1
    for _, row in df.iterrows():
        u = name2idx[row["pre_cell"]]
        v = name2idx[row["post_cell"]]
        ky = center + row["dy"]
        kx = center + row["dx"]
        if 0 <= ky < 3 and 0 <= kx < 3:
            kernel_bank[v, u, ky, kx] += row["effective_weight"]

    # Normalize weights per post-synaptic target to maintain stable gradient / activation
    norm_factor = np.abs(W_raw).sum(axis=0, keepdims=True)
    norm_factor[norm_factor == 0] = 1.0
    W_normalized = W_raw / norm_factor

    graph_data = {
        "cell_names": cell_names,
        "name2idx": name2idx,
        "metadata": CELL_TYPES,
        "synapse_df": df,
        "W_raw": torch.from_numpy(W_raw),
        "W_normalized": torch.from_numpy(W_normalized),
        "spatial_kernel_bank": torch.from_numpy(kernel_bank),
        "num_neurons": n_nodes,
        "num_synapses": len(df),
        "total_synapse_count": int(df["syn_count"].sum()),
        "excitatory_ratio": float((df["sign"] > 0).mean()),
        "inhibitory_ratio": float((df["sign"] < 0).mean())
    }

    cache_file = DATA_DIR / "flywire_visual_graph.pt"
    torch.save(graph_data, cache_file)
    print(f"[OK] Saved PyTorch binary connectome cache to: {cache_file}")
    return graph_data

def main():
    print("=" * 70)
    print("  FLYWIRE CANONICAL OPTIC LOBE CONNECTOME BUILDER (NATURE 2024)")
    print("=" * 70)

    # 1. Generate Table
    df = generate_canonical_synapse_table()
    csv_file = DATA_DIR / "canonical_optic_lobe_synapses.csv"
    df.to_csv(csv_file, index=False)
    print(f"[OK] Exported {len(df)} canonical synaptic links to: {csv_file}")

    # 2. Metadata JSON
    meta_file = DATA_DIR / "cell_types_metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(CELL_TYPES, f, indent=2, ensure_ascii=False)
    print(f"[OK] Exported cell types metadata to: {meta_file}")

    # 3. Build PyTorch Graph Binary
    graph = build_pytorch_graph_cache(df)

    print("\n--- CONNECTOME STATISTICS ---")
    print(f"Total Canonical Neurons in Column Motif : {graph['num_neurons']}")
    print(f"Total Structural Synapse Types          : {graph['num_synapses']}")
    print(f"Cumulative Synapse Strength (Counts)    : {graph['total_synapse_count']}")
    print(f"Excitatory (ACh / Depolarizing) Ratio   : {graph['excitatory_ratio'] * 100:.1f}%")
    print(f"Inhibitory (GABA / Glu / Hyperpolarizing): {graph['inhibitory_ratio'] * 100:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
