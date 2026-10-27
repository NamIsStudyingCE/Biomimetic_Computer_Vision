"""
Milestone GD2 Official Benchmark & Biomimetic Feature Visualization Script
Evaluates:
  1. FlyWire-BioNet (Canonical Connectome + Foveated Columnar Tucker-2, 2.92M params)
  2. Shuffled Connectome Control (Randomly Permuted Synaptic Topology, 2.92M params)
  3. Classical CNN Baseline (Standard Student YOLOv8n, 3.14M params)
  4. Heavy Custom Student Reference (Uncompressed Custom YOLOv8 KD, 6.73M params)

Outputs:
  - results/logs/gd2_benchmark_metrics.json
  - results/figures/gd2_benchmark_comparison.png (300 DPI)
  - results/figures/gd2_bio_feature_maps.png (300 DPI)
"""

import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO

from src.models.flywire_bionet import build_flywire_detector, FlyWireOpticLobeStem

PROJECT_ROOT = Path(r"D:\Biomimetic Computer Vision")
DATA_YAML = Path(r"D:\AI4EmbSys\SVRDD_YOLO_320\data.yaml")
TEST_TXT = Path(r"D:\AI4EmbSys\SVRDD_YOLO_320\test.txt")

WEIGHT_YOLOV8N = Path(r"D:\AI4EmbSys\runs\student_kd_yolov8n\weights\best_kd.pt")
WEIGHT_CUSTOM_HEAVY = Path(r"D:\AI4EmbSys\runs\student_kd_custom_yolov8\weights\best_kd.pt")

WEIGHTS_DIR = PROJECT_ROOT / "results" / "weights"
LOGS_DIR = PROJECT_ROOT / "results" / "logs"
FIGS_DIR = PROJECT_ROOT / "results" / "figures"

CLASS_NAMES = [
    "Longitudinal Crack",
    "Transverse Crack",
    "Alligator Crack",
    "Pothole",
    "Manhole Cover",
    "Transverse Patch",
    "Longitudinal Patch",
]


def evaluate_checkpoint(pt_path: Path, label: str, n_params: int) -> dict:
    print(f"\n[GD2 Benchmark] Evaluating {label} ({n_params:,} params)...")
    t0 = time.perf_counter()
    yolo = YOLO(str(pt_path))
    res = yolo.val(
        data=str(DATA_YAML),
        imgsz=320,
        batch=32,
        workers=0,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    elapsed = time.perf_counter() - t0

    p = float(res.box.mp)
    r = float(res.box.mr)
    map50 = float(res.box.map50)
    map50_95 = float(res.box.map)
    f1 = (2.0 * p * r / (p + r)) if (p + r) > 0 else 0.0
    m_params = n_params / 1e6

    per_class_map50 = {}
    if hasattr(res.box, "ap50") and res.box.ap50 is not None:
        for idx, cname in enumerate(CLASS_NAMES):
            if idx < len(res.box.ap50):
                per_class_map50[cname] = round(float(res.box.ap50[idx]), 4)

    metrics = {
        "model_name": label,
        "parameters": int(n_params),
        "parameters_M": round(m_params, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "precision": round(p, 4),
        "recall": round(r, 4),
        "f1_score": round(f1, 4),
        "efficiency_mAP50_per_MParams": round((map50 * 100.0) / m_params, 2),
        "eval_time_sec": round(elapsed, 2),
        "per_class_mAP50": per_class_map50,
    }
    print(
        f"  -> mAP@0.5: {map50*100:.2f}% | mAP@0.5:0.95: {map50_95*100:.2f}% | "
        f"Precision: {p*100:.2f}% | Recall: {r*100:.2f}% | F1: {f1*100:.2f}%"
    )
    return metrics


def plot_benchmark_comparison(results_dict: dict, out_path: Path):
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
    })
    fig = plt.figure(figsize=(18.5, 6.0), dpi=300)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.32, 1.0, 1.15], wspace=0.34)

    # Panel A: Grouped Bar Chart of Core Detection Metrics
    ax1 = fig.add_subplot(gs[0, 0])
    keys = ["Classical_CNN_YOLOv8n", "Shuffled_Connectome_Control", "FlyWire_BioNet", "Heavy_Custom_Student"]
    labels = ["Classical CNN\n(YOLOv8n\n3.14M)", "Shuffled Control\n(Scrambled\n2.92M)", "FlyWire-BioNet\n(Proposed\n2.92M)", "Heavy Reference\n(Uncompressed\n6.73M)"]

    map50_vals = [results_dict[k]["mAP50"] * 100 for k in keys]
    rec_vals = [results_dict[k]["recall"] * 100 for k in keys]
    prec_vals = [results_dict[k]["precision"] * 100 for k in keys]
    f1_vals = [results_dict[k]["f1_score"] * 100 for k in keys]

    x = np.arange(len(keys))
    w = 0.19
    b1 = ax1.bar(x - 1.5 * w, map50_vals, w, label="mAP@0.5 (%)", color="#10b981", edgecolor="#064e3b", linewidth=0.8)
    b2 = ax1.bar(x - 0.5 * w, rec_vals, w, label="Recall (%)", color="#3b82f6", edgecolor="#1e3a8a", linewidth=0.8)
    b3 = ax1.bar(x + 0.5 * w, prec_vals, w, label="Precision (%)", color="#f59e0b", edgecolor="#78350f", linewidth=0.8)
    b4 = ax1.bar(x + 1.5 * w, f1_vals, w, label="F1-Score (%)", color="#8b5cf6", edgecolor="#4c1d95", linewidth=0.8)

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.8, f"{h:.1f}", ha="center", va="bottom", fontsize=7.2, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=8.5)
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 72)
    ax1.set_title("A. SVRDD Detection Accuracy", pad=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.35)
    ax1.legend(loc="upper left", fontsize=8, ncol=2)

    # Panel B: Parameter vs mAP@0.5 Pareto Efficiency
    ax2 = fig.add_subplot(gs[0, 1])
    colors = ["#ef4444", "#f97316", "#10b981", "#6366f1"]
    markers = ["s", "^", "*", "o"]
    sizes = [130, 140, 260, 130]
    offsets = [(0.45, -0.4), (-0.42, 0.3), (0.0, 1.15), (-0.1, 1.15)]

    for idx, k in enumerate(keys):
        d = results_dict[k]
        ax2.scatter(
            d["parameters_M"], d["mAP50"] * 100,
            c=colors[idx], marker=markers[idx], s=sizes[idx],
            edgecolors="black", linewidth=1.2, zorder=5,
            label=f"{d['model_name']} ({d['parameters_M']:.2f}M)"
        )
        dx, dy = offsets[idx]
        ax2.annotate(
            f"{d['mAP50']*100:.2f}%\n({d['parameters_M']:.2f}M)",
            (d["parameters_M"] + dx, d["mAP50"] * 100 + dy),
            ha="center", fontsize=8.2, fontweight="bold"
        )

    ax2.axvline(3.1395, color="#ef4444", linestyle=":", alpha=0.7, label="YOLOv8n Param Budget (3.14M)")
    ax2.set_xlabel("Total Parameters (Millions)")
    ax2.set_ylabel("mAP@0.5 (%)")
    ax2.set_xlim(2.0, 7.4)
    ax2.set_ylim(34, 50)
    ax2.set_title("B. Parameter Efficiency Frontier", pad=10)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(loc="lower right", fontsize=7.6)

    # Panel C: Per-Class mAP@0.5 Comparison (FlyWire-BioNet vs Classical CNN vs Shuffled)
    ax3 = fig.add_subplot(gs[0, 2])
    y_pos = np.arange(len(CLASS_NAMES))
    h_bar = 0.26

    fw_cls = [results_dict["FlyWire_BioNet"]["per_class_mAP50"].get(c, 0.0) * 100 for c in CLASS_NAMES]
    sh_cls = [results_dict["Shuffled_Connectome_Control"]["per_class_mAP50"].get(c, 0.0) * 100 for c in CLASS_NAMES]
    cn_cls = [results_dict["Classical_CNN_YOLOv8n"]["per_class_mAP50"].get(c, 0.0) * 100 for c in CLASS_NAMES]

    ax3.barh(y_pos + h_bar, fw_cls, h_bar, label="FlyWire-BioNet (2.92M)", color="#10b981", edgecolor="#064e3b")
    ax3.barh(y_pos, sh_cls, h_bar, label="Shuffled Control (2.92M)", color="#f97316", edgecolor="#7c2d12")
    ax3.barh(y_pos - h_bar, cn_cls, h_bar, label="Classical CNN (3.14M)", color="#ef4444", edgecolor="#7f1d1d")

    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([c.replace(" ", "\n") for c in CLASS_NAMES], fontsize=8)
    ax3.set_xlabel("Class AP@0.5 (%)")
    ax3.set_xlim(0, 88)
    ax3.set_title("C. Per-Class Road Damage AP@0.5", pad=10)
    ax3.grid(axis="x", linestyle="--", alpha=0.35)
    ax3.legend(loc="lower right", fontsize=8)

    fig.suptitle(
        "Milestone GD2: Drosophila FlyWire-BioNet vs Symmetrical Baselines on SVRDD (1,000 Test Images, 320x320)",
        fontsize=13, fontweight="bold", y=0.98
    )
    fig.subplots_adjust(top=0.86, bottom=0.14, left=0.05, right=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[GD2 Benchmark] Saved comparison chart to {out_path}")


def _norm_vis(arr: np.ndarray, p_high: float = 99.0) -> np.ndarray:
    hi = max(1e-5, float(np.percentile(arr, p_high)))
    return np.clip(arr / hi, 0.0, 1.0) ** 0.65


def plot_bio_feature_maps(out_path: Path):
    test_paths = [Path(p.strip()) for p in TEST_TXT.read_text().splitlines() if p.strip()]
    sample_path = test_paths[0]
    for p in test_paths[:50]:
        lbl_p = Path(str(p).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")).with_suffix(".txt")
        if lbl_p.exists() and len(lbl_p.read_text().splitlines()) >= 3:
            sample_path = p
            break

    pil_img = Image.open(sample_path).convert("RGB")
    x = TF.to_tensor(pil_img).unsqueeze(0)

    stem_canon = FlyWireOpticLobeStem(mode="canonical", modulation_strength=0.015).eval()
    stem_shuff = FlyWireOpticLobeStem(mode="shuffled", modulation_strength=0.015).eval()

    with torch.no_grad():
        _, maps_canon = stem_canon.extract_bio_features(x)
        _, maps_shuff = stem_shuff.extract_bio_features(x)

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), dpi=300)

    axes[0, 0].imshow(pil_img)
    axes[0, 0].set_title(f"1. Input Road Surface Image\n({sample_path.name[:28]}...)", fontsize=10, fontweight="bold")

    axes[0, 1].imshow(_norm_vis(maps_canon["Lamina_L1_ON"][0, 0].numpy()), cmap="viridis")
    axes[0, 1].set_title("2. Lamina L1 (ON Pathway)\nBright Edge Excitation", fontsize=10, fontweight="bold")

    axes[0, 2].imshow(_norm_vis(maps_canon["Lamina_L2_OFF"][0, 0].numpy()), cmap="magma")
    axes[0, 2].set_title("3. Lamina L2 (OFF Pathway)\nL4->L2 Lateral Inhibition Crack Isolation", fontsize=10, fontweight="bold")

    axes[1, 0].imshow(_norm_vis(maps_canon["Lobula_T5_OFF"][0, 0].numpy(), 99.5), cmap="inferno")
    axes[1, 0].set_title("4. Lobula Plate T5a-d (OFF Motion/Edge)\nDirectional Opponent Crack Correlation", fontsize=10, fontweight="bold")

    axes[1, 1].imshow(_norm_vis(maps_canon["Crack_Saliency"][0, 0].numpy()), cmap="plasma")
    axes[1, 1].set_title("5. Canonical FlyWire Crack Saliency\n(Phase-Aligned L2 + T5 Integration)", fontsize=10, fontweight="bold")

    axes[1, 2].imshow(_norm_vis(maps_shuff["Crack_Saliency"][0, 0].numpy()), cmap="plasma")
    axes[1, 2].set_title("6. Shuffled Connectome Saliency\n(Scrambled Spatial Phase & E/I Noise)", fontsize=10, fontweight="bold")

    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle(
        "Drosophila Optic Lobe (Retina -> Lamina -> Medulla -> Lobula Plate) Neural Feature Extraction on SVRDD",
        fontsize=13, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[GD2 Benchmark] Saved neural feature map visualization to {out_path}")


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    # Clean up any temporary test weights
    for tmp_f in WEIGHTS_DIR.glob("test_*.pt"):
        try:
            tmp_f.unlink()
        except OSError:
            pass

    # 1. Build and save Canonical FlyWire-BioNet & Shuffled Connectome Control
    base_ckpt = torch.load(WEIGHT_CUSTOM_HEAVY, map_location="cpu", weights_only=False)

    canon_model = build_flywire_detector(str(WEIGHT_CUSTOM_HEAVY), mode="canonical")
    n_params_canon = sum(p.numel() for p in canon_model.parameters())
    canon_pt = WEIGHTS_DIR / "flywire_bionet_best.pt"
    ckpt_c = dict(base_ckpt)
    ckpt_c["model"] = canon_model.cpu().half()
    torch.save(ckpt_c, canon_pt)

    shuff_model = build_flywire_detector(str(WEIGHT_CUSTOM_HEAVY), mode="shuffled")
    n_params_shuff = sum(p.numel() for p in shuff_model.parameters())
    shuff_pt = WEIGHTS_DIR / "shuffled_control_best.pt"
    ckpt_s = dict(base_ckpt)
    ckpt_s["model"] = shuff_model.cpu().half()
    torch.save(ckpt_s, shuff_pt)

    n_params_v8n = sum(p.numel() for p in YOLO(str(WEIGHT_YOLOV8N)).model.parameters())
    n_params_heavy = sum(p.numel() for p in YOLO(str(WEIGHT_CUSTOM_HEAVY)).model.parameters())

    # 2. Evaluate all 4 models on SVRDD
    metrics_canon = evaluate_checkpoint(canon_pt, "FlyWire-BioNet (Proposed)", n_params_canon)
    metrics_shuff = evaluate_checkpoint(shuff_pt, "Shuffled Connectome Control", n_params_shuff)
    metrics_v8n = evaluate_checkpoint(WEIGHT_YOLOV8N, "Classical CNN (YOLOv8n)", n_params_v8n)
    metrics_heavy = evaluate_checkpoint(WEIGHT_CUSTOM_HEAVY, "Heavy Custom Student (Ref)", n_params_heavy)

    summary_payload = {
        "benchmark_dataset": "SVRDD_YOLO_320 (1,000 validation/test images, 2,612 instances, 7 classes, 320x320)",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "FlyWire_BioNet": metrics_canon,
            "Shuffled_Connectome_Control": metrics_shuff,
            "Classical_CNN_YOLOv8n": metrics_v8n,
            "Heavy_Custom_Student": metrics_heavy,
        },
        "key_comparisons": {
            "param_reduction_vs_YOLOv8n_pct": round((1.0 - n_params_canon / n_params_v8n) * 100, 2),
            "param_reduction_vs_Heavy_Student_pct": round((1.0 - n_params_canon / n_params_heavy) * 100, 2),
            "mAP50_gain_vs_YOLOv8n_pct": round((metrics_canon["mAP50"] - metrics_v8n["mAP50"]) * 100, 2),
            "recall_gain_vs_YOLOv8n_pct": round((metrics_canon["recall"] - metrics_v8n["recall"]) * 100, 2),
            "mAP50_gain_vs_Shuffled_Control_pct": round((metrics_canon["mAP50"] - metrics_shuff["mAP50"]) * 100, 2),
        }
    }

    log_path = LOGS_DIR / "gd2_benchmark_metrics.json"
    log_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[GD2 Benchmark] Saved metrics JSON to {log_path}")

    # 3. Generate Figures
    plot_benchmark_comparison(summary_payload["models"], FIGS_DIR / "gd2_benchmark_comparison.png")
    plot_bio_feature_maps(FIGS_DIR / "gd2_bio_feature_maps.png")


if __name__ == "__main__":
    main()
