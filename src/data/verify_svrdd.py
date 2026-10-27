"""
SVRDD Dataset Verification and Specification
Biomimetic Computer Vision Project
"""

from pathlib import Path
import json

SVRDD_SOURCE = Path(r"D:\AI4EmbSys\SVRDD_YOLO_320")
TARGET_DIR = Path(r"D:\Biomimetic Computer Vision\data\svrdd")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def verify_svrdd():
    print("=" * 70)
    print("  SVRDD BENCHMARK DATASET SPECIFICATION & VERIFICATION")
    print("=" * 70)

    if not SVRDD_SOURCE.exists():
        raise FileNotFoundError(f"Source SVRDD directory not found at {SVRDD_SOURCE}")

    train_txt = SVRDD_SOURCE / "train.txt"
    val_txt = SVRDD_SOURCE / "val.txt"
    test_txt = SVRDD_SOURCE / "test.txt"
    data_yaml = SVRDD_SOURCE / "data.yaml"

    train_lines = len(train_txt.read_text().strip().splitlines()) if train_txt.exists() else 0
    val_lines = len(val_txt.read_text().strip().splitlines()) if val_txt.exists() else 0
    test_lines = len(test_txt.read_text().strip().splitlines()) if test_txt.exists() else 0

    spec = {
        "dataset_name": "SVRDD (Road Surface Damage Detection)",
        "source_path": str(SVRDD_SOURCE),
        "data_yaml_path": str(data_yaml),
        "image_resolution": [320, 320, 3],
        "num_classes": 7,
        "class_names": ["damage1", "damage2", "damage3", "damage4", "damage5", "damage6", "damage7"],
        "splits": {
            "train": train_lines,
            "val": val_lines,
            "test": test_lines,
            "total": train_lines + val_lines + test_lines
        },
        "target_baselines": {
            "yolov8n_student_standard": {
                "params": 3139461,
                "mAP50": 0.3797,
                "recall": 0.3698
            },
            "yolov8_custom_student_kd": {
                "params": 6726661,
                "mAP50": 0.4657,
                "recall": 0.4497
            }
        }
    }

    out_file = TARGET_DIR / "svrdd_spec.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(spec, f, indent=2, ensure_ascii=False)

    print(f"[OK] SVRDD specification verified and saved to: {out_file}")
    print(f"Total verified samples: {spec['splits']['total']} (Train: {train_lines}, Val: {val_lines}, Test: {test_lines})")
    print(f"Target student to surpass: {spec['target_baselines']['yolov8_custom_student_kd']['params']} params, mAP50={spec['target_baselines']['yolov8_custom_student_kd']['mAP50']}")
    print("=" * 70)

if __name__ == "__main__":
    verify_svrdd()
