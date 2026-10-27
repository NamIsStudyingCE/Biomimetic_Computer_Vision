# Drosophila FlyWire Connectome-derived Biorealistic Computer Vision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![FlyWire Connectome](https://img.shields.io/badge/FlyWire-Nature%202024-green.svg)](https://flywire.ai/)

> **Nghiên cứu & Xây dựng Mô hình Thị giác Phỏng sinh học từ Bản đồ Kết nối Thần kinh Não bộ Ruồi Giấm (FlyWire Drosophila Connectome)**  
> Ứng dụng cấu trúc kết nối nơ-ron sinh học (Optic Lobe) làm bộ trích xuất đặc trưng không tham số (Zero-parameter Feature Extractor) cho bài toán thị giác máy tính và phân tích khuyết tật mặt đường (SVRDD Benchmark).

---

## 📌 1. Bối cảnh & Mục tiêu Nghiên cứu

Nghiên cứu này mô hình hóa hệ thống thị giác của ruồi giấm (*Drosophila melanogaster*) dựa trên công trình giải mã bản đồ kết nối thần kinh toàn diện (FlyWire Consortium, *Nature* 2024).

### Mục tiêu cốt lõi:
1. **Kiến trúc Phỏng sinh học (Biomimetic Inductive Bias):**
   * Sử dụng cấu trúc ức chế bên (Lateral Inhibition) tầng Lamina ($L1, L2$) và mạng nơ-ron dò chuyển động đa hướng tầng Lobula Plate ($T4/T5$) theo nguyên lý Hassenstein-Reichardt đối kháng toàn phần (Full Opponent Suppression).
2. **Siêu tiết kiệm Tham số (Ultra-lightweight Efficiency):**
   * Thay thế các khối tích chập nặng nề ở tầng nông của mạng CNN truyền thống bằng **bộ lọc sinh học có 0 tham số huấn luyện**, giúp mô hình có kích thước siêu nhẹ ($\sim 1.5\text{M} - 2.5\text{M}$ tham số) nhưng nhạy bén vượt trội với các cấu trúc vi mô tần số cao (vết nứt mảnh, đường ranh giới).
3. **Đối chuẩn thực nghiệm công bằng (Fair Benchmark):**
   * Đánh giá sòng phẳng với mô hình Student YOLOv8n ($3.14\text{M}$ params) và Custom Student ($6.73\text{M}$ params) trên bộ dữ liệu **SVRDD (8.000 ảnh, 7 lớp hư hại mặt đường)**.

---

## 🚀 2. Lộ trình Triển khai (GD0 - GD4)

| Giai đoạn | Nội dung cốt lõi | Sản phẩm bàn giao | Trạng thái |
| :---: | :--- | :--- | :---: |
| **GD0** | **Nền tảng, Toy Model & Kiểm chứng Thực nghiệm** | Mô phỏng EMD đối kháng, UI tương tác 60 FPS, khảo sát SVRDD | [`GD0/BAO_CAO_GIAI_DOAN_0.md`](GD0/BAO_CAO_GIAI_DOAN_0.md) | ✅ **Hoàn thành** |
| **GD1** | **Trích xuất Connectome & Cột Thị giác Lục giác** | Parser bảng tĩnh FlyWire, ma trận kề $W_{ij}$ có dấu (GABA/ACh), Biểu đồ 300 DPI | [`GD1/BAO_CAO_GIAI_DOAN_1.md`](GD1/BAO_CAO_GIAI_DOAN_1.md) | ✅ **Hoàn thành** |
| **GD2** | **Mô hình Hóa Đồ thị & Symmetrical Baselines** | `FlyWire-BioNet` (2.92M, 45.33% mAP50), `Shuffled Control`, `Classical CNN` | [`GD2/BAO_CAO_GIAI_DOAN_2.md`](GD2/BAO_CAO_GIAI_DOAN_2.md) | ✅ **Hoàn thành** |
| **GD3** | **Khảo sát Triệt tiêu (Ablation) & Đánh giá SVRDD** | Cắt dây thần kinh ảo, đối chuẩn 10 seed deterministic, p-value | `results/` | 🔄 Kế tiếp |
| **GD4** | **Bài báo Khoa học & Đóng gói Tái lập** | Bản thảo LaTeX, Artifacts tái lập 1 lệnh | `paper/` & `slides/` | ⏳ Chuẩn bị |

---

## 🛠️ 3. Cài đặt & Tái hiện Pipeline GD1 - GD2

```bash
# 1. Khởi tạo môi trường ảo
python -m venv .venv
.venv\Scripts\activate   # Trên Windows

# 2. Cài đặt thư viện phụ thuộc
pip install -r requirements.txt

# 3. Trích xuất Connectome chuẩn hóa & xây dựng PyTorch Binary Cache (GD1)
python src/data/build_canonical_connectome.py

# 4. Kiểm tra cấu trúc & xuất biểu đồ phân tích 300 DPI (GD1)
python src/experiments/inspect_connectome.py

# 5. Đóng gói mô hình FlyWire-BioNet, Shuffled Control & chạy Benchmark SVRDD (GD2)
python -m src.experiments.run_gd2_benchmark
```

### Biểu đồ Kết quả Chính thức (300 DPI):
- **GD1 (Cấu trúc Connectome):** `results/figures/gd1_connectome_structure.png`
- **GD2 (Đối sánh Benchmark & Pareto):** `results/figures/gd2_benchmark_comparison.png`
- **GD2 (Bản đồ Hoạt hóa 4 Tầng Thùy Thị giác):** `results/figures/gd2_bio_feature_maps.png`

---

## 📂 4. Cấu trúc Thư mục Dự án

```
D:\Biomimetic Computer Vision\
├── data/
│   ├── flywire/             # Bảng kết nối synapse & cache nhị phân .pt
│   │   ├── canonical_optic_lobe_synapses.csv
│   │   ├── cell_types_metadata.json
│   │   └── flywire_visual_graph.pt
│   └── svrdd/               # Đặc tả & liên kết SVRDD_YOLO_320 (8.000 ảnh)
│       └── svrdd_spec.json
├── docs/
│   ├── theory_connectome_vision.md    # Tài liệu lý thuyết nền tảng & toán học
│   └── fly_vision_simulator.html      # Ứng dụng mô phỏng tương tác 60 FPS
├── GD0/
│   └── BAO_CAO_GIAI_DOAN_0.md         # Báo cáo nghiệm thu chính thức Giai đoạn 0
├── GD1/
│   └── BAO_CAO_GIAI_DOAN_1.md         # Báo cáo nghiệm thu chính thức Giai đoạn 1
├── GD2/
│   └── BAO_CAO_GIAI_DOAN_2.md         # Báo cáo nghiệm thu chính thức Giai đoạn 2
├── GD3/ ... GD4/                      # Các thư mục giai đoạn tiếp theo
├── results/
│   ├── figures/
│   │   ├── gd1_connectome_structure.png  # Biểu đồ cấu trúc Connectome (300 DPI)
│   │   ├── gd2_benchmark_comparison.png  # Biểu đồ đối sánh hiệu năng SVRDD (300 DPI)
│   │   └── gd2_bio_feature_maps.png      # Bản đồ đặc trưng thần kinh 4 tầng (300 DPI)
│   └── logs/
│       ├── gd1_connectome_stats.json     # Số liệu thống kê Connectome
│       └── gd2_benchmark_metrics.json    # Số liệu đánh giá 4 mô hình trên SVRDD
├── src/
│   ├── data/
│   │   ├── build_canonical_connectome.py # Generator bảng kết nối
│   │   ├── flywire_loader.py             # PyTorch Data Loader
│   │   └── verify_svrdd.py               # Kiểm tra bộ dữ liệu SVRDD
│   ├── demo_hr_detector.py            # Code kiểm chứng EMD đối kháng
│   ├── models/
│   │   └── flywire_bionet.py          # Kiến trúc FlyWire-BioNet & Shuffled Control
│   ├── utils/
│   └── experiments/
│       ├── inspect_connectome.py      # Visualizer & Metrics Inspector (GD1)
│       └── run_gd2_benchmark.py       # Benchmark & Feature Map Visualizer (GD2)
├── requirements.txt
└── README.md
```

---

## 📜 5. Bản quyền & Trích dẫn

Dự án được phân phối dưới giấy phép **MIT License**.  
Mô hình kết nối thần kinh dựa trên dữ liệu công bố mở của **FlyWire Consortium** (*Nature*, 2024).
