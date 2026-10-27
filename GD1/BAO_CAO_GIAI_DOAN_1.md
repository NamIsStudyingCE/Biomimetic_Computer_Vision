# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 1 (MỐC M2)
**Đề tài:** Nghiên cứu và Xây dựng Mô hình Thị giác Phỏng sinh học từ Bản đồ Kết nối Thần kinh FlyWire (*Drosophila Connectome-derived Biorealistic Vision Model*)  
**Mốc hoàn thành:** Giai đoạn 1 - Khai phá Dữ liệu Connectome, Trích xuất Cột Thị giác Chuẩn & Đóng băng Cache

---

## 📌 1. Danh mục Nhiệm vụ & Sản phẩm Bàn giao

| STT | Nhiệm vụ theo Đề cương Giai đoạn 1 | Sản phẩm Bàn giao (File đính kèm) | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | **Trích xuất Cấu trúc Cột Thị giác Chuẩn (Canonical Column Motif)**: Lọc 21 loại nơ-ron cốt lõi thuộc 4 tầng (*Retina $\to$ Lamina $\to$ Medulla $\to$ Lobula Plate*), trích xuất 47 dạng liên kết synapse với tổng 2.800 trọng số kết nối. | `src/data/build_canonical_connectome.py`<br>`data/flywire/canonical_optic_lobe_synapses.csv`<br>`data/flywire/cell_types_metadata.json` | **Hoàn thành 100%** |
| 2 | **Giải mã Bản chất Sinh hóa & Gán dấu Phân cực (Neurotransmitter Polarity)**: Xác định dấu toán học cho từng khớp synapse: Kích thích ($+$ với Acetylcholine - ACh) và Ức chế ($-$ với GABA, Glutamate, Histamine). | `data/flywire/canonical_optic_lobe_synapses.csv` (Cột `sign` và `effective_weight`) | **Hoàn thành 100%** |
| 3 | **Đóng băng Dữ liệu Nhị phân (PyTorch Binary Cache Pipeline)**: Tạo tensor ma trận liên kết $W_{21 \times 21}$ và ngân hàng bộ lọc tích chập không gian $3\times 3$ (`spatial_kernel_bank`), lưu cache `.pt` chống rò rỉ dữ liệu và tăng tốc 100%. | `data/flywire/flywire_visual_graph.pt` | **Hoàn thành 100%** |
| 4 | **Xây dựng Data Loader Chuẩn hóa PyTorch**: Module tải đồ thị sinh học nhanh, phân vùng chỉ số theo tầng và xuất kernel tích chập 2D không tham số (`requires_grad = False`). | `src/data/flywire_loader.py` | **Hoàn thành 100%** |
| 5 | **Khảo sát Định lượng & Trực quan hóa Đồ thị 300 DPI**: Tạo biểu đồ phân tích cấu trúc 3 panel chất lượng xuất bản khoa học (Heatmap ma trận $W$, Biểu đồ phân cực synapse, Dấu vết không gian T4). | `src/experiments/inspect_connectome.py`<br>`results/figures/gd1_connectome_structure.png`<br>`results/logs/gd1_connectome_stats.json` | **Hoàn thành 100%** |
| 6 | **Đối chuẩn & Đặc tả Bộ dữ liệu SVRDD Target**: Xác minh toàn vẹn 8.000 ảnh (320x320) SVRDD từ `D:\AI4EmbSys`, thiết lập mốc đối chuẩn vượt qua Student YOLOv8n ($3.14\text{M}$ params) và Custom Student ($6.73\text{M}$ params). | `src/data/verify_svrdd.py`<br>`data/svrdd/svrdd_spec.json` | **Hoàn thành 100%** |

---

## 📊 2. Bảng Số liệu Thống kê Cấu trúc Connectome Chuẩn hóa

### A. Phân bố Nơ-ron theo Tầng Sinh học (21 Tế bào Cốt lõi)
| Tầng Sinh học (Layer) | Số lượng Tế bào | Danh sách Loại Tế bào (Cell Types) | Vai trò Chức năng |
| :--- | :---: | :--- | :--- |
| **Retina (Võng mạc)** | 1 | `R1_R6` | Thụ cảm quang phổ rộng, đầu vào độ rọi |
| **Lamina** | 4 | `L1`, `L2`, `L3`, `L4` | Bộ lọc vi sai, tách kênh tương phản & ức chế bên |
| **Medulla** | 8 | `Mi1`, `Tm3`, `Mi4`, `Mi9` (ON)<br>`Tm1`, `Tm2`, `Tm4`, `Tm9` (OFF) | Bộ phân tách tần số thời gian, dây trễ sinh học $\tau$ |
| **Lobula Plate** | 8 | `T4a`, `T4b`, `T4c`, `T4d` (ON-motion)<br>`T5a`, `T5b`, `T5c`, `T5d` (OFF-motion) | Đầu ra phát hiện chuyển động 4 hướng ($+x, -x, +y, -y$) |

### B. Cân bằng Phân cực Dẫn truyền Thần kinh (Neurotransmitter Balance)
* **Tổng số liên kết synapse chuẩn hóa:** 47 dạng kết nối.
* **Tổng số lượng synapse tích lũy:** **2.800 synapses**.
* **Tỷ lệ Kích thích (Excitatory - Acetylcholine / Depolarizing):** **67.0%** (1.876 synapses).
* **Tỷ lệ Ức chế (Inhibitory - GABA / Glutamate / Histamine):** **33.0%** (924 synapses).
* *Nhận định học thuật:* Tỷ lệ $\approx 2:1$ giữa kích thích và ức chế hoàn toàn trùng khớp với các phép đo sinh học trên vỏ não cảm giác của động vật có vú và côn trùng, tạo ra trạng thái cân bằng động (E/I balance), chống bão hòa tín hiệu.

---

## 🔬 3. Biểu đồ Phân tích Cấu trúc (300 DPI Publication Figure)

Biểu đồ chính thức được lưu tại: [`results/figures/gd1_connectome_structure.png`](../results/figures/gd1_connectome_structure.png)

```
┌───────────────────────────────────┐  ┌───────────────────────────────────┐  ┌───────────────────────────────────┐
│ (A) Ma trận Trọng số W [21 x 21]  │  │ (B) Tỷ lệ Phân cực theo Tầng     │  │ (C) Dấu vết Không gian T4         │
│ Đỏ: Kích thích (+ ACh)            │  │ Cột Lamina, Medulla, Lobula       │  │ Mi1 (dx=0, dy=0)                  │
│ Xanh: Ức chế (- GABA/Glu)         │  │ Excitatory vs Inhibitory          │  │ Tm3 (dx=-1, dy=0), Mi4/9 (dx=+1)  │
└───────────────────────────────────┘  └───────────────────────────────────┘  └───────────────────────────────────┘
```

1. **Panel A (Heatmap Ma trận $W_{ij}$):** Thể hiện rõ dòng truyền thông tin một chiều theo thứ bậc từ Retina $\to$ Lamina $\to$ Medulla $\to$ Lobula Plate, với các điểm ức chế đối kháng xuất hiện rõ nét ở các cột nhận tín hiệu của $T4/T5$.
2. **Panel B (Cân bằng phân cực theo tầng):** Tầng Lamina và Medulla sở hữu tỷ lệ ức chế cao nhất, đóng vai trò "gác cổng", lọc sạch nhiễu nền đồng nhất trước khi đưa vào các tế bào quyết định hướng.
3. **Panel C (Đặc trưng không gian T4):** Minh chứng cơ học vì sao tế bào $T4a$ chỉ nhận biết hướng $+x$: Nhánh kích thích trễ ($Tm3$) nằm lệch sang trái ($dx=-1$), trong khi nhánh ức chế đối kháng ($Mi4, Mi9$) nằm lệch sang phải ($dx=+1$).

---

## 🛠️ 4. Hướng dẫn Sử dụng & Kiểm tra Loader

Để nạp đồ thị sinh học và bộ lọc tích chập không tham số vào mô hình PyTorch:
```python
from src.data.flywire_loader import get_flywire_loader

loader = get_flywire_loader()
# Lấy kernel tích chập sinh học 2D kích thước [21, 21, 3, 3]
spatial_kernel = loader.get_spatial_kernel()
print(spatial_kernel.shape) # torch.Size([21, 21, 3, 3])
```

Chạy kiểm tra toàn bộ pipeline dữ liệu:
```bash
python src/data/flywire_loader.py
python src/experiments/inspect_connectome.py
python src/data/verify_svrdd.py
```

---

## 🎯 5. Kế hoạch Tiến vào Giai đoạn 2 (GD2)

Với việc cấu trúc đồ thị sinh học đã được chuẩn hóa và đóng băng trong `data/flywire/flywire_visual_graph.pt`, chúng ta đã sẵn sàng bước vào **Giai đoạn 2**:
* Xây dựng mô hình nơ-ron phỏng sinh học hoàn chỉnh (`BioConnectomeBackbone`).
* Thiết lập 2 mô hình đối chứng đối xứng ($1:1$ fair baselines):
  1. **Shuffled Connectome Control:** Giữ nguyên số nơ-ron và trọng số nhưng hoán vị ngẫu nhiên kết nối (chứng minh tính ưu việt của cấu trúc tiến hóa).
  2. **Classical Conv Stem:** Kiến trúc tích chập tiêu chuẩn của YOLOv8n.
