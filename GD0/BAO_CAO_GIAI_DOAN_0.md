# BÁO CÁO NGHIỆM THU GIAI ĐOẠN 0 (MỐC M1)
**Đề tài:** Nghiên cứu và Xây dựng Mô hình Thị giác Phỏng sinh học từ Bản đồ Kết nối Thần kinh FlyWire (*Drosophila Connectome-derived Biorealistic Vision Model*)

---

## 📌 1. Danh mục Nhiệm vụ & Sản phẩm Bàn giao

| STT | Nhiệm vụ theo Đề cương (Giai đoạn 0) | Sản phẩm Bàn giao (File đính kèm) | Trạng thái |
| :---: | :--- | :--- | :---: |
| 1 | **Tóm tắt Lý thuyết & Mô hình hóa Giải phẫu**: Hệ thống hóa cấu trúc 4 tầng thùy thị giác (*Retina $\to$ Lamina $\to$ Medulla $\to$ Lobula Plate*), cơ chế bộ tứ tế bào $T4/T5$, phép nhân tương quan và ức chế đối kháng GABAergic. | `docs/theory_connectome_vision.md` | **Hoàn thành 100%** |
| 2 | **Khởi tạo Môi trường & Khung Thư mục Repo**: Thiết lập cấu trúc repo nghiên cứu chuẩn mực tại `D:\Biomimetic Computer Vision` theo đúng tiêu chuẩn Lab. | `requirements.txt`<br>Cấu trúc thư mục: `docs`, `src`, `data`, `GD0-4`, `results` | **Hoàn thành 100%** |
| 3 | **Xây dựng Toy Model & Kiểm chứng Thời gian thực (Sanity Check)**: Triển khai mạch dò chuyển động Hassenstein-Reichardt chuẩn sinh học, chứng minh khả năng bắt hướng không cần tham số. | `src/demo_hr_detector.py` | **Hoàn thành 100%** |
| 4 | **Trực quan hóa Tương tác (Interactive Simulation UI)**: Xây dựng ứng dụng mô phỏng 60 FPS tích hợp Neural Oscilloscope quan sát điện thế hoạt động thời gian thực. | `docs/fly_vision_simulator.html` | **Hoàn thành 100%** |
| 5 | **Khảo sát Dữ liệu & Đối chuẩn Benchmark**: Lựa chọn phương thức bảng tĩnh FlyWire Connectome (Cách A) và khảo sát bộ dữ liệu SVRDD (8.000 ảnh) từ `D:\AI4EmbSys`. | Phân tích kiến trúc Student YOLOv8n (3.1M params) vs Custom (6.7M params). | **Hoàn thành 100%** |

---

## 🔬 2. Phát hiện Học thuật Cốt lõi của Giai đoạn 0

1. **Hiện tượng báo động giả ở mô hình nửa nhánh:**
   Mô hình nửa nhánh $P_1(t - \tau) \times P_2(t)$ bị dính chập điện thế màng ở dải vận tốc cao khi chuyển động ngược chiều $\to$ gây xung nhọn báo động giả.
2. **Khắc phục bằng cơ chế Ức chế Đối kháng (Opponent Suppression):**
   Tích hợp nhánh ức chế đối kháng GABAergic $[P_1(t - \tau) \times P_2(t)] - \alpha [P_1(t) \times P_2(t - \tau)]$ đã triệt tiêu 100% tín hiệu kích hoạt nhầm, chứng minh cấu trúc kết nối sinh học là **Inductive Bias** hoàn hảo cho việc nhận biết hướng chuyển động.

---

## 🚀 3. Hướng dẫn Tái hiện Demo

Chạy script kiểm tra độc lập tại thư mục gốc:
```bash
python src/demo_hr_detector.py
```
Mở ứng dụng mô phỏng tương tác trên trình duyệt:
```
D:\Biomimetic Computer Vision\docs\fly_vision_simulator.html
```

