# Lý thuyết Nền tảng: Thị giác Phỏng sinh học từ Bản đồ Kết nối Thần kinh FlyWire

> **Đề tài:** Nghiên cứu và Xây dựng Mô hình Thị giác Phỏng sinh học từ Bản đồ Kết nối Thần kinh FlyWire (*Drosophila Connectome-derived Biorealistic Vision Model*)

---

## 1. Giới thiệu Cấu trúc Giải phẫu Thùy Thị giác Ruồi (Optic Lobe)

Não bộ ruồi giấm (*Drosophila melanogaster*) gồm khoảng 139.000 nơ-ron và 50 triệu liên kết synapse (FlyWire Consortium, *Nature* 2024). Trong đó, **hơn 60% nơ-ron dành riêng cho hệ thống xử lý thị giác (Optic Lobe)**.

Hệ thống thị giác ruồi được tổ chức theo cấu trúc dạng tầng đa lớp (Layer-by-layer) phân cấp tương tự mạng Deep Learning:
1. **Mắt kép (Compound Eye / Retina):** Gồm khoảng 800 mắt con (ommatidia) xếp theo lưới lục giác (Hexagonal grid). Mỗi mắt con chứa các tế bào thụ cảm quang $R1-R8$.
2. **Tầng Lamina:** Tiếp nhận tín hiệu từ $R1-R6$, đóng vai trò bộ tiền xử lý lọc nhiễu, tăng độ tương phản cục bộ qua cơ chế ức chế bên (Lateral Inhibition) từ các nơ-ron $L1, L2, L3$.
3. **Tầng Medulla:** Tách tín hiệu thành 2 kênh độc lập:
   - Kênh **ON** (phát hiện độ sáng tăng): Nơ-ron $Mi1, Tm3$.
   - Kênh **OFF** (phát hiện bóng tối/độ sáng giảm): Nơ-ron $Tm1, Tm2, Tm9$.
4. **Tầng Lobula & Lobula Plate:** Bộ vi xử lý tính toán chuyển động đa hướng gồm:
   - Bộ tứ tế bào $T4$ (ON-motion): $T4a$ (Phải), $T4b$ (Trái), $T4c$ (Lên), $T4d$ (Xuống).
   - Bộ tứ tế bào $T5$ (OFF-motion): $T5a, T5b, T5c, T5d$.
   - Tế bào khổng lồ tiếp hợp ($HS, VS, LGMD$): Gom tín hiệu toàn trường thị giác để phát hiện va chạm khẩn cấp.

---

## 2. Mô hình Máy dò Chuyển động Hassenstein-Reichardt (HR Detector)

### A. Mô hình Nửa nhánh (Half-Detector)
$$R_{\text{half}}(t) = P_1(t - \tau) \times P_2(t)$$
- **Hạn chế:** Khi vật thể chuyển động ngược rất nhanh hoặc ánh sáng nền nhấp nháy, tín hiệu đuôi Gauss bị dính chập, gây báo động giả (False-positive).

### B. Mô hình Đối kháng Đầy đủ (Full Opponent Suppression)
Để khắc phục triệt để báo động giả, sinh học sử dụng hai nhánh kích thích và ức chế đối kháng (GABAergic):
$$R_{\text{Full}}(t) = \max\Big(0,\ \big[P_1(t - \tau_1) \times P_2(t)\big] - \alpha \cdot \big[P_1(t) \times P_2(t - \tau_2)\big]\Big)$$
- Khi chuyển động thuận: Nhánh thuận dương lớn, nhánh nghịch bằng 0 $\to$ Phát xung cực đại.
- Khi chuyển động nghịch: Nhánh nghịch lớn hơn nhánh thuận, phép trừ dập tắt tín hiệu về 0 $\to$ Triệt tiêu 100% báo động giả.

---

## 3. Cầu nối với Bài toán Thị giác Máy tính Biên (Embedded Computer Vision - SVRDD)

Trong bài toán phát hiện khuyết tật mặt đường (Road Surface Damage Detection - SVRDD):
- **Đặc trưng mục tiêu:** Vết nứt (cracks), ổ gà (potholes) là các đối tượng có dải tần số không gian cao (mảnh, kéo dài, độ tương phản thấp).
- **Hạn chế của CNN truyền thống (YOLOv8 Stem):** Các lớp tích chập dày đặc và pooling làm mờ biên độ vết nứt và ngốn nhiều tham số tính toán.
- **Ưu thế phỏng sinh học:** Cơ chế ức chế bên (Lateral Inhibition) của tầng Lamina/Medulla hoạt động như bộ lọc High-Pass vi sai tự nhiên với **0 tham số huấn luyện**, giúp làm sắc nét biên nứt trước khi đưa vào các tầng nhận diện sâu hơn.

