# Dự án áp dụng transfer learning cho bài toán phân loại tổn thương da.

## Tổng quan

Project này tập trung vào việc huấn luyện và so sánh các mô hình học sâu cho bài toán image classification trên dữ liệu da liễu.  
Quy trình chính gồm: chuẩn bị dữ liệu, huấn luyện mô hình trên dữ liệu nguồn, fine-tune trên dữ liệu đích, đánh giá kết quả và triển khai demo ứng dụng đơn giản.

## Mục tiêu

- Xây dựng pipeline huấn luyện cho bài toán phân loại ảnh.
- So sánh hiệu quả giữa các mô hình được sử dụng.
- Đánh giá mô hình bằng các chỉ số phù hợp.
- Triển khai một ứng dụng demo đơn giản để thử nghiệm dự đoán.

## Cấu trúc thư mục

```bash
project/
├── data/
├── src/
├── notebooks/
├── checkpoints/
├── logs/
├── results/
├── streamlit_app/
├── docs/
└── README.md
```

### Mô tả các thư mục

- `data/`: chứa dữ liệu gốc, dữ liệu đã xử lý và các file chia tập.
- `src/`: chứa mã nguồn chính của project.
- `notebooks/`: chứa notebook dùng để thử nghiệm, phân tích và kiểm tra nhanh.
- `checkpoints/`: chứa trọng số mô hình đã huấn luyện.
- `logs/`: chứa log trong quá trình train và fine-tune.
- `results/`: chứa kết quả đầu ra như bảng metric, hình ảnh và báo cáo tổng hợp.
- `streamlit_app/`: chứa phần code triển khai ứng dụng demo.
- `docs/`: chứa tài liệu bổ sung như báo cáo, hình minh họa hoặc ghi chú.

## Quy trình thực hiện

1. Chuẩn bị và tiền xử lý dữ liệu.
2. Huấn luyện mô hình trên dữ liệu nguồn.
3. Fine-tune mô hình trên dữ liệu đích.
4. Đánh giá và so sánh kết quả.
5. Triển khai demo ứng dụng.
