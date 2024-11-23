# Federated Learning for Handwriting Recognition

Dự án này triển khai hệ thống Federated Learning để nhận dạng chữ số viết tay sử dụng tập dữ liệu MNIST. Hệ thống bao gồm một server trung tâm và nhiều clients, cho phép training mô hình một cách phân tán mà không cần chia sẻ dữ liệu trực tiếp.

## Tính năng chính

- Federated Learning sử dụng framework Flower
- Training phân tán với nhiều clients
- API server cho inference
- Model CNN cho nhận dạng chữ số viết tay
- Giao diện command line với nhiều tùy chọn cấu hình
- Lưu trữ và theo dõi tiến trình training
- Hỗ trợ số lượng clients linh hoạt

## Cấu trúc thư mục

```
backend/
│
├── federated_learning/
│   ├── __init__.py
│   ├── flwr_client.py      # Client implementation
│   ├── flwr_server.py      # Server implementation
│   └── model.py            # Neural network model
│
├── data/
│   ├── __init__.py
│   └── data_prep.py        # Data preprocessing
│
├── api/
│   ├── __init__.py
│   └── server.py           # API server
│
├── utils/
│   ├── __init__.py
│   └── config.py           # Configuration
│
├── main.py                 # Entry point
└── requirements.txt        # Dependencies
```

## Cài đặt

1. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Trên Windows: venv\Scripts\activate
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Cấu hình

Các cấu hình chính được định nghĩa trong `utils/config.py`:

- `FL_CONFIG`: Cấu hình cho Federated Learning (số rounds, số clients tối thiểu, ...)
- `DATA_CONFIG`: Cấu hình cho dữ liệu và training (batch size, epochs, ...)
- `MODEL_CONFIG`: Cấu hình cho model
- `API_CONFIG`: Cấu hình cho API server

## Sử dụng

### 1. Federated Learning Server

Chạy server với cấu hình mặc định:
```bash
python -m backend.main --mode fl_server
```

Tùy chọn cấu hình:
```bash
python -m backend.main --mode fl_server --min_clients 4 --num_rounds 8
```

Xem tất cả các tùy chọn:
```bash
python -m backend.main --help
```

Các flags cho server:
- `--mode`: Chế độ chạy (fl_server hoặc api)
- `--num_rounds`: Số rounds training
- `--min_clients`: Số clients tối thiểu cần thiết
- `--fraction_fit`: Tỷ lệ clients sử dụng cho training
- `--fraction_evaluate`: Tỷ lệ clients sử dụng cho evaluation
- `--batch_size`: Kích thước batch
- `--local_epochs`: Số epochs training local

### 2. Federated Learning Clients

Chạy client với cấu hình mặc định:
```bash
python -m backend.federated_learning.flwr_client --cid 0
```

Tùy chọn cấu hình:
```bash
python -m backend.federated_learning.flwr_client --cid 0 --batch_size 64 --local_epochs 2
```

Xem tất cả các tùy chọn:
```bash
python -m backend.federated_learning.flwr_client --help
```

Các flags cho client:
- `--cid`: ID của client (bắt buộc)
- `--server_address`: Địa chỉ server
- `--batch_size`: Kích thước batch
- `--local_epochs`: Số epochs training local
- `--validation_split`: Tỷ lệ dữ liệu validation
- `--verbose`: Mức độ hiển thị thông tin (0, 1, 2)

### 3. API Server

Chạy API server:
```bash
python -m backend.main --mode api
```

API Endpoints:
- `/recognize`: POST endpoint cho nhận dạng chữ số
- `/health`: GET endpoint kiểm tra trạng thái server

## Quy trình sử dụng

1. Khởi động server:
```bash
python -m backend.main --mode fl_server --min_clients 3 --num_rounds 5
```

2. Khởi động các clients (trong các terminal khác nhau):
```bash
python -m backend.federated_learning.flwr_client --cid 0
python -m backend.federated_learning.flwr_client --cid 1
python -m backend.federated_learning.flwr_client --cid 2
```

3. Sau khi training xong, chạy API server:
```bash
python -m backend.main --mode api
```

## Kết quả và Log

Sau khi training, hệ thống sẽ tạo:
- Model files trong thư mục `models/`
- Training history trong `models/training_history.json`
- Log files trong terminal

## Troubleshooting

### Server không khởi động
- Kiểm tra port 8080 có đang được sử dụng không
- Đảm bảo đủ quyền để tạo thư mục models/

### Client không kết nối được
- Kiểm tra server đã chạy chưa
- Xác nhận địa chỉ server chính xác
- Đảm bảo Client ID là duy nhất

### Lỗi training
- Kiểm tra RAM đủ cho batch size đã chọn
- Giảm batch size nếu gặp lỗi memory
- Tăng số rounds nếu accuracy chưa đạt yêu cầu

## Dependencies chính

- Python ≥ 3.7
- TensorFlow
- Flower
- Flask
- NumPy
- Matplotlib

## Đóng góp

Vui lòng mở issue hoặc pull request nếu bạn muốn đóng góp cho project.

## License

[MIT License](LICENSE)