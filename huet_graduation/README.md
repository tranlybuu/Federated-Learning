# Federated Learning for Handwriting Recognition

Dự án này triển khai hệ thống Federated Learning cho bài toán nhận dạng chữ số viết tay sử dụng tập dữ liệu MNIST. Hệ thống được thiết kế theo mô hình client-server với ba giai đoạn training riêng biệt và một API server cho inference.

## Tính năng chính

- **Federated Learning đa giai đoạn**:
  - Giai đoạn Initial Training với 2 clients
  - Giai đoạn Additional Training với 3 clients
  - Test-only client cho đánh giá và so sánh

- **Xử lý dữ liệu thông minh**:
  - Phân chia dữ liệu theo ranges cho từng giai đoạn
  - Tự động xử lý và chuẩn hóa dữ liệu
  - Background removal cho ảnh input

- **Model Management**:
  - Lưu trữ và quản lý nhiều phiên bản model
  - Theo dõi model tốt nhất của mỗi giai đoạn
  - So sánh hiệu suất giữa các phiên bản

- **API Service**:
  - REST API cho inference
  - Hỗ trợ cả image upload và URL
  - Health check và monitoring

## Cấu trúc thư mục

```plaintext
backend/
│
├── federated_learning/        # Core Federated Learning implementation
│   ├── __init__.py
│   ├── flwr_client.py        # Client implementation
│   ├── flwr_server.py        # Server implementation
│   └── model.py              # Model architecture
│
├── api/                      # API Server
│   ├── __init__.py
│   └── server.py            # Flask API implementation
│
├── utils/                    # Utilities and Configuration
│   ├── __init__.py
│   └── config.py            # Central configuration
│
├── models/                   # Model storage
│   ├── initial_model.keras
│   ├── global_model_round_*.keras
│   └── best_model.keras
│
└── main.py                  # Entry point
```

## Cài đặt

1. Tạo và kích hoạt môi trường ảo:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

2. Cài đặt dependencies:
```bash
pip install -r requirements.txt
```

## Cấu hình

Toàn bộ cấu hình được tập trung trong `utils/config.py`:

- `MODEL_TEMPLATES`: Đường dẫn và template cho các model
- `TRAINING_CONFIG`: Cấu hình cho các giai đoạn training
- `FL_CONFIG`: Cấu hình Federated Learning
- `DATA_CONFIG`: Cấu hình dữ liệu và training
- `API_CONFIG`: Cấu hình API server

## Sử dụng

### 1. Initial Training Phase

Chạy server:
```bash
python -m backend.main --mode initial --server
```

Chạy clients (trong các terminal khác):
```bash
python -m backend.federated_learning.flwr_client --mode initial --cid 0
python -m backend.federated_learning.flwr_client --mode initial --cid 1
```

### 2. Additional Training Phase

Chạy server:
```bash
python -m backend.main --mode additional --server
```

Chạy clients:
```bash
python -m backend.federated_learning.flwr_client --mode additional --cid 0
python -m backend.federated_learning.flwr_client --mode additional --cid 1
python -m backend.federated_learning.flwr_client --mode additional --cid 2
```

### 3. API Server

```bash
python -m backend.main --mode api
```

### 4. Test Client

```bash
python -m backend.main --mode test-only
```

## Tùy chỉnh Training

Server với cấu hình tùy chỉnh:
```bash
python -m backend.main --mode initial --server --num_rounds 5 --min_clients 3
```

Client với hyperparameters tùy chỉnh:
```bash
python -m backend.federated_learning.flwr_client \
    --mode initial \
    --cid 0 \
    --batch_size 64 \
    --local_epochs 2 \
    --validation_split 0.2
```

## API Endpoints

- `/recognize`: POST - Nhận dạng chữ số từ ảnh
  - Hỗ trợ direct upload hoặc image URL
  - Tự động xử lý và xóa background
  - Trả về chữ số dự đoán và confidence score

- `/health`: GET - Kiểm tra trạng thái server
  - Trạng thái server
  - Thông tin model đang sử dụng
  - Metrics cơ bản

## Monitoring và Logs

- Training logs được lưu trong `models/results/`
- Model checkpoints trong `models/`
- API server logs trong standard output

## Cấu trúc dữ liệu

Initial Training:
- Client 1: Data ranges 0-2
- Client 2: Data ranges 3-4

Additional Training:
- Client 1: Data ranges 5-6
- Client 2: Data ranges 7-8
- Client 3: Data ranges 5,9

## Troubleshooting

1. Server không khởi động:
   - Kiểm tra port 8080 có đang được sử dụng không
   - Đảm bảo đủ quyền truy cập thư mục `models`

2. Client không kết nối được:
   - Kiểm tra server đã chạy chưa
   - Xác nhận địa chỉ server chính xác
   - Đảm bảo Client ID là duy nhất

3. Lỗi training:
   - Kiểm tra RAM đủ cho batch size đã chọn
   - Xem xét giảm batch size
   - Kiểm tra logs trong `models/results/`

## Dependencies chính

- Python ≥ 3.7
- TensorFlow
- Flower
- Flask
- NumPy
- Rembg (cho background removal)
- Pillow

## Đóng góp

Vui lòng mở issue hoặc pull request nếu bạn muốn đóng góp cho project.

## License

[MIT License](LICENSE)