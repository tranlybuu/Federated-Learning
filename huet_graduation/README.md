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

### 1. Initial Training Phase (Giai đoạn Training Ban đầu)

Mục đích:
- Xây dựng model cơ sở với dữ liệu từ các chữ số 0-4
- Sử dụng 2 clients để đảm bảo tính phân tán của dữ liệu
- Tạo nền tảng cho giai đoạn training tiếp theo

Phân chia dữ liệu:
- Client 1: Dữ liệu của các chữ số 0, 1, 2
- Client 2: Dữ liệu của các chữ số 3, 4

Cách thực hiện:

1. Khởi động server:
```bash
python -m backend.main --mode initial --server
```

2. Trong terminal mới, khởi động Client 1:
```bash
python -m backend.federated_learning.flwr_client --mode initial --cid 0
```

3. Trong terminal khác, khởi động Client 2:
```bash
python -m backend.federated_learning.flwr_client --mode initial --cid 1
```

Kết quả:
- Model ban đầu được lưu tại `models/initial_model.keras`
- Các model trung gian được lưu theo rounds
- Kết quả training được lưu trong `models/results/initial_training_results.json`

### 2. Additional Training Phase (Giai đoạn Training Bổ Sung)

Mục đích:
- Mở rộng khả năng nhận dạng cho các chữ số 5-9
- Cải thiện model ban đầu với dữ liệu mới
- Tận dụng kiến thức đã học từ giai đoạn initial
- Thử nghiệm với số lượng clients lớn hơn (3 clients)

Phân chia dữ liệu:
- Client 1: Dữ liệu của các chữ số 5, 6
- Client 2: Dữ liệu của các chữ số 7, 8
- Client 3: Dữ liệu của các chữ số 5, 9

Yêu cầu tiên quyết:
- Đã hoàn thành Initial Training Phase
- File `models/initial_model.keras` tồn tại

Cách thực hiện:

1. Khởi động server:
```bash
python -m backend.main --mode additional --server
```

2. Khởi động ba clients trong các terminal riêng biệt:
```bash
python -m backend.federated_learning.flwr_client --mode additional --cid 0
python -m backend.federated_learning.flwr_client --mode additional --cid 1
python -m backend.federated_learning.flwr_client --mode additional --cid 2
```

Kết quả:
- Model cải tiến được lưu theo rounds
- Model tốt nhất được lưu tại `models/best_additional_model.keras`
- Kết quả training được lưu trong `models/results/additional_training_results.json`

### 3. API Server (Server Phục vụ Dự đoán)

Mục đích:
- Cung cấp endpoint cho việc nhận dạng chữ số
- Sử dụng model tốt nhất để dự đoán
- Cung cấp các API để kiểm tra trạng thái và thông tin model

Features:
- Tự động xóa background của ảnh input
- Hỗ trợ cả upload ảnh trực tiếp và URL
- Trả về kết quả dự đoán kèm độ tin cậy
- API endpoints cho health check và thông tin model

Khởi động server:
```bash
python -m backend.main --mode api
```

Sử dụng API:

1. Upload ảnh trực tiếp:
```bash
curl -X POST -F "image=@digit.png" http://localhost:5000/recognize
```

2. Sử dụng URL ảnh:
```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"url":"http://example.com/digit.png"}' \
     http://localhost:5000/recognize
```

3. Kiểm tra trạng thái:
```bash
curl http://localhost:5000/health
```

### 4. Test Client (Client Kiểm thử)

Mục đích:
- Đánh giá và so sánh hiệu suất của các model
- So sánh kết quả giữa model ban đầu và model sau training
- Cung cấp metrics chi tiết về sự cải thiện
- Kiểm tra độ chính xác trên dữ liệu mới

Features:
- So sánh trực tiếp giữa các phiên bản model
- Tính toán mức độ cải thiện
- Xuất báo cáo chi tiết về hiệu suất
- Hỗ trợ test với dữ liệu tùy chỉnh

Sử dụng:

1. Chạy test client:
```bash
python -m backend.main --mode test-only
```

2. So sánh các model:
```python
from backend.federated_learning.flwr_client import TestOnlyClient

test_client = TestOnlyClient()
results = test_client.evaluate_models()
print(results)
```

3. Test với dữ liệu cụ thể:
```python
comparison = test_client.compare_predictions(custom_data)
print(comparison)
```

Kết quả:
- So sánh accuracy giữa các model
- Thống kê về sự cải thiện
- Matrix nhầm lẫn (confusion matrix)
- Báo cáo chi tiết về hiệu suất

### Quy trình Hoạt động Khuyến nghị:

1. Initial Training:
   - Chạy initial training đầy đủ
   - Kiểm tra kết quả trong thư mục results
   - Đảm bảo accuracy đạt mức chấp nhận được (>85%)

2. Additional Training:
   - Chạy additional training với model ban đầu tốt
   - Theo dõi sự cải thiện qua các rounds
   - Lưu ý sự thay đổi trong accuracy

3. Kiểm thử:
   - Sử dụng test client để đánh giá toàn diện
   - So sánh hiệu suất các model
   - Xác định những cải tiến cần thiết

4. Triển khai:
   - Khởi động API server với model tốt nhất
   - Kiểm tra hiệu suất trong thực tế
   - Theo dõi và ghi log hoạt động

### Notes và Best Practices:

1. Training:
   - Luôn backup model trước khi chạy additional training
   - Theo dõi logs để phát hiện vấn đề sớm
   - Điều chỉnh hyperparameters nếu cần

2. Đánh giá:
   - Sử dụng test client thường xuyên
   - So sánh kết quả giữa các lần training
   - Lưu documentation về các thay đổi

3. API Usage:
   - Monitor API performance
   - Implement rate limiting nếu cần
   - Backup model regularl

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