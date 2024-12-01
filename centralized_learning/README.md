# Centralized Machine Learning for Handwriting Recognition

Đây là các mô hình học máy cho bài toán nhận dạng chữ số viết tay sử dụng tập dữ liệu MNIST.
Trong đó, hình SVM được tham khảo từ Kaggle và mô hình CNN chính là mô hình được sử dụng trong mã nguồn Học liên kết

<hr>

| Mô hình    | Độ chính xác |
|------------|--------------|
| SVM        | 0.97         |
| CNN        | 0.98         |
| FL + CNN   | <= 0.90      |

Đánh giá:
- SVM (Đạt 97%): Kết quả nghiên cứu cho thấy các mô hình huấn luyện trên tập dữ liệu MNIST đạt hiệu suất cao nhất khi dữ liệu được lưu trữ tập trung. Trong đó, mô hình SVM (Support Vector Machine) thể hiện sự ổn định và độ chính xác cao nhờ vào khả năng tối ưu hóa siêu phẳng phân cách giữa các lớp dữ liệu. SVM đặc biệt hiệu quả khi dữ liệu có kích thước nhỏ và phân bố rõ ràng, điều này phù hợp với đặc điểm của tập dữ liệu MNIST.
- CNN (Đạt 98%): Trong khi đó, mô hình CNN (Convolutional Neural Network) không chỉ duy trì độ chính xác cao mà còn vượt trội nhờ khả năng học các đặc trưng phức tạp từ dữ liệu hình ảnh. CNN khai thác hiệu quả cấu trúc không gian của dữ liệu, giúp phân biệt các chữ số viết tay với độ chính xác đáng kể. Việc huấn luyện CNN trên dữ liệu tập trung cho phép mô hình tiếp cận toàn bộ dữ liệu, từ đó tăng cường khả năng tổng quát hóa và giảm thiểu lỗi phân loại.
- FL (sử dụng mô hình CNN ở trên): Khi chuyển sang mô hình Federated Learning (FL) kết hợp với CNN, độ chính xác giảm đáng kể do tính phân tán dữ liệu và hạn chế trong việc trao đổi trực tiếp thông tin giữa các nút mạng. 

<b>Tuy nhiên, mô hình vẫn duy trì một mức độ chính xác đủ cao, cho thấy mô hình này có thể áp dụng hiệu quả trong các tình huống cần bảo mật dữ liệu hoặc khi không thể tập trung dữ liệu hoàn toàn, chẳng hạn như trong lĩnh vực y tế hoặc tài chính. Điều này minh họa rõ ràng sự đánh đổi giữa hiệu suất mô hình và yêu cầu về quyền riêng tư trong các hệ thống học máy phi tập trung.</b>

<hr>

### Tham khảo:
- MNIST-SVM: https://www.kaggle.com/code/adnanzaidi/mnist-svm
