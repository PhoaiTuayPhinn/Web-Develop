# 🏠 House Pricing Prediction - Bài 5 (PyQt6)

## Mô tả
Ứng dụng desktop PyQt6 để dự đoán giá nhà dựa trên các thông số đầu vào, sử dụng model đã được train từ Bài 4.

## Các thành phần
- **app.py**: File chính để chạy ứng dụng PyQt6
- **FileUtil.py**: Class tiện ích để load/save model
- **housingmodel.zip**: File chứa trained model (LinearRegression)
- **ui/MainWindow.py**: File UI được generate từ Qt Designer
- **ui/MainWindow.ui**: File thiết kế giao diện Qt Designer
- **ui/MainWindowEx.py**: Class mở rộng với logic xử lý

## Cài đặt

### 1. Cài đặt thư viện cần thiết
```bash
pip install PyQt6 scikit-learn pandas numpy
```

### 2. Chạy ứng dụng
```bash
python app.py
```

## Cách sử dụng

### Nhập dữ liệu:
1. **Avg. Area Income**: Thu nhập trung bình khu vực (VD: 79545.46)
2. **Avg. Area House Age**: Tuổi nhà trung bình (VD: 5.68)
3. **Avg. Area Number of Rooms**: Số phòng trung bình (VD: 7.01)
4. **Avg. Area Number of Bedrooms**: Số phòng ngủ trung bình (VD: 4.09)
5. **Area Population**: Dân số khu vực (VD: 23086.80)

### Dự đoán:
- Nhấn nút **"Predictions"** để thực hiện dự đoán
- Kết quả sẽ hiển thị ở trường **"House Pricing Prediction"**
- Một hộp thoại thông báo cũng sẽ hiển thị kết quả

### Tính năng:
- ✅ Tự động load model từ file `housingmodel.zip` khi khởi động
- ✅ Validation đầu vào (kiểm tra các trường có được nhập đầy đủ không)
- ✅ Kiểm tra giá trị số hợp lệ
- ✅ Tự động xóa kết quả khi người dùng thay đổi input
- ✅ Hiển thị giá dự đoán với định dạng tiền tệ
- ✅ Thông báo lỗi chi tiết khi có vấn đề

## Cấu trúc code

### MainWindowEx.py
- `__init__()`: Khởi tạo và khai báo biến model
- `setupUi()`: Thiết lập giao diện và cấu hình ban đầu
- `load_model()`: Load model từ file housingmodel.zip
- `signal_and_slot()`: Kết nối các sự kiện với hàm xử lý
- `predict()`: Thực hiện dự đoán giá nhà
- `on_input_changed()`: Xử lý khi người dùng thay đổi input
- `showWindow()`: Hiển thị cửa sổ ứng dụng

## Lưu ý
- File `housingmodel.zip` phải tồn tại trong thư mục gốc
- Model được train bằng sklearn LinearRegression
- Tất cả các trường phải được nhập đầy đủ trước khi dự đoán
- Các giá trị nhập vào phải là số

## Khắc phục sự cố

### Lỗi "Import PyQt6 could not be resolved"
```bash
pip install PyQt6
```

### Lỗi "Model chưa được load"
- Kiểm tra file `housingmodel.zip` có tồn tại không
- Kiểm tra đường dẫn tới file model

### Lỗi "Vui lòng nhập đầy đủ tất cả các trường"
- Đảm bảo tất cả 5 trường input đã được nhập
- Không để trống bất kỳ trường nào

### Lỗi "Invalid Input"
- Các giá trị phải là số
- Không nhập chữ cái hoặc ký tự đặc biệt

## So sánh với Bài 4 (Flask Web)
| Tính năng | Bài 4 (Web) | Bài 5 (Desktop) |
|-----------|-------------|-----------------|
| Framework | Flask | PyQt6 |
| Interface | HTML/CSS/JS | Qt Widgets |
| Deployment | Web Server | Standalone App |
| User Input | Form HTML | Line Edit |
| Validation | JavaScript + Python | PyQt6 Events |
| Display Result | AJAX Response | QLineEdit |

## Tác giả
Huỳnh Quốc Việt - Faculty of Information Systems
