# Bước 1: Import các thư viện cần thiết
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

# Bước 2: Tải dữ liệu CSV mà đề bài cung cấp và lưu dữ liệu vào dự án
df = pd.read_csv('../data/USA_Housing.csv')

# In đầu ra của dataframe để kiểm tra
print(df.head())

# Xem mô tả tổng quan
print(df.describe())

# Bước 3: Tạo cột 'Address_Length' tính độ dài của địa chỉ
df['Address_Length'] = df['Address'].apply(len)

# Bước 4: Chọn chỉ các cột số (cả 'Address_Length' bây giờ là số)
df_numeric = df.select_dtypes(include=[np.number])

# Bước 5: Kiểm tra độ tương quan giữa các cột dữ liệu
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm')  # `annot=True` để hiển thị các giá trị trên heatmap
plt.show()

# Bước 6: Chọn các cột đặc trưng (features) và giá trị mục tiêu (target)
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
y = df['Price']

# Bước 7: Chia dữ liệu thành tập huấn luyện và kiểm tra (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

# Bước 8: Khởi tạo mô hình hồi quy tuyến tính
lm = LinearRegression()

# Bước 9: Huấn luyện mô hình với tập huấn luyện
lm.fit(X_train, y_train)

# Kiểm tra kết quả huấn luyện (có thể in hệ số và intercept của mô hình)
print("Intercept:", lm.intercept_)
print("Coefficients:", lm.coef_)

# Bước 10: Sử dụng mô hình để dự báo
predictions = lm.predict(X_test)

# Dự đoán với một mẫu từ X_test (lưu ý: cần dùng iloc[[0]] để có mảng 2D)
pre1 = lm.predict(X_test.iloc[[0]])  # Đảm bảo sử dụng DataFrame (nếu dùng Series sẽ bị lỗi)
print("kết quả 1 =", pre1)

# Dự đoán với một mẫu mới (lưu ý: truyền vào DataFrame với các cột tương ứng)
new_data = pd.DataFrame([[66774.995817, 5.717143, 7.795215, 4.32, 36788.980327]],
                        columns=['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                                 'Avg. Area Number of Bedrooms', 'Area Population'])
pre2 = lm.predict(new_data)
print("kết quả 2 =", pre2)

# Bước 11: Đánh giá mô hình dự báo
print("Intercept:", lm.intercept_)
coeff_df = pd.DataFrame(lm.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# Bước 12: Kết xuất mô hình ra file zip để tái sử dụng
modelname = "housingmodel.zip"
pickle.dump(lm, open(modelname, 'wb'))  # Lưu mô hình

# Bước 13: Tải lại mô hình đã lưu
trained_model = pickle.load(open(modelname, 'rb'))

# Bước 14: Hiển thị hệ số của mô hình đã tải
features = X.columns
coeff_df = pd.DataFrame(trained_model.coef_, features, columns=['Coefficient'])
print(coeff_df)

# Bước 15: Dự đoán với mô hình đã tải
prediction = trained_model.predict([[66774.995817, 5.717143, 7.795215, 4.32, 36788.980327]])
print("kết quả =", prediction)
