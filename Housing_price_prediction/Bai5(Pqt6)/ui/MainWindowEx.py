from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QTableWidgetItem, QMessageBox
import os
import sys

# Thêm đường dẫn để import FileUtil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FileUtil import FileUtil
from ui.MainWindow import Ui_MainWindow


class MainWindowEx(Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        
    def setupUi(self, MainWindow):
        super().setupUi(MainWindow)
        self.MainWindow = MainWindow
        
        # Load model khi khởi tạo
        self.load_model()
        
        # Kết nối signals và slots
        self.signal_and_slot()
        
        # Set placeholder text cho các input
        self.lineEditIncome.setPlaceholderText("e.g., 79545.46")
        self.lineEditHouseAge.setPlaceholderText("e.g., 5.68")
        self.lineEditNumOfRoom.setPlaceholderText("e.g., 7.01")
        self.lineEditNumOfBedrooms.setPlaceholderText("e.g., 4.09")
        self.lineEditPopulaiton.setPlaceholderText("e.g., 23086.80")
        
        # Set prediction field là read-only
        self.lineEditPrediction.setReadOnly(True)
        
    def load_model(self):
        """Load model từ file housingmodel.zip"""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "housingmodel.zip")
            self.model = FileUtil.load_model(model_path)
            if self.model is None:
                QMessageBox.warning(None, "Warning", "Không thể load model từ housingmodel.zip!")
            else:
                print("Model loaded successfully!")
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Lỗi khi load model: {str(e)}")
            
    def signal_and_slot(self):
        """Kết nối các signals và slots"""
        self.pushButton.clicked.connect(self.predict)
        self.lineEditIncome.textChanged.connect(self.on_input_changed)
        self.lineEditHouseAge.textChanged.connect(self.on_input_changed)
        self.lineEditNumOfRoom.textChanged.connect(self.on_input_changed)
        self.lineEditNumOfBedrooms.textChanged.connect(self.on_input_changed)
        self.lineEditPopulaiton.textChanged.connect(self.on_input_changed)
        
    def predict(self):
        """Thực hiện prediction khi nhấn button"""
        try:
            # Kiểm tra model đã được load chưa
            if self.model is None:
                QMessageBox.warning(self.MainWindow, "Warning", "Model chưa được load!")
                return
            
            # Lấy giá trị từ các input
            income_text = self.lineEditIncome.text().strip()
            house_age_text = self.lineEditHouseAge.text().strip()
            num_of_rooms_text = self.lineEditNumOfRoom.text().strip()
            num_of_bedrooms_text = self.lineEditNumOfBedrooms.text().strip()
            population_text = self.lineEditPopulaiton.text().strip()
            
            # Kiểm tra tất cả các trường đã được nhập chưa
            if not all([income_text, house_age_text, num_of_rooms_text, 
                       num_of_bedrooms_text, population_text]):
                QMessageBox.warning(self.MainWindow, "Warning", 
                                  "Vui lòng nhập đầy đủ tất cả các trường!")
                return
            
            # Chuyển đổi sang float
            income = float(income_text)
            house_age = float(house_age_text)
            num_of_rooms = float(num_of_rooms_text)
            num_of_bedrooms = float(num_of_bedrooms_text)
            population = float(population_text)
            
            # Thực hiện prediction
            result = self.model.predict([[
                income,
                house_age,
                num_of_rooms,
                num_of_bedrooms,
                population
            ]])
            
            # Hiển thị kết quả
            prediction_value = result[0]
            self.lineEditPrediction.setText(f"${prediction_value:,.2f}")
            
            # Hiển thị thông báo thành công
            QMessageBox.information(self.MainWindow, "Success", 
                                  f"Giá nhà dự đoán: ${prediction_value:,.2f}")
            
        except ValueError as e:
            QMessageBox.warning(self.MainWindow, "Invalid Input", 
                              "Vui lòng nhập các giá trị số hợp lệ!")
        except Exception as e:
            QMessageBox.critical(self.MainWindow, "Error", 
                               f"Lỗi khi thực hiện prediction:\n{str(e)}")
    
    def on_input_changed(self):
        """Xóa kết quả prediction khi người dùng thay đổi input"""
        self.lineEditPrediction.clear()
        
    def showWindow(self):
        """Hiển thị cửa sổ"""
        self.MainWindow.show()
