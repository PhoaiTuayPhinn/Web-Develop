from PyQt6.QtWidgets import QApplication, QMainWindow
from ui.MainWindowEx import MainWindowEx
import sys

if __name__ == "__main__":
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = MainWindowEx()
    ui.setupUi(MainWindow)
    ui.showWindow()
    sys.exit(app.exec())


