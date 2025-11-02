import pickle


class FileUtil:

    @staticmethod
    def save_model(model, filename):
        """
        Lưu mô hình vào file với tên được chỉ định.

        :param model: Mô hình cần lưu
        :param filename: Tên file lưu mô hình
        :return: True nếu lưu thành công, False nếu có lỗi
        """
        try:
            with open(filename, 'wb') as file:
                pickle.dump(model, file)
            return True
        except Exception as e:
            print(f"An exception occurred: {e}")
            return False

    @staticmethod
    def load_model(filename):
        """
        Tải mô hình từ file với tên được chỉ định.

        :param filename: Tên file chứa mô hình
        :return: Mô hình đã tải, hoặc None nếu có lỗi
        """
        try:
            with open(filename, 'rb') as file:
                model = pickle.load(file)
            return model
        except Exception as e:
            print(f"An exception occurred: {e}")
            return None
