from tkinter import *
from tkinter import messagebox, ttk
from tkinter.font import Font
from tkinter import filedialog as fd
from src.DataSetView import DataSetView
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from src.FileUtil import FileUtil

class UIPrediction:
    fileName = ""

    def __init__(self):
        pass

    def create_ui(self):
        self.root = Tk()
        self.root.title("House Pricing Prediction - Faculty of Information Systems")
        self.root.geometry("1500x850")

        # Main Panel
        main_panel = PanedWindow(self.root)
        main_panel["bg"] = "yellow"
        main_panel.pack(fill=BOTH, expand=True)

        # Top Panel
        top_panel = PanedWindow(main_panel, height=80)
        top_panel["bg"] = "blue"
        main_panel.add(top_panel)
        top_panel.pack(fill=X, side=TOP, expand=False)

        font = Font(family="tahoma", size=18)
        title_label = Label(top_panel, text='House Pricing Prediction', font=font)
        title_label["bg"] = "yellow"
        top_panel.add(title_label)

        # Center Panel
        center_panel = PanedWindow(main_panel)
        main_panel.add(center_panel)
        center_panel["bg"] = "pink"
        center_panel.pack(fill=BOTH, expand=True)

        # Choose Dataset Panel
        choose_dataset_panel = PanedWindow(center_panel, height=30)
        center_panel.add(choose_dataset_panel)
        choose_dataset_panel.pack(fill=X)

        dataset_label = Label(choose_dataset_panel, text="Select Dataset:")
        self.selectedFileName = StringVar()
        dataset_label.grid(row=0, column=0)

        self.choose_dataset_panel = PanedWindow(choose_dataset_panel)
        self.choose_dataset_button = Button(self.choose_dataset_panel, text="1.Pick Dataset", width=10, command=self.do_pick_data)
        self.view_dataset_button = Button(self.choose_dataset_panel, text="2.View Dataset", width=20, command=self.do_view_dataset)
        self.choose_dataset_button.grid(row=1, column=0)
        self.view_dataset_button.grid(row=1, column=1)
        self.choose_dataset_panel.grid(row=1, column=0)

        # Training Rate
        training_rate_panel = PanedWindow(center_panel, height=30)
        center_panel.add(training_rate_panel)

        training_rate_label = Label(training_rate_panel, text="Training Rate:")
        training_rate_label.grid(row=0, column=0)
        self.training_rate = IntVar()
        self.training_rate.set(80)
        self.training_rate_entry = Entry(training_rate_panel, textvariable=self.training_rate, width=20)
        self.training_rate_entry.grid(row=0, column=1)

        training_rate_panel.pack(fill=X)
        self.train_model_button = Button(training_rate_panel, text="3.Train Model", width=20, command=self.do_train)
        self.train_model_button.grid(row=2, column=0)

        self.status = StringVar()
        self.status.set("Ready")
        self.status_label = Label(training_rate_panel, textvariable=self.status)
        self.status_label.grid(row=3, column=1)

        # RMSE
        rmse_label = Label(training_rate_panel, text="Root Mean Square Error (RMSE):")
        rmse_label.grid(row=4, column=0)
        self.rmse_value = DoubleVar()
        rmse_entry = Entry(training_rate_panel, textvariable=self.rmse_value, width=20)
        rmse_entry.grid(row=4, column=1)

        # Save Model Button
        savemodel_button = Button(training_rate_panel, text="5. Save Model", width=20, command=self.do_save_model)
        savemodel_button.grid(row=5, column=0)

        # Load Model Button
        loadmodel_button = Button(training_rate_panel, text="6. Load Model", width=20, command=self.do_load_model)
        loadmodel_button.grid(row=5, column=1)

        # Prediction Input
        input_prediction_panel = PanedWindow(center_panel)
        input_prediction_panel.pack(fill=BOTH, side=TOP, expand=True)

        area_income_label = Label(input_prediction_panel, text="Avg. Area Income:")
        area_income_label.grid(row=0, column=0)
        self.area_income_value = DoubleVar()
        area_income_entry = Entry(input_prediction_panel, textvariable=self.area_income_value, width=40)
        area_income_entry.grid(row=0, column=1)

        area_house_age_label = Label(input_prediction_panel, text="Avg. Area House Age:")
        area_house_age_label.grid(row=1, column=0)
        self.area_house_age_value = DoubleVar()
        area_house_age_entry = Entry(input_prediction_panel, textvariable=self.area_house_age_value, width=40)
        area_house_age_entry.grid(row=1, column=1)

        area_number_of_rooms_label = Label(input_prediction_panel, text="Avg. Area Number of Rooms:")
        area_number_of_rooms_label.grid(row=2, column=0)
        self.area_number_of_rooms_value = DoubleVar()
        area_number_of_rooms_entry = Entry(input_prediction_panel, textvariable=self.area_number_of_rooms_value, width=40)
        area_number_of_rooms_entry.grid(row=2, column=1)

        area_number_of_bedrooms_label = Label(input_prediction_panel, text="Avg. Area Number of Bedrooms:")
        area_number_of_bedrooms_label.grid(row=3, column=0)
        self.area_number_of_bedrooms_value = DoubleVar()
        area_number_of_bedrooms_entry = Entry(input_prediction_panel, textvariable=self.area_number_of_bedrooms_value, width=40)
        area_number_of_bedrooms_entry.grid(row=3, column=1)

        area_population_label = Label(input_prediction_panel, text="Area Population:")
        area_population_label.grid(row=4, column=0)
        self.area_population_value = DoubleVar()
        area_population_entry = Entry(input_prediction_panel, textvariable=self.area_population_value, width=40)
        area_population_entry.grid(row=4, column=1)

        prediction_button = Button(input_prediction_panel, text="7. Prediction House Pricing", command=self.do_prediction)
        prediction_button.grid(row=5, column=1)

        prediction_price_label = Label(input_prediction_panel, text="Prediction Price:")
        prediction_price_label.grid(row=6, column=0)
        self.prediction_price_value = DoubleVar()
        prediction_price_entry = Entry(input_prediction_panel, textvariable=self.prediction_price_value, width=40)
        prediction_price_entry.grid(row=6, column=1)

        # Designed By Panel
        designedby_panel = PanedWindow(main_panel, height=20)
        designedby_panel["bg"] = "cyan"
        designedby_panel.pack(fill=BOTH, side=BOTTOM)
        designedby_label = Label(designedby_panel, text="Designed by: Huỳnh Quốc Việt")
        designedby_label["bg"] = "cyan"
        designedby_label.pack(side=LEFT)

    def show_ui(self):
        self.root.mainloop()

    def do_pick_data(self):
        filetypes = (("Dataset CSV", "*.csv"), ("All Files", "*.*"))
        s = fd.askopenfilename(title="Choose dataset", initialdir="/", filetypes=filetypes)
        self.selectedFileName.set(s)

    def do_view_dataset(self):
        viewer = DataSetView()
        viewer.create_ui()
        viewer.show_data_listview(self.selectedFileName.get())
        viewer.show_ui()

    def do_train(self):
        ratio = self.training_rate.get()/100
        self.df = pd.read_csv(self.selectedFileName.get())

        self.X = self.df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                          'Avg. Area Number of Bedrooms', 'Area Population']]
        self.y = self.df['Price']

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=1-ratio, random_state=101)

        from sklearn.linear_model import LinearRegression
        self.lm = LinearRegression()
        self.lm.fit(self.X_train, self.y_train)
        self.status.set("Trained is finished")
        messagebox.showinfo("Info", "Trained is finished")

    def do_evaluation(self):
        print(self.lm.intercept_)
        insert_text = self.lm.intercept_

        self.coeff_df = pd.DataFrame(self.lm.coef_, self.X.columns, columns=['Coefficient'])
        print(self.coeff_df)
        self.coefficient_detail_text.insert(END, self.coeff_df)

        predictions = self.lm.predict(self.X_test)
        print(predictions)
        print("self.X_test")
        print(self.X_test)
        print("self.y_test:")
        print(self.y_test)

        y_test_array = np.asarray(self.y_test)
        for i in range(0, len(self.X_test)):
            values = [self.X_test.iloc[i][0], self.X_test.iloc[i][1], self.X_test.iloc[i][2],
                      self.X_test.iloc[i][3], self.X_test.iloc[i][4], y_test_array[i], predictions[i]]
            print(values)
            self.tree.insert('', END, values=values)

    def do_save_model(self):
        FileUtil.save_model(self.lm, "housingmodel.zip")
        messagebox.showinfo("Infor", "exported model to disk successful!")

    def do_load_model(self):
        self.lm = FileUtil.load_model("housingmodel.zip")
        messagebox.showinfo("Infor", "loading model from disk successful!")

    def do_prediction(self):
        result = self.lm.predict([[
            self.area_income_value.get(),
            self.area_house_age_value.get(),
            self.area_number_of_rooms_value.get(),
            self.area_number_of_bedrooms_value.get(),
            self.area_population_value.get()
        ]])
        self.prediction_price_value.set(result[0])
