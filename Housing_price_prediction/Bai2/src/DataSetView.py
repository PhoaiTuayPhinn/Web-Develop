from tkinter import *
from tkinter import ttk
import pandas as pd

class DataSetView:
    def __init__(self):
        pass

    def create_ui(self):
        self.root = Tk()
        self.root.title("Dataset viewer - House Pricing Prediction")
        self.root.geometry("800x600")

        # Tạo panel chứa các thành phần
        main_panel = PanedWindow(self.root)
        main_panel["bg"] = "yellow"
        main_panel.pack(fill=BOTH, expand=True)

        # Define columns
        columns = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
                   'Avg. Area Number of Bedrooms', 'Area Population', 'Price']

        # Create Treeview widget for showing data
        self.tree = ttk.Treeview(main_panel, columns=columns, show="headings")

        # Define headings
        self.tree.heading('Avg. Area Income', text="Avg. Area Income")
        self.tree.heading('Avg. Area House Age', text="Avg. Area House Age")
        self.tree.heading('Avg. Area Number of Rooms', text="Avg. Area Number of Rooms")
        self.tree.heading('Avg. Area Number of Bedrooms', text="Avg. Area Number of Bedrooms")
        self.tree.heading('Area Population', text="Area Population")
        self.tree.heading('Price', text="Price")

        # Scrollbar for treeview
        scrollbar = ttk.Scrollbar(main_panel, orient=VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Make the grid resizeable
        self.tree.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill=Y, expand=True)

        self.show_ui()

    def show_ui(self):
        self.root.mainloop()

    def show_data_listview(self, fileName):
        df = pd.read_csv(fileName)
        for i in range(len(df)):
            values = [df.iloc[i][0], df.iloc[i][1], df.iloc[i][2], df.iloc[i][3], df.iloc[i][4], df.iloc[i][5]]
            self.tree.insert('', 'end', values=values)
