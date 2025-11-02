from tkinter import *

OPTIONS = [
    "model_1",
    "model_2",
    "model_3"
]

root = Tk()

variable = StringVar(root)
variable.set(OPTIONS[0])  # default value

w = OptionMenu(root, variable, *OPTIONS)
w.pack()

def ok():
    print("value is:" + variable.get())

button = Button(root, text="OK", command=ok)
button.pack()

mainloop()
