from tkinter import Tk, Label, Button, PhotoImage
from PIL import Image, ImageTk
import FruitShaninja

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("FRUIT SHANINJA")
        # master.geometry("1280x800")

        image = Image.open("Wiki-background.jpg")
        image = image.resize((640, 400))

        photo = ImageTk.PhotoImage(image)
        self.background = Label(master, image=photo)
        self.background.image = photo
        # self.background.place(x=0, y=0, relwidth=1, relheight=1)
        self.background.pack()

        self.greet_button = Button(master, text="RUN ZEN MODE", command=FruitShaninja.run)
        self.greet_button.pack()

        self.close_button = Button(master, text="Close", command=master.quit)
        self.close_button.pack()



root = Tk()
my_gui = GUI(root)
root.mainloop()