from tkinter import *
from PIL import Image, ImageTk
import FruitShaninja

class GUI:
    def __init__(self, master):
        self.master = master
        master.title("FRUIT SHANINJA")
        # master.geometry("1280x800")

        image = Image.open("WhatsApp Image 2018-11-10 at 23.51.52.jpeg")
        image = image.resize((400,400))

        photo = ImageTk.PhotoImage(image)
        self.background = Label(master, image=photo)
        self.background.image = photo
        # self.background.place(x=0, y=0, relwidth=1, relheight=1)
        self.background.pack()

        self.run = Button(master, text="RUN ZEN MODE", command=FruitShaninja.run, bg = "green")
        self.run.pack()

        self.multiplayer = Button(master, text="MULTIPLAYER MODE", command=FruitShaninja.run, bg="blue")
        self.multiplayer.pack()

        self.close_button = Button(master, text="Close", command=master.quit, bg = "red")
        self.close_button.pack()


root = Tk()
my_gui = GUI(root)
root.mainloop()