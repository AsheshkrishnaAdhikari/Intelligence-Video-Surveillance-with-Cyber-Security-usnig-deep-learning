from tkinter import *
from PIL import ImageTk, Image
import tkinter as tk
import os

app = Tk()
app.title("Welcome")
img =Image.open('bg image.jpg')
bg = ImageTk.PhotoImage(img)

app.geometry("1650x1050")

# Add image
label = Label(app, image=bg)
label.place(x = 0,y = 0)

# Add text
label2 = Label(app, text = "UNUSUAL EVENT DETECTION BY USING DEEP LEARNING",bg="#3db7bf",
               font=("Times New Roman", 24))

label2.pack(pady = 50)


def button1():
    os.system('python Unusual_Event_Weapons.py')
    
def button2():
    os.system('python motion_detector.py')



def Submit():
    pass
    
b1=tk.Button(app,text="Unusual_Event_Weapons",command=button1,bg="#3db7bf",activebackground="#c21d54",fg="black",font=('Arial',16),height=2,width=20)
b1.place(x=500,y=200)
##l1=tk.Label(app,text="",bg="blue",fg="black",font=('Arial',16),height=2,width=20)
##l1.place(row=1,column=6, columnspan=6)
b1=tk.Button(app,text="motion_detector",command=button2,bg="#3db7bf",activebackground="#c21d54",fg="black",font=('Arial',16),height=2,width=20)
b1.place(x=800,y=200)



app.mainloop()

