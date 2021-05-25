import PIL.Image
from fastai import learner, torch_core
import io
from tkinter import *
from fastai.vision.core import *
import pathlib
import ghostscript

# workarond for the error: "cannot instantiate 'PosixPath' on your system"
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


# load_learner requires to re-define all custom functions
def label_func(fname): return fname.parent.name


# load learner using fastai module
# pathlib.Path().absolute() #for current working directory
learn_inf = learner.load_learner(pathlib.Path(
    __file__).parent.absolute()/'export.pkl')

root = Tk()
root.configure(background='gray90')
root.title("Digit Recognition")

root.geometry('640x480')  # windows size
root.resizable(0, 0)  # makes it not resizable

desc = Label(root, text=f'A simple Handwriting digit recognition app. \n Make sure to write the number large and in the center', width=50, height=3,
             borderwidth=1, relief="raised", bg="snow", font=('times', 15)).place(relx=0.5, y=40, anchor=CENTER)


def destroy_widget(widget):
    widget.destroy()


def pred_digit():
    global no, no1
    ps = canvas.postscript(colormode='color')
    # use PIL to convert to PNG
    im1 = PIL.Image.open(io.BytesIO(ps.encode('utf-8')))
    img = im1.resize((28, 28))
    img = PIL.ImageOps.invert(img)
    img = torch_core.TensorImage(image2tensor(img))
    img = PILImage.create(img)
    # predicting the class
    pred, pred_idx, probs = learn_inf.predict(img)
    if probs[pred_idx].item() < 0.75:
        backgr = "salmon"
    else:
        backgr = "cyan"
    no = Label(root, text='Prediction: '+str(pred), width=20, height=1, bg=backgr,
               font=('times', 16, ' bold '))
    no.place(x=350, y=80)

    no1 = Label(root, text=f'Probability: {probs[pred_idx]:.04f}', width=20, height=1, bg=backgr,
                font=('times', 16, ' bold '))
    no1.place(x=350, y=110)
    all = Label(root, text=f'All Predictions: Digit (Prob)', width=27, height=1,
                bg="light cyan", font=('times', 13)).place(x=350, y=160)
    rest = list(range(10))
    for idx, val in enumerate(probs):
        rest[idx] = Label(root, text=f'Prediction: {idx} ({val:.03f})', width=27, height=1,
                          bg="light cyan", font=('times', 12)).place(x=350, y=190 + 20*idx)


def drawing(event):
    #    canvas.configure(background="black")
    x = event.x
    y = event.y
    r = 10
    canvas.create_oval(x-r, y-r, x + r, y + r, fill='black')
    panel5.configure(state=NORMAL)


canvas = Canvas(root, width=280, height=280, highlightthickness=2,
                highlightbackground="midnightblue", cursor="pencil")
canvas.grid(row=0, column=0, pady=2, sticky=W,)
canvas.place(x=43, y=80)
canvas.bind("<B1-Motion>", drawing)

panel5 = Button(root, text='Predict Digit', state=DISABLED, command=pred_digit,
                width=15, borderwidth=0, bg='midnightblue', fg='white', font=('times', 18, 'bold'))
panel5.place(x=80, y=375)


def clear_digit():
    panel5.configure(state=DISABLED)
    canvas.delete("all")


panel6 = Button(root, text='Clear Canvas', width=15, borderwidth=0,
                command=clear_digit, bg='red', fg='white', font=('times', 18, 'bold'))
panel6.place(x=80, y=425)


root.mainloop()  # root window and mainLoop are requirements.
