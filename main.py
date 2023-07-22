from tkinter import *
from tkinter import filedialog

def main():
    root = Tk()
    root.title('Identifier')
    root.geometry('800x600')
    
    # making two frames
    frame1 = LabelFrame(root, padx=130, pady=130)
    frame1.grid(row=0, column=0, padx=10, pady=20)
    
    frame2 = LabelFrame(root, padx=130, pady=130)
    frame2.grid(row=0, column=1, padx=10, pady=20)
    
    # put image selection inside lhs frame
    selection_buttom = Button(frame1, text="Select File")
    selection_buttom.pack()
    
    label = Label(frame2, text="Result....")
    label.pack()
    
    root.mainloop()
    
    
if __name__ == '__main__':
    main()