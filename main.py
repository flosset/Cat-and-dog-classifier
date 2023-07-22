from tkinter import *

def main():
    root = Tk()
    root.title('Identifier')
    root.geometry('800x600')
    
    # making two frames
    frame1 = LabelFrame(root, padx=30, pady=30)
    frame1.grid(row=0, column=0, padx=10, pady=5)
    
    frame2 = LabelFrame(root, padx=30, pady=30)
    frame2.grid(row=0, column=1, padx=10, pady=5)
    
    root.mainloop()
    
    
if __name__ == '__main__':
    main()