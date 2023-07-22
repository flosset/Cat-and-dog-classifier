from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model

def main():
    # create a model to identify cat and dog
    model = create_model()
    
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
    
def create_model():
    '''
    This model will use skip connections since I am just learning
    '''
    X_train, Y_train, X_test, y_test, classes = load_datasets()
    test = Model()
    ...
    

    
    
if __name__ == '__main__':
    main()