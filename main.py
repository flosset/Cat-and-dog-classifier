from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
import os
import numpy as np

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
    X_train, Y_train, X_test, Y_test, classes = load_datasets('dataset/training_set/training_set')  # sending dataset folder as argument (will just  going to use training dataset folder)
    print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    

def load_datasets(directory):
    
    # Get a list of all the classes
    classes = os.listdir(directory)
    
    # Initialize lists to store data and labels
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    
    # Open each class and access all the files
    for class_index, class_name in enumerate(classes):      # We will use number for Y data as well
        class_path = os.path.join(directory, class_name)
        
        # get names of all the images in current class direction
        images = os.listdir(class_path)
        
        # Shuffle the images for better randomness
        np.random.shuffle(images)  # probably doesnt matter
        num_train = int(0.8 * len(images))   # Number of images that we will use to train model
        
        # Open each image and add it to our lists
        for i, image in enumerate(images): 
            image_path = os.path.join(class_path, image)
            # Create 3 dimentional data of image before we add it to our arrays
            img  = Image.open(image_path)
            img = img.resize((64, 64))  # Make image small to make things easier
            img = np.array(img) / 255.0
            
            if i < num_train: # Add first 80% of images to training dataset
                X_train.append(img)
                Y_train.append(class_index)
            else:
                X_test.append(img)
                Y_test.append(class_index)
    
    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    
    # Transpose these one dimentional array to make them vertical instead of horizontal
    Y_train = Y_train.T
    Y_test = Y_test.T
    
    return X_train, Y_train, X_test, Y_test, classes
    
if __name__ == '__main__':
    main()