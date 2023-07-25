from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
import os
from not_used import load_datasets
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.applications import EfficientNetB0
import numpy as np
from tensorflow.keras.preprocessing import image

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
    
    
    img = Image.open('cat.4001.jpg')
    # Resize the image to match your model's input size
    img = img.resize((224, 224))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    model = tf.keras.models.load_model('cat_and_dog_classifier')
    
    predictions = model.predict(img_array)
    class_names = ['cat', 'dog']
    threshold = 0.5
    binary_predictions = (predictions >= threshold).astype(int)
    print(class_names[binary_predictions[0][0]])
    
    
    

    
    
    
if __name__ == '__main__':
    main()