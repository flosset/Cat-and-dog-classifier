from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.models import Model, load_model
import os
from not_used import load_datasets
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.applications import EfficientNetB0
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
    image_size =  (224, 224)
    image_shape = image_size + (3,) # Since each picture has three channels
    batch_size = 70
    train_dataset = image_dataset_from_directory('dataset/training_set/training_set',
                                                                        shuffle=True,
                                                                        batch_size=batch_size,
                                                                        image_size=image_size,
                                                                        seed=40,
                                                                        validation_split=0.2,
                                                                        subset='training')
    validation_dataset = image_dataset_from_directory('dataset/training_set/training_set',
                                                                        shuffle=True,
                                                                        batch_size=batch_size,
                                                                        image_size=image_size,
                                                                        seed=40,
                                                                        validation_split=0.2,
                                                                        subset='validation')
    
    # Prefetch data
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    
    
    
    
def data_augmentor():
    model = tf.keras.Sequential()
    model.add(RandomFlip('horizontal'))
    model.add(RandomRotation(0.2))
    return model

def get_classifier_model(image_shape, augmentor=data_augmentor()):
    model = EfficientNetB0(input_shape=image_shape,
                           include_top=False,
                           weights='imagenet')
    
    
if __name__ == '__main__':
    main()