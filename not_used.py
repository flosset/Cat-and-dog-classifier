import numpy as np
import os
from PIL import Image

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
        
        if class_path == 'dataset/training_set/training_set/.DS_Store':
            continue
        
        # get names of all the images in current class direction
        images = os.listdir(class_path)
        
        # Shuffle the images for better randomness
        np.random.shuffle(images)  # probably doesnt matter
        num_train = int(0.8 * len(images))   # Number of images that we will use to train model
        
        # Open each image and add it to our lists
        for i, image in enumerate(images): 
            image_path = os.path.join(class_path, image)
            
            # Will remove later
            if not image_path.endswith('.jpg'):
                continue
            
            # Create 3 dimentional data of image before we add it to our arrays
            img  = Image.open(image_path)
            img = img.resize((64, 64))  # Make image small to make things easier
            img = np.array(img) / 255.0 # So that pixel value are between 0 and 1
            
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
    
    # Change shape from (x, ) to (x, 1)
    Y_train = Y_train.reshape(-1, 1) # first argument automatically infer size when second is 1  (Y_train shape was e.g. (6000,) before)
    Y_test = Y_test.reshape(-1, 1)
    
    return X_train, Y_train, X_test, Y_test, classes

def junk_code():
    #X_train, Y_train, X_test, Y_test, classes = load_datasets('dataset/training_set/training_set')  # sending dataset folder location as argument (will just  going to use training dataset folder)
    '''print ("number of training examples = " + str(X_train.shape[0]))
    print ("number of test examples = " + str(X_test.shape[0]))
    print ("X_train shape: " + str(X_train.shape))
    print ("Y_train shape: " + str(Y_train.shape))
    print ("X_test shape: " + str(X_test.shape))
    print ("Y_test shape: " + str(Y_test.shape))
    print(classes)'''