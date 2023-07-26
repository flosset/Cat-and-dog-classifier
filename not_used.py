import numpy as np
import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import EfficientNetB0

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
    
    
def model_trainer_and_saver():
    image_size =  (224, 224)
    image_shape = image_size + (3,) # Since each picture has three channels
    batch_size = 32
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
    #train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    model = get_classifier_model(image_shape)
    
    
    
    base_learning_rate = 0.001
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=base_learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    print(model.summary())
    model.fit(train_dataset, validation_data=validation_dataset, epochs=7)
    model.save('cat_and_dog_classifier')
    
def data_augmentor():
    model = tf.keras.Sequential()
    model.add(RandomFlip('horizontal'))
    model.add(RandomRotation(0.2))
    return model

def get_classifier_model(image_shape, augmentor=data_augmentor()):
    model = EfficientNetB0(input_shape=image_shape,
                           include_top=False,
                           weights='imagenet')
    
    # freeze the model (make it untrainable)
    model.trainable = False
    inputs = tf.keras.Input(shape=image_shape)
    x = augmentor(inputs)
    preprocess = tf.keras.applications.efficientnet.preprocess_input
    # Preprocess after augmentor
    x = preprocess(x)
    x = model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) 
    x = tf.keras.layers.Dropout(0.2)(x)
    
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs, outputs)
    
    # retrain last layers of previous model (EfficientNetB0)
    previous_model = model.layers[2]
    previous_model.trainable = True
    #print(len(previous_model.layers))
    fine_tune_at = 200
    for layer in previous_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.layers[2] = previous_model
    
    return model