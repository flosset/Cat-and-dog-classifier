from tkinter import *
from tkinter import filedialog
import tensorflow as tf
from PIL import Image, ImageTk
import numpy as np
from tensorflow.keras.preprocessing import image

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title('Identifier')
        self.root.geometry('800x600')
        
        self.frame1 = LabelFrame(root, padx=130, pady=130)
        self.frame1.grid(row=0, column=0, padx=10, pady=20, columnspan=2)
        
        # Having this variable to garbage collection
        self.selected_image_data = None
        self.image_label = None
        self.image_location = None
        
        self.frame2 = LabelFrame(root, padx=130, pady=130)
        self.frame2.grid(row=0, column=2, padx=10, pady=20, columnspan=2)
        
        self.selection_button = Button(self.frame1, text="Select File", command=self.file_dialog_prompt)
        self.selection_button.pack()
        
        self.result_label = Label(self.frame2, text="Result....")
        self.result_label.pack()
        
        # Stores the name of the animal
        self.result = None
        
        # create a get_result button
        self.get_result_button = Button(text='Get result', command=self.get_result)
        self.get_result_button.grid(row=1, column=1, columnspan=2)
        
    def file_dialog_prompt(self):
        filename = filedialog.askopenfilename(initialdir="/", title="Select a file", filetypes=(("image files", '*.jpg'), ("if above doesnt show the file you want to select", '*.*')))
        self.image_location = filename
        
        if filename:
            self.open_image()
    

    def open_image(self):
        img = Image.open(self.image_location)
        img = img.resize((300, 300))
        image = ImageTk.PhotoImage(img)
        
        if self.image_label:
            self.image_label.destroy()
        self.image_label = Label(self.frame1, image=image)
        self.image_label.pack()#side='bottom', anchor='n')
        # Safe the image in instance attribute
        self.image = image  # Save a reference to the image to avoid garbage collection
        # Change frame internal padding
        self.frame1.config(padx=5, pady=5)
    
    def get_result(self):
        # If location variable is not empty
        if self.image_location:
            animal = classify(self.image_location)
            self.result_label.config(text="It's a " + animal)   

def main():
    root = Tk()
    app = MainApp(root)
    root.mainloop()
    
def classify(file_location):
    '''
    This model will use skip connections since I am just learning
    '''
    img = Image.open(file_location)
    # Resize the image to match your model's input size
    img = img.resize((224, 224))
    # Convert the image to a NumPy array
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    model = tf.keras.models.load_model('cat_and_dog_classifier')
    
    predictions = model.predict(img_array)
    class_names = ['cat😽🐈', 'dog🐕']
    threshold = 0.5 # Value less than 0.5 is cat else dog
    binary_predictions = (predictions >= threshold).astype(int) # returns 3d array
    return class_names[binary_predictions[0][0]]
    
    
if __name__ == '__main__':
    main()