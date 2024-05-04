from tensorflow import keras
import numpy as np
import imutils
import cv2
from PIL import Image
import os
import os
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


class Model:

    angle_model = '05-03_18-34_CNN_model_angle_epochs3500.h5'
    speed_model = '05-03_11-38_CNN_model_speed_epochs1350.h5'

    def __init__(self):
        self.angle_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.angle_model))
        self.speed_model = keras.models.load_model(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.speed_model))
        self.angle_model.summary()
        self.speed_model.summary()

    def preprocess(self, image):
        # Convert the PIL Image to a NumPy array
        img_array = np.array(image)
    
        # Ensure the shape is (height, width, channels)
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension for grayscale images
    
        image_size = (120,120)  # [240, 320] real size of the image.
    
        # Resize the image to the specified size
        img_array = cv2.resize(img_array, image_size)
    
        # Normalize the image array
        img_array = img_array.astype(np.float32) / 255.0  # Scale pixel values to [0, 1]
    
        return img_array



    def predict_angle(self, image):
        
        angle = self.angle_model.predict(np.array([image]))[0]
        
        return angle

    def predict_speed(self, image):
        
        speed = self.speed_model.predict(np.array([image]))[0]
        speed = round(speed[0])
                        
        return speed



    def predict(self, image):
        image = self.preprocess(image)

        angle = self.predict_angle(image)
        speed = self.predict_speed(image)
       
        # Training data was normalised so convert back to car units
        angle = 80 * np.clip(angle, 0, 1) + 50
        speed = 35 * np.clip(speed, 0, 1)
        return angle, speed
