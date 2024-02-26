# test_tf_import.py
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D

from tensorflow import keras
from keras.models import Model
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import MaxPooling2D

def create_cnn_model(input_shape):
    """
    Creates a CNN model with the specified input shape.
    """
    model = Sequential([
        Conv2D(24, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(36, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(48, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        
        Flatten(),
        
        Dense(100, activation='relu'),
        Dropout(0.5),  # Dropout layer to reduce overfitting
        Dense(50, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        Dense(2)  # Output layer with 2 neurons for regression tasks
    ])
    
    # Compile the model with mean squared error loss and an optimizer of choice
    model.compile(optimizer='adam', loss='mse')
    
    return model
