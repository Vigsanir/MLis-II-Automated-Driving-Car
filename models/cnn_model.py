#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D


def create_cnn_model(input_shape, pool_size=(2, 2)):
    """
    Creates a CNN model with the specified input shape.
    """
    model = Sequential([

        Conv2D(24, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size),
        
        Conv2D(36, (5, 5), activation='relu'),
        MaxPooling2D(),
        
        Conv2D(48, (5, 5), activation='relu'),
        MaxPooling2D(pool_size),
        
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size),
        
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


def create_cnn_model_v2(input_shape, pool_size=(2, 2)):
    """
    Creates a CNN model with the specified input shape.
    """
    model = Sequential([

        Conv2D(24, (5, 5), activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size),
        
        Conv2D(36, (5, 5), activation='relu'),
        MaxPooling2D(),
        
        Conv2D(48, (5, 5), activation='relu'),
        MaxPooling2D(pool_size),
        
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size),
        
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
