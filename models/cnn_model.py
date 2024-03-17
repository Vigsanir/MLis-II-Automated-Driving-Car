#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow import keras
from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Input


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


def create_cnn_model_v2(image_shape, pool_size=(2, 2)):
    """
    Creates two CNN models with the specified input shape.
    """
    model_speed = Sequential([
        Input(shape=image_shape + (3,)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', name='output_speed')  # Output layer for 'speed'
    ])

    model_angle = Sequential([
        Input(shape=image_shape + (3,)),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear', name='output_angle')  # Output layer for 'angle'
    ])

    return model_speed, model_angle

def create_cnn_model_v3(image_shape, pool_size=(2, 2)):
    """
    Creates two CNN models with the specified input shape.
    """
    model_speed = Sequential([
        Input(shape=image_shape + (3,), name='input_speed'),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid', name='output_speed')  # Output layer for 'speed'
    ], name='model_speed')

    model_angle = Sequential([
        Input(shape=image_shape + (3,), name='input_angle'),
        Conv2D(32, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(64, (5, 5), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=pool_size),
        Conv2D(256, (3, 3), activation='relu'),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='linear', name='output_angle')  # Output layer for 'angle'
    ], name='model_angle')

    print("Model Speed Summary:")
    model_speed.summary()
    
    print("\nModel Angle Summary:")
    model_angle.summary()
    
    return model_speed, model_angle

############################################################################################################

#image_shape = (int(240/2), int(320/2)) # Half the real size of the image.
#pool_size=(2, 2)
#model_speed, model_angle = create_cnn_model_v3(image_shape, pool_size)

############################################################################################################