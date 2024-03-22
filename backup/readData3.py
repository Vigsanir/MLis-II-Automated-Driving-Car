from utils.debug_utils import debug_log 
import os
from pathlib import Path
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from PIL import Image, UnidentifiedImageError,PngImagePlugin
import logging
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.keras.optimizers import Adadelta  as LegacyAdadelta
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from models.cnn_model import create_cnn_model, create_cnn_model_v2


def get_script_directory():
    try:
        # Check if running in a Jupyter notebook
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Running in a Jupyter notebook
            return os.getcwd()
        else:
            # Running in a standalone Python script
            return Path(__file__).parent
    except NameError:
        # Running in a standalone Python script
        return Path(__file__).parent



def filter_data(path_to_data):
    # Define relative paths
    training_data_relative_path = f"{path_to_data}/training_data/training_data"
    training_labels_relative_path = f"{path_to_data}/training_norm.csv"
    training_labels = pd.read_csv(training_labels_relative_path)
    filenames = os.listdir(training_data_relative_path)
    indices_to_remove = []
    if any('.png' in filename for filename in filenames):

        for index, current_filename in enumerate(filenames):
                   image_path = f"{training_data_relative_path}/{current_filename}"
                   try:
                        Image.open(image_path)
                   except UnidentifiedImageError:
                        os.remove(image_path)
                        indices_to_remove.append(index)
                        continue 
    # Remove rows from training_norm based on the indices_to_remove list
    training_labels = training_labels.drop(training_labels.index[indices_to_remove])
    return training_labels


def construct_training_image_paths(path_to_data):
    training_labels = filter_data(path_to_data)
    training_labels["speed"] = training_labels["speed"].astype(int)
    training_labels["image_id"] = training_labels["image_id"].apply(lambda x: f"{path_to_data}/training_data/training_data/{x}.png")
    training_labels.rename(columns={"image_id": "image_path"}, inplace=True)
    
    train_image_paths_speed_0 = training_labels[training_labels['speed'] == 0].reset_index(drop=True)
    train_image_paths_speed_1 = training_labels[training_labels['speed'] == 1].reset_index(drop=True)
    return train_image_paths_speed_0, train_image_paths_speed_1




def construct_test_image_paths(directory):
    test_data_relative_path = "machine-learning-in-science-ii-2024/test_data/test_data"
    # bug_log(f"Test Data Directory: {construct_test_image_paths}")
    filenames = os.listdir(test_data_relative_path)
    test_image_paths = pd.DataFrame({'image_path': []})
    if any('.png' in filename for filename in filenames):

        for index, current_filename in enumerate(filenames):
           image_path = f"{test_data_relative_path}/{current_filename}"
           try:
                Image.open(image_path)
                test_image_paths.loc[len(test_image_paths)] = str(f"{image_path}.png") # Add a new row to a DataFrame
           except UnidentifiedImageError:
                os.remove(image_path)
                continue 
    return test_image_paths

def build_image_paths(directory):
    # Get the data directory
    path_to_data = f"{directory}/machine-learning-in-science-ii-2024"

    train_image_paths_speed_0, train_image_paths_speed_1 = construct_training_image_paths(path_to_data)
    test_image_paths = construct_test_image_paths(path_to_data)
    return train_image_paths_speed_0, train_image_paths_speed_1, test_image_paths




def build_training_validation_and_evaluation_sets(train_image_paths_0, train_image_paths_1, image_shape, batch_size, eval_split, train_val_split):
    # Combine datasets 0 and 1
    combined_dataset = pd.concat([train_image_paths_0, train_image_paths_1], ignore_index=True)

    # Split into training and validation sets
    train_set, val_set = train_test_split(combined_dataset, test_size=train_val_split[1], random_state=42)

    # Split validation set into evaluation sets for speed and angle
    eval_set_speed, eval_set_angle = train_test_split(val_set, test_size=eval_split, random_state=42)

    # Additional processing as needed (e.g., loading images, data augmentation)

    # Print summary
    print(f"\nFound {len(combined_dataset)} images.")
    print(f"Using {len(train_set)} ({round(len(train_set) / len(combined_dataset) * 100, 1)}%) for training.")
    print(f"Using {len(val_set)} ({round(len(val_set) / len(combined_dataset) * 100, 1)}%) for validation.")
    print(f"Using {len(eval_set_speed)} ({round(len(eval_set_speed) / len(combined_dataset) * 100, 1)}%) for evaluation of speed.")
    print(f"Using {len(eval_set_angle)} ({round(len(eval_set_angle) / len(combined_dataset) * 100, 1)}%) for evaluation of angle.")

    # Additional return statements as needed
    return train_set, val_set, eval_set_speed, eval_set_angle

class CustomDataGenerator(Sequence):
    def __init__(self, data_frame, batch_size, image_shape, augmentations=None, shuffle=True):
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_frame))

    def __len__(self):
        return int(np.ceil(len(self.data_frame) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch = self.data_frame.iloc[start:end]
        images = [self.load_and_augment_image(row['image_path']) for _, row in batch.iterrows()]
        images = np.array(images)
        speeds = batch['speed'].values

        return images, {'output_speed': speeds}


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_and_augment_image(self, image_path):
         img = load_img(image_path, target_size=self.image_shape)
         img = img_to_array(img)
         if self.augmentations:
             img = self.augmentations.random_transform(img)
         img = img / 255.0  # Normalize to [0, 1]
         return img

class CustomDataGenerator_2(Sequence):
    def __init__(self, data_frame, batch_size, image_shape, augmentations=None, shuffle=True):
        self.data_frame = data_frame
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.data_frame))

    def __len__(self):
        return int(np.ceil(len(self.data_frame) / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        batch = self.data_frame.iloc[start:end]
        images = [self.load_and_augment_image(row['image_path']) for _, row in batch.iterrows()]
        images = np.array(images)
        speeds = batch['angle'].values

        return images, {'output_angle': speeds}


    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def load_and_augment_image(self, image_path):
         img = load_img(image_path, target_size=self.image_shape)
         img = img_to_array(img)
         if self.augmentations:
             img = self.augmentations.random_transform(img)
         img = img / 255.0  # Normalize to [0, 1]
         return img

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
# Call the function
# train_set, val_set, eval_set_speed, eval_set_angle = build_training_validation_and_evaluation_sets(train_image_paths_0, train_image_paths_1, image_shape, batch_size, eval_split, train_val_split, speed_weights)


# Set the logging level for the PIL module to suppress debug messages
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.ERROR)

    # Example usage
directory = get_script_directory()
train_image_paths_speed_0, train_image_paths_speed_1, test_image_paths = build_image_paths(directory)
print(train_image_paths_speed_0)
print(train_image_paths_speed_1)
print(test_image_paths)
# Print paths in debug mode
# debug_log("This is a debug message.", data=directory)


# --- DATA HYPERPARAMETERS ---
batch_size = 64
image_shape = (320, 240)
eval_split = 0.1
train_val_split = [0.8, 0.2]

train_set, val_set, eval_set_speed, eval_set_angle = build_training_validation_and_evaluation_sets(train_image_paths_speed_0, train_image_paths_speed_1,
                                                                   image_shape,
                                                                   batch_size,
                                                                    eval_split,
                                                                   train_val_split  )


# --- TRAINING HYPERPARAMETERS ---
epochs = 200
logging = False # log using tensorboard
initial_learning_rate = 1.0
decay_steps = 10000
decay_rate = 0.95
pool_size=(2, 2)


model_speed, model_angle = create_cnn_model_v2(image_shape, pool_size)

lr_schedule = ExponentialDecay(
    initial_learning_rate, decay_steps, decay_rate, staircase=True
)

# Use Adadelta optimizer for both models
optimizer = LegacyAdadelta(learning_rate=lr_schedule)

  # Compile each model separately

model_speed.compile(
   optimizer='Adam',
   loss={'output_speed': BinaryCrossentropy(from_logits=False)},
   loss_weights={'output_speed': 1},
   weighted_metrics=[]
)
model_angle.compile(
   optimizer='Adam',
   loss='mean_squared_error',  # Use mean squared error for regression
   loss_weights={'output_angle': 1},
   weighted_metrics=[]
)

#model_angle.compile(
#   optimizer='Adam',
#   loss={'output_angle':  MeanSquaredError()},
#   loss_weights={'output_angle': 1},
#   weighted_metrics=[]
#)

model_angle.compile(
   optimizer='Adam',
   loss='mean_squared_error',  # Use mean squared error for regression
   loss_weights={'output_angle': 1},
   weighted_metrics=[]
)

# Define augmentations
datagen = ImageDataGenerator(
    channel_shift_range=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)


# Create custom data generators
train_generator = CustomDataGenerator(train_set, batch_size, image_shape, augmentations=datagen)
val_generator = CustomDataGenerator(val_set, batch_size, image_shape)

# Create custom data generators
train_generator_2 = CustomDataGenerator_2(train_set, batch_size, image_shape, augmentations=datagen)
val_generator_2 = CustomDataGenerator_2(val_set, batch_size, image_shape)

# Fit each model separately
history_speed = model_speed.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    verbose=1  # Set verbose to 0 to disable the default progress bar
)

history_angle = model_angle.fit(
    train_generator_2,
    epochs=epochs,
    validation_data=val_generator_2,
    verbose=1  # Set verbose to 0 to disable the default progress bar
)

plot_loss(history_speed)
plot_loss(history_angle)
model_angle.trainable = False
model_angle.save('full_CNN_model_angle.h5')
model_angle.summary()

model_speed.trainable = False
model_speed.save('full_CNN_model_speed.h5')
model_speed.summary()


# # Make predictions using the trained model
#predictions1 = model_angle.predict(test_image_paths)
#predictions2 = model_speed.predict(test_image_paths)
## Create a DataFrame with image IDs and predictions
#results_df = pd.DataFrame({'ImageId': image_ids, 'Angle': predictions[:, 0], 'Speed': predictions[:, 1]})
#
# # Save the results DataFrame to the new submission file
#results_df.to_csv('new_submission.csv', index=False)