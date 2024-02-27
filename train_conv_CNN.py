# train_conv_CNN.py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.debug_utils import debug_log
from readData import construct_paths, load_training_data
from models.cnn_model import create_cnn_model
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator

# Absolute file paths

training_data_dir, training_labels_path, test_data_dir, submission_csv_path = construct_paths()


model_path = 'models/trained_model.h5'

# Load the CSV file into a Pandas DataFrame
labels_df = pd.read_csv(training_labels_path)

# Print the head of the DataFrame in debug mode
debug_log("Head of the DataFrame:", data=labels_df.head())

# Return the path for training and testing data.
training_data_dir, training_labels_path, test_data_dir, submission_csv_path = construct_paths()
# Load training data.
training_images, labels = load_training_data(training_data_dir, training_labels_path, (200, 200))
# Shuffle the training data
#train_imagtraining_imageses, labels = shuffle(training_images, labels)




X_train, X_val, y_train, y_val = train_test_split(training_images, labels, test_size=0.1)
batch_size = 128
epochs = 10
pool_size = (2, 2)
input_shape = X_train.shape[1:]
model = create_cnn_model(input_shape, pool_size)
datagen = ImageDataGenerator(channel_shift_range=0.2)
datagen.fit(X_train)
model.compile(optimizer='Adam', loss='mean_squared_error')
model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), steps_per_epoch=len(X_train)/batch_size,
epochs=epochs, verbose=1, validation_data=(X_val, y_val))
model.trainable = False
model.save('full_CNN_model.h5')
model.summary()
