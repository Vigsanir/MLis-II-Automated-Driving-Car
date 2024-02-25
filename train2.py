# train.py
from models.cnn_model import create_cnn_model
from utils.preprocessing import load_and_preprocess_images
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Absolute file paths
training_data_dir = r"C:\Users\vagel\OneDrive\Υπολογιστής\Ml Nottingham Masters\MLinS II\training_data\training_data"
training_labels_path = r"C:\Users\vagel\OneDrive\Υπολογιστής\Ml Nottingham Masters\MLinS II\training_norm.csv"
model_path = 'models/trained_model.h5'
submission_csv_path = r"C:\Users\vagel\OneDrive\Υπολογιστής\Ml Nottingham Masters\MLinS II\submissionfile.csv"

# Load Dataset
labels_df = pd.read_csv(training_labels_path)

# Assuming load_and_preprocess_images is modified to work with ImageDataGenerator
# Setup the ImageDataGenerator for data augmentation and to read images in batches
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values
    # Include any other preprocessing or augmentation here
)

# Create a data generator for the training data
train_generator, val_generator = load_and_preprocess_images(datagen, training_data_dir, labels_df)

# Create the model
input_shape = (224, 224, 3)  # This should match the target_size in your generators
model = create_cnn_model(input_shape)

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model using the generator
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=val_generator,
    validation_steps=len(val_generator),
    epochs=10
)

# Save the trained model
model.save(model_path)

# Load the trained model
model = tf.keras.models.load_model(model_path)

# Load and preprocess test images
test_images, test_image_ids = load_and_preprocess_images(test_data_dir)  # Make sure this function is correctly implemented

# Predict using the trained model
predictions = model.predict(test_images)

# Denormalize the predictions
denormalized_angles = predictions[:, 0] * 80 + 50  # Adjust index if necessary
denormalized_speeds = predictions[:, 1] * 35       # Adjust index if necessary

# Create the submission DataFrame
submission_df = pd.DataFrame({
    'image_id': test_image_ids,
    'angle': denormalized_angles,
    'speed': denormalized_speeds
})

# Save the submission file
submission_df.to_csv(submission_csv_path, index=False)

print(f"Submission file saved to {submission_csv_path}")
