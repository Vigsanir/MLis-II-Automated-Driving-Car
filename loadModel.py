from keras.models import load_model
from readData2 import build_image_paths, get_script_directory
import pandas as pd
import numpy as np
from keras.preprocessing import image

# Load the model
model_speed = load_model('full_CNN_model_speed.h5')
model_angle = load_model('full_CNN_model_angle.h5')

# Display the summary of the model
    .summary()

# Display the summary of the model
model_angle.summary()

# Load image paths
directory = get_script_directory()
print(directory)
train_image_paths_speed_0, train_image_paths_speed_1, test_image_paths = build_image_paths(directory)

print("START PREDICT!!!    - Yoy! ")
print(test_image_paths)

# Load and preprocess the actual image data
def preprocess_images(image_paths):
    images = []
    for path in image_paths['image_path']:
        img = image.load_img(path, target_size=(64, 64))  # Adjust the target size based on your model's input shape
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        images.append(img_array)

    return np.vstack(images)

test_images = preprocess_images(test_image_paths)

# Make predictions using the trained model
predictions_speed = model_speed.predict(test_images)
predictions_angle = model_angle.predict(test_images)

# Assuming image_ids is defined based on your specific data
image_ids = test_image_paths.index

print(predictions_angle)
print(predictions_speed)

# Create a DataFrame with image IDs and predictions
results_df = pd.DataFrame({'image_id': image_ids, 'angle': predictions_angle.flatten(), 'speed':(predictions_speed.flatten() >= 0.5).astype(int)})

# Save the results DataFrame to the new submission file
results_df.to_csv('Submission_1.csv', index=False)
print("FINISH!! ")

697	1.9170332	3.85E-21