from utils.debug_utils import debug_log  
import os
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy as np



# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

LOCAL_DEBUG_MODE = True
LOCAL_DEBUG_MODE_SHOW_IMG = False

def construct_paths():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths
    training_data_relative_path = "machine-learning-in-science-ii-2024/training_data/training_data"
    training_labels_relative_path = "machine-learning-in-science-ii-2024/training_norm.csv"
    test_data_relative_path = "machine-learning-in-science-ii-2024/test_data/test_data"
    submission_csv_relative_path = "machine-learning-in-science-ii-2024/sampleSubmission.csv"


    # Construct absolute paths
    training_data_dir = os.path.join(script_dir, training_data_relative_path)
    training_labels_path = os.path.join(script_dir, training_labels_relative_path)
    test_data_dir = os.path.join(script_dir, test_data_relative_path)
    submission_csv_path = os.path.join(script_dir, submission_csv_relative_path)

    # Print paths in debug mode
    debug_log("This is a debug message.", data=script_dir)
    debug_log(f"Training Data Directory: {training_data_dir}")
    debug_log(f"Training Labels Path: {training_labels_path}")
    debug_log(f"Test Data Directory: {test_data_dir}")
    debug_log(f"Submission CSV Path: {submission_csv_path}")

    # Return the constructed paths
    return training_data_dir, training_labels_path, test_data_dir, submission_csv_path


def load_training_data(training_data_dir, training_labels_path, desired_size=(320, 240)):
    # Load labels from CSV
    labels_df = pd.read_csv(training_labels_path)

    # Load images
    train_images = []
    # Load labels
    
    for image_id in labels_df['image_id']:
        # Construct the image path
        image_path = os.path.join(training_data_dir, f"{image_id}.png")
        image =  Image.open(image_path)
        # Convert to RGB mode with 24-bit depth
        image = image.convert('RGB')
        # Resize image if needed
        image = image.resize(desired_size)
        
        # Convert image to NumPy array
        image_array = np.array(image) 
        
        # Ensure the shape is (height, width, channels)
        if image_array.shape[-1] != 3:
            image_array = np.transpose(image_array, (1, 0, 2))  # Swap height and width
        
        train_images.append(image_array)



    # Assuming we have one image to show (for example, the first image)
    if LOCAL_DEBUG_MODE_SHOW_IMG:
            
        image_to_show = train_images[0]
        # Display the image
        plt.imshow(image_to_show)
        plt.title("Example Image")
        plt.show()

    # Convert to NumPy arrays
        # Stack images into a single array
   # train_images = np.stack(train_images)
    train_images = np.array(train_images, dtype=np.float32)
    labels = labels_df[['angle', 'speed']].values

    # Print shapes in debug mode
    if LOCAL_DEBUG_MODE:
        debug_log("Shape of train_images:", train_images.shape)
        debug_log("Shape of labels:", labels.shape)

    return train_images, labels




def load_test_images(test_data_dir, image_size=(200, 200)):
    test_images = []
    image_ids = []

    for filename in os.listdir(test_data_dir):
        if filename.endswith(".png"):
            image_path = os.path.join(test_data_dir, filename)
            img = Image.open(image_path)
            # Convert to RGB mode with 24-bit depth
            img = img.convert('RGB')

            # Resize the image to the specified size
            img = img.resize(image_size)
            
            # Convert the PIL Image to a NumPy array
            img_array = np.array(img)
            
            # Ensure the shape is (height, width, channels)
            if img_array.shape[-1] != 3:
                img_array = np.transpose(img_array, (1, 0, 2))  # Swap height and width

            test_images.append(img_array)
            image_ids.append(filename.split(".")[0])  # Extracting the image ID from the filename

    return np.array(test_images, dtype=np.float32), image_ids

# Check if in debug mode before executing example usage
if LOCAL_DEBUG_MODE:
    # Example usage
    debug_log("Run the LOCAL debug logic.")
    training_data_dir, training_labels_path, test_data_dir, submission_csv_path = construct_paths()
    load_training_data(training_data_dir, training_labels_path)

training_data_dir, training_labels_path, test_data_dir, _ = construct_paths()
    
    # Load the CSV file into a Pandas DataFrame
labels_df = pd.read_csv("new_submission.csv")
print(labels_df)
print(test_data_dir)