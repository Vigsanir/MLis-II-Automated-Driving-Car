import os
from pathlib import Path
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

def get_dataset_path():
    # Check if running in Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # Running in Kaggle kernel
        return '/kaggle/input/machine-learning-in-science-ii-2024'
    else:
        # Running in a standalone Python script
        return f"{Path(__file__).parent}/machine-learning-in-science-ii-2024"

def build_training_directory_img(dataset_directory):
    return f"{dataset_directory}/training_data/training_data"

def build_test_directory_img(dataset_directory):
    return f"{dataset_directory}/test_data/test_data"

def construct_images_path(directory_with_images):
    filenames = os.listdir(directory_with_images) 
    image_paths = []
    if any ('.png' in filename for filename in filenames):
        for index, current_filename in enumerate(filenames):
            image_path = f"{directory_with_images}/{current_filename}"
            image_paths.append(image_path)
            
    return image_paths


def build_images_path(dataset_directory):
    training_images_directory = build_training_directory_img(dataset_directory)
    test_images_directory = build_test_directory_img(dataset_directory)
    training_images_path = construct_images_path(training_images_directory)
    test_images_path = construct_images_path(test_images_directory)
    return training_images_path,test_images_path

def get_csv_labels(dataset_directory):
    training_labels_relative_path = f"{dataset_directory}/training_norm.csv" # data frame path
    training_labels = pd.read_csv(training_labels_relative_path)   # store the labels
    return training_labels

def find_corrupted_images(image_paths):
    corrupted_indices = []
    #print(image_paths)

    for i, image_path in enumerate(image_paths):
        try:
            # Attempt to open the image
            with Image.open(image_path) as img:
                # Perform any additional checks if needed
                
                pass
        except (IOError, OSError, Exception) as e:
            # Extract the index from the image_path
            index = int(image_path.split('/')[-1].split('.png')[0])
            # Handle the error (consider the image as corrupted)
            print(f"Error opening image {index}.png: {e}")
            corrupted_indices.append(index)
            print(f'Corupted images: {corrupted_indices}')
    return corrupted_indices


def remove_corrupted_data_labels(training_labels, corrupted_indices):
    # Remove corrupted data from original data label set (training_norm.csv)
    # Extract the 'image_id' column from the original DataFrame
    image_ids = training_labels['image_id']
    
    # Identify the 'image_id' values for the corrupted images
    corrupted_image_ids = image_ids.iloc[corrupted_indices].tolist()

    # Check if each 'image_id' is in the list of corrupted 'image_id' values
    is_not_corrupted = ~image_ids.isin(corrupted_image_ids)

    # Filter the original DataFrame
    data_labels_cleaned = training_labels[is_not_corrupted]

    # Reset the index of the cleaned DataFrame
    data_labels_cleaned = data_labels_cleaned.reset_index(drop=True)
    
    return data_labels_cleaned

def remove_corrupted_data_images(dataset_directory, image_paths, corrupted_indices ):
    # Remove corrupted images from original path list images:
    # Remove corresponding image paths for corrupted images
    training_images_directory = build_training_directory_img(dataset_directory)
    filenames = os.listdir(training_images_directory)

    # Initialize a list to store cleaned image paths
    image_paths_cleaned = []

    for current_filename in filenames:
        image_id = int(current_filename.split('.')[0])
        
        # Check if the image_id is not in the list of corrupted indices
        if image_id not in corrupted_indices:
            image_path = os.path.join(training_images_directory, current_filename)
            image_paths_cleaned.append(image_path)
    return image_paths_cleaned

def check_image_paths_consistency(dataset_directory, train_img, dataset_path):
    print(len(train_img))
    training_labels_test = get_csv_labels(dataset_path)
    image_ids_to_check = [3884, 10171, 3141, 3999, 4895, 8285 , 1017, 13139, 13786, 13782]
    print(training_labels_test)

    # Check if the image IDs exist in the list of cleaned image paths
    for image_id in image_ids_to_check:
        filename_to_check = f"{dataset_directory}/training_data/training_data/{image_id}.png"
        if filename_to_check in train_img:
            print(f"The file path for {filename_to_check} exists in the list.")
        else:
            print(f"The file path for {filename_to_check} does not exist in the list.")

    # Check if the image IDs exist in the DataFrame
    for image_id in image_ids_to_check:
        if image_id in training_labels_test['image_id'].tolist():
            print(f"The image ID {image_id} exists in training_labels_test.")
        else:
            print(f"The image ID {image_id} does not exist in training_labels_test.")

def remove_corrupted_data(dataset_directory, training_labels, image_paths, corrupted_indices):
    # Remove corrupted data from original data label set (training_norm.csv)
    data_labels_cleaned = remove_corrupted_data_labels(training_labels, corrupted_indices)
    # Remove corrupted images from original path list images:
    image_paths_cleaned = remove_corrupted_data_images(dataset_directory, image_paths, corrupted_indices)
    
    # Return cleaned DataFrame and image paths
    return data_labels_cleaned, image_paths_cleaned

def remove_invalid_speed_data(dataset_directory, data_labels, image_paths):
    # Filter rows where 'speed' is not 0 or 1
  
    invalid_speed_filter = (data_labels['speed'] != 0) & (data_labels['speed'] != 1)
    invalid_speed_rows = data_labels[invalid_speed_filter]
    print(f'Removed rows with invalid speed values:\n{invalid_speed_rows}')

    # Get the indices of removed rows
    removed_indices = invalid_speed_rows.index.tolist()
    print(f'Indices of removed rows: {removed_indices}')

    # Print the 'image_id' values of removed rows
    removed_image_ids = invalid_speed_rows['image_id'].tolist()
    print(f'Image_id values of removed rows: {removed_image_ids}')

    # Filter rows where 'speed' is not 0 or 1
    valid_speed_rows = data_labels[~invalid_speed_filter]

    # Extract the 'image_id' column from valid speed rows
    valid_image_ids = valid_speed_rows['image_id'].tolist()
    print(f"Number of entries with valid 'speed' values: {len(valid_speed_rows)}")
    # Filter image_paths based on valid image_ids
    cleaned_image_paths = [f"{dataset_directory}{filename}.png" for filename in valid_speed_rows['image_id'].values]
    # print(f'GOOD PATH: {cleaned_image_paths}')

     # Print paths that are not in cleaned_image_paths
    not_included_paths = [path for path in image_paths if path not in cleaned_image_paths]
    #print(f'Paths not included in cleaned_image_paths:')
    for path in not_included_paths:
       # print(f"NOT INCLUDED {path}")
       pass
    
    # Return cleaned DataFrame and corresponding image paths
    return valid_speed_rows, cleaned_image_paths

def load_test_images(dataset_path, image_size=(200, 200)):
    test_images = []
    image_ids = []
    test_data_dir = build_test_directory_img(dataset_path)
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

    return np.array(test_images), image_ids

def plot_data_scatter(data_labels_cleaned2):
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Scatter plot for 'angle'
    axes[0].scatter(range(len(data_labels_cleaned2['angle'])), data_labels_cleaned2['angle'])
    axes[0].set_title('Scatter Plot: Angle')
    axes[0].set_xlabel('Data Point Index')
    axes[0].set_ylabel('Angle')

    # Scatter plot for 'speed'
    axes[1].scatter(range(len(data_labels_cleaned2['speed'])), data_labels_cleaned2['speed'])
    axes[1].set_title('Scatter Plot: Speed')
    axes[1].set_xlabel('Data Point Index')
    axes[1].set_ylabel('Speed')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

def plot_data_distribution(data_labels_cleaned, data_labels_cleaned2):
    # Create a figure and axes for subplots
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Plot the first box plot for 'angle'
    data_labels_cleaned['angle'].plot(kind='box', ax=axes[0, 0])
    axes[0, 0].set_title('Distribution of Angle - DataFrame 1')

    # Plot the first box plot for 'speed'
    data_labels_cleaned['speed'].plot(kind='box', ax=axes[0, 1])
    axes[0, 1].set_title('Distribution of Speed - DataFrame 1')

    # Histogram for 'angle' in DataFrame 1
    data_labels_cleaned['angle'].plot(kind='hist', bins=20, ax=axes[0, 2])
    axes[0, 2].set_title('Histogram of Angle - DataFrame 1')

    # Scatter plot of 'angle' against 'speed' in DataFrame 1
    axes[0, 3].scatter(data_labels_cleaned['angle'], data_labels_cleaned['speed'])
    axes[0, 3].set_title('Scatter Plot: Angle vs Speed - DataFrame 1')
    axes[0, 3].set_xlabel('Angle')
    axes[0, 3].set_ylabel('Speed')

    # Plot the second box plot for 'angle'
    data_labels_cleaned2['angle'].plot(kind='box', ax=axes[1, 0])
    axes[1, 0].set_title('Distribution of Angle - DataFrame 2')

    # Plot the second box plot for 'speed'
    data_labels_cleaned2['speed'].plot(kind='box', ax=axes[1, 1])
    axes[1, 1].set_title('Distribution of Speed - DataFrame 2')

    # Histogram for 'angle' in DataFrame 2
    data_labels_cleaned2['angle'].plot(kind='hist', bins=20, ax=axes[1, 2])
    axes[1, 2].set_title('Histogram of Angle - DataFrame 2')

    # Scatter plot of 'angle' against 'speed' in DataFrame 2
    axes[1, 3].scatter(data_labels_cleaned2['angle'], data_labels_cleaned2['speed'])
    axes[1, 3].set_title('Scatter Plot: Angle vs Speed - DataFrame 2')
    axes[1, 3].set_xlabel('Angle')
    axes[1, 3].set_ylabel('Speed')

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

def preprocess_dataset(dataset_path):
    # Build image paths
    train_img, test_img = build_images_path(dataset_path)
    
    # Load training labels
    training_labels = get_csv_labels(dataset_path)
    
    # Find corrupted images
    corrupted_indices = find_corrupted_images(train_img)
    
    # Remove corrupted data
    data_labels_cleaned, image_paths_cleaned = remove_corrupted_data(dataset_path, training_labels, train_img, corrupted_indices)
    
    # Remove invalid speed data
    data_labels_cleaned2, image_paths_cleaned2 = remove_invalid_speed_data(dataset_path, data_labels_cleaned, image_paths_cleaned)
    
    # Check image paths consistency
    check_image_paths_consistency(dataset_path, train_img, dataset_path)
    
    return image_paths_cleaned2, data_labels_cleaned2

def build_training_validation_and_evaluation_sets(train_image_paths, data_labels, image_shape, batch_size, eval_split, train_val_split):


    # Split into training and validation sets our data set [images and labels]
    train_set, val_set, train_labels, val_labels = train_test_split(
        train_image_paths,
        data_labels,
        test_size=train_val_split[1],
        random_state=42, # 42 is a random value 
        #stratify=data_labels['speed'] # We want to make sure that we have similar distribution of of the target variable ('speed') 
                                      # is similar in both the training and validation sets.
    )
  
    # Split validation set into evaluation sets for speed and angle
    eval_set_speed, eval_set_angle, eval_labels_speed, eval_labels_angle = train_test_split(
        val_set,
        val_labels,
        test_size=eval_split,
        random_state=42, # 42 is a random value 
        #stratify=data_labels['speed'] # We want to make sure that we have similar distribution of of the target variable ('speed') 
                                      # is similar in both the training and validation sets.
    )
   # Additional processing as needed (e.g., loading images, data augmentation) - Here we can add more images if we'll need.

    # Print summary
    print(f"\nFound {len(train_image_paths)} images.")
    print(f"Using {len(train_set)} ({round(len(train_set) / len(train_image_paths) * 100, 1)}%) for training.")
    print(f"Using {len(val_set)} ({round(len(val_set) / len(train_image_paths) * 100, 1)}%) for validation.")
    print(f"Using {len(eval_set_speed)} ({round(len(eval_set_speed) / len(train_image_paths) * 100, 1)}%) for evaluation of speed.")
    print(f"Using {len(eval_set_angle)} ({round(len(eval_set_angle) / len(train_image_paths) * 100, 1)}%) for evaluation of angle.")
    #print(train_set)

    # Additional return statements as needed
    return train_set, val_set, eval_set_speed, eval_set_angle, train_labels, val_labels, eval_labels_speed, eval_labels_angle

############################################################################################################

#dataset_path = get_dataset_path()
#train_img, test_img = build_images_path(dataset_path)
## print(train_img) # print list with all images.
#print(f"Dataset path: \n{train_img[0]} \n{test_img[0]}")
#
#training_labels = get_csv_labels(dataset_path)
#print(training_labels)
#
#
## Find corrupted images
#corrupted_indices = find_corrupted_images(train_img)
#
#data_labels_cleaned, image_paths_cleaned = remove_corrupted_data(dataset_path, training_labels, train_img, corrupted_indices)
#
## Print the indices of corrupted images
#print("Corrupted Image Indices:", corrupted_indices)
#
#print("REMOVED CORRUPTED IMAGES <IMAGES> :")
#print(f'Labels: new:{len(data_labels_cleaned)} - old:{len(training_labels)}')
#print(f'Images: new:{len(image_paths_cleaned)} - old:{len(train_img)}')
#
#
#data_labels_cleaned2, image_paths_cleaned2 = remove_invalid_speed_data(dataset_path, data_labels_cleaned, image_paths_cleaned)
#
#check_image_paths_consistency(dataset_path, train_img, dataset_path)
#
#print( "REMOVED INVALID DATA <SPEED>:")
#print(f': Labels: New count:{len(data_labels_cleaned2)} ? Previous count:{len(data_labels_cleaned)}')
#print(f': Images: New count:{len(image_paths_cleaned2)} ? Previous count:{len(image_paths_cleaned)}')
#print(f"Type of image_paths_cleaned2 {type(image_paths_cleaned2)}.")


# plot_data_distribution(data_labels_cleaned, data_labels_cleaned2)
# plot_data_scatter(data_labels_cleaned2)
#################################################################################################################