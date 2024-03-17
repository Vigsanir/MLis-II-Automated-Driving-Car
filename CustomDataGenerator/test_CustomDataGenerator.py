import os
from pathlib import Path
import pandas as pd 
from CustomDataGenerator import create_data_generator
from CustomDataGenerator2 import create_data_generator2
import numpy as np



def get_dataset_path():
    # Check if running in Kaggle
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        # Running in Kaggle kernel
        return '/kaggle/input/machine-learning-in-science-ii-2024'
    else:
        # Running in a standalone Python script
        return f'{Path(__file__).parent}/test_images'

print(get_dataset_path())   

def construct_images_path(directory_with_images):
    filenames = os.listdir(directory_with_images) 
    image_paths = []
    if any ('.png' in filename for filename in filenames):
        for index, current_filename in enumerate(filenames):
            image_path = f"{directory_with_images}/{current_filename}"
            image_paths.append(image_path)
            
    return image_paths

def get_csv_labels(dataset_directory):
    training_labels_relative_path = f"{dataset_directory}/test_labels.csv" # data frame path
    training_labels = pd.read_csv(training_labels_relative_path)   # store the labels
    return training_labels

def image_array_to_bit_value(image_array):
    """
    Convert an image array to its corresponding bit value.

    Args:
    image_array: numpy array representing an image where each pixel is represented by an RGB value.

    Returns:
    bit_value: The decimal representation of the binary value derived from the image array.
    """


    # Reshape the image array to get RGB triplets
    image_array_reshaped = image_array.reshape(-1, 3)

    # Take the maximum value across each triplet
    max_values = np.max(image_array_reshaped, axis=1)
    print(max_values)
    # Convert RGB values to binary (0 or 1)
    binary_values = ['1' if np.all(pixel == 255) else '0' for pixel in max_values]

    # Join binary values and convert to decimal
    bit_value = int(''.join(binary_values), 2)

    return bit_value

def test_batch(batch_images, batch_labels, output_label):
    print("Batch Images:")
    for image in batch_images:
        print("[")
        for row in image:
            print(" ".join(str(row[i]) for i in range(len(row))))
        print("]")
    
    print("Batch Labels:", batch_labels)
    
    print("Test Images:")
    for i, (image, label) in enumerate(zip(batch_images, batch_labels)):
        bit_value = image_array_to_bit_value(image)
        value = batch_labels
        if i < len(value):
            print(f"Image {i+1}: Bit Value: {bit_value}, {output_label} Label: {value[i]}")
        else:
            print(f"Image {i+1}: Bit Value: {bit_value}, {output_label} Label: Not Available")

  

file_paths = construct_images_path(get_dataset_path())
print(file_paths)
#  class: list
#['C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/1.png',
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/10.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/11.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/12.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/13.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/14.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/15.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/16.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/2.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/3.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/4.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/5.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/6.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/7.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/8.png', 
#'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/9.png']
training_labels = get_csv_labels(get_dataset_path())
print(training_labels)
#type pandas.core.frame.DataFrame:
#    image_id  angle  speed
#0          1     21     11
#1          2     22     12
#2          3     23     13
#3          4     24     14
#4          5     25     15
#5          6     26     16
#6          7     27     17
#7          8     28     18
#8          9     29     19
#9         10     30     20
#10        11     31     21
#11        12     32     22
#12        13     33     23
#13        14     34     24
#14        15     35     25
#15        16     36     26

# Create data generator
# Create data generator
batch_size = 2
image_shape = (2, 2, 3)  # 2x2 pixels, RGB
output_label = 'speed'
augmentations = None
directory_img= 'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/test_images/'
data_generator = create_data_generator2(directory_img, training_labels, batch_size, image_shape)

epochs = 3
for epoch in range(epochs):
    
    print(f'EPOCH {epoch + 1}')
    data_generator.on_epoch_end()  # Reset the generator at the end of each epoch
    # Iterate over the data generator to get batches
    for batch_images, batch_labels in data_generator:
        print("Batch images shape:", batch_images.shape)
        print("Batch labels shape:", batch_labels.shape)
       # print('AAAAAAAAAAAAAAAAAAA', batch_labels)
        test_batch(batch_images, batch_labels, output_label)



