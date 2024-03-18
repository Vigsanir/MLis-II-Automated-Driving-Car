import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
import os
from pathlib import Path

class CustomDataGenerator(Sequence):
    def __init__(self, input_path, labels_df, batch_size, image_shape, output_label, augmentations=None, shuffle=True):
        super().__init__()
        self.input_path = input_path
        self.labels_df = labels_df
        self.batch_size = batch_size
        self.image_shape = image_shape
        self.output_label = output_label
        self.augmentations = augmentations
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.labels_df))
        self.image_data_generator = ImageDataGenerator()

    def __len__(self):
        return int(np.floor(len(self.labels_df) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indexes[index * self.batch_size: (index + 1) * self.batch_size]
        batch_df = self.labels_df.iloc[indices]
        batch_labels = batch_df[self.output_label].values
        batch_paths = [f"{self.input_path}/{filename}.png" for filename in batch_df['image_id'].values]

        batch_images = []
        for path in batch_paths:
            img = load_img(path, target_size=self.image_shape)
            img = img.convert("RGB")
            img_array = img_to_array(img)
            # Normalize the image array
            img_array = img_array / 255.0  # Scale pixel values to [0, 1]
            batch_images.append(img_array)

        batch_images = np.array(batch_images)
        output_values = np.array(batch_labels)

        return batch_images, output_values

    def on_epoch_end(self):
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        self.on_epoch_end()
        for i in range(len(self)):
            yield self[i]

def create_data_generator(input_path, labels_df, batch_size, image_shape, output_label, augmentations=None, shuffle=True ):
    return CustomDataGenerator(input_path, labels_df, batch_size, image_shape, output_label, augmentations=None, shuffle=True)

def get_dataset_path():
    if 'KAGGLE_KERNEL_RUN_TYPE' in os.environ:
        return '/kaggle/input/machine-learning-in-science-ii-2024'
    else:
        return f'{Path(__file__).parent}/test_images'

def construct_images_path(directory_with_images):
    filenames = os.listdir(directory_with_images) 
    image_paths = []
    if any ('.png' in filename for filename in filenames):
        for index, current_filename in enumerate(filenames):
            image_path = f"{directory_with_images}/{current_filename}"
            image_paths.append(image_path)
            
    return image_paths

def get_csv_labels(dataset_directory):
    training_labels_relative_path = f"{dataset_directory}/test_labels.csv" 
    training_labels = pd.read_csv(training_labels_relative_path)  
    return training_labels

def image_array_to_bit_value(image_array):
    image_array_reshaped = image_array.reshape(-1, 3)
    max_channel_values = np.max(image_array_reshaped, axis=1)
    binary_values = ['1' if channel == 1 else '0' for channel in max_channel_values]
    bit_value = int(''.join(binary_values), 2)
    return bit_value + 1

def test_batch(batch_images, batch_labels, output_label):
    
    
    print("Batch Labels:", batch_labels)
    
    print("Test Images:")
    for i, (image, label) in enumerate(zip(batch_images, batch_labels)):
        print(f"Batch Image {i+1}:")
        print("[")
        for row in image:
            print(" ".join(str(row[i]) for i in range(len(row))))
        print("]")
        bit_value = image_array_to_bit_value(image)
        print(f"Batch image {i+1}: Bit Value: {bit_value}, {output_label} Label: {label}")

#print(get_dataset_path())   
#file_paths = construct_images_path(get_dataset_path())
#print(file_paths)
#
#training_labels = get_csv_labels(get_dataset_path())
#print(training_labels)
#
#batch_size = 8 
#image_shape = (2, 2, 3)  
#output_label = 'angle'
#augmentations = None
#directory_img= 'C:\\Users\\Petru.Sacaleanu\\source\\repos\\MLis-II - The Balcans/CustomDataGenerator/test_images/'
#data_generator = create_data_generator(directory_img, training_labels, batch_size, image_shape, output_label)
#
#epochs = 2
#for epoch in range(epochs):    
#    print(f'EPOCH {epoch + 1}')
#    data_generator.on_epoch_end()  
#    for batch_images, batch_labels in data_generator:
#        print("Batch images shape:", batch_images.shape)
#        print("Batch labels shape:", batch_labels.shape)
#        test_batch(batch_images, batch_labels, output_label)
#