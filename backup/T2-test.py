# Import the CustomDataGenerator class from CustomDataGenerator.py
from CustomDataGenerator import create_data_generator
from panda import pd

# Define a sample dataset and labels
# Define the paths to the images
file_paths = [
    'test_images/1.png', 'test_images/2.png', 'test_images/3.png', 'test_images/4.png',
    'test_images/5.png', 'test_images/6.png', 'test_images/7.png', 'test_images/8.png',
    'test_images/9.png', 'test_images/10.png', 'test_images/11.png', 'test_images/12.png',
    'test_images/13.png', 'test_images/14.png', 'test_images/15.png', 'test_images/16.png'
]
# Example labels
labels = pd.DataFrame({'speed': [20, 30, 40]})

# Define batch size and image shape
batch_size = 2
image_shape = (224, 224)  # Example image shape

# Create the data generator using the create_data_generator function
generator = create_data_generator(file_paths, labels, batch_size, image_shape, output_label='speed')

# Test the generator by iterating over a few batches
num_batches = 3
for i in range(num_batches):
    batch_images, batch_labels = next(generator)
    print(f"Batch {i+1}:")
    print("Images shape:", batch_images.shape)
    print("Labels:", batch_labels)
    print()


# Define the paths to the images

 
# Example labels
labels = pd.DataFrame({'speed': [20, 30, 40, 25, 35, 45, 22, 32, 42, 27, 37, 47, 24, 34, 44, 29]})

## Read the CSV file containing the labels
#labels_df = pd.read_csv('labels.csv')

## Display the first 10 rows and type of labels_df
#print("First 10 rows of data_labels:")
#print(labels_df.head(10))
#print(type(labels_df))

# Extract the 'speed' column as the labels
#labels = labels_df['speed'].tolist()

# Define parameters for the data generator
batch_size = 2
image_shape = (2, 2)
output_label = 'speed'

