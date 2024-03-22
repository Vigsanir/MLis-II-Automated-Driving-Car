import os
from PIL import Image
import cv2
import numpy as np

# Create a folder to save images if it doesn't exist
if not os.path.exists('test_images'):
    os.makedirs('test_images')

# Generate all combinations of 4 bits
for num in range(16):  # Range from 0 to 15 for 16 combinations
    # Convert the number to its binary representation and zero-pad to 4 bits
    binary = format(num, '04b')

    # Convert binary string to list of integers
    pixel_values = [int(bit) * 255 for bit in binary]

    # Reshape pixel values to form a 2x2 image
    pixels = np.array(pixel_values).reshape((2, 2))

    # Create image with pixel values
    img = Image.fromarray(pixels.astype('uint8'), 'L')

    # Save image
    img_path = os.path.join('test_images', f'{num+1}.png')
    img.save(img_path)

    print(f'Image {num} saved as {img_path}')

# Open the image
img_path = 'test_images/9.png'  # Change the path if necessary
img = cv2.imread(img_path)

# Convert the image to a numpy array
img_array = np.asarray(img)

# Print the pixel values
print("Pixel values:")
print(img_array)