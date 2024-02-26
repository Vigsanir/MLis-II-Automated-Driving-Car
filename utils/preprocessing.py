# preprocessing.py
import os
import numpy as np
from PIL import Image

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image and preprocesses it for model training.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert('RGB')  # Ensure image is in RGB format
            img = img.resize(target_size)  # Resize image to target size
            img_array = np.array(img) / 255.0  # Scale pixel values to [0, 1]
            return img_array
    except Exception as e:
        print(f"Error processing image at {image_path}: {e}")
        return None

def load_and_preprocess_images(data_dir, image_ids, target_size=(224, 224)):
    """
    Loads and preprocesses a collection of images given their IDs.
    """
    images = []
    for image_id in image_ids:
        image_path = os.path.join(data_dir, f"{image_id}.png")
        img_array = load_and_preprocess_image(image_path, target_size)
        if img_array is not None:
            images.append(img_array)
    return np.array(images)
