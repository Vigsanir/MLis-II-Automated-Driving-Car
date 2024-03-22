# main.py
import train
from utils.preprocessing import load_and_preprocess_images
from models.cnn_model import create_cnn_model

# Call the training function
if __name__ == '__main__':
    train.train_model()
