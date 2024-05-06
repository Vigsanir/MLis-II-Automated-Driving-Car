# main.py

# Custom module imports
from speed_model import run_speed_model
from angle_model import run_angle_model

# Flags to indicate whether to train the SPEED model and/or the ANGLE model
MODEL_SPEED_FLAG = True   # Set TRUE if want to train the SPEED model
MODEL_ANGLE_FLAG = True  # Set TRUE if want to train the ANGLE model

# SPEED MODEL HYPERPARAMETERS
# Data hyperparameters
batch_size_s = 128    # Use 32 for training. Number of samples per gradient update.
image_shape_s = (120, 120)     # Size of the input images. Real size of the image.
pool_size_s = (2, 2)  # Size of the pooling operation.
augmentation_s = True   # Data augmentation flag, whether to apply data augmentation during training.

# Training hyperparameters
learning_rate_s = 0.0000001  # Specify your desired learning rate
epochs_s = 10  # Number of epochs to train the model.
eval_split_s = 0.1  # Percentage of the data to be used as test set.
train_val_split_s = 0.2  # Percentage of data to be used for training and validation.

logging_s = True  # Logging flag: Set to True if the training process might log various metrics for visualization and analysis using TensorBoard.
FIRST_TRAIN_FLAG_s = True  # Set to True if it is the first train of the model, False if you want to continue training a model that has been trained in the past.                     
DATA_SPLIT_TO_EVALUATE_FLAG_s = True  # Set to True if you want to split the data for evaluation.
saved_speed_model = '03-24_15-47_CNN_model_speed_epochs1050.h5'  # Path to the saved SPEED model

# ANGLE MODEL HYPERPARAMETERS
# Data hyperparameters
batch_size_a = 128    # Use 32 for training. Number of samples per gradient update.
image_shape_a = (120, 120)     # Size of the input images. Real size of the image.
pool_size_a = (2, 2)  # Size of the pooling operation.
augmentation_a = True   # Data augmentation flag, whether to apply data augmentation during training.

# Training hyperparameters
learning_rate_a = 0.0000001  # Specify your desired learning rate
epochs_a = 10  # Number of epochs to train the model.
eval_split_a = 0.1  # Percentage of the data to be used as test set.
train_val_split_a = 0.2  # Percentage of data to be used for training and validation.

logging_a = True  # Logging flag: Set to True if the training process might log various metrics for visualization and analysis using TensorBoard.
FIRST_TRAIN_FLAG_a = True  # Set to True if it is the first train of the model, False if you want to continue training a model that has been trained in the past.                     
DATA_SPLIT_TO_EVALUATE_FLAG_a = True  # Set to True if you want to split the data for evaluation.
saved_angle_model = '03-27_22-06_CNN_model_angle_epochs3000.h5'  # Path to the saved ANGLE model

if __name__ == '__main__':
    # Train the SPEED model if flag is set
    if MODEL_SPEED_FLAG:
        # Packing hyperparameters into tuples for passing to the run_speed_model function
        data_hyperparameters_s = (batch_size_s, image_shape_s, pool_size_s, augmentation_s)
        training_hyperparameters_s = (learning_rate_s, epochs_s, eval_split_s, train_val_split_s)
        setting_s = (logging_s, FIRST_TRAIN_FLAG_s, DATA_SPLIT_TO_EVALUATE_FLAG_s)

        # Calling the function to run the SPEED model with the specified hyperparameters
        run_speed_model(data_hyperparameters_s, training_hyperparameters_s, setting_s, saved_speed_model)

    # Train the ANGLE model if flag is set
    if MODEL_ANGLE_FLAG:
        # Packing hyperparameters into tuples for passing to the run_angle_model function
        data_hyperparameters_a = (batch_size_a, image_shape_a, pool_size_a, augmentation_a)
        training_hyperparameters_a = (learning_rate_a, epochs_a, eval_split_a, train_val_split_a)
        setting_a = (logging_a, FIRST_TRAIN_FLAG_a, DATA_SPLIT_TO_EVALUATE_FLAG_a)

        # Calling the function to run the ANGLE model with the specified hyperparameters
        run_angle_model(data_hyperparameters_a, training_hyperparameters_a, setting_a, saved_angle_model)
