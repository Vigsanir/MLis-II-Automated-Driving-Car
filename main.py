# main.py

from pickle import TRUE
from preprocesing import get_dataset_path, get_project_path, preprocess_dataset, build_training_validation_and_evaluation_sets, build_training_directory_img, load_test_images
from CustomDataGenerator.CustomDataGenerator import create_data_generator
from models.cnn_model import create_cnn_model, create_cnn_model_v2, create_cnn_model_v3, create_cnn_model_v4
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.python.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os
from keras.models import load_model



def compile_model(model, output_label):

    if output_label == "speed":
        model.compile(
            optimizer='Adam',
            loss='binary_crossentropy',  # for binary classification
            metrics=[
                BinaryAccuracy(name='accuracy'),
                Precision(name='precision'),
                Recall(name='recall'),  
                AUC(name='auc'),
                MeanSquaredError(name='mse'),
                MeanAbsoluteError(name='mae')
            ]
        )
    else:
        model.compile(
            optimizer='Adam',
            loss='mean_squared_error',  # for regression
            metrics=[
                MeanSquaredError(name='mse'),
                MeanAbsoluteError(name='mae')
            ]
        )
    return model



def train(dataset_path, model, image_shape, output_label, epochs, directory='trained_models'):
    image_paths, data_labels = preprocess_dataset(dataset_path)
    train_set, val_set, _, _, train_labels, val_labels, _, _ = build_training_validation_and_evaluation_sets(image_paths,
                                                                                                                data_labels,
                                                                                                                image_shape,
                                                                                                                batch_size,
                                                                                                                eval_split,
                                                                                                                train_val_split)

    model = compile_model(model, output_label)    

    training_images_directory = build_training_directory_img(dataset_path)
    train_data_generator = create_data_generator(training_images_directory, train_labels, batch_size, image_shape, output_label)
    val_data_generator = create_data_generator(training_images_directory, val_labels, batch_size, image_shape, output_label)

    history = model.fit(
        train_data_generator,
        epochs=epochs,
        validation_data=val_data_generator,
        verbose=1  # Set verbose to 0 to disable the default progress bar
    )
      # Get the current date
    current_date = datetime.now().strftime("%m-%d_%H-%M")
    # Save the compiled model and trained.
    model.trainable = False
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(os.path.join(directory, f'{current_date}_CNN_model_{output_label}_epochs{epochs}.h5'))

    return history



def test_cnn_model(dataset_path, model, image_shape, output_label, directory='predictions_submision' ):
    # Load test data
    test_images, image_ids = load_test_images(dataset_path, image_shape)

    # Make predictions using the trained model

    predictions  = model.predict(test_images)

    print(f" Prediction {output_label}: {predictions }")
    # Flatten the nested lists
  
    flat_predictions = [item for sublist in predictions for item in sublist]
    # Create a DataFrame with image IDs and predictions
    results_df = pd.DataFrame({'image_id': image_ids, f'{output_label}': flat_predictions})
    print(results_df)
     # Get the current date
    current_date = datetime.now().strftime("%m-%d_%H-%M")
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)

      # Save the results DataFrame to the new submission file with current date
    results_df.to_csv(os.path.join(directory, f'{current_date}_prediction_{output_label}_test_epochs{epochs}.csv'))

def plot_loss_single(history, output_label,epochs, directory='plots'):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title(f'Model Loss {output_label}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')

    current_date = datetime.now().strftime("%m-%d_%H-%M")
        # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the plot in the directory
    plt.savefig(os.path.join(directory, f'{current_date}_Model_Loss_{output_label}_epochs{epochs}.png'))

    plt.show()

    
def plot_loss(history1, history2, epochs, directory='plots'):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss for history1
    axs[0].plot(history1.history['loss'], label='Train')
    axs[0].plot(history1.history['val_loss'], label='Validation')
    axs[0].set_title('Model Loss (History speed)')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend(loc='upper right')

    # Plot loss for history2
    axs[1].plot(history2.history['loss'], label='Train')
    axs[1].plot(history2.history['val_loss'], label='Validation')
    axs[1].set_title('Model Loss (History angle)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Loss')
    axs[1].legend(loc='upper right')

    plt.tight_layout()

    current_date = datetime.now().strftime("%m-%d_%H-%M")
        # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save the plot in the directory
    plt.savefig(os.path.join(directory, f'{current_date}_Model_Loss_combined(S_A)_epochs{epochs}.png'))

    plt.show()

def train_test_model(dataset_path, model, output_label, epochs, image_shape):
    history = train(dataset_path ,model, image_shape, output_label, epochs)
    test_cnn_model(dataset_path, model, image_shape, output_label)
    return history



# Call the training function
if __name__ == '__main__':

    # DATA HYPERPARAMETERS
    batch_size = 128    # Use 32 for training.
    # image_shape = (int(240/2), int(320/2)) # Half the real size of the image.
    image_shape = (120,120)
    eval_split = 0.1
    train_val_split = [0.8, 0.2] # [training_set %, valuation_set %]

    # TRAINING HYPERPARAMETERS 
    learning_rate = 0.001  # Specify your desired learning rate~ 
    epochs = 2
    logging = True # Set to True, the training process might log various metrics (such as loss and accuracy) for visualization and analysis using TensorBoard.
    pool_size=(2, 2)

    FIRST_TRAIN_FLAG = False  # SET TRUE if it is the first train. False if want to continue training a model. Update with model paths
    MODEL_SPEED_FLAG = True  # SET TRUE if want to train the SPEED model
    MODEL_ANGLE_FLAG = True  # SET TRUE if want to train the ANGLE model


    dataset_path = get_dataset_path()

    if FIRST_TRAIN_FLAG:
        # Create and train the models
        model_speed, model_angle = create_cnn_model_v4(image_shape, pool_size)
        if MODEL_SPEED_FLAG:
            history_speed = train_test_model(dataset_path, model_speed, "speed", epochs, image_shape)
        if MODEL_ANGLE_FLAG:
            history_angle = train_test_model(dataset_path, model_angle, "angle", epochs, image_shape)
    else:   
        project_path = get_project_path()
        if MODEL_SPEED_FLAG:
            model_path_speed = f'{project_path}/trained_models/03-18_23-10_CNN_model_speed_epochs200.h5'  # Update with speed model path
            model = load_model(model_path_speed)
                        # Unfreeze all layers for training
            model.trainable = True
            history_speed = train_test_model(dataset_path, model, "speed", epochs, image_shape)

        if MODEL_ANGLE_FLAG:
            model_path_angle = f'{project_path}/trained_models/03-19_07-25_CNN_model_angle_epochs400.h5'  # Update with angle model path
            model = load_model(model_path_angle)
            # Unfreeze all layers for training
            model.trainable = True
            history_angle = train_test_model(dataset_path, model, "angle", epochs, image_shape)

    if MODEL_SPEED_FLAG:
        plot_loss_single(history_speed, "speed", epochs)
    if MODEL_ANGLE_FLAG:
        plot_loss_single(history_angle, "angle", epochs)
    if MODEL_SPEED_FLAG and MODEL_ANGLE_FLAG:
        plot_loss(history_speed, history_angle, epochs)


