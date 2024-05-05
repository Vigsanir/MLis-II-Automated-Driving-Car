# main.py
# Standard library imports
import os
from datetime import datetime

# External library imports
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from pickle import TRUE
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.python.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError

# Custom module imports
from CustomDataGenerator.CustomDataGenerator import create_data_generator
from models.cnn_model import (
    create_cnn_model, 
    create_cnn_model_v2, 
    create_cnn_model_v3, 
    create_cnn_model_v4
)
from evaluation_metrics import (
    print_plot_classification_metrics, 
    print_plot_regression_metrics
)
from preprocesing import (
    get_dataset_path, 
    get_project_path, 
    preprocess_dataset, 
    build_training_validation_and_evaluation_sets, 
    build_training_directory_img, 
    load_test_images, 
    load_and_preprocess_images
)

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



def train(dataset_path, train_labels, val_labels,  model, image_shape, output_label, augumentation, epochs, directory='trained_models'):

    model = compile_model(model, output_label)    

    training_images_directory = build_training_directory_img(dataset_path)
    train_data_generator = create_data_generator(training_images_directory, train_labels, batch_size, image_shape, output_label, augumentation)
    val_data_generator = create_data_generator(training_images_directory, val_labels, batch_size, image_shape, output_label, augumentation)

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



def test_cnn_model(dataset_path, model, image_shape, output_label,DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, epochs, directory='predictions_submision' ):

    if DATA_SPLIT_TO_EVALUATE_FLAG:
        # Load and preprocess images
        [Test_set_path, Test_labels] = evaluate_df
        test_images = load_and_preprocess_images(Test_set_path, [image_shape[1], image_shape[0]])
        image_ids = Test_labels['image_id']
        y_true = Test_labels[output_label]
    else:
        # Load test data
        test_images, image_ids = load_test_images(dataset_path, [image_shape[1], image_shape[0]])
        y_true = None

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

    if DATA_SPLIT_TO_EVALUATE_FLAG:
                  # Save the results DataFrame to the new submission file with current date
        results_df.to_csv(os.path.join(directory, f'{current_date}_prediction_4EVAL_{output_label}_test_epochs{epochs}.csv'))
    else:
        # Save the results DataFrame to the new submission file with current date
        results_df.to_csv(os.path.join(directory, f'{current_date}_prediction_{output_label}_test_epochs{epochs}.csv'))
    return results_df, y_true

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
 

    # Close the current figure
    plt.close()

    
def plot_loss(history1, history2, epochs_speed, epochs_angle , directory='plots'):
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
    epochs = max(epochs_speed, epochs_angle)
    # Save the plot in the directory
    plt.savefig(os.path.join(directory, f'{current_date}_Model_Loss_combined(S_A)_epochs{epochs}.png'))

    plt.show()


    # Close the current figure
    plt.close()

def print_split_data(X_train, X_val, y_train, y_val):
    # Display some samples from the training and validation sets
    print("\nSample from X Training set:")
    for sample in X_train[:3]:
        print(sample)

    print("\nSample from Y Training set:")
    print(y_train[:3])

    print("\nSample from X Validation set:")
    for sample in X_val[:3]:
        print(sample)

    print("\nSample from Y Validation set:")
    print(y_val[:3])

def train_test_model(dataset_path, train_labels, val_labels, model, output_label, augumentation, epochs, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df):
    history = train(dataset_path, train_labels, val_labels, model, image_shape, output_label, augumentation, epochs)
    predicted_values, y_true = test_cnn_model(dataset_path, model, image_shape, output_label, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, epochs)
    return history, predicted_values, y_true



# Call the training function
if __name__ == '__main__':

    # DATA HYPERPARAMETERS
    batch_size = 128    # Use 32 for training.
    image_shape = (120,120)     # [240, 320] real size of the image.
    eval_split = 0.1    # Test_set %
    train_val_split = 0.2 # [training and_validation_set %]
    augumentation = True

    # TRAINING HYPERPARAMETERS 
    learning_rate = 0.0000001  # Specify your desired learning rate~ 
    epochs = 1
    epochs_speed = 300
    epochs_angle = 500
    logging = False # Set to True, the training process might log various metrics (such as loss and accuracy) for visualization and analysis using TensorBoard.
    pool_size=(2, 2)

    FIRST_TRAIN_FLAG = False  # SET TRUE if it is the first train of the model. 
                              # SET FALSE if want to continue training a model that has been trained in the past. Update with model paths
    MODEL_SPEED_FLAG = True   # SET TRUE if want to train the SPEED model
    MODEL_ANGLE_FLAG = True  # SET TRUE if want to train the ANGLE model
    DATA_SPLIT_TO_EVALUATE_FLAG = False # SET TRUE if want to split the data for evaluation.



    dataset_path = get_dataset_path()
    image_paths, data_labels = preprocess_dataset(dataset_path) #<class 'list'> <class 'pandas.core.frame.DataFrame'>
    print(type(image_paths), type(data_labels))

    if DATA_SPLIT_TO_EVALUATE_FLAG:
        TrainVal_set_path, Test_set_path, TrainVal_labels, Test_labels = build_training_validation_and_evaluation_sets(image_paths,
                                                                                                                  data_labels,                                                                                                          
                                                                                                                  eval_split)
        print_split_data(TrainVal_set_path, Test_set_path, TrainVal_labels, Test_labels )

        evaluate_df = [Test_set_path, Test_labels]

    else:
        TrainVal_set_path = image_paths
        TrainVal_labels = data_labels
        evaluate_df = [None, None]

    X_train, X_val, train_labels, val_labels = build_training_validation_and_evaluation_sets(TrainVal_set_path,
                                                                                               TrainVal_labels,
                                                                                               train_val_split)

    print_split_data(X_train, X_val, train_labels, val_labels)

    if FIRST_TRAIN_FLAG:
        # Create and train the models
        model_speed, model_angle = create_cnn_model_v4(image_shape, pool_size)
        if MODEL_SPEED_FLAG:
            history_speed, predicted_speed, y_true_speed = train_test_model(dataset_path, train_labels, val_labels, model_speed, "speed", augumentation, epochs_speed, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df)
        if MODEL_ANGLE_FLAG:
            history_angle, predicted_angle, y_true_angle = train_test_model(dataset_path, train_labels, val_labels, model_angle, "angle", augumentation, epochs_angle, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df)
    else:   
        project_path = get_project_path()
        if MODEL_SPEED_FLAG:
            model_path_speed = f'{project_path}/trained_models/03-24_15-47_CNN_model_speed_epochs1050.h5'  # Update with speed model path
            model = load_model(model_path_speed)
                        # Unfreeze all layers for training
            model.trainable = True
            history_speed, predicted_speed, y_true_speed = train_test_model(dataset_path, train_labels, val_labels, model, "speed", augumentation, epochs_speed, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df)

        if MODEL_ANGLE_FLAG:
            model_path_angle = f'{project_path}/trained_models/03-27_22-06_CNN_model_angle_epochs3000.h5'  # Update with angle model path
            model = load_model(model_path_angle)
            # Unfreeze all layers for training
            model.trainable = True
            history_angle, predicted_angle, y_true_angle = train_test_model(dataset_path, train_labels, val_labels, model, "angle", augumentation, epochs_angle, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df)

    if MODEL_SPEED_FLAG == True and MODEL_ANGLE_FLAG == False:
        if DATA_SPLIT_TO_EVALUATE_FLAG:
            print_plot_classification_metrics(y_true_speed, predicted_speed)
        plot_loss_single(history_speed, "speed", epochs_speed)

    if MODEL_SPEED_FLAG == False and MODEL_ANGLE_FLAG == True:
        if DATA_SPLIT_TO_EVALUATE_FLAG:
            print_plot_regression_metrics(y_true_angle, predicted_angle)
        plot_loss_single(history_angle, "angle", epochs_angle)

    if MODEL_SPEED_FLAG and MODEL_ANGLE_FLAG:
        if DATA_SPLIT_TO_EVALUATE_FLAG:
            print_plot_classification_metrics(y_true_speed, predicted_speed)
            print_plot_regression_metrics(y_true_angle, predicted_angle)
        plot_loss(history_speed, history_angle, epochs_speed, epochs_angle )


