import os
import pandas as pd
from datetime import datetime
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.python.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error

# Custom module imports
from CustomDataGenerator.CustomDataGenerator import create_data_generator
from preprocesing import (
    build_training_directory_img, 
    load_test_images, 
    load_and_preprocess_images
)

def compile_model(model, output_label):
    """
    Compiles the Keras model based on the output label (binary classification or regression).

    Args:
    - model: Keras model to be compiled.
    - output_label: The type of output label, either "speed" for binary classification or anything else for regression.

    Returns:
    - Compiled Keras model.
    """
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

def train(dataset_path, train_labels, val_labels,  model, image_shape, output_label, augumentation, epochs, batch_size, directory='trained_models'):
    """
    Trains the Keras model.

    Args:
    - dataset_path: Path to the dataset.
    - train_labels: DataFrame containing training labels.
    - val_labels: DataFrame containing validation labels.
    - model: Keras model to be trained.
    - image_shape: Tuple representing the image dimensions.
    - output_label: The type of output label, either "speed" for binary classification or anything else for regression.
    - augumentation: Boolean indicating whether data augmentation is enabled.
    - epochs: Number of epochs for training.
    - batch_size: Batch size for training.
    - directory: Directory to save the trained model.

    Returns:
    - Training history.
    """
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

    # Initialize dictionary to store evaluation metrics for each epoch
    evaluation_metrics = []
    # Evaluate the model on the validation data after each epoch
    for epoch in range(epochs):
        # Evaluate the model on the validation data
        evaluation = model.evaluate(val_data_generator)
        # Print the evaluation metrics
        if output_label == "speed":

            print(f'Epoch {epoch + 1} - Validation Loss: {evaluation[0]}, Validation Binary Accuracy: {evaluation[1]}, Validation Precision: {evaluation[2]}, Validation Recall: {evaluation[3]}, Validation AUC: {evaluation[4]}, Validation MSE: {evaluation[5]}, Validation MAE: {evaluation[6]}')

            # Store the evaluation metrics in the list
            evaluation_metrics.append({
                'epoch': epoch + 1,
                'loss': evaluation[0],
                'binary_accuracy': evaluation[1],
                'precision': evaluation[2],
                'recall': evaluation[3],
                'auc': evaluation[4],
                'mse': evaluation[5],
                'mae': evaluation[6]
            })

        if output_label == "angle":

            print(f'Epoch {epoch + 1} - Validation Loss: {evaluation[0]}, Validation MSE: {evaluation[1]}, Validation MAE: {evaluation[2]}')
                  
            # Store the evaluation metrics in the list
            evaluation_metrics.append({
                'epoch': epoch + 1,
                'loss': evaluation[0],
                'mse': evaluation[1],
                'mae': evaluation[2]
            })
        
        
  
    # Get the current date
    current_date = datetime.now().strftime("%m-%d_%H-%M")
    # Save the compiled model and trained.
    model.trainable = False
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(os.path.join(directory, f'{current_date}_CNN_model_{output_label}_epochs{epochs}.h5'))

    return history, evaluation_metrics

def model_predict(dataset_path, model, image_shape, output_label, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, epochs, directory='predictions_submission'):
    """
    Tests the trained CNN model.

    Args:
    - dataset_path: Path to the dataset.
    - model: Trained Keras model.
    - image_shape: Tuple representing the image dimensions.
    - output_label: The type of output label, either "speed" for binary classification or anything else for regression.
    - DATA_SPLIT_TO_EVALUATE_FLAG: Boolean indicating whether to split the data for evaluation.
    - evaluate_df: List containing test dataset paths and labels.
    - epochs: Number of epochs for training.
    - directory: Directory to save the prediction results.

    Returns:
    - DataFrame containing prediction results.
    - True labels (only for evaluation purposes).
    """
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

    #print(f" Prediction {output_label}: {predictions }")
    # Flatten the nested lists
    flat_predictions = [item for sublist in predictions for item in sublist]

    # Create a DataFrame with image IDs and predictions
    results_df = pd.DataFrame({'image_id': image_ids, f'{output_label}': flat_predictions})
    #print(results_df)
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

def train_test_model(dataset_path, train_labels, val_labels, model, output_label, augumentation, epochs, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, batch_size):
    """
    Trains and tests the CNN model.

    Args:
    - dataset_path: Path to the dataset.
    - train_labels: DataFrame containing training labels.
    - val_labels: DataFrame containing validation labels.
    - model: Keras model to be trained.
    - output_label: The type of output label, either "speed" for binary classification or anything else for regression.
    - augumentation: Boolean indicating whether data augmentation is enabled.
    - epochs: Number of epochs for training.
    - image_shape: Tuple representing the image dimensions.
    - DATA_SPLIT_TO_EVALUATE_FLAG: Boolean indicating whether to split the data for evaluation.
    - evaluate_df: List containing test dataset paths and labels.
    - batch_size: Batch size for training.

    Returns:
    - Training history.
    - DataFrame containing prediction results.
    - True labels (only for evaluation purposes).
    """
    history,evaluation_metrics = train(dataset_path, train_labels, val_labels, model, image_shape, output_label, augumentation, epochs, batch_size)
    predicted_values, y_true = model_predict(dataset_path, model, image_shape, output_label, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, epochs)
    return history, evaluation_metrics, predicted_values, y_true
