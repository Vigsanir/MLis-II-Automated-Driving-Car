import os
import pandas as pd
from datetime import datetime
from tensorflow.python.keras.losses import BinaryCrossentropy, MeanSquaredError
from tensorflow.python.keras.metrics import BinaryAccuracy, Precision, Recall, AUC, MeanSquaredError, MeanAbsoluteError
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, mean_squared_error, mean_absolute_error

from CustomDataGenerator.CustomDataGenerator import create_data_generator

from preprocesing import (
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

def train(dataset_path, train_labels, val_labels,  model, image_shape, output_label, augumentation, epochs, batch_size, directory='trained_models'):

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

def train2(dataset_path, train_labels, val_labels, model, image_shape, output_label, augumentation, epochs, batch_size, directory='trained_models'):
    model = compile_model(model, output_label)    

    training_images_directory = build_training_directory_img(dataset_path)
    train_data_generator = create_data_generator(training_images_directory, train_labels, batch_size, image_shape, output_label, augumentation)
    val_data_generator = create_data_generator(training_images_directory, val_labels, batch_size, image_shape, output_label, augumentation)

    train_losses = []
    train_accuracies = []
    train_precisions = []
    train_recalls = []
    train_aucs = []
    train_mses = []
    train_maes = []

    val_losses = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_aucs = []
    val_mses = []
    val_maes = []

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        history = model.fit(
            train_data_generator,
            epochs=1,  # Train for one epoch
            verbose=1,  
            steps_per_epoch=len(train_data_generator),
            validation_data=val_data_generator,
            validation_steps=len(val_data_generator)
        )

        # Extracting training metrics
        train_losses.append(history.history['loss'][0])
        train_accuracies.append(history.history['accuracy'][0])
        train_precisions.append(history.history['precision'][0])
        train_recalls.append(history.history['recall'][0])
        train_aucs.append(history.history['auc'][0])
        train_mses.append(history.history['mse'][0])
        train_maes.append(history.history['mae'][0])

        # Evaluate model on validation data
        val_loss, val_accuracy, val_precision, val_recall, val_auc, val_mse, val_mae = model.evaluate(val_data_generator, verbose=0)

        # Store validation metrics
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_aucs.append(val_auc)
        val_mses.append(val_mse)
        val_maes.append(val_mae)

    # Get the current date
    current_date = datetime.now().strftime("%m-%d_%H-%M")
    
    # Save the compiled model and trained.
    model.trainable = False
    if not os.path.exists(directory):
        os.makedirs(directory)
    model.save(os.path.join(directory, f'{current_date}_CNN_model_{output_label}_epochs{epochs}.h5'))

    # Return training and validation metrics
    return {
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'train_precision': train_precisions,
        'train_recall': train_recalls,
        'train_auc': train_aucs,
        'train_mse': train_mses,
        'train_mae': train_maes,
        'val_loss': val_losses,
        'val_accuracy': val_accuracies,
        'val_precision': val_precisions,
        'val_recall': val_recalls,
        'val_auc': val_aucs,
        'val_mse': val_mses,
        'val_mae': val_maes
    }


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



def train_test_model(dataset_path, train_labels, val_labels, model, output_label, augumentation, epochs, image_shape, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, batch_size):
    history = train(dataset_path, train_labels, val_labels, model, image_shape, output_label, augumentation, epochs, batch_size)
    predicted_values, y_true = test_cnn_model(dataset_path, model, image_shape, output_label, DATA_SPLIT_TO_EVALUATE_FLAG, evaluate_df, epochs)
    return history, predicted_values, y_true
