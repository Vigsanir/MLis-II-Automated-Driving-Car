# train_conv_CNN.py
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.debug_utils import debug_log
from readData import construct_paths, load_training_data, load_test_images
from models.cnn_model import create_cnn_model, create_cnn_model_v2
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error



def load_and_preprocess_data():
    # Return the path for training and testing data.
    training_data_dir, training_labels_path, _, _ = construct_paths()
    
    # Load the CSV file into a Pandas DataFrame
    labels_df = pd.read_csv(training_labels_path)
    
    # Print the head of the DataFrame in debug mode
    debug_log("Head of the DataFrame:", data=labels_df.head())
   
    # Load training data.
    training_images, labels = load_training_data(training_data_dir, training_labels_path, (200, 200))
    
    return training_images, labels


def test_cnn_model(model):
    _, _, test_data_dir, _ = construct_paths()
    # Load test data
    test_images, image_ids = load_test_images(test_data_dir, (200, 200))

    # Make predictions using the trained model
    predictions = model.predict(test_images)

    # Create a DataFrame with image IDs and predictions
    results_df = pd.DataFrame({'ImageId': image_ids, 'Angle': predictions[:, 0], 'Speed': predictions[:, 1]})

     # Save the results DataFrame to the new submission file
    results_df.to_csv('new_submission.csv', index=False)


def train_cnn_model(X_train, y_train, epochs=100, batch_size=128, pool_size=(2, 2)):
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    
    input_shape = X_train.shape[1:]
    
    #model = create_cnn_model(input_shape, pool_size)
    model = create_cnn_model_v2(input_shape, pool_size)
    
    datagen = ImageDataGenerator(channel_shift_range=0.2)
    datagen.fit(X_train)
    
    model.compile(optimizer='Adam', loss='mean_squared_error')
    
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                                  steps_per_epoch=len(X_train)/batch_size,
                                  epochs=epochs,
                                  verbose=1,
                                  validation_data=(X_val, y_val))
    
    model.trainable = False
    model.save('full_CNN_model.h5')
    model.summary()
    
    return model, history

def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

if __name__ == "__main__":
    training_images, labels = load_and_preprocess_data()
    model, training_history = train_cnn_model(training_images, labels)
    test_cnn_model(model)
    plot_loss(training_history)
