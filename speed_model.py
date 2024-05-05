import pandas as pd
from keras.models import load_model
from preprocesing import (
    get_dataset_path, 
    preprocess_dataset, 
    build_training_validation_and_evaluation_sets, 
    get_project_path
    )
from train import (
    train_test_model
    )

from models.cnn_model import (
    create_cnn_model_v4
)

from evaluation_metrics import (
    print_plot_regression_metrics,
    plot_metrics
)

class SpeedModel():
    def __init__(self, data_hyperparameters = (128, (120,120), (2, 2), True ), 
                      training_hyperparameters = (0.01, 2 , 0.1, 0.2), 
                      setting = (False , True, True) ): 
        # DATA HYPERPARAMETERS
        self.batch_size = data_hyperparameters[0]    
        self.image_shape = data_hyperparameters[1]    
        self.pool_size = data_hyperparameters[2]    
        self.augumentation = data_hyperparameters[3]    
        
        # TRAINING HYPERPARAMETERS 
        self.learning_rate = training_hyperparameters[0]  # Specify your desired learning rate~ 
        self.epochs_speed = training_hyperparameters[1]
        self.eval_split = training_hyperparameters[2]    # Test_set %
        self.train_val_split = training_hyperparameters[3] # [training and_validation_set %]
        
        # Setting
        self.logging = setting[0]                    # TRUE = the training process might log various metrics (such as loss and accuracy) for visualization and analysis using TensorBoard.
        self.FIRST_TRAIN_FLAG = setting[1]           #  TRUE = first train of the model. FALSE = continue training a model.
        self.DATA_SPLIT_TO_EVALUATE_FLAG = setting[2] #  TRUE = split the dataset for evaluation.

        self.image_paths = []   # Empty list with image paths
        self.dataset_path = ""   # Empty str fro dataset_path
        self.data_labels = pd.DataFrame() # Empty DataFrame for all labels
        self.train_labels = pd.DataFrame() # Empty DataFrame for labels used for training
        self.val_labels = pd.DataFrame() # Empty DataFrame for labels used for validation
        self.evaluate_df = [None, None]


    def print_split_data(self, X_train, X_val, y_train, y_val):
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

    def update_dataset(self):
        self.dataset_path = get_dataset_path()

        image_paths, data_labels = preprocess_dataset(self.dataset_path)
        if self.DATA_SPLIT_TO_EVALUATE_FLAG:
            TrainVal_set_path, Test_set_path, TrainVal_labels, Test_labels = build_training_validation_and_evaluation_sets(
                                                                            image_paths, data_labels,  self.eval_split)
                                                                                                                      
            print("\n\nFULL DATASET: Print data in Training data and Test data.\n")                                                                                                          
            self.print_split_data(TrainVal_set_path, Test_set_path, TrainVal_labels, Test_labels )
    
            self.evaluate_df = [Test_set_path, Test_labels]
    
        else:
            TrainVal_set_path = image_paths
            TrainVal_labels = data_labels
            self.evaluate_df = [None, None]
        
        X_train, X_val, self.train_labels, val_labels = build_training_validation_and_evaluation_sets(
                                                        TrainVal_set_path, TrainVal_labels, self.train_val_split)

        print("\n\n TRAINING DATASET: Print Data in Training data and Validation data.\n") 
        self.print_split_data(X_train, X_val, self.train_labels, self.val_labels)

    def speed_model_update(self, model):
        self.update_dataset() 
        
        model_speed, _ = create_cnn_model_v4(self.image_shape, self.pool_size)

        if self.FIRST_TRAIN_FLAG:
            history_speed, predicted_speed, y_true_speed = train_test_model(self.dataset_path, self.train_labels, self.val_labels, 
                                                                        model_speed, "speed", self.augumentation, self.epochs_speed, 
                                                                        self.image_shape, self.DATA_SPLIT_TO_EVALUATE_FLAG, self.evaluate_df, self.batch_size)
        else: 
            project_path = get_project_path()
            model_path_speed = f'{project_path}/trained_models/{model}'  # Update with speed model path
            model = load_model(model_path_speed)
            # Unfreeze all layers for training
            model.trainable = True
            history_speed, predicted_speed, y_true_speed = train_test_model(self.dataset_path, self.train_labels, self.val_labels, 
                                                                        model_speed, "speed", self.augumentation, self.epochs_speed, 
                                                                        self.image_shape, self.DATA_SPLIT_TO_EVALUATE_FLAG, self.evaluate_df, self.batch_size)

            if self.DATA_SPLIT_TO_EVALUATE_FLAG:
                print_plot_classification_metrics(y_true_speed, predicted_speed)
            plot_metrics(history_speed, "speed", self.epochs_speed)

    

def run_speed_model(data_hyperparameters, training_hyperparameters, setting, model):
    speed_model = SpeedModel(data_hyperparameters, training_hyperparameters, setting)
    speed_model.speed_model_update(model)


# Call the training function
if __name__ == '__main__':

    # DATA HYPERPARAMETERS
    data_hyperparameters = (
        128,                    # batch_size
        (120, 120),             # image_shape
        (2, 2),                 # pool_size
        True                    # augmentation
    )

    # TRAINING HYPERPARAMETERS 
    training_hyperparameters = (
        0.0000001,  # learning_rate
        3,          # epochs_speed
        0.1,        # eval_split
        0.2         # train_val_split
    )
    
    # Setting
    setting = (
        False,                      # logging: Set to True if the training process might log various metrics for visualization and analysis using TensorBoard.
        False,                      # FIRST_TRAIN_FLAG: Set to True if it is the first train of the model, False if you want to continue training a model that has been trained in the past.
        False                       # DATA_SPLIT_TO_EVALUATE_FLAG: Set to True if you want to split the data for evaluation.
        )

    model =  '03-24_15-47_CNN_model_speed_epochs1050.h5'

    speed_model = SpeedModel(data_hyperparameters, training_hyperparameters, setting)
    speed_model.speed_model_update(model)


