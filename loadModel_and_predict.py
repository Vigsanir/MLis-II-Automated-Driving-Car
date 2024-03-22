
import pandas as pd
from keras.models import load_model


from preprocesing import get_dataset_path,load_test_images



def test_cnn_model(model_path, dataset_path, target_size, output_label):
    # Load the trained model
    model = load_model(model_path)

    # Load test data
    test_images, image_ids = load_test_images(dataset_path, target_size)

    # Make predictions using the trained model
    predictions = model.predict(test_images)
   # print(predictions)
    print(f"Prediction {output_label}: {predictions}")
    
    # Flatten the nested lists
    flat_predictions = [item for sublist in predictions for item in sublist]
   # print(flat_predictions)
    # Create a DataFrame with image IDs and predictions
    results_df = pd.DataFrame({'image_id': image_ids, f'{output_label}': flat_predictions})
    print(results_df)

    # Save the results DataFrame to the new submission file
    results_df.to_csv(f'Test_submission_{output_label}_generate.csv', index=False)

# Example usage
dataset_path = get_dataset_path()
model_path = '03-21_21-48_CNN_model_speed_epochs150.h5'  # Update with your model path
dataset_path = get_dataset_path()  # Update with your test dataset directory
target_size = (120,120)  # Update with your desired target size
output_label = 'speed'  # Update with your desired output label

test_cnn_model(model_path, dataset_path, target_size, output_label)