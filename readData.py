from utils.debug_utils import debug_log  
import os
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))



def construct_paths():
    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Define relative paths
    training_data_relative_path = "machine-learning-in-science-ii-2024/training_data/training_data"
    training_labels_relative_path = "machine-learning-in-science-ii-2024/training_norm.csv"
    test_data_relative_path = "machine-learning-in-science-ii-2024/test_data/test_data"
    submission_csv_relative_path = "machine-learning-in-science-ii-2024/sampleSubmission.csv"

    # Construct absolute paths
    training_data_dir = os.path.join(script_dir, training_data_relative_path)
    training_labels_path = os.path.join(script_dir, training_labels_relative_path)
    test_data_dir = os.path.join(script_dir, test_data_relative_path)
    submission_csv_path = os.path.join(script_dir, submission_csv_relative_path)

    # Print paths in debug mode
    debug_log("This is a debug message.", data=script_dir)
    debug_log(f"Training Data Directory: {training_data_dir}")
    debug_log(f"Training Labels Path: {training_labels_path}")
    debug_log(f"Test Data Directory: {test_data_dir}")
    debug_log(f"Submission CSV Path: {submission_csv_path}")

    # Return the constructed paths
    return training_data_dir, training_labels_path, test_data_dir, submission_csv_path


# Example usage
training_data_dir, training_labels_path, test_data_dir, submission_csv_path = construct_paths()
