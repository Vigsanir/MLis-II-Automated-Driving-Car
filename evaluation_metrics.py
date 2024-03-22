# Standard library imports
import os
from datetime import datetime
import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score

def save_figure_and_data(figure, data, filename):
    current_date = datetime.now().strftime("%m-%d_%H-%M")
    directory = "evaluation_metrics"
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    # Save figure as PNG
    if figure is not None:
        figure.savefig(os.path.join(directory, f"{current_date}_{filename}.png"))
    
    # Save data as text file
    if isinstance(data, dict):  # Check if data is a dictionary
        with open(os.path.join(directory, f"{current_date}_{filename}.txt"), "w") as file:
            for key, value in data.items():
                file.write(f"{key}: {value}\n")
    else:
        np.savetxt(os.path.join(directory, f"{current_date}_{filename}.txt"), data)

def calculate_classification_metrics(y_true, y_pred, threshold=0.5):
    # Convert y_pred to binary predictions
    y_pred_binary = (y_pred['speed'] >= threshold).astype(int)
    
    # Calculate classification metrics
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    accuracy = (y_true == y_pred_binary).mean()

    return precision, recall, f1,accuracy

def print_classification_metrics(y_true, y_pred):

    precision, recall, f1 , accuracy= calculate_classification_metrics(y_true, y_pred)
    
    print("Classification Metrics:")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Save metrics
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    ##save_figure_and_data(None, metrics, "classification_metrics")

def plot_classification_metrics(y_true, y_pred):
    # Convert y_pred to binary predictions
    threshold = 0.5  # Adjust as needed
    y_pred_binary = (y_pred['speed'] >= threshold).astype(int)
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred_binary)
    
    # Compute additional classification metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Confusion Matrix
    axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')
    tick_marks = np.arange(len(set(y_true)))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(tick_marks)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(tick_marks)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        axes[0].text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')

    # Colorbar
    im = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(im, ax=axes[0])

    # Add text annotations for additional classification metrics
    metrics_text = f"\n\n\n\n\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}"
    axes[0].text(0.5, -0.1, metrics_text, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)

    plt.tight_layout()
    plt.show()

    # Save plot
    save_figure_and_data(fig, cm, "confusion_matrix")

    # Close the current figure
    plt.close()


def print_regression_metrics(y_true, y_pred):
    # Extract the 'angle' column from the DataFrame y_pred
    y_pred_angle = y_pred['angle']
    
    # Convert y_true and y_pred_angle to NumPy arrays
    y_true = y_true.values
    y_pred_angle = y_pred_angle.values
    
    # Compute regression metrics
    mae = mean_absolute_error(y_true, y_pred_angle)
    mse = mean_squared_error(y_true, y_pred_angle)
    rmse = mean_squared_error(y_true, y_pred_angle, squared=False)
    r2 = r2_score(y_true, y_pred_angle)

    print("Regression Metrics:")
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("R^2 Score:", r2)

    # Save metrics
    metrics = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R^2 Score": r2
    }
       # Generate the plot
    fig = plot_regression_metrics(y_true, y_pred, metrics)
    
    # Save the figure and metrics
    save_figure_and_data(fig, metrics, "regression_metrics")

    # Return metrics dictionary
    return metrics
    
 


def plot_regression_metrics(y_true, y_pred, metrics):
    # Extract the 'angle' column from the DataFrame y_pred
    y_pred_angle = y_pred['angle']
    
    # Plot scatter plot of Predicted vs Real Values
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot
    axes[0].scatter(y_true, y_pred_angle)
    axes[0].set_title('Scatter Plot of Predicted vs Real Values')
    axes[0].set_xlabel('Real Values')
    axes[0].set_ylabel('Predicted Values')

    # Calculate residuals
    residual = y_true - y_pred_angle

    # Residual Plot
    axes[1].scatter(y_pred_angle, residual)
    axes[1].set_title('Residual Plot')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')


    # Extract individual metrics
    mae_value = metrics["MAE"]
    mse_value = metrics["MSE"]
    rmse_value = metrics["RMSE"]
    r2_score_value = metrics["R^2 Score"]
    metrics_text = f"\n\n\n\n\nMAE: {mae_value}\nMSE: {mse_value}\nRMSE: {rmse_value}\nR^2 Score: {r2_score_value}"
    axes[0].text(0.5, -0.1, metrics_text, horizontalalignment='center', verticalalignment='center', transform=axes[0].transAxes)
    plt.tight_layout()
    plt.show()

    # Save plot
    save_figure_and_data(fig, (y_true, y_pred_angle, residual), "regression_plots")
  
    # Close the current figure
    plt.close()

    return fig


def print_plot_classification_metrics(y_true_cls, y_pred_cls):
    print_classification_metrics(y_true_cls, y_pred_cls)
    plot_classification_metrics(y_true_cls, y_pred_cls)

def print_plot_regression_metrics(y_true_reg, y_pred_reg):
    print_regression_metrics(y_true_reg, y_pred_reg)
    

TEST_FLAG = False
if TEST_FLAG:

    # Example classification metrics
    y_true_cls = [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    y_pred_cls = pd.DataFrame({'speed': [1, 0.999798, 0.986208, 0.989418, 0.967128, 
                                  0.780634, 0.940513, 1, 0.936071, 0]})
    #y_true_cls = [0, 1, 1, 0, 1]
    #y_pred_cls = [0, 1, 1, 1, 0]

    print_plot_classification_metrics(y_true_cls, y_pred_cls)

    # Example regression metrics
    y_true_reg = [0.6, 0.9, 1.3, 1.1, 0.8]
    y_pred_reg = [0.5, 0.8, 1.2, 1.0, 0.7]

    print_plot_regression_metrics(y_true_reg, y_pred_reg)
