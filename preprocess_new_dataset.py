
import os
import csv
from PIL import Image
from preprocesing import get_project_path  

# Get the project path
project_path = get_project_path()

directory = f"{project_path}\\machine-learning-in-science-ii-2024\\training_dataB"
print(directory)

# Get a list of PNG files in the directory
png_files = [file for file in os.listdir(directory) if file.endswith(".png")]

# Sort the list of PNG files
png_files.sort()

# Initialize a counter for image_id
image_id = 1

# Create a CSV file to store the extracted information
with open("training_norm.csv", "w", newline="") as csvfile:
    fieldnames = ['image_id', 'angle', 'speed']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # Process each PNG file
    for filename in png_files:
        # Extract angle and speed from the filename
        angle, speed = filename.split("_")[1:3]
        angle = int(angle)
        speed = int(speed.split(".")[0])
        
        angle_norm = (angle - 50) / 80
        if speed >=35 : speed = 35
        
        speed_norm = (speed - 0) / 35


        # Write the extracted information to the CSV file
        writer.writerow({'image_id': image_id, 'angle': angle_norm, 'speed': speed_norm})

        # Rename the PNG file with the image_id
        os.rename(os.path.join(directory, filename), os.path.join(directory, f"{image_id}.png"))

        # Increment the image_id
        image_id += 1