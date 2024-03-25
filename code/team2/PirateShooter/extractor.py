#!/usr/bin/env python3
import os
import pickle
import cv2
import platform

from recorder.model.datapoint import DataPoint

# When the os is windows, the path is absolute, linux uses a relative path
# Check the operating system
is_windows = platform.system() == "Windows"

# Define the base directory
base_dir = "./datapoints" if not is_windows else r"C:\Users\bryan\Desktop\TM2023-2024\DI\2324-DI-005-Race-to-the-Future\code\team2\PirateShooter\datapoints"

# Loop through each file in the directory
for file in os.listdir(base_dir):
    if file.endswith(".pickle"):
        with open(os.path.join(base_dir, file), "rb") as pickle_file:
            try:
                # Load the list of DataPoint objects from the pickle file
                datapoints = pickle.load(pickle_file)
                
                # Iterate over each DataPoint in the list
                for datapoint in datapoints:
                    print(type(datapoint.image))
                    # Process the DataPoint object
                    if isinstance(datapoint, DataPoint):
                        # Write the image to disk (assuming datapoint.image is a valid image)
                        cv2.imwrite(f"./extracted/img_{datapoint.timestamp}.png", datapoint.image)
                    else:
                        print(f"Invalid datapoint in {file}: {type(datapoint)}")
            except Exception as e:
                print(f"Error loading pickle {file}: {e}")
    