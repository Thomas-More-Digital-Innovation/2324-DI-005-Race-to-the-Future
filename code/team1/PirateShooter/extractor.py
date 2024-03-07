import os
import pickle
import cv2
import platform
import numpy as np

from recorder.model.datapoint import DataPoint

# When the os is windows, the path is absolute, linux uses a relative path
# Check the operating system
is_windows = platform.system() == "Windows"

# Define the base directory
base_dir = "./datapoints" if not is_windows else r"C:\_repos_\2324-DI-005-Race-to-the-Future\code\team1\PirateShooter\datapoints"

# Loop through each file in the directory
for file in os.listdir(base_dir):
    if file.endswith(".pickle"):
        with open(os.path.join(base_dir, file), "rb") as pickle_file:
            try:
                # Load the list of DataPoint objects from the pickle file
                datapoints = pickle.load(pickle_file)
                
                # Iterate over each DataPoint in the list
                for datapoint in datapoints:
                    # Process the DataPoint object
                    if isinstance(datapoint, DataPoint):
                        # Decode bytes to image
                        nparr = np.frombuffer(datapoint.image, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        # Write the image to disk
                        cv2.imwrite(f"img_{datapoint.timestamp}.png", img)
                    else:
                        print(f"Invalid datapoint in {file}: {type(datapoint)}")
            except Exception as e:
                print(f"Error loading pickle {file}: {e}")
