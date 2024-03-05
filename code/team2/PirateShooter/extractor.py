#!/usr/bin/env python3
import os
import pickle
import cv2

from recorder.model.datapoint import DataPoint

files = os.listdir("./datapoints")

for file in files:
    path_parts = file.split(".")

    if path_parts[len(path_parts) - 1] != "pickle":
        continue

    # print(f"Extracting pickle: {file}")

    with open(f"./datapoints/{file}", "rb") as pickle_file:
        datapoint: DataPoint = pickle.load(pickle_file)

        if type(datapoint) is not DataPoint:
            print(f"Invalid pickle: {file}")
            continue

        cv2.imwrite(
            f"./extracted/img_{datapoint.timestamp}.png", datapoint.image)
