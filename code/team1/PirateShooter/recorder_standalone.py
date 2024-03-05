#!/usr/bin/env python3
import functools
import logging
import os
import pickle
from typing import Any
from adafruit_servokit import ServoKit
from recorder.recorder import Recorder
from recorder.model.datapoint import DataPoint

OUTPUT_FILE = "datapoints/datapoint_{id}.pickle"


def callback(latest_timestamp: float, latest_steering_input: float, latest_wheel_input: float, latest_image: Any):
    # print(latest_timestamp, latest_steering_input, latest_wheel_input)

    datapoint = DataPoint(
        latest_timestamp, latest_steering_input, latest_wheel_input, latest_image)

    with open(OUTPUT_FILE.format(id=str(latest_timestamp)), 'wb') as pickle_file:
        pickler = pickle.Pickler(
            pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
        pickler.dump(datapoint)


def callback_controls(latest_steering_input: float, latest_wheel_input: float, servos: ServoKit) -> None:
    steering_angle = (latest_steering_input + 1) * 90
    wheel_angle = (latest_wheel_input + 1) * 90

    if steering_angle > 180:
        steering_angle = 180

    if steering_angle < 0:
        steering_angle = 0

    if wheel_angle > 180:
        wheel_angle = 180

    if wheel_angle < 0:
        wheel_angle = 0

    # Speed limiter
    if wheel_angle > 135:
        wheel_angle = 135

    servos.servo[0].angle = wheel_angle
    servos.servo[1].angle = 180 - steering_angle


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    kit = ServoKit(channels=16)

    recorder = Recorder('/dev/input/event2', 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink',
                        callback, functools.partial(callback_controls, servos=kit))

    try:
        recorder.start()
        recorder.run_indefinitely()
    except KeyboardInterrupt:
        recorder.stop()

        logging.info("recorder stopped")


if __name__ == "__main__":
    main()
