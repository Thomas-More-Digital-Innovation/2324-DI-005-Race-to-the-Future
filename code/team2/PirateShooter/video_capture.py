#!/usr/bin/env python3
import functools
import logging
from typing import Any
from adafruit_servokit import ServoKit
from recorder.recorder import Recorder
import cv2
from cv2 import VideoWriter_fourcc

class VideoRecorder(Recorder):
    def __init__(self, device: str, pipeline: str, callback, control_callback, name="output.mp4", width=1280, height=720, fps=60.0):
        super().__init__(device, pipeline, callback, callback_controls)
        self._name = name
        self._cap = None  # Initialize capture object later
        self._fourcc = VideoWriter_fourcc(*'MP4V')
        self._out = None  # Initialize video writer later
        self._width = width
        self._height = height
        self._fps = fps
        self.control_callback = callback_controls  # Function to control servos

    def start(self):
        super().start()
        # Initialize capture object and video writer within start
        self._cap = cv2.VideoCapture(0)  # Assuming camera 0
        self._out = cv2.VideoWriter(self._name, self._fourcc, self._fps, (self._width, self._height))

    def stop(self):
        super().stop()
        # Release resources when stopping
        if self._cap:
            self._cap.release()
        if self._out:
            self._out.release()

    def process_image(self, image):
        # Write the image frame to the video
        self._out.write(image)
        # Call control callback function with processed image data
        self.control_callback(latest_steering_input=self.latest_steering_input, latest_wheel_input=self.latest_wheel_input, servos=self.recorder.kit)

def callback(latest_timestamp: float, latest_steering_input: float, latest_wheel_input: float, latest_image: Any):
    # You can process other data from the recorder here (optional)
    pass

def callback_controls(latest_steering_input: float, latest_wheel_input: float, servos: ServoKit) -> None:
    # Same logic as before for controlling servos
    steering_angle = (latest_steering_input + 1) * 90
    wheel_angle = (latest_wheel_input + 1) * 90

    if steering_angle > 160:
        steering_angle = 160

    if steering_angle < 20:
        steering_angle = 20

    if wheel_angle > 180:
        wheel_angle = 180

    if wheel_angle < 0:
        wheel_angle = 0

    # Speed limiter
    # 110 = dead slow 120 = slow 135 = normal 180 = no limit - not recommended > see pictures
    if wheel_angle > 120:
        wheel_angle = 120

    if wheel_angle < 45:
        wheel_angle = 45

    servos.servo[0].angle = wheel_angle
    servos.servo[1].angle = 180 - steering_angle

def main():
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

    kit = ServoKit(channels=16)
    
    recorder = VideoRecorder('/dev/input/event2', 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)60/1 ! nvvidconv flip-method=2 ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink',
                            callback, functools.partial(callback_controls, servos=kit))

    try:
        recorder.start()
        recorder.run_indefinitely()
    except KeyboardInterrupt:
        recorder.stop()

    logging.info("recorder stopped")


if __name__ == "__main__":
    main()
