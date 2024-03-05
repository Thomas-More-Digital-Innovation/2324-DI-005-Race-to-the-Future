#!/usr/bin/env python3
import datetime
import functools
import grpc
import logging
import numpy
from adafruit_servokit import ServoKit
from recorder.recorder import Recorder
from recorder.proto import recorder_pb2, recorder_pb2_grpc


SERVER_ADDRESS = '192.168.10.119:50051'


def callback(latest_timestamp, latest_steering_input, latest_wheel_input, latest_image: numpy.ndarray, stub: recorder_pb2_grpc.RecorderServiceStub):
    if latest_image is None:
        return

    datetime_timestamp = datetime.datetime.fromtimestamp(latest_timestamp)

    datapoint = recorder_pb2.SendDataPointRequest(
        timestamp=None, steering_angle=latest_steering_input, wheel_speed=latest_wheel_input, image=latest_image.tobytes())

    # put this outside of the constructor, else the timestamp will not be set in the message
    datapoint.timestamp.FromDatetime(dt=datetime_timestamp)

    stub.SendDataPoint(datapoint)


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

    servos.servo[0].angle = wheel_angle
    servos.servo[1].angle = 180 - steering_angle


def main():
    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    channel = grpc.insecure_channel(SERVER_ADDRESS)
    stub = recorder_pb2_grpc.RecorderServiceStub(channel)

    kit = ServoKit(channels=16)

    recorder = Recorder('/dev/input/event2', 'nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, framerate=(fraction)60/1 ! nvvidconv flip-method= ! video/x-raw, width=(int)1280, height=(int)720, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink',
                        functools.partial(callback, stub=stub), functools.partial(callback_controls, servos=kit))

    try:
        recorder.start()
        recorder.run_indefinitely()
    except KeyboardInterrupt:
        recorder.stop()


if __name__ == "__main__":
    main()
