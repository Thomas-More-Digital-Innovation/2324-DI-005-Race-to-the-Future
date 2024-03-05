#!/usr/bin/env python3
import datetime
from io import BytesIO
import logging
import pickle
import threading
import time
from typing import List
import grpc
from concurrent import futures
import numpy
import cv2

from recorder.proto import recorder_pb2, recorder_pb2_grpc
from recorder.model.datapoint import DataPoint

OUTPUT_FILE = "datapoints/datapoints_{timestamp}_{slice}.pickle"

dump = True
datapoints: List[DataPoint] = []
datapoint_lock = threading.Lock()


class RecorderServicer(recorder_pb2_grpc.RecorderService):
    global dump
    global datapoints
    global datapoint_lock

    def __init__(self) -> None:
        super().__init__()

    def SendDataPoint(self, request, context):
        assert isinstance(request, recorder_pb2.SendDataPointRequest)

        print(f"got datapoint for timestamp: {request.timestamp.ToDatetime()}")

        with datapoint_lock:
            datapoints.append(DataPoint(request.timestamp.ToDatetime().timestamp(
            ), request.steering_angle, request.wheel_speed, request.image))

        return recorder_pb2.SendDataPointResponse()


def data_dump_loop():
    global dump
    global datapoints
    global datapoint_lock

    slice_number = 0

    while dump:
        if len(datapoints) > 30:
            with datapoint_lock:
                with open(OUTPUT_FILE.format(slice=slice_number, timestamp=int(datetime.datetime.now().timestamp())), 'wb') as pickle_file:
                    pickler = pickle.Pickler(
                        pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
                    pickler.dump(datapoints)
                    datapoints = []
                    slice_number += 1

        time.sleep(.1)


def main():
    global dump

    logging.basicConfig(format='%(levelname)s:%(message)s',
                        level=logging.DEBUG)

    data_dump_thread = threading.Thread(target=data_dump_loop)
    data_dump_thread.start()

    try:
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))

        recorder_pb2_grpc.add_RecorderServiceServicer_to_server(
            RecorderServicer(), server)

        server.add_insecure_port("[::]:50051")
        server.start()

        logging.info("started grpc server on [::]:50051")

        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(5)

        dump = False

        data_dump_thread.join()

        logging.info("grpc server stopped")


if __name__ == "__main__":
    main()
