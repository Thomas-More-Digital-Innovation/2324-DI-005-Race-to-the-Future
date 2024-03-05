import logging
import threading
from typing import Any, Callable, List
import evdev
import cv2


class Recorder:
    logger: logging.Logger

    device: evdev.InputDevice
    camera: cv2.VideoCapture
    callback: Callable[[float, float, float, Any], None]
    callback_controls: Callable[[float, float], None]

    record: bool

    input_lock: threading.Lock
    latest_timestamp: float
    latest_wheel_input: float
    latest_steering_input: float

    skip_input_count: int

    image_lock: threading.Lock
    latest_image: Any

    camera_thread: threading.Thread
    input_thread: threading.Thread

    def __init__(self, device_path: str, camera_string: str, callback: Callable[[float, float, float, Any], None], callback_controls: Callable[[float, float], None]) -> None:
        self.logger = logging.getLogger(__name__)

        # do not start recording on initialise
        # we have the start and stop methods for this
        self.record = False

        self.skip_input_count = 0

        self.input_lock = threading.Lock()
        self.image_lock = threading.Lock()

        self.latest_image = None
        self.latest_timestamp = 0.0
        self.latest_steering_input = 0.0
        self.latest_wheel_input = 0.0

        self.device = evdev.InputDevice(device_path)
        self.camera = cv2.VideoCapture(camera_string, cv2.CAP_GSTREAMER)
        self.callback = callback
        self.callback_controls = callback_controls

        if not self.camera.isOpened():
            self.logger.critical("camera could not be opened")
            exit(1)

        self.input_thread = threading.Thread(target=self._device_input_loop)
        self.camera_thread = threading.Thread(target=self._camera_input_loop)

        self.logger.info("recorder has been initialised")

    def _device_input_loop(self) -> None:
        for event in self.device.read_loop():
            if not self.record:
                return

            if event.type == 0:
                continue

            if event.code == evdev.ecodes.BTN_SOUTH:
                self.logger.info("emergency stop")
                continue

            with self.input_lock:
                if event.type == evdev.ecodes.SYN_REPORT:
                    continue

                if event.code in [evdev.ecodes.ABS_RZ, evdev.ecodes.ABS_Z, evdev.ecodes.ABS_X]:
                    self.latest_timestamp = event.timestamp()
                    # self.latest_steering_input = 0.0
                    # self.latest_wheel_input = 0.0

                if event.code == evdev.ecodes.ABS_RY:
                    self.latest_wheel_input = event.value / 255

                if event.code == evdev.ecodes.ABS_RX:
                    self.latest_wheel_input = (1 - (event.value / 255)) - 1

                if event.code == evdev.ecodes.ABS_X:
                    self.latest_steering_input = (event.value / 255) * 2 - 1

                self.skip_input_count += 1

                if self.skip_input_count > 30:
                    self.callback_controls(
                        self.latest_steering_input, self.latest_wheel_input)
                    self.skip_input_count = 0

            # Uncomment to handle every input ever sent to the recorder
            # self._handle_input()

    def _camera_input_loop(self) -> None:
        while self.record:
            ret, frame = self.camera.read()

            if not ret:
                self.logger.warning("could not read camera frame")
                continue

            with self.image_lock:
                self.latest_image = frame

            self._handle_input()

    def _handle_input(self) -> None:
        with self.input_lock and self.image_lock:
            self.callback(self.latest_timestamp, self.latest_steering_input,
                          self.latest_wheel_input, self.latest_image)

    def start(self) -> None:
        self.record = True
        self.input_thread.start()
        self.camera_thread.start()

    def run_indefinitely(self) -> None:
        self.input_thread.join()
        self.camera_thread.join()

    def stop(self) -> None:
        self.record = False
