from dataclasses import dataclass
from typing import Any


@dataclass
class DataPoint:
    """Class for storing a point of data in time."""
    timestamp: float
    steering_angle: float
    wheel_speed: float
    image: Any
