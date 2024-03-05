from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SendDataPointRequest(_message.Message):
    __slots__ = ["image", "steering_angle", "timestamp", "wheel_speed"]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    STEERING_ANGLE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    WHEEL_SPEED_FIELD_NUMBER: _ClassVar[int]
    image: bytes
    steering_angle: float
    timestamp: _timestamp_pb2.Timestamp
    wheel_speed: float
    def __init__(self, timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., steering_angle: _Optional[float] = ..., wheel_speed: _Optional[float] = ..., image: _Optional[bytes] = ...) -> None: ...

class SendDataPointResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
