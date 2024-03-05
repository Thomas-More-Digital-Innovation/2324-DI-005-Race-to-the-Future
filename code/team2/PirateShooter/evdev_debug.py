#!/usr/bin/env python3
import evdev

# devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
# for device in devices:
#     print(device.path, device.name, device.phys)

device = evdev.InputDevice('/dev/input/event2')
print(device)

print(device.capabilities(verbose=True))

for event in device.read_loop():
    if event.type == evdev.ecodes.SYN_REPORT:
        continue

    # if event.code == evdev.ecodes.ABS_RY:
    #     print(event.value / 255)

    # if event.code == evdev.ecodes.ABS_RX:
    #     print((1 - (event.value / 255)) - 1)

    if event.code == evdev.ecodes.ABS_X:
        print(evdev.categorize(event))
        print((event.value / 255) * 2 - 1)

    pass
