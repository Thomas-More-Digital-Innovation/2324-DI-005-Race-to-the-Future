# 2324-DI-005-RACE-TO-THE-FUTURE

## Setup PI Car:

### Requirements:
- Raspberry Pi 4
- SD card
- PI camera
- Playstation 4 controller

1. Check if the battery of the car is charged.
2. Turn the car motor on using the "hidden" switch.
3. Turn the PI on using the Power module button.

### WiFi Connection:

- The PI should automatically connect to the WIFI: **Dino Wireless**.
- Password: **D1n0W1f1@**.
- If not, connect it to a screen and try to change the wifi settings.

### SSH Connection:

- SSH to the PI using the following command:
    ```bash
    ssh rttf-2
    ```
- Password for SSH: **Dino1234**.
  ```bash
  Host rrtf-2
    HostName rttf-2.local
    User rttf-2
  ```

## Setup Nano Car: