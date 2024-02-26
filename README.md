# 2324-DI-005-RACE-TO-THE-FUTURE

Setup PI car:
Prequirements:
- Raspberry Pi 4
- SD card
- PI camera
- Playstation 4 controller

Check if the battery of the car is charged
Turn the car motor on using the "hidden" switch
Turn the PI on using the Power module button

The PI should automaticly connect to the WIFI: Dino IoT
Password is D1n0W1f1@
If not you should connect it to a screen and try to change the wifi settings.

SSH to the PI using the following command:
Password for ssh is rrtf-2 
```bash
Host rrtf-2
  HostName rttf-2.local
  User rttf-2
```

You shouldn't have to run any commands on the SSH in order for it to work with the playstation 4 controller.

The connection with the playstation 4 controller should be automaticly established once the controller is turned on.

Setup Nano car:
