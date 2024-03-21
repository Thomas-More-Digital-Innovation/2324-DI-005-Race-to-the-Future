## Team 2:
Toon
Roel
Tom
Wesley
Bryan

## Car specs:
- Jetson Nano 2GB
- Pi 2 Camera board
- WIFI dongle
- SD card (At least 32 GB)
- Waveshare 8 MP IMX219-D160 Camera Module Wide Angle 160 graden FoV (https://www.amazon.com.be/-/nl/IMX219-D160-compatibel-inbegrepen-ondersteunt-video-opname/dp/B07H2D4WYR/ref=sr_1_3?dib=eyJ2IjoiMSJ9.gZMAc856bvWWiK1EONAOIyV6B2HkBwJALub-Fibr3nANow2jhbZVJ4-xLL5yRS1JLi1mcJiMg9StEreL1tP2mCSxDviua0dU-kvgYxNSxaDGJCnrXey-J5XrV3yJ3ZbchWYFN96WlQkJG_qnUdmNpKkA6CXrrccBdL3RWj3M2wYpYYI6X9AF5zRG7EFJzZXXf6jbZuKpqs4qFQ-8s2y93hFuJK1TRyUJk-Xr0__LUhGyxsP9QM4-LMBLcTzFCjSJITToGXdL8CG1TlzNgEcI_Cg0twcn70pckqJPYh2il20.7iHZ3s6brMzhx4H6s7jiE83A-ku0_Kr-brB5Sj6jE7c&dib_tag=se&keywords=pi%2Bcamera%2Bwide%2BIMX219&qid=1709814728&s=electronics&sr=1-3&th=1)
- Bluetooth dongle (https://www.amazon.com.be/-/en/Edimax-BT-8500-Bluetooth-Network-Card/dp/B08K1C8B81?source=ps-sl-shoppingads-lpcontext&ref_=fplfs&ref_=fplfs&psc=1&smid=A3Q3FYJVX702M2)

## Setup 2GB Jetson Nano Car:
1. Flash the following image onto your SD card: https://github.com/Qengineering/Jetson-Nano-Ubuntu-20-image
2. Insert the SD card into the Nano
3. Boot the Nano by providing power to it
4. The login credentials are as follows:
    - Username: jetson@nano
    - Password: jetson
5. When the Ubuntu is booted, connect to the internet (need a WIFI dongle)
6. Use a tool like GParted to resize the partition to use the full SD card (sudo apt-get install gparted)
7. Create a swap file to have more memory available
    - You can specify the size of the swap file by changing the value of the variable swapfilesize in the script or using the parameter -s | --size [gigabytes] (defaults to 6)
    ```bash
    git clone https://github.com/JetsonHacksNano/installSwapfile
    cd installSwapfile
    ./installSwapfile.sh
    sudo reboot now 
    ```
8. After you have rebooted it is recommended that you update and upgrade all packages using:
    ```bash
    sudo apt-get update -y
    sudo apt-get upgrade -y
    ```
9. To start using the code in this repo you need to first clone it:
    ```bash
    git clone https://github.com/Thomas-More-Digital-Innovation/2324-DI-005-Race-to-the-Future.git
    ```
10. Go into the PirateShooter folder:
    ```bash
    cd 2324-DI-005-Race-to-the-Future/code/team2/PirateShooter
    ```
11. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```

12. If the install gets stuck or takes a long time on grcpio-tools you need to update your pip version:
    ```bash
    pip install --upgrade pip
    ```
    And then run the pip install -r requirements.txt again
13. Connect the camera to the Jetson Nano board in the CSI slot
14. Connect the PS4 controller by bluetooth
15. Find out to which input the PS4 controller is connected:
    ```bash
    ls /dev/input/
    ```
    Or 
    ```bash
    cat /proc/bus/input/devices
    ```
16. Install evdev if not already installed to debug
    ```bash
    sudo apt-get install python3-evdev
    ```
17. Edit the evdev_debug.py file to the correct input number
    ```bash
    sudo nano evdev_debug.py
    ```
18. Edit the file you are using either driver.py or recorder_client.py or recorder_standalone.py to the correct input number
    ```bash
    sudo nano driver.py
    ```
    Or
    ```bash
    sudo nano recorder_client.py
    ```
    Or
    ```bash
    sudo nano recorder_standalone.py
    ```

## Useful commands:
- Check ram usage:
    ```bash
    free -h
    ```


## Useful links: