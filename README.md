# 2324-DI-005-Race-To-The-Future

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

- The PI should automatically connect to the WIFI: **Dino IoT**.
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

## Setup Nano Car 2GB:
We followed the instructions from the following link:
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devki

### Download the 2GB image for the jetson nano from the following link:
https://developer.nvidia.com/embedded/l4t/r32_release_v5.1/r32_release_v5.1/jeston_nano_2gb/jetson-nano-2gb-jp451-sd-card-image.zip

### Flash the image to the SD card:
1. We used Etcher to flash the image to the SD card.
2. Insert the SD card into the Nano
3. Connect a monitor, keyboard and mouse to the Nano.
4. Power the Nano on and it should start the boot process

### Initial Setup:
1. Select the language and keyboard layout.
2. Connect to the internet.
3. Create a user and password.
4. Update the system.

### SSH Connection:

Login: nanorc2
Password: Dino1234
- SSH to the Nano using the following command:
    ```bash
    ssh nanorc2@nanorc2-desktop
    ```
- Password for SSH: **Dino1234**.

### Follow the donkeycar documentation to update the Nano:
Remove Libre Office:

```bash
sudo apt-get remove --purge libreoffice*
sudo apt-get clean
sudo apt-get autoremove
```

And add a 8GB swap file:

```bash
git clone https://github.com/JetsonHacksNano/installSwapfile
cd installSwapfile
./installSwapfile.sh
sudo reboot now 
```

## We skipped step 2a from the donkeycar documentation and went straight to step 3a.

### Step 2a: Free up the serial port (optional. Only needed if you're using the Robohat MM1)

```bash
sudo usermod -aG dialout <your username>
sudo systemctl disable nvgetty
```

### Step 3a: Install System-Wide Dependencies

First install some packages with `apt-get`.

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install -y libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev liblapack-dev libblas-dev gfortran
sudo apt-get install -y python3-dev python3-pip
sudo apt-get install -y libxslt1-dev libxml2-dev libffi-dev libcurl4-openssl-dev libssl-dev libpng-dev libopenblas-dev
sudo apt-get install -y git nano
sudo apt-get install -y openmpi-doc openmpi-bin libopenmpi-dev libopenblas-dev
```

### Step 4a: Setup Python Environment.

#### Setup Virtual Environment

```bash
pip3 install virtualenv
python3 -m virtualenv -p python3 env --system-site-packages
echo "source ~/env/bin/activate" >> ~/.bashrc
source ~/.bashrc
```

#### Setup Python Dependencies

Next, you will need to install packages with `pip`:

```bash
pip3 install -U pip testresources setuptools
pip3 install -U futures==3.1.1 protobuf==3.12.2 pybind11==2.5.0
pip3 install -U cython==0.29.21 pyserial
pip3 install -U future==0.18.2 mock==4.0.2 h5py==2.10.0 keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.3.3
pip3 install -U absl-py==0.9.0 py-cpuinfo==7.0.0 psutil==5.7.2 portpicker==1.3.1 six requests==2.24.0 astor==0.8.1 termcolor==1.1.0 wrapt==1.12.1 google-pasta==0.2.0
pip3 install -U gdown

# This will install tensorflow as a system package
pip3 install --pre --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v45 tensorflow==2.5
```

### Running driver.py
This seemed quite an issue because the jetson nano only allows python version 3.6 and the packages we used require python 3.8
The solution was using a virtual env with python3.8 installed on it and when running commands use:

"myenv" is the name of the virtual env
```bash
sudo ./myenv/bin/python3.8 ./driver.py
```

If this doesn't work it will probably show you that certain packages are missing which you will have to install using

```bash
pip install
```

You can also see the installed packages with their version using
  
```bash
pip list
```

To exit a virtual env use

```bash
deactivate
```

### Problem with camera detection
The camera is not being detected when running driver.py

We used the following github repo documentation to check the camera and got a result when running direct bash commands from this repo:
https://github.com/JetsonHacksNano/CSI-Camera?tab=readme-ov-file

But when running the same in a python script it didn't work. It shows "Unable to open camera" which usually means the OpenCV version or installation is something wrong.

We checked the opencv version using this inline bash script:
  
```bash
python -c "import cv2; print(cv2.__version__)"

```

It showed a correct up to date version (4.5.3) but still the camera was not being detected.

This usually means that the CUDA support inside of openCV is missing, we checked this using the following command:
  
```bash
python -c "import cv2; print(cv2.cuda.getCudaEnabledDeviceCount())"
```

This showed 0 which means that the CUDA support is not enabled.

We tried to fix this by reinstalling openCV with CUDA support using the following command:

```bash

```

To debug the opencv cuda errors, we installed jtop which stands for jetson stats:
https://github.com/rbonghi/jetson_stats

Command to install:

```bash
sudo pip3 install -U jetson-stats
```


## Setup bluetooth connection to the PS4 controller:

## Setup Nano Car 4GB:
We followed the instructions from the following link:
https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devki

### Download the 4GB image for the jetson nano from the following link:
https://developer.nvidia.com/embedded/l4t/r32_release_v5.1/r32_release_v5.1/jeston_nano/jetson-nano-jp451-sd-card-image.zip

### Flash the image to the SD card:
1. We used Etcher to flash the image to the SD card.
2. Insert the SD card into the Nano
3. Connect a monitor, keyboard and mouse to the Nano.
4. Power the Nano on and it should start the boot process