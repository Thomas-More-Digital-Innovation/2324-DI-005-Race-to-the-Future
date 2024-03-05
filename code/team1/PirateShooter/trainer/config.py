import os
AUGMENT = True

# CAMERA
CAMERA_TYPE = "PICAM"   # (PICAM|WEBCAM|CVCAM|CSIC|V4L|MOCK)
IMAGE_W = 512
IMAGE_H = 288
IMAGE_DEPTH = 3         # default RGB=3, make 1 for mono
# CAMERA_FRAMERATE = 20 ^default
CAMERA_FRAMERATE = 15

# For CSIC camera - If the camera is mounted in a rotated position, changing the below parameter will correct the output frame orientation
# CSIC_CAM_GSTREAMER_FLIP_PARM = 0 # (0 => none , 4 => Flip horizontally, 6 => Flip vertically)

# TRAINING
# The DEFAULT_MODEL_TYPE will choose which model will be created at training time. This chooses
# between different neural network designs. You can override this setting by passing the command
# line parameter --type to the python manage.py train and drive commands.
# (linear|categorical|rnn|imu|behavior|3d|localizer|latent)
DEFAULT_MODEL_TYPE = 'linear_one_output'
# how many records to use when doing one pass of gradient decent. Use a smaller number if your gpu is running out of memory.
BATCH_SIZE = 50
# what percent of records to use for training. the remaining used for validation.
TRAIN_TEST_SPLIT = 0.8
MAX_EPOCHS = 100  # how many times to visit all records of your data
STEPS_PER_EPOCH = 20
SHOW_PLOT = True  # would you like to see a pop up display of final loss?
VERBOSE_TRAIN = True  # would you like to see a progress bar with text during training?
# would you like to stop the training if we see it's not improving fit?
USE_EARLY_STOP = True
EARLY_STOP_PATIENCE = 5  # how many epochs to wait before no improvement
# early stop will want this much loss change before calling it improved.
MIN_DELTA = .0005
PRINT_MODEL_SUMMARY = True  # print layers and weights to stdout
OPTIMIZER = "adam"  # adam, sgd, rmsprop, etc.. None accepts default
LEARNING_RATE = 0.001  # only used when OPTIMIZER specified
LEARNING_RATE_DECAY = 0.0  # only used when OPTIMIZER specified
# change to true to automatically send best model during training
SEND_BEST_MODEL_TO_PI = False
# keep images in memory. will speed succesive epochs, but crater if not enough mem.
CACHE_IMAGES = False

# This will remove weights from your model. The primary goal is to increase performance.
PRUNE_CNN = False
PRUNE_PERCENT_TARGET = 75       # The desired percentage of pruning.
# Percenge of pruning that is perform per iteration.
PRUNE_PERCENT_PER_ITERATION = 20
# The max amout of validation loss that is permitted during pruning.
PRUNE_VAL_LOSS_DEGRADATION_LIMIT = 0.2
# percent of dataset used to perform evaluation of model.
PRUNE_EVAL_PERCENT_OF_DATASET = .05

# Region of interst cropping
# only supported in Categorical and Linear models.
# If these crops values are too large, they will cause the stride values to become negative and the model with not be valid.
ROI_CROP_TOP = 0  # the number of rows of pixels to ignore on the top of the image
ROI_CROP_BOTTOM = 0  # the number of rows of pixels to ignore on the bottom of the image


# Model transfer options
# When copying weights during a model transfer operation, should we freeze a certain number of layers
# to the incoming weights and not allow them to change during training?
FREEZE_LAYERS = False  # default False will allow all layers to be modified by training
# when freezing layers, how many layers from the last should be allowed to train?
NUM_LAST_LAYERS_TO_TRAIN = 7


# For the categorical model, this limits the upper bound of the learned throttle
# it's very IMPORTANT that this value is matched from the training PC config.py and the robot.py
# and ideally wouldn't change once set.
MODEL_CATEGORICAL_MAX_THROTTLE_RANGE = 0.5

# RNN or 3D
# some models use a number of images over time. This controls how many.
SEQUENCE_LENGTH = 3

# IMU
# when true, this add a Mpu6050 part and records the data. Can be used with a
HAVE_IMU = False


# BEHAVIORS
# When training the Behavioral Neural Network model, make a list of the behaviors,
# Set the TRAIN_BEHAVIORS = True, and use the BEHAVIOR_LED_COLORS to give each behavior a color
TRAIN_BEHAVIORS = False
BEHAVIOR_LIST = ['Left_Lane', "Right_Lane"]
BEHAVIOR_LED_COLORS = [(0, 10, 0), (10, 0, 0)]  # RGB tuples 0-100 per chanel

# Localizer
# The localizer is a neural network that can learn to predice it's location on the track.
# This is an experimental feature that needs more developement. But it can currently be used
# to predict the segement of the course, where the course is divided into NUM_LOCATIONS segments.
TRAIN_LOCALIZER = False
NUM_LOCATIONS = 10
# when enabled, makes it easier to divide our data into one tub per track length if we make a new tub on each X button press.
BUTTON_PRESS_NEW_TUB = False


# When racing, to give the ai a boost, configure these values.
# the ai will output throttle for this many seconds
AI_LAUNCH_DURATION = 0.0
AI_LAUNCH_THROTTLE = 0.0            # the ai will output this throttle value
# this keypress will enable this boost. It must be enabled before each use to prevent accidental trigger.
AI_LAUNCH_ENABLE_BUTTON = 'R2'
# when False ( default) you will need to hit the AI_LAUNCH_ENABLE_BUTTON for each use. This is safest. When this True, is active on each trip into "local" ai mode.
AI_LAUNCH_KEEP_ENABLED = False

# Scale the output of the throttle of the ai pilot for all model types.
# this multiplier will scale every throttle value for all output from NN models
AI_THROTTLE_MULT = 1.0

# Path following
PATH_FILENAME = "path.pkl"  # the path will be saved to this filename
# the path display will be scaled by this factor in the web page
PATH_SCALE = 5.0
# 255, 255 is the center of the map. This offset controls where the origin is displayed.
PATH_OFFSET = (0, 0)
# after travelling this distance (m), save a path point
PATH_MIN_DIST = 0.3
PID_P = -10.0                       # proportional mult for PID path follower
PID_I = 0.000                       # integral mult for PID path follower
PID_D = -0.2                        # differential mult for PID path follower
PID_THROTTLE = 0.2                  # constant throttle value during path following
SAVE_PATH_BTN = "cross"             # joystick button to save path
# joystick button to press to move car back to origin
RESET_ORIGIN_BTN = "triangle"
