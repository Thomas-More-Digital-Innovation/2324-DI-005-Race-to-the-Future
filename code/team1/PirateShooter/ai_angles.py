# python standard libraries
from PIL import Image
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from imgaug import augmenters as img_aug
import cv2
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.python.keras.losses import MeanSquaredError
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizers import adam_v2
from tensorflow.python.keras.layers import Convolution2D, Dropout, Flatten, Dense, MaxPool2D, Conv2D
from tensorflow.python.keras import Model, Input, Sequential, callbacks
import keras
import tensorflow as tf
import pandas as pd
import numpy as np
from trainer.util.MyCPCallback import MyCPCallback
# from ai_training import TrainingClass
from trainer import config
import os
import random
import fnmatch
import datetime
import pickle
import json
import sys
from recorder.model.datapoint import DataPoint

sys.path.append('code/AI_training')

# data processing
np.set_printoptions(formatter={'float_kind': lambda x: "%.4f" % x})

pd.set_option('display.width', 300)
pd.set_option('display.float_format', '{:,.4f}'.format)
pd.set_option('display.max_colwidth', 200)

# tensorflow
print(f'tf.__version__: {tf.__version__}')

# sklearn

# imaging

###############################################################
# Function to grab the angle from the corresponding json file #
###############################################################


def getJsonValues(currentpath, jsonNumber):
    """
    get the angle and throttle values from the given json
    - jsonPath: str
        - the path of the json"""
    # Opening JSON file
    f = open(f'{currentpath}/{jsonNumber}')
    # returns JSON object as a dictionary
    data = json.load(f)
    # Closing file
    f.close()
    angle = data["user/angle"]
    return angle

##############################
# Function to convert to RGB #
##############################


def my_imread(image_path):
    image = cv2.imread(image_path)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

################################################################################
# Function to perform a random form of data augmentation on the training image #
################################################################################


def random_data_augmentation(image, steering_angle):
    """
    grab an image and the corresponding steering value and perform a random augmentation on it
    - image: cv2 object
        - the image created through the my_imread function
    - steering_angle
        - steering angle corresponding to image from steering_angle list"""
    def pan(image):
        # pan left / right / up / down about 10%
        pan = img_aug.Affine(translate_percent={
                             "x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        image = pan.augment_image(image)
        return image

    def zoom(image):
        # zoom from 100% (no zoom) to 130%
        zoom = img_aug.Affine(scale=(1, 1.3))
        image = zoom.augment_image(image)
        return image

    def adjust_brightness(image):
        # increase or decrease brightness by 30%
        brightness = img_aug.Multiply((0.7, 1.3))
        image = brightness.augment_image(image)
        return image

    def random_flip(image, steering_angle):
        is_flip = random.randint(0, 1)
        if is_flip == 1:
            # randomly flip horizon
            image = cv2.flip(image, 1)
            steering_angle *= -1
        return image, steering_angle

    def blur(image):
        # kernel larger than 5 would make the image way too blurry
        kernel_size = random.randint(1, 5)
        image = cv2.blur(image, (kernel_size, kernel_size))
        return image

    # LIVE ORIGINAL IMAGE TRACKING
    # cv2.imshow("Original Image", image)
    # cv2.waitKey(400)
    # cv2.destroyWindow("Original Image")

    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = blur(image)
    if np.random.rand() < 0.5:
        image = adjust_brightness(image)
    image, steering_angle = random_flip(image, steering_angle)

    # LIVE AUGMENTATION TRACKING
    # cv2.imshow("Augmented image", image)
    # cv2.waitKey(400)
    # cv2.destroyWindow("Augmented image")

    return image, steering_angle

################################
# CONVOLUTIONAL NEURAL NETWORK #
################################


def nvidia_model():
    model = Sequential(name='Nvidia_Model')

    # elu=Expenential Linear Unit, similar to leaky Relu
    # skipping 1st hiddel layer (normalization layer), as we have normalized the data

    # Convolution Layers
    model.add(Conv2D(24, (5, 5), strides=(2, 2), input_shape=(
        288, 512, 3), activation='elu', padding="valid"))
    model.add(Conv2D(36, (5, 5), strides=(2, 2),
              activation='elu', padding="valid"))
    model.add(Conv2D(48, (5, 5), strides=(2, 2),
              activation='elu', padding="valid"))
    model.add(Conv2D(64, (3, 3), activation='elu', padding="valid"))
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Conv2D(64, (3, 3), activation='elu', padding="valid"))

    # Fully Connected Layers
    model.add(Flatten())
    model.add(Dropout(0.2))  # not in original model. added for more robustness
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # output layer: turn angle (from 45-135, 90 is straight, <90 turn left, >90 turn right)
    model.add(Dense(1, activation='sigmoid'))
    model.add(Flatten())
    # since this is a regression problem not classification problem,
    # we use MSE (Mean Squared Error) as loss function
    optimizer = adam_v2.Adam(learning_rate=1e-3)  # lr is learning rate
    loss = MeanSquaredError()
    model.compile(loss=loss, optimizer=optimizer, metrics=[
                  'accuracy', 'MeanSquaredError', 'AUC'])

    return model

######################################################
# Filter out the red tubes needed for edge detection #
######################################################


def img_preprocess(image, input_shape):
    # We cut off the top quarter of the image, because it is not relevant for lane tracking.
    # height, _, _ = image.shape
    # image = image[int(height/3):,:,:]

    # Convert image to HSV because CV uses BGR
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # We isolate the color RED in the image (we take the upper and lower values for all red in the rgb spectrum)
    # Lower mask (0-10)
    lower_red = np.array([0, 100, 100])  # hue (0-180), sat , value
    upper_red = np.array([10, 255, 255])  # hue (0-180), sat , value
    mask0 = cv2.inRange(hsv, lower_red, upper_red)

    # Upper mask (170-180)
    lower_red = np.array([168, 32, 43])  # hue (0-180), sat , value
    upper_red = np.array([180, 255, 255])  # hue (0-180), sat , value
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    # join my masks
    mask = mask0+mask1

    result = image.copy()
    result = cv2.bitwise_and(result, result, mask=mask)

    resized = cv2.resize(result, input_shape, interpolation=cv2.INTER_AREA)
    resized = resized / 255

    # LIVE PREPROCESSED IMAGE TRACKING
    # cv2.imshow("Image passed to CNN", resized)
    # cv2.waitKey(400)
    # cv2.destroyWindow("Image passed to CNN")

    return resized

########################
# Image Data Generator #
########################


def image_data_generator(image_paths, steering_angles, input_shape, batch_size, is_training):
    while True:
        batch_images = []
        batch_steering_angles = []

        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            image_path = image_paths[random_index]
            # image = my_imread(image_paths[random_index])
            image = image_paths[random_index]
            steering_angle = steering_angles[random_index]
            if is_training:
                # training: augment image
                image, steering_angle = random_data_augmentation(
                    image, steering_angle)

            image = img_preprocess(image, input_shape)
            batch_images.append(image)
            batch_steering_angles.append(steering_angle)

        yield(np.asarray(batch_images), np.asarray(batch_steering_angles))


def extractor(datapoint_file):
    with open(f"datapoints/{datapoint_file}", "rb") as pickle_file:
        print(f"Loading {datapoint_file}!")
        datapoint: DataPoint = pickle.load(pickle_file)

        # if type(datapoint) is not DataPoint:
        #     print(f"Invalid pickle: {datapoint_file}")
        #     return
        return datapoint.steering_angle, datapoint.image


#############################################################################################
# PART 1 - grabbing all the image paths and angle values and store them in 2 separate lists #
#############################################################################################
# -------- OLD  ---------
# directory = 'code\AI_training\data'
# jpeg_files = []
# json_files = []
# steering_angles = []
# for currentpath, folders, files in os.walk(directory):
#     for i in files:
#         if i.endswith('.jpg'):
#             jpeg_files.append(os.path.join(currentpath, i))
#         if i.endswith('.json'):
#             json_files.append(i)
#             steering_angles.append(getJsonValues(currentpath, i))
# -----------------------
steering_angles = []
jpeg_files = []

print("Extracting data from pickle files...")
try:
    for datapoint_file in os.listdir("./datapoints"):
        # check if the file is a pickle file
        if datapoint_file.endswith(".pickle"):
            steering_angle, image = extractor(datapoint_file)
            steering_angles.append(steering_angle)
            jpeg_files.append(image)
except:
    print("no more pickle files?")
    pass
#  Terminal Print to check if the right values are in the dataframes
df = pd.DataFrame()
df['ImagePath'] = jpeg_files
df['Angle'] = steering_angles
print(df['ImagePath'])
print(df['Angle'])


###################################################
# PART 2 -  Create a training and validations set #
###################################################
X_train, X_valid, y_train, y_valid = train_test_split(
    jpeg_files, steering_angles, test_size=0.2)
print("Training data: %d\nValidation data: %d" % (len(X_train), len(X_valid)))

###############################
# PART 3 - Create a CNN model #
###############################
model = nvidia_model()
print(model.summary())

#########################################
# PART 4 - DEFINE EARLY STOPPING #
#########################################
model_output_dir = 'models'
temp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def on_best_model(model):
    model_type = config.DEFAULT_MODEL_TYPE
    model_name = "./models/{}-{}-{}-augment-{}".format(
        temp, model_type, config.OPTIMIZER, True)
    # Commented out for TF Lite testing
    # model.save(model_filename, include_optimizer=False)

    # Added for TF Lite testing
    model.save(model_name)

# saves the model weights after each epoch if the validation loss decreased
# checkpoint_callback = callbacks.ModelCheckpoint(filepath=os.path.join(model_output_dir,
# 'lane_navigation_check.h5'), verbose=1, save_best_only=True)


save_best = MyCPCallback(send_model_cb=on_best_model(model),
                         filepath=model_output_dir,
                         monitor='val_loss',
                         verbose=1,
                         save_best_only=True,
                         mode='min',
                         cfg=config)

# stop training if the validation error stops improving.
early_stop = EarlyStopping(monitor='val_loss',
                           min_delta=config.MIN_DELTA,
                           patience=config.EARLY_STOP_PATIENCE,
                           verbose=1,
                           mode='auto')

callbacks_list = [save_best]
if config.USE_EARLY_STOP:
    callbacks_list.append(early_stop)


#############################
# Part 5 - Model Parameters #
#############################

# Image Size
inputShape = (config.IMAGE_W, config.IMAGE_H)
batch_size = config.BATCH_SIZE
epochs = config.MAX_EPOCHS
steps_per_epoch = config.STEPS_PER_EPOCH

##########################
# PART 6 - RUN THE MODEL #
##########################
history = model.fit_generator(image_data_generator(X_train, y_train, input_shape=inputShape, batch_size=batch_size, is_training=True),
                              steps_per_epoch=steps_per_epoch,
                              epochs=epochs,
                              validation_data=image_data_generator(
                                  X_valid, y_valid, input_shape=inputShape, batch_size=10, is_training=False),
                              #   validation_data = (X_valid, y_valid),
                              validation_steps=200,
                              verbose=config.VERBOSE_TRAIN,
                              shuffle=1,
                              callbacks=callbacks_list)
# # always save model output as soon as model finishes training
# model.save(os.path.join(model_output_dir,'lane_navigation_final.h5'))

########################
# PART 7 - PLOT VALIDATION LOSS #
########################
plt.figure(1)

# Only do accuracy if we have that data (e.g. categorical outputs)
if 'angle_out_acc' in history.history:
    plt.subplot(121)

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validate'], loc='upper right')

# summarize history for acc
if 'angle_out_acc' in history.history:
    plt.subplot(122)
    plt.plot(history.history['angle_out_acc'])
    plt.plot(history.history['val_angle_out_acc'])
    plt.title('model angle accuracy')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    # plt.legend(['train', 'validate'], loc='upper left')

plt.savefig("model.png")
# plt.show()


# date_str = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
# history_path = os.path.join(model_output_dir,'history.pickle')
# with open(history_path, 'wb') as f:
#     pickle.dump(history.history, f, pickle.HIGHEST_PROTOCOL)
