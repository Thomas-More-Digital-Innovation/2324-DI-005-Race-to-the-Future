import os

from Tub import Tub
from kerasai import KerasLinear
import numpy as np
from PIL import Image

from kerasai.categorical import KerasCategorical

one_byte_scale = 1.0 / 255.0


def get_model_by_type(model_type, cfg):
    if model_type is None:
        model_type = cfg.DEFAULT_MODEL_TYPE
    print("\"get_model_by_type\" model Type is: {}".format(model_type))

    input_shape = (cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)
    roi_crop = (cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)

    if model_type == "linear":
        kl = KerasLinear(input_shape=input_shape, roi_crop=roi_crop)
    elif model_type == "categorical":
        kl = KerasCategorical(input_shape=input_shape, throttle_range=cfg.MODEL_CATEGORICAL_MAX_THROTTLE_RANGE,
                              roi_crop=roi_crop)
    else:
        raise Exception("unknown model type: %s" % model_type)

    return kl


def expand_path_masks(paths):
    '''
    take a list of paths and expand any wildcards
    returns a new list of paths fully expanded
    '''
    import glob
    expanded_paths = []
    for path in paths:
        if '*' in path or '?' in path:
            mask_paths = glob.glob(path)
            expanded_paths += mask_paths
        else:
            expanded_paths.append(path)

    return expanded_paths


def gather_tub_paths(cfg, tub_names=None):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub paths
    '''
    if tub_names:
        if type(tub_names) == list:
            tub_paths = [os.path.expanduser(n) for n in tub_names]
        else:
            tub_paths = [os.path.expanduser(n) for n in tub_names.split(',')]
        return expand_path_masks(tub_paths)
    else:
        paths = [os.path.join(cfg.DATA_PATH, n) for n in os.listdir(cfg.DATA_PATH)]
        dir_paths = []
        for p in paths:
            if os.path.isdir(p):
                dir_paths.append(p)
        return dir_paths


def gather_tubs(cfg, tub_names):
    '''
    takes as input the configuration, and the comma seperated list of tub paths
    returns a list of Tub objects initialized to each path
    '''

    tub_paths = gather_tub_paths(cfg, tub_names)
    tubs = [Tub(p) for p in tub_paths]

    return tubs


def gather_records(cfg, tub_names, opts=None, verbose=False):
    tubs = gather_tubs(cfg, tub_names)

    records = []

    for tub in tubs:
        if verbose:
            print("[Utils:gather_records] Path : '{}'".format(tub.path))
        record_paths = tub.gather_records()
        records += record_paths

    return records


def get_image_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[0])


def get_record_index(fnm):
    sl = os.path.basename(fnm).split('_')
    return int(sl[1].split('.')[0])


def load_scaled_image_arr(filename, cfg):
    '''
    load an image from the filename, and use the cfg to resize if needed
    also apply cropping and normalize
    '''
    try:
        img = Image.open(filename)
        
        # print(f'IMG: {img}')
        if img.height != cfg.IMAGE_H or img.width != cfg.IMAGE_W:
            img = img.resize((cfg.IMAGE_W, cfg.IMAGE_H))
        
        img_arr = np.array(img)
        # print(img_arr.shape)
        
        img_arr = normalize_and_crop(img_arr, cfg)
        
        croppedImgH = img_arr.shape[0]
        croppedImgW = img_arr.shape[1]
        # print(img_arr.shape)  # should be (720, 1280, 3), is (720, 1280)
        # print(cfg.IMAGE_DEPTH)

        if img_arr.shape[2] == 3 and cfg.IMAGE_DEPTH == 1:
            img_arr = dk.utils.rgb2gray(img_arr).reshape(croppedImgH, croppedImgW, 1)
            # print(img_arr.shape)
    except Exception as e:
        print(e)
        print('failed to load image:', filename)
        img_arr = None
    return img_arr


def normalize_and_crop(img_arr, cfg):
    img_arr = img_arr.astype(np.float32) * one_byte_scale
    if cfg.ROI_CROP_TOP or cfg.ROI_CROP_BOTTOM:
        img_arr = img_crop(img_arr, cfg.ROI_CROP_TOP, cfg.ROI_CROP_BOTTOM)
        if len(img_arr.shape) == 2:
            img_arrH = img_arr.shape[0]
            img_arrW = img_arr.shape[1]
            img_arr = img_arr.reshape(img_arrH, img_arrW, 1)
    return img_arr


def img_crop(img_arr, top, bottom):
    if bottom is 0:
        end = img_arr.shape[0]
    else:
        end = -bottom
    return img_arr[top:end, ...]


def clamp(n, min, max):
    if n < min:
        return min
    if n > max:
        return max
    return n


def linear_unbin(arr, N=15, offset=-1, R=2.0):
    '''
    preform inverse linear_bin, taking
    one hot encoded arr, and get max value
    rescale given R range and offset
    '''
    b = np.argmax(arr)
    a = b * (R / (N + offset)) + offset
    return a


def linear_bin(a, N=15, offset=1, R=2.0):
    '''
    create a bin of length N
    map val A to range R
    offset one hot bin by offset, commonly R/2
    '''
    a = a + offset
    b = round(a / (R / (N - offset)))
    arr = np.zeros(N)
    b = clamp(b, 0, N - 1)
    arr[int(b)] = 1
    return arr
