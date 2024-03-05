import glob
import json
import os
import pickle
import random
import time
from datetime import datetime, timedelta

import zlib
from os.path import join, dirname, splitext, basename

import numpy as np
from PIL import Image
from tensorflow.python import keras

# from kerasai.categorical import KerasCategorical

from util.MyCPCallback import MyCPCallback
from util.utils import get_model_by_type, gather_tub_paths, gather_records, get_image_index, get_record_index, \
    load_scaled_image_arr, linear_bin
from util.augment import augment_image

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('AGG')
    do_plot = True
except:
    do_plot = False
    print("matplotlib not installed")

# Comment out for CPU training, GPU by default
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
figure_format = "png"


def extract_data_from_pickles(cfg, tubs):
    """
    Extracts record_{id}.json and image from a pickle with the same id if exists in the tub.
    Then writes extracted json/jpg along side the source pickle that tub.
    This assumes the format {id}.pickle in the tub directory.
    :param cfg: config with data location configuration. Generally the global config object.
    :param tubs: The list of tubs involved in training.
    :return: implicit None.
    """
    t_paths = gather_tub_paths(cfg, tubs)
    for tub_path in t_paths:
        file_paths = glob.glob(join(tub_path, '*.pickle'))
        print(
            '[AiTraining:extract_data_from_pickles] found {} pickles writing json records and images in tub {}'.format(
                len(file_paths), tub_path))
        for file_path in file_paths:
            # print('loading data from {}'.format(file_paths))
            with open(file_path, 'rb') as f:
                p = zlib.decompress(f.read())
            data = pickle.loads(p)

            base_path = dirname(file_path)
            filename = splitext(basename(file_path))[0]
            image_path = join(base_path, filename + '.jpg')
            img = Image.fromarray(np.uint8(data['val']['cam/image_array']))
            img.save(image_path)

            data['val']['cam/image_array'] = filename + '.jpg'

            with open(join(base_path, 'record_{}.json'.format(filename)), 'w') as f:
                json.dump(data['val'], f)


class TrainingClass():
    def __init__(self):
        print("[TrainingClass] ctor init")

    def on_best_model(self, cfg, model, model_filename):

        # JCLAES: Commented out for TF Lite testing
        # model.save(model_filename, include_optimizer=False)

        # JCLAES: Added for TF Lite testing
        model.save(model_filename)

        if not cfg.SEND_BEST_MODEL_TO_PI:
            return

        on_windows = os.name == 'nt'

    def start_training(self, cfg, dirs, model_name, model_type=None, aug=False):
        """
        use the specified data in tub_names to train an artifical neural network
        saves the output trained model as model_name
        """
        verbose = cfg.VERBOSE_TRAIN
        if model_type is None:
            model_type = cfg.DEFAULT_MODEL_TYPE  # linear

        if model_name and not '.h5' == model_name[-3:]:
            raise Exception("model filename should end with h5")

        gen_records = {}
        opts = {'cfg': cfg}

        if 'linear' in model_type:
            train_type = 'linear'
        else:
            train_type = model_type

        model = get_model_by_type(train_type, cfg)

        # opts['categorical'] = type(model) in [KerasCategorical]
        opts['categorical'] = True
        print('[AiTraining:start_training] training with model type', type(model))

        if cfg.OPTIMIZER:
            model.set_optimizer(
                cfg.OPTIMIZER, cfg.LEARNING_RATE, cfg.LEARNING_RATE_DECAY)

        model.compile()

        if cfg.PRINT_MODEL_SUMMARY:
            print(model.model.summary())

        opts['keras_pilot'] = model
        # opts['continuous'] = False  # continuous
        opts['continuous'] = True  # continuous
        opts['model_type'] = model_type

        extract_data_from_pickles(cfg, dirs)
        print("[AiTraining:start_training] Start gathering records")
        records = gather_records(cfg, dirs, opts, verbose=True)
        print('[AiTraining:start_training] collating %d records ...' %
              (len(records)))
        collate_records(records, gen_records, opts)
        print("[AiTraining:start-training] finished collating records")

        def generator(save_best, opts, data, batch_size, isTrainSet=True, min_records_to_train=1):
            num_records = len(data)
            while True:
                if isTrainSet and opts['continuous']:
                    '''
                    When continuous training, we look for new records after each epoch.
                    This will add new records to the train and validation set.
                    '''
                    records = gather_records(cfg, dirs, opts)
                    if len(records) > num_records:
                        collate_records(records, gen_records, opts)
                        new_num_rec = len(data)
                        if new_num_rec > num_records:
                            print('picked up', new_num_rec -
                                  num_records, 'new records!')
                            num_records = new_num_rec
                            save_best.reset_best()
                    if num_records < min_records_to_train:
                        print("not enough records to train. need %d, have %d. waiting..." % (
                            min_records_to_train, num_records))
                        time.sleep(10)
                        continue

                batch_data = []

                keys = list(data.keys())

                random.shuffle(keys)

                kl = opts['keras_pilot']

                if type(kl.model.output) is list:
                    model_out_shape = (2, 1)
                else:
                    model_out_shape = kl.model.output.shape

                if type(kl.model.input) is list:
                    model_in_shape = (2, 1)
                else:
                    model_in_shape = kl.model.input.shape

                has_imu = False  # type(kl) is KerasIMU
                has_bvh = False  # type(kl) is KerasBehavioral
                img_out = False  # type(kl) is KerasLatent
                loc_out = False  # type(kl) is KerasLocalizer

                if img_out:
                    import cv2

                for key in keys:

                    if not key in data:
                        continue

                    _record = data[key]

                    if _record['train'] != isTrainSet:
                        continue

                    if False:
                        # in continuous mode we need to handle files getting deleted
                        filename = _record['image_path']
                        if not os.path.exists(filename):
                            data.pop(key, None)
                            continue

                    batch_data.append(_record)

                    if len(batch_data) == batch_size:
                        inputs_img = []
                        inputs_imu = []
                        inputs_bvh = []
                        angles = []
                        # throttles = []
                        out_img = []
                        out_loc = []
                        out = []

                        for record in batch_data:
                            # get image data if we don't already have it
                            if record['img_data'] is None:
                                filename = record['image_path']

                                img_arr = load_scaled_image_arr(filename, cfg)

                                if img_arr is None:
                                    break

                                if aug:
                                    orig_shape = img_arr.shape
                                    img2 = np.zeros_like(img_arr)
                                    img_arr = augment_image(
                                        img_arr, dimensions=(cfg.IMAGE_H, cfg.IMAGE_W))

                                    img2[:, :, 0] = img_arr
                                    img2[:, :, 1] = img_arr
                                    img2[:, :, 2] = img_arr
                                    img_arr = img2

                                if cfg.CACHE_IMAGES:
                                    record['img_data'] = img_arr
                            else:
                                img_arr = record['img_data']

                            if img_out:
                                rz_img_arr = cv2.resize(
                                    img_arr, (127, 127)) / 255.0
                                out_img.append(
                                    rz_img_arr[:, :, 0].reshape((127, 127, 1)))

                            if loc_out:
                                out_loc.append(record['location'])

                            if has_imu:
                                inputs_imu.append(record['imu_array'])

                            if has_bvh:
                                inputs_bvh.append(record['behavior_arr'])

                            inputs_img.append(img_arr)
                            angles.append(record['angle'])
                            # throttles.append(record['throttle'])
                            # out.append([record['angle'], record['throttle']])
                            out.append([record['angle']])

                        if img_arr is None:
                            continue

                        img_arr = np.array(inputs_img).reshape(batch_size,
                                                               cfg.IMAGE_H, cfg.IMAGE_W, cfg.IMAGE_DEPTH)

                        if has_imu:
                            X = [img_arr, np.array(inputs_imu)]
                        elif has_bvh:
                            X = [img_arr, np.array(inputs_bvh)]
                        else:
                            X = [img_arr]

                        if img_out:
                            # y = [out_img, np.array(
                            #     angles), np.array(throttles)]
                            y = [out_img, np.array(angles)]
                        elif out_loc:
                            # y = [np.array(angles), np.array(
                            #     throttles), np.array(out_loc)]
                            y = [np.array(angles), np.array(out_loc)]
                        elif model_out_shape[1] == 2:
                            y = [np.array([out]).reshape(batch_size, 2)]
                        else:
                            # y = [np.array(angles), np.array(throttles)]
                            y = [np.array(angles)]

                        yield X, y

                        batch_data = []

        if model_name is None:
            temp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # JCLAES: Commented out for TF Lite
            # model_name = "./models/{}-{}-{}-augment-{}.h5".format(
            #     temp, model_type, cfg.OPTIMIZER, aug)

            # JCLAES: Added for TF Lite
            # This saves the model without .h5 and thus creates a SavedModel format instead of a Keras H5 model.
            model_name = "./models/{}-{}-{}-augment-{}".format(
                temp, model_type, cfg.OPTIMIZER, aug)

        model_path = os.path.expanduser(model_name)
        # checkpoint to save model after each epoch and send best to the pi.
        save_best = MyCPCallback(send_model_cb=self.on_best_model,
                                 filepath=model_path,
                                 monitor='val_loss',
                                 verbose=verbose,
                                 save_best_only=True,
                                 mode='min',
                                 cfg=cfg)

        train_gen = generator(
            save_best, opts, gen_records, cfg.BATCH_SIZE, True)
        val_gen = generator(save_best, opts, gen_records,
                            cfg.BATCH_SIZE, False)

        total_records = len(gen_records)

        num_train = 0
        num_val = 0

        for key, _record in gen_records.items():
            if _record['train'] == True:
                num_train += 1
            else:
                num_val += 1

        print("[AiTrain:start_training] train: %d, val: %d" %
              (num_train, num_val))
        print('[Aitrain:start_training] total records: %d' % (total_records))

        if not False:
            steps_per_epoch = num_train // cfg.BATCH_SIZE
        else:
            steps_per_epoch = 100

        val_steps = num_val // cfg.BATCH_SIZE
        print('[AiTrain:start_training] steps_per_epoch', steps_per_epoch)

        cfg.model_type = model_type
        self.go_train(model, cfg, train_gen, val_gen, gen_records, model_name, steps_per_epoch, val_steps, False,
                      verbose, save_best)
        return model_name

    def go_train(self, kl, cfg, train_gen, val_gen, gen_records, model_name, steps_per_epoch, val_steps, continuous,
                 verbose,
                 save_best=None):

        start = time.time()

        model_path = os.path.expanduser(model_name)

        # checkpoint to save model after each epoch and send best to the pi.
        if save_best is None:
            save_best = MyCPCallback(send_model_cb=self.on_best_model,
                                     filepath=model_path,
                                     monitor='val_loss',
                                     verbose=verbose,
                                     save_best_only=True,
                                     mode='min',
                                     cfg=cfg)

        # stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=cfg.MIN_DELTA,
                                                   patience=cfg.EARLY_STOP_PATIENCE,
                                                   verbose=verbose,
                                                   mode='auto')

        if steps_per_epoch < 2:
            raise Exception(
                "Too little data to train. Please record more records.")

        if continuous:
            epochs = 100000
        else:
            epochs = cfg.MAX_EPOCHS

        workers_count = 1
        use_multiprocessing = False

        callbacks_list = [save_best]

        if cfg.USE_EARLY_STOP and not continuous:
            callbacks_list.append(early_stop)

        history = kl.model.fit_generator(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            verbose=cfg.VERBOSE_TRAIN,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=val_steps,
            workers=workers_count,
            use_multiprocessing=use_multiprocessing)

        full_model_val_loss = min(history.history['val_loss'])
        max_val_loss = full_model_val_loss + cfg.PRUNE_VAL_LOSS_DEGRADATION_LIMIT

        duration_train = time.time() - start
        print("Training completed in %s." %
              str(timedelta(seconds=round(duration_train))))

        print("\n\n----------- Best Eval Loss :%f ---------" % save_best.best)

        if cfg.SHOW_PLOT:
            try:
                if do_plot:
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

                    plt.savefig(model_path + '_loss_acc_%f.%s' %
                                (save_best.best, figure_format))
                #  plt.show()
                else:
                    print("not saving loss graph because matplotlib not set up.")
            except Exception as ex:
                print("problems with loss graph: {}".format(ex))

        # Save tflite, optionally in the int quant format for Coral TPU


def collate_records(records, gen_records, opts):
    '''
    open all the .json records from records list passed in,
    read their contents,
    add them to a list of gen_records, passed in.
    use the opts dict to specify config choices
    '''

    new_records = {}

    for record_path in records:

        basepath = os.path.dirname(record_path)
        index = get_record_index(record_path)
        sample = {'tub_path': basepath, "index": index}

        key = make_key(sample)

        if key in gen_records:
            continue

        try:
            with open(record_path, 'r') as fp:
                json_data = json.load(fp)
        except:
            continue
        try:
            image_filename = json_data["cam/image_array"]
            file_name = os.path.basename(image_filename)
            image_path = os.path.join(basepath, file_name)
        except Exception as e:
            print("[AiTraining:collate_records] error loading data : {}".format(e))
            print("error json stuff {} ... {}".format(record_path, json_data))

        sample['record_path'] = record_path
        sample["image_path"] = image_path
        sample["json_data"] = json_data

        angle = float(json_data['user/angle'])
        # throttle = float(json_data["user/throttle"])

        # TODO: Nicky
        if opts['categorical']:
            angle = linear_bin(angle)
            # throttle = linear_bin(
            #     throttle, N=20, offset=0, R=opts['cfg'].MODEL_CATEGORICAL_MAX_THROTTLE_RANGE)

        sample['angle'] = angle
        # sample['throttle'] = throttle

        try:
            accl_x = float(json_data['imu/acl_x'])
            accl_y = float(json_data['imu/acl_y'])
            accl_z = float(json_data['imu/acl_z'])

            gyro_x = float(json_data['imu/gyr_x'])
            gyro_y = float(json_data['imu/gyr_y'])
            gyro_z = float(json_data['imu/gyr_z'])

            sample['imu_array'] = np.array(
                [accl_x, accl_y, accl_z, gyro_x, gyro_y, gyro_z])
        except:
            pass

        try:
            behavior_arr = np.array(json_data['behavior/one_hot_state_array'])
            sample["behavior_arr"] = behavior_arr
        except:
            pass

        try:
            location_arr = np.array(json_data['location/one_hot_state_array'])
            sample["location"] = location_arr
        except:
            pass

        sample['img_data'] = None

        # Initialise 'train' to False
        sample['train'] = False

        # We need to maintain the correct train - validate ratio across the dataset, even if continous training
        # so don't add this sample to the main records list (gen_records) yet.
        new_records[key] = sample

    # new_records now contains all our NEW samples
    # - set a random selection to be the training samples based on the ratio in CFG file
    shufKeys = list(new_records.keys())
    random.shuffle(shufKeys)
    trainCount = 0
    #  Ratio of samples to use as training data, the remaining are used for evaluation
    targetTrainCount = int(opts['cfg'].TRAIN_TEST_SPLIT * len(shufKeys))
    for key in shufKeys:
        new_records[key]['train'] = True
        trainCount += 1
        if trainCount >= targetTrainCount:
            break
    # Finally add all the new records to the existing list
    gen_records.update(new_records)


def make_key(sample):
    tub_path = sample['tub_path']
    index = sample['index']
    return tub_path + str(index)


def make_next_key(sample, index_offset):
    tub_path = sample['tub_path']
    index = sample['index'] + index_offset
    return tub_path + str(index)
