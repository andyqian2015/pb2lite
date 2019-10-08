#!/usr/bin/env python
# encoding: utf-8

'''
@PROJECT: GetData
@author: andy
@contact: andylina@163.com
@file: tflite_parse.py
@time: 19-9-30 下午2:36
'''

import tensorflow as tf
import numpy as np
import cv2

import _pickle as cPickle

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.per_process_gpu_memory_fraction = 0.4

VALI_RANDOM_LABEL = False # Want to use random label for validation?

# GPU_rate={'GPUpercentage':1.0}

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
NUM_CLASS = 10

vali_dir = '/home/andy/resnet-in-tensorflow/cifar10_data/cifar-10-batches-py/test_batch'


def read_in_all_images(address_list, shuffle=False, is_random_label = False):
    """
    This function reads all training or validation data, shuffles them if needed, and returns the
    images and the corresponding labels as numpy arrays

    :param address_list: a list of paths of cPickle files
    :return: concatenated numpy array of data and labels. Data are in 4D arrays: [num_images,
    image_height, image_width, image_depth] and labels are in 1D arrays: [num_images]
    """
    data = np.array([]).reshape([0, IMG_WIDTH * IMG_HEIGHT * IMG_DEPTH])
    label = np.array([])

    for address in address_list:
        print('Reading images from ' + address)
        batch_data, batch_label = _read_one_batch(address, is_random_label)
        # Concatenate along axis 0 by default
        data = np.concatenate((data, batch_data))
        label = np.concatenate((label, batch_label))

    num_data = len(label)

    # This reshape order is really important. Don't change
    # Reshape is correct. Double checked
    data = data.reshape((num_data, IMG_HEIGHT * IMG_WIDTH, IMG_DEPTH), order='F')
    data = data.reshape((num_data, IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH))

    if shuffle is True:
        print('Shuffling')
        order = np.random.permutation(num_data)
        data = data[order, ...]
        label = label[order]

    data = data.astype(np.float32)
    return data, label

def _read_one_batch(path, is_random_label):
    '''
    The training data contains five data batches in total. The validation data has only one
    batch. This function takes the directory of one batch of data and returns the images and
    corresponding labels as numpy arrays

    :param path: the directory of one batch of data
    :param is_random_label: do you want to use random labels?
    :return: image numpy arrays and label numpy arrays
    '''
    fo = open(path, 'rb')
    dicts = cPickle.load(fo, encoding='iso-8859-1')
    fo.close()

    data = dicts['data']
    if is_random_label is False:
        label = np.array(dicts['labels'])
    else:
        labels = np.random.randint(low=0, high=10, size=10000)
        label = np.array(labels)
    return data, label


def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    validation_array, validation_labels = read_in_all_images([vali_dir],
                                                       is_random_label=VALI_RANDOM_LABEL)
    validation_array = whitening_image(validation_array)

    return validation_array, validation_labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def lite_parse(img, modelpath):

    interpreter = tf.lite.Interpreter(model_path=modelpath)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    print(str(input_details))
    output_details = interpreter.get_output_details()
    print(str(output_details))

    interpreter.set_tensor(input_details[0]["index"], img)

    # 调用模型
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    result = np.squeeze(output_data)
    print(result.shape)
    print(result[0])
    print(result[1])

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        image_np[i,...] = (image_np[i, ...] - mean) / std
    return image_np


if __name__ == "__main__":
    vali_data, vali_labels = read_validation_data()
    vali_data = vali_data[0:128]
    vali_labels = vali_labels[0:128]
    model_path = "/home/andy/resnet-in-tensorflow/models/pb/model.tflite"
    print(vali_data[0])
    while True:
        lite_parse(vali_data, model_path)


