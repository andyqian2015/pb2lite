#!/usr/bin/env python
# encoding: utf-8

'''
@PROJECT: GetData
@author: andy
@contact: andylina@163.com
@file: model_compress.py
@time: 19-9-29 下午2:05
'''

import tensorflow as tf

# input_layer = ["input"]
# input_layer = ["input0"]
input_layer = ["Placeholder"]
# output_layer = ["InceptionV1/Logits/Predictions/Reshape_1"]
# output_layer = ["output_node0"]
output_layer = ["fc/add"]
# graph_def_file = "/home/andy/8TDisk/test5/inception_v1_2016_08_28_frozen.pb/inception_v1_2016_08_28_frozen.pb"
# graph_def_file = "/home/andy/8TDisk/test4/soft.pb"
# graph_def_file = "/home/andy/8TDisk/test4/soft.pb"
# save_model_dir = "/home/andy/resnet-in-tensorflow/logs_test_110"
# save_model_dir = "/home/andy/8TDisk/test5/inception_v1_2016_08_28_frozen.pb/"
# save_model_dir = "/home/andy/resnet-in-tensorflow/models/pb/"
graph_def_file = "/home/andy/resnet-in-tensorflow/models/pb/frozen.pb"
# save_model_dir = "/home/andy/resnet-in-tensorflow/logs_test_110"
# converter = tf.lite.TFLiteConverter.from_saved_model(save_model_dir)
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_layer, output_layer)
# converter.post_training_quantize = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
tflite_model = converter.convert()
open("/home/andy/resnet-in-tensorflow/models/pb/model.tflite", "wb").write(tflite_model)

# import keras
# path = "/home/andy/8TDisk/test2/Step3_2_single.hdf5"
# # converter = tf.lite.TFLiteConverter.from_keras_model_file(path)
# keras.models.load_model(path)


# 查看模型的权重参数的数据类型
# import tensorflow as tf
# from tensorflow.python import pywrap_tensorflow
#
# model_reader = pywrap_tensorflow.NewCheckpointReader("/home/andy/8TDisk/test/soft.pb")
# var_dict = model_reader.get_variable_to_shape_map()
#
# for key in var_dict:
#     print("variable name: ", key)
#     print(model_reader.get_tensor(key).dtype)


# import h5py
# with h5py.File("/home/andy/8TDisk/test2/Step3_2_single.hdf5", "r") as f:
#     for i in f.attrs:
#         print(i)
#     model_structure = f.attrs["model_config"]






