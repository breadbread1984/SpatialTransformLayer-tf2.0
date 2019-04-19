#!/usr/bin/python3

import numpy as np;
import tensorflow as tf;
from AffineLayer import AffineLayer;

class DefaultLocalizationNetwork(tf.keras.Model):

    def __init__(self):

        super(DefaultLocalizationNetwork, self).__init__();
        self.conv = tf.keras.layers.Conv2D(filters = 20, kernel_size = (5,5), strides = (1,1), padding = 'valid');
        self.dense = tf.keras.layers.Dense(units = 6, kernel_initializer = tf.constant_initializer(np.zeros(shape = (20,6), dtype = np.float32)), bias_initializer = tf.constant_initializer(np.array([1, 0, 0, 0, 1, 0], dtype = np.float32)));

    def call(self, inputs):

        results = self.conv(inputs);
        results = tf.math.reduce_mean(results, axis = [1,2]);
        results = self.dense(results);
        results = tf.reshape(results, shape = (-1, 2, 3));
        return results;

class SpatialTransformLayer(tf.keras.layers.Layer):

    def __init__(self, downsample_factor = 1, loc_model = DefaultLocalizationNetwork()):

        super(SpatialTransformLayer, self).__init__();
        self.loc_model = loc_model;
        self.affine_layer = AffineLayer(downsample_factor);

    def call(self, inputs):

        affines = self.loc_model(inputs);
        results = self.affine_layer(inputs, affines);
        return results;

if __name__ == "__main__":

    assert tf.executing_eagerly();
    st = SpatialTransformLayer();
    a = tf.constant(np.random.normal(size = (4,32,32,3)), dtype = tf.float32);
    b = st(a);

