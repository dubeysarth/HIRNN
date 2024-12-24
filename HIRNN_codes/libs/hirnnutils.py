#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):
    y_true = y_true[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))
    y_pred = y_pred[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))


    numerator = tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=1)
    denominator = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=1, keepdims=True)), axis=1)
    
    return numerator / denominator


def nse_metrics(y_true, y_pred):
    y_true = y_true[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))
    y_pred = y_pred[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))

    numerator = tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=1)
    denominator = tf.keras.backend.sum(tf.keras.backend.square(y_true - tf.keras.backend.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / denominator

    return 1.0 - rNSE

def mse_loss(y_true, y_pred):
    y_true = y_true[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))
    y_pred = y_pred[:, 1095:, :]  # skip values in the warmup period (the first 1095 days (3 years))  
    mse_value = tf.keras.backend.mean(tf.keras.backend.square(y_true - y_pred), axis=-1)
    return mse_value

