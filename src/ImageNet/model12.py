import os
import numpy as np
import keras.backend as K
from keras import metrics
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Input, MaxPool2D, Activation, BatchNormalization, UpSampling2D, concatenate, LeakyReLU, Conv2D
from scipy.misc import imread
from skimage import color
import os, sys, inspect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

from dataset import *
from utils import *

IMAGENET_BATCH_SIZE = 128116
EPOCHS = 1000
BATCH_SIZE = 64
VALIDATION_SIZE = 4096
LEARNING_RATE = 0.004
INPUT_SHAPE = (64, 64, 1)
WEIGHTS = 'model12.hdf5'
MODE = 2  # 1: train - 2: visualize


def eacc(y_true, y_pred):
    return K.mean(K.equal(K.round(y_true), K.round(y_pred)))


def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true), axis=-1)


def learning_scheduler(epoch):
    lr = LEARNING_RATE / (2 ** (epoch // 20))
    print('\nlearning rate: ' + str(lr) + '\n')
    return lr


def create_conv(filters, kernel_size, inputs, name=None, strides=(1, 1), bn=True, padding='same', activation='relu'):
    conv = Conv2D(filters, kernel_size, strides=strides, padding=padding, kernel_initializer='he_normal', name=name)(
        inputs)

    if bn == True:
        conv = BatchNormalization()(conv)

    if activation == 'relu':
        conv = Activation(activation)(conv)
    elif activation == 'leakyrelu':
        conv = LeakyReLU()(conv)

    return conv


def create_model():
    inputs = Input(INPUT_SHAPE)
    conv1 = create_conv(64, (3, 3), inputs, 'conv1_1', activation='leakyrelu')
    conv1 = create_conv(64, (3, 3), conv1, 'conv1_2', activation='leakyrelu')
    conv1 = create_conv(64, (3, 3), conv1, 'conv1_3', activation='leakyrelu')
    pool1 = MaxPool2D((2, 2))(conv1)

    conv2 = create_conv(256, (3, 3), pool1, 'conv2_1', activation='leakyrelu')
    conv2 = create_conv(256, (3, 3), conv2, 'conv2_2', activation='leakyrelu')
    conv2 = create_conv(256, (3, 3), conv2, 'conv2_3', activation='leakyrelu')
    pool2 = MaxPool2D((2, 2))(conv2)

    conv3 = create_conv(512, (3, 3), pool2, 'conv3_1', activation='leakyrelu')
    conv3 = create_conv(512, (3, 3), conv3, 'conv3_2', activation='leakyrelu')
    conv3 = create_conv(512, (3, 3), conv3, 'conv3_3', activation='leakyrelu')
    pool3 = MaxPool2D((2, 2))(conv3)

    conv4 = create_conv(1024, (3, 3), pool3, 'conv4_1', activation='leakyrelu')
    conv4 = create_conv(1024, (3, 3), conv4, 'conv4_2', activation='leakyrelu')
    conv4 = create_conv(1024, (3, 3), conv4, 'conv4_3', activation='leakyrelu')

    up5 = create_conv(512, (2, 2), UpSampling2D((2, 2))(conv4), 'up5', activation='relu')
    merge5 = concatenate([conv3, up5], axis=3)
    conv5 = create_conv(512, (3, 3), merge5, 'conv5_1', activation='relu')
    conv5 = create_conv(512, (3, 3), conv5, 'conv5_2', activation='relu')
    conv5 = create_conv(512, (3, 3), conv5, 'conv5_3', activation='relu')

    up6 = create_conv(256, (2, 2), UpSampling2D((2, 2))(conv5), 'up6', activation='relu')
    merge6 = concatenate([conv2, up6], axis=3)
    conv6 = create_conv(256, (3, 3), merge6, 'conv6_1', activation='relu')
    conv6 = create_conv(256, (3, 3), conv6, 'conv6_2', activation='relu')
    conv6 = create_conv(256, (3, 3), conv6, 'conv6_3', activation='relu')

    up7 = create_conv(64, (2, 2), UpSampling2D((2, 2))(conv6), 'up7', activation='relu')
    merge7 = concatenate([conv1, up7], axis=3)
    conv7 = create_conv(64, (3, 3), merge7, 'conv7_1', activation='relu')
    conv7 = create_conv(64, (3, 3), conv7, 'conv7_2', activation='relu')
    conv7 = create_conv(64, (3, 3), conv7, 'conv7_3', activation='relu')
    conv7 = Conv2D(2, (1, 1), padding='same', name='conv7_4')(conv7)

    model = Model(inputs=inputs, outputs=conv7)
    model.compile(optimizer=Adam(lr=LEARNING_RATE, beta_1=0.9),
                  loss='mean_squared_error',
                  metrics=['accuracy', eacc, mse, mae])

    if os.path.exists(WEIGHTS):
        model.load_weights(WEIGHTS)

    return model


if MODE == 1:
    model = create_model()
    model_checkpoint = ModelCheckpoint(
        filepath=WEIGHTS,
        monitor='loss',
        verbose=1,
        save_best_only=True)

    scheduler = LearningRateScheduler(learning_scheduler)

    model.fit_generator(
        imagenet_data_generator(batch_size=BATCH_SIZE, flip=False, scale=255),
        steps_per_epoch=1 * (IMAGENET_BATCH_SIZE // BATCH_SIZE),
        validation_data=imagenet_test_data_generator(batch_size=BATCH_SIZE, scale=255, count=VALIDATION_SIZE),
        validation_steps=VALIDATION_SIZE // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[model_checkpoint, scheduler])

elif MODE == 2:
    model = create_model()
    data_test_yuv, data_test_rgb = load_imagenet_test_data(count=VALIDATION_SIZE)
    Y_channel_test = data_test_yuv[:, :, :, :1]
    UV_channel_test = data_test_yuv[:, :, :, 1:]

    for i in range(0, VALIDATION_SIZE):
        y = Y_channel_test[i]
        uv = UV_channel_test
        uv_pred = np.array(model.predict(y[None, :, :, :]))[0] / 255
        yuv_original = np.r_[(y.T, uv[i][:, :, :1].T, uv[i][:, :, 1:].T)].T
        yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
        show_yuv(yuv_original, yuv_pred)

elif MODE == 3:
    img = imread('290982-64.png')[:,:,0:3]
    INPUT_SHAPE = (img.shape[0], img.shape[1], 1)
    model = create_model()

    yuv = color.rgb2yuv(img)
    y = yuv[:, :, :1]
    uv_pred = np.array(model.predict(y[None, :, :, :]))[0] / 255
    yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
    show_yuv(color.rgb2yuv(img), yuv_pred)
