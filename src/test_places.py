import os
import time
import glob
import numpy as np
from keras.utils import generic_utils
from model import create_models
from dataset import dir_data_generator
from utils import show_lab
import matplotlib.pyplot as plt

EPOCHS = 500
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 100
INPUT_SHAPE_GEN = (256, 256, 1)
INPUT_SHAPE_DIS = (256, 256, 4)
WEIGHTS_GEN = 'weights_places365_lab_gen.hdf5'
WEIGHTS_DIS = 'weights_places365_lab_dis.hdf5'
WEIGHTS_GAN = 'weights_places365_lab_gan.hdf5'

#os.environ['CUDA_VISIBLE_DEVICES'] = '0'

model_gen, model_dis, model_gan = create_models(
    input_shape_gen=INPUT_SHAPE_GEN,
    input_shape_dis=INPUT_SHAPE_DIS,
    output_channels=3,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    loss_weights=[LAMBDA1, LAMBDA2])

if os.path.exists(WEIGHTS_GEN):
    model_gen.load_weights(WEIGHTS_GEN)

if os.path.exists(WEIGHTS_DIS):
    model_dis.load_weights(WEIGHTS_DIS)

if os.path.exists(WEIGHTS_GAN):
    model_gan.load_weights(WEIGHTS_GAN)


PATH = 'test'
TOTAL_SIZE = len(np.array(glob.glob(PATH + '/*.jpg')))
test_data_gen = dir_data_generator(dir=PATH, batch_size=BATCH_SIZE, data_range=(0, TOTAL_SIZE), outType='LAB')


while True:
    data_test_lab, data_test_grey = next(test_data_gen)
    for i in range(0, data_test_lab.shape[0]):
        grey = data_test_grey[i]
        lab_original = data_test_lab[i]
        lab_pred = np.array(model_gen.predict(grey[None, :, :, :]))[0]
        show_lab(lab_original, lab_pred, i)
