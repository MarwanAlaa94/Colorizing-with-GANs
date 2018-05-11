import os
import time
import glob
import numpy as np
from keras.utils import generic_utils
from model import create_models
from dataset import dir_data_generator
from utils import show_yuv
import matplotlib.pyplot as plt

EPOCHS = 200
BATCH_SIZE = 8
LEARNING_RATE = 0.0005
MOMENTUM = 0.5
LAMBDA1 = 1
LAMBDA2 = 100
INPUT_SHAPE_GEN = (256, 256, 1)
INPUT_SHAPE_DIS = (256, 256, 3)
WEIGHTS_GEN = 'weights_places365_yuv_gen.hdf5'
WEIGHTS_DIS = 'weights_places365_yuv_dis.hdf5'
WEIGHTS_GAN = 'weights_places365_yuv_gan.hdf5'
MODE = 1  # 1: train - 2: visualize

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

model_gen, model_dis, model_gan = create_models(
    input_shape_gen=INPUT_SHAPE_GEN,
    input_shape_dis=INPUT_SHAPE_DIS,
    output_channels=2,
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    loss_weights=[LAMBDA1, LAMBDA2])

if os.path.exists(WEIGHTS_GEN):
    model_gen.load_weights(WEIGHTS_GEN)

if os.path.exists(WEIGHTS_DIS):
    model_dis.load_weights(WEIGHTS_DIS)

if os.path.exists(WEIGHTS_GAN):
    model_gan.load_weights(WEIGHTS_GAN)

# model_gen.summary()
# model_dis.summary()
# model_gan.summary()


PATH = '../../../datasets/Places365/val_256'
TOTAL_SIZE = len(np.array(glob.glob(PATH + '/*.jpg')))
TRAIN_SIZE = 30 * 1024
data_gen = dir_data_generator(dir=PATH, batch_size=BATCH_SIZE, data_range=(0, TRAIN_SIZE), outType='YUV')
test_data_gen = dir_data_generator(dir=PATH, batch_size=BATCH_SIZE, data_range=(0, TOTAL_SIZE), outType='YUV')

loss_gen = []
loss_ave_gen = []
loss_dis = []

if MODE == 1:
    print("Start training")
    for e in range(EPOCHS):
        toggle = True
        batch_counter = 1
        batch_total = TRAIN_SIZE // BATCH_SIZE
        progbar = generic_utils.Progbar(batch_total * BATCH_SIZE)
        start = time.time()
        dis_res = 0
        loss_gen = []

        while batch_counter < batch_total:

            batch_counter += 1
            data_yuv, data_rgb = next(data_gen)
            data_y = data_yuv[:, :, :, :1] * 255
            data_uv = data_yuv[:, :, :, 1:] * 255

            if batch_counter % 2 == 0:
                toggle = not toggle
                if toggle:
                    x_dis = np.concatenate((model_gen.predict(data_y), data_y), axis=3)
                    y_dis = np.zeros((BATCH_SIZE, 1))
                    #y_dis = np.ones((BATCH_SIZE, 1)) * .1
                else:
                    x_dis = np.concatenate((data_uv, data_y), axis=3)
                    y_dis = np.ones((BATCH_SIZE, 1))
                    y_dis = np.ones((BATCH_SIZE, 1)) * .9

                dis_res = model_dis.train_on_batch(x_dis, y_dis)

            model_dis.trainable = False
            x_gen = data_y
            y_gen = np.ones((BATCH_SIZE, 1))
            x_output = data_uv
            gan_res = model_gan.train_on_batch(x_gen, [y_gen, x_output])
            model_dis.trainable = True

            progbar.add(BATCH_SIZE,
                        values=[("D loss", dis_res),
                                ("G total loss", gan_res[0]),
                                ("G loss", gan_res[1]),
                                ("G L1", gan_res[2]),
                                ("pacc", gan_res[5]),
                                ("acc", gan_res[6])])

            #loss_gen.append(gan_res[0])
            #loss_ave_gen.append(np.average(np.array(loss_gen)))

            # plt.ion()
            # plt.clf()
            # #plt.plot(loss_gen[-100:], 'b', loss_ave_gen[-100:], 'r')
            # plt.plot(loss_ave_gen, 'r')
            # plt.show()
            # plt.pause(0.0001)

            if batch_counter % 1000 == 0:
                print('')
                print('Saving weights...')
                model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
                model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
                model_gan.save_weights(WEIGHTS_GAN, overwrite=True)
                np.save('loss_gen', loss_gen)
                np.save('loss_ave_gen', loss_ave_gen)

        print("")
        data_test_yuv, data_test_rgb = next(test_data_gen)
        data_test_y = data_test_yuv[:, :, :, :1] * 255
        data_test_uv = data_test_yuv[:, :, :, 1:] * 255

        ev = model_gan.evaluate(data_test_y, [np.ones((data_test_y.shape[0], 1)), data_test_uv])
        ev = np.round(np.array(ev), 4)
        print('Epoch %s/%s, Time: %s' % (e + 1, EPOCHS, round(time.time() - start)))
        print('G total loss: %s - G loss: %s - G L1: %s: pacc: %s - acc: %s' % (ev[0], ev[1], ev[2], ev[5], ev[6]))
        print('')
        model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
        model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
        model_gan.save_weights(WEIGHTS_GAN, overwrite=True)

elif MODE == 2:
    while True:
        data_test_yuv, data_test_rgb = next(test_data_gen)
        data_test_yuv = data_test_yuv * 255
        data_test_y = data_test_yuv[:, :, :, :1]
        data_test_uv = data_test_yuv[:, :, :, 1:]
        for i in range(0, data_test_yuv.shape[0]):
            y = data_test_y[i]
            yuv_original = data_test_yuv[i]
            uv_pred = np.array(model_gen.predict(y[None, :, :, :]))[0]
            yuv_pred = np.r_[(y.T, uv_pred.T[:1], uv_pred.T[1:])].T
            show_yuv(yuv_original / 255, yuv_pred / 255)
