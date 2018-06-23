import os

import time
import glob
import numpy as np
from keras.utils import generic_utils
from model import create_models
from dataset import dir_data_generator
from utils import show_lab
from utils import evaluate
import matplotlib.pyplot as plt

EPOCHS = 20
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
MODE = 3 # 1: train - 2: visualize

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

# model_gen.summary()
# model_dis.summary()
# model_gan.summary()


PATH = 'datasetsdisney'
TOTAL_SIZE = len(np.array(glob.glob(PATH + '/*.jpg')))
TRAIN_SIZE =  15 * 1024
print("len is ", str(TOTAL_SIZE))
data_gen = dir_data_generator(dir=PATH, batch_size=BATCH_SIZE, data_range=(0, TRAIN_SIZE), outType='LAB')
test_data_gen = dir_data_generator(dir=PATH, batch_size=BATCH_SIZE, data_range=(TRAIN_SIZE, TOTAL_SIZE), outType='LAB')
eval_data = dir_data_generator(dir='disneyBest', batch_size=BATCH_SIZE, data_range=(0, 8), outType
='LAB')
loss_gen = []
loss_ave_gen = []
loss_dis = []

if MODE == 1:
    print("data size = %d", TOTAL_SIZE)
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
            data_lab, data_grey = next(data_gen)

            if batch_counter % 2 == 0:
                toggle = not toggle
                if toggle:
                    x_dis = np.concatenate((model_gen.predict(data_grey), data_grey), axis=3)
                    y_dis = np.zeros((BATCH_SIZE, 1))
                    y_dis = np.ones((BATCH_SIZE, 1)) * .1
                else:
                    x_dis = np.concatenate((data_lab, data_grey), axis=3)
                    y_dis = np.ones((BATCH_SIZE, 1))
                    y_dis = np.ones((BATCH_SIZE, 1)) * .9

                dis_res = model_dis.train_on_batch(x_dis, y_dis)
            model_dis.trainable = False
            x_gen = data_grey
            y_gen = np.ones((BATCH_SIZE, 1))
            x_output = data_lab
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
            if batch_counter % 100 == 0:
                print('')
                print('Saving weights...')
                model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
                model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
                model_gan.save_weights(WEIGHTS_GAN, overwrite=True)
                for j in range(0, 3):
                   data_test_lab, data_test_grey = next(test_data_gen)
                   for i in range(0, data_test_lab.shape[0]):  
                      grey = data_test_grey[i]
                      lab_original = data_test_lab[i]
                      lab_pred = np.array(model_gen.predict(grey[None, :, :, :]))[0]
                      show_lab(lab_original, lab_pred, i + (j * BATCH_SIZE))
        print("")
        print("Start training3")
        data_test_lab, data_test_grey = next(test_data_gen)
        ev = model_gan.evaluate(data_test_grey, [np.ones((data_test_grey.shape[0], 1)), data_test_lab])
        ev = np.round(np.array(ev), 4)
        print('Epoch %s/%s, Time: %s' % (e + 1, EPOCHS, round(time.time() - start)))
        print('G total loss: %s - G loss: %s - G L1: %s: pacc: %s - acc: %s' % (ev[0], ev[1], ev[2], ev[5], ev[6]))
        print('')
        model_gen.save_weights(WEIGHTS_GEN, overwrite=True)
        model_dis.save_weights(WEIGHTS_DIS, overwrite=True)
        model_gan.save_weights(WEIGHTS_GAN, overwrite=True)


elif MODE == 2:
    while True:
        data_test_lab, data_test_grey = next(test_data_gen)
        for i in range(0, data_test_lab.shape[0]):
            grey = data_test_grey[i]
            lab_original = data_test_lab[i]
            lab_pred = np.array(model_gen.predict(grey[None, :, :, :]))[0]
            show_lab(lab_original, lab_pred, i)

elif MODE == 3:
    sum_accuracy = 0
    generator_loss = 0
    count = 0
    toggle = True
    discriminator_loss = 0
    print('Evaluating')
    while True:
        data_test_lab, data_test_grey = next(test_data_gen)
        ev = model_gan.evaluate(data_test_grey, [np.ones((data_test_grey.shape[0], 1)), data_test_lab])
        ev = np.round(np.array(ev), 4)
        generator_loss = generator_loss + ev[0]
        toggle = not toggle
        if toggle:
           x_dis = np.concatenate((model_gen.predict(data_test_grey), data_test_grey), axis=3)
           y_dis = np.ones((BATCH_SIZE, 1)) * .1   
        else:
           x_dis = np.concatenate((data_test_lab, data_test_grey), axis=3)
           y_dis = np.ones((BATCH_SIZE, 1)) * .9
               
        discriminator_loss = discriminator_loss + model_dis.train_on_batch(x_dis, y_dis)    
        for i in range(0, data_test_lab.shape[0]):
            grey = data_test_grey[i]
            lab_original = data_test_lab[i]
            lab_pred = np.array(model_gen.predict(grey[None, :, :, :]))[0]
            sum_accuracy = sum_accuracy + evaluate(lab_original, lab_pred, 2)
            count = count + 1

            if count % 300  == 0:
                print('The pixelwise accuracy is ' + str(sum_accuracy/count)) 
                print ('The generator loss is ' + str((generator_loss / count) * 8))
                print ('The discriminator loss  is ' + str((discriminator_loss / count) * 8))
                data_eval_lab, data_eval_grey = next(eval_data)
                for j in range(0, data_eval_lab.shape[0]):
                    eval_grey = data_eval_grey[j]
                    eval_lab_original = data_eval_lab[j]
                    eval_lab_pred = np.array(model_gen.predict(eval_grey[None, :, :, :]))[0]
                    show_lab(eval_lab_original, eval_lab_pred, j)
