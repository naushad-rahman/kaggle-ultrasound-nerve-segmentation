from __future__ import print_function

import os
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.callbacks import ModelCheckpoint
from image_processing import load_train_data, load_test_data
from model import get_unet
from helper import print_text

IMG_ROWS = 96
IMG_COLS = 96
PRED_DIR = 'preds'
BATCH_SIZE = 32
EPOCHS = 20
VALID_SPLIT = 0.2


def pre_process(images):
    images_p = np.ndarray((images.shape[0], IMG_ROWS, IMG_COLS), dtype=np.uint8)
    for i in range(images.shape[0]):
        images_p[i] = resize(images[i], (IMG_COLS, IMG_ROWS), preserve_range=True)

    images_p = images_p[..., np.newaxis]
    return images_p


def train_and_predict():
    print_text('Loading and pre-processing train data.')
    images_train, images_mask_train = load_train_data()

    images_train = pre_process(images_train)
    images_mask_train = pre_process(images_mask_train)

    images_train = images_train.astype('float32')
    mean = np.mean(images_train)  # mean for data centering
    std = np.std(images_train)  # std for data normalization

    images_train -= mean
    images_train /= std

    images_mask_train = images_mask_train.astype('float32')
    images_mask_train /= 255.  # scale masks to [0, 1]

    print_text('Creating and compiling model.')
    model = get_unet(IMG_ROWS, IMG_COLS)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print_text('Fitting model.')
    model.fit(images_train, images_mask_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, shuffle=True,
              validation_split=VALID_SPLIT,
              callbacks=[model_checkpoint])

    print_text('Loading and pre-processing test data.')
    images_test, images_id_test = load_test_data()
    images_test = pre_process(images_test)

    images_test = images_test.astype('float32')
    images_test -= mean
    images_test /= std

    print_text('Loading saved weights.')
    model.load_weights('weights.h5')

    print_text('Predicting masks on test data.')
    images_mask_test = model.predict(images_test, verbose=1)
    np.save('images_mask_test.npy', images_mask_test)

    print_text('Saving predicted masks to files.')
    if not os.path.exists(PRED_DIR):
        os.mkdir(PRED_DIR)
    for image, image_id in zip(images_mask_test, images_id_test):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(PRED_DIR, str(image_id) + '_pred.png'), image)


if __name__ == '__main__':
    train_and_predict()
