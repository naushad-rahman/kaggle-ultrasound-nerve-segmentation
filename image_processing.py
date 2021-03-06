from __future__ import print_function

import os
import numpy as np

from skimage.io import imread
from helper import print_text

DATA_DIR = 'raw/'

IMAGE_ROWS = 420
IMAGE_COLS = 580


def create_train_data():
    train_data_path = os.path.join(DATA_DIR, 'train')
    images = os.listdir(train_data_path)
    total = len(images) / 2

    imgs = np.ndarray((total, IMAGE_ROWS, IMAGE_COLS), dtype=np.uint8)
    imgs_mask = np.ndarray((total, IMAGE_ROWS, IMAGE_COLS), dtype=np.uint8)

    i = 0
    print_text('Creating training images.')
    for image_name in images:
        if 'mask' in image_name:
            continue
        image_mask_name = image_name.split('.')[0] + '_mask.tif'
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)
        img_mask = imread(os.path.join(train_data_path, image_mask_name), as_grey=True)

        img = np.array([img])
        img_mask = np.array([img_mask])

        imgs[i] = img
        imgs_mask[i] = img_mask

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print_text('Loading done.')

    np.save('imgs_train.npy', imgs)
    np.save('imgs_mask_train.npy', imgs_mask)
    print_text('Saving image data to .npy files is done.')


def load_train_data():
    imgs_train = np.load('imgs_train.npy')
    imgs_mask_train = np.load('imgs_mask_train.npy')
    return imgs_train, imgs_mask_train


def create_test_data():
    train_data_path = os.path.join(DATA_DIR, 'test')
    images = os.listdir(train_data_path)
    total = len(images)

    imgs = np.ndarray((total, IMAGE_ROWS, IMAGE_COLS), dtype=np.uint8)
    imgs_id = np.ndarray((total, ), dtype=np.int32)

    i = 0
    print_text('Creating test images...')
    for image_name in images:
        img_id = int(image_name.split('.')[0])
        img = imread(os.path.join(train_data_path, image_name), as_grey=True)

        img = np.array([img])

        imgs[i] = img
        imgs_id[i] = img_id

        if i % 100 == 0:
            print('Done: {0}/{1} images'.format(i, total))
        i += 1
    print_text('Loading done.')

    np.save('imgs_test.npy', imgs)
    np.save('imgs_mask_test.npy', imgs_id)
    print_text('Saving to .npy files done.')


def load_test_data():
    test_images = np.load('imgs_test.npy')
    image_masks = np.load('imgs_mask_test.npy')
    return test_images, image_masks


if __name__ == '__main__':
    create_train_data()
    create_test_data()
