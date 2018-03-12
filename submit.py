from __future__ import print_function

import numpy as np
from skimage.transform import resize
from image_processing import IMAGE_COLS, IMAGE_ROWS, load_test_data


def prep(image):
    image = image.astype('float32')
    image = (image > 0.5).astype(np.uint8)  # threshold
    image = resize(image, (IMAGE_COLS, IMAGE_ROWS), preserve_range=True)
    return image


def run_length_enc(label):
    from itertools import chain
    x = label.transpose().flatten()
    y = np.where(x > 0)[0]
    if len(y) < 10:  # consider as empty
        return ''
    z = np.where(np.diff(y) > 1)[0]
    start = np.insert(y[z+1], 0, y[0])
    end = np.append(y[z], y[-1])
    length = end - start
    res = [[s+1, l+1] for s, l in zip(list(start), list(length))]
    res = list(chain.from_iterable(res))
    return ' '.join([str(r) for r in res])


def submit():
    test_images, image_masks = load_test_data()
    test_images = np.load('imgs_mask_test.npy')

    arg_sort = np.argsort(image_masks)
    test_image_masks = image_masks[arg_sort]
    test_images = test_images[arg_sort]

    total = test_images.shape[0]
    masks = []
    rles = []
    for i in range(total):
        image = test_images[i, 0]
        image = prep(image)
        rle = run_length_enc(image)

        rles.append(rle)
        masks.append(test_image_masks[i])

        if i % 100 == 0:
            print('{}/{}'.format(i, total))

    first_row = 'image,pixels'
    file_name = 'submission.csv'

    with open(file_name, 'w+') as f:
        f.write(first_row + '\n')
        for i in range(total):
            s = str(masks[i]) + ',' + rles[i]
            f.write(s + '\n')


if __name__ == '__main__':
    submit()
