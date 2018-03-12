from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

SMOOTH = 1.
LEARN_RATE = 1e-5
PADDING = 'same'
ACT_F = 'relu'
KERNEL = (3, 3)


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2. * intersection + SMOOTH) / (K.sum(y_true) + K.sum(y_pred) + SMOOTH)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet(image_rows, image_cols):
    inputs = Input((image_rows, image_cols, 1))
    conv_layer_1 = Conv2D(filters=32, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(inputs)
    conv_layer_1 = Conv2D(filters=32, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv_layer_1)

    conv_layer_2 = Conv2D(filters=64, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(pool1)
    conv_layer_2 = Conv2D(filters=64, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv_layer_2)

    conv_layer_3 = Conv2D(filters=128, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(pool2)
    conv_layer_3 = Conv2D(filters=128, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv_layer_3)

    conv_layer_4 = Conv2D(filters=256, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(pool3)
    conv_layer_4 = Conv2D(filters=256, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv_layer_4)

    conv_layer_5 = Conv2D(filters=512, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(pool4)
    conv_layer_5 = Conv2D(filters=512, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_5)

    transp_layer_5 = Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding=PADDING)(conv_layer_5)
    up6 = concatenate([transp_layer_5, conv_layer_4], axis=3)
    conv_layer_6 = Conv2D(filters=256, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(up6)
    conv_layer_6 = Conv2D(filters=256, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_6)

    transp_layer_6 = Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding=PADDING)(conv_layer_6)
    up7 = concatenate([transp_layer_6, conv_layer_3], axis=3)
    conv_layer_7 = Conv2D(filters=128, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(up7)
    conv_layer_7 = Conv2D(filters=128, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_7)

    transp_layer_7 = Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding=PADDING)(conv_layer_7)
    up8 = concatenate([transp_layer_7, conv_layer_2], axis=3)
    conv_layer_8 = Conv2D(filters=64, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(up8)
    conv_layer_8 = Conv2D(filters=64, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_8)

    transp_layer_8 = Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding=PADDING)(conv_layer_8)
    up9 = concatenate([transp_layer_8, conv_layer_1], axis=3)
    conv_layer_9 = Conv2D(filters=32, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(up9)
    conv_layer_9 = Conv2D(filters=32, kernel_size=KERNEL, activation=ACT_F, padding=PADDING)(conv_layer_9)

    conv_layer_10 = Conv2D(filters=1, kernel_size=(1, 1), activation='sigmoid')(conv_layer_9)

    model = Model(inputs=[inputs], outputs=[conv_layer_10])

    model.compile(optimizer=Adam(lr=LEARN_RATE), loss=dice_coef_loss, metrics=[dice_coef])

    return model
