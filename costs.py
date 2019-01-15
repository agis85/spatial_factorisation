import numpy as np
from keras import backend as K


def dice(y_true, y_pred, smooth=0.1):
    y_pred = y_pred[..., 0:y_true.shape[-1]]

    # Symbolically compute the intersection
    y_int = y_true * y_pred
    return np.mean((2 * np.sum(y_int, axis=(1, 2, 3)) + smooth)
                   / (np.sum(y_true, axis=(1, 2, 3)) + np.sum(y_pred, axis=(1, 2, 3)) + smooth))


def dice_coef(y_true, y_pred):
    '''
    DICE coefficient.
    :param y_true: a tensor of ground truth data
    :param y_pred: a tensor of predicted data
    '''
    # Symbolically compute the intersection
    intersection = K.sum(y_true * y_pred, axis=(1, 2, 3)) + 0.1
    union = K.sum(y_true, axis=(1, 2, 3)) + K.sum(y_pred, axis=(1, 2, 3)) + 0.1
    return K.mean(2 * intersection / union, axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)
