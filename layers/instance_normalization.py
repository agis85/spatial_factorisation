
from keras.engine.topology import Layer
import keras.backend as K


class InstanceNormalization(Layer):
    '''Instance Normalization adapted from https://github.com/PiscesDream/CycleGAN-keras'''

    def __init__(self, **kwargs):
        super(InstanceNormalization, self).__init__()
        self.inshape = kwargs['inshape'] if 'inshape' in kwargs else None

    def build(self, input_shape):
        self.scale = self.add_weight(name='scale', shape=(input_shape[get_channel_dim()],), initializer="one", trainable=True)
        self.shift = self.add_weight(name='shift', shape=(input_shape[get_channel_dim()],), initializer="zero", trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, x, mask=None):
        if get_channel_dim() == 1:
            h, w = 2, 3
            exp_dim = -1
        else:
            h, w = 1, 2
            exp_dim = 1

        x_shape = self.inshape if self.inshape else x.shape
        hw = K.cast(x_shape[h] * x_shape[w], K.floatx())
        mu = K.sum(x, [h, w]) / hw
        mu_vec = K.expand_dims(K.expand_dims(mu, 1), 1)
        sig2 = K.sum(K.square(x - mu_vec), [h, w]) / hw
        sig2_vec = K.expand_dims(K.expand_dims(sig2, 1), 1)
        y = (x - mu_vec) / (K.sqrt(sig2_vec) + K.epsilon())

        scale = K.expand_dims(K.expand_dims(K.expand_dims(self.scale, 0), exp_dim), exp_dim)
        shift = K.expand_dims(K.expand_dims(K.expand_dims(self.shift, 0), exp_dim), exp_dim)
        return scale * y + shift

    def compute_output_shape(self, input_shape):
        return input_shape

def get_channel_dim():
    if K.image_data_format() == 'channels_first':
        return 1
    return -1